"""Ingestion pipeline: scan, parse, embed, and index PDFs.

The service is designed to be interrupted. Every file is processed
independently and its state recorded in the SQLite manifest before and after
work, so a reboot, crash, or Ctrl-C leaves the knowledge base consistent:
already-indexed papers are skipped on the next run, and anything caught
mid-flight is retried.

Files are processed one at a time on purpose. Parsing a paper with figure
rendering peaks around 400-500 MB; running several in parallel is what would
push a 16 GB laptop into swap.
"""

from __future__ import annotations

import fnmatch
import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from engine.config import Settings
from engine.embeddings import backend_fingerprint
from engine.locking import ingest_lock
from engine.manifest import PARSER_VERSION, DocumentRecord, Manifest, file_digest
from engine.models import document_uid
from engine.pdf_parser import PdfParseError, parse_pdf
from engine.vector_store import VectorStore

LOGGER = logging.getLogger(__name__)

#: Callback signature: (stage, message, current, total).
ProgressCallback = Callable[[str, str, int, int], None]

_MANIFEST_EMBEDDING_KEY = "embedding_fingerprint"


@dataclass(slots=True)
class IngestReport:
    """Summary of one ingestion run."""

    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    removed: int = 0
    chunks: int = 0
    figures: int = 0
    tables: int = 0
    elapsed_seconds: float = 0.0
    errors: list[tuple[str, str]] = field(default_factory=list)
    #: Filenames a dry run would have (re-)ingested.
    planned: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a one-line human-readable summary."""
        if self.planned:
            return f"would re-ingest {len(self.planned)} document(s), skip {self.skipped}"
        return (
            f"indexed={self.indexed} skipped={self.skipped} failed={self.failed} "
            f"removed={self.removed} chunks={self.chunks} figures={self.figures} "
            f"tables={self.tables} in {self.elapsed_seconds:.1f}s"
        )


def discover_pdfs(settings: Settings) -> list[Path]:
    """Return every readable PDF under the configured corpus directory.

    Hidden files and common temporary artefacts from sync tools are ignored.
    """
    root = settings.pdf_dir
    if not root.is_dir():
        LOGGER.warning("PDF directory does not exist: %s", root)
        return []

    pattern = "**/*.pdf" if settings.ingest_recursive else "*.pdf"
    found: list[Path] = []
    for path in sorted(root.glob(pattern)):
        if not path.is_file():
            continue
        if path.name.startswith(".") or path.name.startswith("~$"):
            continue
        found.append(path)
    return found


def _build_record(path: Path, settings: Settings, *, with_digest: bool = True) -> DocumentRecord:
    """Create a manifest record describing the file as it is on disk now."""
    stat = path.stat()
    try:
        rel = path.resolve().relative_to(settings.pdf_dir.resolve()).as_posix()
    except ValueError:
        rel = path.name
    return DocumentRecord(
        doc_uid=document_uid(path, settings.pdf_dir),
        rel_path=rel,
        abs_path=str(path.resolve()),
        document_name=path.name,
        size_bytes=stat.st_size,
        mtime=stat.st_mtime,
        sha256=file_digest(path) if with_digest else "",
        status="pending",
    )


@dataclass(slots=True)
class ReindexSelector:
    """Chooses which documents a re-index should touch.

    An empty selector means "every document". Any active criterion narrows the
    set, and criteria combine with OR: ``--failed --stale`` re-ingests documents
    that are either failed or stale.
    """

    pattern: str | None = None
    failed_only: bool = False
    stale_only: bool = False

    @property
    def is_narrowed(self) -> bool:
        """Whether this selector targets a subset rather than the whole corpus."""
        return bool(self.pattern) or self.failed_only or self.stale_only

    def describe(self) -> str:
        """Human-readable summary of the criteria."""
        if not self.is_narrowed:
            return "all documents"
        parts: list[str] = []
        if self.pattern:
            parts.append(f"matching {self.pattern!r}")
        if self.failed_only:
            parts.append("previously failed")
        if self.stale_only:
            parts.append(f"parsed by a version other than {PARSER_VERSION}")
        return " or ".join(parts)

    def matches(self, path: Path, record: DocumentRecord | None) -> bool:
        """Return ``True`` if ``path`` should be re-ingested."""
        if not self.is_narrowed:
            return True
        if self.pattern:
            needle = self.pattern.lower()
            name = path.name.lower()
            if fnmatch.fnmatch(name, needle) or needle in name:
                return True
            if fnmatch.fnmatch(str(path).lower(), needle):
                return True
        if self.failed_only and record is not None and record.status == "failed":
            return True
        if self.stale_only and (record is None or record.parser_version != PARSER_VERSION):
            return True
        return False


class IngestionService:
    """Coordinates parsing, embedding, and manifest bookkeeping."""

    def __init__(
        self,
        settings: Settings,
        store: VectorStore,
        manifest: Manifest,
        progress: ProgressCallback | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._manifest = manifest
        self._progress = progress or (lambda *_: None)

    def _emit(self, stage: str, message: str, current: int = 0, total: int = 0) -> None:
        """Report progress, never letting a UI callback break ingestion."""
        try:
            self._progress(stage, message, current, total)
        except Exception:  # pragma: no cover - defensive
            LOGGER.debug("Progress callback raised", exc_info=True)

    # ------------------------------------------------------------ integrity
    def check_embedding_backend(self) -> str | None:
        """Detect an embedding-backend change that would corrupt the index.

        Returns:
            A warning message if the stored fingerprint differs, else ``None``.
        """
        current = backend_fingerprint(self._settings)
        stored = self._manifest.get_meta(_MANIFEST_EMBEDDING_KEY)
        if not stored:
            self._manifest.set_meta(_MANIFEST_EMBEDDING_KEY, current)
            return None
        if stored != current:
            return (
                f"Embedding backend changed ({stored} -> {current}). Existing vectors "
                "live in a different space. Run `python main.py reindex` to rebuild."
            )
        return None

    def recover_interrupted(self) -> int:
        """Requeue documents left mid-ingest by a previous process."""
        stale = self._manifest.reset_stale_indexing()
        if stale:
            self._emit("recover", f"Requeued {len(stale)} interrupted document(s)", 0, 0)
        return len(stale)

    # ------------------------------------------------------------- ingestion
    def ingest_file(self, path: Path, *, force: bool = False) -> tuple[bool, str]:
        """Parse and index one PDF.

        Args:
            path: PDF to ingest.
            force: Re-index even if the manifest says the file is unchanged.

        Returns:
            ``(changed, message)`` where ``changed`` is ``True`` only when the
            vector store was actually written to.
        """
        if not path.is_file():
            return False, f"{path.name}: not found"

        try:
            record = _build_record(path, self._settings)
        except OSError as exc:
            return False, f"{path.name}: unreadable ({exc})"

        if not force and not self._manifest.needs_ingest(record):
            return False, f"{path.name}: unchanged"

        self._manifest.upsert_pending(record)
        self._manifest.set_status(record.doc_uid, "indexing")

        try:
            parsed = parse_pdf(path, self._settings)
        except PdfParseError as exc:
            self._manifest.set_status(record.doc_uid, "failed", str(exc))
            return False, f"{path.name}: parse failed ({exc})"
        except Exception as exc:  # pragma: no cover - unexpected parser fault
            LOGGER.exception("Unexpected parse failure for %s", path)
            self._manifest.set_status(record.doc_uid, "failed", str(exc))
            return False, f"{path.name}: parse crashed ({exc})"

        if not parsed.chunks and not parsed.figures:
            self._manifest.set_status(
                record.doc_uid, "failed", "no extractable text (scanned PDF without OCR?)"
            )
            return False, f"{path.name}: no extractable text"

        previous_images = set(self._store.figure_image_paths(record.doc_uid))

        try:
            self._store.replace_document(record.doc_uid, parsed.chunks, parsed.figures)
        except Exception as exc:
            LOGGER.exception("Vector write failed for %s", path)
            self._manifest.set_status(record.doc_uid, "failed", str(exc))
            return False, f"{path.name}: indexing failed ({exc})"

        # A re-parse can emit fewer or differently numbered figures; drop the
        # files the new parse no longer references.
        self._delete_figure_files(previous_images - {f.image_path for f in parsed.figures})

        self._manifest.mark_indexed(
            record.doc_uid,
            page_count=parsed.page_count,
            chunk_count=len(parsed.chunks),
            figure_count=len(parsed.figures),
            table_count=parsed.table_count,
        )
        for warning in parsed.warnings[:5]:
            LOGGER.debug("%s: %s", path.name, warning)

        return True, (
            f"{path.name}: {parsed.page_count}p, {len(parsed.chunks)} chunks, "
            f"{parsed.table_count} tables, {len(parsed.figures)} figures"
        )

    def _delete_figure_files(self, paths: Iterable[str]) -> int:
        """Delete extracted figure files, returning how many were removed.

        Only paths inside ``assets/images`` are touched. Metadata is data, not
        a command: a stored path pointing anywhere else is ignored rather than
        followed, so a hand-edited or corrupted index cannot delete arbitrary
        files.
        """
        images_root = self._settings.images_dir.resolve()
        removed = 0
        for raw in paths:
            if not raw:
                continue
            try:
                target = Path(raw).resolve()
                target.relative_to(images_root)
            except (ValueError, OSError):
                continue
            try:
                target.unlink()
                removed += 1
            except FileNotFoundError:
                pass
            except OSError as exc:  # pragma: no cover - permissions
                LOGGER.debug("Could not delete %s: %s", target, exc)
        return removed

    def remove_document(self, doc_uid: str) -> None:
        """Drop a document from the vector store, the manifest, and disk.

        Its extracted figures are deleted too, so replacing a corpus does not
        leave orphaned PNGs behind.
        """
        stale_images = self._store.figure_image_paths(doc_uid)
        self._store.delete_document(doc_uid)
        self._manifest.delete(doc_uid)
        freed = self._delete_figure_files(stale_images)
        if freed:
            LOGGER.debug("Removed %d figure file(s) for %s", freed, doc_uid)

    def sync(
        self,
        *,
        force: bool = False,
        prune: bool = True,
        selector: ReindexSelector | None = None,
        dry_run: bool = False,
    ) -> IngestReport:
        """Bring the index in line with the corpus directory.

        New and modified papers are indexed, unchanged ones skipped, and (when
        ``prune`` is set) papers deleted from disk are removed from the index.

        Args:
            force: Re-ingest even when the manifest says a file is unchanged.
            prune: Remove index entries for files no longer on disk. Ignored
                when ``selector`` narrows the run, since the documents outside
                the selection are still present and must not be dropped.
            selector: Restricts the run to a subset of documents.
            dry_run: Report what would happen without touching the index.
        """
        started = time.monotonic()
        report = IngestReport()
        selector = selector or ReindexSelector()

        if selector.is_narrowed:
            prune = False

        if dry_run:
            # A dry run reports through its caller, not the progress line.
            self._plan(report, selector=selector, force=force)
            report.elapsed_seconds = time.monotonic() - started
            return report

        with ingest_lock(self._settings.state_dir / "ingest.lock"):
            self._sync_locked(report, force=force, prune=prune, selector=selector)

        report.elapsed_seconds = time.monotonic() - started
        self._emit("complete", report.summary(), 0, 0)
        return report

    def _plan(self, report: IngestReport, *, selector: ReindexSelector, force: bool) -> None:
        """Populate ``report.planned`` without modifying anything."""
        for path in discover_pdfs(self._settings):
            record = self._manifest.get(document_uid(path, self._settings.pdf_dir))
            if not selector.matches(path, record):
                continue
            if not force and record is not None and record.status == "indexed":
                try:
                    fresh = _build_record(path, self._settings)
                except OSError:
                    continue
                if not self._manifest.needs_ingest(fresh):
                    report.skipped += 1
                    continue
            report.planned.append(path.name)

    def _sync_locked(
        self,
        report: IngestReport,
        *,
        force: bool,
        prune: bool,
        selector: ReindexSelector | None = None,
    ) -> None:
        """Body of :meth:`sync`, executed while holding the ingestion lock."""
        selector = selector or ReindexSelector()
        warning = self.check_embedding_backend()
        if warning:
            LOGGER.warning(warning)
            report.errors.append(("configuration", warning))

        self.recover_interrupted()

        paths = discover_pdfs(self._settings)
        total = len(paths)
        self._emit("scan", f"Found {total} PDF(s) in {self._settings.pdf_dir}", 0, total)

        present: list[str] = []
        selected: list[Path] = []
        for path in paths:
            uid = document_uid(path, self._settings.pdf_dir)
            present.append(uid)
            if selector.matches(path, self._manifest.get(uid)):
                selected.append(path)

        total = len(selected)
        if selector.is_narrowed:
            self._emit("scan", f"Selected {total} document(s): {selector.describe()}", 0, total)

        for index, path in enumerate(selected, start=1):
            uid = document_uid(path, self._settings.pdf_dir)
            self._emit("ingest", path.name, index, total)
            try:
                changed, message = self.ingest_file(path, force=force)
            except Exception as exc:  # pragma: no cover - last-resort guard
                LOGGER.exception("Ingest crashed on %s", path)
                report.failed += 1
                report.errors.append((path.name, str(exc)))
                continue

            if changed:
                report.indexed += 1
                document = self._manifest.get(uid)
                if document:
                    report.chunks += document.chunk_count
                    report.figures += document.figure_count
                    report.tables += document.table_count
                LOGGER.info(message)
            elif "unchanged" in message:
                report.skipped += 1
            else:
                report.failed += 1
                report.errors.append((path.name, message))
                LOGGER.warning(message)
            self._emit("done-file", message, index, total)

        if prune:
            for orphan in self._manifest.prune_missing(present):
                self.remove_document(orphan.doc_uid)
                report.removed += 1
                LOGGER.info("Removed deleted document: %s", orphan.document_name)

    # ----------------------------------------------------------------- info
    def status(self) -> dict[str, object]:
        """Return a snapshot combining manifest and vector-store state."""
        totals = self._manifest.totals()
        counts = self._store.counts()
        return {
            "pdf_dir": str(self._settings.pdf_dir),
            "documents_indexed": totals.get("documents", 0),
            "pages": totals.get("pages", 0),
            "chunks_manifest": totals.get("chunks", 0),
            "figures_manifest": totals.get("figures", 0),
            "tables": totals.get("tables", 0),
            "vectors_chunks": counts.get("chunks", 0),
            "vectors_figures": counts.get("figures", 0),
            "by_status": self._manifest.counts_by_status(),
            "embedding": backend_fingerprint(self._settings),
        }


def build_service(
    settings: Settings, progress: ProgressCallback | None = None
) -> tuple[IngestionService, VectorStore, Manifest]:
    """Construct the ingestion stack and return its components.

    The caller owns the returned :class:`Manifest` and must close it.
    """
    settings.ensure_directories()
    store = VectorStore(settings)
    manifest = Manifest(settings.manifest_path)
    service = IngestionService(settings, store, manifest, progress)
    return service, store, manifest
