"""Watchdog-based monitoring of the paper folder.

Two behaviours matter for correctness here:

**Debouncing.** A single file copy emits a burst of ``created``/``modified``
events. Each path is therefore held in a pending set and only ingested once it
has been quiet for ``watch_debounce_seconds``.

**Stability.** A large PDF being copied in is visible long before it is
complete; parsing it then would index a truncated file. Before ingesting, the
watcher requires the file's size to stay unchanged across
``watch_stability_seconds``.

Ingestion runs on a single worker thread, so the watcher never parses two
papers at once and never blocks the watchdog observer.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from engine.config import Settings
from engine.models import document_uid
from ingest import IngestionService

LOGGER = logging.getLogger(__name__)

#: Callback signature: (event, message).
WatcherCallback = Callable[[str, str], None]


def _is_pdf(path: str | Path) -> bool:
    """Return ``True`` for real PDF paths, ignoring editor/sync temp files."""
    name = Path(path).name
    if name.startswith(".") or name.startswith("~$"):
        return False
    return name.lower().endswith(".pdf")


class _PdfEventHandler(FileSystemEventHandler):
    """Translates watchdog events into queued ingest/remove intents."""

    def __init__(self, monitor: "FolderWatcher") -> None:
        self._monitor = monitor

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory and _is_pdf(event.src_path):
            self._monitor.queue_upsert(Path(str(event.src_path)))

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory and _is_pdf(event.src_path):
            self._monitor.queue_upsert(Path(str(event.src_path)))

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        if _is_pdf(event.src_path):
            self._monitor.queue_remove(Path(str(event.src_path)))
        dest = getattr(event, "dest_path", None)
        if dest and _is_pdf(dest):
            self._monitor.queue_upsert(Path(str(dest)))

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory and _is_pdf(event.src_path):
            self._monitor.queue_remove(Path(str(event.src_path)))


class FolderWatcher:
    """Watches the corpus directory and keeps the index in sync."""

    def __init__(
        self,
        settings: Settings,
        service: IngestionService,
        callback: WatcherCallback | None = None,
    ) -> None:
        self._settings = settings
        self._service = service
        self._callback = callback or (lambda event, message: None)

        self._observer: Observer | None = None
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()

        self._lock = threading.Lock()
        self._pending_upsert: dict[Path, float] = {}
        self._pending_remove: dict[Path, float] = {}
        self._sizes: dict[Path, tuple[int, float]] = {}

    # ------------------------------------------------------------- queueing
    def queue_upsert(self, path: Path) -> None:
        """Mark a file for (re)indexing once it settles."""
        with self._lock:
            self._pending_upsert[path] = time.monotonic()
            self._pending_remove.pop(path, None)

    def queue_remove(self, path: Path) -> None:
        """Mark a file for removal from the index."""
        with self._lock:
            self._pending_remove[path] = time.monotonic()
            self._pending_upsert.pop(path, None)
            self._sizes.pop(path, None)

    def _emit(self, event: str, message: str) -> None:
        """Notify the caller, never letting a UI callback kill the watcher."""
        try:
            self._callback(event, message)
        except Exception:  # pragma: no cover - defensive
            LOGGER.debug("Watcher callback raised", exc_info=True)

    # ------------------------------------------------------------- lifecycle
    def start(self, *, initial_sync: bool = True) -> None:
        """Begin watching. Optionally performs a full sync first."""
        self._settings.ensure_directories()

        if initial_sync:
            self._emit("sync-start", f"Scanning {self._settings.pdf_dir}")
            report = self._service.sync()
            self._emit("sync-complete", report.summary())

        handler = _PdfEventHandler(self)
        self._observer = Observer()
        self._observer.schedule(
            handler, str(self._settings.pdf_dir), recursive=self._settings.ingest_recursive
        )
        self._observer.start()

        self._stop.clear()
        self._worker = threading.Thread(target=self._drain_loop, name="kb-ingest", daemon=True)
        self._worker.start()
        self._emit("watching", f"Watching {self._settings.pdf_dir} for changes")
        LOGGER.info("Watching %s", self._settings.pdf_dir)

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the observer and worker thread."""
        self._stop.set()
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout=timeout)
            except Exception:  # pragma: no cover
                LOGGER.debug("Observer shutdown raised", exc_info=True)
            self._observer = None
        if self._worker is not None:
            self._worker.join(timeout=timeout)
            self._worker = None
        LOGGER.info("Watcher stopped")

    def __enter__(self) -> "FolderWatcher":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    # ----------------------------------------------------------- processing
    def _is_stable(self, path: Path) -> bool:
        """Return ``True`` when a file's size has stopped changing.

        Guards against indexing a PDF that is still being copied in.
        """
        try:
            size = path.stat().st_size
        except OSError:
            return False

        now = time.monotonic()
        previous = self._sizes.get(path)
        if previous is None or previous[0] != size:
            self._sizes[path] = (size, now)
            return False
        return (now - previous[1]) >= self._settings.watch_stability_seconds

    def _due(self, pending: dict[Path, float], now: float) -> list[Path]:
        """Return paths whose debounce window has elapsed."""
        delay = self._settings.watch_debounce_seconds
        return [path for path, stamp in pending.items() if now - stamp >= delay]

    def _drain_loop(self) -> None:
        """Worker loop: apply debounced, stabilised changes one at a time."""
        while not self._stop.is_set():
            self._stop.wait(1.0)
            if self._stop.is_set():
                break
            now = time.monotonic()

            with self._lock:
                removals = self._due(self._pending_remove, now)
                for path in removals:
                    self._pending_remove.pop(path, None)
                upserts = self._due(self._pending_upsert, now)

            for path in removals:
                self._handle_remove(path)

            for path in upserts:
                if not path.exists():
                    with self._lock:
                        self._pending_upsert.pop(path, None)
                    continue
                if not self._is_stable(path):
                    continue  # still being written; re-check next tick
                with self._lock:
                    self._pending_upsert.pop(path, None)
                self._sizes.pop(path, None)
                self._handle_upsert(path)

    def _handle_upsert(self, path: Path) -> None:
        """Ingest one settled file."""
        self._emit("ingest-start", path.name)
        try:
            changed, message = self._service.ingest_file(path)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Watcher ingest failed for %s", path)
            self._emit("error", f"{path.name}: {exc}")
            return
        self._emit("ingest-complete" if changed else "ingest-skipped", message)
        LOGGER.info("Watcher: %s", message)

    def _handle_remove(self, path: Path) -> None:
        """Remove a deleted file from the index."""
        uid = document_uid(path, self._settings.pdf_dir)
        try:
            self._service.remove_document(uid)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Watcher removal failed for %s", path)
            self._emit("error", f"{path.name}: {exc}")
            return
        self._emit("removed", f"{path.name}: removed from index")
        LOGGER.info("Watcher: removed %s", path.name)
