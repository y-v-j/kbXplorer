"""Crash-safe ingestion bookkeeping backed by SQLite.

The manifest is the source of truth for *what has already been indexed*. It
lets the application answer three questions cheaply after any interruption
(reboot, SIGKILL, power loss):

* Which files are new or have changed since the last run? (content hash)
* Which files were mid-ingest when we died? (``status == 'indexing'``)
* Which indexed files have since been deleted from disk?

SQLite runs in WAL mode with ``synchronous=FULL`` so a hard power cut cannot
leave a torn manifest.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

LOGGER = logging.getLogger(__name__)

DocStatus = Literal["pending", "indexing", "indexed", "failed", "missing"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_uid        TEXT PRIMARY KEY,
    rel_path       TEXT NOT NULL,
    abs_path       TEXT NOT NULL,
    document_name  TEXT NOT NULL,
    size_bytes     INTEGER NOT NULL DEFAULT 0,
    mtime          REAL    NOT NULL DEFAULT 0,
    sha256         TEXT    NOT NULL DEFAULT '',
    page_count     INTEGER NOT NULL DEFAULT 0,
    chunk_count    INTEGER NOT NULL DEFAULT 0,
    figure_count   INTEGER NOT NULL DEFAULT 0,
    table_count    INTEGER NOT NULL DEFAULT 0,
    status         TEXT    NOT NULL DEFAULT 'pending',
    error          TEXT    NOT NULL DEFAULT '',
    parser_version TEXT    NOT NULL DEFAULT '',
    indexed_at     TEXT    NOT NULL DEFAULT '',
    updated_at     TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_rel_path ON documents(rel_path);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

#: Bump when parser output changes in a way that invalidates existing vectors.
PARSER_VERSION = "1.1.0"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_digest(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 of a file, read in 1 MiB blocks to bound memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


@dataclass(slots=True)
class DocumentRecord:
    """One row of the ``documents`` table."""

    doc_uid: str
    rel_path: str
    abs_path: str
    document_name: str
    size_bytes: int = 0
    mtime: float = 0.0
    sha256: str = ""
    page_count: int = 0
    chunk_count: int = 0
    figure_count: int = 0
    table_count: int = 0
    status: DocStatus = "pending"
    error: str = ""
    parser_version: str = ""
    indexed_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "DocumentRecord":
        return cls(**{key: row[key] for key in row.keys()})


class Manifest:
    """Thread-safe SQLite wrapper tracking per-document ingestion state."""

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ------------------------------------------------------------------ core
    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """Run a block inside a lock-guarded transaction."""
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def close(self) -> None:
        """Close the underlying connection."""
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:  # pragma: no cover - best effort
                LOGGER.debug("Manifest already closed", exc_info=True)

    def __enter__(self) -> "Manifest":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------- accessors
    def get(self, doc_uid: str) -> DocumentRecord | None:
        """Return the record for ``doc_uid``, or ``None`` if unknown."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM documents WHERE doc_uid = ?", (doc_uid,)
            ).fetchone()
        return DocumentRecord.from_row(row) if row else None

    def all_documents(self) -> list[DocumentRecord]:
        """Return every tracked document, ordered by name."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM documents ORDER BY document_name COLLATE NOCASE"
            ).fetchall()
        return [DocumentRecord.from_row(row) for row in rows]

    def by_status(self, status: DocStatus) -> list[DocumentRecord]:
        """Return every document currently in ``status``."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM documents WHERE status = ? ORDER BY document_name", (status,)
            ).fetchall()
        return [DocumentRecord.from_row(row) for row in rows]

    def counts_by_status(self) -> dict[str, int]:
        """Return a ``{status: count}`` summary."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM documents GROUP BY status"
            ).fetchall()
        return {row["status"]: row["n"] for row in rows}

    def totals(self) -> dict[str, int]:
        """Return aggregate chunk/figure/table/page totals for indexed docs."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT COUNT(*)             AS documents,
                       COALESCE(SUM(page_count), 0)   AS pages,
                       COALESCE(SUM(chunk_count), 0)  AS chunks,
                       COALESCE(SUM(figure_count), 0) AS figures,
                       COALESCE(SUM(table_count), 0)  AS tables
                FROM documents WHERE status = 'indexed'
                """
            ).fetchone()
        return {key: int(row[key]) for key in row.keys()}

    # ------------------------------------------------------------- mutations
    def upsert_pending(self, record: DocumentRecord) -> None:
        """Insert or refresh a document row, preserving prior counts."""
        now = _utcnow()
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    doc_uid, rel_path, abs_path, document_name, size_bytes, mtime,
                    sha256, status, error, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, '', ?)
                ON CONFLICT(doc_uid) DO UPDATE SET
                    rel_path      = excluded.rel_path,
                    abs_path      = excluded.abs_path,
                    document_name = excluded.document_name,
                    size_bytes    = excluded.size_bytes,
                    mtime         = excluded.mtime,
                    sha256        = excluded.sha256,
                    status        = excluded.status,
                    error         = '',
                    updated_at    = excluded.updated_at
                """,
                (
                    record.doc_uid,
                    record.rel_path,
                    record.abs_path,
                    record.document_name,
                    record.size_bytes,
                    record.mtime,
                    record.sha256,
                    record.status,
                    now,
                ),
            )

    def set_status(self, doc_uid: str, status: DocStatus, error: str = "") -> None:
        """Update just the status (and optional error text) of a document."""
        with self._tx() as conn:
            conn.execute(
                "UPDATE documents SET status = ?, error = ?, updated_at = ? WHERE doc_uid = ?",
                (status, error[:2000], _utcnow(), doc_uid),
            )

    def mark_indexed(
        self,
        doc_uid: str,
        *,
        page_count: int,
        chunk_count: int,
        figure_count: int,
        table_count: int,
    ) -> None:
        """Record a successful ingest with its output counts."""
        now = _utcnow()
        with self._tx() as conn:
            conn.execute(
                """
                UPDATE documents SET
                    status = 'indexed', error = '', page_count = ?, chunk_count = ?,
                    figure_count = ?, table_count = ?, parser_version = ?,
                    indexed_at = ?, updated_at = ?
                WHERE doc_uid = ?
                """,
                (
                    page_count,
                    chunk_count,
                    figure_count,
                    table_count,
                    PARSER_VERSION,
                    now,
                    now,
                    doc_uid,
                ),
            )

    def delete(self, doc_uid: str) -> None:
        """Remove a document row entirely."""
        with self._tx() as conn:
            conn.execute("DELETE FROM documents WHERE doc_uid = ?", (doc_uid,))

    def reset_stale_indexing(self) -> list[DocumentRecord]:
        """Demote documents left in ``indexing`` by a crash back to ``pending``.

        Returns the affected records so the caller can log or re-queue them.
        """
        stale = self.by_status("indexing")
        if stale:
            with self._tx() as conn:
                conn.execute(
                    "UPDATE documents SET status='pending', updated_at=? WHERE status='indexing'",
                    (_utcnow(),),
                )
            LOGGER.warning("Reset %d document(s) left mid-ingest by a previous run", len(stale))
        return stale

    def needs_ingest(self, record: DocumentRecord) -> bool:
        """Return ``True`` if the on-disk file differs from what we indexed."""
        existing = self.get(record.doc_uid)
        if existing is None or existing.status != "indexed":
            return True
        if existing.parser_version != PARSER_VERSION:
            return True
        if existing.sha256 and record.sha256:
            return existing.sha256 != record.sha256
        return existing.size_bytes != record.size_bytes or existing.mtime != record.mtime

    # ------------------------------------------------------------------ meta
    def set_meta(self, key: str, value: str) -> None:
        """Store a small key/value pair (e.g. embedding backend fingerprint)."""
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def get_meta(self, key: str, default: str = "") -> str:
        """Read a key/value pair written by :meth:`set_meta`."""
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def prune_missing(self, present_uids: Sequence[str]) -> list[DocumentRecord]:
        """Return indexed documents whose UID is absent from ``present_uids``."""
        present = set(present_uids)
        return [doc for doc in self.all_documents() if doc.doc_uid not in present]
