"""Advisory inter-process locking for ingestion.

ChromaDB's persistent client keeps its index in SQLite. Two processes writing
the same directory at once (a stray ``main.py watch`` alongside
``main.py tui --watch``, say) can deadlock on that SQLite file or interleave
partial document replacements.

Ingestion therefore takes an advisory ``flock`` on ``state/ingest.lock``. Readers
— queries and the TUI's retrieval path — never take the lock; only writers do.
The lock is released automatically when the process exits, including on SIGKILL
or power loss, because it lives in the kernel rather than on disk.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path

LOGGER = logging.getLogger(__name__)


class IngestLockBusy(RuntimeError):
    """Raised when another process already holds the ingestion lock."""


@contextmanager
def ingest_lock(lock_path: Path, *, blocking: bool = False) -> Iterator[None]:
    """Hold an advisory exclusive lock for the duration of the block.

    Args:
        lock_path: File used as the lock. Created if absent.
        blocking: Wait for the lock instead of failing immediately.

    Raises:
        IngestLockBusy: If the lock is held and ``blocking`` is ``False``.

    Note:
        On platforms without ``fcntl`` (Windows) this degrades to a no-op, which
        is acceptable: the lock is advisory and only guards a rare race.
    """
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows
        LOGGER.debug("fcntl unavailable; ingestion lock disabled")
        yield
        return

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB

    try:
        try:
            fcntl.flock(handle, flags)
        except OSError as exc:
            raise IngestLockBusy(
                f"another process is already ingesting (lock: {lock_path}). "
                "Wait for it to finish, or stop it first."
            ) from exc

        try:
            os.ftruncate(handle, 0)
            os.write(handle, f"{os.getpid()}\n".encode())
        except OSError:  # pragma: no cover - informational only
            pass

        yield
    finally:
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        except OSError:  # pragma: no cover
            pass
        os.close(handle)
