"""Central, file-backed configuration for the local research knowledge base.

All tunables live in a single :class:`Settings` dataclass.  Defaults are chosen
for a CPU-only 11th-gen Intel laptop with 16 GB of RAM: the embedding backend is
the quantised ONNX build of ``all-MiniLM-L6-v2`` (~150 MB resident instead of
~700 MB for the PyTorch build), and the default LLM is a 3B quantised model.

Precedence (lowest to highest): dataclass defaults -> ``config.json`` ->
``KB_*`` environment variables.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Final

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
CONFIG_FILE: Final[Path] = PROJECT_ROOT / "config.json"

#: Environment variable prefix used for overrides, e.g. ``KB_OLLAMA_MODEL``.
ENV_PREFIX: Final[str] = "KB_"

_PATH_FIELDS: Final[frozenset[str]] = frozenset(
    {"pdf_dir", "chroma_dir", "assets_dir", "state_dir", "log_dir"}
)


@dataclass(slots=True)
class Settings:
    """Runtime configuration for every component of the knowledge base."""

    # ----------------------------------------------------------------- paths
    #: Source corpus. Only this directory is indexed; the reference textbooks
    #: that live one level up in ``pdf_files/`` are deliberately excluded.
    pdf_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "pdf_files" / "input_pdf_files")
    chroma_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "chroma_db")
    assets_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "assets")
    state_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "state")
    log_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")

    # ----------------------------------------------------------- collections
    text_collection: str = "paper_chunks"
    figure_collection: str = "paper_figures"

    # ------------------------------------------------------------ embeddings
    #: ``"onnx"`` (default, low memory) or ``"sentence-transformers"``.
    embedding_backend: str = "onnx"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 64

    # -------------------------------------------------------------- chunking
    chunk_target_chars: int = 1100
    chunk_overlap_chars: int = 180
    min_chunk_chars: int = 120

    # --------------------------------------------------------------- figures
    extract_figures: bool = True
    #: Render DPI for caption-anchored figure regions (vector-safe).
    figure_dpi: int = 160
    #: Minimum region area in PDF points^2 before a figure is kept.
    figure_min_area: float = 10_000.0
    figure_max_per_page: int = 8
    #: Also dump embedded raster images that no rendered region already covers.
    extract_embedded_images: bool = True
    embedded_image_min_pixels: int = 40_000

    # ---------------------------------------------------------------- tables
    extract_tables: bool = True
    table_max_rows: int = 80
    table_max_chars: int = 6_000

    # ------------------------------------------------------------- retrieval
    top_k_text: int = 5
    top_k_figures: int = 3
    #: Candidates pulled from the vector index before lexical re-ranking.
    retrieval_candidates: int = 24
    #: Weight of the lexical (keyword-overlap) score in the hybrid ranking.
    lexical_weight: float = 0.35
    max_context_chars: int = 4_500

    # ---------------------------------------------------------------- ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    #: Models tried, in order, when ``ollama_model`` is not installed.
    ollama_fallback_models: tuple[str, ...] = (
        "llama3.2:3b",
        "qwen2.5:7b-instruct-q4_K_M",
        "qwen3.5:4b",
        "phi3:mini",
    )
    ollama_num_ctx: int = 4096
    ollama_temperature: float = 0.1
    #: Hard cap on generated tokens. On CPU each token costs ~0.1 s, so an
    #: unbounded answer is the difference between 40 s and several minutes.
    ollama_num_predict: int = 700
    #: Keep the model resident between questions; reloading a 2 GB model
    #: from disk dominates time-to-first-token on a memory-tight machine.
    ollama_keep_alive: str = "30m"
    ollama_timeout: float = 600.0
    #: Suppress ``<think>`` blocks emitted by reasoning models (e.g. qwen3.x).
    strip_reasoning_tokens: bool = True

    # --------------------------------------------------------------- ingest
    watch_debounce_seconds: float = 4.0
    #: A file must keep a stable size for this long before it is ingested.
    watch_stability_seconds: float = 2.0
    ingest_recursive: bool = True

    # ------------------------------------------------------------------ misc
    log_level: str = "INFO"

    # ------------------------------------------------------------- helpers
    @property
    def images_dir(self) -> Path:
        """Directory holding extracted figure PNGs."""
        return self.assets_dir / "images"

    @property
    def manifest_path(self) -> Path:
        """SQLite file tracking per-document ingestion state."""
        return self.state_dir / "manifest.db"

    def ensure_directories(self) -> None:
        """Create every directory the application writes to."""
        for path in (
            self.pdf_dir,
            self.chroma_dir,
            self.assets_dir,
            self.images_dir,
            self.state_dir,
            self.log_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the settings."""
        raw = asdict(self)
        return {
            key: (str(value) if isinstance(value, Path) else list(value) if isinstance(value, tuple) else value)
            for key, value in raw.items()
        }

    def save(self, path: Path | None = None) -> Path:
        """Persist the settings to ``config.json`` and return the written path."""
        target = path or CONFIG_FILE
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target


def _coerce(name: str, raw: Any, current: Any) -> Any:
    """Coerce ``raw`` to the type of the existing ``current`` field value."""
    if name in _PATH_FIELDS:
        return Path(str(raw)).expanduser()
    if isinstance(current, bool):
        if isinstance(raw, str):
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        return bool(raw)
    if isinstance(current, tuple):
        if isinstance(raw, str):
            return tuple(part.strip() for part in raw.split(",") if part.strip())
        return tuple(raw)
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return type(current)(raw) if current is not None else raw


def load_settings(config_file: Path | None = None) -> Settings:
    """Build a :class:`Settings` instance from defaults, file, and environment.

    Unknown or malformed keys are logged and ignored rather than raising, so a
    stale ``config.json`` can never prevent the application from starting.
    """
    settings = Settings()
    valid = {f.name for f in fields(Settings)}

    source = config_file or CONFIG_FILE
    if source.is_file():
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("Ignoring unreadable config file %s: %s", source, exc)
            payload = {}
        for key, value in payload.items():
            if key not in valid:
                LOGGER.debug("Ignoring unknown config key %r", key)
                continue
            try:
                setattr(settings, key, _coerce(key, value, getattr(settings, key)))
            except (TypeError, ValueError) as exc:
                LOGGER.warning("Ignoring bad config value for %r: %s", key, exc)

    for key in valid:
        env_key = f"{ENV_PREFIX}{key.upper()}"
        if env_key not in os.environ:
            continue
        try:
            setattr(settings, key, _coerce(key, os.environ[env_key], getattr(settings, key)))
        except (TypeError, ValueError) as exc:
            LOGGER.warning("Ignoring bad value in %s: %s", env_key, exc)

    return settings


def configure_logging(settings: Settings, *, to_file: bool = True, quiet: bool = False) -> None:
    """Configure root logging for CLI and TUI entry points.

    Args:
        settings: Active settings (supplies log level and log directory).
        to_file: Also write a rotating-free plain log to ``logs/knowledge_base.log``.
        quiet: Suppress console output — used by the TUI, whose stdout is the UI.
    """
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = []

    if not quiet:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("%(levelname)-8s %(name)s: %(message)s"))
        handlers.append(console)

    if to_file:
        file_handler = logging.FileHandler(settings.log_dir / "knowledge_base.log", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        handlers=handlers,
        force=True,
    )
    # These libraries are extremely chatty at INFO.
    for noisy in ("chromadb", "httpx", "httpcore", "urllib3", "pdfminer", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
