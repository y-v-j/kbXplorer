"""Registry of named knowledge bases.

Each knowledge base is a corpus directory plus its own vector store, extracted
figures, and ingestion state. The expensive resources — the embedding model
(~250 MB) and the Ollama LLM (~2.6 GB) — are shared across all of them, so a
second knowledge base costs only its own index: roughly 50 MB resident for a
100-paper corpus. Switching is therefore cheap enough to do interactively.

The registry lives in ``knowledge_bases.json`` at the project root. On first use
it is seeded with a ``default`` entry pointing at whatever paths the current
settings describe, so an existing single-corpus installation keeps working with
no migration.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from engine.config import PROJECT_ROOT, Settings

LOGGER = logging.getLogger(__name__)

REGISTRY_FILE = PROJECT_ROOT / "knowledge_bases.json"

#: Name used for the entry seeded from the existing settings.
DEFAULT_NAME = "default"

#: Where derived index data lives for bases added by name.
BASES_ROOT = PROJECT_ROOT / "bases"


class RegistryError(RuntimeError):
    """Raised for unknown names, duplicates, or an unusable registry file."""


def _resolve(value: str | Path) -> Path:
    """Resolve a possibly relative path against the project root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _store(path: Path) -> str:
    """Render a path relative to the project root when possible."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


@dataclass(slots=True)
class KnowledgeBaseRef:
    """Everything needed to point the application at one corpus."""

    name: str
    pdf_dir: Path
    chroma_dir: Path
    assets_dir: Path
    state_dir: Path
    description: str = ""

    @classmethod
    def derived(cls, name: str, pdf_dir: Path, description: str = "") -> "KnowledgeBaseRef":
        """Build a base whose index data lives under ``bases/<name>/``.

        The corpus directory stays wherever the user keeps it; only the
        generated data (vectors, figures, manifest) is placed under the project.
        """
        root = BASES_ROOT / name
        return cls(
            name=name,
            pdf_dir=_resolve(pdf_dir),
            chroma_dir=root / "chroma_db",
            assets_dir=root / "assets",
            state_dir=root / "state",
            description=description,
        )

    @classmethod
    def from_settings(cls, settings: Settings, name: str = DEFAULT_NAME) -> "KnowledgeBaseRef":
        """Capture the paths of an existing single-corpus installation."""
        return cls(
            name=name,
            pdf_dir=settings.pdf_dir,
            chroma_dir=settings.chroma_dir,
            assets_dir=settings.assets_dir,
            state_dir=settings.state_dir,
            description="Original knowledge base",
        )

    @classmethod
    def from_dict(cls, name: str, payload: dict[str, Any]) -> "KnowledgeBaseRef":
        """Rebuild a reference from its serialised form."""
        root = BASES_ROOT / name
        return cls(
            name=name,
            pdf_dir=_resolve(payload["pdf_dir"]),
            chroma_dir=_resolve(payload.get("chroma_dir", root / "chroma_db")),
            assets_dir=_resolve(payload.get("assets_dir", root / "assets")),
            state_dir=_resolve(payload.get("state_dir", root / "state")),
            description=str(payload.get("description", "")),
        )

    def to_dict(self) -> dict[str, str]:
        """Serialise, keeping in-project paths relative for portability."""
        return {
            "pdf_dir": _store(self.pdf_dir),
            "chroma_dir": _store(self.chroma_dir),
            "assets_dir": _store(self.assets_dir),
            "state_dir": _store(self.state_dir),
            "description": self.description,
        }

    @property
    def images_dir(self) -> Path:
        return self.assets_dir / "images"

    def apply(self, settings: Settings) -> Settings:
        """Return a copy of ``settings`` pointed at this knowledge base."""
        return replace(
            settings,
            pdf_dir=self.pdf_dir,
            chroma_dir=self.chroma_dir,
            assets_dir=self.assets_dir,
            state_dir=self.state_dir,
        )

    def exists(self) -> bool:
        """Whether the corpus directory is present."""
        return self.pdf_dir.is_dir()

    def pdf_count(self) -> int:
        """Number of PDFs currently in the corpus directory."""
        if not self.pdf_dir.is_dir():
            return 0
        return sum(
            1
            for p in self.pdf_dir.rglob("*.pdf")
            if p.is_file() and not p.name.startswith((".", "~$"))
        )


class Registry:
    """Loads, mutates, and persists the set of known knowledge bases."""

    def __init__(self, bases: dict[str, KnowledgeBaseRef], active: str, path: Path) -> None:
        self._bases = bases
        self._active = active
        self._path = path

    # ------------------------------------------------------------- loading
    @classmethod
    def load(cls, settings: Settings, path: Path | None = None) -> "Registry":
        """Load the registry, seeding a ``default`` entry on first use."""
        target = path or REGISTRY_FILE
        bases: dict[str, KnowledgeBaseRef] = {}
        active = DEFAULT_NAME

        if target.is_file():
            try:
                payload = json.loads(target.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                LOGGER.warning("Unreadable registry %s (%s); using defaults", target, exc)
                payload = {}
            for name, entry in (payload.get("bases") or {}).items():
                try:
                    bases[name] = KnowledgeBaseRef.from_dict(name, entry)
                except (KeyError, TypeError, ValueError) as exc:
                    LOGGER.warning("Skipping malformed registry entry %r: %s", name, exc)
            active = str(payload.get("active") or DEFAULT_NAME)

        if not bases:
            bases[DEFAULT_NAME] = KnowledgeBaseRef.from_settings(settings)
            active = DEFAULT_NAME

        if active not in bases:
            active = next(iter(bases))

        return cls(bases, active, target)

    # ------------------------------------------------------------ accessors
    @property
    def active_name(self) -> str:
        return self._active

    @property
    def active(self) -> KnowledgeBaseRef:
        """The currently selected knowledge base."""
        return self._bases[self._active]

    def names(self) -> list[str]:
        """Registered names, active first, then alphabetical."""
        others = sorted(n for n in self._bases if n != self._active)
        return [self._active, *others]

    def all(self) -> list[KnowledgeBaseRef]:
        """Every registered base, active first."""
        return [self._bases[n] for n in self.names()]

    def get(self, name: str) -> KnowledgeBaseRef:
        """Look up a base by name.

        Raises:
            RegistryError: If no such base is registered.
        """
        try:
            return self._bases[name]
        except KeyError:
            known = ", ".join(sorted(self._bases)) or "none"
            raise RegistryError(f"unknown knowledge base {name!r} (known: {known})") from None

    def __contains__(self, name: object) -> bool:
        return name in self._bases

    def __len__(self) -> int:
        return len(self._bases)

    # ------------------------------------------------------------ mutation
    def add(self, ref: KnowledgeBaseRef, *, replace_existing: bool = False) -> None:
        """Register a new base.

        Raises:
            RegistryError: If the name is taken and ``replace_existing`` is not set.
        """
        if ref.name in self._bases and not replace_existing:
            raise RegistryError(f"knowledge base {ref.name!r} already exists")
        self._bases[ref.name] = ref

    def remove(self, name: str) -> KnowledgeBaseRef:
        """Unregister a base and return it. Does not delete any files."""
        ref = self.get(name)
        if len(self._bases) == 1:
            raise RegistryError("cannot remove the only knowledge base")
        del self._bases[name]
        if self._active == name:
            self._active = next(iter(self._bases))
        return ref

    def set_active(self, name: str) -> KnowledgeBaseRef:
        """Mark a base as active and return it."""
        ref = self.get(name)
        self._active = name
        return ref

    def save(self) -> Path:
        """Persist the registry to disk."""
        payload = {
            "active": self._active,
            "bases": {name: ref.to_dict() for name, ref in sorted(self._bases.items())},
        }
        self._path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return self._path
