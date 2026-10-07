"""Typed data structures shared across the ingestion, storage, and RAG layers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

ContentType = Literal["text", "table", "figure"]
FigureKind = Literal["figure", "table", "image"]


def document_uid(pdf_path: Path, root: Path) -> str:
    """Return a stable identifier for a document based on its path.

    The UID is derived from the path relative to the corpus root (falling back
    to the absolute path), so it survives content edits — which is what lets a
    re-ingest replace the previous vectors for the same file.
    """
    try:
        key = pdf_path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        key = pdf_path.resolve().as_posix()
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


@dataclass(slots=True)
class PageLine:
    """A single synthesised text line on a page.

    PDFs have no native line concept; ``number`` is assigned by sorting the
    page's text spans into reading order (column-aware). It is stable for a
    given file and parser version, and is what citations refer to.
    """

    number: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    column: int = 0

    @property
    def width(self) -> float:
        return self.x1 - self.x0


@dataclass(slots=True)
class Chunk:
    """A retrievable unit of text carrying exact structural provenance."""

    doc_uid: str
    document_name: str
    source_path: str
    page_number: int
    line_start: int
    line_end: int
    content_type: ContentType
    text: str
    chunk_index: int
    caption: str = ""
    image_path: str = ""

    @property
    def chunk_id(self) -> str:
        """Deterministic ChromaDB id, so re-ingest upserts rather than duplicates."""
        return f"{self.doc_uid}:p{self.page_number}:c{self.chunk_index}"

    def citation_label(self) -> str:
        """Render the canonical citation string for this chunk."""
        return f"[{self.document_name}, Page {self.page_number}, Line {self.line_start}-{self.line_end}]"

    def to_metadata(self) -> dict[str, Any]:
        """Return ChromaDB-safe metadata (scalars only)."""
        return {
            "doc_uid": self.doc_uid,
            "document_name": self.document_name,
            "source_path": self.source_path,
            "page_number": self.page_number,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "content_type": self.content_type,
            "chunk_index": self.chunk_index,
            "caption": self.caption,
            "image_path": self.image_path,
        }


@dataclass(slots=True)
class FigureRecord:
    """A figure or table region extracted from a page, indexed by its caption."""

    doc_uid: str
    document_name: str
    source_path: str
    page_number: int
    line_start: int
    line_end: int
    caption: str
    image_path: str
    kind: FigureKind
    figure_index: int
    label: str = ""

    @property
    def figure_id(self) -> str:
        """Deterministic ChromaDB id for the figure collection."""
        return f"{self.doc_uid}:p{self.page_number}:f{self.figure_index}"

    def to_metadata(self) -> dict[str, Any]:
        """Return ChromaDB-safe metadata (scalars only)."""
        return {
            "doc_uid": self.doc_uid,
            "document_name": self.document_name,
            "source_path": self.source_path,
            "page_number": self.page_number,
            "page": self.page_number,  # spec-mandated alias
            "line_start": self.line_start,
            "line_end": self.line_end,
            "image_path": self.image_path,
            "caption": self.caption,
            "kind": self.kind,
            "label": self.label,
            "content_type": "figure",
            "figure_index": self.figure_index,
        }


@dataclass(slots=True)
class ParsedDocument:
    """Everything the parser recovered from one PDF."""

    doc_uid: str
    document_name: str
    source_path: str
    page_count: int
    chunks: list[Chunk] = field(default_factory=list)
    figures: list[FigureRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def table_count(self) -> int:
        return sum(1 for chunk in self.chunks if chunk.content_type == "table")

    @property
    def text_chunk_count(self) -> int:
        return sum(1 for chunk in self.chunks if chunk.content_type == "text")


@dataclass(slots=True)
class Retrieved:
    """A scored search hit returned by the vector store."""

    id: str
    text: str
    metadata: dict[str, Any]
    distance: float
    dense_score: float = 0.0
    lexical_score: float = 0.0
    score: float = 0.0

    @property
    def document_name(self) -> str:
        return str(self.metadata.get("document_name", "unknown.pdf"))

    @property
    def page_number(self) -> int:
        return int(self.metadata.get("page_number", 0) or 0)

    @property
    def line_start(self) -> int:
        return int(self.metadata.get("line_start", 0) or 0)

    @property
    def line_end(self) -> int:
        return int(self.metadata.get("line_end", 0) or 0)

    @property
    def content_type(self) -> str:
        return str(self.metadata.get("content_type", "text"))

    @property
    def image_path(self) -> str:
        return str(self.metadata.get("image_path", "") or "")

    @property
    def caption(self) -> str:
        return str(self.metadata.get("caption", "") or "")

    def citation_label(self) -> str:
        """Render the canonical citation string for this hit."""
        return f"[{self.document_name}, Page {self.page_number}, Line {self.line_start}-{self.line_end}]"


@dataclass(slots=True)
class Citation:
    """A citation parsed out of an LLM answer and checked against retrieval."""

    document_name: str
    page: int
    line_start: int
    line_end: int
    raw: str
    verified: bool = False
    snippet: str = ""
    content_type: str = "text"
    image_path: str = ""
    caption: str = ""
    source_path: str = ""

    def label(self) -> str:
        return f"[{self.document_name}, Page {self.page}, Line {self.line_start}-{self.line_end}]"


@dataclass(slots=True)
class RetrievalBundle:
    """The hybrid context assembled for one question."""

    query: str
    chunks: list[Retrieved] = field(default_factory=list)
    figures: list[Retrieved] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.chunks and not self.figures


@dataclass(slots=True)
class AnswerResult:
    """Final output of a RAG turn."""

    query: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    bundle: RetrievalBundle | None = None
    model: str = ""
    elapsed_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def verified_citations(self) -> list[Citation]:
        return [c for c in self.citations if c.verified]

    @property
    def unverified_citations(self) -> list[Citation]:
        return [c for c in self.citations if not c.verified]
