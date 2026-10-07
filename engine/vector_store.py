"""Persistent ChromaDB wrapper with hybrid (dense + lexical) retrieval.

Two collections are maintained:

``paper_chunks``
    Text and Markdown-table chunks, the main answer context.
``paper_figures``
    Figure and table captions, each carrying ``image_path``, ``page`` and
    ``line_start`` so the UI can open the extracted PNG.

Document ids are deterministic (``<doc_uid>:p<page>:c<index>``), so re-ingesting
a file upserts in place instead of duplicating. All writes are batched to keep
peak memory bounded on a 16 GB machine.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterable, Sequence
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from engine.config import Settings
from engine.embeddings import ChromaEmbeddingFunction, build_embedding_function
from engine.models import Chunk, FigureRecord, Retrieved

LOGGER = logging.getLogger(__name__)

#: Rows pushed to Chroma per add() call.
_WRITE_BATCH = 256

_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")

_STOPWORDS = frozenset(
    """the a an and or of for to in on at by with from as is are was were be been
    it its this that these those which what how why when where who whom can could
    should would may might will shall do does did not no nor but if then than
    there here their them they we you your our us""".split()
)


def _tokenize(text: str) -> set[str]:
    """Lower-case content words used for the lexical half of the hybrid score."""
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


class VectorStoreError(RuntimeError):
    """Raised when the vector database cannot be opened or queried."""


class VectorStore:
    """Thin, typed facade over the two persistent ChromaDB collections."""

    def __init__(self, settings: Settings, embedding_function: ChromaEmbeddingFunction | None = None) -> None:
        self._settings = settings
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._embedder = embedding_function or build_embedding_function(settings)

        try:
            self._client = chromadb.PersistentClient(
                path=str(settings.chroma_dir),
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
        except Exception as exc:
            raise VectorStoreError(f"cannot open ChromaDB at {settings.chroma_dir}: {exc}") from exc

        self._chunks = self._get_collection(settings.text_collection)
        self._figures = self._get_collection(settings.figure_collection)

    def _get_collection(self, name: str) -> chromadb.Collection:
        """Fetch or create a cosine-space collection bound to our embedder."""
        try:
            return self._client.get_or_create_collection(
                name=name,
                embedding_function=self._embedder,  # type: ignore[arg-type]
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise VectorStoreError(f"cannot open collection {name!r}: {exc}") from exc

    # --------------------------------------------------------------- writing
    def replace_document(self, doc_uid: str, chunks: Sequence[Chunk], figures: Sequence[FigureRecord]) -> None:
        """Atomically swap all vectors for one document.

        Existing rows for ``doc_uid`` are deleted first, so a re-ingest after an
        edit never leaves stale chunks behind.
        """
        self.delete_document(doc_uid)
        self.add_chunks(chunks)
        self.add_figures(figures)

    def add_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Add text/table chunks in bounded batches. Returns the count written."""
        written = 0
        for batch in _batched(chunks, _WRITE_BATCH):
            if not batch:
                continue
            try:
                self._chunks.upsert(
                    ids=[c.chunk_id for c in batch],
                    documents=[c.text for c in batch],
                    metadatas=[c.to_metadata() for c in batch],
                )
            except Exception as exc:
                raise VectorStoreError(f"failed writing chunk batch: {exc}") from exc
            written += len(batch)
        return written

    def add_figures(self, figures: Sequence[FigureRecord]) -> int:
        """Index figure captions. Returns the count written."""
        written = 0
        for batch in _batched(figures, _WRITE_BATCH):
            if not batch:
                continue
            try:
                self._figures.upsert(
                    ids=[f.figure_id for f in batch],
                    documents=[_figure_document(f) for f in batch],
                    metadatas=[f.to_metadata() for f in batch],
                )
            except Exception as exc:
                raise VectorStoreError(f"failed writing figure batch: {exc}") from exc
            written += len(batch)
        return written

    def delete_document(self, doc_uid: str) -> None:
        """Remove every chunk and figure belonging to ``doc_uid``."""
        for collection in (self._chunks, self._figures):
            try:
                collection.delete(where={"doc_uid": doc_uid})
            except Exception as exc:  # pragma: no cover - delete of absent id
                LOGGER.debug("Delete for %s in %s: %s", doc_uid, collection.name, exc)

    def reset(self) -> None:
        """Drop and recreate both collections."""
        for name in (self._settings.text_collection, self._settings.figure_collection):
            try:
                self._client.delete_collection(name)
            except Exception:  # pragma: no cover - collection may not exist
                LOGGER.debug("Collection %s absent during reset", name)
        self._chunks = self._get_collection(self._settings.text_collection)
        self._figures = self._get_collection(self._settings.figure_collection)

    # --------------------------------------------------------------- reading
    def counts(self) -> dict[str, int]:
        """Return row counts for both collections."""
        try:
            return {"chunks": self._chunks.count(), "figures": self._figures.count()}
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Could not count collections: %s", exc)
            return {"chunks": 0, "figures": 0}

    def documents(self) -> list[str]:
        """Return the distinct document names currently indexed."""
        try:
            payload = self._chunks.get(include=["metadatas"], limit=100_000)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Could not enumerate documents: %s", exc)
            return []
        names = {str(m.get("document_name", "")) for m in (payload.get("metadatas") or [])}
        return sorted(n for n in names if n)

    def figure_image_paths(self, doc_uid: str) -> list[str]:
        """Return the on-disk image paths recorded for one document's figures.

        Read before deleting a document so the extracted PNGs can be removed
        alongside its vectors instead of being stranded on disk.
        """
        try:
            payload = self._figures.get(
                where={"doc_uid": doc_uid}, include=["metadatas"], limit=100_000
            )
        except Exception as exc:  # pragma: no cover - absent collection
            LOGGER.debug("Could not list figures for %s: %s", doc_uid, exc)
            return []
        return [
            str(meta.get("image_path", ""))
            for meta in (payload.get("metadatas") or [])
            if meta and meta.get("image_path")
        ]

    def query_chunks(
        self,
        query: str,
        *,
        n_results: int,
        candidates: int,
        lexical_weight: float,
        where: dict[str, Any] | None = None,
    ) -> list[Retrieved]:
        """Hybrid search over text and table chunks.

        Dense cosine similarity supplies the candidate set; a lexical
        keyword-overlap score then re-ranks it. Pure vector search alone tends
        to miss exact identifiers (gene names, PDB codes, tool names) that
        matter in this corpus.
        """
        hits = self._query(self._chunks, query, candidates, where)
        return _rerank(query, hits, lexical_weight)[:n_results]

    def query_figures(self, query: str, *, n_results: int) -> list[Retrieved]:
        """Semantic search over figure and table captions."""
        hits = self._query(self._figures, query, max(n_results * 3, 9), None)
        return _rerank(query, hits, 0.25)[:n_results]

    def _query(
        self,
        collection: chromadb.Collection,
        query: str,
        n_results: int,
        where: dict[str, Any] | None,
    ) -> list[Retrieved]:
        """Run one Chroma query and normalise the response shape."""
        if not query.strip():
            return []
        try:
            available = collection.count()
        except Exception:  # pragma: no cover
            available = n_results
        if available == 0:
            return []

        try:
            raw = collection.query(
                query_texts=[query],
                n_results=min(n_results, available),
                where=where or None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"query failed on {collection.name}: {exc}") from exc

        ids = (raw.get("ids") or [[]])[0]
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]

        results: list[Retrieved] = []
        for i, doc_id in enumerate(ids):
            distance = float(dists[i]) if i < len(dists) else 1.0
            results.append(
                Retrieved(
                    id=str(doc_id),
                    text=str(docs[i]) if i < len(docs) else "",
                    metadata=dict(metas[i]) if i < len(metas) and metas[i] else {},
                    distance=distance,
                    dense_score=max(0.0, 1.0 - distance),
                )
            )
        return results


def _figure_document(figure: FigureRecord) -> str:
    """Build the text that represents a figure in the vector space."""
    parts = [figure.label, figure.caption]
    return " ".join(p for p in parts if p).strip() or figure.caption


def _rerank(query: str, hits: list[Retrieved], lexical_weight: float) -> list[Retrieved]:
    """Blend dense similarity with keyword overlap and sort descending."""
    query_tokens = _tokenize(query)
    weight = min(max(lexical_weight, 0.0), 1.0)

    for hit in hits:
        if query_tokens:
            overlap = query_tokens & _tokenize(hit.text)
            # Sub-linear in the number of matched terms, normalised by query size.
            hit.lexical_score = math.log1p(len(overlap)) / math.log1p(len(query_tokens))
        else:
            hit.lexical_score = 0.0
        hit.score = (1.0 - weight) * hit.dense_score + weight * hit.lexical_score

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits


def _batched(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    """Yield ``items`` in slices of at most ``size``."""
    for start in range(0, len(items), size):
        yield items[start : start + size]
