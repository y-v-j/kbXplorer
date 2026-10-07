"""Embedding backends for the knowledge base.

Both backends serve the *same* model — ``sentence-transformers/all-MiniLM-L6-v2``
(384 dimensions, cosine space). They differ only in runtime:

``onnx`` (default)
    ChromaDB's bundled quantised ONNX build. ~150 MB resident, no PyTorch, and
    the model is cached under ``~/.cache/chroma`` so it works offline after the
    first run. This is the right default on a 16 GB CPU-only machine.

``sentence-transformers``
    The PyTorch build. Higher fidelity in the last decimal place, but pulls in
    torch and roughly 700 MB of resident memory. Opt in via
    ``KB_EMBEDDING_BACKEND=sentence-transformers`` when RAM is plentiful.

Switching backends changes the vector space, so :func:`backend_fingerprint` is
recorded in the manifest and the CLI refuses to mix vectors from two backends
in one collection without an explicit re-index.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from engine.config import Settings

LOGGER = logging.getLogger(__name__)

#: Dimensionality of all-MiniLM-L6-v2.
EMBEDDING_DIM = 384


@runtime_checkable
class ChromaEmbeddingFunction(Protocol):
    """The callable interface ChromaDB expects from an embedding function."""

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002 - Chroma's name
        ...


class EmbeddingUnavailableError(RuntimeError):
    """Raised when the requested embedding backend cannot be constructed."""


def backend_fingerprint(settings: Settings) -> str:
    """Return a string identifying the active vector space.

    Used to detect a backend change that would invalidate stored embeddings.
    """
    return f"{settings.embedding_backend}:{settings.embedding_model}:{EMBEDDING_DIM}"


def build_embedding_function(settings: Settings) -> ChromaEmbeddingFunction:
    """Construct the ChromaDB embedding function for the configured backend.

    Args:
        settings: Active configuration.

    Returns:
        A ChromaDB-compatible embedding function.

    Raises:
        EmbeddingUnavailableError: If the backend's dependencies are missing or
            the model cannot be loaded.
    """
    backend = settings.embedding_backend.strip().lower()

    if backend in {"onnx", "default", "chroma"}:
        try:
            from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

            LOGGER.info("Embedding backend: ONNX all-MiniLM-L6-v2 (low memory)")
            return ONNXMiniLM_L6_V2()
        except Exception as exc:  # pragma: no cover - depends on environment
            raise EmbeddingUnavailableError(
                "Could not initialise the ONNX MiniLM embedder. The model is "
                "downloaded once to ~/.cache/chroma; check network access on "
                f"first run. Original error: {exc}"
            ) from exc

    if backend in {"sentence-transformers", "sentence_transformers", "st", "torch"}:
        try:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

            LOGGER.info("Embedding backend: sentence-transformers/%s", settings.embedding_model)
            return SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model)
        except Exception as exc:
            raise EmbeddingUnavailableError(
                "Could not initialise the sentence-transformers embedder. Install "
                "it with `pip install sentence-transformers` (CPU torch), or set "
                f"KB_EMBEDDING_BACKEND=onnx. Original error: {exc}"
            ) from exc

    raise EmbeddingUnavailableError(
        f"Unknown embedding backend {settings.embedding_backend!r}. "
        "Use 'onnx' or 'sentence-transformers'."
    )


def embed_texts(function: ChromaEmbeddingFunction, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts, normalising the return type across backends."""
    if not texts:
        return []
    vectors: Any = function(texts)
    return [list(map(float, vector)) for vector in vectors]
