"""Retrieval-augmented generation over the local paper corpus.

The pipeline is: embed the question -> hybrid retrieval of text chunks, tables
and figure captions -> build a context block in which every passage is labelled
with its exact citation -> instruct the local LLM to answer only from that
block -> parse and verify every citation it emits.

Answers stream, because on a CPU-only machine time-to-first-token is what makes
the interface feel alive; a 3B model runs at roughly 8-15 tok/s here.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from engine.citations import (
    annotate_unverified,
    citations_from_context,
    parse_citations,
    verify_citations,
)
from engine.config import Settings
from engine.models import AnswerResult, RetrievalBundle, Retrieved
from engine.ollama_client import OllamaConnector, OllamaUnavailableError
from engine.vector_store import VectorStore

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise research assistant answering questions about a \
corpus of scientific papers. You have been given numbered context passages \
extracted from those papers.

Rules you must follow without exception:
1. Answer ONLY using the provided context passages. The context is your entire \
world; you have no other knowledge.
2. If the context does not contain the answer, say exactly: "The provided papers \
do not contain enough information to answer this question." Do not speculate.
3. Every factual claim must carry a citation in this EXACT format: \
[DocName.pdf, Page X, Line Y-Z]
4. Copy the citation values verbatim from the "CITATION:" line of the passage \
you used. Never invent, adjust, or interpolate a document name, page number, or \
line number.
5. Multiple citations for one claim are written back to back: \
[A.pdf, Page 1, Line 2-5] [B.pdf, Page 3, Line 9-14]
6. Be concise and technical. Do not pad the answer or restate the question."""

EventKind = Literal["context", "token", "done", "error"]


@dataclass(slots=True)
class RagEvent:
    """One step of a streaming RAG turn."""

    kind: EventKind
    text: str = ""
    bundle: RetrievalBundle | None = None
    result: AnswerResult | None = None


class RagEngine:
    """Hybrid retrieval plus grounded, citation-checked generation."""

    def __init__(self, settings: Settings, store: VectorStore, connector: OllamaConnector) -> None:
        self._settings = settings
        self._store = store
        self._ollama = connector

    # ------------------------------------------------------------ retrieval
    def retrieve(self, query: str) -> RetrievalBundle:
        """Run hybrid retrieval for ``query`` across chunks and figures."""
        settings = self._settings
        chunks = self._store.query_chunks(
            query,
            n_results=settings.top_k_text,
            candidates=settings.retrieval_candidates,
            lexical_weight=settings.lexical_weight,
        )
        figures = self._store.query_figures(query, n_results=settings.top_k_figures)
        return RetrievalBundle(query=query, chunks=chunks, figures=figures)

    # --------------------------------------------------------------- prompt
    def build_context(self, bundle: RetrievalBundle) -> str:
        """Render retrieved passages into a citation-labelled context block.

        Each passage carries an explicit ``CITATION:`` line holding the exact
        string the model must copy, which measurably reduces malformed and
        hallucinated references from small models.
        """
        budget = self._settings.max_context_chars
        blocks: list[str] = []
        used = 0

        for index, hit in enumerate(bundle.chunks, start=1):
            label = "TABLE" if hit.content_type == "table" else "PASSAGE"
            body = hit.text.strip()
            entry = (
                f"[{index}] {label}\n"
                f"CITATION: {hit.citation_label()}\n"
                f"CONTENT:\n{body}\n"
            )
            if used + len(entry) > budget and blocks:
                break
            blocks.append(entry)
            used += len(entry)

        for offset, figure in enumerate(bundle.figures, start=len(blocks) + 1):
            caption = figure.caption or figure.text
            entry = (
                f"[{offset}] FIGURE\n"
                f"CITATION: {figure.citation_label()}\n"
                f"CAPTION: {caption.strip()}\n"
            )
            if used + len(entry) > budget and blocks:
                break
            blocks.append(entry)
            used += len(entry)

        return "\n".join(blocks)

    def build_user_prompt(self, query: str, bundle: RetrievalBundle) -> str:
        """Assemble the user turn: context block followed by the question."""
        context = self.build_context(bundle)
        return (
            "CONTEXT PASSAGES\n"
            "================\n"
            f"{context}\n"
            "================\n\n"
            f"QUESTION: {query}\n\n"
            "Answer using only the context above, citing every claim in the "
            "format [DocName.pdf, Page X, Line Y-Z]."
        )

    # ----------------------------------------------------------- generation
    def answer_stream(self, query: str) -> Iterator[RagEvent]:
        """Stream a grounded answer for ``query``.

        Yields a ``context`` event once retrieval completes (so the reference
        pane can populate immediately), then ``token`` events, then a final
        ``done`` event carrying the verified :class:`AnswerResult`. Failures
        surface as an ``error`` event rather than an exception, so the caller
        never has to wrap the loop.
        """
        started = time.monotonic()
        warnings: list[str] = []

        try:
            bundle = self.retrieve(query)
        except Exception as exc:
            LOGGER.exception("Retrieval failed")
            yield RagEvent(kind="error", text=f"Retrieval failed: {exc}")
            return

        yield RagEvent(kind="context", bundle=bundle)

        if bundle.is_empty:
            message = (
                "No indexed content matched that question. "
                "Check that ingestion has run (`python main.py status`)."
            )
            yield RagEvent(kind="token", text=message)
            yield RagEvent(
                kind="done",
                result=AnswerResult(
                    query=query,
                    answer=message,
                    bundle=bundle,
                    elapsed_seconds=time.monotonic() - started,
                    warnings=["empty retrieval"],
                ),
            )
            return

        model = self._ollama.resolve_model()
        prompt = self.build_user_prompt(query, bundle)
        pieces: list[str] = []

        try:
            for fragment in self._ollama.stream_chat(SYSTEM_PROMPT, prompt, model=model):
                pieces.append(fragment)
                yield RagEvent(kind="token", text=fragment)
        except OllamaUnavailableError as exc:
            LOGGER.warning("Ollama unavailable: %s", exc)
            result = self._retrieval_only_result(query, bundle, str(exc), started)
            yield RagEvent(kind="token", text=result.answer)
            yield RagEvent(kind="done", result=result)
            return

        answer = "".join(pieces).strip()
        context_hits: list[Retrieved] = [*bundle.chunks, *bundle.figures]
        citations = verify_citations(parse_citations(answer), context_hits)

        if not citations:
            warnings.append("The model returned no citations; showing retrieved sources instead.")
            citations = citations_from_context(bundle.chunks[:3])
        else:
            unverified = [c for c in citations if not c.verified]
            if unverified:
                warnings.append(
                    f"{len(unverified)} of {len(citations)} citations could not be matched "
                    "to retrieved context and are marked [unverified]."
                )
                answer = annotate_unverified(answer, citations)

        yield RagEvent(
            kind="done",
            result=AnswerResult(
                query=query,
                answer=answer,
                citations=citations,
                bundle=bundle,
                model=model,
                elapsed_seconds=time.monotonic() - started,
                warnings=warnings,
            ),
        )

    def answer(self, query: str) -> AnswerResult:
        """Blocking convenience wrapper returning only the final result."""
        result: AnswerResult | None = None
        for event in self.answer_stream(query):
            if event.kind == "done" and event.result is not None:
                result = event.result
            elif event.kind == "error":
                return AnswerResult(query=query, answer=event.text, warnings=[event.text])
        return result or AnswerResult(query=query, answer="", warnings=["no result produced"])

    # ------------------------------------------------------------- fallback
    def _retrieval_only_result(
        self, query: str, bundle: RetrievalBundle, error: str, started: float
    ) -> AnswerResult:
        """Build a useful answer from retrieval alone when the LLM is down."""
        lines = [
            "The local LLM is unavailable, so no synthesised answer was generated.",
            f"Reason: {error}",
            "",
            "Start it with `ollama serve`, then ask again. The most relevant "
            "passages retrieved for this question are:",
            "",
        ]
        for index, hit in enumerate(bundle.chunks[:5], start=1):
            snippet = " ".join(hit.text.split())[:280]
            lines.append(f"{index}. {hit.citation_label()}")
            lines.append(f"   {snippet}...")
            lines.append("")

        return AnswerResult(
            query=query,
            answer="\n".join(lines).strip(),
            citations=citations_from_context(bundle.chunks[:5]),
            bundle=bundle,
            elapsed_seconds=time.monotonic() - started,
            warnings=["ollama unavailable - retrieval-only mode"],
        )
