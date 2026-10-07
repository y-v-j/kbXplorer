"""Parsing and verification of LLM-emitted citations.

A 3B model will produce well-formed-looking citations that point at pages it
never saw. Prompting alone does not fix this, so every citation an answer makes
is parsed and checked against the metadata of the chunks that were actually
retrieved. A citation is *verified* only when its document, page, and line range
all correspond to real retrieved context.

Unverified citations are reported, never silently trusted — the TUI marks them
and the CLI lists them under a warning.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from engine.models import Citation, Retrieved

LOGGER = logging.getLogger(__name__)

#: Canonical form: ``[DocName.pdf, Page 4, Line 12-18]``.
#: Tolerates 'Lines', an en/em dash, a missing end line, and stray whitespace.
CITATION_RE = re.compile(
    r"\[\s*(?P<doc>[^\[\],]+?\.pdf)\s*,\s*"
    r"[Pp]ages?\s*(?P<page>\d{1,5})\s*,\s*"
    r"[Ll]ines?\s*(?P<start>\d{1,6})\s*"
    r"(?:[-‐-―]\s*(?P<end>\d{1,6}))?\s*\]"
)


def parse_citations(text: str) -> list[Citation]:
    """Extract every citation-shaped span from an answer.

    Duplicates are collapsed, preserving first-seen order.
    """
    seen: set[tuple[str, int, int, int]] = set()
    citations: list[Citation] = []

    for match in CITATION_RE.finditer(text):
        document = match.group("doc").strip()
        page = int(match.group("page"))
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        if end < start:
            start, end = end, start

        key = (document.lower(), page, start, end)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            Citation(
                document_name=document,
                page=page,
                line_start=start,
                line_end=end,
                raw=match.group(0),
            )
        )
    return citations


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int, *, slack: int) -> bool:
    """Return ``True`` if two inclusive line ranges overlap within ``slack``."""
    return a_start <= b_end + slack and b_start <= a_end + slack


def verify_citations(
    citations: Sequence[Citation],
    retrieved: Sequence[Retrieved],
    *,
    line_slack: int = 5,
) -> list[Citation]:
    """Check each citation against the retrieved context and enrich it in place.

    A citation is verified when a retrieved chunk shares its document name and
    page, and its line range overlaps (within ``line_slack`` lines, which
    absorbs a model quoting a slightly narrower span than the chunk).

    Verified citations gain the chunk's snippet, content type, and image path so
    the reference pane can render them without another lookup.

    Args:
        citations: Citations parsed from the answer.
        retrieved: The chunks and figures that were actually supplied as context.
        line_slack: Tolerance in lines when comparing ranges.

    Returns:
        The same citation objects, mutated with verification results.
    """
    by_document: dict[str, list[Retrieved]] = {}
    for hit in retrieved:
        by_document.setdefault(hit.document_name.lower(), []).append(hit)

    for citation in citations:
        candidates = by_document.get(citation.document_name.lower(), [])
        if not candidates:
            # Tolerate the model dropping or mangling the .pdf suffix.
            stem = citation.document_name.lower().removesuffix(".pdf")
            for name, hits in by_document.items():
                if name.removesuffix(".pdf") == stem:
                    candidates = hits
                    break

        match: Retrieved | None = None
        for hit in candidates:
            if hit.page_number != citation.page:
                continue
            if _overlaps(citation.line_start, citation.line_end, hit.line_start, hit.line_end, slack=line_slack):
                match = hit
                break
        if match is None:
            # Same page is still a partial hit; prefer it over nothing.
            for hit in candidates:
                if hit.page_number == citation.page:
                    match = hit
                    break

        if match is None:
            citation.verified = False
            continue

        citation.verified = _overlaps(
            citation.line_start, citation.line_end, match.line_start, match.line_end, slack=line_slack
        )
        citation.snippet = match.text[:700]
        citation.content_type = match.content_type
        citation.image_path = match.image_path
        citation.caption = match.caption
        citation.source_path = str(match.metadata.get("source_path", "") or "")

    return list(citations)


def citations_from_context(retrieved: Sequence[Retrieved]) -> list[Citation]:
    """Build citations directly from retrieved context.

    Used for the reference pane when the LLM is unavailable, and as the
    fallback when an answer cites nothing at all.
    """
    citations: list[Citation] = []
    for hit in retrieved:
        citations.append(
            Citation(
                document_name=hit.document_name,
                page=hit.page_number,
                line_start=hit.line_start,
                line_end=hit.line_end,
                raw=hit.citation_label(),
                verified=True,
                snippet=hit.text[:700],
                content_type=hit.content_type,
                image_path=hit.image_path,
                caption=hit.caption,
                source_path=str(hit.metadata.get("source_path", "") or ""),
            )
        )
    return citations


def annotate_unverified(text: str, citations: Sequence[Citation]) -> str:
    """Flag unverifiable citations inline so a reader cannot mistake them.

    The citation text is preserved and suffixed with ``[unverified]`` rather
    than removed, keeping the answer readable while making the gap explicit.
    """
    result = text
    for citation in citations:
        if citation.verified or not citation.raw:
            continue
        if f"{citation.raw} [unverified]" in result:
            continue
        result = result.replace(citation.raw, f"{citation.raw} [unverified]")
    return result


def coverage_ratio(citations: Sequence[Citation]) -> float:
    """Return the fraction of citations that were verified (1.0 when none)."""
    if not citations:
        return 1.0
    return sum(1 for c in citations if c.verified) / len(citations)
