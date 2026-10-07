"""Layout-aware PDF parsing with synthetic, citable line numbers.

Design notes
------------
**Line numbers.** PDFs store positioned glyph runs, not lines. This module
synthesises a per-page line numbering by collecting text spans, detecting the
column layout, sorting into reading order, and numbering from 1. The result is
deterministic for a given file and :data:`PARSER_VERSION`, which is what makes
``[doc.pdf, Page 4, Line 12-18]`` a reproducible coordinate.

**Figures.** ``page.get_images()`` only returns embedded *raster* XObjects, so
it silently misses vector artwork — which is most plots in a scientific paper.
The primary extraction path here is therefore caption-anchored: locate a
``Figure N``/``Table N`` caption, infer the graphic region adjacent to it, and
render that region to PNG. This captures vector and raster figures alike.
Embedded rasters not covered by a rendered region are additionally exported.

**Tables.** ``pdfplumber`` produces the best Markdown tables but is 10-20x
slower than PyMuPDF, so PyMuPDF's fast ``find_tables()`` is used as a detector
and pdfplumber runs only on the pages that actually contain a table.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from engine.config import Settings
from engine.manifest import PARSER_VERSION
from engine.models import Chunk, FigureRecord, PageLine, ParsedDocument, document_uid

LOGGER = logging.getLogger(__name__)

#: Matches the opening of a figure/table caption.
CAPTION_RE = re.compile(
    r"^\s*(?P<kind>figure|fig\.?|table|scheme|chart|plate|exhibit)\s*"
    r"(?P<num>\d{1,3}[a-z]?|[ivxlc]{1,6})\b\s*[.:)–—-]?",
    re.IGNORECASE,
)

#: Fraction of page width a line must straddle to count as "full width".
_SPAN_TOLERANCE = 0.08
#: Minimum share of lines on each side before a page is called two-column.
_COLUMN_SHARE = 0.15
#: Vertical rounding (points) applied when sorting lines into reading order.
_Y_QUANTUM = 3.0


class PdfParseError(RuntimeError):
    """Raised when a PDF cannot be opened or is structurally unusable."""


@dataclass(slots=True)
class _PageLayout:
    """Per-page geometry derived once and reused by every extractor."""

    lines: list[PageLine]
    columns: list[tuple[float, float]]
    width: float
    height: float

    def line_range_for_rect(self, rect: pymupdf.Rect) -> tuple[int, int]:
        """Return the (start, end) line numbers overlapping ``rect``.

        Falls back to the line nearest the rectangle when nothing intersects,
        so a figure always carries a usable line anchor.
        """
        hits = [
            line.number
            for line in self.lines
            if not (line.y1 < rect.y0 - 2 or line.y0 > rect.y1 + 2)
            and not (line.x1 < rect.x0 - 2 or line.x0 > rect.x1 + 2)
        ]
        if hits:
            return min(hits), max(hits)
        if not self.lines:
            return 0, 0
        nearest = min(self.lines, key=lambda ln: abs((ln.y0 + ln.y1) / 2 - (rect.y0 + rect.y1) / 2))
        return nearest.number, nearest.number


def _order_lines(
    raw: Sequence[tuple[pymupdf.Rect, str]], width: float, height: float
) -> tuple[list[int], list[int], list[tuple[float, float]]]:
    """Sort a page's text lines into reading order.

    A whole-page column vote is not enough: the first page of a typical paper
    carries a full-width title block above a two-column body, and a single
    verdict for the page either scrambles the columns or mis-orders the title.
    So the page is segmented into horizontal *bands* instead. A line that
    straddles the vertical midline (title, running head, a section heading
    spanning both columns, a wide equation, a footer) closes the current band
    and is emitted on its own; runs of non-straddling lines form a column band
    whose left column is emitted in full before its right column.

    Returns:
        ``(order, column_per_line, column_bounds)`` where ``order`` holds
        indices into ``raw`` in reading order and a column of ``-1`` marks a
        full-width line.
    """
    if not raw:
        return [], [], [(0.0, width)]

    mid = width / 2.0
    tol = width * _SPAN_TOLERANCE
    total = len(raw)

    def classify(rect: pymupdf.Rect) -> int:
        """-1 = full width, 0 = left column, 1 = right column."""
        if rect.x0 < mid - tol and rect.x1 > mid + tol:
            return -1
        return 0 if (rect.x0 + rect.x1) / 2 < mid else 1

    columns = [classify(rect) for rect, _ in raw]
    left_count = columns.count(0)
    right_count = columns.count(1)

    by_y = sorted(
        range(total),
        key=lambda i: (round(raw[i][0].y0 / _Y_QUANTUM), raw[i][0].x0),
    )

    # Without a healthy population on both sides the page is single-column;
    # plain top-to-bottom order is correct and safer than forcing a split.
    if left_count < total * _COLUMN_SHARE or right_count < total * _COLUMN_SHARE:
        return by_y, [0] * total, [(0.0, width)]

    left_rects = [raw[i][0] for i in range(total) if columns[i] == 0]
    right_rects = [raw[i][0] for i in range(total) if columns[i] == 1]
    bounds = [
        (min(r.x0 for r in left_rects), max(r.x1 for r in left_rects)),
        (min(r.x0 for r in right_rects), max(r.x1 for r in right_rects)),
    ]

    order: list[int] = []
    band: list[int] = []

    def flush_band() -> None:
        """Emit the pending column band: the whole left column, then the right."""
        if not band:
            return
        order.extend(i for i in band if columns[i] == 0)
        order.extend(i for i in band if columns[i] == 1)
        band.clear()

    for index in by_y:
        if columns[index] == -1:
            flush_band()
            order.append(index)
        else:
            band.append(index)
    flush_band()

    return order, columns, bounds


def build_page_layout(page: pymupdf.Page) -> _PageLayout:
    """Extract text lines from a page and number them in reading order."""
    width = float(page.rect.width)
    height = float(page.rect.height)

    raw: list[tuple[pymupdf.Rect, str]] = []
    try:
        payload = page.get_text("dict")
    except Exception as exc:  # pragma: no cover - corrupt page
        LOGGER.debug("get_text failed on page %s: %s", page.number, exc)
        return _PageLayout(lines=[], columns=[(0.0, width)], width=width, height=height)

    for block in payload.get("blocks", []):
        if block.get("type") != 0:  # 0 == text block
            continue
        for line in block.get("lines", []):
            # Skip rotated text: vertical watermarks ("Downloaded from ...")
            # and margin notes are not part of the reading flow, and they
            # fragment the bands that reading order is derived from.
            direction = line.get("dir", (1.0, 0.0))
            if abs(float(direction[1])) > 0.1:
                continue
            text = "".join(span.get("text", "") for span in line.get("spans", []))
            if not text.strip():
                continue
            bbox = line.get("bbox")
            if not bbox:
                continue
            raw.append((pymupdf.Rect(bbox), text.strip()))

    order, column_idx, bounds = _order_lines(raw, width, height)

    lines: list[PageLine] = []
    for number, idx in enumerate(order, start=1):
        rect, text = raw[idx]
        lines.append(
            PageLine(
                number=number,
                text=text,
                x0=float(rect.x0),
                y0=float(rect.y0),
                x1=float(rect.x1),
                y1=float(rect.y1),
                column=column_idx[idx],
            )
        )
    return _PageLayout(lines=lines, columns=bounds, width=width, height=height)


def chunk_page_lines(
    lines: Sequence[PageLine],
    *,
    target_chars: int,
    overlap_chars: int,
    min_chars: int,
) -> list[tuple[int, int, str]]:
    """Group consecutive lines into overlapping chunks.

    Chunks never span pages, so ``page_number`` on a citation is always exact.

    Returns:
        A list of ``(line_start, line_end, text)`` tuples.
    """
    if not lines:
        return []

    chunks: list[tuple[int, int, str]] = []
    start = 0
    n = len(lines)

    while start < n:
        buffer: list[str] = []
        size = 0
        end = start
        while end < n and size < target_chars:
            text = lines[end].text
            buffer.append(text)
            size += len(text) + 1
            end += 1

        body = "\n".join(buffer).strip()
        if body:
            chunks.append((lines[start].number, lines[end - 1].number, body))

        if end >= n:
            break

        # Step back far enough to create the requested character overlap.
        back = 0
        acc = 0
        while end - back - 1 > start and acc < overlap_chars:
            acc += len(lines[end - back - 1].text) + 1
            back += 1
        start = max(start + 1, end - back)

    # Fold a runt tail chunk into its predecessor.
    if len(chunks) > 1 and len(chunks[-1][2]) < min_chars:
        prev_start, _, prev_text = chunks[-2]
        last_start, last_end, last_text = chunks.pop()
        chunks[-1] = (prev_start, last_end, f"{prev_text}\n{last_text}")
    return chunks


def _caption_blocks(page: pymupdf.Page, layout: _PageLayout) -> list[tuple[pymupdf.Rect, str, str, str]]:
    """Find caption blocks on a page.

    Returns:
        Tuples of ``(rect, caption_text, kind, label)`` where ``kind`` is one of
        ``"figure"`` or ``"table"``.
    """
    found: list[tuple[pymupdf.Rect, str, str, str]] = []
    try:
        blocks = page.get_text("blocks")
    except Exception:  # pragma: no cover - corrupt page
        return found

    for block in blocks:
        if len(block) < 7 or block[6] != 0:
            continue
        text = (block[4] or "").strip()
        if not text:
            continue
        first_line = text.splitlines()[0]
        match = CAPTION_RE.match(first_line)
        if not match:
            continue
        raw_kind = match.group("kind").lower().rstrip(".")
        kind = "table" if raw_kind == "table" else "figure"
        label = f"{match.group('kind').strip().rstrip('.').title()} {match.group('num')}"
        caption = " ".join(text.split())[:800]
        found.append((pymupdf.Rect(block[:4]), caption, kind, label))
    return found


def _graphics_rects(page: pymupdf.Page, drawings: Sequence[dict]) -> list[pymupdf.Rect]:
    """Return the bounding boxes of every vector drawing and raster image."""
    rects: list[pymupdf.Rect] = []
    for drawing in drawings:
        drect = drawing.get("rect")
        if drect is not None:
            rects.append(pymupdf.Rect(drect))
    try:
        for info in page.get_image_info():
            ibox = info.get("bbox")
            if ibox:
                rects.append(pymupdf.Rect(ibox))
    except Exception:  # pragma: no cover
        pass
    return rects


def _column_bounds_for(rect: pymupdf.Rect, layout: _PageLayout) -> tuple[float, float]:
    """Return the x-range of the column that ``rect`` sits in."""
    if len(layout.columns) == 1:
        return layout.columns[0]
    centre = (rect.x0 + rect.x1) / 2
    best = min(layout.columns, key=lambda b: abs((b[0] + b[1]) / 2 - centre))
    # A caption wider than its column (full-width figure) keeps its own span.
    if rect.x1 - rect.x0 > (best[1] - best[0]) * 1.35:
        return min(b[0] for b in layout.columns), max(b[1] for b in layout.columns)
    return best


def _clamp_vertically(
    x0: float,
    x1: float,
    caption_rect: pymupdf.Rect,
    layout: _PageLayout,
    *,
    below: bool,
) -> pymupdf.Rect | None:
    """Bound a graphic band vertically by the nearest text spanning ``[x0, x1]``."""
    pad = 3.0
    if below:
        blockers = [
            ln.y0
            for ln in layout.lines
            if ln.y0 >= caption_rect.y1 + 1 and not (ln.x1 < x0 - 5 or ln.x0 > x1 + 5)
        ]
        bottom = min(blockers) - pad if blockers else layout.height - pad
        if bottom - caption_rect.y1 < 35:
            return None
        return pymupdf.Rect(x0, caption_rect.y1 + 1, x1, bottom)

    blockers = [
        ln.y1
        for ln in layout.lines
        if ln.y1 <= caption_rect.y0 - 1 and not (ln.x1 < x0 - 5 or ln.x0 > x1 + 5)
    ]
    top = max(blockers) + pad if blockers else pad
    if caption_rect.y0 - top < 35:
        return None
    return pymupdf.Rect(x0, top, x1, caption_rect.y0 - 1)


def _figure_region(
    caption_rect: pymupdf.Rect,
    layout: _PageLayout,
    graphics: Sequence[pymupdf.Rect],
    *,
    prefer_below: bool,
) -> pymupdf.Rect | None:
    """Infer the graphic area belonging to a caption.

    Figures normally sit above their caption and tables below, so the preferred
    side is tried first. The band is initially bounded by the caption's own
    column, then widened to the true horizontal extent of the artwork, because
    multi-column papers routinely run a figure across the full page while its
    caption stays inside one column. After widening the vertical bounds are
    recomputed over the wider span, so a running header or footer cannot creep
    into the render. If that recomputation collapses the band -- the signature
    of a genuinely column-width figure sitting beside prose -- the narrow
    region is kept instead.
    """
    col_x0, col_x1 = _column_bounds_for(caption_rect, layout)
    pad = 3.0

    for below in ((True, False) if prefer_below else (False, True)):
        narrow = _clamp_vertically(col_x0 - pad, col_x1 + pad, caption_rect, layout, below=below)
        if narrow is None:
            continue

        inside = [g for g in graphics if narrow.intersects(g)]
        if not inside:
            continue

        union_x0 = min(g.x0 for g in inside)
        union_x1 = max(g.x1 for g in inside)
        if union_x0 < narrow.x0 - 2 or union_x1 > narrow.x1 + 2:
            wide_x0 = max(0.0, min(narrow.x0, union_x0) - pad)
            wide_x1 = min(layout.width, max(narrow.x1, union_x1) + pad)
            widened = _clamp_vertically(wide_x0, wide_x1, caption_rect, layout, below=below)
            if widened is not None and widened.height >= max(35.0, narrow.height * 0.5):
                return widened
        return narrow
    return None


def extract_figures_for_page(
    page: pymupdf.Page,
    layout: _PageLayout,
    *,
    doc_uid: str,
    document_name: str,
    source_path: str,
    page_number: int,
    settings: Settings,
    start_index: int,
) -> list[FigureRecord]:
    """Extract caption-anchored figure/table regions from one page as PNGs."""
    records: list[FigureRecord] = []
    captions = _caption_blocks(page, layout)
    if not captions:
        return records

    try:
        drawings = page.get_drawings()
    except Exception:  # pragma: no cover
        drawings = []

    graphics = _graphics_rects(page, drawings)
    rendered: list[pymupdf.Rect] = []
    rendered_labels: set[str] = set()
    index = start_index

    for caption_rect, caption, kind, label in captions:
        if len(records) >= settings.figure_max_per_page:
            break
        region = _figure_region(caption_rect, layout, graphics, prefer_below=(kind == "table"))
        if region is None:
            continue
        if region.get_area() < settings.figure_min_area:
            continue
        if any(region.intersects(prev) and prev.get_area() > 0 for prev in rendered):
            continue

        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(document_name).stem)[:60]
        filename = f"{stem}_{doc_uid[:8]}_p{page_number:04d}_f{index:03d}.png"
        out_path = settings.images_dir / filename
        try:
            pixmap = page.get_pixmap(clip=region, dpi=settings.figure_dpi, alpha=False)
            pixmap.save(str(out_path))
            pixmap = None  # release immediately; pixmaps are the memory hot spot
        except Exception as exc:
            LOGGER.debug("Figure render failed (%s p%d): %s", document_name, page_number, exc)
            continue

        line_start, line_end = layout.line_range_for_rect(caption_rect)
        records.append(
            FigureRecord(
                doc_uid=doc_uid,
                document_name=document_name,
                source_path=source_path,
                page_number=page_number,
                line_start=line_start,
                line_end=line_end,
                caption=caption,
                image_path=str(out_path.resolve()),
                kind="table" if kind == "table" else "figure",
                figure_index=index,
                label=label,
            )
        )
        rendered.append(region)
        if label:
            rendered_labels.add(label.lower())
        index += 1

    if settings.extract_embedded_images:
        records.extend(
            _extract_embedded_images(
                page,
                layout,
                doc_uid=doc_uid,
                document_name=document_name,
                source_path=source_path,
                page_number=page_number,
                settings=settings,
                start_index=index,
                covered=rendered,
                captions=captions,
                rendered_labels=rendered_labels,
            )
        )
    return records


def _extract_embedded_images(
    page: pymupdf.Page,
    layout: _PageLayout,
    *,
    doc_uid: str,
    document_name: str,
    source_path: str,
    page_number: int,
    settings: Settings,
    start_index: int,
    covered: Sequence[pymupdf.Rect],
    captions: Sequence[tuple[pymupdf.Rect, str, str, str]],
    rendered_labels: set[str],
) -> list[FigureRecord]:
    """Export embedded raster images not already represented by a rendered region.

    A raster is skipped when it overlaps a rendered region *or* when its nearest
    caption was already rendered, which prevents one figure being indexed twice.
    """
    records: list[FigureRecord] = []
    index = start_index
    document = page.parent

    try:
        image_list = page.get_images(full=True)
    except Exception:  # pragma: no cover
        return records

    for img in image_list:
        if len(records) + len(covered) >= settings.figure_max_per_page:
            break
        xref = img[0]
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            continue
        if not rects:
            continue
        rect = rects[0]
        if any(rect.intersects(prev) for prev in covered):
            continue

        try:
            base = document.extract_image(xref)
        except Exception:
            continue
        width, height = int(base.get("width", 0)), int(base.get("height", 0))
        if width * height < settings.embedded_image_min_pixels:
            continue

        caption, label = "", ""
        nearest: float | None = None
        for crect, ctext, _kind, clabel in captions:
            distance = min(abs(crect.y0 - rect.y1), abs(rect.y0 - crect.y1))
            if distance < 90 and (nearest is None or distance < nearest):
                nearest, caption, label = distance, ctext, clabel

        # The rendered region for this caption already represents the figure.
        if label and label.lower() in rendered_labels:
            continue
        # Uncaptioned artwork is usually a logo or rule; demand a larger footprint.
        if not caption:
            if width * height < settings.embedded_image_min_pixels * 4:
                continue
            caption = f"Untitled image on page {page_number} of {document_name}"

        ext = str(base.get("ext", "png")).lower()
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(document_name).stem)[:60]
        out_path = settings.images_dir / f"{stem}_{doc_uid[:8]}_p{page_number:04d}_img{index:03d}.{ext}"
        try:
            out_path.write_bytes(base["image"])
        except Exception as exc:
            LOGGER.debug("Embedded image write failed: %s", exc)
            continue

        line_start, line_end = layout.line_range_for_rect(rect)
        records.append(
            FigureRecord(
                doc_uid=doc_uid,
                document_name=document_name,
                source_path=source_path,
                page_number=page_number,
                line_start=line_start,
                line_end=line_end,
                caption=caption,
                image_path=str(out_path.resolve()),
                kind="image",
                figure_index=index,
                label=label,
            )
        )
        index += 1
    return records


def _table_to_markdown(rows: Sequence[Sequence[str | None]], *, max_rows: int) -> str:
    """Render an extracted table as a GitHub-flavoured Markdown table."""
    cleaned: list[list[str]] = []
    for row in rows[:max_rows]:
        cells = [" ".join((cell or "").split()) for cell in row]
        if any(cells):
            cleaned.append(cells)
    if len(cleaned) < 2:
        return ""

    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]

    # Drop columns that are empty in every row - pdfplumber emits many on
    # ruled scientific tables and they make the Markdown unreadable.
    keep = [i for i in range(width) if any(row[i] for row in cleaned)]
    if len(keep) < 2:
        return ""
    cleaned = [[row[i] for i in keep] for row in cleaned]

    header, *body = cleaned
    header = [cell or f"col{i + 1}" for i, cell in enumerate(header)]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(out)


def _detect_table_pages(document: pymupdf.Document) -> dict[int, list[pymupdf.Rect]]:
    """Cheaply locate pages containing tables using PyMuPDF's table finder."""
    hits: dict[int, list[pymupdf.Rect]] = {}
    for index in range(document.page_count):
        try:
            page = document.load_page(index)
            finder = page.find_tables()
            rects = [pymupdf.Rect(table.bbox) for table in finder.tables]
        except Exception:  # pragma: no cover - table finder is best-effort
            continue
        if rects:
            hits[index] = rects
    return hits


def _extract_tables(
    pdf_path: Path,
    table_pages: dict[int, list[pymupdf.Rect]],
    layouts: dict[int, _PageLayout],
    *,
    doc_uid: str,
    document_name: str,
    source_path: str,
    settings: Settings,
    chunk_counter: dict[int, int],
) -> tuple[list[Chunk], list[str]]:
    """Extract Markdown tables with pdfplumber, only on pages known to have one."""
    chunks: list[Chunk] = []
    warnings: list[str] = []
    if not table_pages:
        return chunks, warnings

    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover
        return chunks, [f"pdfplumber unavailable: {exc}"]

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_index, rects in sorted(table_pages.items()):
                if page_index >= len(pdf.pages):
                    continue
                try:
                    plumber_page = pdf.pages[page_index]
                    tables = plumber_page.extract_tables()
                except Exception as exc:
                    warnings.append(f"table extraction failed on page {page_index + 1}: {exc}")
                    continue

                layout = layouts.get(page_index)
                page_number = page_index + 1
                for order, rows in enumerate(tables):
                    markdown = _table_to_markdown(rows, max_rows=settings.table_max_rows)
                    if not markdown:
                        continue
                    markdown = markdown[: settings.table_max_chars]
                    rect = rects[order] if order < len(rects) else None
                    if layout is not None and rect is not None:
                        line_start, line_end = layout.line_range_for_rect(rect)
                    elif layout is not None and layout.lines:
                        line_start, line_end = 1, layout.lines[-1].number
                    else:
                        line_start, line_end = 0, 0

                    index = chunk_counter.get(page_number, 0)
                    chunk_counter[page_number] = index + 1
                    chunks.append(
                        Chunk(
                            doc_uid=doc_uid,
                            document_name=document_name,
                            source_path=source_path,
                            page_number=page_number,
                            line_start=line_start,
                            line_end=line_end,
                            content_type="table",
                            text=f"Table extracted from page {page_number}:\n\n{markdown}",
                            chunk_index=index,
                        )
                    )
                    try:
                        plumber_page.flush_cache()
                    except Exception:  # pragma: no cover
                        pass
    except Exception as exc:
        warnings.append(f"pdfplumber could not open document: {exc}")

    return chunks, warnings


def parse_pdf(pdf_path: Path, settings: Settings) -> ParsedDocument:
    """Parse one PDF into chunks and figure records.

    Args:
        pdf_path: Absolute path to the PDF.
        settings: Active configuration.

    Returns:
        A fully populated :class:`ParsedDocument`.

    Raises:
        PdfParseError: If the file cannot be opened by PyMuPDF.
    """
    document_name = pdf_path.name
    source_path = str(pdf_path.resolve())
    uid = document_uid(pdf_path, settings.pdf_dir)

    try:
        document = pymupdf.open(str(pdf_path))
    except Exception as exc:
        raise PdfParseError(f"cannot open {document_name}: {exc}") from exc

    parsed = ParsedDocument(
        doc_uid=uid,
        document_name=document_name,
        source_path=source_path,
        page_count=0,
    )

    try:
        if document.needs_pass:
            raise PdfParseError(f"{document_name} is password protected")
        parsed.page_count = document.page_count

        layouts: dict[int, _PageLayout] = {}
        chunk_counter: dict[int, int] = {}
        figure_counter = 0

        for page_index in range(document.page_count):
            page_number = page_index + 1
            try:
                page = document.load_page(page_index)
            except Exception as exc:
                parsed.warnings.append(f"page {page_number} unreadable: {exc}")
                continue

            layout = build_page_layout(page)
            layouts[page_index] = layout

            for line_start, line_end, body in chunk_page_lines(
                layout.lines,
                target_chars=settings.chunk_target_chars,
                overlap_chars=settings.chunk_overlap_chars,
                min_chars=settings.min_chunk_chars,
            ):
                index = chunk_counter.get(page_number, 0)
                chunk_counter[page_number] = index + 1
                parsed.chunks.append(
                    Chunk(
                        doc_uid=uid,
                        document_name=document_name,
                        source_path=source_path,
                        page_number=page_number,
                        line_start=line_start,
                        line_end=line_end,
                        content_type="text",
                        text=body,
                        chunk_index=index,
                    )
                )

            if settings.extract_figures:
                try:
                    figures = extract_figures_for_page(
                        page,
                        layout,
                        doc_uid=uid,
                        document_name=document_name,
                        source_path=source_path,
                        page_number=page_number,
                        settings=settings,
                        start_index=figure_counter,
                    )
                except Exception as exc:
                    parsed.warnings.append(f"figure extraction failed on page {page_number}: {exc}")
                    figures = []
                parsed.figures.extend(figures)
                figure_counter += len(figures)

        if settings.extract_tables:
            table_pages = _detect_table_pages(document)
            table_chunks, table_warnings = _extract_tables(
                pdf_path,
                table_pages,
                layouts,
                doc_uid=uid,
                document_name=document_name,
                source_path=source_path,
                settings=settings,
                chunk_counter=chunk_counter,
            )
            parsed.chunks.extend(table_chunks)
            parsed.warnings.extend(table_warnings)
    finally:
        document.close()

    LOGGER.debug(
        "Parsed %s: %d pages, %d chunks (%d tables), %d figures",
        document_name,
        parsed.page_count,
        len(parsed.chunks),
        parsed.table_count,
        len(parsed.figures),
    )
    return parsed


__all__ = [
    "PARSER_VERSION",
    "PdfParseError",
    "build_page_layout",
    "chunk_page_lines",
    "parse_pdf",
]
