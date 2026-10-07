# Development Notes

Design decisions, the reasoning behind them, and the things that will bite you
if you change them. Written for whoever maintains this next.

Target hardware: Intel i5-1135G7 (4 cores / 8 threads, Tiger Lake), 16 GB RAM,
no GPU, Python 3.12.

---

## 1. The three hard problems

Everything else in this codebase is plumbing. These three are where the
engineering actually is.

### 1.1 PDFs have no line numbers

The spec asks for citations of the form `[doc.pdf, Page 4, Line 12-18]`. PDFs
store positioned glyph runs, not lines, and certainly not line numbers. There is
nothing to read out.

**What we do.** `engine/pdf_parser.py:build_page_layout()` synthesises them:

1. Pull text lines with geometry via `page.get_text("dict")`.
2. Detect the column layout (`_assign_columns`) — a page is two-column when
   fewer than 25% of lines straddle the vertical midline and each side holds at
   least 15% of the lines.
3. Sort by `(column, y0 quantised to 3 pt, x0)`. The quantisation matters:
   without it, sub-pixel baseline jitter shuffles words on the same visual line.
4. Number from 1.

Full-width bands are special-cased. A line straddling the midline in the top 22%
of the page gets column `-1` (running head, title) and in the bottom 18% column
`99` (footer), so they sort outside the body columns instead of corrupting
reading order in the middle.

**Band segmentation, not a whole-page column vote.** The first cut used a
single verdict per page: count how many lines straddle the midline, decide
"one column" or "two", sort accordingly. That is wrong for the most valuable
page in every paper. A title page carries a full-width title block above a
two-column body; on one test paper's first page, 27% of lines straddled the midline, which
tripped the single-column fallback and **interleaved the two columns of the
abstract** — fragments of the two columns alternated line by line. The abstract is
the highest-value retrieval target in a paper, and it was being scrambled.

`_order_lines()` segments the page into horizontal *bands* instead. A line that
straddles the midline (title, running head, a heading spanning both columns, a
wide equation, a footer) closes the current band and is emitted alone; runs of
non-straddling lines form a column band whose left column is emitted in full
before its right. Title pages and pure two-column pages both come out correct,
and a full-width figure mid-page correctly splits the surrounding text.

Rotated lines (`line["dir"][1] != 0`) are dropped before ordering. Vertical
watermarks — a publisher's `Downloaded from ...` notice down the page edge — are
not part of the reading flow and were fragmenting bands.

**Consequences you must accept.** These numbers are an artefact of *our* parser.
They are stable for a given file and `PARSER_VERSION`, which is what makes a
citation reproducible and lets the reference pane show the exact source text.
They do **not** correspond to line numbers printed in a journal margin. If you
change the sorting, the quantum, or the column heuristic, every stored citation
silently shifts — bump `PARSER_VERSION` in `engine/manifest.py`, which forces a
re-index. (This happened once already: 1.0.0 → 1.1.0 for band segmentation.)

### 1.2 `get_images()` misses most scientific figures

The brainstorming notes (and most tutorials) build figure extraction on
`page.get_images()`. That returns embedded raster XObjects only. A plot exported
from matplotlib, R, or Illustrator is *vector* content — there is no image object
to extract, and this approach returns nothing at all for it.

This was verified on the corpus: for a figure drawn as a pure vector diagram, the
raster path finds fragments; the render path captures the whole figure.

**What we do.** Caption-anchored region rendering, in `_figure_region()`:

1. Find caption blocks matching `CAPTION_RE` (`Figure|Fig.|Table|Scheme|Chart|
   Plate|Exhibit` + number).
2. Infer the adjacent graphic band — above the caption for figures, below for
   tables, falling back to the other side.
3. Bound it vertically by the nearest body text over that x-range.
4. **Widen** to the true horizontal extent of the artwork, then **re-clamp
   vertically over the wider span**.
5. Render with `page.get_pixmap(clip=region, dpi=160)`.

Step 4 is subtle and was the source of two bugs during development:

* Without widening, a full-width figure whose caption sits in one column gets
  cropped to that column. (Caught on a full-width vector figure during testing.)
* Widening naively then lets the running header bleed into the render, because
  the vertical bound was computed for the narrow span. Hence the re-clamp.
* Re-clamping can *collapse* the band when the figure genuinely is column-width
  and prose sits beside it. So if the re-clamped height drops below half the
  narrow height, we keep the narrow region.

Embedded rasters are still exported, but only when they do not overlap a
rendered region **and** their nearest caption was not already rendered
(`rendered_labels`). Without that second check every figure is indexed twice —
once as a render, once as a raster — with identical captions, which pollutes
retrieval.

Uncaptioned artwork must clear 4× the size threshold, because at the base
threshold you collect journal logos and rules.

### 1.3 Small models fabricate citations

A 3B model produces well-formed citations pointing at pages it never saw.
Prompting reduces this; it does not fix it. Treating the model's output as
trustworthy would make the entire premise of the project false.

**What we do**, in `engine/citations.py`:

* `parse_citations()` — tolerant regex (en/em dashes, `Line`/`Lines`, missing end
  line, reversed ranges, case variation), deduplicated.
* `verify_citations()` — each citation must match a *retrieved* chunk on
  document name and page, with line ranges overlapping within 5 lines of slack.
  The slack absorbs a model quoting a narrower span than the chunk it came from.
* Failures are marked `[unverified]` inline and shown with `?` in the dock. They
  are never dropped silently — a hidden failure is worse than a visible one.
* If an answer cites nothing at all, the dock falls back to the top retrieved
  passages so the user still sees provenance.

The prompt reinforces this by putting an explicit `CITATION:` line on every
context passage holding the exact string to copy. Measured on this corpus,
`llama3.2:3b` reaches 3/3 verified on well-posed questions.

---

## 2. Choices that were not obvious

### Embedding backend: ONNX over PyTorch

The spec names `sentence-transformers/all-MiniLM-L6-v2`. ChromaDB bundles the
*same model* as a quantised ONNX build. Resident memory is ~150 MB versus
~700 MB for the PyTorch path, and it avoids a ~2.5 GB torch install. On a machine
where free RAM is the binding constraint, that is the correct default.

Both backends are implemented (`engine/embeddings.py`); switch with
`KB_EMBEDDING_BACKEND=sentence-transformers`. Retrieval quality is
indistinguishable on this corpus.

**Trap:** the two backends produce different vector spaces. Mixing them silently
corrupts retrieval. `backend_fingerprint()` is written to the manifest and
checked on every ingest; a mismatch warns and tells you to `reindex`.

**Trap:** the ONNX model downloads once to `~/.cache/chroma` (79 MB). First run
needs network. After that it is fully offline.

### pdfplumber only where PyMuPDF found a table

pdfplumber produces the best Markdown tables but runs 10–20× slower than
PyMuPDF — measured at 0.5–2 s/page against ~0.1 s/page. Running it on all 1213
pages would dominate ingestion.

So PyMuPDF's `find_tables()` is the **detector** (fast, and it gives us the
bbox for line-range mapping) and pdfplumber is the **extractor**, opened only on
pages already known to contain a table. pdfplumber loads pages lazily, so this
is genuinely cheap. `flush_cache()` is called per page to bound memory.

`_table_to_markdown()` drops all-empty columns; pdfplumber emits many of them on
ruled scientific tables and they make the output unreadable.

### Hybrid retrieval, not pure vector search

Dense-only retrieval misses exact identifiers — accession codes, gene names,
software tool names — which is exactly what gets asked about in a research corpus.
`_rerank()` blends cosine similarity with keyword overlap at
`lexical_weight=0.35`. The lexical term is `log1p(matched) / log1p(query terms)`,
sub-linear so one rare match does not dominate.

Observable in practice: for a question naming a specific tool, the top
hit scores dense 0.84 / lexical 0.92, and a chunk with dense 0.69 but lexical
1.00 is correctly promoted into the top 3.

### Deterministic IDs and document-scoped replacement

Chunk IDs are `{doc_uid}:p{page}:c{index}`, where `doc_uid` is a SHA-1 of the
path *relative to the corpus root* — derived from path, not content, so it
survives edits. Re-ingesting deletes by `where={"doc_uid": ...}` then re-adds,
so an edited paper never leaves stale chunks behind.

### Sequential ingestion

Parsing one paper with figure rendering peaks around 400–500 MB; the full
process peaks near 1.2 GB with ONNX and Chroma loaded. Parallel workers are what
would push a 16 GB laptop into swap. Ingestion is deliberately single-file, and
pixmaps are released immediately after `save()` — they are the memory hot spot.

### `Enter` is context-sensitive

The spec assigns `Enter` both "submit question" and "open figure". They are
resolved by focus: `Input.Submitted` submits, `ListView.Selected` on a citation
row opens. `Ctrl+O` works from anywhere.

---

## 3. Crash safety

The user asked for robustness across reboots and updates. The manifest
(`engine/manifest.py`) is the mechanism.

* SQLite in **WAL mode with `synchronous=FULL`** — a hard power cut cannot tear
  the manifest.
* Status transitions bracket the work: `pending → indexing → indexed|failed`,
  each committed before/after parsing.
* `reset_stale_indexing()` runs at the start of every sync and demotes anything
  left in `indexing` (the signature of a crash) back to `pending` for retry.
* `needs_ingest()` compares SHA-256 first, falling back to size+mtime, and also
  re-ingests when `PARSER_VERSION` changed.
* Deleted files are pruned from both the vector store and the manifest.

Net effect: `python main.py ingest` is idempotent and always safe to re-run.
Interrupt it at any point and run it again.

### Re-indexing

`ingest` is content-driven: it only re-parses a file whose SHA-256 changed. That
is the wrong tool when the *parser* changed and the files did not. `reindex`
covers that case, and `ReindexSelector` (in `ingest.py`) keeps it proportionate:

| Selector | Matches |
|---|---|
| *(none)* | every document |
| `--only PATTERN` | filename glob, or plain substring |
| `--failed` | manifest status `failed` |
| `--stale` | `parser_version != PARSER_VERSION` |

Criteria combine with OR. Two invariants matter:

* **A narrowed run never prunes.** `sync()` forces `prune=False` whenever the
  selector is narrowed — otherwise every document outside the selection would
  look "missing" and be deleted from the index.
* **`--rebuild` refuses a selector.** Dropping the whole store and then
  re-ingesting a subset would silently discard everything else. This is the one
  destructive path, so it also prompts unless `--yes` is passed.

Ordinary `reindex` needs no store reset: `replace_document()` deletes by
`doc_uid` before re-adding, so each document is swapped atomically and an
interrupted run leaves the rest of the index intact. `--rebuild` exists for the
one case that genuinely needs it — an embedding-backend change, where the old
vectors occupy a different space and cannot be compared against the new ones.

`--dry-run` routes through `_plan()`, which applies the same selector and
`needs_ingest()` logic without taking the ingestion lock or writing anything.

The watcher adds a second safety property: `_is_stable()` requires a file's size
to be unchanged across `watch_stability_seconds` before parsing, so a large PDF
still being copied in is never indexed half-written. Events are debounced per
path (a single copy emits a burst of `created`/`modified`), and ingestion runs on
one worker thread so the observer is never blocked.

---

## 4. Performance, measured

On the target machine, CPU only:

| Stage | Measurement |
|---|---|
| Parse, text only | ~0.1 s/page |
| Parse, with figure rendering | 0.9–2.0 s/page |
| pdfplumber table extraction | 0.5–2 s/page (gated) |
| Embedding | ~150 chunks/s |
| Retrieval (dense + rerank) | < 100 ms |
| `llama3.2:3b`, warm, 4.5 k context | ~30–60 s per answer |
| Peak RSS, ingestion | ~1.2 GB |

### The memory finding

During development the machine was at **13 GB/15 GB RAM used and 7.6 GB/7.6 GB
swap consumed** — swap fully exhausted. Under those conditions a single answer
took **200 s with 120 s time-to-first-token**. That is thrashing, not compute:
after unloading idle Ollama models, available RAM went from 2.0 GB to 5.1 GB.

This is why `main.py doctor` explicitly checks `MemAvailable` and `SwapFree` and
fails the check when swap is exhausted. If answers are slow, that is the first
thing to look at — not the model size.

Settings were tuned accordingly:

| Setting | Was | Now | Why |
|---|---|---|---|
| `ollama_num_ctx` | 8192 | 4096 | Halves KV cache; prompt eval is the latency driver |
| `max_context_chars` | 7000 | 4500 | Prompt size dominates time-to-first-token |
| `top_k_text` | 6 | 5 | Smaller prompt, negligible recall loss |
| `ollama_num_predict` | — | 700 | Caps rambling; each token costs ~0.1 s |
| `ollama_keep_alive` | — | 30m | Avoids reloading a 2 GB model between questions |

---

## 4b. Multiple knowledge bases

`engine/registry.py` holds named corpora in `knowledge_bases.json`; each entry
carries its own `pdf_dir`, `chroma_dir`, `assets_dir`, and `state_dir`.
`KnowledgeBaseRef.apply()` returns a `Settings` copy pointed at one of them, so
every existing component works unchanged — nothing below the registry knows
that more than one corpus exists.

On first run the registry seeds a `default` entry from whatever the current
settings describe, so a single-corpus installation keeps working with no
migration step.

**Why switching is cheap.** Measured on the 94-paper corpus: the ONNX embedding
model costs ~248 MB and the Ollama model ~2.6 GB, and both are *shared*. An open
vector store adds only ~51 MB. Switching therefore costs tens of megabytes, not
gigabytes, which is what makes it viable as an interactive action on a 16 GB
machine. `_switch_worker` closes the previous manifest and drops the previous
store; RSS measured flat (~370 MB) across six alternating switches.

**One trap worth remembering.** `App._registry` is Textual's own widget set. The
first version of this stored the knowledge-base registry as `self._registry` and
silently broke `push_screen` and shutdown (`'Registry' object has no attribute
'discard'`). It is `self._kb_registry` for that reason. If you add state to the
app, check it against `App.__init__` first — that was the only collision among
the twelve attributes this class defines, but there are 53 to collide with.

The switch runs on a thread worker because opening a ChromaDB collection touches
disk, and it is guarded by `_switching` so a second Ctrl+K mid-swap is refused
rather than racing.

## 5. Known limitations

* **Scanned PDFs without a text layer** yield nothing. They are marked `failed`
  with a clear reason. OCR them first (`ocrmypdf`).
* **Line numbers are synthetic** — see §1.1. Reproducible, but not the journal's.
* **Table extraction degrades on complex layouts.** Multi-level headers and
  merged cells produce imperfect Markdown. The table is still retrievable and
  cited correctly; only its formatting suffers.
* **Caption detection is regex-based.** Papers whose captions do not begin with
  `Figure`/`Fig.`/`Table`/`Scheme`/`Chart`/`Plate` yield no figures. Extend
  `CAPTION_RE`.
* **Three-column layouts** are treated as two-column. Rare in this corpus.
* **No cross-document reasoning beyond retrieval.** The model sees the top-k
  passages; it cannot perform a corpus-wide aggregation ("how many papers
  used X?").
* **No conversation memory.** Each question is independent. Follow-ups must be
  self-contained. This is deliberate — carrying history would consume the
  context budget that citations depend on.

---

## 6. Extension points

**Better answers on the same hardware.** Raise `top_k_text` and
`max_context_chars` only if RAM allows; the cost is linear in prompt size. A
reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) would improve precision at
~200 ms/query and ~100 MB.

**Better figures.** `pymupdf_layout` (PyMuPDF suggests it at import) gives
stronger layout analysis. Raising `figure_dpi` to 200–300 improves quality
roughly quadratically in file size.

**Inline terminal images.** Kitty/WezTerm support graphics protocols; `chafa`
could render figures directly in the dock instead of shelling out to
`xdg-open`. Not installed here, hence the external-viewer approach.

**Conversation memory.** Add a rolling summary to the system prompt, but budget
it explicitly against `max_context_chars`.

**Larger model.** `qwen2.5:7b-instruct-q4_K_M` gives better synthesis at 3–5
tok/s — roughly 2–3 minutes per answer here. Only worth it with ≥8 GB free.

---

## 7. Testing notes

There is no automated test suite; validation was done against the live corpus.
If you add one, the high-value targets are pure and easy to test:

* `chunk_page_lines()` — overlap, runt-tail folding, line-range correctness.
* `parse_citations()` — the dash/case/reversed-range variants.
* `verify_citations()` — the slack boundary and the same-page partial match.
* `_table_to_markdown()` — empty-column dropping, ragged rows.
* `_order_lines()` — one-column, two-column, and title pages with a full-width
  block above a two-column body (the case that motivated band segmentation).
* `_ThinkFilter` — tags split across streaming fragments.

`Manifest` is straightforward to test against a `tmp_path` SQLite file.

Manual checks worth repeating after parser changes:

```bash
python main.py doctor
python main.py ingest --force            # on a 3-4 paper subset
python main.py query "<a question one of those papers answers>"
# expect: every citation verified
```

Then open a rendered figure and confirm it is neither cropped nor polluted with
body text — that single check catches most `_figure_region()` regressions.
