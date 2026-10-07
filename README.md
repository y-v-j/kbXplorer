# kbXplorer

A local-first, CPU-only knowledge base over a folder of PDF research papers.
Ask questions in a terminal interface and get answers where **every claim
carries a citation down to the page and line range**, with extracted figures
one keypress away.

Nothing leaves the machine: parsing, embeddings, vector search, and language
model all run locally.

```
┌─ QA Console ───────────────────────────┬─ Citations & Figures ──────────┐
│ you 14:22:07  Which preprocessing step │ ✓ 1. ¶ study_a.pdf             │
│               improved accuracy most?  │      Page 6, Line 41-58        │
│                                        │ ✓ 2. 🖼 study_a.pdf   [image]  │
│ assistant  Filtering low-quality reads │      Page 5, Line 1-6          │
│ raised accuracy from 81% to 93%        │ ────────────────────────────── │
│ [study_a.pdf, Page 6, Line 41-58], the │ Status: verified               │
│ largest gain of any step in Fig. 2     │ Caption: Fig. 2 | Accuracy     │
│ [study_a.pdf, Page 5, Line 1-6].       │ after each preprocessing step  │
│ ┌────────────────────────────────────┐ │ Image: assets/images/…f001.png │
│ │ Ask a question about your papers…  │ │ Press Enter or Ctrl+O to open. │
└─┴────────────────────────────────────┴─┴────────────────────────────────┘
  Answered in 38.4s · llama3.2:3b · 2/2 citations verified
```

*The paper names, questions, and answers in this README are synthetic, for
illustration only.*

---

## Quick start

```bash
# 1. Get the code
git clone https://github.com/y-v-j/kbXplorer.git
cd kbXplorer

# 2. Create an environment (venv shown; a conda env works just as well)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Make sure Ollama is running and has a model
ollama serve &                 # if not already running
ollama pull llama3.2:3b        # ~2 GB, the recommended default

# 4. Check the environment (also creates the working directories)
python main.py doctor

# 5. Add your papers
cp /path/to/papers/*.pdf pdf_files/input_pdf_files/

# 6. Index the corpus (one-time, ~20 min for ~100 papers)
python main.py ingest

# 7. Launch the interface
python main.py tui
```

The first `ingest` downloads the ONNX embedding model (~79 MB) to
`~/.cache/chroma`, so it needs network access once. After that everything runs
offline.

---

## Example session

*Synthetic example — the papers, questions, and answers are invented.*

Imagine a corpus of two papers, `study_a.pdf` (a methods paper) and
`study_b.pdf` (a review). A one-shot query from the shell:

```text
$ python main.py query "What learning rate was used for fine-tuning?"

Q: What learning rate was used for fine-tuning?

Fine-tuning used a learning rate of 3e-4 with cosine decay over 50 epochs
[study_a.pdf, Page 4, Line 22-30]. A sweep from 1e-5 to 1e-2 found 3e-4
best on the validation split [study_b.pdf, Page 9, Line 5-14].

Sources
  ✓ 1. [study_a.pdf, Page 4, Line 22-30]  [text]
  ? 2. [study_b.pdf, Page 9, Line 5-14]  [text]
! 1 of 2 citations could not be matched to retrieved context and are marked [unverified].

41.7s · llama3.2:3b · 1/2 citations verified
```

The first citation matches a passage that was actually retrieved, so it is
verified (`✓`). The second points at a page the model was never shown, so it
is flagged (`?`) rather than trusted.

When the papers do not cover a question, the model is instructed to say so
instead of guessing, and the top retrieved passages are listed so you can check
for yourself:

```text
$ python main.py query "Who funded the follow-up trial?"

Q: Who funded the follow-up trial?

The provided papers do not contain enough information to answer this question.

Sources
  ✓ 1. [study_b.pdf, Page 2, Line 1-14]  [text]
  ✓ 2. [study_a.pdf, Page 11, Line 30-44]  [text]
  ✓ 3. [study_a.pdf, Page 1, Line 18-35]  [text]
! The model returned no citations; showing retrieved sources instead.

29.3s · llama3.2:3b · 3/3 citations verified
```

---

## Requirements

| | |
|---|---|
| Python | 3.11+ (built and tested on 3.12) |
| Ollama | Running locally, with at least one model pulled |
| RAM | 16 GB, of which **~4 GB must be free** when answering |
| Disk | ~10 GB (models, vectors, extracted figures) |
| GPU | Not required, not used |
| OS | Linux (`xdg-open`), macOS (`open`), Windows |

The repository ships **code only** — no papers, vectors, or figures. The corpus
lives in `pdf_files/input_pdf_files/`, which is created on first run and is
git-ignored along with every generated directory. Subdirectories are scanned
recursively; PDFs placed one level up in `pdf_files/` (reference textbooks, for
instance) are deliberately **not** indexed.

---

## Commands

```bash
python main.py                     # same as `tui`
python main.py ingest              # index new or changed papers
python main.py ingest --force      # re-index everything
python main.py ingest --no-prune   # keep vectors for files deleted from disk
python main.py watch               # index, then auto-index folder changes
python main.py watch --no-initial-sync   # skip the startup scan
python main.py tui                 # dual-pane interface (default command)
python main.py tui --watch         # …with the folder watcher running
python main.py query "what sample size did the study use?"   # one-shot, no TUI
python main.py status              # index statistics and failed documents
python main.py doctor              # diagnose environment problems
python main.py kb list             # registered knowledge bases
python main.py reset --images      # delete the index and figures
```

### Re-indexing

`ingest` only touches files whose content changed. When you need to re-parse
papers that are *already* indexed — after changing a parser setting, upgrading
the parser, or when one paper came out badly — use `reindex`. Selectors narrow
the work so one bad document does not cost a full corpus rebuild, and
`--dry-run` shows the plan before anything is written.

```bash
python main.py reindex --dry-run             # what would be re-parsed?
python main.py reindex                       # re-parse every document in place
python main.py reindex --only "study_a*"     # one paper (glob or substring)
python main.py reindex --failed              # retry only what failed
python main.py reindex --stale               # only docs from an older parser
python main.py reindex --stale --failed      # selectors combine with OR
python main.py reindex --rebuild             # drop the whole store first
python main.py ingest --only "review" --dry-run   # same filters on ingest
```

| Situation | Command |
|---|---|
| Edited `CAPTION_RE`, `figure_dpi`, chunk sizes | `reindex` |
| Bumped `PARSER_VERSION` | `reindex --stale` (or plain `ingest` — it detects this) |
| One paper parsed badly | `reindex --only "<name>"` |
| Some papers failed (e.g. after OCR-ing them) | `reindex --failed` |
| Changed `embedding_backend` | `reindex --rebuild` — the vector space changed |
| Corpus looks corrupt | `reset --images --yes` then `ingest` |

`reindex` replaces each document's vectors atomically (delete by `doc_uid`, then
re-add), so it is safe to interrupt and re-run. Only `--rebuild` drops the whole
store; it refuses to combine with a selector and asks for confirmation unless
`--yes` is passed. A narrowed run never prunes, so documents outside the
selection are left untouched.

Useful overrides:

```bash
python main.py --model qwen3.5:4b tui       # different LLM
python main.py --pdf-dir ~/papers ingest    # different corpus
python main.py --kb genomics status         # a specific knowledge base
python main.py --log-level DEBUG ingest     # verbose logging
```

---

## Keyboard shortcuts

| Key | Action |
|---|---|
| `Enter` (input box) | Submit the question |
| `Enter` (citation row) | Open that figure in the system image viewer |
| `Ctrl+O` | Open the highlighted citation's figure |
| `Tab` / `Shift+Tab` | Move between the input box and the citation list |
| `↑` / `↓` | Move through citations (detail pane follows) |
| `Ctrl+K` | Switch knowledge base (`Esc` closes the picker) |
| `Ctrl+R` | Rescan the corpus for new papers |
| `Ctrl+S` | Print index status into the chat |
| `Ctrl+L` | Clear the conversation |
| `Ctrl+Q` | Quit |

`Enter` is context-sensitive by design: it submits in the input box and opens a
figure on a citation row, which resolves the collision in the original spec.
When a citation has no figure, `Enter`/`Ctrl+O` opens the source PDF instead.

---

## How it works

```
pdf_files/input_pdf_files/*.pdf
        │
        ▼
  ┌───────────────┐   PyMuPDF: text spans → column detection → reading order
  │  pdf_parser   │   → synthetic per-page line numbers
  │               │   caption-anchored region render → assets/images/*.png
  │               │   PyMuPDF table detect → pdfplumber → Markdown
  └───────┬───────┘
          ▼
  ┌───────────────┐   all-MiniLM-L6-v2 (ONNX, 384-d, cosine)
  │  vector_store │   paper_chunks  — text and tables
  │   (ChromaDB)  │   paper_figures — captions + image_path
  └───────┬───────┘
          ▼
  ┌───────────────┐   dense retrieval → lexical re-rank → context block
  │  rag_engine   │   → Ollama (streaming) → parse citations → VERIFY
  └───────┬───────┘
          ▼
     tui/app.py  (60% chat · 40% citation dock)
```

### Citations are verified, not trusted

A 3B model will happily emit a correctly formatted citation pointing at a page
it never saw. Every citation in an answer is therefore parsed and checked
against the metadata of the chunks that were actually retrieved: the document
name, the page, and an overlapping line range must all match. Citations that
fail are shown with `?` in the dock (and in the `query` source list) along
with a warning, never silently accepted. The status bar reports the ratio (`3/3 citations verified`).
If an answer cites nothing at all, the dock falls back to the top retrieved
passages so you still see where the context came from.

### About line numbers

PDFs have no line concept — only positioned glyph runs. Line numbers here are
*synthesised*: text spans are collected per page, columns are detected, spans
are sorted into reading order, and numbered from 1. They are deterministic for a
given file and parser version, so a citation is reproducible and the reference
pane can show you the exact source text. They will **not** match line numbers
printed in a journal's margin.

### About figures

`page.get_images()` only returns embedded raster images, so it silently misses
vector artwork — which is most plots in a scientific paper. The primary path
here is caption-anchored instead: find a `Figure N` / `Table N` caption, infer
the graphic region next to it, widen it to the artwork's true extent, and render
that region to PNG at 160 DPI. This captures vector and raster figures alike.
Embedded rasters not already covered by a rendered region are exported too.

### Without Ollama

If the Ollama daemon is down, retrieval still runs: questions return the most
relevant passages with their citations, and the interface says that synthesis
is disabled. If the configured model is not installed, the first installed
entry of `ollama_fallback_models` is used, then any installed model.

---

## Adding papers

Drop a PDF into `pdf_files/input_pdf_files/`, then either:

* press `Ctrl+R` in the TUI,
* run `python main.py ingest`, or
* leave `python main.py watch` (or `tui --watch`) running — new files are
  detected and indexed automatically.

The watcher waits for a file's size to stop changing before parsing, so a large
PDF still being copied in is never indexed half-written. Deleted files are
removed from the index on the next scan.

Only one process can ingest at a time. Ingestion takes an advisory lock on
`state/ingest.lock`; a second writer (say, `watch` alongside `tui --watch`)
stops with a `Busy:` message instead of corrupting the index. Queries never
take the lock.

---

## Replacing the corpus / running several knowledge bases

### Replace everything with a different set of papers

Delete the old PDFs, copy the new ones in, then ingest:

```bash
rm pdf_files/input_pdf_files/*.pdf
cp /path/to/new/papers/*.pdf pdf_files/input_pdf_files/
python main.py ingest
```

That is all that is required. `ingest` prunes documents that no longer exist —
their vectors, manifest rows, and extracted figure PNGs are all removed — and
indexes the new ones. The report line shows both halves, e.g.
`indexed=2 ... removed=3`. Verify with `python main.py status`.

If you would rather start from a guaranteed-clean slate (or the index was
damaged), reset first:

```bash
python main.py reset --images --yes    # clears vectors, state, and figures
python main.py ingest
```

Only the PDFs are irreplaceable; everything else rebuilds from them.

### Several knowledge bases, switchable inside the TUI

Register each corpus once, then switch with **Ctrl+K** without leaving the app:

```bash
python main.py kb add genomics --pdf-dir ~/papers/genomics \
    --description "RNA-seq and variant calling"
python main.py --kb genomics ingest      # index it
python main.py tui                       # Ctrl+K to switch
```

| Command | Does |
|---|---|
| `kb list` | Show every base, its corpus size, and which is active |
| `kb add <name> --pdf-dir PATH` | Register a corpus (PDFs stay where they are) |
| `kb add <name> --pdf-dir PATH --force` | Overwrite an existing entry |
| `kb use <name>` | Set the default base for new sessions |
| `kb remove <name>` | Unregister — **deletes no files** |
| `--kb <name> <command>` | Run any command against one base |

Inside the TUI, `Ctrl+K` opens a picker showing each base with its PDF count and
indexed passages. Switching swaps the vector store, the reference dock, and the
pane title; the chat transcript stays so you can compare answers across corpora.
Your choice is remembered, so the next launch reopens the same base.

The registry is stored in `knowledge_bases.json` at the project root. Until it
exists, a single `default` base is derived from the current settings, so a
single-corpus setup needs no registration. The file is local machine state and
is git-ignored.

Only the corpus folder is yours to place — the generated index for a named base
lives under `bases/<name>/`. `kb remove` unregisters without deleting anything;
delete `bases/<name>/` by hand if you want the index gone too.

**Cost.** The embedding model (~250 MB) and the Ollama model (~2.6 GB) are shared
across all bases, so a second knowledge base costs only its own index —
**measured at 51 MB resident for a 94-paper corpus**. Switching repeatedly does
not accumulate memory: RSS stays flat across alternating switches because the
previous store and manifest are released. On 16 GB you can register as many as
you like.

### Keeping a knowledge base entirely outside the project

Every path is configurable, so a corpus and its index can live anywhere. Point
the four directory settings somewhere new:

```bash
export KB_PDF_DIR=~/kb/genomics/pdfs
export KB_CHROMA_DIR=~/kb/genomics/chroma
export KB_STATE_DIR=~/kb/genomics/state
export KB_ASSETS_DIR=~/kb/genomics/assets
python main.py ingest && python main.py tui
```

> **Note:** these variables only take effect while `knowledge_bases.json` does
> not exist. Once it does — after any `kb add`, `kb use`, or a `Ctrl+K` switch —
> the active base's paths from the registry override `KB_PDF_DIR`,
> `KB_CHROMA_DIR`, `KB_STATE_DIR`, `KB_ASSETS_DIR`, and the same keys in
> `config.json`. In that case, use `kb add` instead. `--pdf-dir` still
> overrides the corpus folder, but not where the index is stored.

A small wrapper script per knowledge base is the tidiest form:

```bash
#!/usr/bin/env bash
# ~/bin/kb-genomics
export KB_PDF_DIR=~/kb/genomics/pdfs KB_CHROMA_DIR=~/kb/genomics/chroma
export KB_STATE_DIR=~/kb/genomics/state KB_ASSETS_DIR=~/kb/genomics/assets
cd /path/to/kbXplorer
exec python main.py "$@"
```

The two indexes never see each other, and the embedding model and Ollama are
shared, so a second knowledge base costs only its own vectors and figures.

---

## Configuration

Defaults live in `engine/config.py`. Override them with a `config.json` in the
project root, or with `KB_*` environment variables (which win). Any setting name
works as `KB_<NAME>` in upper case:

```bash
KB_OLLAMA_MODEL=qwen3.5:4b python main.py tui
KB_TOP_K_TEXT=8 KB_MAX_CONTEXT_CHARS=6000 python main.py query "…"
KB_FIGURE_DPI=200 python main.py ingest --force
```

```json
{
  "ollama_model": "qwen3.5:4b",
  "top_k_text": 6,
  "figure_dpi": 200
}
```

Unknown or malformed keys are logged and ignored, so a stale `config.json`
never stops the app from starting.

Frequently adjusted settings:

| Setting | Default | Notes |
|---|---|---|
| `ollama_model` | `llama3.2:3b` | 3B is the sweet spot on a laptop CPU |
| `ollama_host` | `http://localhost:11434` | Point at another Ollama daemon |
| `ollama_num_ctx` | `4096` | Larger costs prompt-eval time and RAM |
| `ollama_num_predict` | `700` | Caps answer length; each token ~0.1 s |
| `ollama_keep_alive` | `30m` | Keeps the model loaded between questions |
| `top_k_text` | `5` | Passages retrieved per question |
| `top_k_figures` | `3` | Figure captions retrieved per question |
| `lexical_weight` | `0.35` | Keyword weight in the hybrid re-rank |
| `max_context_chars` | `4500` | Prompt size is the main latency driver |
| `figure_dpi` | `160` | Raise for sharper figures, larger files |
| `extract_figures` / `extract_tables` | `true` | Turn off for a faster text-only ingest |
| `embedding_backend` | `onnx` | `sentence-transformers` for the torch build |
| `strip_reasoning_tokens` | `true` | Hide `<think>` blocks from reasoning models |

The PyTorch backend is optional and not in `requirements.txt`. Install the CPU
build with
`pip install sentence-transformers --index-url https://download.pytorch.org/whl/cpu`.

---

## Performance

Measured on an i5-1135G7 (4 cores / 8 threads), 16 GB RAM, CPU only:

| Operation | Time |
|---|---|
| Ingest, per page | ~0.9–2.0 s (figure rendering dominates) |
| Ingest, 94 papers / 1213 pages | ~20–30 min, one time |
| Embedding | ~150 chunks/s |
| Retrieval | < 100 ms |
| Answer, `llama3.2:3b`, warm | ~30–60 s |
| Answer, 7B q4 | 2–3 min — not recommended here |

**Free RAM is the dominant factor.** With under ~3 GB available the machine
swaps and a single answer can take minutes. `python main.py doctor` checks this
explicitly. If answers are slow, close memory-heavy applications (a browser with
many tabs is usually the culprit) and run `ollama stop <model>` to unload a
model you are not using.

---

## Troubleshooting

**`Ollama unavailable`** — the daemon is not running. Start it with `ollama
serve`. Retrieval and citations keep working without it; only answer synthesis
is disabled, and the interface says so.

**Answers take minutes** — check `python main.py doctor` for the RAM and swap
lines. This is nearly always memory pressure, not the model.

**`Busy: another process is already ingesting`** — a second `ingest`, `watch`,
or `tui --watch` is writing to the same index. Let it finish or stop it, then
retry.

**`Could not initialise the ONNX MiniLM embedder`** — the embedding model could
not be downloaded on first run. Check network access; the model is cached in
`~/.cache/chroma` afterwards.

**A paper shows as `failed`** — run `python main.py status` to see the reason.
The usual cause is a scanned PDF with no text layer; it needs OCR
(e.g. `ocrmypdf in.pdf out.pdf`) before it can be indexed. Password-protected
PDFs also fail.

**No figures extracted from a paper** — its captions may not start with
`Figure`/`Fig.`/`Table`/`Scheme`/`Chart`/`Plate`/`Exhibit`. Adjust `CAPTION_RE`
in `engine/pdf_parser.py`.

**Text from a paper looks scrambled or interleaved** — the page layout defeated
column detection. `engine/pdf_parser.py:_order_lines()` segments each page into
horizontal bands and orders the columns within each; rotated text (vertical
journal watermarks) is dropped before ordering. Both run on every ingest, so new
papers are handled the same way. If a specific layout still misreads, the knobs
are `_SPAN_TOLERANCE`, `_COLUMN_SHARE`, and `_Y_QUANTUM` at the top of that
module — changing any of them shifts line numbering, so bump `PARSER_VERSION` in
`engine/manifest.py` and re-index afterwards.

**Changed the embedding backend** — the vector space changed and old vectors are
meaningless. Ingestion warns and tells you to rebuild; run
`python main.py reindex --rebuild`.

**`KB_PDF_DIR` (or another path variable) is ignored** — a `knowledge_bases.json`
exists and its active entry wins. See the note under
[Keeping a knowledge base entirely outside the project](#keeping-a-knowledge-base-entirely-outside-the-project).

**Interrupted ingestion** — just run `python main.py ingest` again. Documents
caught mid-flight are detected and retried; completed ones are skipped.

---

## Project layout

```
kbXplorer/
├── main.py                  CLI entry point
├── ingest.py                ingestion pipeline, re-index selectors
├── requirements.txt
├── README.md
├── instructions.md          the original project specification
├── engine/
│   ├── config.py            settings, env overrides, logging
│   ├── models.py            typed data structures
│   ├── manifest.py          SQLite ingestion state (crash-safe), PARSER_VERSION
│   ├── locking.py           advisory inter-process ingestion lock
│   ├── registry.py          named knowledge bases (multi-corpus)
│   ├── pdf_parser.py        layout, line numbering, figures, tables
│   ├── embeddings.py        ONNX / sentence-transformers backends
│   ├── vector_store.py      ChromaDB wrapper, hybrid retrieval
│   ├── ollama_client.py     LLM connector, health, streaming
│   ├── citations.py         citation parsing and verification
│   └── rag_engine.py        retrieval → prompt → grounded answer
├── tui/
│   ├── app.py               dual-pane Textual application
│   ├── kb_picker.py         Ctrl+K knowledge-base switcher
│   ├── widgets.py           citation rows, detail pane, viewer launch
│   └── styles.tcss          layout and theme
└── watcher/
    └── folder_watcher.py    debounced, stability-checked folder monitor
```

Created at runtime and git-ignored:

```
├── pdf_files/input_pdf_files/   your corpus (default base)
├── chroma_db/                   persistent vectors
├── assets/images/               extracted figures
├── state/                       manifest.db and ingest.lock
├── logs/                        knowledge_base.log
├── bases/<name>/                index data for each base added with `kb add`
├── knowledge_bases.json         knowledge-base registry
└── config.json                  optional settings overrides
```

---

## License

The kbXplorer source code is released under the [MIT License](LICENSE).
Copyright (c) 2026 Yogesh.

### Third-party licenses

kbXplorer does not bundle its dependencies; `pip` installs each one under its
own license:

| Package | License |
|---|---|
| PyMuPDF | AGPL-3.0, or a commercial license from Artifex |
| pdfplumber, onnxruntime, ollama, textual, rich | MIT |
| chromadb, tokenizers, watchdog | Apache-2.0 |
| Pillow | MIT-CMU |
| sentence-transformers (optional) | Apache-2.0 |

PyMuPDF is the one to be aware of. Personal and research use of kbXplorer is
unaffected, and the MIT license on this code is compatible with the AGPL. If
you distribute kbXplorer together with PyMuPDF, or offer it as a network
service, as part of a closed-source product, you must either meet the AGPL-3.0
terms or obtain a commercial PyMuPDF license from Artifex.
