You are an expert Bioinformatician, Python Programmer and Systems Software Engineer. Build a complete, modular, local-first CLI/TUI application that creates a queryable Knowledge Base from a folder of local PDF research papers.

### TECHNICAL SPECIFICATIONS & HARDWARE CONSTRAINTS
- Operating Constraints: Must run smoothly on CPU only (No GPU support required, Intel 11th Gen i5/i7, 16 GB RAM with about 150 GB of Hard Drive space). Keep memory footprint low while development as well as execution of the completed product.
- Language & Core Frameworks: Python 3.12
- TUI Interface: Textual (PyTermGUI/Rich powered split-pane interface)
- Local Vector Database: ChromaDB (Persistent storage at `./chroma_db`)
- Embeddings Model: sentence-transformers/all-MiniLM-L6-v2 (CPU-optimized, low memory)
- PDF Ingestion & Parsing: PyMuPDF (`fitz`) and `pdfplumber` (for layout awareness, line numbers, and figure extraction)
- Folder Monitoring: `watchdog` library
- Local LLM Integration: Ollama Python Client (`ollama`), querying locally served models (default: `llama3.2:3b` or `qwen2.5:7b-instruct-q4_K_M`) OR any other open source (freely available) suitable model for this purpose.

---

### ARCHITECTURE & CORE REQUIREMENTS

1. INGESTION PIPELINE (`ingest.py`)
   - Monitor a specified directory (`./pdf_files) using `watchdog`. Automatically index any new or updated `.pdf` files.
   - Parse PDFs into structured text chunks while preserving exact structural metadata:
     * `document_name`: Filename
     * `page_number`: Page number (1-indexed)
     * `line_start` & `line_end`: Estimated line positions calculated per page layout
     * `content_type`: "text", "table", or "figure"
   - Extract figures and images using PyMuPDF (`fitz`):
     * Extract embedded images to `./assets/images/` with unique filenames.
     * Use spatial heuristics (bounding box coordinates) to detect nearby text blocks starting with "Figure", "Fig.", or "Table" to extract figure captions.
     * Index figure captions into a dedicated ChromaDB collection (`paper_figures`) with `image_path`, `page`, and `line_start` saved in metadata.
   - Extract tables as Markdown text using `pdfplumber` and store them in the main vector collection.

2. RETRIEVAL & RAG SYSTEM (`rag_engine.py`)
   - Given a user query, generate embeddings using `all-MiniLM-L6-v2`.
   - Perform hybrid context retrieval from ChromaDB (retrieving relevant text chunks, tables, and top matching figures/captions).
   - Construct a prompt for the local Ollama LLM instructing it to answer strictly based on the provided context.
   - Require the LLM to provide citations for every claim using a strict format: `[DocName.pdf, Page X, Line Y-Z]`.

3. TERMINAL USER INTERFACE (`app.py` using Textual)
   - Interface Layout (Dual-Pane Split Screen):
     * LEFT PANE (60% width): Interactive QA Chat Console. Displays system messages, user inputs, and streaming LLM responses. Contains a bottom text input box.
     * RIGHT PANE (40% width): Citation & Visual Reference Dock. Dynamic view updating based on retrieved results or clicked citations.
   - Right Pane Details:
     * Displays extracted exact citations (Document title, Page number, Line range, and raw text chunk snippet).
     * If a figure or table is cited/matched, display the caption and image path.
     * Add an interactive button or shortcut (`Ctrl+O` or `Enter`) to trigger the system's default image viewer (`xdg-open` / `open`) to instantly view the extracted high-res PNG figure.

---

### DELIVERABLES EXPECTED
1. Complete project directory structure:
   - `main.py` (CLI entry point)
   - `tui/` (Textual UI layout and components)
   - `engine/` (PDF parser, ChromaDB vector store wrapper, Ollama connector)
   - `watcher/` (Watchdog paper folder monitor)
   - `requirements.txt`
2. Fully typed, clean, production-grade Python code with clear docstrings and error handling (e.g., handling missing Ollama service gracefully).
3. Clear startup instructions in a README section (how to pull the Ollama model, set up virtualenv, and start the app).

### BRAINSTORMING IDEAS:
Refer to the author's private brainstorming notes (not included in this repository).
