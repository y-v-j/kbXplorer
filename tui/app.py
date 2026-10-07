"""Textual dual-pane interface for the research knowledge base.

Layout
------
* **Left (60%)** - QA chat console with a docked input box. System notices,
  questions, and streaming answers are appended as separate blocks.
* **Right (40%)** - Citation and visual reference dock. The upper list holds one
  row per citation; the lower panel shows the selected citation's page, line
  range, source snippet, caption, and image path.

``Enter`` is context-sensitive: in the input box it submits the question, and on
a citation row it opens that figure in the system image viewer. ``Ctrl+O`` does
the same from anywhere, which resolves the ambiguity in the original spec.

Generation runs on a Textual thread worker, so a CPU-bound 3B model streaming at
~10 tokens/second never blocks the event loop.
"""

from __future__ import annotations

import logging
from datetime import datetime

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Input, ListView, Static

from engine.citations import citations_from_context
from engine.config import Settings
from engine.models import AnswerResult, Citation, RetrievalBundle
from engine.ollama_client import OllamaConnector
from engine.rag_engine import RagEngine
from engine.registry import KnowledgeBaseRef, Registry
from engine.vector_store import VectorStore
from ingest import IngestionService, build_service
from tui.kb_picker import KnowledgeBasePicker
from tui.widgets import CitationDetail, CitationItem, open_in_system_viewer

LOGGER = logging.getLogger(__name__)


class KnowledgeBaseApp(App[None]):
    """The dual-pane terminal application."""

    CSS_PATH = "styles.tcss"
    TITLE = "Research Knowledge Base"

    BINDINGS = [
        Binding("ctrl+o", "open_figure", "Open figure", priority=True),
        Binding("ctrl+k", "switch_base", "Switch KB", priority=True),
        Binding("ctrl+r", "rescan", "Rescan corpus"),
        Binding("ctrl+l", "clear_chat", "Clear chat"),
        Binding("ctrl+s", "show_status", "Status"),
        Binding("ctrl+q", "quit", "Quit", priority=True),
    ]

    def __init__(
        self,
        settings: Settings,
        engine: RagEngine,
        connector: OllamaConnector,
        store: VectorStore,
        service: IngestionService,
        registry: Registry | None = None,
        manifest: object | None = None,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._engine = engine
        self._connector = connector
        self._store = store
        self._service = service
        # NB: not ``_registry`` - Textual's App uses that name internally for
        # its widget set, and shadowing it breaks push_screen and shutdown.
        self._kb_registry = registry
        self._manifest = manifest

        self._citations: list[Citation] = []
        self._stream_target: Static | None = None
        self._stream_buffer = ""
        self._busy = False
        self._switching = False

    # ------------------------------------------------------------- compose
    def compose(self) -> ComposeResult:
        """Build the widget tree."""
        yield Header(show_clock=True)
        with Horizontal(id="main-split"):
            with Vertical(id="chat-pane"):
                yield VerticalScroll(id="chat-log")
                with Vertical(id="input-row"):
                    yield Input(
                        placeholder="Ask a question about your papers…",
                        id="question-input",
                    )
            with Vertical(id="dock-pane"):
                yield ListView(id="citation-list")
                with VerticalScroll(id="detail-scroll"):
                    yield CitationDetail(id="detail-body")
        yield Static("", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        """Set pane titles, greet the user, and probe backend health."""
        self.query_one("#chat-pane").border_title = "QA Console"
        self.query_one("#dock-pane").border_title = "Citations & Figures"
        self.query_one(CitationDetail).show_placeholder()
        self.query_one("#question-input", Input).focus()

        self._refresh_titles()
        counts = self._store.counts()
        self._system(
            f"Knowledge base ready — {counts['chunks']} passages and "
            f"{counts['figures']} figures indexed from {self._settings.pdf_dir}."
        )
        if self._kb_registry is not None and len(self._kb_registry) > 1:
            self._system(
                f"{len(self._kb_registry)} knowledge bases registered — press Ctrl+K to switch."
            )
        self._set_status("Checking Ollama…")
        self._check_backend()

    # -------------------------------------------------------------- helpers
    def _set_status(self, message: str) -> None:
        """Update the one-line status bar."""
        self.query_one("#status-bar", Static).update(message)

    def _append(self, renderable: Text | str, *, classes: str = "") -> Static:
        """Append a block to the chat log and scroll to it."""
        log = self.query_one("#chat-log", VerticalScroll)
        widget = Static(renderable, classes=classes)
        log.mount(widget)
        log.scroll_end(animate=False)
        return widget

    def _system(self, message: str) -> None:
        """Append a system notice."""
        text = Text()
        text.append("system  ", style="bold cyan")
        text.append(message, style="dim")
        self._append(text)

    def _error(self, message: str) -> None:
        """Append an error notice."""
        text = Text()
        text.append("error   ", style="bold red")
        text.append(message)
        self._append(text)

    @work(thread=True, exclusive=True, group="health")
    def _check_backend(self) -> None:
        """Probe Ollama off the event loop and report the result."""
        status = self._connector.health()
        self.call_from_thread(self._set_status, status.describe())
        if not status.available:
            self.call_from_thread(
                self._system,
                "Ollama is not reachable — answers are disabled, but retrieval "
                "and citations still work. Start it with `ollama serve`.",
            )
        elif not status.models:
            self.call_from_thread(
                self._system, "Ollama has no models installed. Run: ollama pull llama3.2:3b"
            )

    # ------------------------------------------------------------ questions
    @on(Input.Submitted, "#question-input")
    def _on_question(self, event: Input.Submitted) -> None:
        """Handle a submitted question."""
        question = event.value.strip()
        if not question:
            return
        if self._busy:
            self._set_status("Still answering the previous question…")
            return

        event.input.value = ""
        stamp = datetime.now().strftime("%H:%M:%S")
        text = Text()
        text.append(f"you {stamp}  ", style="bold green")
        text.append(question)
        self._append(text)

        self._busy = True
        self._stream_buffer = ""
        header = Text()
        header.append("assistant  ", style="bold magenta")
        self._stream_target = self._append(header)
        self._set_status("Retrieving context…")
        self._run_query(question)

    @work(thread=True, exclusive=True, group="rag")
    def _run_query(self, question: str) -> None:
        """Stream a RAG answer on a worker thread."""
        try:
            for event in self._engine.answer_stream(question):
                if event.kind == "context" and event.bundle is not None:
                    self.call_from_thread(self._show_context, event.bundle)
                elif event.kind == "token":
                    self.call_from_thread(self._stream_token, event.text)
                elif event.kind == "done" and event.result is not None:
                    self.call_from_thread(self._finish, event.result)
                elif event.kind == "error":
                    self.call_from_thread(self._error, event.text)
                    self.call_from_thread(self._release)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Query worker failed")
            self.call_from_thread(self._error, f"Query failed: {exc}")
            self.call_from_thread(self._release)

    def _release(self) -> None:
        """Re-enable input after a turn ends."""
        self._busy = False
        self._stream_target = None

    def _stream_token(self, fragment: str) -> None:
        """Append a streamed fragment to the in-flight answer block."""
        if self._stream_target is None:
            return
        self._stream_buffer += fragment
        text = Text()
        text.append("assistant  ", style="bold magenta")
        text.append(self._stream_buffer)
        self._stream_target.update(text)
        self.query_one("#chat-log", VerticalScroll).scroll_end(animate=False)

    def _show_context(self, bundle: RetrievalBundle) -> None:
        """Populate the dock from retrieval, before the answer arrives."""
        self._set_status(
            f"Retrieved {len(bundle.chunks)} passages and {len(bundle.figures)} figures — generating…"
        )
        preview = citations_from_context([*bundle.chunks, *bundle.figures])
        self._load_citations(preview, note="retrieved context")

    def _finish(self, result: AnswerResult) -> None:
        """Render the final answer state and its verified citations."""
        if result.citations:
            self._load_citations(result.citations, note="cited by answer")

        for warning in result.warnings:
            self._system(warning)

        verified = len(result.verified_citations)
        total = len(result.citations)
        self._set_status(
            f"Answered in {result.elapsed_seconds:.1f}s · {result.model or 'no model'} · "
            f"{verified}/{total} citations verified"
        )
        self._release()

    # ------------------------------------------------------------ citations
    def _load_citations(self, citations: list[Citation], *, note: str) -> None:
        """Replace the dock contents with ``citations``."""
        self._citations = citations
        listing = self.query_one("#citation-list", ListView)
        listing.clear()
        for index, citation in enumerate(citations, start=1):
            listing.append(CitationItem(citation, index))

        self.query_one("#dock-pane").border_title = f"Citations & Figures ({len(citations)} · {note})"
        detail = self.query_one(CitationDetail)
        if citations:
            # Highlight the first row up front, otherwise ListView starts at
            # index None and the first arrow press is spent selecting row 0.
            listing.index = 0
            detail.show_citation(citations[0])
        else:
            listing.index = None
            detail.show_placeholder()

    @on(ListView.Highlighted, "#citation-list")
    def _on_highlight(self, event: ListView.Highlighted) -> None:
        """Show the highlighted citation in the detail panel."""
        item = event.item
        if isinstance(item, CitationItem):
            self.query_one(CitationDetail).show_citation(item.citation)

    @on(ListView.Selected, "#citation-list")
    def _on_select(self, event: ListView.Selected) -> None:
        """Enter on a citation row opens its figure."""
        item = event.item
        if isinstance(item, CitationItem):
            self.query_one(CitationDetail).show_citation(item.citation)
            self._open_citation(item.citation)

    def _current_citation(self) -> Citation | None:
        """Return the citation currently highlighted in the dock."""
        listing = self.query_one("#citation-list", ListView)
        item = listing.highlighted_child
        if isinstance(item, CitationItem):
            return item.citation
        return self._citations[0] if self._citations else None

    def _open_citation(self, citation: Citation) -> None:
        """Open a citation's image, falling back to the source PDF."""
        target = citation.image_path or citation.source_path
        if not target:
            self._set_status("This citation has no figure or file to open.")
            return
        ok, message = open_in_system_viewer(target)
        self._set_status(message if ok else f"⚠ {message}")

    def _refresh_titles(self) -> None:
        """Show the active knowledge base in the pane title and app subtitle."""
        name = self._kb_registry.active_name if self._kb_registry is not None else ""
        chat = self.query_one("#chat-pane")
        chat.border_title = f"QA Console — {name}" if name else "QA Console"
        self.sub_title = name

    def action_switch_base(self) -> None:
        """Open the knowledge-base picker."""
        if self._kb_registry is None:
            self._set_status("No knowledge-base registry available.")
            return
        if self._busy or self._switching:
            self._set_status("Busy — wait for the current operation to finish.")
            return

        counts = {self._kb_registry.active_name: self._store.counts()["chunks"]}
        self.push_screen(
            KnowledgeBasePicker(self._kb_registry.all(), self._kb_registry.active_name, counts),
            self._on_base_chosen,
        )

    def _on_base_chosen(self, name: str | None) -> None:
        """Handle the picker's result."""
        if not name or self._kb_registry is None:
            return
        if name == self._kb_registry.active_name:
            self._set_status(f"Already using '{name}'.")
            return
        ref = self._kb_registry.get(name)
        if not ref.exists():
            self._error(f"Corpus folder for '{name}' is missing: {ref.pdf_dir}")
            return
        self._switching = True
        self._set_status(f"Switching to '{name}'…")
        self._switch_worker(ref)

    @work(thread=True, exclusive=True, group="switch")
    def _switch_worker(self, ref: KnowledgeBaseRef) -> None:
        """Tear down the current index and open the selected one.

        Runs off the event loop because opening a ChromaDB collection touches
        disk. The previous manifest is closed and the previous store dropped so
        switching does not accumulate memory.
        """
        try:
            old_manifest = self._manifest
            settings = ref.apply(self._settings)
            settings.ensure_directories()
            service, store, manifest = build_service(settings)
        except Exception as exc:
            LOGGER.exception("Switch to %s failed", ref.name)
            self.call_from_thread(self._error, f"Could not open '{ref.name}': {exc}")
            self.call_from_thread(self._finish_switch, None)
            return

        if old_manifest is not None:
            try:
                old_manifest.close()
            except Exception:  # pragma: no cover - best effort
                LOGGER.debug("Closing previous manifest failed", exc_info=True)

        self._settings = settings
        self._store = store
        self._service = service
        self._manifest = manifest
        self._engine = RagEngine(settings, store, self._connector)
        if self._kb_registry is not None:
            self._kb_registry.set_active(ref.name)
            try:
                self._kb_registry.save()
            except OSError as exc:  # pragma: no cover
                LOGGER.warning("Could not persist active knowledge base: %s", exc)

        self.call_from_thread(self._finish_switch, ref)

    def _finish_switch(self, ref: KnowledgeBaseRef | None) -> None:
        """Update the interface after a switch attempt."""
        self._switching = False
        if ref is None:
            self._set_status("Switch failed.")
            return
        self._refresh_titles()
        self._load_citations([], note="none yet")
        counts = self._store.counts()
        self._system(
            f"Switched to '{ref.name}' — {counts['chunks']} passages and "
            f"{counts['figures']} figures from {ref.pdf_dir}."
        )
        if counts["chunks"] == 0:
            self._system(
                f"'{ref.name}' has no indexed content yet. Press Ctrl+R to index it."
            )
        self._set_status(
            f"Knowledge base: {ref.name} · {counts['chunks']} passages"
        )

    # -------------------------------------------------------------- actions
    def action_open_figure(self) -> None:
        """Open the highlighted citation's figure in the system viewer."""
        citation = self._current_citation()
        if citation is None:
            self._set_status("No citation selected.")
            return
        self._open_citation(citation)

    def action_clear_chat(self) -> None:
        """Clear the conversation transcript."""
        log = self.query_one("#chat-log", VerticalScroll)
        for child in list(log.children):
            child.remove()
        self._system("Conversation cleared.")

    def action_show_status(self) -> None:
        """Print an index summary into the chat log."""
        status = self._service.status()
        lines = [
            f"corpus            {status['pdf_dir']}",
            f"documents indexed {status['documents_indexed']}  ({status['pages']} pages)",
            f"passage vectors   {status['vectors_chunks']}",
            f"figure vectors    {status['vectors_figures']}",
            f"tables extracted  {status['tables']}",
            f"embedding         {status['embedding']}",
            f"by status         {status['by_status']}",
        ]
        self._system("Index status:\n  " + "\n  ".join(lines))

    def action_rescan(self) -> None:
        """Re-scan the corpus directory for new or changed papers."""
        if self._busy:
            self._set_status("Busy — try again once the answer completes.")
            return
        self._system("Rescanning corpus…")
        self._set_status("Rescanning…")
        self._rescan_worker()

    @work(thread=True, exclusive=True, group="ingest")
    def _rescan_worker(self) -> None:
        """Run a corpus sync off the event loop."""
        try:
            report = self._service.sync()
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Rescan failed")
            self.call_from_thread(self._error, f"Rescan failed: {exc}")
            return
        self.call_from_thread(self._system, f"Rescan complete — {report.summary()}")
        self.call_from_thread(self._set_status, "Rescan complete")

    # ------------------------------------------------------------- watcher
    def on_watcher_event(self, event: str, message: str) -> None:
        """Receive folder-watcher notifications from its background thread."""
        try:
            self.call_from_thread(self._system, f"watcher: {message}")
        except Exception:  # pragma: no cover - app may be shutting down
            LOGGER.debug("Dropped watcher event %s", event)
