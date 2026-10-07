"""Modal screen for switching between knowledge bases."""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import ListItem, ListView, Static

from engine.registry import KnowledgeBaseRef


class BaseItem(ListItem):
    """One selectable knowledge base."""

    def __init__(self, ref: KnowledgeBaseRef, *, is_active: bool, indexed: int | None) -> None:
        super().__init__()
        self.ref = ref
        self._is_active = is_active
        self._indexed = indexed

    def compose(self) -> ComposeResult:
        """Render the name, corpus size, and index state."""
        ref = self.ref
        text = Text()
        text.append("● " if self._is_active else "  ", style="green" if self._is_active else "dim")
        text.append(f"{ref.name}", style="bold")
        if self._is_active:
            text.append("  (current)", style="dim green")
        text.append("\n")

        if not ref.exists():
            text.append("    corpus folder missing: ", style="red")
            text.append(f"{ref.pdf_dir}\n", style="dim red")
        else:
            pdfs = ref.pdf_count()
            detail = f"    {pdfs} PDF{'s' if pdfs != 1 else ''}"
            if self._indexed is not None:
                detail += f" · {self._indexed} passages indexed"
            text.append(f"{detail}\n", style="dim cyan")
        if ref.description:
            text.append(f"    {ref.description}\n", style="dim italic")
        yield Static(text)


class KnowledgeBasePicker(ModalScreen[str | None]):
    """Dialog listing the registered knowledge bases.

    Dismisses with the chosen name, or ``None`` when cancelled.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    DEFAULT_CSS = """
    KnowledgeBasePicker {
        align: center middle;
    }
    #picker-box {
        width: 74;
        height: auto;
        max-height: 80%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #picker-title {
        text-style: bold;
        color: $accent;
        padding-bottom: 1;
    }
    #picker-list {
        height: auto;
        max-height: 20;
        background: $surface;
    }
    #picker-list > ListItem {
        padding: 0 1;
        background: $surface;
    }
    #picker-list > ListItem.--highlight {
        background: $accent 30%;
    }
    #picker-hint {
        padding-top: 1;
        color: $text-muted;
    }
    """

    def __init__(
        self,
        refs: list[KnowledgeBaseRef],
        active_name: str,
        counts: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self._refs = refs
        self._active_name = active_name
        self._counts = counts or {}

    def compose(self) -> ComposeResult:
        """Build the dialog."""
        with Vertical(id="picker-box"):
            yield Static("Switch knowledge base", id="picker-title")
            yield ListView(
                *[
                    BaseItem(
                        ref,
                        is_active=(ref.name == self._active_name),
                        indexed=self._counts.get(ref.name),
                    )
                    for ref in self._refs
                ],
                id="picker-list",
            )
            yield Static(
                "↑↓ move · Enter switch · Esc cancel     "
                "add one with:  python main.py kb add <name> --pdf-dir <path>",
                id="picker-hint",
            )

    def on_mount(self) -> None:
        """Focus the list and highlight the active base."""
        listing = self.query_one("#picker-list", ListView)
        listing.focus()
        for index, ref in enumerate(self._refs):
            if ref.name == self._active_name:
                listing.index = index
                break
        else:
            listing.index = 0 if self._refs else None

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Return the chosen base name."""
        item = event.item
        if isinstance(item, BaseItem):
            self.dismiss(item.ref.name)

    def action_cancel(self) -> None:
        """Close without switching."""
        self.dismiss(None)
