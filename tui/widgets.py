"""Reusable widgets for the knowledge-base TUI."""

from __future__ import annotations

import logging
import platform
import subprocess
from pathlib import Path

from rich.text import Text
from textual.widgets import ListItem, Static

from engine.models import Citation

LOGGER = logging.getLogger(__name__)


def open_in_system_viewer(path: str | Path) -> tuple[bool, str]:
    """Open a file with the platform's default application.

    Uses ``xdg-open`` on Linux, ``open`` on macOS, and ``os.startfile`` on
    Windows. The child process is detached so closing the viewer never blocks
    the TUI, and its output is discarded so it cannot corrupt the terminal.

    Returns:
        ``(success, message)`` suitable for display in the status bar.
    """
    target = Path(path).expanduser()
    if not target.exists():
        return False, f"File not found: {target.name}"

    system = platform.system()
    try:
        if system == "Linux":
            command = ["xdg-open", str(target)]
        elif system == "Darwin":
            command = ["open", str(target)]
        elif system == "Windows":
            import os

            os.startfile(str(target))  # type: ignore[attr-defined]  # noqa: S606
            return True, f"Opened {target.name}"
        else:
            return False, f"Unsupported platform: {system}"

        subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError:
        return False, "No image viewer found (install xdg-utils)"
    except Exception as exc:  # pragma: no cover - platform dependent
        LOGGER.exception("Viewer launch failed")
        return False, f"Could not open viewer: {exc}"

    return True, f"Opened {target.name}"


class CitationItem(ListItem):
    """A single citation row in the reference dock."""

    def __init__(self, citation: Citation, index: int) -> None:
        super().__init__()
        self.citation = citation
        self._index = index

    def compose(self):
        """Render the row: marker, document, page/line, and content type."""
        citation = self.citation
        marker = "✓" if citation.verified else "?"
        style = "green" if citation.verified else "yellow"

        if citation.content_type == "figure":
            icon = "🖼"
        elif citation.content_type == "table":
            icon = "▦"
        else:
            icon = "¶"

        text = Text()
        text.append(f"{marker} ", style=style)
        text.append(f"{self._index}. ", style="dim")
        text.append(f"{icon} ")
        text.append(f"{citation.document_name}\n", style="bold")
        text.append(
            f"    Page {citation.page}, Line {citation.line_start}-{citation.line_end}",
            style="dim cyan",
        )
        if citation.image_path:
            text.append("  [image]", style="magenta")
        yield Static(text)


class CitationDetail(Static):
    """Detail view for the currently selected citation."""

    def show_placeholder(self) -> None:
        """Render the empty state."""
        self.update(
            Text.from_markup(
                "[dim]Select a citation to inspect the source passage.\n\n"
                "[b]Enter[/b] or [b]Ctrl+O[/b] opens an associated figure in "
                "your image viewer.[/dim]"
            )
        )

    def show_citation(self, citation: Citation) -> None:
        """Render one citation's full provenance and text snippet."""
        text = Text()
        status = "verified against retrieved context" if citation.verified else "NOT verified"
        text.append("Status: ", style="bold")
        text.append(f"{status}\n", style="green" if citation.verified else "yellow")

        text.append("Document: ", style="bold")
        text.append(f"{citation.document_name}\n")
        text.append("Page: ", style="bold")
        text.append(f"{citation.page}    ")
        text.append("Lines: ", style="bold")
        text.append(f"{citation.line_start}-{citation.line_end}\n")
        text.append("Type: ", style="bold")
        text.append(f"{citation.content_type}\n")

        if citation.caption:
            text.append("\nCaption\n", style="bold")
            text.append(f"{citation.caption}\n", style="italic")

        if citation.image_path:
            text.append("\nImage\n", style="bold")
            text.append(f"{citation.image_path}\n", style="magenta")
            text.append("Press Enter or Ctrl+O to open.\n", style="dim")

        if citation.snippet:
            text.append("\nSource text\n", style="bold")
            text.append(f"{citation.snippet}\n", style="white")

        if citation.source_path:
            text.append("\nFile\n", style="bold")
            text.append(f"{citation.source_path}\n", style="dim")

        self.update(text)
