"""Rich rendering helpers for Copex TUI."""

import io
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

# NOTE: Avoid __future__.annotations for standalone import in tests (Python 3.14
# dataclasses resolves string annotations via sys.modules).
from rich.box import ROUNDED
from rich.console import Console, Group, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:  # pragma: no cover
    # Only for type checking; runtime imports are lazy.
    from copex.ui import Icons as Icons
    from copex.ui import Theme as Theme


def _get_ui():
    """Lazy import Theme/Icons from copex.ui."""
    from copex.ui import Icons, Theme

    return Icons, Theme


def __getattr__(name: str):
    # Allow `from copex.tui.render import Theme, Icons` without importing copex.ui
    # during module import.
    if name == "Icons":
        return _get_ui()[0]
    if name == "Theme":
        return _get_ui()[1]
    raise AttributeError(name)


__all__ = [
    "RenderConfig",
    "render_to_ansi",
    "render_status_bar",
    "render_tool_call_collapsed",
    "render_tool_call_expanded",
    "render_palette",
    "render_reasoning_collapsed",
    "render_reasoning_expanded",
    "render_message",
    "render_prompt_prefix",
    "render_continuation_prefix",
    "render_help_panel",
    "render_stash_indicator",
    "render_spinner",
    "Theme",
    "Icons",
]


@dataclass
class RenderConfig:
    """Configuration for rendering."""

    width: int | None = None
    show_reasoning: bool = True
    show_tools: bool = True
    compact_mode: bool = False
    syntax_theme: str = "monokai"


def render_to_ansi(
    renderable: RenderableType,
    *,
    width: int | None = None,
    force_terminal: bool = True,
) -> str:
    """
    Render a Rich renderable to ANSI escape codes.

    This is useful for prompt_toolkit integration where we need
    ANSI-formatted strings.
    """
    console = Console(
        file=io.StringIO(),
        width=width or 120,
        force_terminal=force_terminal,
        color_system="truecolor",
        legacy_windows=False,
    )
    console.print(renderable, end="")
    return console.file.getvalue()


def render_status_bar(
    model: str,
    reasoning: str,
    tokens: int | None,
    cost: float | None,
    is_streaming: bool = False,
    activity: str = "idle",
    width: int | None = None,
) -> str:
    """Render the status bar as ANSI text."""
    Icons, Theme = _get_ui()
    parts = []

    # Model
    parts.append(Text(f"  {model} ", style=f"bold {Theme.PRIMARY}"))

    # Separator
    parts.append(Text("│", style=Theme.BORDER))

    # Reasoning effort
    parts.append(Text(f" {Icons.BRAIN} {reasoning} ", style=Theme.ACCENT))

    # Separator
    parts.append(Text("│", style=Theme.BORDER))

    # Tokens
    if tokens is None:
        parts.append(Text(" N/A tokens ", style=Theme.INFO))
    else:
        parts.append(Text(f" {tokens:,} tokens ", style=Theme.INFO))

    # Separator
    parts.append(Text("│", style=Theme.BORDER))

    # Cost
    if cost is None:
        parts.append(Text(" $N/A ", style=Theme.WARNING))
    else:
        parts.append(Text(f" ${cost:.4f} ", style=Theme.SUCCESS if cost < 1.0 else Theme.WARNING))

    # Activity indicator (right aligned)
    if is_streaming:
        spinner = "⠋"  # Will be animated by caller
        activity_text = f" {spinner} {activity} "
    else:
        activity_text = f" {Icons.DONE} ready "

    # Build full bar
    result = Text()
    for part in parts:
        result.append_text(part)

    # Add activity on right
    result.append(" " * 5)  # Spacer
    result.append(activity_text, style=Theme.PRIMARY if is_streaming else Theme.MUTED)

    return render_to_ansi(result, width=width)


def render_tool_call_collapsed(
    name: str,
    icon: str,
    status: str,
    duration: float | None = None,
    is_selected: bool = False,
) -> Text:
    """Render a collapsed tool call as a single line."""
    Icons, Theme = _get_ui()
    text = Text()

    # Status indicator
    if status == "running":
        text.append("⏳ ", style=Theme.WARNING)
    elif status == "success":
        text.append(f"{Icons.DONE} ", style=Theme.SUCCESS)
    else:
        text.append(f"{Icons.ERROR} ", style=Theme.ERROR)

    # Icon and name
    text.append(f"{icon} ", style=Theme.WARNING)
    text.append(name, style=f"bold {Theme.WARNING}" if status == "running" else Theme.MUTED)

    # Duration
    if duration is not None:
        text.append(f" ({duration:.1f}s)", style=Theme.MUTED)

    # Expand hint
    text.append("  ▸", style=Theme.MUTED)

    # Selection indicator
    if is_selected:
        text.stylize("reverse")

    return text


def render_tool_call_expanded(
    name: str,
    icon: str,
    status: str,
    arguments: dict[str, Any],
    result: str | None = None,
    duration: float | None = None,
) -> Panel:
    """Render an expanded tool call with details."""
    Icons, Theme = _get_ui()
    elements = []

    # Header
    header = Text()
    if status == "running":
        header.append("⏳ ", style=Theme.WARNING)
    elif status == "success":
        header.append(f"{Icons.DONE} ", style=Theme.SUCCESS)
    else:
        header.append(f"{Icons.ERROR} ", style=Theme.ERROR)
    header.append(f"{icon} ", style=Theme.WARNING)
    header.append(name, style=f"bold {Theme.WARNING}")
    if duration is not None:
        header.append(f" ({duration:.1f}s)", style=Theme.MUTED)
    elements.append(header)

    # Arguments
    if arguments:
        args_table = Table(show_header=False, box=None, padding=(0, 1))
        args_table.add_column("Key", style=Theme.MUTED)
        args_table.add_column("Value", style=Theme.MESSAGE)

        for key, value in arguments.items():
            val_str = str(value)
            if len(val_str) > 80:
                val_str = val_str[:77] + "..."
            args_table.add_row(key, val_str)

        elements.append(Text())
        elements.append(args_table)

    # Result
    if result:
        elements.append(Text())
        elements.append(Text("Result:", style=Theme.SUBHEADER))

        # Truncate long results
        result_display = result
        if len(result) > 500:
            result_display = result[:500] + f"\n... ({len(result) - 500} more chars)"

        elements.append(Text(result_display, style=Theme.MUTED))

    return Panel(
        Group(*elements),
        title=f"[{Theme.WARNING}]{icon} {name}[/{Theme.WARNING}]",
        title_align="left",
        border_style=Theme.BORDER_ACTIVE if status == "running" else Theme.BORDER,
        padding=(0, 1),
        box=ROUNDED,
    )


def render_palette(
    commands: list[tuple[str, str, str, int]],  # (id, label, description, score)
    query: str,
    selected_index: int,
    width: int | None = None,
) -> str:
    """Render the command palette as ANSI text."""
    Icons, Theme = _get_ui()
    elements = []

    # Header with search box
    header = Text()
    header.append(" 🔍 ", style=Theme.PRIMARY)
    header.append(query if query else "Type to search...", style="bold" if query else Theme.MUTED)
    header.append("│", style=Theme.MUTED)
    elements.append(header)
    elements.append(Text())

    # Command list
    if not commands:
        elements.append(Text("  No matching commands", style=Theme.MUTED))
    else:
        for i, (cmd_id, label, description, score) in enumerate(commands[:10]):
            line = Text()

            # Selection indicator
            if i == selected_index:
                line.append(" ▸ ", style=f"bold {Theme.PRIMARY}")
            else:
                line.append("   ")

            # Label
            line.append(label, style="bold" if i == selected_index else "")

            # Description
            line.append(f"  {description}", style=Theme.MUTED)

            elements.append(line)

    # Help text
    elements.append(Text())
    help_text = Text()
    help_text.append("↑↓", style="bold")
    help_text.append(" navigate  ", style=Theme.MUTED)
    help_text.append("Enter", style="bold")
    help_text.append(" select  ", style=Theme.MUTED)
    help_text.append("Esc", style="bold")
    help_text.append(" close", style=Theme.MUTED)
    elements.append(help_text)

    panel = Panel(
        Group(*elements),
        title=f"[{Theme.PRIMARY}]Command Palette[/{Theme.PRIMARY}]",
        title_align="left",
        border_style=Theme.BORDER_ACTIVE,
        padding=(0, 1),
        box=ROUNDED,
        width=min(width or 80, 80),
    )

    return render_to_ansi(panel, width=width)


def render_reasoning_collapsed(
    reasoning: str,
    char_count: int,
) -> Text:
    """Render collapsed reasoning as a single line."""
    Icons, Theme = _get_ui()
    text = Text()
    text.append(f"{Icons.BRAIN} ", style=Theme.ACCENT)
    text.append(f"Reasoning ({char_count:,} chars)", style=Theme.MUTED)
    text.append("  ▸", style=Theme.MUTED)
    return text


def render_reasoning_expanded(reasoning: str) -> Panel:
    """Render expanded reasoning panel."""
    Icons, Theme = _get_ui()
    return Panel(
        Markdown(reasoning),
        title=f"[{Theme.ACCENT}]{Icons.BRAIN} Reasoning[/{Theme.ACCENT}]",
        title_align="left",
        border_style=Theme.BORDER,
        padding=(0, 1),
        box=ROUNDED,
    )


def render_message(content: str, is_streaming: bool = False) -> Panel:
    """Render the main message panel."""
    Icons, Theme = _get_ui()
    if is_streaming:
        # Show raw text while streaming for better performance
        text = Text(content, style=Theme.MESSAGE)
        text.append("▌", style=f"bold {Theme.PRIMARY}")
        renderable: RenderableType = text
    else:
        # Render as markdown when complete
        renderable = Markdown(content)

    return Panel(
        renderable,
        title=f"[{Theme.PRIMARY}]{Icons.ROBOT} Response[/{Theme.PRIMARY}]",
        title_align="left",
        border_style=Theme.BORDER_ACTIVE if is_streaming else Theme.BORDER,
        padding=(0, 1),
        box=ROUNDED,
    )


def render_prompt_prefix(model: str, is_multiline: bool = False) -> str:
    """Render the prompt prefix."""
    Icons, Theme = _get_ui()
    text = Text()
    text.append(f"{Icons.ARROW_RIGHT} ", style=f"bold {Theme.SUCCESS}")

    if is_multiline:
        return render_to_ansi(text)

    return render_to_ansi(text)


def render_continuation_prefix(line_number: int) -> str:
    """Render the continuation prefix for multiline input."""
    Icons, Theme = _get_ui()
    text = Text()
    text.append("   ... ", style=Theme.MUTED)
    return render_to_ansi(text)


def render_help_panel(shortcuts: list[tuple[str, str]]) -> str:
    """Render the help panel with keyboard shortcuts."""
    Icons, Theme = _get_ui()
    table = Table(
        show_header=True,
        header_style=f"bold {Theme.PRIMARY}",
        box=ROUNDED,
        padding=(0, 2),
    )
    table.add_column("Shortcut", style="bold")
    table.add_column("Action", style=Theme.MESSAGE)

    for shortcut, description in shortcuts:
        table.add_row(shortcut, description)

    panel = Panel(
        table,
        title=f"[{Theme.PRIMARY}]⌨️ Keyboard Shortcuts[/{Theme.PRIMARY}]",
        title_align="left",
        border_style=Theme.BORDER,
        padding=(0, 1),
        box=ROUNDED,
    )

    return render_to_ansi(panel)


def render_stash_indicator(count: int, current_index: int) -> Text:
    """Render the stash indicator for status bar."""
    Icons, Theme = _get_ui()
    text = Text()
    text.append("📋 ", style=Theme.INFO)
    text.append(f"{current_index + 1}/{count}", style=Theme.INFO)
    return text


def render_spinner(frame: int) -> str:
    """Get a spinner frame."""
    spinners = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    return spinners[frame % len(spinners)]
