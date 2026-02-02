"""Rich rendering helpers for Copex TUI."""

import io
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

# NOTE: Avoid __future__.annotations for standalone import in tests (Python 3.14
# dataclasses resolves string annotations via sys.modules).

from rich.box import ROUNDED
from rich.console import Console, Group, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.style import Style
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
    "Spinners",
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
    "render_welcome_banner",
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
    duration: str | None = None,
    is_streaming: bool = False,
    activity: str = "idle",
    width: int | None = None,
    spinner_frame: int = 0,
) -> str:
    """Render the status bar as ANSI text."""
    Icons, Theme = _get_ui()
    parts = []

    # Elegant separators
    sep = " │ "
    sep_style = Theme.MUTED

    # Model with icon
    parts.append(Text(f"  {Icons.ROBOT} ", style=Theme.MUTED))
    parts.append(Text(model, style=f"bold {Theme.PRIMARY}"))
    parts.append(Text(sep, style=sep_style))

    # Reasoning effort with brain icon
    parts.append(Text(f"{Icons.BRAIN} ", style=Theme.ACCENT))
    parts.append(Text(reasoning, style=Theme.ACCENT))
    parts.append(Text(sep, style=sep_style))

    # Duration
    if duration:
        parts.append(Text(f"{Icons.CLOCK} ", style=Theme.MUTED))
        parts.append(Text(duration, style=Theme.INFO))
        parts.append(Text(sep, style=sep_style))

    # Tokens with formatted number
    if tokens is None:
        parts.append(Text("— tokens", style=Theme.MUTED))
    else:
        parts.append(Text(f"{tokens:,}", style=Theme.INFO))
        parts.append(Text(" tokens", style=Theme.MUTED))
    parts.append(Text(sep, style=sep_style))

    # Cost with color coding
    if cost is None:
        parts.append(Text("$—", style=Theme.MUTED))
    else:
        cost_style = Theme.SUCCESS if cost < 0.10 else (Theme.WARNING if cost < 1.0 else Theme.ERROR)
        parts.append(Text(f"${cost:.4f}", style=cost_style))

    # Activity indicator (right side with spinner)
    if is_streaming:
        spinner = render_spinner(spinner_frame, style="pulse")
        activity_style = Theme.PRIMARY if activity == "responding" else (
            Theme.ACCENT if activity == "reasoning" else Theme.WARNING
        )
        parts.append(Text(f"  {spinner} ", style=f"bold {activity_style}"))
        parts.append(Text(activity, style=activity_style))
    else:
        parts.append(Text(f"  {Icons.DONE} ", style=Theme.SUCCESS))
        parts.append(Text("ready", style=Theme.MUTED))
    
    # Build full bar
    result = Text()
    for part in parts:
        result.append_text(part)
    
    return render_to_ansi(result, width=width)


def _format_tool_args(arguments: dict[str, Any] | None, max_len: int = 48) -> str:
    """Format a compact argument preview for tool calls."""
    if not arguments:
        return ""

    preferred_keys = ("path", "file", "command", "pattern", "query", "url", "name")
    parts: list[str] = []

    def add_part(key: str, value: Any) -> None:
        val_str = str(value).replace("\n", " ")
        if len(val_str) > 32:
            val_str = val_str[:29] + "..."
        parts.append(f"{key}={val_str}")

    for key in preferred_keys:
        if key in arguments and arguments[key] is not None:
            add_part(key, arguments[key])

    if not parts:
        for key, value in list(arguments.items())[:2]:
            add_part(str(key), value)

    summary = " ".join(parts)
    if len(summary) > max_len:
        summary = summary[: max_len - 1] + "…"
    return summary


def render_tool_call_collapsed(
    name: str,
    icon: str,
    status: str,
    duration: float | None = None,
    arguments: dict[str, Any] | None = None,
    is_selected: bool = False,
    spinner_frame: int = 0,
) -> Text:
    """Render a collapsed tool call as a single line."""
    Icons, Theme = _get_ui()
    text = Text()

    # Status indicator with appropriate styling
    if status == "running":
        spinner = render_spinner(spinner_frame, style="dots")
        text.append(f"{spinner} ", style=f"bold {Theme.WARNING}")
    elif status == "success":
        text.append(f"{Icons.DONE} ", style=Theme.SUCCESS)
    else:
        text.append(f"{Icons.ERROR} ", style=Theme.ERROR)

    # Icon and name with cleaner styling
    text.append(f"{icon} ", style=Theme.MUTED)
    name_style = Theme.WARNING if status == "running" else Theme.MESSAGE
    text.append(name, style=name_style)

    args_preview = _format_tool_args(arguments)
    if args_preview:
        text.append(f" • {args_preview}", style=Theme.MUTED)

    # Duration with subtle formatting
    if duration is not None:
        text.append(f"  {duration:.1f}s", style=Theme.MUTED)
    elif status == "running":
        text.append("  • in progress", style=Theme.MUTED)

    # Status label for quick scan
    status_label = "Running" if status == "running" else ("OK" if status == "success" else "Failed")
    status_style = Theme.WARNING if status == "running" else (Theme.SUCCESS if status == "success" else Theme.ERROR)
    text.append(f"  {status_label}", style=status_style)

    # Expand hint (subtle chevron)
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
    elif status == "running":
        header.append(" (in progress)", style=Theme.MUTED)
    header_status = "Running" if status == "running" else ("OK" if status == "success" else "Failed")
    header.append(
        f" • {header_status}",
        style=Theme.WARNING if status == "running" else (Theme.SUCCESS if status == "success" else Theme.ERROR),
    )
    elements.append(header)
    
    # Arguments
    elements.append(Text("Arguments", style=Theme.SUBHEADER))
    if arguments:
        args_table = Table(show_header=False, box=None, padding=(0, 1))
        args_table.add_column("Key", style=Theme.MUTED)
        args_table.add_column("Value", style=Theme.MESSAGE, overflow="fold")
        
        for key, value in arguments.items():
            val_str = str(value)
            if len(val_str) > 80:
                val_str = val_str[:77] + "..."
            args_table.add_row(key, val_str)
        
        elements.append(args_table)
    else:
        elements.append(Text("No arguments.", style=Theme.MUTED))
    
    # Result
    elements.append(Text())
    elements.append(Text("Output", style=Theme.SUBHEADER))
    if result:
        # Truncate long results
        result_display = result
        if len(result) > 500:
            result_display = result[:500] + f"\n... ({len(result) - 500} more chars)"
        
        elements.append(Text(result_display, style=Theme.MUTED))
    else:
        if status == "running":
            elements.append(Text("Output pending…", style=Theme.WARNING))
        elif status == "error":
            elements.append(Text("No output.", style=Theme.ERROR))
        else:
            elements.append(Text("No output.", style=Theme.MUTED))

    if status == "error" and result:
        elements.append(Text())
        elements.append(Text("Check logs for details.", style=Theme.ERROR))
    
    if status == "running":
        border_style = Theme.BORDER_ACTIVE
    elif status == "success":
        border_style = Theme.SUCCESS
    elif status == "error":
        border_style = Theme.ERROR
    else:
        border_style = Theme.BORDER

    body = Group(*elements)
    return Panel(
        body,
        title=f"[{Theme.WARNING}]▾ {icon} {name}[/{Theme.WARNING}]",
        title_align="left",
        border_style=border_style,
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
    header.append(f" {Icons.SEARCH} ", style=Theme.PRIMARY)
    if query:
        header.append(query, style=f"bold {Theme.MESSAGE}")
    else:
        header.append("Type to search…", style=Theme.MUTED)
    header.append("  │ ", style=Theme.MUTED)
    header.append(f"{len(commands)} results", style=Theme.MUTED if commands else Theme.MUTED)
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
                line.append("   ", style=Theme.MUTED)

            # Label
            label_style = f"bold {Theme.PRIMARY}" if i == selected_index else f"bold {Theme.MESSAGE}"
            line.append(label, style=label_style)

            # Description
            desc_style = Theme.MUTED if i != selected_index else Theme.SUBHEADER
            line.append(" — ", style=Theme.MUTED)
            line.append(description, style=desc_style)

            elements.append(line)
    
    # Help text
    elements.append(Text())
    help_text = Text()
    help_text.append("↑↓", style="bold")
    help_text.append(" navigate  ", style=Theme.MUTED)
    help_text.append("Enter", style="bold")
    help_text.append(" select  ", style=Theme.MUTED)
    help_text.append("Backspace", style="bold")
    help_text.append(" delete  ", style=Theme.MUTED)
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
    text.append(f"Reasoning • {char_count:,} chars", style=Theme.MUTED)
    text.append("  ▸", style=Theme.MUTED)
    return text


def render_reasoning_expanded(reasoning: str) -> Panel:
    """Render expanded reasoning panel."""
    Icons, Theme = _get_ui()
    content = Markdown(reasoning) if reasoning.strip() else Text("No reasoning provided.", style=Theme.MUTED)
    return Panel(
        content,
        title=f"[{Theme.ACCENT}]▾ {Icons.BRAIN} Reasoning[/{Theme.ACCENT}]",
        title_align="left",
        border_style=Theme.BORDER,
        padding=(0, 1),
        box=ROUNDED,
    )


def render_message(content: str, is_streaming: bool = False, spinner_frame: int = 0) -> Panel:
    """Render the main message panel."""
    Icons, Theme = _get_ui()
    if is_streaming:
        # Show raw text while streaming for better performance
        text = Text(content, style=Theme.MESSAGE)
        # Blinking cursor effect using different cursor chars
        cursors = ["▏", "▎", "▍", "▌", "▋", "▊", "▉", "█", "▉", "▊", "▋", "▌", "▍", "▎"]
        cursor = cursors[spinner_frame % len(cursors)]
        text.append(cursor, style=f"bold {Theme.PRIMARY}")
        renderable: RenderableType = text
        title_icon = render_spinner(spinner_frame, style="pulse")
        border = Theme.BORDER_ACTIVE
    else:
        # Render as markdown when complete
        renderable = Markdown(content)
        title_icon = Icons.ROBOT
        border = Theme.BORDER
    
    title_prefix = "▾ " if not is_streaming else ""
    return Panel(
        renderable,
        title=f"[{Theme.PRIMARY}]{title_prefix}{title_icon} Response[/{Theme.PRIMARY}]",
        title_align="left",
        border_style=border,
        padding=(0, 1),
        box=ROUNDED,
    )


def render_prompt_prefix(model: str, is_multiline: bool = False) -> str:
    """Render the prompt prefix with elegant styling."""
    Icons, Theme = _get_ui()
    text = Text()
    
    # Use a clean arrow with subtle styling
    if is_multiline:
        text.append("  ", style=Theme.MUTED)
        text.append("┃ ", style=Theme.BORDER)
    else:
        text.append(f"{Icons.ARROW_RIGHT} ", style=f"bold {Theme.SUCCESS}")
    
    return render_to_ansi(text)


def render_continuation_prefix(line_number: int) -> str:
    """Render the continuation prefix for multiline input."""
    Icons, Theme = _get_ui()
    text = Text()
    text.append(f"   ... ", style=Theme.MUTED)
    return render_to_ansi(text)


def render_help_panel(shortcuts: list[tuple[str, str, str]]) -> str:
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
    table.add_column("Context", style=Theme.MUTED)
    
    for shortcut, description, context in shortcuts:
        context_label = "all" if context == "always" else context
        table.add_row(shortcut, description, context_label)
    
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


class Spinners:
    """Collection of spinner styles for different contexts."""
    
    # Braille dots - smooth, default for most contexts
    BRAILLE = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    
    # Bouncing dots - good for loading
    DOTS = ["⠁", "⠂", "⠄", "⡀", "⢀", "⠠", "⠐", "⠈"]
    
    # Moon phases - elegant for long waits
    MOON = ["🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘"]
    
    # Simple line - minimal
    LINE = ["⎯", "\\", "|", "/"]
    
    # Growing bar - good for progress-like activity
    BAR = ["▏", "▎", "▍", "▌", "▋", "▊", "▉", "█", "▉", "▊", "▋", "▌", "▍", "▎"]
    
    # Pulse dot - clean and modern
    PULSE = ["○", "◔", "◑", "◕", "●", "◕", "◑", "◔"]
    
    # Arc spinner - professional look
    ARC = ["◜", "◠", "◝", "◞", "◡", "◟"]


def render_spinner(frame: int, style: str = "braille") -> str:
    """Get a spinner frame.
    
    Args:
        frame: Current frame number
        style: Spinner style (braille, dots, moon, line, bar, pulse, arc)
    
    Returns:
        Single character for the spinner frame
    """
    spinners = {
        "braille": Spinners.BRAILLE,
        "dots": Spinners.DOTS,
        "moon": Spinners.MOON,
        "line": Spinners.LINE,
        "bar": Spinners.BAR,
        "pulse": Spinners.PULSE,
        "arc": Spinners.ARC,
    }
    frames = spinners.get(style, Spinners.BRAILLE)
    return frames[frame % len(frames)]


def render_welcome_banner(
    model: str,
    reasoning: str,
    version: str = "",
    width: int | None = None,
) -> str:
    """Render a stylish welcome banner for the TUI."""
    Icons, Theme = _get_ui()
    
    # ASCII art logo - compact and elegant
    logo_lines = [
        "┌─────────────────────────────────────┐",
        "│   ██████╗ ██████╗ ██████╗ ███████╗  │",
        "│  ██╔════╝██╔═══██╗██╔══██╗██╔════╝  │",
        "│  ██║     ██║   ██║██████╔╝█████╗    │",
        "│  ██║     ██║   ██║██╔═══╝ ██╔══╝    │",
        "│  ╚██████╗╚██████╔╝██║     ███████╗  │",
        "│   ╚═════╝ ╚═════╝ ╚═╝     ╚══════╝  │",
        "│          Copilot Extended           │",
        "└─────────────────────────────────────┘",
    ]
    
    elements = []
    
    # Logo with gradient effect (top to bottom: primary -> accent)
    for i, line in enumerate(logo_lines):
        # Gradient from primary to accent
        if i < 3:
            style = Theme.PRIMARY
        elif i < 6:
            style = Theme.ACCENT
        else:
            style = Theme.MUTED
        elements.append(Text(line, style=style))
        elements.append(Text("\n"))
    
    elements.append(Text("\n"))
    
    # Status info in a clean table format
    info = Text()
    info.append(f"  {Icons.ROBOT} ", style=Theme.PRIMARY)
    info.append("Model: ", style=Theme.MUTED)
    info.append(model, style=f"bold {Theme.PRIMARY}")
    info.append(f"  {Icons.BRAIN} ", style=Theme.ACCENT)
    info.append("Reasoning: ", style=Theme.MUTED)
    info.append(reasoning, style=f"bold {Theme.ACCENT}")
    if version:
        info.append(f"  v{version}", style=Theme.MUTED)
    elements.append(info)
    elements.append(Text("\n\n"))
    
    # Quick help
    help_text = Text()
    help_text.append("  ", style="")
    help_text.append("Ctrl+P", style="bold")
    help_text.append(" palette  ", style=Theme.MUTED)
    help_text.append("Ctrl+N", style="bold")
    help_text.append(" new session  ", style=Theme.MUTED)
    help_text.append("Ctrl+C", style="bold")
    help_text.append(" cancel  ", style=Theme.MUTED)
    help_text.append("Ctrl+D", style="bold")
    help_text.append(" exit", style=Theme.MUTED)
    elements.append(help_text)
    
    result = Text()
    for elem in elements:
        result.append_text(elem)
    
    return render_to_ansi(result, width=width)
