"""Beautiful CLI UI components for Copex."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from rich.box import ROUNDED
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

# ═══════════════════════════════════════════════════════════════════════════════
# Theme and Colors
# ═══════════════════════════════════════════════════════════════════════════════

class Theme:
    """Color theme for the UI."""

    # Brand colors
    PRIMARY = "cyan"
    SECONDARY = "blue"
    ACCENT = "magenta"

    # Status colors
    SUCCESS = "green"
    WARNING = "yellow"
    ERROR = "red"
    INFO = "blue"

    # Content colors
    REASONING = "dim italic"
    MESSAGE = "white"
    CODE = "bright_white"
    MUTED = "dim"

    # UI elements
    BORDER = "bright_black"
    BORDER_ACTIVE = "cyan"
    HEADER = "bold cyan"
    SUBHEADER = "bold white"


# ═══════════════════════════════════════════════════════════════════════════════
# Icons and Symbols
# ═══════════════════════════════════════════════════════════════════════════════

class Icons:
    """Unicode icons for the UI."""

    # Status
    THINKING = "◐"
    DONE = "✓"
    ERROR = "✗"
    WARNING = "⚠"
    INFO = "ℹ"

    # Actions
    TOOL = "⚡"
    FILE_READ = "📖"
    FILE_WRITE = "📝"
    FILE_CREATE = "📄"
    SEARCH = "🔍"
    TERMINAL = "💻"
    GLOBE = "🌐"

    # Navigation
    ARROW_RIGHT = "→"
    ARROW_DOWN = "↓"
    BULLET = "•"

    # Misc
    SPARKLE = "✨"
    BRAIN = "🧠"
    ROBOT = "🤖"
    LIGHTNING = "⚡"
    CLOCK = "⏱"


# ═══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════════

class ActivityType(str, Enum):
    """Types of activities to display."""
    THINKING = "thinking"
    REASONING = "reasoning"
    RESPONDING = "responding"
    TOOL_CALL = "tool_call"
    WAITING = "waiting"
    DONE = "done"
    ERROR = "error"


@dataclass
class ToolCallInfo:
    """Information about a tool call."""
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str | None = None
    status: str = "running"  # running, success, error
    duration: float | None = None

    @property
    def icon(self) -> str:
        """Get appropriate icon for the tool."""
        name_lower = self.name.lower()
        if "read" in name_lower or "view" in name_lower:
            return Icons.FILE_READ
        elif "write" in name_lower or "edit" in name_lower:
            return Icons.FILE_WRITE
        elif "create" in name_lower:
            return Icons.FILE_CREATE
        elif "search" in name_lower or "grep" in name_lower or "glob" in name_lower:
            return Icons.SEARCH
        elif "shell" in name_lower or "bash" in name_lower or "powershell" in name_lower:
            return Icons.TERMINAL
        elif "web" in name_lower or "fetch" in name_lower:
            return Icons.GLOBE
        return Icons.TOOL


@dataclass
class UIState:
    """Current state of the UI."""
    activity: ActivityType = ActivityType.WAITING
    reasoning: str = ""
    message: str = ""
    tool_calls: list[ToolCallInfo] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    model: str = ""
    retries: int = 0

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def elapsed_str(self) -> str:
        elapsed = self.elapsed
        if elapsed < 60:
            return f"{elapsed:.1f}s"
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        return f"{minutes}m {seconds:.0f}s"


# ═══════════════════════════════════════════════════════════════════════════════
# UI Components
# ═══════════════════════════════════════════════════════════════════════════════

class CopexUI:
    """Beautiful UI for Copex CLI."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self.state = UIState()
        self._live: Live | None = None
        self._spinners = ["◐", "◓", "◑", "◒"]
        self._spinner_idx = 0

    def _get_spinner(self) -> str:
        """Get current spinner frame."""
        self._spinner_idx = (self._spinner_idx + 1) % len(self._spinners)
        return self._spinners[self._spinner_idx]

    def _build_header(self) -> Text:
        """Build the header with model and status."""
        header = Text()
        header.append(f"{Icons.ROBOT} ", style=Theme.PRIMARY)
        header.append("Copex", style=Theme.HEADER)
        if self.state.model:
            header.append(f" • {self.state.model}", style=Theme.MUTED)
        header.append(f" • {self.state.elapsed_str}", style=Theme.MUTED)
        if self.state.retries > 0:
            header.append(f" • {self.state.retries} retries", style=Theme.WARNING)
        return header

    def _build_activity_indicator(self) -> Text:
        """Build the current activity indicator."""
        indicator = Text()

        if self.state.activity == ActivityType.THINKING:
            indicator.append(f" {self._get_spinner()} ", style=f"bold {Theme.PRIMARY}")
            indicator.append("Thinking...", style=Theme.PRIMARY)
        elif self.state.activity == ActivityType.REASONING:
            indicator.append(f" {Icons.BRAIN} ", style=f"bold {Theme.ACCENT}")
            indicator.append("Reasoning...", style=Theme.ACCENT)
        elif self.state.activity == ActivityType.RESPONDING:
            indicator.append(f" {self._get_spinner()} ", style=f"bold {Theme.SUCCESS}")
            indicator.append("Responding...", style=Theme.SUCCESS)
        elif self.state.activity == ActivityType.TOOL_CALL:
            indicator.append(f" {Icons.LIGHTNING} ", style=f"bold {Theme.WARNING}")
            indicator.append("Executing tools...", style=Theme.WARNING)
        elif self.state.activity == ActivityType.DONE:
            indicator.append(f" {Icons.DONE} ", style=f"bold {Theme.SUCCESS}")
            indicator.append("Complete", style=Theme.SUCCESS)
        elif self.state.activity == ActivityType.ERROR:
            indicator.append(f" {Icons.ERROR} ", style=f"bold {Theme.ERROR}")
            indicator.append("Error", style=Theme.ERROR)
        else:
            indicator.append(f" {self._get_spinner()} ", style=Theme.MUTED)
            indicator.append("Waiting...", style=Theme.MUTED)

        return indicator

    def _build_reasoning_panel(self) -> Panel | None:
        """Build the reasoning panel if there's reasoning content."""
        if not self.state.reasoning:
            return None

        # Truncate for live display
        reasoning = self.state.reasoning
        if len(reasoning) > 500:
            reasoning = reasoning[-500:] + "..."

        content = Text(reasoning, style=Theme.REASONING)
        if self.state.activity == ActivityType.REASONING:
            content.append("▌", style=f"bold {Theme.ACCENT}")

        return Panel(
            content,
            title=f"[{Theme.ACCENT}]{Icons.BRAIN} Reasoning[/{Theme.ACCENT}]",
            title_align="left",
            border_style=Theme.BORDER,
            padding=(0, 1),
        )

    def _build_tool_calls_panel(self) -> Panel | None:
        """Build the tool calls panel."""
        if not self.state.tool_calls:
            return None

        tree = Tree(f"[{Theme.WARNING}]{Icons.TOOL} Tool Calls[/{Theme.WARNING}]")

        for tool in self.state.tool_calls[-5:]:  # Show last 5
            status_style = {
                "running": Theme.WARNING,
                "success": Theme.SUCCESS,
                "error": Theme.ERROR,
            }.get(tool.status, Theme.MUTED)

            # Build tool info
            tool_text = Text()
            tool_text.append(f"{tool.icon} ", style=status_style)
            tool_text.append(tool.name, style=f"bold {status_style}")

            # Add key arguments (truncated)
            if tool.arguments:
                args_preview = self._format_args_preview(tool.arguments)
                if args_preview:
                    tool_text.append(f" {args_preview}", style=Theme.MUTED)

            if tool.duration:
                tool_text.append(f" ({tool.duration:.1f}s)", style=Theme.MUTED)

            branch = tree.add(tool_text)

            # Add result preview if available
            if tool.result and tool.status != "running":
                result_preview = tool.result[:100]
                if len(tool.result) > 100:
                    result_preview += "..."
                branch.add(Text(result_preview, style=Theme.MUTED))

        if len(self.state.tool_calls) > 5:
            tree.add(Text(f"... and {len(self.state.tool_calls) - 5} more", style=Theme.MUTED))

        return Panel(
            tree,
            border_style=Theme.BORDER,
            padding=(0, 1),
        )

    def _format_args_preview(self, args: dict[str, Any], max_len: int = 60) -> str:
        """Format arguments for preview."""
        if not args:
            return ""

        parts = []
        for key, value in args.items():
            if key in ("path", "file", "command", "pattern", "query"):
                val_str = str(value)[:40]
                if len(str(value)) > 40:
                    val_str += "..."
                parts.append(f"{key}={val_str}")

        result = " ".join(parts)
        if len(result) > max_len:
            result = result[:max_len] + "..."
        return result

    def _build_message_panel(self) -> Panel | None:
        """Build the message panel."""
        if not self.state.message:
            return None

        # For live display, show raw text with cursor
        content = Text(self.state.message)
        if self.state.activity == ActivityType.RESPONDING:
            content.append("▌", style=f"bold {Theme.PRIMARY}")

        return Panel(
            content,
            title=f"[{Theme.PRIMARY}]{Icons.ROBOT} Response[/{Theme.PRIMARY}]",
            title_align="left",
            border_style=Theme.BORDER_ACTIVE if self.state.activity == ActivityType.RESPONDING else Theme.BORDER,
            padding=(0, 1),
        )

    def build_live_display(self) -> Group:
        """Build the complete live display."""
        elements = []

        # Header
        elements.append(self._build_header())
        elements.append(Text())  # Spacer

        # Activity indicator
        elements.append(self._build_activity_indicator())
        elements.append(Text())  # Spacer

        # Reasoning (if any)
        reasoning_panel = self._build_reasoning_panel()
        if reasoning_panel:
            elements.append(reasoning_panel)
            elements.append(Text())

        # Tool calls (if any)
        tool_panel = self._build_tool_calls_panel()
        if tool_panel:
            elements.append(tool_panel)
            elements.append(Text())

        # Message (if any)
        message_panel = self._build_message_panel()
        if message_panel:
            elements.append(message_panel)

        return Group(*elements)

    def build_final_display(self) -> Group:
        """Build the final formatted display after streaming completes."""
        elements = []

        # Reasoning panel (collapsed/summary)
        if self.state.reasoning:
            elements.append(Panel(
                Markdown(self.state.reasoning),
                title=f"[{Theme.ACCENT}]{Icons.BRAIN} Reasoning[/{Theme.ACCENT}]",
                title_align="left",
                border_style=Theme.BORDER,
                padding=(0, 1),
            ))
            elements.append(Text())

        # Tool calls summary
        if self.state.tool_calls:
            successful = sum(1 for t in self.state.tool_calls if t.status == "success")
            failed = sum(1 for t in self.state.tool_calls if t.status == "error")

            summary = Text()
            summary.append(f"{Icons.TOOL} ", style=Theme.WARNING)
            summary.append(f"{len(self.state.tool_calls)} tool calls", style=Theme.WARNING)
            if successful:
                summary.append(f" • {Icons.DONE} {successful} succeeded", style=Theme.SUCCESS)
            if failed:
                summary.append(f" • {Icons.ERROR} {failed} failed", style=Theme.ERROR)

            elements.append(summary)
            elements.append(Text())

        # Main response with markdown
        if self.state.message:
            elements.append(Panel(
                Markdown(self.state.message),
                title=f"[{Theme.PRIMARY}]{Icons.ROBOT} Response[/{Theme.PRIMARY}]",
                title_align="left",
                border_style=Theme.BORDER_ACTIVE,
                padding=(0, 1),
                box=ROUNDED,
            ))

        # Footer with stats
        footer = Text()
        footer.append(f"\n{Icons.DONE} ", style=Theme.SUCCESS)
        footer.append(f"Completed in {self.state.elapsed_str}", style=Theme.MUTED)
        if self.state.retries > 0:
            footer.append(f" • {self.state.retries} retries", style=Theme.WARNING)
        elements.append(footer)

        return Group(*elements)

    # ═══════════════════════════════════════════════════════════════════════════
    # Public Methods
    # ═══════════════════════════════════════════════════════════════════════════

    def reset(self, model: str = "") -> None:
        """Reset state for a new interaction."""
        self.state = UIState(model=model)

    def set_activity(self, activity: ActivityType) -> None:
        """Set the current activity."""
        self.state.activity = activity

    def add_reasoning(self, delta: str) -> None:
        """Add reasoning content."""
        self.state.reasoning += delta
        if self.state.activity != ActivityType.REASONING:
            self.state.activity = ActivityType.REASONING

    def add_message(self, delta: str) -> None:
        """Add message content."""
        self.state.message += delta
        if self.state.activity != ActivityType.RESPONDING:
            self.state.activity = ActivityType.RESPONDING

    def add_tool_call(self, tool: ToolCallInfo) -> None:
        """Add a tool call."""
        self.state.tool_calls.append(tool)
        self.state.activity = ActivityType.TOOL_CALL

    def update_tool_call(self, name: str, status: str, result: str | None = None, duration: float | None = None) -> None:
        """Update a tool call status."""
        for tool in reversed(self.state.tool_calls):
            if tool.name == name and tool.status == "running":
                tool.status = status
                tool.result = result
                tool.duration = duration
                break

    def increment_retries(self) -> None:
        """Increment retry count."""
        self.state.retries += 1

    def set_final_content(self, message: str, reasoning: str | None = None) -> None:
        """Set final content."""
        if message:
            self.state.message = message
        if reasoning:
            self.state.reasoning = reasoning
        self.state.activity = ActivityType.DONE


# ═══════════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════════

def print_welcome(console: Console, model: str, reasoning: str) -> None:
    """Print the welcome banner."""
    console.print()
    console.print(Panel(
        Text.from_markup(
            f"[{Theme.HEADER}]{Icons.ROBOT} Copex[/{Theme.HEADER}] "
            f"[{Theme.MUTED}]- Copilot Extended[/{Theme.MUTED}]\n\n"
            f"[{Theme.MUTED}]Model:[/{Theme.MUTED}] [{Theme.PRIMARY}]{model}[/{Theme.PRIMARY}]\n"
            f"[{Theme.MUTED}]Reasoning:[/{Theme.MUTED}] [{Theme.PRIMARY}]{reasoning}[/{Theme.PRIMARY}]\n\n"
            f"[{Theme.MUTED}]Type [bold]exit[/bold] to quit, [bold]new[/bold] for fresh session[/{Theme.MUTED}]\n"
            f"[{Theme.MUTED}]Press [bold]Shift+Enter[/bold] for newline[/{Theme.MUTED}]"
        ),
        border_style=Theme.BORDER_ACTIVE,
        box=ROUNDED,
        padding=(0, 2),
    ))
    console.print()


def print_user_prompt(console: Console, prompt: str) -> None:
    """Print the user's prompt."""
    console.print()
    console.print(Text("❯ ", style=f"bold {Theme.SUCCESS}"), end="")

    # Truncate long prompts for display
    if len(prompt) > 200:
        display_prompt = prompt[:200] + "..."
    else:
        display_prompt = prompt
    console.print(Text(display_prompt, style="bold"))
    console.print()


def print_error(console: Console, error: str) -> None:
    """Print an error message."""
    console.print(Panel(
        Text(f"{Icons.ERROR} {error}", style=Theme.ERROR),
        border_style=Theme.ERROR,
        title="Error",
        title_align="left",
    ))


def print_retry(console: Console, attempt: int, max_attempts: int, error: str) -> None:
    """Print a retry notification."""
    console.print(Text(
        f" {Icons.WARNING} Retry {attempt}/{max_attempts}: {error[:50]}...",
        style=Theme.WARNING,
    ))


def print_tool_call(console: Console, name: str, args: dict[str, Any] | None = None) -> None:
    """Print a tool call notification."""
    tool = ToolCallInfo(name=name, arguments=args or {})

    text = Text()
    text.append(f" {tool.icon} ", style=Theme.WARNING)
    text.append(name, style=f"bold {Theme.WARNING}")

    if args:
        preview = ""
        if "path" in args:
            preview = f" path={args['path']}"
        elif "command" in args:
            cmd = str(args['command'])[:40]
            preview = f" cmd={cmd}..."
        elif "pattern" in args:
            preview = f" pattern={args['pattern']}"
        if preview:
            text.append(preview, style=Theme.MUTED)

    console.print(text)


def print_tool_result(console: Console, name: str, success: bool, duration: float | None = None) -> None:
    """Print a tool result notification."""
    icon = Icons.DONE if success else Icons.ERROR
    style = Theme.SUCCESS if success else Theme.ERROR

    text = Text()
    text.append(f"   {icon} ", style=style)
    text.append(name, style=f"bold {style}")
    if duration:
        text.append(f" ({duration:.1f}s)", style=Theme.MUTED)

    console.print(text)
