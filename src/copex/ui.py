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
from rich.table import Table
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


THEME_PRESETS = {
    "default": {
        "PRIMARY": "cyan",
        "SECONDARY": "blue",
        "ACCENT": "magenta",
        "SUCCESS": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "INFO": "blue",
        "REASONING": "dim italic",
        "MESSAGE": "white",
        "CODE": "bright_white",
        "MUTED": "dim",
        "BORDER": "bright_black",
        "BORDER_ACTIVE": "cyan",
        "HEADER": "bold cyan",
        "SUBHEADER": "bold white",
    },
    "midnight": {
        "PRIMARY": "bright_cyan",
        "SECONDARY": "bright_blue",
        "ACCENT": "bright_magenta",
        "SUCCESS": "bright_green",
        "WARNING": "bright_yellow",
        "ERROR": "bright_red",
        "INFO": "bright_blue",
        "REASONING": "dim italic",
        "MESSAGE": "white",
        "CODE": "bright_white",
        "MUTED": "grey70",
        "BORDER": "grey39",
        "BORDER_ACTIVE": "bright_cyan",
        "HEADER": "bold bright_cyan",
        "SUBHEADER": "bold bright_white",
    },
    "mono": {
        "PRIMARY": "white",
        "SECONDARY": "white",
        "ACCENT": "white",
        "SUCCESS": "white",
        "WARNING": "white",
        "ERROR": "white",
        "INFO": "white",
        "REASONING": "dim",
        "MESSAGE": "white",
        "CODE": "white",
        "MUTED": "dim",
        "BORDER": "grey66",
        "BORDER_ACTIVE": "white",
        "HEADER": "bold white",
        "SUBHEADER": "bold white",
    },
    "sunset": {
        "PRIMARY": "bright_yellow",
        "SECONDARY": "bright_red",
        "ACCENT": "bright_magenta",
        "SUCCESS": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "INFO": "bright_yellow",
        "REASONING": "dim italic",
        "MESSAGE": "white",
        "CODE": "bright_white",
        "MUTED": "grey70",
        "BORDER": "grey39",
        "BORDER_ACTIVE": "bright_yellow",
        "HEADER": "bold bright_yellow",
        "SUBHEADER": "bold bright_white",
    },
    "tokyo": {
        # Tokyo Night inspired palette - modern and easy on the eyes
        "PRIMARY": "#7aa2f7",  # Soft blue
        "SECONDARY": "#9ece6a",  # Green
        "ACCENT": "#bb9af7",  # Purple
        "SUCCESS": "#9ece6a",  # Green
        "WARNING": "#e0af68",  # Orange/Yellow
        "ERROR": "#f7768e",  # Red/Pink
        "INFO": "#7dcfff",  # Light blue
        "REASONING": "italic #565f89",  # Muted italic
        "MESSAGE": "#c0caf5",  # Light text
        "CODE": "#a9b1d6",  # Code text
        "MUTED": "#565f89",  # Muted gray
        "BORDER": "#3b4261",  # Dark border
        "BORDER_ACTIVE": "#7aa2f7",  # Active border (primary)
        "HEADER": "bold #7aa2f7",  # Bold primary
        "SUBHEADER": "bold #c0caf5",  # Bold text
    },
}


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

    tool_id: str = field(default_factory=lambda: "")
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str | None = None
    status: str = "running"  # running, success, error
    duration: float | None = None
    started_at: float = field(default_factory=time.time)

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

    @property
    def elapsed(self) -> float:
        if self.duration is not None:
            return self.duration
        return time.time() - self.started_at


@dataclass
class HistoryEntry:
    """A single conversation turn."""

    role: str  # "user" or "assistant"
    content: str
    reasoning: str | None = None
    tool_calls: list[ToolCallInfo] = field(default_factory=list)


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
    last_update: float = field(default_factory=time.time)
    history: list[HistoryEntry] = field(default_factory=list)
    # Token usage tracking
    input_tokens: int | None = None
    output_tokens: int | None = None

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

    @property
    def idle(self) -> float:
        return time.time() - self.last_update

    @property
    def idle_str(self) -> str:
        idle = self.idle
        if idle < 60:
            return f"{idle:.1f}s"
        minutes = int(idle // 60)
        seconds = idle % 60
        return f"{minutes}m {seconds:.0f}s"


# ═══════════════════════════════════════════════════════════════════════════════
# UI Components
# ═══════════════════════════════════════════════════════════════════════════════


class CopexUI:
    """Beautiful UI for Copex CLI."""

    def __init__(
        self,
        console: Console | None = None,
        *,
        theme: str = "default",
        density: str = "extended",
        show_all_tools: bool = False,
    ):
        self.console = console or Console()
        self.set_theme(theme)
        self.density = density
        self.state = UIState()
        self._live: Live | None = None
        self._spinners = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._spinner_idx = 0
        self._last_frame_at = 0.0
        self._dot_frames = [".", "..", "..."]
        self.show_all_tools = show_all_tools
        self._max_live_message_chars = 2000 if density == "extended" else 900
        self._max_live_reasoning_chars = 800 if density == "extended" else 320

    def _get_spinner(self) -> str:
        """Get current spinner frame."""
        return self._spinners[self._spinner_idx]

    def _get_dots(self) -> str:
        """Get current dot animation frame."""
        return self._dot_frames[self._spinner_idx % len(self._dot_frames)]

    def _advance_frame(self) -> None:
        """Advance animation frame."""
        now = time.time()
        if now - self._last_frame_at < 0.08:
            return
        self._last_frame_at = now
        self._spinner_idx = (self._spinner_idx + 1) % len(self._spinners)

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
        """Build the current activity indicator with fixed width to prevent shifting."""
        indicator = Text()
        dots = self._get_dots()
        spinner = self._get_spinner()

        # Fixed width for activity text to prevent elapsed time from shifting
        # "Executing tools" is longest at 15 chars + "..." = 18 chars
        activity_width = 18

        if self.state.activity == ActivityType.THINKING:
            indicator.append(f" {spinner} ", style=f"bold {Theme.PRIMARY}")
            label = f"Thinking{dots}"
            indicator.append(label.ljust(activity_width), style=Theme.PRIMARY)
        elif self.state.activity == ActivityType.REASONING:
            indicator.append(f" {spinner} ", style=f"bold {Theme.ACCENT}")
            label = f"Reasoning{dots}"
            indicator.append(label.ljust(activity_width), style=Theme.ACCENT)
        elif self.state.activity == ActivityType.RESPONDING:
            indicator.append(f" {spinner} ", style=f"bold {Theme.SUCCESS}")
            label = f"Responding{dots}"
            indicator.append(label.ljust(activity_width), style=Theme.SUCCESS)
        elif self.state.activity == ActivityType.TOOL_CALL:
            indicator.append(f" {spinner} ", style=f"bold {Theme.WARNING}")
            label = f"Executing tools{dots}"
            indicator.append(label.ljust(activity_width), style=Theme.WARNING)
        elif self.state.activity == ActivityType.DONE:
            indicator.append(f" {Icons.DONE} ", style=f"bold {Theme.SUCCESS}")
            indicator.append("Complete".ljust(activity_width), style=Theme.SUCCESS)
        elif self.state.activity == ActivityType.ERROR:
            indicator.append(f" {Icons.ERROR} ", style=f"bold {Theme.ERROR}")
            indicator.append("Error".ljust(activity_width), style=Theme.ERROR)
        else:
            indicator.append(f" {spinner} ", style=Theme.MUTED)
            label = f"Waiting{dots}"
            indicator.append(label.ljust(activity_width), style=Theme.MUTED)

        return indicator

    def _build_reasoning_panel(self) -> Panel | None:
        """Build the reasoning panel if there's reasoning content."""
        if not self.state.reasoning:
            return None

        # Truncate for live display
        reasoning = self.state.reasoning
        if len(reasoning) > self._max_live_reasoning_chars:
            reasoning = "..." + reasoning[-self._max_live_reasoning_chars :]

        content = Text(reasoning, style=Theme.REASONING)
        if self.state.activity == ActivityType.REASONING:
            content.append("▌", style=f"bold {Theme.ACCENT}")

        return Panel(
            content,
            title=f"[{Theme.ACCENT}]{Icons.BRAIN} Reasoning[/{Theme.ACCENT}]",
            title_align="left",
            border_style=Theme.BORDER_ACTIVE
            if self.state.activity == ActivityType.REASONING
            else Theme.BORDER,
            padding=(0, 1),
            box=ROUNDED,
        )

    def _build_tool_calls_panel(self) -> Panel | None:
        """Build the tool calls panel."""
        if not self.state.tool_calls:
            return None

        spinner = self._get_spinner()
        running = sum(1 for t in self.state.tool_calls if t.status == "running")
        successful = sum(1 for t in self.state.tool_calls if t.status == "success")
        failed = sum(1 for t in self.state.tool_calls if t.status == "error")
        title_parts = [f"{Icons.TOOL} Tools"]
        if running:
            title_parts.append(f"{running} running")
        if successful:
            title_parts.append(f"{successful} ok")
        if failed:
            title_parts.append(f"{failed} failed")
        title = f"[{Theme.WARNING}]{' • '.join(title_parts)}[/{Theme.WARNING}]"

        tree = Tree(f"[{Theme.WARNING}]{Icons.TOOL} Tool Calls[/{Theme.WARNING}]")

        max_tools = 5 if self.density == "extended" else 3
        tools_to_show = (
            self.state.tool_calls if self.show_all_tools else self.state.tool_calls[-max_tools:]
        )
        for tool in tools_to_show:
            status_style = {
                "running": Theme.WARNING,
                "success": Theme.SUCCESS,
                "error": Theme.ERROR,
            }.get(tool.status, Theme.MUTED)

            # Build tool info
            tool_text = Text()
            status_icon = (
                spinner
                if tool.status == "running"
                else (Icons.DONE if tool.status == "success" else Icons.ERROR)
            )
            tool_text.append(f"{status_icon} ", style=status_style)
            tool_text.append(f"{tool.icon} ", style=status_style)
            tool_text.append(tool.name, style=f"bold {status_style}")

            # Add key arguments (truncated)
            if tool.arguments and self.density == "extended":
                args_preview = self._format_args_preview(tool.arguments)
                if args_preview:
                    tool_text.append(f" {args_preview}", style=Theme.MUTED)

            if tool.status == "running":
                tool_text.append(f" ({tool.elapsed:5.1f}s)", style=Theme.MUTED)
            elif tool.duration:
                tool_text.append(f" ({tool.duration:5.1f}s)", style=Theme.MUTED)

            branch = tree.add(tool_text)

            # Add result preview if available
            if tool.result and tool.status != "running":
                result_preview = tool.result[:100]
                if len(tool.result) > 100:
                    result_preview += "..."
                branch.add(Text(result_preview, style=Theme.MUTED))

        if len(self.state.tool_calls) > max_tools:
            if self.show_all_tools:
                tree.add(Text("Showing all tools (use /tools to collapse)", style=Theme.MUTED))
            else:
                tree.add(
                    Text(
                        f"... and {len(self.state.tool_calls) - max_tools} more (use /tools to expand)",
                        style=Theme.MUTED,
                    )
                )

        border_style = Theme.BORDER
        if self.state.activity == ActivityType.TOOL_CALL or running:
            border_style = Theme.BORDER_ACTIVE
        if failed:
            border_style = Theme.ERROR

        return Panel(
            tree,
            title=title,
            title_align="left",
            border_style=border_style,
            padding=(0, 1),
            box=ROUNDED,
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

        # Show full message content (no truncation) so box expands with content
        content = Text(self.state.message, style=Theme.MESSAGE)
        if self.state.activity == ActivityType.RESPONDING:
            content.append("▌", style=f"bold {Theme.PRIMARY}")

        return Panel(
            content,
            title=f"[{Theme.PRIMARY}]{Icons.ROBOT} Response[/{Theme.PRIMARY}]",
            title_align="left",
            border_style=Theme.BORDER_ACTIVE
            if self.state.activity == ActivityType.RESPONDING
            else Theme.BORDER,
            padding=(0, 1),
            box=ROUNDED,
        )

    def _build_status_panel(self) -> Panel:
        """Build a status panel with live progress details."""
        activity = self._build_activity_indicator()
        message_chars = len(self.state.message)
        reasoning_chars = len(self.state.reasoning)
        running_tools = sum(1 for t in self.state.tool_calls if t.status == "running")
        successful_tools = sum(1 for t in self.state.tool_calls if t.status == "success")
        failed_tools = sum(1 for t in self.state.tool_calls if t.status == "error")

        message_text = Text()
        message_text.append(f"{Icons.ROBOT} ", style=Theme.PRIMARY)
        message_text.append(f"{message_chars} chars", style=Theme.PRIMARY)

        reasoning_text = Text()
        reasoning_text.append(f"{Icons.BRAIN} ", style=Theme.ACCENT)
        reasoning_text.append(f"{reasoning_chars} chars", style=Theme.ACCENT)

        tools_text = Text()
        tools_text.append(f"{Icons.TOOL} ", style=Theme.WARNING)
        if not self.state.tool_calls:
            tools_text.append("no tools", style=Theme.MUTED)
        else:
            parts = []
            if running_tools:
                parts.append(f"{running_tools} running")
            if successful_tools:
                parts.append(f"{successful_tools} ok")
            if failed_tools:
                parts.append(f"{failed_tools} failed")
            tools_text.append(
                " • ".join(parts), style=Theme.WARNING if not failed_tools else Theme.ERROR
            )

        elapsed_text = Text()
        elapsed_text.append(f"{Icons.CLOCK} ", style=Theme.MUTED)
        elapsed_text.append(f"{self.state.elapsed_str} elapsed", style=Theme.MUTED)

        updated_text = Text()
        updated_text.append(f"{Icons.SPARKLE} ", style=Theme.MUTED)
        updated_text.append(f"updated {self.state.idle_str} ago", style=Theme.MUTED)

        model_text = Text()
        if self.state.model:
            model_text.append(f"{Icons.ROBOT} ", style=Theme.PRIMARY)
            model_text.append(self.state.model, style=Theme.MUTED)
        else:
            model_text.append(f"{Icons.ROBOT} default model", style=Theme.MUTED)

        retry_text = Text()
        if self.state.retries:
            retry_text.append(f"{Icons.WARNING} ", style=Theme.WARNING)
            retry_text.append(f"{self.state.retries} retries", style=Theme.WARNING)
        else:
            retry_text.append(f"{Icons.DONE} no retries", style=Theme.MUTED)

        grid = Table.grid(expand=True)
        grid.add_column(justify="left")
        if self.density == "extended":
            grid.add_column(justify="center")
            grid.add_column(justify="right")
            grid.add_row(activity, elapsed_text, updated_text)
            grid.add_row(message_text, reasoning_text, tools_text)
            grid.add_row(model_text, Text(), retry_text)
        else:
            grid.add_column(justify="right")
            grid.add_row(activity, elapsed_text)
            grid.add_row(message_text, tools_text)

        if self.state.activity == ActivityType.ERROR:
            border_style = Theme.ERROR
        elif self.state.activity == ActivityType.DONE:
            border_style = Theme.SUCCESS
        elif self.state.activity == ActivityType.WAITING:
            border_style = Theme.BORDER
        else:
            border_style = Theme.BORDER_ACTIVE

        title = f"[{Theme.PRIMARY}]{Icons.ROBOT} Copex[/{Theme.PRIMARY}]"
        if self.state.model:
            title += f" [{Theme.MUTED}]• {self.state.model}[/{Theme.MUTED}]"

        content = Group(grid, Text()) if self.density == "extended" else grid

        return Panel(
            content,
            title=title,
            title_align="left",
            border_style=border_style,
            padding=(0, 1),
            box=ROUNDED,
        )

    def build_live_display(self) -> Group:
        """Build the complete live display."""
        self._advance_frame()
        elements = []

        # Status panel
        elements.append(self._build_status_panel())
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

    def _build_history_panel(self) -> Panel | None:
        """Build the conversation history panel."""
        if not self.state.history:
            return None

        elements = []
        for i, entry in enumerate(self.state.history):
            if entry.role == "user":
                # User message
                user_text = Text()
                user_text.append("❯ ", style=f"bold {Theme.SUCCESS}")
                # Truncate long user messages
                content = entry.content
                if len(content) > 200:
                    content = content[:200] + "..."
                user_text.append(content, style="bold")
                elements.append(user_text)
                elements.append(Text())  # Spacer
            else:
                # Assistant message
                if entry.reasoning and self.density == "extended":
                    elements.append(
                        Panel(
                            Markdown(entry.reasoning),
                            title=f"[{Theme.ACCENT}]{Icons.BRAIN} Reasoning[/{Theme.ACCENT}]",
                            title_align="left",
                            border_style=Theme.BORDER,
                            padding=(0, 1),
                            box=ROUNDED,
                        )
                    )
                    elements.append(Text())

                elements.append(
                    Panel(
                        Markdown(entry.content),
                        title=f"[{Theme.PRIMARY}]{Icons.ROBOT} Response[/{Theme.PRIMARY}]",
                        title_align="left",
                        border_style=Theme.BORDER,
                        padding=(0, 1),
                        box=ROUNDED,
                    )
                )
                elements.append(Text())  # Spacer between turns

        if not elements:
            return None

        return Panel(
            Group(*elements),
            title=f"[{Theme.MUTED}]Conversation History ({len([e for e in self.state.history if e.role == 'user'])} turns)[/{Theme.MUTED}]",
            title_align="left",
            border_style=Theme.BORDER,
            padding=(0, 1),
            box=ROUNDED,
        )

    def build_final_display(self) -> Group:
        """Build the final formatted display after streaming completes."""
        elements = []

        # Reasoning panel (collapsed/summary)
        if self.state.reasoning and self.density == "extended":
            elements.append(
                Panel(
                    Markdown(self.state.reasoning),
                    title=f"[{Theme.ACCENT}]{Icons.BRAIN} Reasoning[/{Theme.ACCENT}]",
                    title_align="left",
                    border_style=Theme.BORDER,
                    padding=(0, 1),
                    box=ROUNDED,
                )
            )
            elements.append(Text())

        # Main response with markdown
        if self.state.message:
            elements.append(
                Panel(
                    Markdown(self.state.message),
                    title=f"[{Theme.PRIMARY}]{Icons.ROBOT} Response[/{Theme.PRIMARY}]",
                    title_align="left",
                    border_style=Theme.BORDER_ACTIVE,
                    padding=(0, 1),
                    box=ROUNDED,
                )
            )

        # Summary panel
        elements.append(self._build_summary_panel())

        return Group(*elements)

    # ═══════════════════════════════════════════════════════════════════════════
    # Public Methods
    # ═══════════════════════════════════════════════════════════════════════════

    def reset(self, model: str = "", preserve_history: bool = False) -> None:
        """Reset state for a new interaction."""
        old_history = self.state.history if preserve_history else []
        self.state = UIState(model=model, history=old_history)

    def set_activity(self, activity: ActivityType) -> None:
        """Set the current activity."""
        self.state.activity = activity
        self._touch()

    def add_reasoning(self, delta: str) -> None:
        """Add reasoning content."""
        self.state.reasoning += delta
        if self.state.activity != ActivityType.REASONING:
            self.state.activity = ActivityType.REASONING
        self._touch()

    def add_message(self, delta: str) -> None:
        """Add message content."""
        self.state.message += delta
        if self.state.activity != ActivityType.RESPONDING:
            self.state.activity = ActivityType.RESPONDING
        self._touch()

    def add_tool_call(self, tool: ToolCallInfo) -> None:
        """Add a tool call."""
        self.state.tool_calls.append(tool)
        self.state.activity = ActivityType.TOOL_CALL
        self._touch()

    def update_tool_call(
        self,
        tool_id: str | None,
        name: str,
        status: str,
        result: str | None = None,
        duration: float | None = None,
    ) -> None:
        """Update a tool call status by tool_id (preferred) or by name as fallback."""
        updated = False
        if tool_id:
            for tool in reversed(self.state.tool_calls):
                if getattr(tool, "tool_id", None) == tool_id and tool.status == "running":
                    tool.status = status
                    tool.result = result
                    tool.duration = duration
                    updated = True
                    break
        if not updated:
            for tool in reversed(self.state.tool_calls):
                if tool.name == name and tool.status == "running":
                    tool.status = status
                    tool.result = result
                    tool.duration = duration
                    break
        if self.state.activity == ActivityType.TOOL_CALL:
            running_tools = any(tool.status == "running" for tool in self.state.tool_calls)
            if not running_tools:
                self.state.activity = ActivityType.THINKING
        self._touch()

    def increment_retries(self) -> None:
        """Increment retry count."""
        self.state.retries += 1
        self._touch()

    def set_final_content(self, message: str, reasoning: str | None = None) -> None:
        """Set final content."""
        if message:
            self.state.message = message
        if reasoning:
            self.state.reasoning = reasoning
        self.state.activity = ActivityType.DONE
        self._touch()

    def add_user_message(self, content: str) -> None:
        """Add a user message to history."""
        self.state.history.append(HistoryEntry(role="user", content=content))
        self._touch()

    def finalize_assistant_response(self) -> None:
        """Finalize the current assistant response and add to history."""
        if self.state.message:
            self.state.history.append(
                HistoryEntry(
                    role="assistant",
                    content=self.state.message,
                    reasoning=self.state.reasoning if self.state.reasoning else None,
                    tool_calls=list(self.state.tool_calls),
                )
            )
        self._touch()

    def set_usage(self, input_tokens: int | None, output_tokens: int | None) -> None:
        """Set token usage counts."""
        self.state.input_tokens = input_tokens
        self.state.output_tokens = output_tokens
        self._touch()

    def _touch(self) -> None:
        """Update last activity timestamp."""
        self.state.last_update = time.time()

    def _build_summary_panel(self) -> Panel:
        """Build a summary panel for completed output."""
        summary = Table.grid(expand=True)
        summary.add_column(justify="left")
        summary.add_column(justify="right")

        # First row: timing and retries
        elapsed_text = Text()
        elapsed_text.append(f"{Icons.CLOCK} ", style=Theme.MUTED)
        elapsed_text.append(f"{self.state.elapsed_str}", style=Theme.MUTED)

        retry_text = Text()
        if self.state.retries:
            retry_text.append(f"{Icons.WARNING} ", style=Theme.WARNING)
            retry_text.append(f"{self.state.retries} retries", style=Theme.WARNING)
        else:
            retry_text.append(f"{Icons.DONE} no retries", style=Theme.MUTED)

        summary.add_row(elapsed_text, retry_text)

        # Second row: token counts (if available)
        if self.state.input_tokens is not None or self.state.output_tokens is not None:
            token_left = Text()
            token_left.append("📊 ", style=Theme.INFO)
            token_left.append("Tokens: ", style=Theme.MUTED)
            inp = self.state.input_tokens or 0
            out = self.state.output_tokens or 0
            token_left.append(f"{inp:,} in", style=Theme.INFO)
            token_left.append(" / ", style=Theme.MUTED)
            token_left.append(f"{out:,} out", style=Theme.INFO)

            total_text = Text()
            total_text.append(f"Total: {inp + out:,}", style=Theme.MUTED)
            summary.add_row(token_left, total_text)

        # Third row: tool calls (if any)
        if self.state.tool_calls:
            successful = sum(1 for t in self.state.tool_calls if t.status == "success")
            failed = sum(1 for t in self.state.tool_calls if t.status == "error")
            tool_left = Text()
            tool_left.append(f"{Icons.TOOL} ", style=Theme.WARNING)
            tool_left.append(f"{len(self.state.tool_calls)} tool calls", style=Theme.WARNING)

            tool_right = Text()
            if successful:
                tool_right.append(f"{Icons.DONE} {successful} ok", style=Theme.SUCCESS)
            if failed:
                if tool_right:
                    tool_right.append(" • ", style=Theme.MUTED)
                tool_right.append(f"{Icons.ERROR} {failed} failed", style=Theme.ERROR)
            summary.add_row(tool_left, tool_right)

        return Panel(
            summary,
            title=f"[{Theme.SUCCESS}]{Icons.DONE} Summary[/{Theme.SUCCESS}]",
            title_align="left",
            border_style=Theme.BORDER_ACTIVE,
            padding=(0, 1),
            box=ROUNDED,
        )

    def _build_progress_bar(self, width: int = 28) -> Text:
        """Build a smooth animated progress bar."""
        if self.density == "compact":
            width = min(20, width)
        if width < 10:
            width = 10
        pos = (self._spinner_idx // 2) % width
        trail = 1
        bar = ["░"] * width
        for offset in range(trail):
            idx = (pos - offset) % width
            bar[idx] = "█"

        if self.state.activity == ActivityType.ERROR:
            color = Theme.ERROR
        elif self.state.activity == ActivityType.DONE:
            color = Theme.SUCCESS
        elif self.state.activity == ActivityType.TOOL_CALL:
            color = Theme.WARNING
        elif self.state.activity == ActivityType.REASONING:
            color = Theme.ACCENT
        elif self.state.activity == ActivityType.RESPONDING:
            color = Theme.SUCCESS
        else:
            color = Theme.PRIMARY

        bar_text = Text()
        bar_text.append("Progress ", style=Theme.MUTED)
        bar_text.append("[" + "".join(bar) + "]", style=color)
        return bar_text

    def set_theme(self, theme: str) -> None:
        """Apply a theme preset."""
        apply_theme(theme)


def apply_theme(theme: str) -> None:
    """Apply a theme preset globally."""
    palette = THEME_PRESETS.get(theme, THEME_PRESETS["default"])
    for key, value in palette.items():
        setattr(Theme, key, value)


# ═══════════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════════


def print_welcome(
    console: Console,
    model: str,
    reasoning: str,
    theme: str | None = None,
    density: str | None = None,
) -> None:
    """Print the welcome banner."""
    console.print()
    console.print(
        Panel(
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
        )
    )
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
    console.print(
        Panel(
            Text(f"{Icons.ERROR} {error}", style=Theme.ERROR),
            border_style=Theme.ERROR,
            title="Error",
            title_align="left",
        )
    )


def print_retry(console: Console, attempt: int, max_attempts: int, error: str) -> None:
    """Print a retry notification."""
    console.print(
        Text(
            f" {Icons.WARNING} Retry {attempt}/{max_attempts}: {error[:50]}...",
            style=Theme.WARNING,
        )
    )


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
            cmd = str(args["command"])[:40]
            preview = f" cmd={cmd}..."
        elif "pattern" in args:
            preview = f" pattern={args['pattern']}"
        if preview:
            text.append(preview, style=Theme.MUTED)

    console.print(text)


def print_tool_result(
    console: Console, name: str, success: bool, duration: float | None = None
) -> None:
    """Print a tool result notification."""
    icon = Icons.DONE if success else Icons.ERROR
    style = Theme.SUCCESS if success else Theme.ERROR

    text = Text()
    text.append(f"   {icon} ", style=style)
    text.append(name, style=f"bold {style}")
    if duration:
        text.append(f" ({duration:.1f}s)", style=Theme.MUTED)

    console.print(text)


# ═══════════════════════════════════════════════════════════════════════════════
# Progress Components for Ralph and Plan
# ═══════════════════════════════════════════════════════════════════════════════


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours}h {mins}m"


def build_progress_bar(
    progress: float,
    width: int = 30,
    filled_char: str = "━",
    empty_char: str = "░",
    style_filled: str | None = None,
    style_empty: str | None = None,
) -> Text:
    """
    Build a progress bar as a Rich Text object.

    Args:
        progress: Progress fraction (0.0 to 1.0)
        width: Width of the bar in characters
        filled_char: Character for filled portion
        empty_char: Character for empty portion
        style_filled: Style for filled portion
        style_empty: Style for empty portion

    Returns:
        Rich Text object with styled progress bar
    """
    progress = max(0.0, min(1.0, progress))
    filled = int(width * progress)
    empty = width - filled

    bar = Text()
    bar.append(filled_char * filled, style=style_filled or Theme.PRIMARY)
    bar.append(empty_char * empty, style=style_empty or Theme.MUTED)
    return bar


class RalphUI:
    """
    Beautiful UI for Ralph Wiggum loops.

    Displays iteration progress, completion promise status, and timing information.
    """

    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self._spinners = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._spinner_idx = 0
        self._start_time = time.time()

    def _get_spinner(self) -> str:
        self._spinner_idx = (self._spinner_idx + 1) % len(self._spinners)
        return self._spinners[self._spinner_idx]

    def print_header(
        self,
        prompt: str,
        max_iterations: int | None,
        completion_promise: str | None,
        model: str = "",
        reasoning: str = "",
    ) -> None:
        """Print the Ralph loop header."""
        self._start_time = time.time()

        elements = []

        # Title
        title_text = Text()
        title_text.append("🔄 ", style=Theme.WARNING)
        title_text.append("Ralph Wiggum Loop", style=f"bold {Theme.WARNING}")
        elements.append(title_text)
        elements.append(Text())

        # Config info
        info_table = Table.grid(padding=(0, 2))
        info_table.add_column(style=Theme.MUTED)
        info_table.add_column(style=Theme.MESSAGE)

        if model:
            info_table.add_row(f"{Icons.ROBOT} Model:", model)
        if reasoning:
            info_table.add_row(f"{Icons.BRAIN} Reasoning:", reasoning)
        if max_iterations:
            info_table.add_row("📊 Max iterations:", str(max_iterations))
        if completion_promise:
            info_table.add_row("🎯 Completion promise:", f'"{completion_promise}"')

        elements.append(info_table)
        elements.append(Text())

        # Truncated prompt preview
        prompt_preview = prompt[:150] + "..." if len(prompt) > 150 else prompt
        elements.append(Text(f"📝 Task: {prompt_preview}", style=Theme.MUTED))

        self.console.print(
            Panel(
                Group(*elements),
                title=f"[{Theme.WARNING}]🔄 Ralph Loop Starting[/{Theme.WARNING}]",
                title_align="left",
                border_style=Theme.WARNING,
                padding=(0, 1),
                box=ROUNDED,
            )
        )
        self.console.print()

    def print_iteration(
        self,
        iteration: int,
        max_iterations: int | None,
        response_preview: str,
        is_complete: bool = False,
    ) -> None:
        """Print iteration progress."""
        elapsed = time.time() - self._start_time

        # Build progress line
        progress_line = Text()

        if is_complete:
            progress_line.append(f"{Icons.DONE} ", style=f"bold {Theme.SUCCESS}")
            progress_line.append("Complete", style=Theme.SUCCESS)
        else:
            progress_line.append(f"{self._get_spinner()} ", style=f"bold {Theme.WARNING}")
            progress_line.append(f"Iteration {iteration}", style=Theme.WARNING)

        if max_iterations:
            progress_line.append(f"/{max_iterations}", style=Theme.MUTED)

            # Progress bar
            progress = iteration / max_iterations
            progress_line.append("  ")
            progress_line.append_text(build_progress_bar(progress, width=20))
            progress_line.append(f"  {int(progress * 100)}%", style=Theme.MUTED)

        # Elapsed time
        progress_line.append(f"  {Icons.CLOCK} {format_duration(elapsed)}", style=Theme.MUTED)

        self.console.print(progress_line)

        # Response preview (truncated)
        if response_preview:
            preview = response_preview[:200] + "..." if len(response_preview) > 200 else response_preview
            self.console.print(Text(f"  └─ {preview}", style=Theme.MUTED))
        self.console.print()

    def print_complete(
        self,
        iteration: int,
        completion_reason: str,
        total_time: float | None = None,
    ) -> None:
        """Print loop completion summary."""
        elapsed = total_time or (time.time() - self._start_time)

        elements = []

        # Summary stats
        summary = Table.grid(expand=True)
        summary.add_column(justify="left")
        summary.add_column(justify="right")

        completed_text = Text()
        completed_text.append(f"{Icons.DONE} ", style=Theme.SUCCESS)
        completed_text.append(f"{iteration} iterations", style=Theme.SUCCESS)

        time_text = Text()
        time_text.append(f"{Icons.CLOCK} ", style=Theme.MUTED)
        time_text.append(format_duration(elapsed), style=Theme.MUTED)

        summary.add_row(completed_text, time_text)

        reason_text = Text()
        reason_text.append("Reason: ", style=Theme.MUTED)
        reason_text.append(completion_reason, style=Theme.PRIMARY)
        summary.add_row(reason_text, Text())

        elements.append(summary)

        self.console.print(
            Panel(
                Group(*elements),
                title=f"[{Theme.SUCCESS}]{Icons.DONE} Ralph Loop Complete[/{Theme.SUCCESS}]",
                title_align="left",
                border_style=Theme.SUCCESS,
                padding=(0, 1),
                box=ROUNDED,
            )
        )
        self.console.print()


class PlanUI:
    """
    Beautiful UI for Plan execution.

    Displays step-by-step progress, ETA, and execution statistics.
    """

    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self._spinners = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._spinner_idx = 0
        self._start_time = time.time()

    def _get_spinner(self) -> str:
        self._spinner_idx = (self._spinner_idx + 1) % len(self._spinners)
        return self._spinners[self._spinner_idx]

    def print_plan_header(
        self,
        task: str,
        step_count: int,
        model: str = "",
        reasoning: str = "",
    ) -> None:
        """Print the plan header."""
        self._start_time = time.time()

        elements = []

        # Config info
        info_table = Table.grid(padding=(0, 2))
        info_table.add_column(style=Theme.MUTED)
        info_table.add_column(style=Theme.MESSAGE)

        if model:
            info_table.add_row(f"{Icons.ROBOT} Model:", model)
        if reasoning:
            info_table.add_row(f"{Icons.BRAIN} Reasoning:", reasoning)
        info_table.add_row("📋 Steps:", str(step_count))

        elements.append(info_table)
        elements.append(Text())

        # Task description
        task_preview = task[:200] + "..." if len(task) > 200 else task
        elements.append(Text(f"📝 Task: {task_preview}", style=Theme.MESSAGE))

        self.console.print(
            Panel(
                Group(*elements),
                title=f"[{Theme.PRIMARY}]📋 Plan Execution[/{Theme.PRIMARY}]",
                title_align="left",
                border_style=Theme.PRIMARY,
                padding=(0, 1),
                box=ROUNDED,
            )
        )
        self.console.print()

    def print_plan_overview(
        self,
        steps: list[tuple[int, str, str]],  # (number, description, status)
    ) -> None:
        """Print the plan overview with all steps."""
        elements = []

        for number, description, status in steps:
            status_icon = {
                "pending": "⬜",
                "running": "🔄",
                "completed": "✅",
                "failed": "❌",
                "skipped": "⏭️",
            }.get(status, "⬜")

            status_style = {
                "pending": Theme.MUTED,
                "running": Theme.WARNING,
                "completed": Theme.SUCCESS,
                "failed": Theme.ERROR,
                "skipped": Theme.MUTED,
            }.get(status, Theme.MUTED)

            line = Text()
            line.append(f"{status_icon} ", style=status_style)
            line.append(f"Step {number}: ", style=f"bold {status_style}")
            line.append(description, style=status_style)
            elements.append(line)

        self.console.print(
            Panel(
                Group(*elements),
                title=f"[{Theme.MUTED}]Plan Steps[/{Theme.MUTED}]",
                title_align="left",
                border_style=Theme.BORDER,
                padding=(0, 1),
                box=ROUNDED,
            )
        )
        self.console.print()

    def print_step_start(
        self,
        step_number: int,
        total_steps: int,
        description: str,
    ) -> None:
        """Print step start notification."""
        progress = (step_number - 1) / total_steps

        line = Text()
        line.append(f"{self._get_spinner()} ", style=f"bold {Theme.WARNING}")
        line.append(f"Step {step_number}/{total_steps}: ", style=f"bold {Theme.WARNING}")
        line.append(description, style=Theme.MESSAGE)

        self.console.print(line)

        # Progress bar
        bar_line = Text()
        bar_line.append("  ")
        bar_line.append_text(build_progress_bar(progress, width=30))
        bar_line.append(f"  {int(progress * 100)}%", style=Theme.MUTED)
        self.console.print(bar_line)
        self.console.print()

    def print_step_complete(
        self,
        step_number: int,
        total_steps: int,
        duration: float,
        result_preview: str | None = None,
        eta_remaining: float | None = None,
    ) -> None:
        """Print step completion notification."""
        line = Text()
        line.append(f"{Icons.DONE} ", style=f"bold {Theme.SUCCESS}")
        line.append(f"Step {step_number}/{total_steps} complete ", style=Theme.SUCCESS)
        line.append(f"({format_duration(duration)})", style=Theme.MUTED)

        self.console.print(line)

        if result_preview:
            preview = result_preview[:150] + "..." if len(result_preview) > 150 else result_preview
            self.console.print(Text(f"  └─ {preview}", style=Theme.MUTED))

        if eta_remaining is not None:
            eta_line = Text()
            eta_line.append("  ", style="")
            eta_line.append(f"{Icons.CLOCK} ", style=Theme.INFO)
            eta_line.append(f"Estimated remaining: ~{format_duration(eta_remaining)}", style=Theme.INFO)
            self.console.print(eta_line)

        self.console.print()

    def print_step_failed(
        self,
        step_number: int,
        total_steps: int,
        error: str,
        duration: float,
    ) -> None:
        """Print step failure notification."""
        line = Text()
        line.append(f"{Icons.ERROR} ", style=f"bold {Theme.ERROR}")
        line.append(f"Step {step_number}/{total_steps} failed ", style=Theme.ERROR)
        line.append(f"({format_duration(duration)})", style=Theme.MUTED)

        self.console.print(line)

        error_preview = error[:200] + "..." if len(error) > 200 else error
        self.console.print(Text(f"  └─ {error_preview}", style=Theme.ERROR))
        self.console.print(Text("  └─ Checkpoint saved. Resume with: copex plan --resume", style=Theme.MUTED))
        self.console.print()

    def print_plan_complete(
        self,
        completed_steps: int,
        failed_steps: int,
        total_steps: int,
        total_time: float,
        total_tokens: int = 0,
    ) -> None:
        """Print plan completion summary."""
        elements = []

        # Status line
        if failed_steps == 0 and completed_steps == total_steps:
            status_line = Text()
            status_line.append(f"{Icons.DONE} ", style=f"bold {Theme.SUCCESS}")
            status_line.append(f"{completed_steps}/{total_steps} steps completed successfully!", style=Theme.SUCCESS)
            border_style = Theme.SUCCESS
            title_style = Theme.SUCCESS
        elif failed_steps > 0:
            status_line = Text()
            status_line.append(f"{Icons.WARNING} ", style=f"bold {Theme.WARNING}")
            status_line.append(f"{completed_steps}/{total_steps} completed, {failed_steps} failed", style=Theme.WARNING)
            border_style = Theme.WARNING
            title_style = Theme.WARNING
        else:
            status_line = Text()
            status_line.append(f"{Icons.INFO} ", style=f"bold {Theme.INFO}")
            status_line.append(f"{completed_steps}/{total_steps} steps completed", style=Theme.INFO)
            border_style = Theme.INFO
            title_style = Theme.INFO

        elements.append(status_line)
        elements.append(Text())

        # Stats table
        stats = Table.grid(expand=True)
        stats.add_column(justify="left")
        stats.add_column(justify="right")

        time_text = Text()
        time_text.append(f"{Icons.CLOCK} ", style=Theme.MUTED)
        time_text.append(f"Total time: {format_duration(total_time)}", style=Theme.MUTED)

        avg_text = Text()
        if completed_steps > 0:
            avg = total_time / completed_steps
            avg_text.append(f"Avg: {format_duration(avg)}/step", style=Theme.MUTED)

        stats.add_row(time_text, avg_text)

        if total_tokens > 0:
            token_text = Text()
            token_text.append(f"{Icons.LIGHTNING} ", style=Theme.INFO)
            token_text.append(f"Tokens: {total_tokens:,}", style=Theme.INFO)
            stats.add_row(token_text, Text())

        elements.append(stats)

        self.console.print(
            Panel(
                Group(*elements),
                title=f"[{title_style}]{Icons.DONE} Plan Complete[/{title_style}]",
                title_align="left",
                border_style=border_style,
                padding=(0, 1),
                box=ROUNDED,
            )
        )
        self.console.print()
