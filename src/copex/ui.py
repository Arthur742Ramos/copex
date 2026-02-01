"""Beautiful CLI UI components for Copex - Codex CLI inspired.

This module provides polished terminal UI components with:
- Shimmer animations (like Codex CLI)
- Terminal color detection for light/dark themes
- Adaptive color blending
- Smooth spinner animations
- Clean status indicators
- Diff visualization with line numbers
- Professional color scheme
- Token usage display with cost calculation
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from rich.box import ROUNDED, MINIMAL
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown

from copex.ui_components import TokenUsageDisplay
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

# ═══════════════════════════════════════════════════════════════════════════════
# Terminal Detection & Color System (Codex-inspired)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_terminal_colors() -> tuple[tuple[int, int, int] | None, tuple[int, int, int] | None]:
    """
    Detect terminal foreground and background colors.
    Returns (fg_rgb, bg_rgb) or (None, None) if detection fails.
    
    Uses OSC 10/11 escape sequences on Unix.
    """
    if sys.platform == "win32" or not sys.stdout.isatty():
        return None, None
    
    # Skip detection in non-interactive environments
    if os.environ.get("CI") or os.environ.get("NO_COLOR"):
        return None, None
    
    # For now, return None and let is_light_theme() use heuristics
    # Full OSC detection would require raw terminal I/O
    return None, None


def is_light_theme() -> bool:
    """
    Detect if terminal is using a light theme.
    Uses multiple heuristics:
    1. COLORFGBG environment variable
    2. Common terminal-specific variables
    3. macOS Terminal.app detection
    """
    # Check COLORFGBG (format: "fg;bg" where bg > 7 usually means light)
    colorfgbg = os.environ.get("COLORFGBG", "")
    if ";" in colorfgbg:
        try:
            _, bg = colorfgbg.rsplit(";", 1)
            bg_idx = int(bg)
            # In 16-color mode, indices 7, 15 are often white/light
            if bg_idx in (7, 15):
                return True
            if bg_idx == 0:
                return False
        except (ValueError, IndexError):
            pass
    
    # macOS Terminal.app detection
    term_program = os.environ.get("TERM_PROGRAM", "")
    if term_program == "Apple_Terminal":
        # Apple Terminal defaults to light
        return True
    
    # iTerm2 can report theme
    iterm_profile = os.environ.get("ITERM_PROFILE", "").lower()
    if "light" in iterm_profile:
        return True
    if "dark" in iterm_profile:
        return False
    
    # VS Code terminal
    if os.environ.get("VSCODE_INJECTION"):
        vscode_theme = os.environ.get("VSCODE_TERM_PROFILE", "").lower()
        if "light" in vscode_theme:
            return True
    
    # Default to dark (most developer terminals are dark)
    return False


def supports_true_color() -> bool:
    """Check if terminal supports 24-bit RGB colors."""
    colorterm = os.environ.get("COLORTERM", "")
    return colorterm in ("truecolor", "24bit")


def supports_256_color() -> bool:
    """Check if terminal supports 256 colors."""
    term = os.environ.get("TERM", "")
    return "256color" in term or supports_true_color()


def blend_color(
    fg: tuple[int, int, int],
    bg: tuple[int, int, int],
    alpha: float,
) -> tuple[int, int, int]:
    """Blend fg over bg with alpha (0-1)."""
    r = int(fg[0] * alpha + bg[0] * (1 - alpha))
    g = int(fg[1] * alpha + bg[1] * (1 - alpha))
    b = int(fg[2] * alpha + bg[2] * (1 - alpha))
    return (min(255, max(0, r)), min(255, max(0, g)), min(255, max(0, b)))


def rgb_to_ansi256(r: int, g: int, b: int) -> int:
    """Convert RGB to nearest ANSI 256 color index."""
    # Check grayscale ramp first (232-255)
    if r == g == b:
        if r < 8:
            return 16
        if r > 248:
            return 231
        return round((r - 8) / 247 * 24) + 232
    
    # 6x6x6 color cube (16-231)
    def to_cube(v: int) -> int:
        if v < 48:
            return 0
        if v < 115:
            return 1
        return (v - 35) // 40
    
    return 16 + 36 * to_cube(r) + 6 * to_cube(g) + to_cube(b)


# ═══════════════════════════════════════════════════════════════════════════════
# Theme and Colors (adaptive to terminal)
# ═══════════════════════════════════════════════════════════════════════════════

class Theme:
    """Color theme for the UI - adapts to terminal colors."""

    # These are defaults, can be overridden by apply_theme()
    _is_light = is_light_theme()
    
    # Brand colors (ANSI compatible, per Codex style guide)
    PRIMARY = "cyan"
    SECONDARY = "blue"
    ACCENT = "magenta"

    # Status colors
    SUCCESS = "green"
    WARNING = "yellow"
    ERROR = "red"
    INFO = "cyan"

    # Content colors (adapt to light/dark)
    REASONING = "dim italic"
    MESSAGE = "default"
    CODE = "bright_white" if not _is_light else "black"
    MUTED = "dim"

    # UI elements
    BORDER = "bright_black" if not _is_light else "grey70"
    BORDER_ACTIVE = "cyan"
    HEADER = "bold cyan"
    SUBHEADER = "bold"


THEME_PRESETS = {
    "default": {
        "PRIMARY": "cyan",
        "SECONDARY": "blue",
        "ACCENT": "magenta",
        "SUCCESS": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "INFO": "cyan",
        "REASONING": "dim italic",
        "MESSAGE": "default",
        "CODE": "bright_white",
        "MUTED": "dim",
        "BORDER": "bright_black",
        "BORDER_ACTIVE": "cyan",
        "HEADER": "bold cyan",
        "SUBHEADER": "bold",
    },
    "light": {
        "PRIMARY": "blue",
        "SECONDARY": "cyan",
        "ACCENT": "magenta",
        "SUCCESS": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "INFO": "blue",
        "REASONING": "dim italic",
        "MESSAGE": "default",
        "CODE": "black",
        "MUTED": "grey50",
        "BORDER": "grey70",
        "BORDER_ACTIVE": "blue",
        "HEADER": "bold blue",
        "SUBHEADER": "bold",
    },
    "midnight": {
        "PRIMARY": "bright_cyan",
        "SECONDARY": "bright_blue",
        "ACCENT": "bright_magenta",
        "SUCCESS": "bright_green",
        "WARNING": "bright_yellow",
        "ERROR": "bright_red",
        "INFO": "bright_cyan",
        "REASONING": "dim italic",
        "MESSAGE": "default",
        "CODE": "bright_white",
        "MUTED": "grey70",
        "BORDER": "grey39",
        "BORDER_ACTIVE": "bright_cyan",
        "HEADER": "bold bright_cyan",
        "SUBHEADER": "bold bright_white",
    },
    "mono": {
        "PRIMARY": "default",
        "SECONDARY": "default",
        "ACCENT": "default",
        "SUCCESS": "default",
        "WARNING": "default",
        "ERROR": "default",
        "INFO": "default",
        "REASONING": "dim",
        "MESSAGE": "default",
        "CODE": "default",
        "MUTED": "dim",
        "BORDER": "dim",
        "BORDER_ACTIVE": "default",
        "HEADER": "bold",
        "SUBHEADER": "bold",
    },
    "codex": {
        # Matches Codex CLI's minimalist style
        "PRIMARY": "cyan",
        "SECONDARY": "blue",
        "ACCENT": "magenta",
        "SUCCESS": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "INFO": "cyan",
        "REASONING": "dim italic",
        "MESSAGE": "default",
        "CODE": "cyan",
        "MUTED": "dim",
        "BORDER": "dim",
        "BORDER_ACTIVE": "cyan",
        "HEADER": "bold",
        "SUBHEADER": "bold",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Icons and Symbols (with ASCII fallback)
# ═══════════════════════════════════════════════════════════════════════════════

class Icons:
    """Unicode icons for the UI."""

    # Status
    THINKING = "◐"
    DONE = "✓"
    ERROR = "✗"
    WARNING = "⚠"
    INFO = "ℹ"

    # Spinner frames (Codex style - dot blink)
    SPINNER_FRAMES = ["•", "◦"]
    
    # Alternative: Braille spinner for more animation
    BRAILLE_SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

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
    TREE_CORNER = "└"
    TREE_VERT = "│"
    TREE_TEE = "├"

    # Misc
    SPARKLE = "✨"
    BRAIN = "🧠"
    ROBOT = "🤖"
    LIGHTNING = "⚡"
    CLOCK = "⏱"
    ELLIPSIS = "⋮"


class ASCIIIcons:
    """ASCII fallback icons."""

    THINKING = "..."
    DONE = "[OK]"
    ERROR = "[X]"
    WARNING = "[!]"
    INFO = "[i]"

    SPINNER_FRAMES = [".", "o"]
    BRAILLE_SPINNER = ["/", "-", "\\", "|"]

    TOOL = "*"
    FILE_READ = "R"
    FILE_WRITE = "W"
    FILE_CREATE = "+"
    SEARCH = "?"
    TERMINAL = "$"
    GLOBE = "@"

    ARROW_RIGHT = "->"
    ARROW_DOWN = "v"
    BULLET = "*"
    TREE_CORNER = "`-"
    TREE_VERT = "|"
    TREE_TEE = "|-"

    SPARKLE = "*"
    BRAIN = "!"
    ROBOT = "AI"
    LIGHTNING = "*"
    CLOCK = "t"
    ELLIPSIS = ":"


# ═══════════════════════════════════════════════════════════════════════════════
# Shimmer Animation (Codex-inspired)
# ═══════════════════════════════════════════════════════════════════════════════

_SHIMMER_START = time.time()


def shimmer_text(text: str, use_rgb: bool = True) -> Text:
    """
    Create a shimmering text effect like Codex CLI.
    
    A bright band sweeps across the text periodically.
    Works best with true color terminals.
    """
    if not text:
        return Text()
    
    chars = list(text)
    now = time.time() - _SHIMMER_START
    
    # Shimmer parameters (match Codex)
    padding = 10
    period = len(chars) + padding * 2
    sweep_seconds = 2.0
    pos = (now % sweep_seconds) / sweep_seconds * period
    band_half_width = 5.0
    
    # Base and highlight colors
    is_light = is_light_theme()
    if is_light:
        base_rgb = (80, 80, 80)  # Dark gray for light theme
        highlight_rgb = (0, 0, 0)  # Black highlight
    else:
        base_rgb = (128, 128, 128)  # Gray
        highlight_rgb = (255, 255, 255)  # White highlight
    
    result = Text()
    
    for i, ch in enumerate(chars):
        i_pos = i + padding
        dist = abs(i_pos - pos)
        
        # Cosine falloff for smooth band
        if dist <= band_half_width:
            import math
            t = 0.5 * (1 + math.cos(math.pi * dist / band_half_width))
        else:
            t = 0.0
        
        if use_rgb and supports_true_color():
            # True color shimmer
            r, g, b = blend_color(highlight_rgb, base_rgb, t * 0.9)
            result.append(ch, style=f"bold rgb({r},{g},{b})")
        else:
            # Fallback: use bold/dim for shimmer effect
            if t < 0.2:
                result.append(ch, style="dim")
            elif t < 0.6:
                result.append(ch, style="default")
            else:
                result.append(ch, style="bold")
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Diff Visualization (Codex-inspired)
# ═══════════════════════════════════════════════════════════════════════════════

def render_diff(
    diff_text: str,
    wrap_width: int = 80,
    show_line_numbers: bool = True,
) -> Text:
    """
    Render a unified diff with colors and line numbers.
    
    Inspired by Codex CLI's diff_render.rs.
    """
    result = Text()
    
    lines = diff_text.split("\n")
    old_ln = 0
    new_ln = 0
    ln_width = 4  # Default width
    
    # Pre-scan to find max line number for width
    for line in lines:
        if line.startswith("@@"):
            # Parse hunk header: @@ -start,count +start,count @@
            try:
                parts = line.split()
                if len(parts) >= 3:
                    new_part = parts[2]  # e.g., "+1,5"
                    if new_part.startswith("+"):
                        new_start = int(new_part[1:].split(",")[0])
                        ln_width = max(ln_width, len(str(new_start + 100)))
            except (ValueError, IndexError):
                pass
    
    for line in lines:
        if line.startswith("@@"):
            # Hunk header
            try:
                parts = line.split()
                if len(parts) >= 3:
                    old_part = parts[1]  # e.g., "-1,5"
                    new_part = parts[2]  # e.g., "+1,5"
                    if old_part.startswith("-"):
                        old_ln = int(old_part[1:].split(",")[0])
                    if new_part.startswith("+"):
                        new_ln = int(new_part[1:].split(",")[0])
            except (ValueError, IndexError):
                old_ln, new_ln = 1, 1
            
            if result:
                gutter = " " * (ln_width + 1)
                result.append(f"{gutter}{Icons.ELLIPSIS}\n", style="dim")
            continue
        
        if line.startswith("+") and not line.startswith("+++"):
            # Addition
            if show_line_numbers:
                result.append(f"{new_ln:>{ln_width}} ", style="dim")
            result.append(f"+{line[1:]}\n", style="green")
            new_ln += 1
        elif line.startswith("-") and not line.startswith("---"):
            # Deletion
            if show_line_numbers:
                result.append(f"{old_ln:>{ln_width}} ", style="dim")
            result.append(f"-{line[1:]}\n", style="red")
            old_ln += 1
        elif line.startswith(" "):
            # Context
            if show_line_numbers:
                result.append(f"{new_ln:>{ln_width}} ", style="dim")
            result.append(f" {line[1:]}\n", style="default")
            old_ln += 1
            new_ln += 1
        elif line.startswith("---") or line.startswith("+++"):
            # File headers - skip or show dimmed
            pass
        elif line.startswith("diff ") or line.startswith("index "):
            # Git diff headers - skip
            pass
    
    return result


def format_file_change_summary(
    path: str,
    added: int = 0,
    removed: int = 0,
    operation: str = "Edited",
) -> Text:
    """Format a file change summary line like Codex CLI."""
    result = Text()
    result.append(f"• {operation} ", style="bold")
    result.append(path, style="default")
    result.append(" (", style="default")
    result.append(f"+{added}", style="green")
    result.append(" ", style="default")
    result.append(f"-{removed}", style="red")
    result.append(")", style="default")
    return result


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
    id: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str | None = None
    status: str = "running"  # running, success, error
    duration: float | None = None
    started_at: float = field(default_factory=time.time)

    def icon(self, icon_set: type[Icons] | type[ASCIIIcons] = Icons) -> str:
        """Get appropriate icon for the tool."""
        name_lower = self.name.lower()
        if "read" in name_lower or "view" in name_lower:
            return icon_set.FILE_READ
        elif "write" in name_lower or "edit" in name_lower:
            return icon_set.FILE_WRITE
        elif "create" in name_lower:
            return icon_set.FILE_CREATE
        elif "search" in name_lower or "grep" in name_lower or "glob" in name_lower:
            return icon_set.SEARCH
        elif "shell" in name_lower or "bash" in name_lower or "powershell" in name_lower:
            return icon_set.TERMINAL
        elif "web" in name_lower or "fetch" in name_lower:
            return icon_set.GLOBE
        return icon_set.TOOL

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
    input_tokens: int = 0
    output_tokens: int = 0
    token_usage: dict[str, int] = field(default_factory=dict)  # input, output, reasoning

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def elapsed_str(self) -> str:
        """Format elapsed time like Codex: 0s, 1m 00s, 1h 00m 00s."""
        elapsed = int(self.elapsed)
        if elapsed < 60:
            return f"{elapsed}s"
        if elapsed < 3600:
            minutes = elapsed // 60
            seconds = elapsed % 60
            return f"{minutes}m {seconds:02d}s"
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60
        return f"{hours}h {minutes:02d}m {seconds:02d}s"

    @property
    def idle(self) -> float:
        return time.time() - self.last_update


# ═══════════════════════════════════════════════════════════════════════════════
# UI Components (Codex-inspired)
# ═══════════════════════════════════════════════════════════════════════════════

class CopexUI:
    """
    Beautiful UI for Copex CLI - Codex CLI inspired.
    
    Key features:
    - Shimmer animation on status text
    - Adaptive to terminal colors (light/dark)
    - Clean, minimal status line
    - Smooth 32ms frame rate
    - Tool call tree display
    """

    def __init__(
        self,
        console: Console | None = None,
        *,
        theme: str = "default",
        density: str = "extended",
        show_all_tools: bool = False,
        show_reasoning: bool = True,
        ascii_icons: bool = False,
        animations: bool = True,
    ):
        self.console = console or Console()
        self.set_theme(theme)
        self.density = density
        self.state = UIState()
        self._dirty = True
        self._live: Live | None = None
        
        # Animation state
        self._animations = animations and self.console.is_terminal
        self._frame_idx = 0
        self._last_frame_at = 0.0
        self._frame_interval = 0.032  # 32ms like Codex
        
        self.show_all_tools = show_all_tools
        self.show_reasoning = show_reasoning
        self._icon_set = ASCIIIcons if ascii_icons else Icons
        
        # Auto-detect light theme
        if is_light_theme() and theme == "default":
            self.set_theme("light")

    def _advance_frame(self) -> None:
        """Advance animation frame at 32ms intervals."""
        now = time.time()
        if now - self._last_frame_at >= self._frame_interval:
            self._last_frame_at = now
            self._frame_idx = (self._frame_idx + 1) % 60  # 60 frame cycle

    def _get_spinner(self) -> str:
        """Get current spinner frame."""
        if not self._animations:
            return self._icon_set.BULLET
        
        # Use Braille spinner for smooth animation
        frames = self._icon_set.BRAILLE_SPINNER
        return frames[self._frame_idx % len(frames)]

    def _build_status_line(self) -> Text:
        """
        Build the main status line like Codex CLI:
        • Working (5s • Esc to interrupt)
        """
        line = Text()
        
        # Spinner
        if self._animations and self.state.activity in (
            ActivityType.THINKING, ActivityType.REASONING,
            ActivityType.RESPONDING, ActivityType.TOOL_CALL
        ):
            spinner_text = shimmer_text(self._icon_set.BULLET)
            line.append_text(spinner_text)
        else:
            icon = {
                ActivityType.DONE: self._icon_set.DONE,
                ActivityType.ERROR: self._icon_set.ERROR,
            }.get(self.state.activity, self._icon_set.BULLET)
            style = {
                ActivityType.DONE: f"bold {Theme.SUCCESS}",
                ActivityType.ERROR: f"bold {Theme.ERROR}",
            }.get(self.state.activity, "dim")
            line.append(icon, style=style)
        
        line.append(" ", style="default")
        
        # Activity text with shimmer
        activity_labels = {
            ActivityType.THINKING: "Thinking",
            ActivityType.REASONING: "Reasoning",
            ActivityType.RESPONDING: "Writing",
            ActivityType.TOOL_CALL: "Working",
            ActivityType.WAITING: "Waiting",
            ActivityType.DONE: "Done",
            ActivityType.ERROR: "Error",
        }
        label = activity_labels.get(self.state.activity, "Working")
        
        if self._animations and self.state.activity not in (ActivityType.DONE, ActivityType.ERROR, ActivityType.WAITING):
            line.append_text(shimmer_text(label))
        else:
            style = {
                ActivityType.DONE: Theme.SUCCESS,
                ActivityType.ERROR: Theme.ERROR,
            }.get(self.state.activity, "default")
            line.append(label, style=style)
        
        # Elapsed time and interrupt hint
        line.append(" ", style="default")
        line.append(f"({self.state.elapsed_str}", style="dim")
        
        if self.state.activity in (ActivityType.THINKING, ActivityType.REASONING,
                                   ActivityType.RESPONDING, ActivityType.TOOL_CALL):
            line.append(" • ", style="dim")
            line.append("Esc", style="default")
            line.append(" to interrupt", style="dim")
        
        line.append(")", style="dim")
        
        return line

    def _build_details_line(self) -> Text | None:
        """Build optional details line below status."""
        if not self.state.tool_calls:
            return None
        
        running = [t for t in self.state.tool_calls if t.status == "running"]
        if not running:
            return None
        
        # Show current tool being executed
        tool = running[-1]
        result = Text()
        result.append(f"  {self._icon_set.TREE_CORNER} ", style="dim")
        result.append(tool.name, style="dim")
        
        if tool.arguments:
            # Show key argument
            for key in ("path", "command", "pattern", "query", "file"):
                if key in tool.arguments:
                    val = str(tool.arguments[key])[:50]
                    if len(str(tool.arguments[key])) > 50:
                        val += "..."
                    result.append(f" {key}=", style="dim")
                    result.append(val, style="dim italic")
                    break
        
        return result

    def _build_reasoning_panel(self) -> Panel | None:
        """Build the reasoning panel."""
        if not self.show_reasoning or not self.state.reasoning:
            return None
        
        # Truncate for live display
        reasoning = self.state.reasoning
        max_chars = 800 if self.density == "extended" else 400
        if len(reasoning) > max_chars:
            reasoning = "..." + reasoning[-max_chars:]
        
        content = Text(reasoning, style=Theme.REASONING)
        if self.state.activity == ActivityType.REASONING:
            content.append("▌", style=f"bold {Theme.ACCENT}")
        
        return Panel(
            content,
            title=f"[{Theme.ACCENT}]{self._icon_set.BRAIN} Reasoning[/{Theme.ACCENT}]",
            title_align="left",
            border_style=Theme.BORDER_ACTIVE if self.state.activity == ActivityType.REASONING else Theme.BORDER,
            box=MINIMAL if self.density == "compact" else ROUNDED,
            padding=(0, 1),
        )

    def _build_tool_calls_panel(self) -> Panel | None:
        """Build the tool calls panel."""
        if not self.state.tool_calls:
            return None
        
        running = sum(1 for t in self.state.tool_calls if t.status == "running")
        successful = sum(1 for t in self.state.tool_calls if t.status == "success")
        failed = sum(1 for t in self.state.tool_calls if t.status == "error")
        
        # Compact summary for title
        parts = []
        if running:
            parts.append(f"{running} running")
        if successful:
            parts.append(f"{successful} ok")
        if failed:
            parts.append(f"{failed} failed")
        summary = " • ".join(parts) if parts else "Tools"
        
        title = f"[{Theme.WARNING}]{self._icon_set.TOOL} {summary}[/{Theme.WARNING}]"
        
        # Build tree
        tree = Tree("")
        max_tools = 5 if self.density == "extended" else 3
        tools_to_show = self.state.tool_calls if self.show_all_tools else self.state.tool_calls[-max_tools:]
        
        for tool in tools_to_show:
            status_style = {
                "running": Theme.WARNING,
                "success": Theme.SUCCESS,
                "error": Theme.ERROR,
            }.get(tool.status, Theme.MUTED)
            
            tool_text = Text()
            
            # Status icon
            if tool.status == "running":
                tool_text.append(self._get_spinner(), style=status_style)
            elif tool.status == "success":
                tool_text.append(self._icon_set.DONE, style=status_style)
            else:
                tool_text.append(self._icon_set.ERROR, style=status_style)
            
            tool_text.append(" ", style="default")
            tool_text.append(tool.icon(self._icon_set), style=status_style)
            tool_text.append(" ", style="default")
            tool_text.append(tool.name, style=f"bold {status_style}")
            
            # Duration
            if tool.status == "running":
                tool_text.append(f" ({tool.elapsed:.1f}s)", style="dim")
            elif tool.duration:
                tool_text.append(f" ({tool.duration:.1f}s)", style="dim")
            
            branch = tree.add(tool_text)
            
            # Result preview
            if tool.result and tool.status != "running" and self.density == "extended":
                preview = tool.result[:80]
                if len(tool.result) > 80:
                    preview += "..."
                branch.add(Text(preview, style="dim"))
        
        if len(self.state.tool_calls) > max_tools and not self.show_all_tools:
            tree.add(Text(f"... +{len(self.state.tool_calls) - max_tools} more", style="dim"))
        
        border_style = Theme.BORDER
        if running:
            border_style = Theme.BORDER_ACTIVE
        if failed:
            border_style = Theme.ERROR
        
        return Panel(
            tree,
            title=title,
            title_align="left",
            border_style=border_style,
            box=MINIMAL if self.density == "compact" else ROUNDED,
            padding=(0, 1),
        )

    def _build_message_panel(self) -> Panel | None:
        """Build the message/response panel."""
        if not self.state.message:
            return None
        
        content = Text(self.state.message, style=Theme.MESSAGE)
        if self.state.activity == ActivityType.RESPONDING:
            content.append("▌", style=f"bold {Theme.PRIMARY}")
        
        return Panel(
            content,
            title=f"[{Theme.PRIMARY}]{self._icon_set.ROBOT} Response[/{Theme.PRIMARY}]",
            title_align="left",
            border_style=Theme.BORDER_ACTIVE if self.state.activity == ActivityType.RESPONDING else Theme.BORDER,
            box=MINIMAL if self.density == "compact" else ROUNDED,
            padding=(0, 1),
        )

    def _build_summary_line(self) -> Text:
        """Build a compact summary line for completed output."""
        line = Text()
        
        # Elapsed
        line.append(f"{self._icon_set.CLOCK} ", style="dim")
        line.append(self.state.elapsed_str, style="dim")
        
        # Tools summary
        if self.state.tool_calls:
            successful = sum(1 for t in self.state.tool_calls if t.status == "success")
            failed = sum(1 for t in self.state.tool_calls if t.status == "error")
            line.append(f" • {self._icon_set.TOOL} ", style="dim")
            if failed:
                line.append(f"{successful} ok, ", style="green")
                line.append(f"{failed} failed", style="red")
            else:
                line.append(f"{successful} tools", style="green")
        
        # Retries
        if self.state.retries:
            line.append(f" • {self._icon_set.WARNING} ", style="dim")
            line.append(f"{self.state.retries} retries", style="yellow")
        
        # Token usage with cost calculation using TokenUsageDisplay
        if self.state.input_tokens > 0 or self.state.output_tokens > 0:
            usage_display = TokenUsageDisplay(
                input_tokens=self.state.input_tokens,
                output_tokens=self.state.output_tokens,
                model=self.state.model if self.state.model else None,
                compact=True,
                show_cost=True,
            )
            line.append(" • ", style="dim")
            line.append_text(usage_display.to_text())
        elif self.state.token_usage:
            # Fallback to dict-based display
            total = sum(self.state.token_usage.values())
            line.append(f" • {total:,} tokens", style="dim")
        
        return line

    def build_live_display(self) -> Group:
        """Build the complete live display."""
        self._advance_frame()
        elements = []
        
        # Main status line (Codex style)
        elements.append(self._build_status_line())
        
        # Details line (current tool)
        details = self._build_details_line()
        if details:
            elements.append(details)
        
        elements.append(Text())  # Spacer
        
        # Reasoning panel
        if self.density == "extended":
            reasoning_panel = self._build_reasoning_panel()
            if reasoning_panel:
                elements.append(reasoning_panel)
                elements.append(Text())
        
        # Tool calls panel
        tool_panel = self._build_tool_calls_panel()
        if tool_panel:
            elements.append(tool_panel)
            elements.append(Text())
        
        # Message panel
        message_panel = self._build_message_panel()
        if message_panel:
            elements.append(message_panel)
        
        return Group(*elements)

    def build_final_display(self) -> Group:
        """Build the final formatted display after streaming completes."""
        elements = []
        
        # Reasoning (markdown rendered)
        if self.show_reasoning and self.state.reasoning and self.density == "extended":
            elements.append(Panel(
                Markdown(self.state.reasoning),
                title=f"[{Theme.ACCENT}]{self._icon_set.BRAIN} Reasoning[/{Theme.ACCENT}]",
                title_align="left",
                border_style=Theme.BORDER,
                box=ROUNDED,
                padding=(0, 1),
            ))
            elements.append(Text())
        
        # Main response (markdown rendered)
        if self.state.message:
            elements.append(Panel(
                Markdown(self.state.message),
                title=f"[{Theme.PRIMARY}]{self._icon_set.ROBOT} Response[/{Theme.PRIMARY}]",
                title_align="left",
                border_style=Theme.BORDER_ACTIVE,
                box=ROUNDED,
                padding=(0, 1),
            ))
        
        # Summary line
        elements.append(Text())
        elements.append(self._build_summary_line())
        
        return Group(*elements)

    # ═══════════════════════════════════════════════════════════════════════════
    # Public Methods
    # ═══════════════════════════════════════════════════════════════════════════

    def reset(self, model: str = "", preserve_history: bool = False) -> None:
        """Reset UI state for a new interaction."""
        old_history = self.state.history if preserve_history else []
        self.state = UIState(model=model, history=old_history)
        self._frame_idx = 0
        self._touch()

    def set_activity(self, activity: ActivityType) -> None:
        """Set the current activity indicator."""
        self.state.activity = activity
        self._touch()

    def add_reasoning(self, delta: str) -> None:
        """Append reasoning content."""
        if not self.show_reasoning:
            return
        self.state.reasoning += delta
        if self.state.activity != ActivityType.REASONING:
            self.state.activity = ActivityType.REASONING
        self._touch()

    def add_message(self, delta: str) -> None:
        """Append message content."""
        self.state.message += delta
        if self.state.activity != ActivityType.RESPONDING:
            self.state.activity = ActivityType.RESPONDING
        self._touch()

    def add_tool_call(self, tool: ToolCallInfo) -> None:
        """Track a tool call."""
        self.state.tool_calls.append(tool)
        self.state.activity = ActivityType.TOOL_CALL
        self._touch()

    def update_tool_call(
        self,
        name: str,
        status: str,
        result: str | None = None,
        duration: float | None = None,
        tool_call_id: str | None = None,
    ) -> None:
        """Update a tool call status."""
        for tool in reversed(self.state.tool_calls):
            if tool_call_id and tool.id == tool_call_id:
                tool.status = status
                tool.result = result
                tool.duration = duration
                break
            if tool.name == name and tool.status == "running" and tool_call_id is None:
                tool.status = status
                tool.result = result
                tool.duration = duration
                break
        
        # Check if any tools still running
        if self.state.activity == ActivityType.TOOL_CALL:
            if not any(t.status == "running" for t in self.state.tool_calls):
                self.state.activity = ActivityType.THINKING
        self._touch()

    def increment_retries(self) -> None:
        """Increment retry counter."""
        self.state.retries += 1
        self._touch()

    def set_token_usage(self, input_tokens: int = 0, output_tokens: int = 0, reasoning_tokens: int = 0) -> None:
        """Set token usage statistics."""
        self.state.token_usage = {
            "input": input_tokens,
            "output": output_tokens,
            "reasoning": reasoning_tokens,
        }
        self.state.input_tokens = input_tokens
        self.state.output_tokens = output_tokens + reasoning_tokens
        self._touch()

    def set_final_content(self, message: str, reasoning: str | None = None) -> None:
        """Set final content and mark as done."""
        if message:
            self.state.message = message
        if reasoning and self.show_reasoning:
            self.state.reasoning = reasoning
        self.state.activity = ActivityType.DONE
        self._touch()

    def add_user_message(self, content: str) -> None:
        """Add a user message to history."""
        self.state.history.append(HistoryEntry(role="user", content=content))
        self._touch()

    def finalize_assistant_response(self) -> None:
        """Finalize and store assistant response in history."""
        if self.state.message:
            self.state.history.append(HistoryEntry(
                role="assistant",
                content=self.state.message,
                reasoning=self.state.reasoning if self.show_reasoning else None,
                tool_calls=list(self.state.tool_calls),
            ))
        self._touch()

    def consume_dirty(self) -> bool:
        """Check and clear dirty flag."""
        if self._dirty:
            self._dirty = False
            return True
        return False

    def _touch(self) -> None:
        """Mark state as updated."""
        self.state.last_update = time.time()
        self._dirty = True

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
    ascii_icons: bool = False,
) -> None:
    """Print the welcome banner."""
    icon_set = ASCIIIcons if ascii_icons else Icons
    
    welcome_text = Text()
    welcome_text.append(f"{icon_set.ROBOT} Copex", style=Theme.HEADER)
    welcome_text.append(" - Copilot Extended\n\n", style="dim")
    welcome_text.append("Model: ", style="dim")
    welcome_text.append(f"{model}\n", style=Theme.PRIMARY)
    welcome_text.append("Reasoning: ", style="dim")
    welcome_text.append(f"{reasoning}\n\n", style=Theme.PRIMARY)
    welcome_text.append("Type ", style="dim")
    welcome_text.append("exit", style="bold")
    welcome_text.append(" to quit, ", style="dim")
    welcome_text.append("new", style="bold")
    welcome_text.append(" for fresh session\n", style="dim")
    welcome_text.append("Press ", style="dim")
    welcome_text.append("Esc+Enter", style="bold")
    welcome_text.append(" for newline", style="dim")
    
    console.print()
    console.print(Panel(
        welcome_text,
        border_style=Theme.BORDER_ACTIVE,
        box=ROUNDED,
        padding=(0, 2),
    ))
    console.print()


def print_user_prompt(console: Console, prompt: str, *, ascii_icons: bool = False) -> None:
    """Print the user's prompt."""
    icon_set = ASCIIIcons if ascii_icons else Icons
    
    console.print()
    text = Text()
    text.append(f"{icon_set.ARROW_RIGHT} ", style=f"bold {Theme.SUCCESS}")
    
    # Truncate long prompts
    if len(prompt) > 200:
        text.append(prompt[:200] + "...", style="bold")
    else:
        text.append(prompt, style="bold")
    
    console.print(text)
    console.print()


def print_error(console: Console, error: str, *, ascii_icons: bool = False) -> None:
    """Print an error message."""
    icon_set = ASCIIIcons if ascii_icons else Icons
    console.print(Panel(
        Text(f"{icon_set.ERROR} {error}", style=Theme.ERROR),
        border_style=Theme.ERROR,
        title="Error",
        title_align="left",
        box=ROUNDED,
    ))


def print_retry(
    console: Console,
    attempt: int,
    max_attempts: int,
    error: str,
    *,
    ascii_icons: bool = False,
) -> None:
    """Print a retry notification."""
    icon_set = ASCIIIcons if ascii_icons else Icons
    console.print(Text(
        f" {icon_set.WARNING} Retry {attempt}/{max_attempts}: {error[:50]}...",
        style=Theme.WARNING,
    ))


def print_tool_call(
    console: Console,
    name: str,
    args: dict[str, Any] | None = None,
    *,
    ascii_icons: bool = False,
) -> None:
    """Print a tool call notification."""
    tool = ToolCallInfo(name=name, arguments=args or {})
    icon_set = ASCIIIcons if ascii_icons else Icons
    
    text = Text()
    text.append(f" {tool.icon(icon_set)} ", style=Theme.WARNING)
    text.append(name, style=f"bold {Theme.WARNING}")
    
    if args:
        for key in ("path", "command", "pattern"):
            if key in args:
                val = str(args[key])[:40]
                text.append(f" {key}=", style="dim")
                text.append(val, style="dim italic")
                break
    
    console.print(text)


def print_tool_result(
    console: Console,
    name: str,
    success: bool,
    duration: float | None = None,
    *,
    ascii_icons: bool = False,
) -> None:
    """Print a tool result notification."""
    icon_set = ASCIIIcons if ascii_icons else Icons
    icon = icon_set.DONE if success else icon_set.ERROR
    style = Theme.SUCCESS if success else Theme.ERROR
    
    text = Text()
    text.append(f"   {icon} ", style=style)
    text.append(name, style=f"bold {style}")
    if duration:
        text.append(f" ({duration:.1f}s)", style="dim")
    
    console.print(text)
