"""Beautiful interactive mode for Copex - inspired by Claude Code, OpenCode, and Aider.

This module provides a polished interactive chat experience with:
- Clean prompt showing model name
- Animated spinner during thinking with elapsed time
- Smooth streaming with live markdown rendering
- Compact, collapsible tool call display
- Persistent status bar with session info
- Clean stats line after each response
"""

from __future__ import annotations

import asyncio
import io
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import ANSI, FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from rich.box import ROUNDED
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.style import Style as RichStyle
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from copex.client import Copex, StreamChunk
    from copex.config import CopexConfig
    from copex.models import Model, ReasoningEffort


# ═══════════════════════════════════════════════════════════════════════════════
# Theme and Icons
# ═══════════════════════════════════════════════════════════════════════════════


class Colors:
    """Modern color palette."""

    # Brand
    PRIMARY = "#7aa2f7"  # Soft blue
    SECONDARY = "#9ece6a"  # Green
    ACCENT = "#bb9af7"  # Purple

    # Status
    SUCCESS = "#9ece6a"
    WARNING = "#e0af68"
    ERROR = "#f7768e"
    INFO = "#7dcfff"

    # Text
    TEXT = "#c0caf5"
    TEXT_MUTED = "#565f89"
    TEXT_DIM = "#3b4261"

    # Borders
    BORDER = "#3b4261"
    BORDER_ACTIVE = "#7aa2f7"


class Icons:
    """Unicode icons."""

    PROMPT = "❯"
    THINKING = "◐"
    DONE = "✓"
    ERROR = "✗"
    TOOL = "⚡"
    BRAIN = "🧠"
    ROBOT = "🤖"
    CLOCK = "⏱"
    ARROW = "→"
    CHEVRON_RIGHT = "›"
    CHEVRON_DOWN = "▾"
    SPARKLE = "✨"


class Spinners:
    """Spinner animation frames."""

    DOTS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    PULSE = ["○", "◔", "◑", "◕", "●", "◕", "◑", "◔"]
    ARC = ["◜", "◠", "◝", "◞", "◡", "◟"]


# ═══════════════════════════════════════════════════════════════════════════════
# Tool Call State
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ToolCall:
    """Represents a tool call during streaming."""

    tool_id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str | None = None
    status: str = "running"  # running, success, error
    started_at: float = field(default_factory=time.time)
    duration: float | None = None

    @property
    def elapsed(self) -> float:
        if self.duration is not None:
            return self.duration
        return time.time() - self.started_at

    @property
    def icon(self) -> str:
        """Get tool-specific icon."""
        name_lower = self.name.lower()
        if "read" in name_lower or "view" in name_lower:
            return "📖"
        elif "write" in name_lower or "edit" in name_lower:
            return "📝"
        elif "shell" in name_lower or "bash" in name_lower or "exec" in name_lower:
            return "💻"
        elif "search" in name_lower or "grep" in name_lower:
            return "🔍"
        elif "web" in name_lower or "fetch" in name_lower:
            return "🌐"
        return "⚡"


# ═══════════════════════════════════════════════════════════════════════════════
# Streaming UI State
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class StreamState:
    """Current state of streaming response."""

    phase: str = "idle"  # idle, thinking, reasoning, responding, tool_call, done, error
    message: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    retries: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    model: str = ""

    # Animation state
    _frame: int = 0
    _last_frame_time: float = 0.0

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

    def advance_frame(self) -> None:
        """Advance animation frame at ~10fps."""
        now = time.time()
        if now - self._last_frame_time >= 0.1:
            self._frame = (self._frame + 1) % len(Spinners.DOTS)
            self._last_frame_time = now

    @property
    def spinner(self) -> str:
        return Spinners.DOTS[self._frame]


# ═══════════════════════════════════════════════════════════════════════════════
# Live Streaming Renderer
# ═══════════════════════════════════════════════════════════════════════════════


class StreamRenderer:
    """Renders streaming response with beautiful live updates."""

    def __init__(self, console: Console, state: StreamState, compact: bool = False):
        self.console = console
        self.state = state
        self.compact = compact
        self._show_reasoning = True
        self._expand_tools = False
        self._max_tool_preview = 3

    def _format_elapsed(self) -> Text:
        """Format elapsed time with icon."""
        text = Text()
        text.append(f"{Icons.CLOCK} ", style=Colors.TEXT_MUTED)
        text.append(self.state.elapsed_str, style=Colors.TEXT_MUTED)
        return text

    def _build_status_line(self) -> Text:
        """Build the status line showing current activity."""
        text = Text()
        spinner = self.state.spinner
        self.state.advance_frame()

        if self.state.phase == "thinking":
            text.append(f" {spinner} ", style=f"bold {Colors.PRIMARY}")
            text.append("Thinking", style=Colors.PRIMARY)
            text.append("...", style=Colors.TEXT_MUTED)
        elif self.state.phase == "reasoning":
            text.append(f" {spinner} ", style=f"bold {Colors.ACCENT}")
            text.append("Reasoning", style=Colors.ACCENT)
            text.append("...", style=Colors.TEXT_MUTED)
        elif self.state.phase == "responding":
            text.append(f" {spinner} ", style=f"bold {Colors.SUCCESS}")
            text.append("Responding", style=Colors.SUCCESS)
            text.append("...", style=Colors.TEXT_MUTED)
        elif self.state.phase == "tool_call":
            running = sum(1 for t in self.state.tool_calls if t.status == "running")
            text.append(f" {spinner} ", style=f"bold {Colors.WARNING}")
            if running == 1:
                tool = next(t for t in self.state.tool_calls if t.status == "running")
                text.append(f"Running {tool.name}", style=Colors.WARNING)
            else:
                text.append(f"Running {running} tools", style=Colors.WARNING)
            text.append("...", style=Colors.TEXT_MUTED)
        elif self.state.phase == "done":
            text.append(f" {Icons.DONE} ", style=f"bold {Colors.SUCCESS}")
            text.append("Complete", style=Colors.SUCCESS)
        elif self.state.phase == "error":
            text.append(f" {Icons.ERROR} ", style=f"bold {Colors.ERROR}")
            text.append("Error", style=Colors.ERROR)
        else:
            text.append("   ", style="")
            text.append("Ready", style=Colors.TEXT_MUTED)

        # Add elapsed time on the right
        text.append("  ")
        text.append_text(self._format_elapsed())

        # Add model name
        if self.state.model:
            text.append("  │  ", style=Colors.TEXT_DIM)
            text.append(self.state.model, style=Colors.TEXT_MUTED)

        return text

    def _build_tool_calls_compact(self) -> Text | None:
        """Build compact tool calls summary."""
        if not self.state.tool_calls:
            return None

        text = Text()
        running = [t for t in self.state.tool_calls if t.status == "running"]
        completed = [t for t in self.state.tool_calls if t.status != "running"]

        # Show running tools first
        for tool in running[-self._max_tool_preview :]:
            spinner = self.state.spinner
            text.append(f"\n  {spinner} ", style=f"bold {Colors.WARNING}")
            text.append(tool.icon + " ", style=Colors.WARNING)
            text.append(tool.name, style=Colors.WARNING)
            # Show key argument
            if tool.arguments:
                arg_preview = self._format_arg_preview(tool.arguments)
                if arg_preview:
                    text.append(f" {arg_preview}", style=Colors.TEXT_MUTED)
            text.append(f"  {tool.elapsed:.1f}s", style=Colors.TEXT_MUTED)

        # Show completed tools (last few)
        for tool in completed[-self._max_tool_preview :]:
            icon = Icons.DONE if tool.status == "success" else Icons.ERROR
            style = Colors.SUCCESS if tool.status == "success" else Colors.ERROR
            text.append(f"\n  {icon} ", style=style)
            text.append(tool.icon + " ", style=Colors.TEXT_MUTED)
            text.append(tool.name, style=Colors.TEXT_MUTED)
            if tool.duration:
                text.append(f"  {tool.duration:.1f}s", style=Colors.TEXT_MUTED)

        # Show count if more
        total = len(self.state.tool_calls)
        shown = min(total, self._max_tool_preview * 2)
        if total > shown:
            text.append(f"\n  ... and {total - shown} more", style=Colors.TEXT_MUTED)

        return text

    def _format_arg_preview(self, args: dict[str, Any], max_len: int = 40) -> str:
        """Format a brief argument preview."""
        for key in ("path", "file", "command", "pattern", "query"):
            if key in args and args[key]:
                val = str(args[key])
                if len(val) > max_len:
                    val = val[: max_len - 3] + "..."
                return f"{key}={val}"
        return ""

    def _build_reasoning_panel(self) -> Panel | None:
        """Build reasoning panel if available."""
        if not self.state.reasoning or not self._show_reasoning:
            return None

        # Truncate for live display
        content = self.state.reasoning
        max_chars = 600 if not self.compact else 300
        if len(content) > max_chars:
            content = "..." + content[-max_chars:]

        text = Text(content, style=f"italic {Colors.TEXT_MUTED}")

        # Add cursor if still reasoning
        if self.state.phase == "reasoning":
            text.append("▌", style=f"bold {Colors.ACCENT}")

        border = Colors.BORDER_ACTIVE if self.state.phase == "reasoning" else Colors.BORDER
        return Panel(
            text,
            title=f"[{Colors.ACCENT}]{Icons.BRAIN} Reasoning[/{Colors.ACCENT}]",
            title_align="left",
            border_style=border,
            padding=(0, 1),
            box=ROUNDED,
        )

    def _build_message_panel(self) -> Panel | None:
        """Build message panel with streaming content."""
        if not self.state.message:
            return None

        # During streaming, show raw text for performance
        if self.state.phase in ("responding", "thinking", "reasoning", "tool_call"):
            content = Text(self.state.message, style=Colors.TEXT)
            # Blinking cursor
            if self.state.phase == "responding":
                cursors = ["▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]
                cursor = cursors[self.state._frame % len(cursors)]
                content.append(cursor, style=f"bold {Colors.PRIMARY}")
        else:
            # Final render with markdown
            content = Markdown(self.state.message)

        border = Colors.BORDER_ACTIVE if self.state.phase == "responding" else Colors.BORDER
        return Panel(
            content,
            title=f"[{Colors.PRIMARY}]{Icons.ROBOT} Response[/{Colors.PRIMARY}]",
            title_align="left",
            border_style=border,
            padding=(0, 1),
            box=ROUNDED,
        )

    def build(self) -> Group:
        """Build the complete live display."""
        elements = []

        # Status line (always shown)
        elements.append(self._build_status_line())

        # Tool calls (compact inline)
        tools_text = self._build_tool_calls_compact()
        if tools_text:
            elements.append(tools_text)
            elements.append(Text())  # Spacer

        # Reasoning panel (if available)
        if not self.compact:
            reasoning_panel = self._build_reasoning_panel()
            if reasoning_panel:
                elements.append(Text())  # Spacer
                elements.append(reasoning_panel)

        # Message panel
        message_panel = self._build_message_panel()
        if message_panel:
            elements.append(Text())  # Spacer
            elements.append(message_panel)

        return Group(*elements)


# ═══════════════════════════════════════════════════════════════════════════════
# Final Output Renderer
# ═══════════════════════════════════════════════════════════════════════════════


def render_final_output(
    console: Console,
    state: StreamState,
    show_reasoning: bool = True,
    show_stats: bool = True,
) -> None:
    """Render the final formatted output after streaming completes."""
    elements = []

    # Reasoning panel (collapsible style)
    if state.reasoning and show_reasoning:
        console.print(
            Panel(
                Markdown(state.reasoning),
                title=f"[{Colors.ACCENT}]{Icons.BRAIN} Reasoning[/{Colors.ACCENT}]",
                title_align="left",
                border_style=Colors.BORDER,
                padding=(0, 1),
                box=ROUNDED,
            )
        )
        console.print()

    # Main response with markdown
    if state.message:
        console.print(
            Panel(
                Markdown(state.message),
                title=f"[{Colors.PRIMARY}]{Icons.ROBOT} Response[/{Colors.PRIMARY}]",
                title_align="left",
                border_style=Colors.BORDER_ACTIVE,
                padding=(0, 1),
                box=ROUNDED,
            )
        )

    # Stats line
    if show_stats:
        console.print(_build_stats_line(state))
    console.print()


def _build_stats_line(state: StreamState) -> Text:
    """Build a clean stats line."""
    text = Text()

    # Elapsed time
    text.append(f"{Icons.CLOCK} ", style=Colors.TEXT_MUTED)
    text.append(state.elapsed_str, style=Colors.TEXT_MUTED)

    # Token counts
    if state.input_tokens is not None or state.output_tokens is not None:
        text.append("  │  ", style=Colors.TEXT_DIM)
        inp = state.input_tokens or 0
        out = state.output_tokens or 0
        text.append(f"{inp:,}", style=Colors.INFO)
        text.append(" in / ", style=Colors.TEXT_MUTED)
        text.append(f"{out:,}", style=Colors.INFO)
        text.append(" out", style=Colors.TEXT_MUTED)

    # Tool calls
    if state.tool_calls:
        successful = sum(1 for t in state.tool_calls if t.status == "success")
        failed = sum(1 for t in state.tool_calls if t.status == "error")
        text.append("  │  ", style=Colors.TEXT_DIM)
        text.append(f"{Icons.TOOL} ", style=Colors.WARNING)
        if failed:
            text.append(f"{successful} ok, {failed} failed", style=Colors.WARNING)
        else:
            text.append(f"{len(state.tool_calls)} tools", style=Colors.TEXT_MUTED)

    # Retries
    if state.retries:
        text.append("  │  ", style=Colors.TEXT_DIM)
        text.append(f"{state.retries} retries", style=Colors.WARNING)

    return text


# ═══════════════════════════════════════════════════════════════════════════════
# Interactive Session
# ═══════════════════════════════════════════════════════════════════════════════


class SlashCompleter(Completer):
    """Completer for slash commands."""

    COMMANDS = [
        "/model",
        "/models",
        "/reasoning",
        "/new",
        "/status",
        "/help",
        "/clear",
        "exit",
        "quit",
    ]

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.lstrip()
        if not text.startswith("/") and text not in ("exit", "quit", "ex", "qu"):
            return
        for cmd in self.COMMANDS:
            if cmd.lower().startswith(text.lower()):
                yield Completion(cmd, start_position=-len(text))


def _build_prompt_message(model: str, reasoning: str) -> FormattedText:
    """Build the prompt message showing model info."""
    return FormattedText(
        [
            ("class:model", f"[{model}"),
            ("class:sep", "/"),
            ("class:reasoning", f"{reasoning}"),
            ("class:model", "] "),
            ("class:prompt", f"{Icons.PROMPT} "),
        ]
    )


PROMPT_STYLE = Style.from_dict(
    {
        "model": "#7aa2f7 bold",
        "sep": "#565f89",
        "reasoning": "#bb9af7",
        "prompt": "#9ece6a bold",
        "continuation": "#565f89",
    }
)


async def run_interactive(config: "CopexConfig") -> None:
    """Run the beautiful interactive chat loop."""
    from pathlib import Path

    from copex import __version__
    from copex.client import Copex, StreamChunk
    from copex.config import save_last_model
    from copex.models import Model, ReasoningEffort, normalize_reasoning_effort, parse_reasoning_effort

    console = Console()
    client = Copex(config)
    await client.start()

    # Build prompt session
    history_path = Path.home() / ".copex" / "history"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    bindings = KeyBindings()

    @bindings.add("enter")
    def handle_enter(event):
        buffer = event.app.current_buffer
        if buffer.document.text.strip():
            buffer.validate_and_handle()
        else:
            buffer.reset()

    @bindings.add("escape", "enter")
    def handle_newline(event):
        event.app.current_buffer.insert_text("\n")

    session: PromptSession = PromptSession(
        message=_build_prompt_message(
            config.model.value.split("/")[-1],  # Just model name, not provider
            config.reasoning_effort.value,
        ),
        history=FileHistory(str(history_path)),
        key_bindings=bindings,
        completer=SlashCompleter(),
        complete_while_typing=True,
        multiline=True,
        prompt_continuation=lambda width, ln, is_soft: "  ... ",
        style=PROMPT_STYLE,
    )

    # Print welcome banner
    _print_welcome(console, config.model.value, config.reasoning_effort.value, __version__)

    try:
        while True:
            try:
                prompt = await session.prompt_async()
            except EOFError:
                break
            except KeyboardInterrupt:
                break

            prompt = prompt.strip()
            if not prompt:
                continue

            # Handle commands
            cmd = prompt.lower()
            if cmd in ("exit", "quit"):
                break

            if cmd in ("/new", "new"):
                client.new_session()
                console.print(f"[{Colors.SUCCESS}]{Icons.DONE} New session started[/{Colors.SUCCESS}]")
                continue

            if cmd in ("/help", "help"):
                _print_help(console)
                continue

            if cmd in ("/status", "status"):
                _print_status(console, client)
                continue

            if cmd in ("/models", "models"):
                _print_models(console, client.config.model)
                continue

            if cmd in ("/clear", "clear"):
                console.clear()
                continue

            if cmd.startswith("/model "):
                model_name = prompt.split(maxsplit=1)[1].strip()
                try:
                    new_model = Model(model_name)
                    client.config.model = new_model
                    save_last_model(new_model)
                    normalized, warning = normalize_reasoning_effort(new_model, client.config.reasoning_effort)
                    if warning:
                        console.print(f"[{Colors.WARNING}]{warning}[/{Colors.WARNING}]")
                    client.config.reasoning_effort = normalized
                    client.new_session()
                    session.message = _build_prompt_message(
                        new_model.value.split("/")[-1],
                        normalized.value,
                    )
                    console.print(f"[{Colors.SUCCESS}]{Icons.DONE} Switched to {new_model.value}[/{Colors.SUCCESS}]")
                except ValueError:
                    console.print(f"[{Colors.ERROR}]Unknown model: {model_name}[/{Colors.ERROR}]")
                continue

            if cmd.startswith("/reasoning "):
                level = prompt.split(maxsplit=1)[1].strip()
                try:
                    requested = parse_reasoning_effort(level)
                    if requested is None:
                        raise ValueError(level)
                    normalized, warning = normalize_reasoning_effort(client.config.model, requested)
                    if warning:
                        console.print(f"[{Colors.WARNING}]{warning}[/{Colors.WARNING}]")
                    client.config.reasoning_effort = normalized
                    client.new_session()
                    session.message = _build_prompt_message(
                        client.config.model.value.split("/")[-1],
                        normalized.value,
                    )
                    console.print(f"[{Colors.SUCCESS}]{Icons.DONE} Reasoning set to {normalized.value}[/{Colors.SUCCESS}]")
                except ValueError:
                    valid = ", ".join(r.value for r in ReasoningEffort)
                    console.print(f"[{Colors.ERROR}]Invalid level. Valid: {valid}[/{Colors.ERROR}]")
                continue

            # Send message
            await _stream_message(console, client, prompt)

    except KeyboardInterrupt:
        console.print(f"\n[{Colors.WARNING}]Goodbye!{Icons.SPARKLE}[/{Colors.WARNING}]")
    finally:
        await client.stop()


async def _stream_message(console: Console, client: "Copex", prompt: str) -> None:
    """Stream a message with beautiful live updates."""
    from copex.client import StreamChunk

    state = StreamState(model=client.config.model.value)
    state.phase = "thinking"
    renderer = StreamRenderer(console, state)
    refresh_stop = asyncio.Event()

    def on_chunk(chunk: StreamChunk) -> None:
        if chunk.type == "message":
            if chunk.is_final:
                pass  # Content already captured
            else:
                state.message += chunk.delta
                if state.phase != "responding":
                    state.phase = "responding"
        elif chunk.type == "reasoning":
            if not chunk.is_final:
                state.reasoning += chunk.delta
                if state.phase != "reasoning":
                    state.phase = "reasoning"
        elif chunk.type == "tool_call":
            tool = ToolCall(
                tool_id=chunk.tool_id or "",
                name=chunk.tool_name or "unknown",
                arguments=chunk.tool_args or {},
            )
            state.tool_calls.append(tool)
            state.phase = "tool_call"
        elif chunk.type == "tool_result":
            for tool in reversed(state.tool_calls):
                if (chunk.tool_id and tool.tool_id == chunk.tool_id) or tool.name == chunk.tool_name:
                    if tool.status == "running":
                        tool.status = "success" if chunk.tool_success is not False else "error"
                        tool.result = chunk.tool_result
                        tool.duration = chunk.tool_duration or tool.elapsed
                        break
            # Check if any tools still running
            if not any(t.status == "running" for t in state.tool_calls):
                if state.message:
                    state.phase = "responding"
                else:
                    state.phase = "thinking"
        elif chunk.type == "system":
            state.retries += 1

    async def refresh_loop(live: Live) -> None:
        while not refresh_stop.is_set():
            live.update(renderer.build())
            await asyncio.sleep(0.1)

    # Print user prompt
    console.print()
    user_text = Text()
    user_text.append(f"{Icons.PROMPT} ", style=f"bold {Colors.SECONDARY}")
    display_prompt = prompt[:200] + "..." if len(prompt) > 200 else prompt
    user_text.append(display_prompt, style="bold")
    console.print(user_text)
    console.print()

    # Stream with live display
    with Live(console=console, refresh_per_second=10, transient=True) as live:
        live.update(renderer.build())
        refresh_task = asyncio.create_task(refresh_loop(live))
        try:
            response = await client.send(prompt, on_chunk=on_chunk)
            # Use streamed content if available
            if not state.message and response.content:
                state.message = response.content
            if not state.reasoning and response.reasoning:
                state.reasoning = response.reasoning
            state.input_tokens = response.prompt_tokens
            state.output_tokens = response.completion_tokens
            state.retries = response.retries
            state.phase = "done"
        except Exception as e:
            state.phase = "error"
            console.print(f"[{Colors.ERROR}]{Icons.ERROR} Error: {e}[/{Colors.ERROR}]")
            raise
        finally:
            refresh_stop.set()
            try:
                await refresh_task
            except asyncio.CancelledError:
                pass

    # Render final output
    render_final_output(console, state)


def _print_welcome(console: Console, model: str, reasoning: str, version: str) -> None:
    """Print welcome banner."""
    console.print()
    banner = Text()
    banner.append("  ╭──────────────────────────────────────╮\n", style=Colors.PRIMARY)
    banner.append("  │ ", style=Colors.PRIMARY)
    banner.append(f"  {Icons.ROBOT} Copex", style=f"bold {Colors.PRIMARY}")
    banner.append(" - Copilot Extended", style=Colors.TEXT_MUTED)
    banner.append("     │\n", style=Colors.PRIMARY)
    banner.append("  ╰──────────────────────────────────────╯\n", style=Colors.PRIMARY)
    console.print(banner)

    info = Text()
    info.append("  Model: ", style=Colors.TEXT_MUTED)
    info.append(model, style=f"bold {Colors.PRIMARY}")
    info.append("  │  ", style=Colors.TEXT_DIM)
    info.append("Reasoning: ", style=Colors.TEXT_MUTED)
    info.append(reasoning, style=f"bold {Colors.ACCENT}")
    info.append("  │  ", style=Colors.TEXT_DIM)
    info.append(f"v{version}", style=Colors.TEXT_MUTED)
    console.print(info)
    console.print()

    help_text = Text()
    help_text.append("  ", style="")
    for key, desc in [("Esc+Enter", "newline"), ("/help", "commands"), ("exit", "quit")]:
        help_text.append(key, style="bold")
        help_text.append(f" {desc}  ", style=Colors.TEXT_MUTED)
    console.print(help_text)
    console.print()


def _print_help(console: Console) -> None:
    """Print help message."""
    console.print()
    cmds = [
        ("/model <name>", "Change model"),
        ("/reasoning <level>", "Change reasoning (low/medium/high/xhigh)"),
        ("/models", "List available models"),
        ("/new", "Start new session"),
        ("/status", "Show current settings"),
        ("/clear", "Clear screen"),
        ("exit", "Quit"),
    ]
    for cmd, desc in cmds:
        console.print(f"  [{Colors.PRIMARY}]{cmd:20}[/{Colors.PRIMARY}] {desc}")
    console.print()


def _print_status(console: Console, client: "Copex") -> None:
    """Print current status."""
    console.print()
    console.print(f"  Model:     [{Colors.PRIMARY}]{client.config.model.value}[/{Colors.PRIMARY}]")
    console.print(f"  Reasoning: [{Colors.ACCENT}]{client.config.reasoning_effort.value}[/{Colors.ACCENT}]")
    console.print()


def _print_models(console: Console, current: "Model") -> None:
    """Print available models."""
    from copex.models import Model

    console.print()
    for model in Model:
        marker = f"[{Colors.SUCCESS}]→[/{Colors.SUCCESS}] " if model == current else "  "
        style = f"bold {Colors.PRIMARY}" if model == current else Colors.TEXT_MUTED
        console.print(f"  {marker}[{style}]{model.value}[/{style}]")
    console.print()
