"""Beautiful interactive mode for Copex - inspired by GitHub Copilot CLI.

This module provides a polished interactive chat experience with:
- Clean banner showing working directory and model
- Simple `>` prompt with Tab completion for commands
- Animated spinner during thinking with elapsed time
- Smooth streaming with live markdown rendering
- Tool call display with ✓/✗ status and duration
- Clean stats line (API time, tokens, tools)
- Keyboard shortcuts (Ctrl+L, ↑/↓ history, Esc+Enter multiline)
- Intuitive commands (/model, /reasoning, /models, /status, /help)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

if TYPE_CHECKING:
    from copex.client import Copex, StreamChunk
    from copex.config import CopexConfig
    from copex.models import Model


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
    """Unicode icons used in UI."""

    DONE = "✓"
    ERROR = "✗"
    CLOCK = "⏱"


class Spinners:
    """Spinner animation frames."""

    DOTS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


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
        """Build the status line - minimal and calm."""
        text = Text()
        spinner = self.state.spinner
        self.state.advance_frame()

        # Just show spinner + brief phase indicator, all in dim muted style
        if self.state.phase == "thinking":
            text.append(f"  {spinner} ", style=Colors.TEXT_MUTED)
        elif self.state.phase == "reasoning":
            text.append(f"  {spinner} ", style=Colors.TEXT_MUTED)
        elif self.state.phase == "responding":
            text.append(f"  {spinner} ", style=Colors.TEXT_MUTED)
        elif self.state.phase == "tool_call":
            sum(1 for t in self.state.tool_calls if t.status == "running")
            tool = next((t for t in self.state.tool_calls if t.status == "running"), None)
            text.append(f"  {spinner} ", style=Colors.TEXT_MUTED)
            if tool:
                text.append(f"{tool.name}", style=Colors.TEXT_MUTED)
        elif self.state.phase == "done":
            text.append(f"  {Icons.DONE} ", style=Colors.TEXT_MUTED)
        elif self.state.phase == "error":
            text.append(f"  {Icons.ERROR} ", style=Colors.ERROR)
        else:
            text.append("    ", style="")

        # Add elapsed time
        text.append_text(self._format_elapsed())

        return text

    def _build_tool_calls_compact(self) -> Text | None:
        """Build compact tool calls - clean like Copilot CLI."""
        if not self.state.tool_calls:
            return None

        text = Text()
        running = [t for t in self.state.tool_calls if t.status == "running"]
        completed = [t for t in self.state.tool_calls if t.status != "running"]

        # Show running tools with spinner and key info
        for tool in running[-self._max_tool_preview :]:
            spinner = self.state.spinner
            text.append(f"{spinner} ", style=Colors.TEXT_MUTED)
            text.append(tool.name, style=Colors.INFO)
            # Show key argument inline
            if tool.arguments:
                arg_preview = self._format_arg_preview(tool.arguments)
                if arg_preview:
                    text.append(f" {arg_preview}", style=Colors.TEXT_DIM)
            text.append("\n")

        # Show completed tools - checkmark or x
        for tool in completed[-self._max_tool_preview :]:
            if tool.status == "success":
                text.append("✓ ", style=Colors.SUCCESS)
            else:
                text.append("✗ ", style=Colors.ERROR)
            text.append(tool.name, style=Colors.TEXT_DIM)
            if tool.duration:
                text.append(f" ({tool.duration:.1f}s)", style=Colors.TEXT_DIM)
            text.append("\n")

        # Show count if more
        total = len(self.state.tool_calls)
        shown = min(total, self._max_tool_preview * 2)
        if total > shown:
            text.append(f"+{total - shown} more\n", style=Colors.TEXT_DIM)

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

    def _build_reasoning_panel(self) -> Text | None:
        """Build reasoning text (no panel, minimal style)."""
        if not self.state.reasoning or not self._show_reasoning:
            return None

        # Truncate for live display
        content = self.state.reasoning
        max_chars = 600 if not self.compact else 300
        if len(content) > max_chars:
            content = "..." + content[-max_chars:]

        text = Text()
        text.append(content, style=f"dim italic {Colors.TEXT_MUTED}")

        # Add cursor if still reasoning
        if self.state.phase == "reasoning":
            text.append("▌", style=Colors.TEXT_MUTED)

        return text

    def _build_message_panel(self) -> Text | Markdown | None:
        """Build message content (no panel, clean text)."""
        if not self.state.message:
            return None

        # During streaming, show raw text for performance
        if self.state.phase in ("responding", "thinking", "reasoning", "tool_call"):
            content = Text(self.state.message, style=Colors.TEXT)
            # Simple blinking cursor during streaming
            if self.state.phase == "responding":
                content.append("▌", style=f"bold {Colors.PRIMARY}")
            return content
        else:
            # Final render with markdown
            return Markdown(self.state.message)

    def build(self) -> Group:
        """Build the complete live display - minimal and clean."""
        elements = []

        # Tool calls only when running (minimal)
        tools_text = self._build_tool_calls_compact()
        if tools_text:
            elements.append(tools_text)

        # Reasoning (if available and not compact)
        if not self.compact:
            reasoning = self._build_reasoning_panel()
            if reasoning:
                elements.append(Text())  # Spacer
                elements.append(reasoning)

        # Message content
        message = self._build_message_panel()
        if message:
            if elements:
                elements.append(Text())  # Spacer
            elements.append(message)

        # If nothing to show yet, show spinner with elapsed time
        if not elements and self.state.phase in ("thinking", "reasoning"):
            spinner = self.state.spinner
            self.state.advance_frame()
            wait_text = Text()
            wait_text.append(f"{spinner} ", style=Colors.TEXT_MUTED)
            wait_text.append("Thinking", style=f"dim {Colors.TEXT_MUTED}")
            wait_text.append(f" ({self.state.elapsed_str})", style=Colors.TEXT_DIM)
            elements.append(wait_text)

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
    # Reasoning (subtle, no emoji)
    if state.reasoning and show_reasoning:
        text = Text(state.reasoning, style=f"dim italic {Colors.TEXT_MUTED}")
        console.print(text)
        console.print()

    # Main response with markdown (no panel)
    if state.message:
        console.print(Markdown(state.message))
        console.print()

    # Stats line
    if show_stats:
        console.print(_build_stats_line(state))


def _build_stats_line(state: StreamState) -> Text:
    """Build a clean stats line - styled like Copilot CLI."""
    text = Text()

    # Header label
    text.append("API time: ", style=Colors.TEXT_DIM)
    text.append(state.elapsed_str, style=Colors.TEXT_MUTED)

    # Token counts (formatted nicely)
    if state.input_tokens is not None or state.output_tokens is not None:
        text.append("  Tokens: ", style=Colors.TEXT_DIM)
        inp = state.input_tokens or 0
        out = state.output_tokens or 0
        text.append(f"{inp:,} in, {out:,} out", style=Colors.TEXT_MUTED)

    # Tool calls
    if state.tool_calls:
        failed = sum(1 for t in state.tool_calls if t.status == "error")
        text.append("  Tools: ", style=Colors.TEXT_DIM)
        if failed:
            text.append(f"{len(state.tool_calls)} ({failed} failed)", style=Colors.WARNING)
        else:
            text.append(str(len(state.tool_calls)), style=Colors.TEXT_MUTED)

    # Retries
    if state.retries:
        text.append("  Retries: ", style=Colors.TEXT_DIM)
        text.append(str(state.retries), style=Colors.WARNING)

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
    """Build the prompt message - clean like Copilot CLI."""
    return FormattedText(
        [
            ("class:prompt", "> "),
        ]
    )


PROMPT_STYLE = Style.from_dict(
    {
        "prompt": "#7aa2f7 bold",
        "continuation": "#565f89",
    }
)


async def run_interactive(config: CopexConfig) -> None:
    """Run the beautiful interactive chat loop."""
    from pathlib import Path

    from copex import __version__
    from copex.client import Copex
    from copex.config import save_last_model
    from copex.models import (
        Model,
        ReasoningEffort,
        normalize_reasoning_effort,
        parse_reasoning_effort,
    )

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

    @bindings.add("c-l")
    def handle_clear(event):
        console.clear()

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
        prompt_continuation=lambda width, ln, is_soft: ". ",
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
            if cmd in ("exit", "quit", "q"):
                break

            if cmd in ("/new", "new"):
                client.new_session()
                console.print(f"[{Colors.SUCCESS}]New session started[/{Colors.SUCCESS}]")
                continue

            if cmd in ("/help", "help", "?"):
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
                    normalized, warning = normalize_reasoning_effort(
                        new_model, client.config.reasoning_effort
                    )
                    if warning:
                        console.print(f"[{Colors.WARNING}]{warning}[/{Colors.WARNING}]")
                    client.config.reasoning_effort = normalized
                    client.new_session()
                    session.message = _build_prompt_message(
                        new_model.value.split("/")[-1],
                        normalized.value,
                    )
                    console.print(
                        f"[{Colors.SUCCESS}]Switched to {new_model.value}[/{Colors.SUCCESS}]"
                    )
                except ValueError:
                    console.print(f"[{Colors.ERROR}]Unknown model: {model_name}[/{Colors.ERROR}]")
                    console.print(
                        f"[{Colors.TEXT_DIM}]Use /models to see available options[/{Colors.TEXT_DIM}]"
                    )
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
                    console.print(
                        f"[{Colors.SUCCESS}]Reasoning set to {normalized.value}[/{Colors.SUCCESS}]"
                    )
                except ValueError:
                    valid = ", ".join(r.value for r in ReasoningEffort)
                    console.print(f"[{Colors.ERROR}]Invalid level: {level}[/{Colors.ERROR}]")
                    console.print(f"[{Colors.TEXT_DIM}]Valid options: {valid}[/{Colors.TEXT_DIM}]")
                continue

            # Send message
            await _stream_message(console, client, prompt)

    except KeyboardInterrupt:
        console.print()  # Clean line
    finally:
        console.print(f"[{Colors.TEXT_MUTED}]Goodbye[/{Colors.TEXT_MUTED}]")
        await client.stop()


async def _stream_message(console: Console, client: Copex, prompt: str) -> None:
    """Stream a message with beautiful live updates."""

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
                if (
                    chunk.tool_id and tool.tool_id == chunk.tool_id
                ) or tool.name == chunk.tool_name:
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

    # Print user prompt (subtle echo)
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
        except Exception as e:  # Catch-all: show error and continue interactive session
            state.phase = "error"
            # Don't re-raise - show error and continue session
            console.print()
            console.print(f"[{Colors.ERROR}]Error: {e}[/{Colors.ERROR}]")
            return
        finally:
            refresh_stop.set()
            try:
                await refresh_task
            except asyncio.CancelledError:
                pass

    # Render final output
    render_final_output(console, state)


def _print_welcome(console: Console, model: str, reasoning: str, version: str) -> None:
    """Print welcome banner - clean like Copilot CLI."""
    from pathlib import Path

    console.print()

    # Title line - simple and clean
    title = Text()
    title.append("copex", style=f"bold {Colors.PRIMARY}")
    title.append(f" v{version}", style=Colors.TEXT_DIM)
    console.print(title)
    console.print()

    # Working directory (like Copilot CLI)
    cwd = Path.cwd()
    home = Path.home()
    try:
        display_path = f"~/{cwd.relative_to(home)}"
    except ValueError:
        display_path = str(cwd)

    console.print(f"[{Colors.TEXT_MUTED}]Working in: {display_path}[/{Colors.TEXT_MUTED}]")
    console.print(f"[{Colors.TEXT_DIM}]Model: {model} · Reasoning: {reasoning}[/{Colors.TEXT_DIM}]")
    console.print()
    console.print(f"[{Colors.TEXT_DIM}]Type /help for commands, exit to quit[/{Colors.TEXT_DIM}]")
    console.print()


def _print_help(console: Console) -> None:
    """Print help message - clean like Copilot CLI."""
    console.print()
    console.print("[bold]Commands[/bold]")
    console.print()
    cmds = [
        ("/model <name>", "Change model"),
        ("/reasoning <level>", "Set reasoning (low/medium/high/xhigh)"),
        ("/models", "List available models"),
        ("/new", "Start new session"),
        ("/status", "Show current settings"),
        ("/clear", "Clear screen"),
        ("exit", "Quit"),
    ]
    for cmd, desc in cmds:
        console.print(
            f"  [{Colors.PRIMARY}]{cmd:22}[/{Colors.PRIMARY}] [{Colors.TEXT_MUTED}]{desc}[/{Colors.TEXT_MUTED}]"
        )

    console.print()
    console.print("[bold]Keyboard[/bold]")
    console.print()
    console.print(
        f"  [{Colors.PRIMARY}]{'Enter':22}[/{Colors.PRIMARY}] [{Colors.TEXT_MUTED}]Send message[/{Colors.TEXT_MUTED}]"
    )
    console.print(
        f"  [{Colors.PRIMARY}]{'Esc + Enter':22}[/{Colors.PRIMARY}] [{Colors.TEXT_MUTED}]New line[/{Colors.TEXT_MUTED}]"
    )
    console.print(
        f"  [{Colors.PRIMARY}]{'Tab':22}[/{Colors.PRIMARY}] [{Colors.TEXT_MUTED}]Complete command[/{Colors.TEXT_MUTED}]"
    )
    console.print(
        f"  [{Colors.PRIMARY}]{'↑ / ↓':22}[/{Colors.PRIMARY}] [{Colors.TEXT_MUTED}]History navigation[/{Colors.TEXT_MUTED}]"
    )
    console.print(
        f"  [{Colors.PRIMARY}]{'Ctrl+L':22}[/{Colors.PRIMARY}] [{Colors.TEXT_MUTED}]Clear screen[/{Colors.TEXT_MUTED}]"
    )
    console.print(
        f"  [{Colors.PRIMARY}]{'Ctrl+C':22}[/{Colors.PRIMARY}] [{Colors.TEXT_MUTED}]Cancel/Exit[/{Colors.TEXT_MUTED}]"
    )
    console.print()


def _print_status(console: Console, client: Copex) -> None:
    """Print current status - clean format."""
    from pathlib import Path

    console.print()
    console.print("[bold]Current Session[/bold]")
    console.print()

    # Working directory
    cwd = Path.cwd()
    home = Path.home()
    try:
        display_path = f"~/{cwd.relative_to(home)}"
    except ValueError:
        display_path = str(cwd)

    console.print(
        f"  [{Colors.TEXT_DIM}]Directory:[/{Colors.TEXT_DIM}]  [{Colors.TEXT_MUTED}]{display_path}[/{Colors.TEXT_MUTED}]"
    )
    console.print(
        f"  [{Colors.TEXT_DIM}]Model:[/{Colors.TEXT_DIM}]      [{Colors.PRIMARY}]{client.config.model.value}[/{Colors.PRIMARY}]"
    )
    console.print(
        f"  [{Colors.TEXT_DIM}]Reasoning:[/{Colors.TEXT_DIM}]  [{Colors.ACCENT}]{client.config.reasoning_effort.value}[/{Colors.ACCENT}]"
    )
    console.print()


def _print_models(console: Console, current: Model) -> None:
    """Print available models."""
    from copex.models import Model

    console.print()
    console.print("[bold]Available Models[/bold]")
    console.print()
    for model in Model:
        if model == current:
            console.print(
                f"  [{Colors.SUCCESS}]→[/{Colors.SUCCESS}] [{Colors.PRIMARY} bold]{model.value}[/{Colors.PRIMARY} bold] [{Colors.TEXT_DIM}](current)[/{Colors.TEXT_DIM}]"
            )
        else:
            console.print(f"    [{Colors.TEXT_MUTED}]{model.value}[/{Colors.TEXT_MUTED}]")
    console.print()
    console.print(f"[{Colors.TEXT_DIM}]Use /model <name> to switch[/{Colors.TEXT_DIM}]")
    console.print()
