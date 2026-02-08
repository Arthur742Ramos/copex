"""Main TUI Application for Copex."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI, FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import (
    ConditionalContainer,
    Float,
    FloatContainer,
    HSplit,
    Window,
)
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.styles import Style
from rich.console import Console

from .history import CombinedHistoryManager
from .keymap import Action, KeymapManager
from .palette import CommandCategory, CommandPalette, PaletteCommand
from .render import (
    render_message,
    render_prompt_prefix,
    render_reasoning_expanded,
    render_spinner,
    render_to_ansi,
    render_tool_call_collapsed,
    render_tool_call_expanded,
)
from .state import SessionState, ToolCallState, TuiMode, TuiState

# Conditional imports - these are only needed at runtime
if TYPE_CHECKING:
    pass


# Styles for prompt_toolkit
TUI_STYLE = Style.from_dict(
    {
        # Input area
        "prompt": "bold fg:#9ece6a",
        "continuation": "fg:#6b7280",
        # Status bar
        "status": "bg:#1f2430 fg:#e6e6e6",
        "status.model": "bold fg:#7dcfff",
        "status.reasoning": "fg:#bb9af7",
        "status.tokens": "fg:#7aa2f7",
        "status.duration": "fg:#89b4fa",
        "status.activity": "fg:#e0af68",
        "status.sep": "fg:#3b4252",
        "status.dim": "fg:#9aa4b2",
        "status.divider": "fg:#1f2937",
        "status.last": "fg:#a5b4fc",
        "status.ready": "fg:#8ab4f8",
        "status.active": "fg:#e0af68 bold",
        "status.kpi": "fg:#cbd5f5",
        "status.kpi.dim": "fg:#94a3b8",
        "status.label": "fg:#94a3b8",
        "status.metric": "fg:#cbd5f5",
        # Palette
        "palette": "bg:#0f172a fg:#e5e7eb",
        "palette.item": "fg:#e5e7eb",
        "palette.selected": "bg:#1d4ed8 fg:#f8fafc bold",
        "palette.description": "fg:#9ca3af",
        "palette.description.selected": "bg:#1d4ed8 fg:#e2e8f0",
        "palette.shortcut": "fg:#a78bfa",
        "palette.meta": "fg:#60a5fa",
        "palette.no_results": "fg:#94a3b8 italic",
        "palette.meta.selected": "bg:#1d4ed8 fg:#c7d2fe",
        "palette.shortcut.selected": "bg:#1d4ed8 fg:#e9d5ff",
        "palette.border": "fg:#1f2937",
        "palette.info": "fg:#94a3b8",
        "palette.query": "bold fg:#f8fafc",
        "palette.query.placeholder": "fg:#94a3b8 italic",
        "palette.icon": "fg:#93c5fd",
        "palette.icon.selected": "bg:#1d4ed8 fg:#dbeafe",
        "palette.more": "fg:#64748b",
        # Messages
        "user": "bold fg:ansigreen",
        "assistant": "fg:ansiwhite",
        "reasoning": "italic fg:ansigray",
        "tool": "fg:ansiyellow",
        # Stash indicator
        "stash": "fg:#60a5fa",
        # Input hint
        "input_hint": "fg:#9ca3af italic",
        "input_hint.key": "bold fg:#e2e8f0",
        "input_hint.sep": "fg:#4b5563",
    }
)


@dataclass
class TuiApp:
    """
    Main TUI Application for Copex.

    Integrates:
    - prompt_toolkit for input handling
    - Rich for rendering
    - State management
    - Command palette
    - History and stash
    """

    # Core components (always available)
    state: TuiState = field(default_factory=TuiState)
    palette: CommandPalette = field(default_factory=CommandPalette)
    keymap: KeymapManager = field(default_factory=KeymapManager)
    history_manager: CombinedHistoryManager = field(default_factory=CombinedHistoryManager)

    # Rich console for rendering
    console: Console = field(default_factory=Console)

    # Internal state
    _running: bool = False
    _spinner_frame: int = 0
    _last_render: float = 0.0
    _app: Application | None = None
    _input_buffer: Buffer | None = None

    # These will be set in run() - they require copex imports
    _config: Any = None
    _client: Any = None
    _metrics: Any = None
    _current_send_task: asyncio.Task[None] | None = None

    def __post_init__(self) -> None:
        """Initialize components after dataclass creation."""
        # Register keymap handlers
        self._register_handlers()

        # Register palette actions
        self._register_palette_actions()

        # Set up state change callback
        self.state.on_state_change = self._on_state_change

    def _register_handlers(self) -> None:
        """Register action handlers."""
        handlers = {
            Action.SEND: self._handle_send,
            Action.NEWLINE: self._handle_newline,
            Action.CANCEL: self._handle_cancel,
            Action.CLEAR_INPUT: self._handle_clear_input,
            Action.HISTORY_PREV: self._handle_history_prev,
            Action.HISTORY_NEXT: self._handle_history_next,
            Action.OPEN_PALETTE: self._handle_open_palette,
            Action.CLOSE_PALETTE: self._handle_close_palette,
            Action.PALETTE_UP: self._handle_palette_up,
            Action.PALETTE_DOWN: self._handle_palette_down,
            Action.PALETTE_SELECT: self._handle_palette_select,
            Action.STASH_SAVE: self._handle_stash_save,
            Action.STASH_RESTORE: self._handle_stash_restore,
            Action.STASH_CYCLE: self._handle_stash_cycle,
            Action.TOGGLE_REASONING: self._handle_toggle_reasoning,
            Action.TOGGLE_TOOLS: self._handle_toggle_tools,
            Action.NEW_SESSION: self._handle_new_session,
            Action.CLEAR_SCREEN: self._handle_clear_screen,
            Action.MODEL_PICKER: self._handle_model_picker,
            Action.REASONING_PICKER: self._handle_reasoning_picker,
            Action.EXIT: self._handle_exit,
        }

        for action, handler in handlers.items():
            self.keymap.register_handler(action, handler)

    def _register_palette_actions(self) -> None:
        """Register actions for palette commands."""
        actions = {
            "session:new": self._handle_new_session,
            "session:clear": self._handle_clear_screen,
            "view:tools:toggle": self._handle_toggle_tools,
            "view:reasoning:toggle": self._handle_toggle_reasoning,
            "view:statusbar:toggle": self._handle_toggle_statusbar,
            "export:json": lambda: self._handle_export("json"),
            "export:markdown": lambda: self._handle_export("markdown"),
            "export:metrics": lambda: self._handle_export("metrics"),
            "help:shortcuts": self._handle_show_shortcuts,
            "help:about": self._handle_show_about,
        }

        for cmd_id, action in actions.items():
            self.palette.set_action(cmd_id, action)

    def _on_state_change(self) -> None:
        """Called when state changes."""
        if self._app:
            self._app.invalidate()

    # =========================================================================
    # Action Handlers
    # =========================================================================

    def _handle_send(self) -> None:
        """Handle send action."""
        if not self._input_buffer:
            return

        text = self._input_buffer.text.strip()
        if not text:
            return

        # Add to history
        self.history_manager.add_to_history(text, model=self.state.session.model_label)

        # Clear input
        self._input_buffer.reset()

        # Send message (will be handled by main loop)
        self.state.input_buffer = text
        self.state.notify_change()

    def _handle_newline(self) -> None:
        """Insert a newline in the input."""
        if self._input_buffer:
            self._input_buffer.insert_text("\n")

    def _handle_cancel(self) -> None:
        """Handle cancel action."""
        if self.state.mode == TuiMode.PALETTE:
            self._handle_close_palette()
        elif self.state.session.is_streaming:
            # Best-effort cancellation:
            # - Cancel our local task
            # - Ask the underlying Copilot session to abort the current turn
            if self._current_send_task and not self._current_send_task.done():
                self._current_send_task.cancel()

            if self._client is not None:
                try:
                    asyncio.get_running_loop().create_task(self._client.abort())
                except RuntimeError:
                    # No running loop (shouldn't happen inside the TUI), ignore.
                    pass

            self.state.session.is_streaming = False
            self.state.session.is_thinking = False
            self.state.session.current_activity = "cancelled"
            self._show_notification("Request cancelled")
            self.state.notify_change()

    def _handle_clear_input(self) -> None:
        """Clear the input buffer."""
        if self._input_buffer:
            self._input_buffer.reset()

    def _handle_history_prev(self) -> None:
        """Navigate to previous history."""
        if not self._input_buffer:
            return

        prev = self.history_manager.history_up(self._input_buffer.text)
        if prev is not None:
            self._input_buffer.document = self._input_buffer.document.__class__(prev)

    def _handle_history_next(self) -> None:
        """Navigate to next history."""
        if not self._input_buffer:
            return

        next_text = self.history_manager.history_down(self._input_buffer.text)
        if next_text is not None:
            self._input_buffer.document = self._input_buffer.document.__class__(next_text)

    def _handle_open_palette(self) -> None:
        """Open the command palette."""
        self.palette.reset()
        self.state.open_palette()

    def _handle_close_palette(self) -> None:
        """Close the command palette."""
        self.palette.reset()
        self.state.close_palette()

    def _handle_palette_up(self) -> None:
        """Move palette selection up."""
        if self.state.mode == TuiMode.PALETTE:
            results = self.palette.search(self.state.palette_query)
            self.state.move_palette_selection(-1, max_index=len(results) - 1)

    def _handle_palette_down(self) -> None:
        """Move palette selection down."""
        if self.state.mode == TuiMode.PALETTE:
            results = self.palette.search(self.state.palette_query)
            self.state.move_palette_selection(1, max_index=len(results) - 1)

    def _handle_palette_select(self) -> None:
        """Select current palette item."""
        if self.state.mode != TuiMode.PALETTE:
            return

        results = self.palette.search(self.state.palette_query)
        if results and 0 <= self.state.palette_selected < len(results):
            cmd, _ = results[self.state.palette_selected]
            self._execute_palette_command(cmd)

    def _execute_palette_command(self, cmd: PaletteCommand) -> None:
        """Execute a palette command."""
        if cmd.action:
            cmd.action()
            self.state.close_palette()
        elif cmd.subcommands:
            self.palette.push_commands(cmd.subcommands)
            self.state.update_palette_query("")
        elif cmd.value is not None:
            # Handle model/reasoning selection.
            # In normal runtime we use the Model/ReasoningEffort enums for type
            # consistency; in isolated tests (where copex.models may not import)
            # we fall back to plain strings.
            if cmd.id.startswith("model:"):
                try:
                    from copex.models import Model

                    model_enum = Model(cmd.value)
                    self.state.session.model = model_enum
                    if self._config:
                        self._config.model = model_enum
                except (ValueError, ImportError):
                    self.state.session.model = cmd.value
            elif cmd.id.startswith("reasoning:"):
                try:
                    from copex.models import ReasoningEffort

                    effort_enum = ReasoningEffort(cmd.value)
                    self.state.session.reasoning_effort = effort_enum
                    if self._config:
                        self._config.reasoning_effort = effort_enum
                except (ValueError, ImportError):
                    self.state.session.reasoning_effort = cmd.value
            self.state.close_palette()

    def _handle_stash_save(self) -> None:
        """Save current input to stash."""
        if not self._input_buffer:
            return

        text = self._input_buffer.text
        cursor_pos = self._input_buffer.cursor_position

        if self.history_manager.stash_draft(text, cursor_pos):
            self._input_buffer.reset()
            self._show_notification("Draft saved to stash")

    def _handle_stash_restore(self) -> None:
        """Restore from stash."""
        result = self.history_manager.restore_draft()
        if result and self._input_buffer:
            content, cursor_pos = result
            self._input_buffer.document = self._input_buffer.document.__class__(content, cursor_pos)
            self._show_notification("Draft restored from stash")

    def _handle_stash_cycle(self) -> None:
        """Cycle through stash."""
        result = self.history_manager.cycle_stash()
        if result and self._input_buffer:
            content, cursor_pos = result
            self._input_buffer.document = self._input_buffer.document.__class__(content, cursor_pos)

    def _handle_toggle_reasoning(self) -> None:
        """Toggle reasoning panel visibility."""
        self.state.show_reasoning = not self.state.show_reasoning
        self.state.notify_change()

    def _handle_toggle_tools(self) -> None:
        """Toggle tool calls expansion."""
        if self.state.expand_all_tools:
            self.state.collapse_all_tool_calls()
        else:
            self.state.expand_all_tool_calls()

    def _handle_toggle_statusbar(self) -> None:
        """Toggle status bar visibility."""
        self.state.show_status_bar = not self.state.show_status_bar
        self.state.notify_change()

    def _handle_new_session(self) -> None:
        """Start a new session."""
        if self._client:
            self._client.new_session()
        self.state.clear_current_response()
        self.state.messages = []
        # Reset session state with current model/reasoning
        model = self._config.model if self._config else "claude-opus-4.5"
        reasoning = self._config.reasoning_effort if self._config else "xhigh"
        self.state.session = SessionState(model=model, reasoning_effort=reasoning)
        self._show_notification("New session started")

    def _handle_clear_screen(self) -> None:
        """Clear the screen."""
        self.console.clear()

    def _handle_model_picker(self) -> None:
        """Open the model picker."""
        self.palette.reset()
        self.state.open_palette()

        cmd = self.palette.get_command("model")
        if cmd and cmd.subcommands:
            self.palette.push_commands(cmd.subcommands)
            self.state.update_palette_query("")
        else:
            # Fallback to query filter
            self.state.update_palette_query("model")

    def _handle_reasoning_picker(self) -> None:
        """Open the reasoning picker."""
        self.palette.reset()
        self.state.open_palette()

        cmd = self.palette.get_command("reasoning")
        if cmd and cmd.subcommands:
            self.palette.push_commands(cmd.subcommands)
            self.state.update_palette_query("")
        else:
            self.state.update_palette_query("reasoning")

    def _handle_exit(self) -> None:
        """Exit the application."""
        if self._input_buffer and self._input_buffer.text.strip():
            # Don't exit if there's input, clear instead
            self._input_buffer.reset()
        else:
            self._running = False
            if self._app:
                self._app.exit()

    def _show_notification(self, message: str) -> None:
        """Show a brief notification."""
        # For now, just print - can be enhanced with a toast
        self.console.print(f"[dim]{message}[/dim]")

    def _handle_show_shortcuts(self) -> None:
        """Show keyboard shortcuts."""
        from .render import render_help_panel

        content = render_help_panel(self.keymap.get_help_text())
        self.console.print(content)

    def _handle_show_about(self) -> None:
        """Show about info."""
        from copex.cli import __version__

        self.console.print(f"[bold]Copex TUI[/bold] v{__version__}")

    def _handle_export(self, export_type: str) -> None:
        """Export conversation or metrics."""
        export_dir = Path.home() / ".copex"
        export_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        if export_type == "json":
            path = export_dir / f"copex_chat_{timestamp}.json"
            data = {
                "model": self.state.session.model_label,
                "reasoning_effort": self.state.session.reasoning_label,
                "messages": self.state.messages,
            }
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            self._show_notification(f"Exported JSON to {path}")
        elif export_type == "markdown":
            path = export_dir / f"copex_chat_{timestamp}.md"
            lines = []
            for msg in self.state.messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                lines.append(f"## {role.capitalize()}\n\n{content}\n")
                reasoning = msg.get("reasoning")
                if reasoning:
                    lines.append(f"> Reasoning:\n>\n> {reasoning}\n")
            path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
            self._show_notification(f"Exported Markdown to {path}")
        elif export_type == "metrics":
            path = export_dir / f"copex_metrics_{timestamp}.csv"
            if self._metrics:
                self._metrics.export_csv(path)
                self._show_notification(f"Exported metrics to {path}")
            else:
                self._show_notification("No metrics available")

    # =========================================================================
    # Rendering
    # =========================================================================

    def _get_status_bar(self) -> FormattedText:
        """Build the status bar content."""
        if not self.state.show_status_bar:
            return FormattedText([])

        from copex.ui import Icons

        parts: list[tuple[str, str]] = []

        def add_sep() -> None:
            parts.append(("class:status.sep", " │ "))

        # Model
        parts.append(("class:status.model", f" {Icons.ROBOT} {self.state.session.model_label} "))
        add_sep()

        # Reasoning
        parts.append(
            ("class:status.reasoning", f" {Icons.BRAIN} {self.state.session.reasoning_label} ")
        )
        add_sep()

        # Duration
        parts.append(
            ("class:status.duration", f" {Icons.CLOCK} {self.state.session.session_duration_str} ")
        )
        add_sep()

        # Tokens (real counts when available; otherwise show unknown)
        if self.state.session.request_count == 0 or self.state.session.tokens_complete:
            token_value = f"{self.state.session.total_tokens:,}"
        else:
            token_value = "—"
        parts.append(("class:status.metric", f" {token_value} tokens "))
        add_sep()

        # Cost (real cost when available; otherwise show unknown)
        if self.state.session.request_count == 0 or self.state.session.cost_complete:
            cost_text = f"${self.state.session.total_cost:.4f}"
        else:
            cost_text = "$—"
        parts.append(("class:status.cost", f" {cost_text} "))

        # Requests + last request duration
        add_sep()
        req_label = "req" if self.state.session.request_count == 1 else "reqs"
        parts.append(("class:status.kpi", f" {self.state.session.request_count} {req_label} "))
        if self.state.session.last_request_duration is not None:
            parts.append(("class:status.kpi.dim", "• "))
            parts.append(
                ("class:status.last", f"{self.state.session.last_request_duration:.1f}s last ")
            )
        elif self.state.session.request_count == 0:
            parts.append(("class:status.label", " • no requests "))

        # Activity (right side)
        add_sep()
        activity_label = self.state.session.current_activity.replace("_", " ").capitalize()
        if self.state.session.is_streaming:
            spinner = render_spinner(self._spinner_frame, style="braille")
            parts.append(("class:status.active", f" {spinner} {activity_label} "))
        else:
            parts.append(("class:status.ready", f" {Icons.DONE} ready "))

        # Stash indicator
        if self.history_manager.has_stash:
            add_sep()
            count = self.history_manager.stash_count
            pos = self.history_manager.stash_position + 1
            parts.append(("class:stash", f" 📋 {pos}/{count} "))

        return FormattedText(parts)

    def _get_input_hint(self) -> FormattedText:
        """Build context-aware input hints."""
        parts: list[tuple[str, str]] = []

        def add_hint(key: str, label: str) -> None:
            if parts:
                parts.append(("class:input_hint.sep", " • "))
            parts.append(("class:input_hint.key", key))
            parts.append(("class:input_hint", f" {label}"))

        if self.state.mode == TuiMode.PALETTE:
            add_hint("↑↓", "navigate")
            add_hint("Enter", "select")
            if self.palette.has_parent:
                add_hint("Backspace", "back")
                add_hint("Esc", "close")
            else:
                add_hint("Esc", "close")
            add_hint("Ctrl+P", "close palette")
            return FormattedText(parts)

        if self.state.session.is_streaming:
            tools_label = "collapse tools" if self.state.expand_all_tools else "expand tools"
            reasoning_label = "hide reasoning" if self.state.show_reasoning else "show reasoning"
            add_hint("Ctrl+C", "cancel")
            add_hint("Ctrl+T", tools_label)
            add_hint("Ctrl+G", reasoning_label)
            add_hint("Ctrl+P", "palette")
            return FormattedText(parts)

        has_input = bool(self._input_buffer and self._input_buffer.text.strip())
        if has_input:
            tools_label = "collapse tools" if self.state.expand_all_tools else "expand tools"
            reasoning_label = "hide reasoning" if self.state.show_reasoning else "show reasoning"
            add_hint("Enter", "send")
            add_hint("Ctrl+J", "newline")
            add_hint("Ctrl+U", "clear")
            add_hint("Ctrl+P", "palette")
            add_hint("Ctrl+M", "model")
            add_hint("Ctrl+E", "reasoning")
            add_hint("Ctrl+S", "stash")
            add_hint("Ctrl+R", "restore")
            add_hint("Ctrl+T", tools_label)
            add_hint("Ctrl+G", reasoning_label)
        else:
            tools_label = "collapse tools" if self.state.expand_all_tools else "expand tools"
            reasoning_label = "hide reasoning" if self.state.show_reasoning else "show reasoning"
            add_hint("Ctrl+N", "new session")
            add_hint("Ctrl+P", "palette")
            add_hint("Ctrl+M", "model")
            add_hint("Ctrl+E", "reasoning")
            add_hint("Ctrl+L", "clear screen")
            add_hint("Ctrl+D", "exit")
            add_hint("Ctrl+T", tools_label)
            add_hint("Ctrl+G", reasoning_label)
        return FormattedText(parts)

    def _get_palette_content(self) -> FormattedText:
        """Build the palette content."""
        if self.state.mode != TuiMode.PALETTE:
            return FormattedText([])

        results = self.palette.search(self.state.palette_query)

        parts: list[tuple[str, str]] = []

        # Search box
        from copex.ui import Icons

        category_icons = {
            CommandCategory.MODEL: Icons.ROBOT,
            CommandCategory.REASONING: Icons.BRAIN,
            CommandCategory.SESSION: Icons.SPARKLE,
            CommandCategory.EXPORT: Icons.FILE_WRITE,
            CommandCategory.VIEW: Icons.TOOL,
            CommandCategory.HELP: Icons.INFO,
        }

        parts.append(("class:palette.meta", " 🔍 "))
        if self.state.palette_query:
            parts.append(("class:palette.query", self.state.palette_query))
        else:
            parts.append(("class:palette.query.placeholder", "Type to search…"))
        if results:
            result_label = "result" if len(results) == 1 else "results"
            parts.append(("class:palette.info", f"  {len(results)} {result_label}"))
        else:
            parts.append(("class:palette.info", "  0 results"))
        parts.append(("", "\n\n"))

        # Results
        max_results = 8
        for i, (cmd, _score) in enumerate(results[:max_results]):
            selected = i == self.state.palette_selected
            prefix = "▸" if selected else " "
            label_style = "class:palette.selected" if selected else "class:palette.item"
            desc_style = (
                "class:palette.description.selected" if selected else "class:palette.description"
            )
            meta_style = "class:palette.meta.selected" if selected else "class:palette.meta"
            shortcut_style = (
                "class:palette.shortcut.selected" if selected else "class:palette.shortcut"
            )
            icon_style = "class:palette.icon.selected" if selected else "class:palette.icon"
            icon = category_icons.get(cmd.category, Icons.INFO)

            parts.append((label_style, f" {prefix} "))
            parts.append((icon_style, f"{icon} "))
            parts.append((label_style, f"{cmd.label} "))
            if cmd.shortcut:
                parts.append((shortcut_style, f"{cmd.shortcut} "))
            parts.append((meta_style, f"[{cmd.category.value.upper()}] "))

            description = cmd.description
            if len(description) > 44:
                description = description[:41] + "…"
            parts.append((desc_style, description))
            parts.append(("", "\n"))
        if len(results) > max_results:
            parts.append(("class:palette.more", f"  … {len(results) - max_results} more"))
            parts.append(("", "\n"))

        if not results:
            parts.append(("class:palette.no_results", "  No matching commands\n"))

        # Help
        parts.append(("", "\n"))
        parts.append(("class:palette.shortcut", "↑↓"))
        parts.append(("class:palette.description", " navigate  "))
        parts.append(("class:palette.shortcut", "Enter"))
        parts.append(("class:palette.description", " select  "))
        if self.palette.has_parent:
            parts.append(("class:palette.shortcut", "Backspace"))
            parts.append(("class:palette.description", " back  "))
        parts.append(("class:palette.shortcut", "Esc"))
        parts.append(("class:palette.description", " close  "))
        parts.append(("class:palette.shortcut", "Ctrl+P"))
        parts.append(("class:palette.description", " toggle"))

        return FormattedText(parts)

    def _get_main_content(self) -> ANSI:
        """Render the main conversation area."""
        # Keep rendering lightweight; prompt_toolkit may call this often.
        from rich.box import ROUNDED
        from rich.console import Group
        from rich.markdown import Markdown
        from rich.panel import Panel
        from rich.text import Text

        from copex.ui import Icons, Theme

        renderables: list[Any] = []

        # Show recent history (avoid massive re-rendering for long sessions).
        history = self.state.messages[-50:]
        for msg in history:
            role = msg.get("role")
            content = msg.get("content") or ""

            if role == "user":
                renderables.append(
                    Panel(
                        Text(content, style=Theme.SUCCESS),
                        title=f"[{Theme.SUCCESS}]▸ {Icons.ARROW_RIGHT} You[/{Theme.SUCCESS}]",
                        title_align="left",
                        border_style=Theme.BORDER,
                        padding=(0, 1),
                        box=ROUNDED,
                    )
                )
                continue

            if role == "assistant":
                renderables.append(
                    Panel(
                        Markdown(content),
                        title=f"[{Theme.PRIMARY}]▾ {Icons.ROBOT} Assistant[/{Theme.PRIMARY}]",
                        title_align="left",
                        border_style=Theme.BORDER,
                        padding=(0, 1),
                        box=ROUNDED,
                    )
                )

                reasoning = msg.get("reasoning")
                if self.state.show_reasoning and reasoning:
                    renderables.append(render_reasoning_expanded(str(reasoning)))

                # Persisted tool calls for this completed turn.
                tool_calls = msg.get("tool_calls") or []
                for tc in tool_calls:
                    name = str(tc.get("name", "unknown"))
                    status = str(tc.get("status", "success"))
                    icon = ToolCallState(name=name).icon
                    renderables.append(
                        render_tool_call_collapsed(
                            name=name,
                            icon=icon,
                            status=status,
                            duration=tc.get("duration"),
                            arguments=tc.get("arguments"),
                            is_selected=False,
                        )
                    )

        # Current in-flight response
        if self.state.session.is_streaming:
            if self.state.show_reasoning and self.state.current_reasoning:
                renderables.append(render_reasoning_expanded(self.state.current_reasoning))

            if self.state.current_message:
                renderables.append(render_message(self.state.current_message, is_streaming=True))

            # Tool calls for the in-flight request
            for tc in self.state.tool_calls:
                duration = (
                    tc.duration
                    if tc.duration is not None
                    else (tc.elapsed if tc.status == "running" else None)
                )
                if self.state.expand_all_tools or tc.is_expanded:
                    renderables.append(
                        render_tool_call_expanded(
                            name=tc.name,
                            icon=tc.icon,
                            status=tc.status,
                            arguments=tc.arguments,
                            result=tc.result,
                            duration=duration,
                        )
                    )
                else:
                    renderables.append(
                        render_tool_call_collapsed(
                            name=tc.name,
                            icon=tc.icon,
                            status=tc.status,
                            duration=duration,
                            arguments=tc.arguments,
                            is_selected=False,
                        )
                    )

        if not renderables:
            empty_text = Text()
            empty_text.append(f"{Icons.SPARKLE} ", style=Theme.ACCENT)
            empty_text.append("No messages yet. ", style=Theme.MUTED)
            empty_text.append("Type below to start", style=Theme.MESSAGE)
            empty_text.append(" • Ctrl+P palette • Ctrl+N new session", style=Theme.MUTED)
            renderables.append(empty_text)

        return ANSI(render_to_ansi(Group(*renderables)))

    def _build_layout(self) -> Layout:
        """Build the prompt_toolkit layout."""
        # Input buffer
        self._input_buffer = Buffer(
            name="input",
            multiline=True,
        )

        # Status bar at bottom
        status_bar = ConditionalContainer(
            content=Window(
                content=FormattedTextControl(self._get_status_bar),
                height=1,
                style="class:status",
            ),
            filter=Condition(lambda: self.state.show_status_bar),
        )

        status_separator = ConditionalContainer(
            content=Window(
                content=FormattedTextControl([("class:status.divider", "─" * 200)]),
                height=1,
                style="class:status",
            ),
            filter=Condition(lambda: self.state.show_status_bar),
        )

        # Input area with prompt
        prompt_window = Window(
            content=FormattedTextControl(
                lambda: ANSI(
                    render_prompt_prefix(
                        self.state.session.model_label,
                        is_multiline=(
                            self.state.mode != TuiMode.PALETTE
                            and bool(self._input_buffer and self._input_buffer.multiline)
                        ),
                    )
                )
            ),
            height=1,
        )
        input_window = Window(
            content=BufferControl(buffer=self._input_buffer),
            height=Dimension(min=1, max=10),
        )
        input_hint_window = Window(
            content=FormattedTextControl(self._get_input_hint),
            height=1,
            style="class:input_hint",
        )
        input_area = HSplit([prompt_window, input_window, input_hint_window])

        # Main content area (for messages)
        main_content = Window(
            content=FormattedTextControl(self._get_main_content),
            height=Dimension(weight=1),
            wrap_lines=True,
        )

        # Palette overlay (float)
        palette_float = Float(
            content=ConditionalContainer(
                content=Window(
                    content=FormattedTextControl(self._get_palette_content),
                    width=70,
                    height=15,
                    style="class:palette class:palette.border",
                ),
                filter=Condition(lambda: self.state.mode == TuiMode.PALETTE),
            ),
            left=2,
            top=2,
        )

        # Main layout
        root = FloatContainer(
            content=HSplit(
                [
                    main_content,
                    input_area,
                    status_separator,
                    status_bar,
                ]
            ),
            floats=[palette_float],
        )

        return Layout(root, focused_element=input_window)

    def _build_keybindings(self) -> KeyBindings:
        """Build keybindings for the application."""
        kb = KeyBindings()

        # Mode getter for conditional bindings
        def get_mode() -> str:
            if self.state.mode == TuiMode.PALETTE:
                return "palette"
            return "input"

        # Enter to send (when not in palette and has content)
        @kb.add("enter")
        def handle_enter(event) -> None:
            if self.state.mode == TuiMode.PALETTE:
                self._handle_palette_select()
            elif self._input_buffer and self._input_buffer.text.strip():
                self._handle_send()
            # else: do nothing (empty input)

        # Multiline input
        #
        # Many terminals *cannot* send a distinct Shift+Enter sequence; they
        # commonly emit ESC followed by Enter. We bind both Ctrl+J (reliable)
        # and the ESC+Enter sequence to insert a newline.
        @kb.add("c-j")
        def handle_ctrl_j(event) -> None:
            self._handle_newline()

        @kb.add("escape", "enter")
        def handle_escape_enter(event) -> None:
            # Treat ESC+Enter as "newline" in input mode (fixes Shift+Enter in
            # common terminal configurations).
            if self.state.mode != TuiMode.PALETTE:
                self._handle_newline()

        # Note: prompt_toolkit does not have a first-class Shift+Enter key.
        # Terminals that emit ESC+Enter (common) are handled above, and Ctrl+J
        # is the reliable primary newline binding.

        # Ctrl+P for palette
        @kb.add("c-p")
        def handle_ctrl_p(event) -> None:
            if self.state.mode == TuiMode.PALETTE:
                self._handle_close_palette()
            else:
                self._handle_open_palette()

        # Escape to close palette
        @kb.add("escape")
        def handle_escape(event) -> None:
            if self.state.mode == TuiMode.PALETTE:
                if self.palette.pop_commands():
                    self.state.update_palette_query("")
                else:
                    self._handle_close_palette()

        # Arrow keys in palette
        @kb.add("up")
        def handle_up(event) -> None:
            if self.state.mode == TuiMode.PALETTE:
                self._handle_palette_up()
            else:
                self._handle_history_prev()

        @kb.add("down")
        def handle_down(event) -> None:
            if self.state.mode == TuiMode.PALETTE:
                self._handle_palette_down()
            else:
                self._handle_history_next()

        @kb.add("backspace", filter=Condition(lambda: self.state.mode == TuiMode.PALETTE))
        def handle_backspace(event) -> None:
            if self.state.palette_query:
                self.state.update_palette_query(self.state.palette_query[:-1])
            else:
                if self.palette.pop_commands():
                    self.state.update_palette_query("")
                else:
                    self._handle_close_palette()

        @kb.add("<any>", filter=Condition(lambda: self.state.mode == TuiMode.PALETTE))
        def handle_any(event) -> None:
            if event.data:
                self.state.update_palette_query(self.state.palette_query + event.data)

        # Stash commands
        @kb.add("c-s")
        def handle_ctrl_s(event) -> None:
            self._handle_stash_save()

        @kb.add("c-r")
        def handle_ctrl_r(event) -> None:
            self._handle_stash_restore()

        @kb.add("c-u")
        def handle_ctrl_u(event) -> None:
            self._handle_clear_input()

        # Toggle commands
        @kb.add("c-t")
        def handle_ctrl_t(event) -> None:
            self._handle_toggle_tools()

        @kb.add("c-g")
        def handle_ctrl_g(event) -> None:
            self._handle_toggle_reasoning()

        # Session commands
        @kb.add("c-n")
        def handle_ctrl_n(event) -> None:
            self._handle_new_session()

        @kb.add("c-l")
        def handle_ctrl_l(event) -> None:
            self._handle_clear_screen()

        # Exit
        @kb.add("c-d")
        def handle_ctrl_d(event) -> None:
            if not self._input_buffer or not self._input_buffer.text.strip():
                self._handle_exit()

        @kb.add("c-c")
        def handle_ctrl_c(event) -> None:
            self._handle_cancel()

        @kb.add("c-q")
        def handle_ctrl_q(event) -> None:
            self._handle_exit()

        return kb

    # =========================================================================
    # Main Loop
    # =========================================================================

    async def _process_message(self, prompt: str) -> None:
        """Process a user message."""
        if not self._client:
            return

        self.state.clear_current_response()
        self.state.session.start_request()

        # Add user message to history
        self.state.messages.append(
            {
                "role": "user",
                "content": prompt,
                "timestamp": time.time(),
            }
        )

        # Streaming callback - defined here to capture chunk type
        def on_chunk(chunk: Any) -> None:
            if chunk.type == "message":
                if chunk.is_final and chunk.content is not None:
                    self.state.current_message = chunk.content
                    self.state.session.current_activity = "responding"
                    self.state.notify_change()
                else:
                    self.state.add_message_delta(chunk.delta)
            elif chunk.type == "reasoning":
                if chunk.is_final and chunk.content is not None:
                    self.state.current_reasoning = chunk.content
                    self.state.session.current_activity = "reasoning"
                    self.state.notify_change()
                else:
                    self.state.add_reasoning_delta(chunk.delta)
            elif chunk.type == "tool_call":
                self.state.add_tool_call(
                    name=chunk.tool_name or "unknown",
                    arguments=chunk.tool_args or {},
                )
            elif chunk.type == "tool_result":
                status = "success" if chunk.tool_success is not False else "error"
                self.state.update_tool_call(
                    chunk.tool_name or "unknown",
                    result=chunk.tool_result,
                    status=status,
                    duration=chunk.tool_duration,
                )

        try:
            response = await self._client.send(
                prompt,
                on_chunk=on_chunk,
                metrics=self._metrics,
            )
            tokens, cost, prompt_tokens, completion_tokens = self._extract_usage(response)
            self.state.session.end_request(
                tokens=tokens,
                cost=cost,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

            # Add assistant message (persist tool calls for exporting/viewing).
            tool_calls = [
                {
                    "name": tc.name,
                    "arguments": tc.arguments,
                    "result": tc.result,
                    "status": tc.status,
                    "duration": tc.duration,
                }
                for tc in self.state.tool_calls
            ]

            self.state.messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                    "reasoning": response.reasoning,
                    "tool_calls": tool_calls,
                    "timestamp": time.time(),
                }
            )

            # Clear transient streaming buffers (the completed message is now in
            # self.state.messages).
            self.state.clear_current_response()

        except Exception as e:  # Catch-all: streaming errors shown to user, session continues
            self.state.session.is_streaming = False
            self.state.session.current_activity = "error"
            self.console.print(f"[red]Error: {e}[/red]")
        finally:
            self._current_send_task = None

    async def run(self, config: Any = None) -> None:
        """Run the TUI application.

        Args:
            config: CopexConfig instance (imported at runtime to avoid circular imports)
        """
        # Import copex modules at runtime to avoid circular imports
        from copex.client import Copex
        from copex.config import CopexConfig
        from copex.metrics import get_collector

        # Initialize config
        self._config = config or CopexConfig()
        self._metrics = get_collector()

        # Set initial state from config
        self.state.session.model = self._config.model
        self.state.session.reasoning_effort = self._config.reasoning_effort

        # Initialize client
        self._client = Copex(self._config)
        await self._client.start()

        self._running = True

        # Build application
        layout = self._build_layout()
        keybindings = self._build_keybindings()

        self._app = Application(
            layout=layout,
            key_bindings=keybindings,
            style=TUI_STYLE,
            full_screen=False,
            mouse_support=True,
        )

        # Welcome message
        from copex.ui import print_welcome

        print_welcome(self.console, self._config.model.value, self._config.reasoning_effort.value)

        async def spinner_loop() -> None:
            while self._running:
                self._spinner_frame = (self._spinner_frame + 1) % 10000
                if self._app:
                    self._app.invalidate()
                await asyncio.sleep(0.08)

        async def input_loop() -> None:
            while self._running:
                if self.state.input_buffer and not self.state.session.is_streaming:
                    prompt = self.state.input_buffer
                    self.state.input_buffer = ""
                    self._current_send_task = asyncio.create_task(self._process_message(prompt))
                    await self._current_send_task
                await asyncio.sleep(0.05)

        spinner_task = asyncio.create_task(spinner_loop())
        input_task = asyncio.create_task(input_loop())

        try:
            await self._app.run_async()
        except (KeyboardInterrupt, EOFError):
            pass
        finally:
            self._running = False
            for task in (spinner_task, input_task):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            await self._client.stop()
            self._running = False

    def _extract_usage(
        self, response: Any
    ) -> tuple[int | None, float | None, int | None, int | None]:
        """Extract *real* token usage and cost from a response.

        Important: The metrics subsystem contains token/cost *estimates*.
        For the TUI status bar we prefer to show "unknown" rather than an
        incorrect number.
        """
        prompt_tokens: int | None = None
        completion_tokens: int | None = None

        usage = getattr(response, "usage", None)
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
        else:
            prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
            completion_tokens = getattr(usage, "completion_tokens", None) if usage else None

        total_tokens: int | None = None
        if prompt_tokens is not None or completion_tokens is not None:
            total_tokens = int(prompt_tokens or 0) + int(completion_tokens or 0)

        cost_val = getattr(response, "cost", None)
        cost: float | None = None
        if cost_val is not None:
            try:
                cost = float(cost_val)
            except (TypeError, ValueError):
                cost = None

        return total_tokens, cost, prompt_tokens, completion_tokens


async def run_tui(config: Any = None) -> None:
    """Run the Copex TUI.

    Args:
        config: CopexConfig instance (optional)
    """
    app = TuiApp()
    await app.run(config)


def main() -> None:
    """Entry point for the TUI."""
    import asyncio

    asyncio.run(run_tui())


if __name__ == "__main__":
    main()
