"""TUI state management for Copex."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

# NOTE:
# These modules are intentionally importable as standalone files in tests
# (via importlib.util.module_from_spec) without being registered in sys.modules.
# Python 3.14's dataclasses will crash when __future__.annotations is enabled in
# such a loading mode (all annotations become strings, which dataclasses tries
# to resolve via sys.modules). So we avoid __future__.annotations here.
from copex.models import Model, ReasoningEffort

DEFAULT_MODEL: Model | str = Model.CLAUDE_OPUS_4_5
DEFAULT_REASONING: ReasoningEffort | str = ReasoningEffort.XHIGH

if TYPE_CHECKING:
    pass


class TuiMode(str, Enum):
    """Current TUI mode."""

    NORMAL = "normal"
    PALETTE = "palette"
    PICKER = "picker"


class PanelState(str, Enum):
    """State of a collapsible panel."""

    COLLAPSED = "collapsed"
    EXPANDED = "expanded"


@dataclass
class ToolCallState:
    """State of a tool call with collapse tracking."""

    tool_id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str | None = None
    status: str = "running"  # running, success, error
    started_at: float = field(default_factory=time.time)
    duration: float | None = None
    panel_state: PanelState = PanelState.COLLAPSED

    @property
    def is_expanded(self) -> bool:
        return self.panel_state == PanelState.EXPANDED

    def toggle(self) -> None:
        """Toggle between collapsed and expanded."""
        if self.panel_state == PanelState.COLLAPSED:
            self.panel_state = PanelState.EXPANDED
        else:
            self.panel_state = PanelState.COLLAPSED

    @property
    def elapsed(self) -> float:
        if self.duration is not None:
            return self.duration
        return time.time() - self.started_at

    @property
    def icon(self) -> str:
        """Get appropriate icon for the tool."""
        from copex.ui import Icons

        name_lower = self.name.lower()
        if "read" in name_lower or "view" in name_lower:
            return Icons.FILE_READ
        elif "write" in name_lower or "edit" in name_lower:
            return Icons.FILE_WRITE
        elif "create" in name_lower:
            return Icons.FILE_CREATE
        elif "search" in name_lower or "grep" in name_lower or "glob" in name_lower:
            return Icons.SEARCH
        elif "shell" in name_lower or "bash" in name_lower:
            return Icons.TERMINAL
        elif "web" in name_lower or "fetch" in name_lower:
            return Icons.GLOBE
        return Icons.TOOL

    @property
    def summary(self) -> str:
        """One-line summary for collapsed state."""
        status_icon = (
            "⏳" if self.status == "running" else ("✓" if self.status == "success" else "✗")
        )
        status_label = (
            "Running"
            if self.status == "running"
            else ("OK" if self.status == "success" else "Failed")
        )
        duration = f" ({self.elapsed:.1f}s)" if self.elapsed else ""
        return f"{status_icon} {self.icon} {self.name} • {status_label}{duration}"


@dataclass
class SessionState:
    """State for a Copex session."""

    model: Model | str = DEFAULT_MODEL
    reasoning_effort: ReasoningEffort | str = DEFAULT_REASONING

    # Metrics (only updated when real usage/cost are provided by the SDK)
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost: float = 0.0
    request_count: int = 0

    # Whether totals are complete. If any request lacks real usage/cost, we mark
    # the aggregates as unknown rather than showing misleading zeros/estimates.
    tokens_complete: bool = True
    cost_complete: bool = True

    # Timing
    session_start: float = field(default_factory=time.time)
    last_request_start: float | None = None
    last_request_duration: float | None = None

    # Activity
    is_streaming: bool = False
    is_thinking: bool = False
    current_activity: str = "idle"

    def start_request(self) -> None:
        """Mark the start of a new request."""
        self.last_request_start = time.time()
        self.is_streaming = True
        self.is_thinking = True
        self.request_count += 1

    def end_request(
        self,
        *,
        tokens: int | None = None,
        cost: float | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        """Mark the end of a request."""
        if self.last_request_start:
            self.last_request_duration = time.time() - self.last_request_start
        self.is_streaming = False
        self.is_thinking = False
        self.current_activity = "idle"

        # Prefer explicit prompt/completion token counts when provided.
        if tokens is None and (prompt_tokens is not None or completion_tokens is not None):
            tokens = int(prompt_tokens or 0) + int(completion_tokens or 0)

        if prompt_tokens is not None:
            self.prompt_tokens += int(prompt_tokens)
        if completion_tokens is not None:
            self.completion_tokens += int(completion_tokens)

        if tokens is None:
            self.tokens_complete = False
        else:
            self.total_tokens += int(tokens)

        if cost is None:
            self.cost_complete = False
        else:
            self.total_cost += float(cost)

    @property
    def session_duration(self) -> float:
        """Total session duration in seconds."""
        return time.time() - self.session_start

    @property
    def session_duration_str(self) -> str:
        """Human-readable session duration."""
        duration = self.session_duration
        if duration < 60:
            return f"{duration:.0f}s"
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        if minutes < 60:
            return f"{minutes}m {seconds:02d}s"
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours}h {minutes:02d}m"

    @property
    def model_label(self) -> str:
        """String label for current model."""
        return self.model.value if hasattr(self.model, "value") else str(self.model)

    @property
    def reasoning_label(self) -> str:
        """String label for current reasoning effort."""
        return (
            self.reasoning_effort.value
            if hasattr(self.reasoning_effort, "value")
            else str(self.reasoning_effort)
        )


@dataclass
class PromptDraft:
    """A saved prompt draft."""

    content: str
    cursor_position: int = 0
    created_at: float = field(default_factory=time.time)


@dataclass
class TuiState:
    """Complete TUI state."""

    # Session
    session: SessionState = field(default_factory=SessionState)

    # Current mode
    mode: TuiMode = TuiMode.NORMAL

    # Input state
    input_buffer: str = ""
    cursor_position: int = 0

    # Prompt stash (for Ctrl+S/Ctrl+R)
    stashed_drafts: list[PromptDraft] = field(default_factory=list)
    stash_index: int = -1
    max_stash_size: int = 10

    # Current response
    current_message: str = ""
    current_reasoning: str = ""
    tool_calls: list[ToolCallState] = field(default_factory=list)

    # Conversation history
    messages: list[dict[str, Any]] = field(default_factory=list)

    # Palette state
    palette_query: str = ""
    palette_selected: int = 0

    # UI preferences
    show_status_bar: bool = True
    show_reasoning: bool = True
    expand_all_tools: bool = False

    # Callbacks
    on_state_change: Callable[[], None] | None = None

    def notify_change(self) -> None:
        """Notify listeners of state change."""
        if self.on_state_change:
            self.on_state_change()

    # Stash operations
    def stash_prompt(self) -> bool:
        """Save current input to stash. Returns True if saved."""
        if not self.input_buffer.strip():
            return False

        draft = PromptDraft(
            content=self.input_buffer,
            cursor_position=self.cursor_position,
        )

        # Remove oldest if at max size
        if len(self.stashed_drafts) >= self.max_stash_size:
            self.stashed_drafts.pop(0)

        self.stashed_drafts.append(draft)
        self.stash_index = len(self.stashed_drafts) - 1
        self.input_buffer = ""
        self.cursor_position = 0
        self.notify_change()
        return True

    def restore_stash(self, index: int | None = None) -> bool:
        """Restore a stashed prompt. Returns True if restored."""
        if not self.stashed_drafts:
            return False

        if index is None:
            # Use last stash by default
            index = len(self.stashed_drafts) - 1

        if 0 <= index < len(self.stashed_drafts):
            draft = self.stashed_drafts[index]
            self.input_buffer = draft.content
            self.cursor_position = draft.cursor_position
            self.stash_index = index
            self.notify_change()
            return True

        return False

    def pop_stash(self) -> PromptDraft | None:
        """Pop the most recent stash."""
        if not self.stashed_drafts:
            return None

        draft = self.stashed_drafts.pop()
        self.input_buffer = draft.content
        self.cursor_position = draft.cursor_position
        self.stash_index = len(self.stashed_drafts) - 1
        self.notify_change()
        return draft

    def cycle_stash(self, direction: int = -1) -> bool:
        """Cycle through stash. Returns True if changed."""
        if not self.stashed_drafts:
            return False

        new_index = (self.stash_index + direction) % len(self.stashed_drafts)
        return self.restore_stash(new_index)

    # Tool call operations
    def add_tool_call(
        self, tool_id: str | None = None, name: str = "", arguments: dict[str, Any] | None = None
    ) -> None:
        """Add a new tool call."""
        self.tool_calls.append(
            ToolCallState(tool_id=tool_id or "", name=name, arguments=arguments or {})
        )
        self.notify_change()

    def update_tool_call(self, tool_id: str | None = None, **kwargs: Any) -> None:
        """Update a tool call by tool_id."""
        if tool_id is None:
            # Fallback to updating last running by name if no id provided
            for tc in self.tool_calls:
                if tc.name == kwargs.get("name"):
                    for key, value in kwargs.items():
                        if hasattr(tc, key):
                            setattr(tc, key, value)
                    break
        else:
            for tc in self.tool_calls:
                if tc.tool_id == tool_id:
                    for key, value in kwargs.items():
                        if hasattr(tc, key):
                            setattr(tc, key, value)
                    break
        self.notify_change()

    def toggle_tool_call(self, index: int) -> None:
        """Toggle a tool call's expanded state."""
        if 0 <= index < len(self.tool_calls):
            self.tool_calls[index].toggle()
            self.notify_change()

    def expand_all_tool_calls(self) -> None:
        """Expand all tool calls."""
        for tc in self.tool_calls:
            tc.panel_state = PanelState.EXPANDED
        self.expand_all_tools = True
        self.notify_change()

    def collapse_all_tool_calls(self) -> None:
        """Collapse all tool calls."""
        for tc in self.tool_calls:
            tc.panel_state = PanelState.COLLAPSED
        self.expand_all_tools = False
        self.notify_change()

    # Response operations
    def clear_current_response(self) -> None:
        """Clear the current response state."""
        self.current_message = ""
        self.current_reasoning = ""
        self.tool_calls = []
        self.notify_change()

    def add_message_delta(self, delta: str) -> None:
        """Add a delta to the current message."""
        self.current_message += delta
        self.session.current_activity = "responding"
        self.session.is_thinking = False
        self.notify_change()

    def add_reasoning_delta(self, delta: str) -> None:
        """Add a delta to the current reasoning."""
        self.current_reasoning += delta
        self.session.current_activity = "reasoning"
        self.notify_change()

    # Palette operations
    def open_palette(self) -> None:
        """Open the command palette."""
        self.mode = TuiMode.PALETTE
        self.palette_query = ""
        self.palette_selected = 0
        self.notify_change()

    def close_palette(self) -> None:
        """Close the command palette."""
        self.mode = TuiMode.NORMAL
        self.palette_query = ""
        self.palette_selected = 0
        self.notify_change()

    def update_palette_query(self, query: str) -> None:
        """Update the palette search query."""
        self.palette_query = query
        self.palette_selected = 0
        self.notify_change()

    def move_palette_selection(self, direction: int, max_index: int | None = None) -> None:
        """Move palette selection up or down.

        The selection is clamped to [0, max_index]. If max_index < 0 (empty
        result list), we keep the selection at 0.
        """
        next_index = self.palette_selected + direction
        if max_index is not None:
            if max_index < 0:
                self.palette_selected = 0
                self.notify_change()
                return
            next_index = max(0, min(next_index, max_index))
        else:
            next_index = max(0, next_index)
        self.palette_selected = next_index
        self.notify_change()
