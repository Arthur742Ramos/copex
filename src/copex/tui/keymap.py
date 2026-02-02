"""Keybinding definitions for Copex TUI."""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable

# NOTE: Avoid __future__.annotations for standalone import in tests (Python 3.14
# dataclasses resolves string annotations via sys.modules).

# prompt_toolkit is optional at module load time
if TYPE_CHECKING:
    from prompt_toolkit.key_binding import KeyBindings


class Action(str, Enum):
    """TUI actions that can be bound to keys."""

    # Input actions
    SEND = "send"
    NEWLINE = "newline"
    CANCEL = "cancel"
    CLEAR_INPUT = "clear_input"

    # Navigation
    HISTORY_PREV = "history_prev"
    HISTORY_NEXT = "history_next"
    SCROLL_UP = "scroll_up"
    SCROLL_DOWN = "scroll_down"

    # Palette
    OPEN_PALETTE = "open_palette"
    CLOSE_PALETTE = "close_palette"
    PALETTE_UP = "palette_up"
    PALETTE_DOWN = "palette_down"
    PALETTE_SELECT = "palette_select"

    # Stash
    STASH_SAVE = "stash_save"
    STASH_RESTORE = "stash_restore"
    STASH_CYCLE = "stash_cycle"

    # View toggles
    TOGGLE_REASONING = "toggle_reasoning"
    TOGGLE_TOOLS = "toggle_tools"
    TOGGLE_STATUSBAR = "toggle_statusbar"
    EXPAND_TOOLS = "expand_tools"
    COLLAPSE_TOOLS = "collapse_tools"

    # Session
    NEW_SESSION = "new_session"
    CLEAR_SCREEN = "clear_screen"

    # Quick model/reasoning
    MODEL_PICKER = "model_picker"
    REASONING_PICKER = "reasoning_picker"

    # Exit
    EXIT = "exit"


@dataclass
class KeyBinding:
    """A single key binding."""

    keys: tuple[str, ...]
    action: Action
    description: str
    when: str = "always"  # always, input, palette


# Default key bindings
DEFAULT_BINDINGS: list[KeyBinding] = [
    # Input actions
    KeyBinding(
        keys=("enter",),
        action=Action.SEND,
        description="Send message (when input has content)",
        when="input",
    ),
    KeyBinding(
        keys=("c-j",),  # Ctrl+J
        action=Action.NEWLINE,
        description="Insert newline (primary)",
        when="input",
    ),
    KeyBinding(
        keys=("escape", "enter"),
        action=Action.NEWLINE,
        description="Insert newline (Shift+Enter in many terminals; sent as Esc then Enter)",
        when="input",
    ),
    KeyBinding(
        keys=("c-c",),
        action=Action.CANCEL,
        description="Cancel current operation",
        when="always",
    ),
    KeyBinding(
        keys=("c-u",),
        action=Action.CLEAR_INPUT,
        description="Clear input buffer",
        when="input",
    ),
    # Navigation
    KeyBinding(
        keys=("up",),
        action=Action.HISTORY_PREV,
        description="Previous history item",
        when="input",
    ),
    KeyBinding(
        keys=("down",),
        action=Action.HISTORY_NEXT,
        description="Next history item",
        when="input",
    ),
    KeyBinding(
        keys=("pageup",),
        action=Action.SCROLL_UP,
        description="Scroll up",
        when="always",
    ),
    KeyBinding(
        keys=("pagedown",),
        action=Action.SCROLL_DOWN,
        description="Scroll down",
        when="always",
    ),
    # Palette
    KeyBinding(
        keys=("c-p",),
        action=Action.OPEN_PALETTE,
        description="Open command palette",
        when="input",
    ),
    KeyBinding(
        keys=("escape",),
        action=Action.CLOSE_PALETTE,
        description="Close palette / Cancel",
        when="palette",
    ),
    KeyBinding(
        keys=("up",),
        action=Action.PALETTE_UP,
        description="Move selection up",
        when="palette",
    ),
    KeyBinding(
        keys=("down",),
        action=Action.PALETTE_DOWN,
        description="Move selection down",
        when="palette",
    ),
    KeyBinding(
        keys=("enter",),
        action=Action.PALETTE_SELECT,
        description="Select item",
        when="palette",
    ),
    # Stash
    KeyBinding(
        keys=("c-s",),
        action=Action.STASH_SAVE,
        description="Save prompt to stash",
        when="input",
    ),
    KeyBinding(
        keys=("c-r",),
        action=Action.STASH_RESTORE,
        description="Restore from stash",
        when="input",
    ),
    KeyBinding(
        keys=("c-x", "c-r"),  # Ctrl+X, Ctrl+R sequence
        action=Action.STASH_CYCLE,
        description="Cycle through stash",
        when="input",
    ),
    # View toggles
    KeyBinding(
        keys=("c-t",),
        action=Action.TOGGLE_TOOLS,
        description="Toggle tool calls view",
        when="always",
    ),
    KeyBinding(
        keys=("c-g",),  # Ctrl+G for reasoning (G for "thinking")
        action=Action.TOGGLE_REASONING,
        description="Toggle reasoning view",
        when="always",
    ),
    # Quick pickers
    KeyBinding(
        keys=("c-m",),
        action=Action.MODEL_PICKER,
        description="Quick model picker",
        when="input",
    ),
    KeyBinding(
        keys=("c-e",),
        action=Action.REASONING_PICKER,
        description="Quick reasoning picker",
        when="input",
    ),
    # Session
    KeyBinding(
        keys=("c-n",),
        action=Action.NEW_SESSION,
        description="Start new session",
        when="input",
    ),
    KeyBinding(
        keys=("c-l",),
        action=Action.CLEAR_SCREEN,
        description="Clear screen",
        when="always",
    ),
    # Exit
    KeyBinding(
        keys=("c-d",),
        action=Action.EXIT,
        description="Exit (when input empty)",
        when="input",
    ),
    KeyBinding(
        keys=("c-q",),
        action=Action.EXIT,
        description="Exit immediately",
        when="always",
    ),
]


class KeymapManager:
    """Manages keybindings for the TUI."""

    def __init__(self) -> None:
        self.bindings = list(DEFAULT_BINDINGS)
        self._handlers: dict[Action, Callable[[], None]] = {}

    def register_handler(self, action: Action, handler: Callable[[], None]) -> None:
        """Register a handler for an action."""
        self._handlers[action] = handler

    def get_handler(self, action: Action) -> Callable[[], None] | None:
        """Get the handler for an action."""
        return self._handlers.get(action)

    def get_bindings_for_context(self, context: str) -> list[KeyBinding]:
        """Get bindings that apply to a context (always, input, palette)."""
        return [b for b in self.bindings if b.when == context or b.when == "always"]

    def get_shortcut_display(self, action: Action) -> str:
        """Get human-readable shortcut for an action."""
        for binding in self.bindings:
            if binding.action == action:
                return self._format_keys(binding.keys)
        return ""

    def _format_keys(self, keys: tuple[str, ...]) -> str:
        """Format keys for display."""
        result = []
        for key in keys:
            if key.startswith("c-"):
                result.append(f"Ctrl+{key[2:].upper()}")
            elif key.startswith("s-"):
                result.append(f"Shift+{key[2:].upper()}")
            elif key.startswith("a-"):
                result.append(f"Alt+{key[2:].upper()}")
            else:
                result.append(key.capitalize())
        return " ".join(result)

    def get_help_text(self) -> list[tuple[str, str, str]]:
        """Get list of (shortcut, description, context) for help display."""
        seen_actions = set()
        result = []
        for binding in self.bindings:
            if binding.action not in seen_actions:
                seen_actions.add(binding.action)
                shortcut = self._format_keys(binding.keys)
                context = binding.when
                result.append((shortcut, binding.description, context))
        return result

    def build_prompt_toolkit_bindings(
        self,
        get_mode: Callable[[], str],
    ) -> "KeyBindings":
        """Build prompt_toolkit KeyBindings from our bindings."""
        # Import at runtime to avoid import errors when prompt_toolkit is not installed
        from prompt_toolkit.key_binding import KeyBindings

        kb = KeyBindings()

        for binding in self.bindings:
            self._add_binding(kb, binding, get_mode)

        return kb

    def _add_binding(
        self,
        kb: "KeyBindings",
        binding: KeyBinding,
        get_mode: Callable[[], str],
    ) -> None:
        """Add a single binding to the KeyBindings object."""
        keys = binding.keys
        action = binding.action
        when = binding.when

        # Convert key strings to prompt_toolkit format
        pt_keys = self._convert_keys(keys)

        @kb.add(*pt_keys)
        def handler(event, action=action, when=when) -> None:
            current_mode = get_mode()

            # Check if binding applies to current mode
            if when != "always" and when != current_mode:
                # Pass through for default behavior
                event.app.current_buffer.insert_text(event.data)
                return

            # Get and call the handler
            handler_fn = self._handlers.get(action)
            if handler_fn:
                handler_fn()

    def _convert_keys(self, keys: tuple[str, ...]) -> tuple:
        """Convert our key format to prompt_toolkit format."""
        # Import at runtime
        from prompt_toolkit.keys import Keys

        result = []
        for key in keys:
            if key == "enter":
                result.append(Keys.Enter)
            elif key == "escape":
                result.append(Keys.Escape)
            elif key == "up":
                result.append(Keys.Up)
            elif key == "down":
                result.append(Keys.Down)
            elif key == "left":
                result.append(Keys.Left)
            elif key == "right":
                result.append(Keys.Right)
            elif key == "pageup":
                result.append(Keys.PageUp)
            elif key == "pagedown":
                result.append(Keys.PageDown)
            elif key == "tab":
                result.append(Keys.Tab)
            elif key == "s-tab":
                result.append(Keys.BackTab)
            elif key == "s-enter":
                result.append("s-enter")
            elif key.startswith("c-"):
                # Control key
                result.append(f"c-{key[2:]}")
            elif key.startswith("s-"):
                # Shift key (for letters)
                result.append(f"s-{key[2:]}")
            elif key.startswith("a-"):
                # Alt key
                result.append(f"escape {key[2:]}")
            else:
                result.append(key)
        return tuple(result)
