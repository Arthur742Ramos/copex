"""Copex TUI - Terminal User Interface with command palette and enhanced input handling.

This module provides an enhanced TUI experience for Copex with:
- Command Palette (Ctrl+P) - searchable actions for model/reasoning/session/export
- Status bar - always visible: model, reasoning, tokens, cost, spinner
- Collapsible tool calls - one-line summary, expand on demand
- Better multiline input - Ctrl+J for newline (primary). Some terminals map Shift+Enter to Esc+Enter.
- Prompt stash - save/restore drafts (Ctrl+S/Ctrl+R)

Usage:
    from copex.tui import run_tui, TuiApp

    # Run with defaults
    await run_tui()

    # Run with custom config
    from copex import CopexConfig
    config = CopexConfig(model=Model.CLAUDE_OPUS_4_5)
    await run_tui(config)

Components:
    TuiApp - Main application class
    TuiState - Complete TUI state management
    SessionState - Session-level state (model, tokens, cost)
    CommandPalette - Searchable command palette with fuzzy matching
    KeymapManager - Keybinding management
    PromptHistory - Persistent prompt history
    PromptStash - Save/restore prompt drafts
"""

# Core components that don't require prompt_toolkit
from .history import CombinedHistoryManager, PromptHistory, PromptStash
from .keymap import Action, KeyBinding, KeymapManager
from .palette import (
    CommandCategory,
    CommandPalette,
    PaletteCommand,
    filter_commands,
    fuzzy_match,
)
from .render import (
    RenderConfig,
    render_help_panel,
    render_message,
    render_palette,
    render_status_bar,
    render_to_ansi,
    render_tool_call_collapsed,
    render_tool_call_expanded,
)
from .state import PanelState, SessionState, ToolCallState, TuiMode, TuiState

# App requires prompt_toolkit - import lazily
try:
    from .app import TuiApp, main, run_tui

    _HAS_PROMPT_TOOLKIT = True
except ImportError:
    TuiApp = None  # type: ignore
    main = None  # type: ignore
    run_tui = None  # type: ignore
    _HAS_PROMPT_TOOLKIT = False

__all__ = [
    # Main app (requires prompt_toolkit)
    "TuiApp",
    "run_tui",
    "main",
    # State
    "TuiState",
    "TuiMode",
    "SessionState",
    "ToolCallState",
    "PanelState",
    # Palette
    "CommandPalette",
    "CommandCategory",
    "PaletteCommand",
    "fuzzy_match",
    "filter_commands",
    # Keymap
    "KeymapManager",
    "KeyBinding",
    "Action",
    # History
    "PromptHistory",
    "PromptStash",
    "CombinedHistoryManager",
    # Render
    "RenderConfig",
    "render_to_ansi",
    "render_status_bar",
    "render_palette",
    "render_message",
    "render_tool_call_collapsed",
    "render_tool_call_expanded",
    "render_help_panel",
]
