"""Tests for TUI module components.

These tests are designed to be standalone and not require the copilot SDK.
"""

from __future__ import annotations

import sys

import pytest

# Ensure local package imports work even when modules are loaded via
# importlib.util.spec_from_file_location.
if "src" not in sys.path:
    sys.path.insert(0, "src")


class TestTuiState:
    """Tests for TUI state management."""
    
    def test_tui_mode_enum(self) -> None:
        """Test TuiMode enum values."""
        # Import directly to avoid triggering parent package
        import sys
        sys.path.insert(0, "src")
        
        # Import the module file directly
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "state", "src/copex/tui/state.py"
        )
        state_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(state_mod)
        
        TuiMode = state_mod.TuiMode
        
        assert TuiMode.NORMAL == "normal"
        assert TuiMode.PALETTE == "palette"
        assert TuiMode.PICKER == "picker"
    
    def test_panel_state_enum(self) -> None:
        """Test PanelState enum values."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "state", "src/copex/tui/state.py"
        )
        state_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(state_mod)
        
        PanelState = state_mod.PanelState
        
        assert PanelState.COLLAPSED == "collapsed"
        assert PanelState.EXPANDED == "expanded"
    
    def test_tool_call_state(self) -> None:
        """Test ToolCallState dataclass."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "state", "src/copex/tui/state.py"
        )
        state_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(state_mod)
        
        ToolCallState = state_mod.ToolCallState
        PanelState = state_mod.PanelState
        
        tc = ToolCallState(name="view", arguments={"path": "/test"})
        
        assert tc.name == "view"
        assert tc.arguments == {"path": "/test"}
        assert tc.status == "running"
        assert tc.panel_state == PanelState.COLLAPSED
        assert not tc.is_expanded
        
        tc.toggle()
        assert tc.is_expanded
        assert tc.panel_state == PanelState.EXPANDED
        
        tc.toggle()
        assert not tc.is_expanded
    
    def test_tool_call_icon(self) -> None:
        """Test ToolCallState icon property."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "state", "src/copex/tui/state.py"
        )
        state_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(state_mod)
        
        ToolCallState = state_mod.ToolCallState
        
        assert ToolCallState(name="view").icon == "📖"
        assert ToolCallState(name="read_file").icon == "📖"
        assert ToolCallState(name="edit").icon == "📝"
        assert ToolCallState(name="write_file").icon == "📝"
        assert ToolCallState(name="create_file").icon == "📄"
        assert ToolCallState(name="grep").icon == "🔍"
        assert ToolCallState(name="glob_search").icon == "🔍"
        assert ToolCallState(name="bash").icon == "💻"
        assert ToolCallState(name="shell_command").icon == "💻"
        assert ToolCallState(name="web_fetch").icon == "🌐"
        assert ToolCallState(name="unknown_tool").icon == "⚡"
    
    def test_session_state(self) -> None:
        """Test SessionState dataclass."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "state", "src/copex/tui/state.py"
        )
        state_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(state_mod)
        
        SessionState = state_mod.SessionState
        
        session = SessionState()
        
        assert session.model == "claude-opus-4.5"
        assert session.reasoning_effort == "xhigh"
        assert session.total_tokens == 0
        assert session.total_cost == 0.0
        assert not session.is_streaming
        assert session.current_activity == "idle"
        
        session.start_request()
        assert session.is_streaming
        assert session.is_thinking
        assert session.request_count == 1
        
        session.end_request(tokens=100, cost=0.01)
        assert not session.is_streaming
        assert session.total_tokens == 100
        assert session.total_cost == 0.01
    
    def test_tui_state_stash(self) -> None:
        """Test TuiState stash operations."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "state", "src/copex/tui/state.py"
        )
        state_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(state_mod)
        
        TuiState = state_mod.TuiState
        
        state = TuiState()
        
        # Empty stash shouldn't save
        assert not state.stash_prompt()
        assert len(state.stashed_drafts) == 0
        
        # Save a draft
        state.input_buffer = "test prompt"
        assert state.stash_prompt()
        assert len(state.stashed_drafts) == 1
        assert state.input_buffer == ""
        
        # Restore
        assert state.restore_stash()
        assert state.input_buffer == "test prompt"
    
    def test_tui_state_palette(self) -> None:
        """Test TuiState palette operations."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "state", "src/copex/tui/state.py"
        )
        state_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(state_mod)
        
        TuiState = state_mod.TuiState
        TuiMode = state_mod.TuiMode
        
        state = TuiState()
        
        assert state.mode == TuiMode.NORMAL
        
        state.open_palette()
        assert state.mode == TuiMode.PALETTE
        assert state.palette_query == ""
        assert state.palette_selected == 0
        
        state.update_palette_query("model")
        assert state.palette_query == "model"
        
        state.move_palette_selection(1)
        assert state.palette_selected == 1
        
        state.close_palette()
        assert state.mode == TuiMode.NORMAL

    def test_tui_app_builds_keybindings_and_layout(self) -> None:
        """Smoke test: the TUI app constructs without invalid key bindings."""
        import sys

        sys.path.insert(0, "src")
        from copex.tui.app import TuiApp

        app = TuiApp()
        kb = app._build_keybindings()
        layout = app._build_layout()

        assert kb is not None
        assert layout is not None


class TestPalette:
    """Tests for command palette."""
    
    def test_fuzzy_match_exact(self) -> None:
        """Test exact substring matches."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "palette", "src/copex/tui/palette.py"
        )
        palette_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(palette_mod)
        
        fuzzy_match = palette_mod.fuzzy_match
        
        match, score = fuzzy_match("model", "Change Model")
        assert match
        assert score > 0
        
        match, score = fuzzy_match("Model", "Change Model")
        assert match  # Case insensitive
    
    def test_fuzzy_match_partial(self) -> None:
        """Test partial matches."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "palette", "src/copex/tui/palette.py"
        )
        palette_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(palette_mod)
        
        fuzzy_match = palette_mod.fuzzy_match
        
        match, score = fuzzy_match("cm", "Change Model")
        assert match
        
        match, score = fuzzy_match("chmd", "Change Model")
        assert match
    
    def test_fuzzy_match_no_match(self) -> None:
        """Test non-matches."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "palette", "src/copex/tui/palette.py"
        )
        palette_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(palette_mod)
        
        fuzzy_match = palette_mod.fuzzy_match
        
        match, score = fuzzy_match("xyz", "Change Model")
        assert not match
        assert score == 0
    
    def test_command_palette_search(self) -> None:
        """Test CommandPalette search."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "palette", "src/copex/tui/palette.py"
        )
        palette_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(palette_mod)
        
        CommandPalette = palette_mod.CommandPalette
        
        palette = CommandPalette()
        
        # Empty query returns all
        results = palette.search("")
        assert len(results) > 0
        
        # Search for model
        results = palette.search("model")
        assert len(results) > 0
        assert any("model" in r[0].id.lower() for r in results)
        
        # Search for session
        results = palette.search("session")
        assert len(results) > 0
        assert any("session" in r[0].id.lower() for r in results)


class TestKeymap:
    """Tests for keymap manager."""
    
    def test_default_bindings(self) -> None:
        """Test default keybindings are defined."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "keymap", "src/copex/tui/keymap.py"
        )
        keymap_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(keymap_mod)
        
        DEFAULT_BINDINGS = keymap_mod.DEFAULT_BINDINGS
        Action = keymap_mod.Action
        
        # Check key actions are bound
        actions = {b.action for b in DEFAULT_BINDINGS}
        assert Action.SEND in actions
        assert Action.NEWLINE in actions
        assert Action.OPEN_PALETTE in actions
        assert Action.STASH_SAVE in actions
        assert Action.STASH_RESTORE in actions
    
    def test_keymap_manager_shortcuts(self) -> None:
        """Test KeymapManager shortcut display."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "keymap", "src/copex/tui/keymap.py"
        )
        keymap_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(keymap_mod)
        
        KeymapManager = keymap_mod.KeymapManager
        Action = keymap_mod.Action
        
        keymap = KeymapManager()
        
        # Check shortcut display
        assert keymap.get_shortcut_display(Action.SEND) == "Enter"
        assert keymap.get_shortcut_display(Action.OPEN_PALETTE) == "Ctrl+P"
        assert keymap.get_shortcut_display(Action.STASH_SAVE) == "Ctrl+S"
    
    def test_help_text(self) -> None:
        """Test help text generation."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "keymap", "src/copex/tui/keymap.py"
        )
        keymap_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(keymap_mod)
        
        KeymapManager = keymap_mod.KeymapManager
        
        keymap = KeymapManager()
        help_items = keymap.get_help_text()
        
        assert len(help_items) > 0
        # Each item should be (shortcut, description)
        for shortcut, desc in help_items:
            assert isinstance(shortcut, str)
            assert isinstance(desc, str)
            assert len(desc) > 0


class TestHistory:
    """Tests for history and stash management."""
    
    def test_prompt_stash(self) -> None:
        """Test PromptStash operations."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "history", "src/copex/tui/history.py"
        )
        history_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(history_mod)
        
        PromptStash = history_mod.PromptStash
        
        stash = PromptStash(persist=False)
        
        # Empty stash
        assert len(stash) == 0
        assert stash.pop() is None
        
        # Push items
        assert stash.push("first")
        assert stash.push("second")
        assert len(stash) == 2
        
        # Peek
        entry = stash.peek()
        assert entry is not None
        assert entry.content == "second"
        assert len(stash) == 2  # Peek doesn't remove
        
        # Pop
        entry = stash.pop()
        assert entry is not None
        assert entry.content == "second"
        assert len(stash) == 1
        
        entry = stash.pop()
        assert entry.content == "first"
        assert len(stash) == 0
    
    def test_stash_cycling(self) -> None:
        """Test cycling through stash."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "history", "src/copex/tui/history.py"
        )
        history_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(history_mod)
        
        PromptStash = history_mod.PromptStash
        
        stash = PromptStash(persist=False)
        stash.push("a")
        stash.push("b")
        stash.push("c")
        
        # Start at last item
        assert stash.current_index == 2
        
        # Cycle prev
        entry = stash.cycle_prev()
        assert entry.content == "b"
        
        entry = stash.cycle_prev()
        assert entry.content == "a"
        
        # Wrap around
        entry = stash.cycle_prev()
        assert entry.content == "c"
    
    def test_combined_history_manager(self) -> None:
        """Test CombinedHistoryManager."""
        import importlib.util
        import tempfile
        from pathlib import Path
        
        spec = importlib.util.spec_from_file_location(
            "history", "src/copex/tui/history.py"
        )
        history_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(history_mod)
        
        CombinedHistoryManager = history_mod.CombinedHistoryManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CombinedHistoryManager(history_dir=Path(tmpdir))
            
            # Add to history
            manager.add_to_history("test prompt", model="gpt-5")
            
            # Stash draft
            assert manager.stash_draft("draft content", cursor_position=5)
            assert manager.has_stash
            assert manager.stash_count == 1
            
            # Restore draft
            result = manager.restore_draft()
            assert result is not None
            content, cursor = result
            assert content == "draft content"
            assert cursor == 5


class TestRender:
    """Tests for render helpers."""
    
    def test_theme_constants(self) -> None:
        """Test Theme class has required constants."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "render", "src/copex/tui/render.py"
        )
        render_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(render_mod)
        
        Theme = render_mod.Theme
        
        assert hasattr(Theme, "PRIMARY")
        assert hasattr(Theme, "SECONDARY")
        assert hasattr(Theme, "SUCCESS")
        assert hasattr(Theme, "ERROR")
        assert hasattr(Theme, "WARNING")
    
    def test_icons_constants(self) -> None:
        """Test Icons class has required constants."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "render", "src/copex/tui/render.py"
        )
        render_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(render_mod)
        
        Icons = render_mod.Icons
        
        assert hasattr(Icons, "DONE")
        assert hasattr(Icons, "ERROR")
        assert hasattr(Icons, "BRAIN")
        assert hasattr(Icons, "ROBOT")
        assert hasattr(Icons, "TOOL")
    
    def test_render_spinner(self) -> None:
        """Test spinner animation."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "render", "src/copex/tui/render.py"
        )
        render_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(render_mod)
        
        render_spinner = render_mod.render_spinner
        
        frames = [render_spinner(i) for i in range(10)]
        assert len(set(frames)) > 1  # Multiple unique frames
    
    def test_render_to_ansi(self) -> None:
        """Test ANSI rendering."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "render", "src/copex/tui/render.py"
        )
        render_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(render_mod)
        
        from rich.text import Text
        
        render_to_ansi = render_mod.render_to_ansi
        
        text = Text("Hello", style="bold")
        result = render_to_ansi(text)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_render_status_bar(self) -> None:
        """Test status bar rendering."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "render", "src/copex/tui/render.py"
        )
        render_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(render_mod)
        
        render_status_bar = render_mod.render_status_bar
        
        result = render_status_bar(
            model="gpt-5.2-codex",
            reasoning="xhigh",
            tokens=12345,
            cost=0.0234,
            is_streaming=True,
            activity="responding",
        )
        
        assert isinstance(result, str)
        assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
