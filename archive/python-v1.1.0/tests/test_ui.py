"""Tests for UI components."""

from __future__ import annotations

import time

import pytest

from copex.ui import ActivityType, CopexUI, Theme, ToolCallInfo


class TestCopexUISpinner:
    """Tests for spinner animation (Codex CLI inspired)."""

    def test_spinner_icons_have_braille_frames(self):
        """Icons class should have braille spinner frames."""
        from copex.ui import Icons
        assert len(Icons.BRAILLE_SPINNER) == 10
        assert Icons.BRAILLE_SPINNER == ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def test_spinner_advances_after_interval(self):
        """Spinner should advance after 32ms frame interval."""
        ui = CopexUI()
        initial_idx = ui._frame_idx

        # Force time to pass
        ui._last_frame_at = time.time() - 0.05  # 50ms ago (> 32ms)
        ui._advance_frame()

        assert ui._frame_idx == (initial_idx + 1) % 60

    def test_spinner_does_not_advance_too_quickly(self):
        """Spinner should not advance if less than 32ms passed."""
        ui = CopexUI()
        ui._advance_frame()  # First advance
        initial_idx = ui._frame_idx

        # Immediately try again
        ui._advance_frame()

        assert ui._frame_idx == initial_idx  # Should not have changed

    def test_get_spinner_returns_icon(self):
        """_get_spinner should return a spinner icon."""
        ui = CopexUI()
        from copex.ui import Icons
        spinner = ui._get_spinner()
        # Should return a braille spinner frame or bullet
        assert spinner in Icons.BRAILLE_SPINNER or spinner == Icons.BULLET


class TestCopexUIMessagePanel:
    """Tests for message panel building."""

    def test_message_panel_shows_full_content(self):
        """Message panel should show full content without truncation."""
        ui = CopexUI()
        long_message = "x" * 5000  # Long message
        ui.state.message = long_message

        panel = ui._build_message_panel()

        assert panel is not None
        # The content should contain the full message (no truncation)
        # Panel's renderable is a Text object
        assert long_message in str(panel.renderable)

    def test_message_panel_none_when_empty(self):
        """Message panel should be None when no message."""
        ui = CopexUI()
        ui.state.message = ""

        panel = ui._build_message_panel()

        assert panel is None

    def test_message_panel_shows_cursor_when_responding(self):
        """Message panel should show cursor when actively responding."""
        ui = CopexUI()
        ui.state.message = "Hello"
        ui.state.activity = ActivityType.RESPONDING

        panel = ui._build_message_panel()

        assert panel is not None
        assert "▌" in str(panel.renderable)


class TestCopexUIState:
    """Tests for UI state management."""

    def test_add_message_updates_activity(self):
        """Adding message should set activity to RESPONDING."""
        ui = CopexUI()
        ui.state.activity = ActivityType.THINKING

        ui.add_message("Hello")

        assert ui.state.activity == ActivityType.RESPONDING
        assert ui.state.message == "Hello"

    def test_add_reasoning_updates_activity(self):
        """Adding reasoning should set activity to REASONING."""
        ui = CopexUI()
        ui.state.activity = ActivityType.THINKING

        ui.add_reasoning("Thinking...")

        assert ui.state.activity == ActivityType.REASONING
        assert ui.state.reasoning == "Thinking..."

    def test_reset_clears_state(self):
        """Reset should clear all state."""
        ui = CopexUI()
        ui.state.message = "Hello"
        ui.state.reasoning = "Thinking"
        ui.state.activity = ActivityType.RESPONDING

        ui.reset(model="gpt-4")

        assert ui.state.message == ""
        assert ui.state.reasoning == ""
        assert ui.state.activity == ActivityType.WAITING
        assert ui.state.model == "gpt-4"


def test_ui_dirty_flag_consumed():
    """Dirty flag should toggle when UI state updates."""
    ui = CopexUI()
    assert ui.consume_dirty() is True
    assert ui.consume_dirty() is False
    ui.add_message("Hi")
    assert ui.consume_dirty() is True


class TestStreamingContext:
    """Tests for StreamingContext double-buffered streaming."""

    def test_streaming_context_creation(self):
        """Should create a streaming context."""
        ui = CopexUI()
        ctx = ui.streaming_context()
        assert ctx is not None
        assert ctx.ui is ui

    def test_streaming_context_enter_exit(self):
        """Should properly enter and exit context."""
        from rich.console import Console
        # Force non-TTY to avoid actual Live display in tests
        console = Console(force_terminal=False)
        ui = CopexUI(console=console)
        
        with ui.streaming_context() as ctx:
            assert ctx is not None
            # Non-TTY should have no live display
            assert ctx._live is None

    def test_streaming_context_refresh_rate_limiting(self):
        """Should respect rate limiting on refresh."""
        from rich.console import Console
        console = Console(force_terminal=False)
        ui = CopexUI(console=console)
        
        with ui.streaming_context(refresh_per_second=10) as ctx:
            # First refresh should succeed
            ui.add_message("test")
            result1 = ctx.refresh()
            assert result1 is True
            
            # Immediate second refresh should be rate-limited
            result2 = ctx.refresh()
            assert result2 is False  # Rate limited

    def test_streaming_context_dirty_flag_optimization(self):
        """Should only refresh when content has changed."""
        from rich.console import Console
        console = Console(force_terminal=False)
        ui = CopexUI(console=console)
        
        with ui.streaming_context() as ctx:
            # Add content - sets dirty flag
            ui.add_message("test")
            
            # Sleep to exceed rate limit
            import time
            time.sleep(0.05)
            
            # First non-forced refresh should succeed (dirty flag is True)
            result1 = ctx.refresh()
            assert result1 is True  # Dirty flag was True
            
            # Dirty flag should now be consumed
            time.sleep(0.05)  # Wait for rate limit
            
            # Second refresh without changes should fail
            result2 = ctx.refresh()
            assert result2 is False  # Nothing dirty anymore

    def test_streaming_context_force_refresh(self):
        """Should force refresh when requested."""
        from rich.console import Console
        console = Console(force_terminal=False)
        ui = CopexUI(console=console)
        
        with ui.streaming_context() as ctx:
            # Force should work even without dirty flag
            result = ctx.refresh(force=True)
            assert result is True

    def test_streaming_context_frame_count(self):
        """Should track frame count."""
        from rich.console import Console
        console = Console(force_terminal=False)
        ui = CopexUI(console=console)
        
        with ui.streaming_context() as ctx:
            assert ctx.frame_count == 0
            ctx.refresh(force=True)
            assert ctx.frame_count == 1
            ctx.refresh(force=True)
            assert ctx.frame_count == 2

    def test_streaming_context_is_active(self):
        """Should report active state correctly."""
        from rich.console import Console
        console = Console(force_terminal=False)
        ui = CopexUI(console=console)
        
        ctx = ui.streaming_context()
        assert ctx.is_active is False
        
        with ctx:
            # Non-TTY means no Live, but context is "entered"
            pass
        
        assert ctx.is_active is False

    def test_streaming_context_builds_frames(self):
        """Should build complete frames for display."""
        from rich.console import Console
        console = Console(force_terminal=False)
        ui = CopexUI(console=console)
        ui.add_message("Hello world")
        
        with ui.streaming_context() as ctx:
            ctx.refresh(force=True)
            # Frame should be built
            assert ctx._current_frame is not None


class TestThemeDetection:
    """Tests for terminal theme detection."""

    def test_terminal_info_detect(self):
        """TerminalInfo.detect() should return detection results."""
        from copex.ui import TerminalInfo
        
        TerminalInfo.reset_cache()
        info = TerminalInfo.detect()
        
        assert "is_tty" in info
        assert "is_light" in info
        assert "supports_true_color" in info
        assert "supports_256_color" in info
        assert "term_program" in info
        assert "no_color" in info

    def test_terminal_info_caches_results(self):
        """TerminalInfo should cache detection results."""
        from copex.ui import TerminalInfo
        
        TerminalInfo.reset_cache()
        info1 = TerminalInfo.detect()
        info2 = TerminalInfo.detect()
        
        assert info1 is info2  # Same object (cached)

    def test_terminal_info_reset_cache(self):
        """TerminalInfo.reset_cache() should clear cache."""
        from copex.ui import TerminalInfo
        
        TerminalInfo.reset_cache()
        info1 = TerminalInfo.detect()
        
        TerminalInfo.reset_cache()
        info2 = TerminalInfo.detect()
        
        assert info1 is not info2  # Different objects

    def test_is_light_theme_function(self):
        """is_light_theme() should return boolean."""
        from copex.ui import is_light_theme, TerminalInfo
        
        TerminalInfo.reset_cache()
        result = is_light_theme()
        
        assert isinstance(result, bool)

    def test_supports_color_functions(self):
        """supports_256_color and supports_true_color should return booleans."""
        from copex.ui import supports_256_color, supports_true_color, TerminalInfo
        
        TerminalInfo.reset_cache()
        
        assert isinstance(supports_256_color(), bool)
        assert isinstance(supports_true_color(), bool)

    def test_apply_theme_valid(self):
        """apply_theme() should apply a valid theme."""
        from copex.ui import apply_theme, Theme
        
        # Save originals
        orig_primary = Theme.PRIMARY
        
        result = apply_theme("midnight")
        
        assert result is True
        assert Theme.PRIMARY == "bright_cyan"
        
        # Restore
        apply_theme("default")
        assert Theme.PRIMARY == "cyan"

    def test_apply_theme_invalid(self):
        """apply_theme() should return False for invalid theme."""
        from copex.ui import apply_theme
        
        result = apply_theme("nonexistent-theme")
        
        assert result is False

    def test_list_themes(self):
        """list_themes() should return list of theme names."""
        from copex.ui import list_themes
        
        themes = list_themes()
        
        assert isinstance(themes, list)
        assert "default" in themes
        assert "light" in themes
        assert "dark-256" in themes
        assert "light-256" in themes
        assert "high-contrast-dark" in themes
        assert "github-dark" in themes
        assert "solarized-light" in themes

    def test_get_theme_for_terminal(self):
        """get_theme_for_terminal() should return a theme name."""
        from copex.ui import get_theme_for_terminal, THEME_PRESETS, TerminalInfo
        
        TerminalInfo.reset_cache()
        theme_name = get_theme_for_terminal()
        
        assert theme_name in THEME_PRESETS

    def test_auto_apply_theme(self):
        """auto_apply_theme() should apply and return theme name."""
        from copex.ui import auto_apply_theme, THEME_PRESETS, TerminalInfo
        
        TerminalInfo.reset_cache()
        theme_name = auto_apply_theme()
        
        assert theme_name in THEME_PRESETS

    def test_get_recommended_theme(self):
        """get_recommended_theme() should return valid theme."""
        from copex.ui import get_recommended_theme, THEME_PRESETS, TerminalInfo
        
        TerminalInfo.reset_cache()
        theme_name = get_recommended_theme()
        
        assert theme_name in THEME_PRESETS

    def test_get_theme_preview(self):
        """get_theme_preview() should return preview string."""
        from copex.ui import get_theme_preview
        
        preview = get_theme_preview("default")
        
        assert "Theme: default" in preview
        assert "PRIMARY" in preview
        assert "SUCCESS" in preview

    def test_get_theme_preview_invalid(self):
        """get_theme_preview() should handle invalid theme."""
        from copex.ui import get_theme_preview
        
        preview = get_theme_preview("nonexistent")
        
        assert "not found" in preview

    def test_theme_presets_have_required_keys(self):
        """All theme presets should have required keys."""
        from copex.ui import THEME_PRESETS
        
        required = {
            "PRIMARY", "SECONDARY", "ACCENT", "SUCCESS", "WARNING", "ERROR",
            "INFO", "REASONING", "MESSAGE", "CODE", "MUTED", "BORDER",
            "BORDER_ACTIVE", "HEADER", "SUBHEADER"
        }
        
        for name, preset in THEME_PRESETS.items():
            for key in required:
                assert key in preset, f"Theme '{name}' missing key '{key}'"

    def test_256_color_themes_use_color_codes(self):
        """256-color themes should use color(N) format."""
        from copex.ui import THEME_PRESETS
        
        for name in ["dark-256", "light-256"]:
            preset = THEME_PRESETS[name]
            # At least some should use color codes
            color_count = sum(1 for v in preset.values() if "color(" in str(v))
            assert color_count >= 5, f"Theme '{name}' should use more color codes"


class TestThemeDetectionWithEnv:
    """Tests for theme detection with environment variable mocking."""

    def test_copex_theme_env_light(self, monkeypatch):
        """COPEX_THEME=light should force light mode."""
        from copex.ui import TerminalInfo, _detect_light_theme
        
        monkeypatch.setenv("COPEX_THEME", "light")
        TerminalInfo.reset_cache()
        
        assert _detect_light_theme() is True

    def test_copex_theme_env_dark(self, monkeypatch):
        """COPEX_THEME=dark should force dark mode."""
        from copex.ui import TerminalInfo, _detect_light_theme
        
        monkeypatch.setenv("COPEX_THEME", "dark")
        TerminalInfo.reset_cache()
        
        assert _detect_light_theme() is False

    def test_colorfgbg_light_detection(self, monkeypatch):
        """COLORFGBG with light bg should detect light."""
        from copex.ui import TerminalInfo, _detect_light_theme
        
        monkeypatch.delenv("COPEX_THEME", raising=False)
        monkeypatch.setenv("COLORFGBG", "0;15")  # Black on white
        TerminalInfo.reset_cache()
        
        assert _detect_light_theme() is True

    def test_colorfgbg_dark_detection(self, monkeypatch):
        """COLORFGBG with dark bg should detect dark."""
        from copex.ui import TerminalInfo, _detect_light_theme
        
        monkeypatch.delenv("COPEX_THEME", raising=False)
        monkeypatch.setenv("COLORFGBG", "15;0")  # White on black
        TerminalInfo.reset_cache()
        
        assert _detect_light_theme() is False

    def test_true_color_detection(self, monkeypatch):
        """COLORTERM=truecolor should enable true color."""
        from copex.ui import _supports_true_color
        
        monkeypatch.setenv("COLORTERM", "truecolor")
        
        assert _supports_true_color() is True

    def test_256_color_detection(self, monkeypatch):
        """TERM with 256color should enable 256 colors."""
        from copex.ui import _supports_256_color
        
        monkeypatch.setenv("TERM", "xterm-256color")
        monkeypatch.delenv("COLORTERM", raising=False)
        
        assert _supports_256_color() is True

    def test_no_color_env_returns_mono(self, monkeypatch):
        """NO_COLOR should return mono theme."""
        from copex.ui import get_recommended_theme, TerminalInfo
        
        monkeypatch.setenv("NO_COLOR", "1")
        TerminalInfo.reset_cache()
        
        assert get_recommended_theme() == "mono"

    def test_apple_terminal_default_light(self, monkeypatch):
        """Apple_Terminal should default to light."""
        from copex.ui import TerminalInfo, _detect_light_theme
        
        monkeypatch.delenv("COPEX_THEME", raising=False)
        monkeypatch.delenv("COLORFGBG", raising=False)
        monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
        TerminalInfo.reset_cache()
        
        assert _detect_light_theme() is True

    def test_iterm_light_profile(self, monkeypatch):
        """iTerm with 'light' in profile should detect light."""
        from copex.ui import TerminalInfo, _detect_light_theme
        
        monkeypatch.delenv("COPEX_THEME", raising=False)
        monkeypatch.delenv("COLORFGBG", raising=False)
        monkeypatch.delenv("TERM_PROGRAM", raising=False)
        monkeypatch.setenv("ITERM_PROFILE", "Solarized Light")
        TerminalInfo.reset_cache()
        
        assert _detect_light_theme() is True

    def test_iterm_dark_profile(self, monkeypatch):
        """iTerm with 'dark' in profile should detect dark."""
        from copex.ui import TerminalInfo, _detect_light_theme
        
        monkeypatch.delenv("COPEX_THEME", raising=False)
        monkeypatch.delenv("COLORFGBG", raising=False)
        monkeypatch.delenv("TERM_PROGRAM", raising=False)
        monkeypatch.setenv("ITERM_PROFILE", "Dracula")
        TerminalInfo.reset_cache()
        
        assert _detect_light_theme() is False
