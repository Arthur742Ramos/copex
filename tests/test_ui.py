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
