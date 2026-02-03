"""Tests for UI components."""

from __future__ import annotations

import time

import pytest

from copex.ui import ActivityType, CopexUI, Theme, ToolCallInfo


class TestCopexUISpinner:
    """Tests for spinner animation."""

    def test_spinner_has_10_frames(self):
        """Spinner should have 10 braille frames for smooth animation."""
        ui = CopexUI()
        assert len(ui._spinners) == 10
        assert ui._spinners == ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def test_spinner_advances_after_interval(self):
        """Spinner should advance after 80ms."""
        ui = CopexUI()
        initial_idx = ui._spinner_idx

        # Force time to pass
        ui._last_frame_at = time.time() - 0.1  # 100ms ago
        ui._advance_frame()

        assert ui._spinner_idx == (initial_idx + 1) % len(ui._spinners)

    def test_spinner_does_not_advance_too_quickly(self):
        """Spinner should not advance if less than 80ms passed."""
        ui = CopexUI()
        ui._advance_frame()  # First advance
        initial_idx = ui._spinner_idx

        # Immediately try again
        ui._advance_frame()

        assert ui._spinner_idx == initial_idx  # Should not have changed

    def test_get_spinner_returns_current_frame(self):
        """_get_spinner should return the current frame."""
        ui = CopexUI()
        ui._spinner_idx = 3
        assert ui._get_spinner() == "⠸"


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


class TestProgressComponents:
    """Tests for progress bar and formatting utilities."""

    def test_format_duration_seconds(self):
        """format_duration should handle seconds."""
        from copex.ui import format_duration

        assert format_duration(30.5) == "30.5s"
        assert format_duration(0.5) == "0.5s"
        assert format_duration(59.9) == "59.9s"

    def test_format_duration_minutes(self):
        """format_duration should handle minutes."""
        from copex.ui import format_duration

        assert format_duration(90) == "1m 30s"
        assert format_duration(3599) == "59m 59s"

    def test_format_duration_hours(self):
        """format_duration should handle hours."""
        from copex.ui import format_duration

        assert format_duration(3600) == "1h 0m"
        assert format_duration(7200) == "2h 0m"
        assert format_duration(3660) == "1h 1m"

    def test_build_progress_bar_empty(self):
        """Progress bar at 0% should be all empty chars."""
        from copex.ui import build_progress_bar

        bar = build_progress_bar(0.0, width=10)
        bar_str = str(bar)
        assert "░" * 10 in bar_str

    def test_build_progress_bar_full(self):
        """Progress bar at 100% should be all filled chars."""
        from copex.ui import build_progress_bar

        bar = build_progress_bar(1.0, width=10)
        bar_str = str(bar)
        assert "━" * 10 in bar_str

    def test_build_progress_bar_half(self):
        """Progress bar at 50% should be half filled."""
        from copex.ui import build_progress_bar

        bar = build_progress_bar(0.5, width=10)
        bar_str = str(bar)
        assert "━" * 5 in bar_str
        assert "░" * 5 in bar_str

    def test_build_progress_bar_clamps_values(self):
        """Progress bar should clamp values to 0.0-1.0."""
        from copex.ui import build_progress_bar

        # Values > 1.0 should be clamped to 1.0
        bar = build_progress_bar(1.5, width=10)
        bar_str = str(bar)
        assert "━" * 10 in bar_str

        # Values < 0.0 should be clamped to 0.0
        bar = build_progress_bar(-0.5, width=10)
        bar_str = str(bar)
        assert "░" * 10 in bar_str


class TestRalphUI:
    """Tests for RalphUI components."""

    def test_ralph_ui_creation(self):
        """RalphUI should be creatable."""
        from copex.ui import RalphUI
        from rich.console import Console

        ui = RalphUI(Console())
        assert ui is not None

    def test_ralph_ui_spinner(self):
        """RalphUI should have spinner functionality."""
        from copex.ui import RalphUI
        from rich.console import Console

        ui = RalphUI(Console())
        spinner1 = ui._get_spinner()
        spinner2 = ui._get_spinner()
        # Spinner should cycle
        assert spinner1 != spinner2 or len(ui._spinners) == 1


class TestPlanUI:
    """Tests for PlanUI components."""

    def test_plan_ui_creation(self):
        """PlanUI should be creatable."""
        from copex.ui import PlanUI
        from rich.console import Console

        ui = PlanUI(Console())
        assert ui is not None

    def test_plan_ui_spinner(self):
        """PlanUI should have spinner functionality."""
        from copex.ui import PlanUI
        from rich.console import Console

        ui = PlanUI(Console())
        spinner1 = ui._get_spinner()
        spinner2 = ui._get_spinner()
        # Spinner should cycle
        assert spinner1 != spinner2 or len(ui._spinners) == 1
