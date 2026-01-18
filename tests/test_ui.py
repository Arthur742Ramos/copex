"""Tests for UI components."""

from __future__ import annotations

import time

import pytest

from copex.ui import ActivityType, CopexUI, Theme, ToolCallInfo, UIState


class TestUIState:
    """Tests for UIState dataclass."""

    def test_total_tokens(self):
        """total_tokens should sum input and output tokens."""
        state = UIState()
        state.input_tokens = 100
        state.output_tokens = 200
        assert state.total_tokens == 300

    def test_estimated_cost(self):
        """estimated_cost should calculate based on tokens."""
        state = UIState()
        state.input_tokens = 50000
        state.output_tokens = 50000
        # 100K tokens * $0.00001 = $1.00
        assert state.estimated_cost == pytest.approx(1.0, rel=0.01)

    def test_elapsed_uses_completed_at_when_set(self):
        """elapsed should use completed_at if available."""
        state = UIState()
        state.start_time = 100.0
        state.completed_at = 105.0
        assert state.elapsed == 5.0

    def test_elapsed_uses_current_time_when_not_completed(self):
        """elapsed should use current time if not completed."""
        state = UIState()
        state.start_time = time.time() - 2.0
        state.completed_at = None
        assert 1.9 < state.elapsed < 2.5


class TestCopexUISpinner:
    """Tests for spinner animation."""

    def test_spinner_varies_by_activity(self):
        """Spinner should change based on activity type."""
        ui = CopexUI()
        
        ui.state.activity = ActivityType.THINKING
        thinking_spinner = ui._get_spinner()
        
        ui.state.activity = ActivityType.TOOL_CALL
        tool_spinner = ui._get_spinner()
        
        # All activities use the same spinner set now
        assert thinking_spinner in ui._spinners
        assert tool_spinner in ui._spinners

    def test_spinner_default_for_waiting(self):
        """Default spinner for waiting activity."""
        ui = CopexUI()
        ui.state.activity = ActivityType.WAITING
        spinner = ui._get_spinner()
        assert spinner in ui._spinners

    def test_spinner_advances_after_interval(self):
        """Spinner should advance after interval."""
        ui = CopexUI()
        initial_idx = ui._spinner_idx

        # Force time to pass
        ui._last_frame_at = time.time() - 0.1  # 100ms ago
        ui._advance_frame()

        assert ui._spinner_idx == (initial_idx + 1) % len(ui._spinners)

    def test_spinner_does_not_advance_too_quickly(self):
        """Spinner should not advance if less than interval passed."""
        ui = CopexUI()
        ui._advance_frame()  # First advance
        initial_idx = ui._spinner_idx

        # Immediately try again
        ui._advance_frame()

        assert ui._spinner_idx == initial_idx  # Should not have changed


class TestCopexUITokens:
    """Tests for token tracking."""

    def test_set_tokens(self):
        """set_tokens should update token counts."""
        ui = CopexUI()
        ui.set_tokens(input_tokens=100, output_tokens=200)
        
        assert ui.state.input_tokens == 100
        assert ui.state.output_tokens == 200

    def test_add_tokens(self):
        """add_tokens should accumulate token counts."""
        ui = CopexUI()
        ui.set_tokens(input_tokens=100, output_tokens=100)
        ui.add_tokens(input_tokens=50, output_tokens=150)
        
        assert ui.state.input_tokens == 150
        assert ui.state.output_tokens == 250


class TestCopexUIBell:
    """Tests for bell toggle."""

    def test_toggle_bell_default_off(self):
        """Bell should be off by default."""
        ui = CopexUI()
        assert ui._bell_on_complete is False

    def test_toggle_bell_returns_new_state(self):
        """toggle_bell should return new state."""
        ui = CopexUI()
        assert ui.toggle_bell() is True
        assert ui._bell_on_complete is True
        assert ui.toggle_bell() is False
        assert ui._bell_on_complete is False


class TestCopexUIClipboard:
    """Tests for clipboard/export functionality."""

    def test_get_last_response_returns_message(self):
        """get_last_response should return current message."""
        ui = CopexUI()
        ui.state.message = "Hello, world!"
        assert ui.get_last_response() == "Hello, world!"

    def test_get_last_response_returns_none_when_empty(self):
        """get_last_response should return None when no message."""
        ui = CopexUI()
        ui.state.message = ""
        assert ui.get_last_response() is None

    def test_export_conversation_includes_history(self):
        """export_conversation should include conversation history."""
        ui = CopexUI()
        ui.state.model = "gpt-5.2-codex"
        ui.add_user_message("Hello")
        ui.state.message = "Hi there!"
        ui.finalize_assistant_response()
        
        markdown = ui.export_conversation()
        
        assert "# Copex Conversation" in markdown
        assert "gpt-5.2-codex" in markdown
        assert "Hello" in markdown
        assert "Hi there!" in markdown
        assert "👤 User" in markdown
        assert "🤖 Assistant" in markdown

    def test_export_conversation_includes_tool_calls(self):
        """export_conversation should include tool call info."""
        ui = CopexUI()
        ui.add_user_message("Run a command")
        ui.add_tool_call(ToolCallInfo(name="bash", status="success"))
        ui.state.message = "Done!"
        ui.finalize_assistant_response()
        
        markdown = ui.export_conversation()
        
        assert "Tool Calls" in markdown
        assert "bash" in markdown


class TestCopexUIFinalContent:
    """Tests for final content handling."""

    def test_set_final_content_sets_completed_at(self):
        """set_final_content should set completed_at timestamp."""
        ui = CopexUI()
        ui.state.completed_at = None
        
        ui.set_final_content("Done!")
        
        assert ui.state.completed_at is not None
        assert ui.state.activity == ActivityType.DONE

    def test_set_final_content_with_reasoning(self):
        """set_final_content should handle reasoning."""
        ui = CopexUI()
        ui.set_final_content("Answer", reasoning="I thought about it")
        
        assert ui.state.message == "Answer"
        assert ui.state.reasoning == "I thought about it"


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
        ui.state.input_tokens = 100
        ui.state.output_tokens = 200

        ui.reset(model="gpt-4")

        assert ui.state.message == ""
        assert ui.state.reasoning == ""
        assert ui.state.activity == ActivityType.WAITING
        assert ui.state.model == "gpt-4"
        assert ui.state.input_tokens == 0
        assert ui.state.output_tokens == 0

