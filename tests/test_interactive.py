"""Tests for the new interactive module."""

import pytest
import time

from copex.interactive import (
    Colors,
    Icons,
    Spinners,
    ToolCall,
    StreamState,
    StreamRenderer,
    SlashCompleter,
    _build_stats_line,
)
from rich.console import Console


class TestColors:
    """Test the Colors class."""

    def test_has_primary_color(self):
        assert Colors.PRIMARY == "#7aa2f7"

    def test_has_success_color(self):
        assert Colors.SUCCESS == "#9ece6a"

    def test_has_error_color(self):
        assert Colors.ERROR == "#f7768e"


class TestIcons:
    """Test the Icons class."""

    def test_has_done_icon(self):
        assert Icons.DONE == "✓"

    def test_has_error_icon(self):
        assert Icons.ERROR == "✗"

    def test_has_clock_icon(self):
        assert Icons.CLOCK == "⏱"


class TestSpinners:
    """Test the Spinners class."""

    def test_dots_has_frames(self):
        assert len(Spinners.DOTS) == 10


class TestToolCall:
    """Test the ToolCall dataclass."""

    def test_creation(self):
        tool = ToolCall(tool_id="1", name="read_file")
        assert tool.tool_id == "1"
        assert tool.name == "read_file"
        assert tool.status == "running"

    def test_icon_for_read(self):
        tool = ToolCall(tool_id="1", name="read_file")
        assert tool.icon == "📖"

    def test_icon_for_write(self):
        tool = ToolCall(tool_id="1", name="write_file")
        assert tool.icon == "📝"

    def test_icon_for_shell(self):
        tool = ToolCall(tool_id="1", name="bash_shell")
        assert tool.icon == "💻"

    def test_icon_for_search(self):
        tool = ToolCall(tool_id="1", name="grep_search")
        assert tool.icon == "🔍"

    def test_icon_for_web(self):
        tool = ToolCall(tool_id="1", name="web_fetch")
        assert tool.icon == "🌐"

    def test_icon_default(self):
        tool = ToolCall(tool_id="1", name="unknown_tool")
        assert tool.icon == "⚡"

    def test_elapsed_when_running(self):
        tool = ToolCall(tool_id="1", name="test")
        time.sleep(0.1)
        assert tool.elapsed >= 0.1

    def test_elapsed_when_finished(self):
        tool = ToolCall(tool_id="1", name="test", duration=5.0)
        assert tool.elapsed == 5.0


class TestStreamState:
    """Test the StreamState dataclass."""

    def test_creation(self):
        state = StreamState()
        assert state.phase == "idle"
        assert state.message == ""
        assert state.reasoning == ""
        assert len(state.tool_calls) == 0

    def test_elapsed_str_seconds(self):
        state = StreamState()
        state.start_time = time.time() - 30
        assert "30" in state.elapsed_str or "29" in state.elapsed_str

    def test_elapsed_str_minutes(self):
        state = StreamState()
        state.start_time = time.time() - 90
        assert "m" in state.elapsed_str

    def test_spinner_frames(self):
        state = StreamState()
        assert state.spinner in Spinners.DOTS

    def test_advance_frame(self):
        state = StreamState()
        state._last_frame_time = 0  # Force advance
        initial_frame = state._frame
        state.advance_frame()
        assert state._frame == (initial_frame + 1) % len(Spinners.DOTS)


class TestStreamRenderer:
    """Test the StreamRenderer class."""

    def test_creation(self):
        console = Console(force_terminal=True)
        state = StreamState()
        renderer = StreamRenderer(console, state)
        assert renderer.state is state
        assert renderer.compact is False

    def test_build_returns_group(self):
        console = Console(force_terminal=True)
        state = StreamState()
        state.phase = "thinking"
        renderer = StreamRenderer(console, state)
        result = renderer.build()
        assert result is not None

    def test_format_arg_preview(self):
        console = Console(force_terminal=True)
        state = StreamState()
        renderer = StreamRenderer(console, state)
        
        args = {"path": "/some/file.py", "other": "ignored"}
        preview = renderer._format_arg_preview(args)
        assert "path=" in preview
        assert "/some/file.py" in preview


class TestSlashCompleter:
    """Test the SlashCompleter class."""

    def test_has_commands(self):
        completer = SlashCompleter()
        assert "/model" in completer.COMMANDS
        assert "/models" in completer.COMMANDS
        assert "/help" in completer.COMMANDS
        assert "/new" in completer.COMMANDS
        assert "exit" in completer.COMMANDS


class TestBuildStatsLine:
    """Test the _build_stats_line function."""

    def test_basic_stats(self):
        state = StreamState()
        state.start_time = time.time() - 10
        result = _build_stats_line(state)
        assert result is not None
        # Should contain elapsed time
        plain = result.plain
        assert "10" in plain or "9" in plain

    def test_with_tokens(self):
        state = StreamState()
        state.input_tokens = 1000
        state.output_tokens = 500
        result = _build_stats_line(state)
        plain = result.plain
        assert "1,000" in plain
        assert "500" in plain

    def test_with_tool_calls(self):
        state = StreamState()
        state.tool_calls = [
            ToolCall(tool_id="1", name="test1", status="success"),
            ToolCall(tool_id="2", name="test2", status="success"),
        ]
        result = _build_stats_line(state)
        plain = result.plain
        assert "Tools: 2" in plain

    def test_with_retries(self):
        state = StreamState()
        state.retries = 3
        result = _build_stats_line(state)
        plain = result.plain
        assert "Retries: 3" in plain


class TestFormatArgPreview:
    """Test the _format_arg_preview helper."""

    def test_with_path(self):
        from copex.interactive import StreamState, StreamRenderer
        from rich.console import Console
        console = Console(force_terminal=True)
        state = StreamState()
        renderer = StreamRenderer(console, state)
        
        result = renderer._format_arg_preview({"path": "/test/file.py"})
        assert "path=/test/file.py" in result

    def test_with_long_value(self):
        from copex.interactive import StreamState, StreamRenderer
        from rich.console import Console
        console = Console(force_terminal=True)
        state = StreamState()
        renderer = StreamRenderer(console, state)
        
        long_path = "/very/long/path/that/exceeds/the/maximum/allowed/length/for/preview"
        result = renderer._format_arg_preview({"path": long_path})
        assert "..." in result
        assert len(result) <= 50

    def test_empty_args(self):
        from copex.interactive import StreamState, StreamRenderer
        from rich.console import Console
        console = Console(force_terminal=True)
        state = StreamState()
        renderer = StreamRenderer(console, state)
        
        result = renderer._format_arg_preview({})
        assert result == ""
