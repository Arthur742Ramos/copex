"""Tests for copex.log_render — JSONL log rendering helpers."""

from __future__ import annotations

import io
import json

import pytest

from rich.console import Console

from copex.log_render import (
    RenderState,
    _classify_event,
    _coerce_float,
    _coerce_int,
    _extract_content,
    _extract_data,
    _extract_delta_content,
    _extract_error_message,
    _extract_event_type,
    _extract_timestamp,
    _extract_tool_args,
    _extract_tool_calls,
    _extract_tool_id,
    _extract_tool_name,
    _extract_tool_output,
    _extract_tool_results,
    _flush_buffers,
    _format_arg_preview,
    _truncate,
    render_jsonl,
)


def _make_console() -> Console:
    return Console(file=io.StringIO(), force_terminal=True, width=120)


def _console_text(console: Console) -> str:
    f = console.file
    assert isinstance(f, io.StringIO)
    return f.getvalue()


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


class TestExtractEventType:
    def test_from_type(self) -> None:
        assert _extract_event_type({"type": "tool.call"}) == "tool.call"

    def test_from_event(self) -> None:
        assert _extract_event_type({"event": "message"}) == "message"

    def test_none(self) -> None:
        assert _extract_event_type({}) is None


class TestExtractData:
    def test_dict_data(self) -> None:
        assert _extract_data({"data": {"k": "v"}}) == {"k": "v"}

    def test_non_dict(self) -> None:
        assert _extract_data({"data": "string"}) is None

    def test_payload_key(self) -> None:
        assert _extract_data({"payload": {"x": 1}}) == {"x": 1}


class TestExtractContent:
    def test_content_key(self) -> None:
        assert _extract_content({"content": "hello"}) == "hello"

    def test_text_key(self) -> None:
        assert _extract_content({"text": "world"}) == "world"

    def test_empty_fallback(self) -> None:
        assert _extract_content({}) == ""

    def test_non_string_content(self) -> None:
        assert _extract_content({"content": 42}) == "42"


class TestExtractDeltaContent:
    def test_delta_key(self) -> None:
        assert _extract_delta_content({"delta": "chunk"}) == "chunk"

    def test_delta_content_key(self) -> None:
        assert _extract_delta_content({"delta_content": "dc"}) == "dc"

    def test_empty(self) -> None:
        assert _extract_delta_content({}) == ""


class TestExtractTimestamp:
    def test_string_timestamp(self) -> None:
        assert _extract_timestamp({"timestamp": "2025-01-01"}, None) == "2025-01-01"

    def test_numeric_timestamp(self) -> None:
        result = _extract_timestamp({"timestamp": 1704067200}, None)
        assert result is not None  # Converted to ISO

    def test_from_data(self) -> None:
        result = _extract_timestamp({}, {"time": "12:00"})
        assert result == "12:00"

    def test_none(self) -> None:
        assert _extract_timestamp({}, None) is None


class TestExtractToolHelpers:
    def test_tool_name_from_name(self) -> None:
        assert _extract_tool_name({"name": "bash"}) == "bash"

    def test_tool_name_from_state(self) -> None:
        state = RenderState()
        state.tool_names["id1"] = "write_file"
        assert _extract_tool_name({"id": "id1"}, state) == "write_file"

    def test_tool_name_unknown(self) -> None:
        assert _extract_tool_name({}) == "unknown"

    def test_tool_id(self) -> None:
        assert _extract_tool_id({"tool_call_id": "tc1"}) == "tc1"

    def test_tool_id_none(self) -> None:
        assert _extract_tool_id({}) is None

    def test_tool_args_dict(self) -> None:
        assert _extract_tool_args({"arguments": {"a": 1}}) == {"a": 1}

    def test_tool_args_none(self) -> None:
        assert _extract_tool_args({}) is None

    def test_tool_calls_list(self) -> None:
        calls = _extract_tool_calls({"tool_calls": [{"name": "a"}, {"name": "b"}]})
        assert len(calls) == 2

    def test_tool_calls_empty(self) -> None:
        assert _extract_tool_calls({}) == []

    def test_tool_results_list(self) -> None:
        results = _extract_tool_results({"tool_results": [{"output": "ok"}]})
        assert len(results) == 1

    def test_tool_results_empty(self) -> None:
        assert _extract_tool_results({}) == []


class TestExtractToolOutput:
    def test_string_output(self) -> None:
        assert _extract_tool_output({"output": "hello"}) == "hello"

    def test_dict_with_content(self) -> None:
        result = _extract_tool_output({"output": {"content": "inner"}})
        assert result == "inner"

    def test_dict_fallback(self) -> None:
        result = _extract_tool_output({"output": {"key": "val"}})
        assert result is not None  # JSON-serialized

    def test_none(self) -> None:
        assert _extract_tool_output({}) is None


class TestExtractErrorMessage:
    def test_message_key(self) -> None:
        assert _extract_error_message({"message": "oops"}) == "oops"

    def test_error_key(self) -> None:
        assert _extract_error_message({"error": "fail"}) == "fail"

    def test_fallback_json(self) -> None:
        result = _extract_error_message({"unknown": "data"})
        assert "unknown" in result  # JSON-encoded


# ---------------------------------------------------------------------------
# Coercion
# ---------------------------------------------------------------------------


class TestCoercion:
    def test_coerce_int(self) -> None:
        assert _coerce_int(42) == 42
        assert _coerce_int("10") == 10
        assert _coerce_int(None) is None
        assert _coerce_int("abc") is None

    def test_coerce_float(self) -> None:
        assert _coerce_float(3.14) == pytest.approx(3.14)
        assert _coerce_float("2.5") == pytest.approx(2.5)
        assert _coerce_float(None) is None
        assert _coerce_float("xyz") is None


# ---------------------------------------------------------------------------
# Classify event
# ---------------------------------------------------------------------------


class TestClassifyEvent:
    @pytest.mark.parametrize(
        ("event_type", "expected"),
        [
            ("progress", "progress"),
            ("usage_report", "usage"),
            ("tool.execution_partial_result", "tool_partial"),
            ("tool.execution_complete", "tool_result"),
            ("tool.execution_start", "tool_call"),
            ("assistant.reasoning_delta", "assistant_reasoning_delta"),
            ("assistant.reasoning", "assistant_reasoning"),
            ("assistant.message_delta", "assistant_message_delta"),
            ("user.message", "user"),
            ("assistant.message", "assistant_message"),
            ("turn_end", "turn_end"),
            ("error", "error"),
            ("system", "system"),
        ],
    )
    def test_classify(self, event_type: str, expected: str) -> None:
        assert _classify_event(event_type, {}, None) == expected

    def test_classify_role_user(self) -> None:
        assert _classify_event(None, {"role": "user", "content": "hi"}, None) == "user"

    def test_classify_role_assistant(self) -> None:
        assert (
            _classify_event(None, {"role": "assistant", "content": "hi"}, None)
            == "assistant_message"
        )

    def test_classify_unknown(self) -> None:
        assert _classify_event(None, {}, None) == "unknown"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


class TestFormatArgPreview:
    def test_path_arg(self) -> None:
        assert "path=" in _format_arg_preview({"path": "/foo/bar"})

    def test_command_arg(self) -> None:
        assert "cmd=" in _format_arg_preview({"command": "ls -la"})

    def test_empty(self) -> None:
        assert _format_arg_preview(None) == ""
        assert _format_arg_preview({}) == ""


class TestTruncate:
    def test_short(self) -> None:
        assert _truncate("abc", 10) == "abc"

    def test_long(self) -> None:
        result = _truncate("a" * 100, 20)
        assert len(result) == 20
        assert result.endswith("...")


# ---------------------------------------------------------------------------
# Flush buffers
# ---------------------------------------------------------------------------


class TestFlushBuffers:
    def test_flushes_both_buffers(self) -> None:
        state = RenderState(message_buffer="msg", reasoning_buffer="think")
        console = _make_console()
        _flush_buffers(state, console, None)
        assert state.message_buffer == ""
        assert state.reasoning_buffer == ""
        text = _console_text(console)
        assert "Reasoning" in text
        assert "Assistant" in text

    def test_no_output_when_empty(self) -> None:
        state = RenderState()
        console = _make_console()
        _flush_buffers(state, console, None)
        assert _console_text(console) == ""


# ---------------------------------------------------------------------------
# Full render_jsonl
# ---------------------------------------------------------------------------


class TestRenderJsonl:
    def test_renders_assistant_message(self) -> None:
        events = [
            {"type": "assistant.message", "data": {"content": "Hello world"}},
        ]
        stream = io.StringIO("\n".join(json.dumps(e) for e in events))
        console = _make_console()
        render_jsonl(stream, console)
        text = _console_text(console)
        assert "Assistant" in text

    def test_renders_error_event(self) -> None:
        events = [{"type": "error", "data": {"message": "something broke"}}]
        stream = io.StringIO(json.dumps(events[0]))
        console = _make_console()
        render_jsonl(stream, console)
        text = _console_text(console)
        assert "Error" in text

    def test_invalid_json_shows_error(self) -> None:
        stream = io.StringIO("not valid json\n")
        console = _make_console()
        render_jsonl(stream, console)
        text = _console_text(console)
        assert "Invalid JSON" in text

    def test_empty_stream(self) -> None:
        stream = io.StringIO("")
        console = _make_console()
        render_jsonl(stream, console)
        assert _console_text(console) == ""

    def test_delta_accumulation(self) -> None:
        events = [
            {"type": "assistant.message_delta", "data": {"delta": "Hel"}},
            {"type": "assistant.message_delta", "data": {"delta": "lo"}},
            {"type": "assistant.message", "data": {"content": ""}},
        ]
        stream = io.StringIO("\n".join(json.dumps(e) for e in events))
        console = _make_console()
        render_jsonl(stream, console)
        text = _console_text(console)
        assert "Assistant" in text

    def test_tool_call_and_result(self) -> None:
        events = [
            {
                "type": "tool.execution_start",
                "data": {"name": "bash", "arguments": {"command": "ls"}},
            },
            {
                "type": "tool.execution_complete",
                "data": {"name": "bash", "output": "file.txt", "success": True},
            },
        ]
        stream = io.StringIO("\n".join(json.dumps(e) for e in events))
        console = _make_console()
        render_jsonl(stream, console)
        text = _console_text(console)
        assert "bash" in text

    def test_non_dict_event(self) -> None:
        stream = io.StringIO('"just a string"\n')
        console = _make_console()
        render_jsonl(stream, console)
        text = _console_text(console)
        assert "just a string" in text
