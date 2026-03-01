"""Tests for agent functionality.

Tests the AgentSession loop, AgentTurn/AgentResult dataclasses,
JSON Lines output, CLI integration, and edge cases.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from copex.agent import AgentResult, AgentSession, AgentTurn, DEFAULT_MAX_TURNS
from copex.config import CopexConfig
from copex.streaming import Response, StreamChunk


# ---------------------------------------------------------------------------
# Helpers — mock client that simulates multi-turn agent interactions
# ---------------------------------------------------------------------------


def _chunks_for_tool_calls(
    tool_calls: list[dict[str, Any]],
) -> list[StreamChunk]:
    """Build StreamChunk objects that simulate tool call/result events."""
    chunks: list[StreamChunk] = []
    for i, tc in enumerate(tool_calls):
        tid = f"tool-{i}"
        chunks.append(StreamChunk(
            type="tool_call",
            tool_id=tid,
            tool_name=tc.get("name", "unknown"),
            tool_args=tc.get("arguments", {}),
        ))
        chunks.append(StreamChunk(
            type="tool_result",
            tool_id=tid,
            tool_name=tc.get("name", "unknown"),
            tool_result="ok",
            tool_success=True,
            tool_duration=0.1,
        ))
    return chunks


class FakeClient:
    """Mock client whose send() returns pre-scripted responses.

    Each entry in *responses* is a tuple of (content, tool_call_dicts).
    Tool calls are delivered via on_chunk callbacks to match the real flow.
    """

    def __init__(
        self,
        responses: list[tuple[str, list[dict[str, Any]]]] | None = None,
    ):
        self._responses = list(responses or [])
        self.prompts_seen: list[str] = []
        self.config = CopexConfig()

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def send(
        self,
        prompt: str,
        *,
        tools: list[Any] | None = None,
        on_chunk: Any = None,
        metrics: Any = None,
    ) -> Response:
        self.prompts_seen.append(prompt)
        if self._responses:
            content, tc_dicts = self._responses.pop(0)
        else:
            content, tc_dicts = ("done", [])

        if on_chunk and tc_dicts:
            for chunk in _chunks_for_tool_calls(tc_dicts):
                on_chunk(chunk)

        return Response(content=content)


def _simple(content: str) -> tuple[str, list[dict[str, Any]]]:
    """Response with no tool calls."""
    return (content, [])


def _with_tools(
    content: str, tools: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    """Response with tool calls."""
    return (content, tools)


def run(coro):
    return asyncio.run(coro)


# ===========================================================================
# 1. AgentTurn / AgentResult dataclass tests
# ===========================================================================


class TestAgentDataclasses:

    def test_agent_turn_fields(self):
        turn = AgentTurn(turn=1, content="Hello", tool_calls=[], stop_reason="end_turn")
        assert turn.turn == 1
        assert turn.content == "Hello"
        assert turn.tool_calls == []
        assert turn.stop_reason == "end_turn"

    def test_agent_turn_with_tool_calls(self):
        calls = [{"name": "read_file", "arguments": {"path": "a.txt"}}]
        turn = AgentTurn(turn=2, content="", tool_calls=calls)
        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0]["name"] == "read_file"

    def test_agent_result_fields(self):
        t1 = AgentTurn(turn=1, content="step 1")
        t2 = AgentTurn(turn=2, content="final", stop_reason="end_turn")
        result = AgentResult(turns=[t1, t2], final_content="final", total_turns=2)
        assert result.total_turns == 2
        assert result.final_content == "final"
        assert len(result.turns) == 2

    def test_agent_result_empty(self):
        result = AgentResult()
        assert result.turns == []
        assert result.final_content == ""
        assert result.total_turns == 0

    def test_agent_turn_to_dict(self):
        turn = AgentTurn(
            turn=1,
            content="Hello",
            tool_calls=[{"name": "bash", "arguments": {"cmd": "ls"}}],
            stop_reason="end_turn",
        )
        d = turn.to_dict()
        assert d["turn"] == 1
        assert d["content"] == "Hello"
        assert d["stop_reason"] == "end_turn"
        assert d["tool_calls"][0]["name"] == "bash"
        assert json.dumps(d)  # JSON-serializable

    def test_agent_turn_to_json(self):
        turn = AgentTurn(turn=1, content="hi")
        parsed = json.loads(turn.to_json())
        assert parsed["turn"] == 1
        assert parsed["content"] == "hi"

    def test_agent_result_to_json(self):
        t = AgentTurn(turn=1, content="done", stop_reason="end_turn")
        result = AgentResult(turns=[t], final_content="done", stop_reason="end_turn")
        d = result.to_dict()
        assert "turns" in d
        assert d["final_content"] == "done"
        assert json.loads(result.to_json())


# ===========================================================================
# 2. AgentSession basics
# ===========================================================================


class TestAgentSessionBasics:

    def test_create_agent_session(self):
        client = FakeClient([_simple("Hello world")])
        session = AgentSession(client)
        assert session is not None

    def test_simple_prompt_returns_result(self):
        client = FakeClient([_simple("Hello world")])
        session = AgentSession(client)
        result = run(session.run("Say hello"))
        assert result.final_content == "Hello world"
        assert len(result.turns) >= 1
        assert result.stop_reason == "end_turn"

    def test_prompt_is_passed_to_client(self):
        client = FakeClient([_simple("ok")])
        session = AgentSession(client)
        run(session.run("specific test prompt"))
        assert client.prompts_seen[0] == "specific test prompt"

    def test_context_manager(self):
        client = FakeClient([_simple("ok")])

        async def _test():
            async with AgentSession(client) as session:
                return await session.run("prompt")

        result = run(_test())
        assert result.final_content == "ok"

    def test_total_turns_set(self):
        client = FakeClient([_simple("ok")])
        session = AgentSession(client)
        result = run(session.run("hi"))
        assert result.total_turns == 1


# ===========================================================================
# 3. Turn limiting
# ===========================================================================


class TestTurnLimiting:

    def test_stops_at_max_turns(self):
        tool_resp = _with_tools("thinking", [{"name": "bash", "arguments": {"cmd": "echo"}}])
        client = FakeClient([tool_resp] * 20)
        session = AgentSession(client, max_turns=3)
        result = run(session.run("keep going"))
        assert len(result.turns) <= 3

    def test_max_turns_stop_reason(self):
        tool_resp = _with_tools("going", [{"name": "bash", "arguments": {"cmd": "echo"}}])
        client = FakeClient([tool_resp] * 20)
        session = AgentSession(client, max_turns=2)
        result = run(session.run("go"))
        assert result.stop_reason == "max_turns"

    def test_last_turn_stop_reason_is_max_turns(self):
        tool_resp = _with_tools("going", [{"name": "bash", "arguments": {"cmd": "echo"}}])
        client = FakeClient([tool_resp] * 20)
        session = AgentSession(client, max_turns=2)
        result = run(session.run("go"))
        assert result.turns[-1].stop_reason == "max_turns"

    def test_default_max_turns_is_reasonable(self):
        client = FakeClient([_simple("done")])
        session = AgentSession(client)
        assert session.max_turns == DEFAULT_MAX_TURNS
        assert DEFAULT_MAX_TURNS >= 1
        assert DEFAULT_MAX_TURNS <= 200


# ===========================================================================
# 4. Graceful completion (end_turn)
# ===========================================================================


class TestGracefulCompletion:

    def test_stops_on_no_tool_calls(self):
        responses = [
            _with_tools("running tool", [{"name": "bash", "arguments": {"cmd": "ls"}}]),
            _simple("All done!"),
        ]
        client = FakeClient(responses)
        session = AgentSession(client, max_turns=10)
        result = run(session.run("Do something"))
        assert result.final_content == "All done!"
        assert len(result.turns) == 2

    def test_end_turn_stop_reason(self):
        client = FakeClient([_simple("Final answer")])
        session = AgentSession(client)
        result = run(session.run("Answer me"))
        assert result.stop_reason == "end_turn"
        assert result.turns[-1].stop_reason == "end_turn"

    def test_single_turn_no_tools(self):
        client = FakeClient([_simple("Instant answer")])
        session = AgentSession(client, max_turns=10)
        result = run(session.run("question"))
        assert len(result.turns) == 1
        assert result.final_content == "Instant answer"


# ===========================================================================
# 5. Tool call handling
# ===========================================================================


class TestToolCallHandling:

    def test_tool_calls_recorded_in_turn(self):
        responses = [
            _with_tools("Checking", [{"name": "read_file", "arguments": {"path": "foo.py"}}]),
            _simple("Done"),
        ]
        client = FakeClient(responses)
        session = AgentSession(client)
        result = run(session.run("Read foo.py"))
        first_turn = result.turns[0]
        assert len(first_turn.tool_calls) > 0
        assert first_turn.tool_calls[0]["name"] == "read_file"

    def test_multi_tool_calls_in_one_turn(self):
        responses = [
            _with_tools("Reading", [
                {"name": "read_file", "arguments": {"path": "a.py"}},
                {"name": "read_file", "arguments": {"path": "b.py"}},
            ]),
            _simple("Analyzed both"),
        ]
        client = FakeClient(responses)
        session = AgentSession(client)
        result = run(session.run("Read both"))
        assert len(result.turns[0].tool_calls) == 2

    def test_tool_results_fed_back(self):
        responses = [
            _with_tools("Calling", [{"name": "bash", "arguments": {"cmd": "echo"}}]),
            _simple("Got it"),
        ]
        client = FakeClient(responses)
        session = AgentSession(client)
        run(session.run("Run command"))
        assert len(client.prompts_seen) == 2

    def test_continue_prompt_used_for_subsequent_turns(self):
        responses = [
            _with_tools("step 1", [{"name": "bash", "arguments": {"cmd": "ls"}}]),
            _simple("step 2"),
        ]
        client = FakeClient(responses)
        session = AgentSession(client, continue_prompt="Keep going")
        run(session.run("initial"))
        assert client.prompts_seen[0] == "initial"
        assert client.prompts_seen[1] == "Keep going"

    def test_tool_call_captures_result_metadata(self):
        responses = [
            _with_tools("Running", [{"name": "bash", "arguments": {"cmd": "echo"}}]),
            _simple("Done"),
        ]
        client = FakeClient(responses)
        session = AgentSession(client)
        result = run(session.run("go"))
        tc = result.turns[0].tool_calls[0]
        assert tc["result"] == "ok"
        assert tc["success"] is True
        assert tc["duration"] == 0.1


# ===========================================================================
# 6. Error handling
# ===========================================================================


class TestErrorHandling:

    def test_sdk_error_produces_error_stop_reason(self):
        client = FakeClient()
        client.send = AsyncMock(side_effect=RuntimeError("SDK exploded"))
        session = AgentSession(client)
        result = run(session.run("trigger error"))
        assert result.stop_reason == "error"
        assert result.turns[-1].stop_reason == "error"

    def test_error_message_captured(self):
        client = FakeClient()
        client.send = AsyncMock(side_effect=RuntimeError("boom"))
        session = AgentSession(client)
        result = run(session.run("fail"))
        assert result.error is not None
        assert "boom" in result.error
        assert result.turns[-1].error is not None

    def test_timeout_error_handled(self):
        client = FakeClient()
        client.send = AsyncMock(side_effect=asyncio.TimeoutError())
        session = AgentSession(client)
        result = run(session.run("slow prompt"))
        assert result.stop_reason == "error"

    def test_error_preserves_previous_turns(self):
        call_count = 0

        class FailOnSecondClient(FakeClient):
            async def send(self, prompt, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    on_chunk = kwargs.get("on_chunk")
                    if on_chunk:
                        for c in _chunks_for_tool_calls([{"name": "bash", "arguments": {}}]):
                            on_chunk(c)
                    return Response(content="First turn ok")
                raise RuntimeError("Network failed")

        client = FailOnSecondClient()
        session = AgentSession(client, max_turns=10)
        result = run(session.run("multi-turn"))
        assert len(result.turns) >= 2
        assert result.turns[0].content == "First turn ok"
        assert result.turns[-1].stop_reason == "error"
        assert result.stop_reason == "error"

    def test_error_on_first_turn(self):
        client = FakeClient()
        client.send = AsyncMock(side_effect=ConnectionError("refused"))
        session = AgentSession(client)
        result = run(session.run("fail fast"))
        assert len(result.turns) == 1
        assert result.turns[0].stop_reason == "error"
        assert result.turns[0].turn == 1


# ===========================================================================
# 7. JSON output format
# ===========================================================================


class TestJsonOutputFormat:

    def test_agent_turn_json_roundtrip(self):
        tc = {"name": "bash", "arguments": {"cmd": "echo hi"}, "result": "hi", "success": True}
        turn = AgentTurn(turn=1, content="Hello", tool_calls=[tc], stop_reason="end_turn")
        parsed = json.loads(turn.to_json())
        assert parsed["turn"] == 1
        assert parsed["content"] == "Hello"
        assert parsed["stop_reason"] == "end_turn"
        assert parsed["tool_calls"][0]["name"] == "bash"

    def test_agent_result_json_lines(self):
        turns = [
            AgentTurn(turn=1, content="step 1"),
            AgentTurn(turn=2, content="step 2", stop_reason="end_turn"),
        ]
        result = AgentResult(turns=turns, final_content="step 2")
        lines = [t.to_json() for t in result.turns]
        assert len(lines) == 2
        for line in lines:
            parsed = json.loads(line)
            assert "turn" in parsed
            assert "content" in parsed
            assert "tool_calls" in parsed
            assert "stop_reason" in parsed

    def test_json_contains_duration_ms(self):
        turn = AgentTurn(turn=1, content="x", duration_ms=123.456)
        d = turn.to_dict()
        assert d["duration_ms"] == 123.5

    def test_json_contains_error_field(self):
        turn = AgentTurn(turn=1, content="", stop_reason="error", error="bad thing")
        d = turn.to_dict()
        assert d["error"] == "bad thing"


# ===========================================================================
# 8. Stop reasons
# ===========================================================================


class TestStopReasons:

    def test_end_turn_reason(self):
        client = FakeClient([_simple("Done")])
        session = AgentSession(client)
        result = run(session.run("simple"))
        assert result.stop_reason == "end_turn"
        assert result.turns[-1].stop_reason == "end_turn"

    def test_max_turns_reason(self):
        tool_resp = _with_tools("w", [{"name": "bash", "arguments": {"cmd": "echo"}}])
        client = FakeClient([tool_resp] * 10)
        session = AgentSession(client, max_turns=1)
        result = run(session.run("go"))
        assert result.stop_reason == "max_turns"

    def test_error_reason(self):
        client = FakeClient()
        client.send = AsyncMock(side_effect=Exception("kaboom"))
        session = AgentSession(client)
        result = run(session.run("fail"))
        assert result.stop_reason == "error"

    def test_result_stop_reason_matches_last_turn_for_end_turn(self):
        client = FakeClient([_simple("Done")])
        session = AgentSession(client)
        result = run(session.run("test"))
        assert result.stop_reason == result.turns[-1].stop_reason

    def test_intermediate_turn_has_none_stop_reason(self):
        """Intermediate turns with tool calls should have stop_reason=None."""
        responses = [
            _with_tools("step 1", [{"name": "bash", "arguments": {"cmd": "ls"}}]),
            _simple("step 2 final"),
        ]
        client = FakeClient(responses)
        session = AgentSession(client, max_turns=10)
        result = run(session.run("multi"))
        if len(result.turns) > 1:
            assert result.turns[0].stop_reason is None


# ===========================================================================
# 9. Model parameter
# ===========================================================================


class TestModelParameter:

    def test_model_passed_to_session(self):
        client = FakeClient([_simple("ok")])
        session = AgentSession(client, model="gpt-5.2-codex")
        assert session.model == "gpt-5.2-codex"

    def test_default_model_is_none(self):
        client = FakeClient([_simple("ok")])
        session = AgentSession(client)
        assert session.model is None

    def test_agent_cli_model_option(self):
        from copex.cli import app
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["agent", "--help"])
        assert "--model" in result.output


# ===========================================================================
# 10. Edge cases
# ===========================================================================


class TestEdgeCases:

    def test_empty_prompt(self):
        client = FakeClient([_simple("handled empty")])
        session = AgentSession(client)
        result = run(session.run(""))
        assert len(result.turns) >= 1

    def test_zero_max_turns(self):
        client = FakeClient([_simple("never reached")])
        session = AgentSession(client, max_turns=0)
        result = run(session.run("prompt"))
        assert len(result.turns) == 0
        assert result.stop_reason == "max_turns"

    def test_single_max_turn_with_tools(self):
        client = FakeClient([
            _with_tools("one turn", [{"name": "bash", "arguments": {"cmd": "ls"}}]),
        ])
        session = AgentSession(client, max_turns=1)
        result = run(session.run("go"))
        assert len(result.turns) == 1
        assert result.stop_reason == "max_turns"

    def test_single_max_turn_no_tools(self):
        client = FakeClient([_simple("answer")])
        session = AgentSession(client, max_turns=1)
        result = run(session.run("go"))
        assert len(result.turns) == 1
        assert result.stop_reason == "end_turn"

    def test_large_max_turns_finishes_normally(self):
        client = FakeClient([_simple("quick answer")])
        session = AgentSession(client, max_turns=10000)
        result = run(session.run("answer this"))
        assert len(result.turns) == 1
        assert result.final_content == "quick answer"

    def test_empty_content_response(self):
        client = FakeClient([_simple("")])
        session = AgentSession(client)
        result = run(session.run("prompt"))
        assert len(result.turns) >= 1

    def test_tool_call_with_empty_arguments(self):
        responses = [
            _with_tools("calling", [{"name": "get_status", "arguments": {}}]),
            _simple("done"),
        ]
        client = FakeClient(responses)
        session = AgentSession(client)
        result = run(session.run("check"))
        assert len(result.turns) == 2

    def test_duration_ms_is_positive(self):
        client = FakeClient([_simple("fast")])
        session = AgentSession(client)
        result = run(session.run("go"))
        for turn in result.turns:
            assert turn.duration_ms >= 0

    def test_total_duration_ms_is_positive(self):
        client = FakeClient([_simple("fast")])
        session = AgentSession(client)
        result = run(session.run("go"))
        assert result.total_duration_ms >= 0

    def test_whitespace_only_prompt(self):
        client = FakeClient([_simple("ok")])
        session = AgentSession(client)
        result = run(session.run("   "))
        assert len(result.turns) >= 1


# ===========================================================================
# CLI integration tests
# ===========================================================================


class TestCLIIntegration:

    def test_agent_command_exists(self):
        from copex.cli import app
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["agent", "--help"])
        assert result.exit_code == 0, f"agent --help failed: {result.output}"

    def test_agent_help_shows_options(self):
        from copex.cli import app
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["agent", "--help"])
        assert "--json" in result.output
        assert "--max-turns" in result.output
        assert "--model" in result.output

    def test_agent_no_prompt_exits_nonzero(self):
        from copex.cli import app
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["agent"])
        assert result.exit_code != 0


# ===========================================================================
# run_streaming tests
# ===========================================================================


class TestRunStreaming:

    def test_streaming_yields_turns(self):
        responses = [
            _with_tools("step 1", [{"name": "bash", "arguments": {"cmd": "ls"}}]),
            _simple("step 2"),
        ]
        client = FakeClient(responses)

        async def _test():
            session = AgentSession(client, max_turns=10)
            turns = []
            async for turn in session.run_streaming("go"):
                turns.append(turn)
            return turns

        turns = run(_test())
        assert len(turns) == 2
        assert turns[0].turn == 1
        assert turns[1].turn == 2

    def test_streaming_stops_on_error(self):
        client = FakeClient()
        client.send = AsyncMock(side_effect=RuntimeError("fail"))

        async def _test():
            session = AgentSession(client)
            turns = []
            async for turn in session.run_streaming("go"):
                turns.append(turn)
            return turns

        turns = run(_test())
        assert len(turns) == 1
        assert turns[0].stop_reason == "error"

    def test_streaming_stops_on_no_tools(self):
        client = FakeClient([_simple("done")])

        async def _test():
            session = AgentSession(client)
            turns = []
            async for turn in session.run_streaming("go"):
                turns.append(turn)
            return turns

        turns = run(_test())
        assert len(turns) == 1
        assert turns[0].stop_reason == "end_turn"


# ===========================================================================
# _extract_tool_calls fallback (raw_events)
# ===========================================================================


class TestExtractToolCallsFallback:
    """Tests for the raw_events fallback in _extract_tool_calls."""

    def test_extract_from_raw_events(self):
        from copex.agent import _extract_tool_calls

        class FakeResponse:
            raw_events = [
                {"type": "tool.call", "data": {"name": "bash", "arguments": {"cmd": "ls"}}},
            ]

        result = _extract_tool_calls(FakeResponse())
        assert len(result) == 1
        assert result[0]["name"] == "bash"
        assert result[0]["arguments"] == {"cmd": "ls"}

    def test_extract_skips_non_dict_events(self):
        from copex.agent import _extract_tool_calls

        class FakeResponse:
            raw_events = ["not a dict", 42, {"type": "tool.call", "data": {"name": "read"}}]

        result = _extract_tool_calls(FakeResponse())
        assert len(result) == 1
        assert result[0]["name"] == "read"

    def test_extract_skips_non_tool_call_events(self):
        from copex.agent import _extract_tool_calls

        class FakeResponse:
            raw_events = [{"type": "message"}, {"type": "tool.result"}]

        result = _extract_tool_calls(FakeResponse())
        assert len(result) == 0

    def test_extract_no_raw_events(self):
        from copex.agent import _extract_tool_calls

        class FakeResponse:
            pass

        result = _extract_tool_calls(FakeResponse())
        assert result == []

    def test_extract_tool_call_without_arguments(self):
        from copex.agent import _extract_tool_calls

        class FakeResponse:
            raw_events = [
                {"type": "tool.call", "data": {"name": "get_status"}},
            ]

        result = _extract_tool_calls(FakeResponse())
        assert len(result) == 1
        assert result[0]["name"] == "get_status"
        assert "arguments" not in result[0]

    def test_extract_tool_call_data_not_dict(self):
        from copex.agent import _extract_tool_calls

        class FakeResponse:
            raw_events = [
                {"type": "tool.call", "data": "not a dict"},
            ]

        result = _extract_tool_calls(FakeResponse())
        assert len(result) == 0
