from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from copex.ralph import RalphConfig, RalphState, RalphWiggum, ralph_loop


@pytest.mark.asyncio
async def test_loop_records_history_until_max_iterations():
    client = SimpleNamespace(send=AsyncMock(side_effect=[
        SimpleNamespace(content="iteration-1"),
        SimpleNamespace(content="iteration-2"),
    ]))
    ralph = RalphWiggum(client, RalphConfig(delay_between_iterations=0))

    state = await ralph.loop("build task", max_iterations=2)

    assert state.completed is True
    assert state.completion_reason == "max_iterations (2)"
    assert state.history == ["iteration-1", "iteration-2"]
    assert client.send.await_count == 2


@pytest.mark.asyncio
async def test_loop_stops_when_completion_promise_is_met():
    client = SimpleNamespace(send=AsyncMock(return_value=SimpleNamespace(
        content="<promise>DONE</promise>"
    )))
    ralph = RalphWiggum(client, RalphConfig(delay_between_iterations=0))

    state = await ralph.loop("finish work", completion_promise="DONE", max_iterations=5)

    assert state.completed is True
    assert state.completion_reason == "promise: DONE"
    assert state.iteration == 1
    assert state.history == ["<promise>DONE</promise>"]


@pytest.mark.asyncio
async def test_loop_stops_after_consecutive_failures():
    client = SimpleNamespace(send=AsyncMock(side_effect=[
        RuntimeError("boom-1"),
        RuntimeError("boom-2"),
        RuntimeError("boom-3"),
    ]))
    config = RalphConfig(delay_between_iterations=0, max_consecutive_errors=2, continue_on_error=True)
    ralph = RalphWiggum(client, config)

    state = await ralph.loop("failing task", max_iterations=10)

    assert state.completed is True
    assert state.completion_reason == "errors: 2 consecutive failures"
    assert state.history == []
    assert client.send.await_count == 2


@pytest.mark.asyncio
async def test_loop_raises_when_continue_on_error_is_disabled():
    client = SimpleNamespace(send=AsyncMock(side_effect=RuntimeError("hard fail")))
    ralph = RalphWiggum(client, RalphConfig(delay_between_iterations=0, continue_on_error=False))

    with pytest.raises(RuntimeError, match="hard fail"):
        await ralph.loop("failing task", max_iterations=3)


@pytest.mark.asyncio
async def test_cancel_marks_state_as_cancelled():
    client = SimpleNamespace(send=AsyncMock())
    ralph = RalphWiggum(client, RalphConfig(delay_between_iterations=0))

    async def _send(_prompt):
        ralph.cancel()
        return SimpleNamespace(content="partial result")

    client.send.side_effect = _send
    state = await ralph.loop("cancel me", max_iterations=5)

    assert state.completed is True
    assert state.completion_reason == "cancelled"
    assert state.history == ["partial result"]


@pytest.mark.asyncio
async def test_callbacks_receive_iteration_and_final_state():
    client = SimpleNamespace(send=AsyncMock(side_effect=[
        SimpleNamespace(content="first pass"),
        SimpleNamespace(content="<promise>DONE</promise>"),
    ]))
    ralph = RalphWiggum(client, RalphConfig(delay_between_iterations=0))
    iterations = []
    completed_states: list[RalphState] = []

    state = await ralph.loop(
        "run callbacks",
        completion_promise="DONE",
        max_iterations=10,
        on_iteration=lambda i, content: iterations.append((i, content)),
        on_complete=lambda final_state: completed_states.append(final_state),
    )

    assert iterations == [(1, "first pass")]
    assert completed_states == [state]
    assert state.history == ["first pass", "<promise>DONE</promise>"]


def test_check_promise_normalizes_whitespace_and_case():
    client = SimpleNamespace(send=AsyncMock())
    ralph = RalphWiggum(client)

    assert ralph._check_promise("<promise>  all done  </promise>", "ALL DONE") is True
    assert ralph._check_promise("<promise>something else</promise>", "ALL DONE") is False


@pytest.mark.asyncio
async def test_ralph_loop_convenience_function():
    client = SimpleNamespace(send=AsyncMock(return_value=SimpleNamespace(content="one")))
    with patch("copex.ralph.asyncio.sleep", new=AsyncMock()):
        state = await ralph_loop(client, "task", max_iterations=1)

    assert isinstance(state, RalphState)
    assert state.history == ["one"]
    assert state.completion_reason == "max_iterations (1)"
