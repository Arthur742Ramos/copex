from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from copex.client import Copex, StreamChunk
from copex.config import CopexConfig
from copex.models import EventType


class FakeSession:
    def __init__(self, events, messages=None):
        self._events = events
        self._messages = messages or []
        self._handler = None

    def on(self, handler):
        self._handler = handler

        def unsubscribe():
            self._handler = None

        return unsubscribe

    async def send(self, _options):
        for event in self._events:
            if self._handler:
                self._handler(event)

    async def get_messages(self):
        return self._messages


class DummyClient:
    async def start(self):
        return None


def build_event(event_type: str, **data):
    return SimpleNamespace(type=event_type, data=SimpleNamespace(**data))


def run(coro):
    return asyncio.run(coro)


def test_send_uses_message_delta_and_final_content():
    events = [
        build_event(EventType.ASSISTANT_MESSAGE_DELTA.value, delta_content="Hello "),
        build_event(EventType.ASSISTANT_MESSAGE_DELTA.value, delta_content="world"),
        build_event(EventType.ASSISTANT_MESSAGE.value, content="Hello world"),
        build_event(EventType.SESSION_IDLE.value),
    ]
    session = FakeSession(events)
    client = Copex(CopexConfig())
    client._started = True
    client._client = DummyClient()
    client._session = session

    chunks: list[StreamChunk] = []
    response = run(client.send("prompt", on_chunk=chunks.append))

    assert response.content == "Hello world"
    assert "".join(c.delta for c in chunks if c.type == "message") == "Hello world"


def test_send_uses_transformed_content_when_content_missing():
    events = [
        build_event(EventType.ASSISTANT_MESSAGE.value, content="", transformed_content="Fallback"),
        build_event(EventType.SESSION_IDLE.value),
    ]
    session = FakeSession(events)
    client = Copex(CopexConfig())
    client._started = True
    client._client = DummyClient()
    client._session = session

    response = run(client.send("prompt"))

    assert response.content == "Fallback"


def test_send_sets_tool_result_chunks():
    result_obj = SimpleNamespace(content="ok")
    events = [
        build_event(EventType.TOOL_EXECUTION_START.value, tool_name="read_file", arguments={"path": "a.txt"}),
        build_event(EventType.TOOL_EXECUTION_PARTIAL_RESULT.value, tool_name="read_file", partial_output="partial"),
        build_event(
            EventType.TOOL_EXECUTION_COMPLETE.value,
            tool_name="read_file",
            result=result_obj,
            success=True,
            duration=1.25,
        ),
        build_event(EventType.SESSION_IDLE.value),
    ]
    session = FakeSession(events)
    client = Copex(CopexConfig())
    client._started = True
    client._client = DummyClient()
    client._session = session

    chunks: list[StreamChunk] = []
    response = run(client.send("prompt", on_chunk=chunks.append))

    assert response.content == ""
    tool_calls = [c for c in chunks if c.type == "tool_call"]
    tool_results = [c for c in chunks if c.type == "tool_result"]
    assert tool_calls and tool_calls[0].tool_name == "read_file"
    assert tool_results[-1].tool_result == "ok"
    assert tool_results[-1].tool_success is True
    assert tool_results[-1].tool_duration == 1.25


def test_send_falls_back_to_session_history():
    """Non-streaming mode should fall back to history when no content received."""
    events = [
        build_event(EventType.ASSISTANT_TURN_END.value),
    ]
    message_event = SimpleNamespace(
        type=EventType.ASSISTANT_MESSAGE.value,
        data=SimpleNamespace(content="From history"),
    )
    session = FakeSession(events, messages=[message_event])
    client = Copex(CopexConfig())
    client._started = True
    client._client = DummyClient()
    client._session = session

    response = run(client.send("prompt"))

    assert response.content == "From history"


def test_streaming_does_not_use_history_fallback():
    """Streaming mode should NOT fall back to history - avoids stale content from previous turns."""
    events = [
        build_event(EventType.ASSISTANT_REASONING_DELTA.value, delta_content="Thinking..."),
        build_event(EventType.ASSISTANT_REASONING.value, content="Thinking..."),
        build_event(EventType.ASSISTANT_TURN_END.value),
    ]
    # History contains stale content from a previous turn
    stale_message = SimpleNamespace(
        type=EventType.ASSISTANT_MESSAGE.value,
        data=SimpleNamespace(content="Stale previous response"),
    )
    session = FakeSession(events, messages=[stale_message])
    client = Copex(CopexConfig())
    client._started = True
    client._client = DummyClient()
    client._session = session

    chunks: list[StreamChunk] = []
    response = run(client.send("prompt", on_chunk=chunks.append))

    # Should NOT contain stale history content
    assert response.content != "Stale previous response"
    assert response.content == ""
    # Reasoning should be captured
    assert response.reasoning == "Thinking..."


def test_streaming_uses_deltas_not_history():
    """Streaming should use accumulated deltas, not history fallback."""
    events = [
        build_event(EventType.ASSISTANT_MESSAGE_DELTA.value, delta_content="Current "),
        build_event(EventType.ASSISTANT_MESSAGE_DELTA.value, delta_content="response"),
        build_event(EventType.ASSISTANT_TURN_END.value),
    ]
    # History contains different (stale) content
    stale_message = SimpleNamespace(
        type=EventType.ASSISTANT_MESSAGE.value,
        data=SimpleNamespace(content="Old stale message"),
    )
    session = FakeSession(events, messages=[stale_message])
    client = Copex(CopexConfig())
    client._started = True
    client._client = DummyClient()
    client._session = session

    chunks: list[StreamChunk] = []
    response = run(client.send("prompt", on_chunk=chunks.append))

    # Should use streamed content, not history
    assert response.content == "Current response"
    assert "".join(c.delta for c in chunks if c.type == "message") == "Current response"


def test_streaming_with_reasoning_captures_both():
    """Streaming with reasoning should capture both reasoning and message content."""
    events = [
        build_event(EventType.ASSISTANT_REASONING_DELTA.value, delta_content="Let me think"),
        build_event(EventType.ASSISTANT_REASONING.value, content="Let me think"),
        build_event(EventType.ASSISTANT_MESSAGE_DELTA.value, delta_content="Here's my answer"),
        build_event(EventType.ASSISTANT_MESSAGE.value, content="Here's my answer"),
        build_event(EventType.ASSISTANT_TURN_END.value),
    ]
    session = FakeSession(events)
    client = Copex(CopexConfig())
    client._started = True
    client._client = DummyClient()
    client._session = session

    chunks: list[StreamChunk] = []
    response = run(client.send("prompt", on_chunk=chunks.append))

    assert response.content == "Here's my answer"
    assert response.reasoning == "Let me think"
    
    reasoning_chunks = [c for c in chunks if c.type == "reasoning"]
    message_chunks = [c for c in chunks if c.type == "message"]
    assert len(reasoning_chunks) >= 1
    assert len(message_chunks) >= 1


def test_session_error_raises():
    events = [
        build_event(EventType.SESSION_ERROR.value, message="boom"),
    ]
    session = FakeSession(events)
    client = Copex(CopexConfig())
    client._started = True
    client._client = DummyClient()
    client._session = session

    with pytest.raises(RuntimeError, match="boom"):
        run(client.send("prompt"))
