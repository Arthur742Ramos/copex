from __future__ import annotations

import asyncio
from types import SimpleNamespace
import threading

import pytest

from copex.client import Copex, StreamChunk
import copex.client as client_module
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


class MockRpcClient:
    def __init__(self, owner, session_factory):
        self._owner = owner
        self._session_factory = session_factory
        self._sessions = {}
        self._next_id = 0

    async def request(self, method, _payload):
        if method != "session.create":
            raise RuntimeError(f"Unexpected RPC method: {method}")
        self._next_id += 1
        session_id = f"session-{self._next_id}"
        session = self._session_factory()
        if self._owner is not None:
            self._owner.sessions_created += 1
        self._sessions[session_id] = session
        return {"sessionId": session_id, "workspacePath": None}


class FakeCopilotSession:
    def __init__(self, session_id, rpc_client, _workspace_path=None):
        self._inner = rpc_client._sessions[session_id]

    def on(self, handler):
        return self._inner.on(handler)

    async def send(self, options):
        return await self._inner.send(options)

    async def get_messages(self):
        return await self._inner.get_messages()

    async def destroy(self):
        return await self._inner.destroy()


client_module.CopilotSession = FakeCopilotSession


def build_event(event_type: str, **data):
    return SimpleNamespace(type=event_type, data=SimpleNamespace(**data))


def run(coro):
    return asyncio.run(coro)


def test_send_uses_message_delta_and_final_content():
    events = [
        build_event(EventType.ASSISTANT_MESSAGE_DELTA.value, delta_content="Hello "),
        build_event(EventType.ASSISTANT_MESSAGE_DELTA.value, delta_content="world"),
        build_event(
            EventType.ASSISTANT_MESSAGE.value,
            content="Hello world",
            usage={"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
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

    assert response.content == "Hello world"
    assert response.token_usage is not None
    assert response.token_usage.total == 8
    assert "".join(c.delta for c in chunks if c.type == "message") == "Hello world"


def test_send_uses_transformed_content_when_content_missing():
    events = [
        build_event(
            EventType.ASSISTANT_MESSAGE.value,
            content="",
            transformed_content="Fallback",
            usage={"prompt": 2, "completion": 4},
        ),
        build_event(EventType.SESSION_IDLE.value),
    ]
    session = FakeSession(events)
    client = Copex(CopexConfig())
    client._started = True
    client._client = DummyClient()
    client._session = session

    response = run(client.send("prompt"))

    assert response.content == "Fallback"
    assert response.token_usage is not None
    assert response.token_usage.total == 6


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
    # Disable auto_continue and set max_retries=1 for fast failure
    from copex.config import RetryConfig
    config = CopexConfig(auto_continue=False, retry=RetryConfig(max_retries=1, base_delay=0.1))
    client = Copex(config)
    client._started = True
    client._client = DummyClient()
    client._session = session

    with pytest.raises(RuntimeError, match="boom"):
        run(client.send("prompt"))


def test_auto_continue_after_exhausted_retries():
    """Test that auto-continue triggers with session recovery after retries exhausted."""
    call_count = 0
    prompts_seen = []
    class CountingSession:
        def __init__(self):
            self._handler = None

        def on(self, handler):
            self._handler = handler
            return lambda: None

        async def send(self, options):
            nonlocal call_count
            call_count += 1
            prompts_seen.append(options.get("prompt"))
            # First few calls fail, then succeed
            if call_count <= 4:  # 2 retries + 1 auto-continue + 1 more retry
                if self._handler:
                    self._handler(SimpleNamespace(
                        type=EventType.SESSION_ERROR.value,
                        data=SimpleNamespace(message="500 Internal Server Error")
                    ))
            else:
                if self._handler:
                    self._handler(SimpleNamespace(
                        type=EventType.ASSISTANT_MESSAGE.value,
                        data=SimpleNamespace(content="success!")
                    ))
                    self._handler(SimpleNamespace(
                        type=EventType.SESSION_IDLE.value,
                        data=None
                    ))

        async def get_messages(self):
            return []

        async def destroy(self):
            pass

    class MockClient:
        def __init__(self):
            self._client = MockRpcClient(self, CountingSession)
            self._sessions_lock = threading.Lock()
            self._sessions = {}
            self.sessions_created = 0

        async def start(self):
            return None

        async def create_session(self, options):
            self.sessions_created += 1
            return CountingSession()

    from copex.config import RetryConfig
    config = CopexConfig(
        auto_continue=True,
        continue_prompt="Keep going",
        retry=RetryConfig(max_retries=2, max_auto_continues=2, base_delay=0.1)
    )
    client = Copex(config)
    client._started = True
    client._client = MockClient()
    client._session = CountingSession()

    result = run(client.send("original prompt"))

    assert result.content == "success!"
    assert result.auto_continues >= 1
    # Session recovery should have created a new session
    assert client._client.sessions_created >= 1
    # Should have seen recovery prompt (contains "Keep going")
    assert any("Keep going" in p for p in prompts_seen if p)
