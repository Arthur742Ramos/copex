from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from copex.client import Copex, StreamChunk
from copex.config import COPILOT_CLI_NOT_FOUND_MESSAGE, CopexConfig, RetryConfig
from copex.exceptions import CopexTimeoutError
from copex.models import EventType


def run(coro):
    return asyncio.run(coro)


def build_event(event_type: str, **data):
    return SimpleNamespace(type=event_type, data=SimpleNamespace(**data))


class FakeSession:
    def __init__(self, scripts: list[list[SimpleNamespace]], messages: list | None = None):
        self._scripts = list(scripts)
        self._messages = messages or []
        self._handler = None
        self.send_count = 0
        self.prompts: list[str | None] = []
        self.destroy = AsyncMock()
        self.abort = AsyncMock()

    def on(self, handler):
        self._handler = handler

        def unsubscribe():
            self._handler = None

        return unsubscribe

    async def send(self, options):
        self.send_count += 1
        self.prompts.append(options.get("prompt"))
        if not self._scripts:
            return
        for event in self._scripts.pop(0):
            if self._handler:
                self._handler(event)

    async def get_messages(self):
        return self._messages


@pytest.fixture(autouse=True)
def _disable_memory_capture(monkeypatch):
    monkeypatch.setattr("copex.client.auto_capture_memory", lambda *args, **kwargs: None)


class TestClientSessionLifecycle:
    def test_start_initializes_client_once(self):
        mock_sdk_client = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
        with patch("copex.client.CopilotClient", return_value=mock_sdk_client) as mock_ctor:
            client = Copex(CopexConfig())
            run(client.start())
            run(client.start())

        mock_ctor.assert_called_once()
        assert mock_sdk_client.start.await_count == 1

    def test_start_surfaces_clear_error_when_cli_lookup_fails(self):
        with patch("copex.client.CopilotClient", side_effect=TypeError("missing cli")):
            client = Copex(CopexConfig())
            with pytest.raises(RuntimeError, match=COPILOT_CLI_NOT_FOUND_MESSAGE):
                run(client.start())

    def test_ensure_session_uses_fallback_model_and_drops_reasoning(self):
        created_session = FakeSession([])
        mock_sdk_client = SimpleNamespace(
            create_session=AsyncMock(return_value=created_session),
            start=AsyncMock(),
            stop=AsyncMock(),
        )
        client = Copex(CopexConfig())
        client._started = True
        client._client = mock_sdk_client
        client._current_model = "fallback-model"

        with patch("copex.models.model_supports_reasoning", return_value=False):
            session = run(client._ensure_session())

        assert session is created_session
        create_options = mock_sdk_client.create_session.await_args.args[0]
        assert create_options["model"] == "fallback-model"
        assert "reasoning_effort" not in create_options

    def test_new_session_clears_and_destroys_existing_session(self):
        client = Copex(CopexConfig())
        session = FakeSession([])
        client._session = session

        client.new_session()

        assert client._session is None
        assert session.destroy.await_count == 1


class TestClientSendAndStream:
    def test_send_collects_usage_metadata(self):
        events = [[
            build_event(EventType.ASSISTANT_MESSAGE.value, content="Hello"),
            build_event("assistant.usage", input_tokens=12, output_tokens=34, cost=0.5, model="m1"),
            build_event(EventType.SESSION_IDLE.value),
        ]]
        session = FakeSession(events)
        client = Copex(CopexConfig())
        client._started = True
        client._client = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
        client._session = session

        response = run(client.send("prompt"))

        assert response.content == "Hello"
        assert response.prompt_tokens == 12
        assert response.completion_tokens == 34
        assert response.cost == 0.5
        assert response.server_model == "m1"

    def test_stream_yields_chunks(self):
        events = [[
            build_event(EventType.ASSISTANT_MESSAGE_DELTA.value, delta_content="Hello "),
            build_event(EventType.ASSISTANT_MESSAGE.value, content="Hello world"),
            build_event(EventType.SESSION_IDLE.value),
        ]]
        session = FakeSession(events)
        client = Copex(CopexConfig())
        client._started = True
        client._client = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
        client._session = session

        async def _collect():
            chunks = []
            async for chunk in client.stream("prompt"):
                chunks.append(chunk)
            return chunks

        chunks = run(_collect())

        assert any(c.type == "message" and c.delta for c in chunks)
        assert any(c.type == "message" and c.is_final for c in chunks)

    def test_tool_call_argument_parsing_handles_json_and_raw(self):
        events = [[
            build_event(EventType.TOOL_CALL.value, name="read_file", arguments='{"path":"a.txt"}'),
            build_event(EventType.TOOL_CALL.value, name="run_shell", arguments="ls -la"),
            build_event(EventType.ASSISTANT_MESSAGE.value, content="done"),
            build_event(EventType.SESSION_IDLE.value),
        ]]
        session = FakeSession(events)
        client = Copex(CopexConfig())
        client._started = True
        client._client = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
        client._session = session

        chunks: list[StreamChunk] = []
        run(client.send("prompt", on_chunk=chunks.append))
        tool_calls = [c for c in chunks if c.type == "tool_call"]

        assert tool_calls[0].tool_name == "read_file"
        assert tool_calls[0].tool_args == {"path": "a.txt"}
        assert tool_calls[1].tool_name == "run_shell"
        assert tool_calls[1].tool_args == {"raw": "ls -la"}

    def test_send_offloads_approval_workflow_to_executor(self, monkeypatch):
        created: list[object] = []
        main_thread = threading.get_ident()

        class FakeApprovalWorkflow:
            def __init__(self, *args, **kwargs):
                self.review_thread = None
                self.apply_thread = None
                self.log_thread = None
                created.append(self)

            def review_tool_call(self, tool_name, tool_args, *, cwd=None):
                self.review_thread = threading.get_ident()
                return ["reviewed"]

            def apply_post_tool_decisions(self, reviewed):
                self.apply_thread = threading.get_ident()
                return ["applied"]

            def log_execution_event(self, reviewed, *, success, result=None, error=None):
                self.log_thread = threading.get_ident()

        monkeypatch.setattr("copex.approval.ApprovalWorkflow", FakeApprovalWorkflow)

        events = [[
            build_event(
                EventType.TOOL_CALL.value,
                name="write_file",
                arguments='{"path":"a.txt","content":"updated"}',
            ),
            build_event(
                EventType.TOOL_EXECUTION_COMPLETE.value,
                tool_name="write_file",
                success=True,
                result=SimpleNamespace(content="ok"),
            ),
            build_event(EventType.ASSISTANT_MESSAGE.value, content="done"),
            build_event(EventType.SESSION_IDLE.value),
        ]]
        session = FakeSession(events)
        client = Copex(CopexConfig(approval_mode="manual"))
        client._started = True
        client._client = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
        client._session = session

        response = run(client.send("prompt"))

        assert response.content == "done"
        workflow = created[0]
        assert workflow.review_thread is not None and workflow.review_thread != main_thread
        assert workflow.apply_thread is not None and workflow.apply_thread != main_thread
        assert workflow.log_thread is not None and workflow.log_thread != main_thread


class TestClientRetryAndErrors:
    def test_non_retryable_error_raises_without_retry(self):
        events = [[build_event(EventType.SESSION_ERROR.value, message="HTTP 401 Unauthorized")]]
        session = FakeSession(events)
        config = CopexConfig(
            auto_continue=False,
            retry=RetryConfig(max_retries=3, retry_on_any_error=False, base_delay=0.1),
        )
        client = Copex(config)
        client._started = True
        client._client = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
        client._session = session

        with pytest.raises(RuntimeError, match="401"):
            run(client.send("prompt"))
        assert session.send_count == 1

    def test_retryable_error_retries_then_succeeds(self):
        events = [
            [build_event(EventType.SESSION_ERROR.value, message="HTTP 500")],
            [
                build_event(EventType.ASSISTANT_MESSAGE.value, content="Recovered"),
                build_event(EventType.SESSION_IDLE.value),
            ],
        ]
        session = FakeSession(events)
        config = CopexConfig(
            auto_continue=False,
            retry=RetryConfig(max_retries=2, retry_on_any_error=False, base_delay=0.1),
        )
        client = Copex(config)
        client._started = True
        client._client = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
        client._session = session

        with patch("copex.client.asyncio.sleep", new=AsyncMock()):
            response = run(client.send("prompt"))

        assert response.content == "Recovered"
        assert response.retries == 1
        assert session.send_count == 2

    def test_recover_session_rebuilds_prompt_from_context(self):
        old_messages = [
            SimpleNamespace(
                type=EventType.USER_MESSAGE,
                data=SimpleNamespace(content="Need help with auth"),
            ),
            SimpleNamespace(
                type=EventType.ASSISTANT_MESSAGE,
                data=SimpleNamespace(content="I can help with that."),
            ),
        ]
        old_session = FakeSession([], messages=old_messages)
        new_session = FakeSession([])
        mock_sdk_client = SimpleNamespace(
            create_session=AsyncMock(return_value=new_session),
            start=AsyncMock(),
            stop=AsyncMock(),
        )
        client = Copex(CopexConfig())
        client._started = True
        client._client = mock_sdk_client
        client._session = old_session

        recovered_session, recovered_prompt = run(client._recover_session(None))

        assert recovered_session is new_session
        assert "User: Need help with auth" in recovered_prompt
        assert "Assistant: I can help with that." in recovered_prompt
        assert client.config.continue_prompt in recovered_prompt
        assert old_session.destroy.await_count == 1

    def test_send_once_times_out_after_inactivity(self):
        session = FakeSession([[]])
        client = Copex(CopexConfig())
        client.config.timeout = 0.01

        with pytest.raises(CopexTimeoutError, match="timed out"):
            run(client._send_once(session, "prompt", None, None))
