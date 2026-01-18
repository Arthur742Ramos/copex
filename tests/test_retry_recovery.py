"""Comprehensive tests for retry logic and session recovery."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from copex.client import Copex, StreamChunk
from copex.config import CopexConfig, RetryConfig
from copex.models import EventType


def run(coro):
    return asyncio.run(coro)


def build_event(event_type: str, **data):
    return SimpleNamespace(type=event_type, data=SimpleNamespace(**data))


class MockClient:
    """Mock CopilotClient that tracks session creation."""

    def __init__(self, session_factory):
        self.session_factory = session_factory
        self.sessions_created = 0

    async def start(self):
        pass

    async def create_session(self, options):
        self.sessions_created += 1
        return self.session_factory()


class ScriptedSession:
    """Session that follows a script of responses for each send() call."""

    def __init__(self, script: list[list], messages: list | None = None):
        """
        Args:
            script: List of event lists. Each send() pops one list and fires those events.
            messages: Messages to return from get_messages() for context recovery.
        """
        self._script = list(script)
        self._messages = messages or []
        self._handler = None
        self.send_count = 0
        self.prompts_received = []

    def on(self, handler):
        self._handler = handler
        return lambda: None

    async def send(self, options):
        self.send_count += 1
        self.prompts_received.append(options.get("prompt"))

        if self._script:
            events = self._script.pop(0)
            for event in events:
                if self._handler:
                    self._handler(event)

    async def get_messages(self):
        return self._messages

    async def destroy(self):
        pass


# =============================================================================
# Basic Retry Tests
# =============================================================================


class TestBasicRetry:
    """Tests for basic retry behavior within a single session."""

    def test_success_on_first_try(self):
        """No retries needed when first call succeeds."""
        session = ScriptedSession([
            [build_event(EventType.ASSISTANT_MESSAGE.value, content="Hello!"),
             build_event(EventType.SESSION_IDLE.value)],
        ])
        config = CopexConfig(retry=RetryConfig(max_retries=3, base_delay=0.1))
        client = Copex(config)
        client._started = True
        client._client = MockClient(lambda: session)
        client._session = session

        result = run(client.send("Hi"))

        assert result.content == "Hello!"
        assert result.retries == 0
        assert result.auto_continues == 0
        assert session.send_count == 1

    def test_success_after_one_retry(self):
        """Succeeds on second attempt after one failure."""
        session = ScriptedSession([
            [build_event(EventType.SESSION_ERROR.value, message="500 error")],
            [build_event(EventType.ASSISTANT_MESSAGE.value, content="Recovered!"),
             build_event(EventType.SESSION_IDLE.value)],
        ])
        config = CopexConfig(retry=RetryConfig(max_retries=3, base_delay=0.1))
        client = Copex(config)
        client._started = True
        client._client = MockClient(lambda: session)
        client._session = session

        result = run(client.send("Hi"))

        assert result.content == "Recovered!"
        assert result.retries == 1
        assert result.auto_continues == 0
        assert session.send_count == 2

    def test_success_after_max_retries(self):
        """Succeeds on the last retry attempt."""
        session = ScriptedSession([
            [build_event(EventType.SESSION_ERROR.value, message="500")],
            [build_event(EventType.SESSION_ERROR.value, message="500")],
            [build_event(EventType.SESSION_ERROR.value, message="500")],
            [build_event(EventType.ASSISTANT_MESSAGE.value, content="Finally!"),
             build_event(EventType.SESSION_IDLE.value)],
        ])
        config = CopexConfig(
            auto_continue=False,
            retry=RetryConfig(max_retries=3, base_delay=0.1)
        )
        client = Copex(config)
        client._started = True
        client._client = MockClient(lambda: session)
        client._session = session

        result = run(client.send("Hi"))

        assert result.content == "Finally!"
        assert result.retries == 3
        assert session.send_count == 4

    def test_retries_use_same_prompt(self):
        """Retries within limit use the original prompt."""
        session = ScriptedSession([
            [build_event(EventType.SESSION_ERROR.value, message="500")],
            [build_event(EventType.SESSION_ERROR.value, message="500")],
            [build_event(EventType.ASSISTANT_MESSAGE.value, content="OK"),
             build_event(EventType.SESSION_IDLE.value)],
        ])
        config = CopexConfig(
            auto_continue=False,
            retry=RetryConfig(max_retries=3, base_delay=0.1)
        )
        client = Copex(config)
        client._started = True
        client._client = MockClient(lambda: session)
        client._session = session

        run(client.send("Original prompt"))

        # All retries should use the same prompt
        assert all(p == "Original prompt" for p in session.prompts_received)

    def test_fails_after_exhausting_retries_without_auto_continue(self):
        """Raises error when retries exhausted and auto_continue=False."""
        session = ScriptedSession([
            [build_event(EventType.SESSION_ERROR.value, message="500")],
            [build_event(EventType.SESSION_ERROR.value, message="500")],
            [build_event(EventType.SESSION_ERROR.value, message="500")],
            [build_event(EventType.SESSION_ERROR.value, message="500")],
        ])
        config = CopexConfig(
            auto_continue=False,
            retry=RetryConfig(max_retries=3, base_delay=0.1)
        )
        client = Copex(config)
        client._started = True
        client._client = MockClient(lambda: session)
        client._session = session

        with pytest.raises(RuntimeError, match="500"):
            run(client.send("Hi"))

        # Should have tried 4 times (1 initial + 3 retries)
        assert session.send_count == 4


# =============================================================================
# Session Recovery Tests
# =============================================================================


class TestSessionRecovery:
    """Tests for session recovery after retry exhaustion."""

    def test_session_recovery_creates_new_session(self):
        """After retries exhausted, creates a fresh session."""
        sessions = []

        def create_session():
            # First session fails all attempts, second succeeds
            if len(sessions) == 0:
                s = ScriptedSession([
                    [build_event(EventType.SESSION_ERROR.value, message="500")],
                    [build_event(EventType.SESSION_ERROR.value, message="500")],
                ])
            else:
                s = ScriptedSession([
                    [build_event(EventType.ASSISTANT_MESSAGE.value, content="New session works!"),
                     build_event(EventType.SESSION_IDLE.value)],
                ])
            sessions.append(s)
            return s

        mock_client = MockClient(create_session)
        config = CopexConfig(
            auto_continue=True,
            retry=RetryConfig(max_retries=1, max_auto_continues=2, base_delay=0.1)
        )
        client = Copex(config)
        client._started = True
        client._client = mock_client
        client._session = create_session()

        result = run(client.send("Original"))

        assert result.content == "New session works!"
        assert result.auto_continues >= 1
        assert mock_client.sessions_created >= 1

    def test_tool_state_mismatch_recovers_session(self):
        """Tool state mismatch error triggers session recovery."""
        tool_error = (
            "Model call failed: messages.0.content.1: unexpected `tool_use_id` found in "
            "`tool_result` blocks: toolu_123. Each `tool_result` block must have a "
            "corresponding `tool_use` block in the previous message."
        )
        sessions = []

        def create_session():
            if not sessions:
                s = ScriptedSession([
                    [build_event(EventType.SESSION_ERROR.value, message=tool_error)],
                ])
            else:
                s = ScriptedSession([
                    [build_event(EventType.ASSISTANT_MESSAGE.value, content="Recovered!"),
                     build_event(EventType.SESSION_IDLE.value)],
                ])
            sessions.append(s)
            return s

        mock_client = MockClient(create_session)
        config = CopexConfig(
            auto_continue=True,
            retry=RetryConfig(max_retries=2, max_auto_continues=2, base_delay=0.1)
        )
        client = Copex(config)
        client._started = True
        client._client = mock_client
        client._session = create_session()

        result = run(client.send("Hi"))

        assert result.content == "Recovered!"
        assert result.auto_continues == 1
        assert mock_client.sessions_created == 1

    def test_session_recovery_preserves_context(self):
        """Recovery prompt includes conversation history."""
        # First session has message history
        first_session = ScriptedSession(
            script=[
                [build_event(EventType.SESSION_ERROR.value, message="500")],
                [build_event(EventType.SESSION_ERROR.value, message="500")],
            ],
            messages=[
                SimpleNamespace(
                    type=EventType.USER_MESSAGE.value,
                    data=SimpleNamespace(content="What is 2+2?")
                ),
                SimpleNamespace(
                    type=EventType.ASSISTANT_MESSAGE.value,
                    data=SimpleNamespace(content="The answer is 4.")
                ),
            ]
        )

        recovery_prompt_seen = []

        class RecoverySession:
            def __init__(self):
                self._handler = None

            def on(self, handler):
                self._handler = handler
                return lambda: None

            async def send(self, options):
                recovery_prompt_seen.append(options.get("prompt"))
                if self._handler:
                    self._handler(build_event(EventType.ASSISTANT_MESSAGE.value, content="Continued!"))
                    self._handler(build_event(EventType.SESSION_IDLE.value))

            async def get_messages(self):
                return []

            async def destroy(self):
                pass

        session_count = [0]

        def create_session():
            session_count[0] += 1
            return RecoverySession()

        mock_client = MockClient(create_session)
        config = CopexConfig(
            auto_continue=True,
            continue_prompt="Keep going",
            retry=RetryConfig(max_retries=1, max_auto_continues=2, base_delay=0.1)
        )
        client = Copex(config)
        client._started = True
        client._client = mock_client
        client._session = first_session

        result = run(client.send("Original"))

        assert result.content == "Continued!"
        assert len(recovery_prompt_seen) >= 1
        # Check that the recovery prompt contains the conversation context
        prompt = recovery_prompt_seen[0]
        assert "What is 2+2?" in prompt
        assert "The answer is 4" in prompt
        assert "Keep going" in prompt

    def test_multiple_recovery_cycles(self):
        """Can recover multiple times before succeeding."""
        cycle = [0]

        def create_session():
            cycle[0] += 1
            if cycle[0] < 3:
                # First two sessions fail after their retries
                return ScriptedSession([
                    [build_event(EventType.SESSION_ERROR.value, message="500")],
                    [build_event(EventType.SESSION_ERROR.value, message="500")],
                ])
            else:
                # Third session succeeds
                return ScriptedSession([
                    [build_event(EventType.ASSISTANT_MESSAGE.value, content="Third time's the charm!"),
                     build_event(EventType.SESSION_IDLE.value)],
                ])

        mock_client = MockClient(create_session)
        config = CopexConfig(
            auto_continue=True,
            retry=RetryConfig(max_retries=1, max_auto_continues=3, base_delay=0.1)
        )
        client = Copex(config)
        client._started = True
        client._client = mock_client
        client._session = create_session()

        result = run(client.send("Hi"))

        assert result.content == "Third time's the charm!"
        assert result.auto_continues == 2  # Two recovery cycles before success

    def test_fails_after_max_auto_continues(self):
        """Eventually fails when max_auto_continues is exhausted."""
        def create_failing_session():
            return ScriptedSession([
                [build_event(EventType.SESSION_ERROR.value, message="persistent 500")],
                [build_event(EventType.SESSION_ERROR.value, message="persistent 500")],
            ])

        mock_client = MockClient(create_failing_session)
        config = CopexConfig(
            auto_continue=True,
            retry=RetryConfig(max_retries=1, max_auto_continues=2, base_delay=0.1)
        )
        client = Copex(config)
        client._started = True
        client._client = mock_client
        client._session = create_failing_session()

        with pytest.raises(RuntimeError, match="persistent 500"):
            run(client.send("Hi"))

        # Should have created new sessions for recovery attempts
        # Initial + 2 auto-continues = 3 total sessions (but initial is set directly)
        assert mock_client.sessions_created == 2


# =============================================================================
# Error Classification Tests
# =============================================================================


class TestErrorClassification:
    """Tests for which errors trigger retry vs immediate failure."""

    def test_non_retryable_error_raises_immediately(self):
        """Non-retryable errors are raised without retry."""
        session = ScriptedSession([
            [build_event(EventType.SESSION_ERROR.value, message="Invalid syntax")],
        ])
        config = CopexConfig(
            retry=RetryConfig(
                max_retries=5,
                retry_on_any_error=False,
                retry_on_errors=["500", "502", "503"],
                base_delay=0.1
            )
        )
        client = Copex(config)
        client._started = True
        client._client = MockClient(lambda: session)
        client._session = session

        with pytest.raises(RuntimeError, match="Invalid syntax"):
            run(client.send("Hi"))

        # Should have only tried once
        assert session.send_count == 1

    def test_retry_on_any_error_retries_all_errors(self):
        """With retry_on_any_error=True, all errors trigger retry."""
        session = ScriptedSession([
            [build_event(EventType.SESSION_ERROR.value, message="Random error")],
            [build_event(EventType.ASSISTANT_MESSAGE.value, content="OK"),
             build_event(EventType.SESSION_IDLE.value)],
        ])
        config = CopexConfig(
            retry=RetryConfig(max_retries=3, retry_on_any_error=True, base_delay=0.1)
        )
        client = Copex(config)
        client._started = True
        client._client = MockClient(lambda: session)
        client._session = session

        result = run(client.send("Hi"))

        assert result.content == "OK"
        assert result.retries == 1

    def test_retries_on_500_errors(self):
        """500 errors trigger retry with default config."""
        session = ScriptedSession([
            [build_event(EventType.SESSION_ERROR.value, message="500 Internal Server Error")],
            [build_event(EventType.ASSISTANT_MESSAGE.value, content="OK"),
             build_event(EventType.SESSION_IDLE.value)],
        ])
        config = CopexConfig(
            retry=RetryConfig(
                max_retries=3,
                retry_on_any_error=False,
                retry_on_errors=["500"],
                base_delay=0.1
            )
        )
        client = Copex(config)
        client._started = True
        client._client = MockClient(lambda: session)
        client._session = session

        result = run(client.send("Hi"))

        assert result.content == "OK"
        assert result.retries == 1

    def test_retries_on_rate_limit(self):
        """Rate limit errors trigger retry."""
        session = ScriptedSession([
            [build_event(EventType.SESSION_ERROR.value, message="Rate limit exceeded")],
            [build_event(EventType.ASSISTANT_MESSAGE.value, content="OK"),
             build_event(EventType.SESSION_IDLE.value)],
        ])
        config = CopexConfig(
            retry=RetryConfig(
                max_retries=3,
                retry_on_any_error=False,
                retry_on_errors=["rate limit"],
                base_delay=0.1
            )
        )
        client = Copex(config)
        client._started = True
        client._client = MockClient(lambda: session)
        client._session = session

        result = run(client.send("Hi"))

        assert result.content == "OK"


# =============================================================================
# Streaming Callback Tests
# =============================================================================


class TestStreamingCallbacks:
    """Tests for on_chunk callbacks during retry/recovery."""

    def test_retry_notification_sent_to_callback(self):
        """Retry attempts notify via on_chunk callback."""
        session = ScriptedSession([
            [build_event(EventType.SESSION_ERROR.value, message="500 error")],
            [build_event(EventType.ASSISTANT_MESSAGE.value, content="OK"),
             build_event(EventType.SESSION_IDLE.value)],
        ])
        config = CopexConfig(
            auto_continue=False,
            retry=RetryConfig(max_retries=3, base_delay=0.1)
        )
        client = Copex(config)
        client._started = True
        client._client = MockClient(lambda: session)
        client._session = session

        chunks = []
        result = run(client.send("Hi", on_chunk=chunks.append))

        # Should have received a system chunk about retry
        system_chunks = [c for c in chunks if c.type == "system"]
        assert len(system_chunks) >= 1
        assert "Retry" in system_chunks[0].delta
        assert "1/" in system_chunks[0].delta

    def test_recovery_notification_sent_to_callback(self):
        """Session recovery notifies via on_chunk callback."""
        sessions = []

        def create_session():
            if len(sessions) == 0:
                s = ScriptedSession([
                    [build_event(EventType.SESSION_ERROR.value, message="500")],
                    [build_event(EventType.SESSION_ERROR.value, message="500")],
                ])
            else:
                s = ScriptedSession([
                    [build_event(EventType.ASSISTANT_MESSAGE.value, content="OK"),
                     build_event(EventType.SESSION_IDLE.value)],
                ])
            sessions.append(s)
            return s

        mock_client = MockClient(create_session)
        config = CopexConfig(
            auto_continue=True,
            retry=RetryConfig(max_retries=1, max_auto_continues=2, base_delay=0.1)
        )
        client = Copex(config)
        client._started = True
        client._client = mock_client
        client._session = create_session()

        chunks = []
        result = run(client.send("Hi", on_chunk=chunks.append))

        # Should have system chunks about both retry and recovery
        system_chunks = [c for c in chunks if c.type == "system"]
        system_text = "".join(c.delta for c in system_chunks)
        assert "Retry" in system_text
        assert "Auto-continue" in system_text or "recovered" in system_text


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests for retry/recovery logic."""

    def test_empty_context_recovery(self):
        """Recovery works even with no conversation history."""
        first_session = ScriptedSession(
            script=[
                [build_event(EventType.SESSION_ERROR.value, message="500")],
                [build_event(EventType.SESSION_ERROR.value, message="500")],
            ],
            messages=[]  # No history
        )

        recovery_prompt_seen = []

        class RecoverySession:
            def __init__(self):
                self._handler = None

            def on(self, handler):
                self._handler = handler
                return lambda: None

            async def send(self, options):
                recovery_prompt_seen.append(options.get("prompt"))
                if self._handler:
                    self._handler(build_event(EventType.ASSISTANT_MESSAGE.value, content="OK"))
                    self._handler(build_event(EventType.SESSION_IDLE.value))

            async def get_messages(self):
                return []

            async def destroy(self):
                pass

        mock_client = MockClient(lambda: RecoverySession())
        config = CopexConfig(
            auto_continue=True,
            continue_prompt="Keep going",
            retry=RetryConfig(max_retries=1, max_auto_continues=2, base_delay=0.1)
        )
        client = Copex(config)
        client._started = True
        client._client = mock_client
        client._session = first_session

        result = run(client.send("Hi"))

        assert result.content == "OK"
        # With no history, should just use continue_prompt
        assert recovery_prompt_seen[0] == "Keep going"

    def test_context_extraction_failure_handled(self):
        """Recovery continues even if get_messages() fails."""
        class FailingSession:
            def __init__(self):
                self._handler = None
                self.send_count = 0

            def on(self, handler):
                self._handler = handler
                return lambda: None

            async def send(self, options):
                self.send_count += 1
                if self._handler:
                    self._handler(build_event(EventType.SESSION_ERROR.value, message="500"))

            async def get_messages(self):
                raise RuntimeError("get_messages failed!")

            async def destroy(self):
                pass

        class SuccessSession:
            def __init__(self):
                self._handler = None

            def on(self, handler):
                self._handler = handler
                return lambda: None

            async def send(self, options):
                if self._handler:
                    self._handler(build_event(EventType.ASSISTANT_MESSAGE.value, content="Recovered"))
                    self._handler(build_event(EventType.SESSION_IDLE.value))

            async def get_messages(self):
                return []

            async def destroy(self):
                pass

        session_count = [0]

        def create_session():
            session_count[0] += 1
            return SuccessSession()

        mock_client = MockClient(create_session)
        config = CopexConfig(
            auto_continue=True,
            retry=RetryConfig(max_retries=1, max_auto_continues=2, base_delay=0.1)
        )
        client = Copex(config)
        client._started = True
        client._client = mock_client
        client._session = FailingSession()

        result = run(client.send("Hi"))

        assert result.content == "Recovered"

    def test_session_destroy_failure_handled(self):
        """Recovery continues even if session.destroy() fails."""
        class FailingDestroySession:
            def __init__(self):
                self._handler = None

            def on(self, handler):
                self._handler = handler
                return lambda: None

            async def send(self, options):
                if self._handler:
                    self._handler(build_event(EventType.SESSION_ERROR.value, message="500"))

            async def get_messages(self):
                return []

            async def destroy(self):
                raise RuntimeError("destroy failed!")

        class SuccessSession:
            def __init__(self):
                self._handler = None

            def on(self, handler):
                self._handler = handler
                return lambda: None

            async def send(self, options):
                if self._handler:
                    self._handler(build_event(EventType.ASSISTANT_MESSAGE.value, content="OK"))
                    self._handler(build_event(EventType.SESSION_IDLE.value))

            async def get_messages(self):
                return []

            async def destroy(self):
                pass

        mock_client = MockClient(lambda: SuccessSession())
        config = CopexConfig(
            auto_continue=True,
            retry=RetryConfig(max_retries=1, max_auto_continues=2, base_delay=0.1)
        )
        client = Copex(config)
        client._started = True
        client._client = mock_client
        client._session = FailingDestroySession()

        result = run(client.send("Hi"))

        assert result.content == "OK"

    def test_response_tracks_retry_and_continue_counts(self):
        """Response object correctly tracks retries and auto_continues."""
        cycle = [0]

        def create_session():
            cycle[0] += 1
            if cycle[0] == 1:
                # First session: 2 retries then recovery
                return ScriptedSession([
                    [build_event(EventType.SESSION_ERROR.value, message="500")],
                    [build_event(EventType.SESSION_ERROR.value, message="500")],
                    [build_event(EventType.SESSION_ERROR.value, message="500")],
                ])
            elif cycle[0] == 2:
                # Second session: 1 retry then recovery
                return ScriptedSession([
                    [build_event(EventType.SESSION_ERROR.value, message="500")],
                    [build_event(EventType.SESSION_ERROR.value, message="500")],
                    [build_event(EventType.SESSION_ERROR.value, message="500")],
                ])
            else:
                # Third session: succeeds after 1 retry
                return ScriptedSession([
                    [build_event(EventType.SESSION_ERROR.value, message="500")],
                    [build_event(EventType.ASSISTANT_MESSAGE.value, content="Finally!"),
                     build_event(EventType.SESSION_IDLE.value)],
                ])

        mock_client = MockClient(create_session)
        config = CopexConfig(
            auto_continue=True,
            retry=RetryConfig(max_retries=2, max_auto_continues=3, base_delay=0.1)
        )
        client = Copex(config)
        client._started = True
        client._client = mock_client
        client._session = create_session()

        result = run(client.send("Hi"))

        assert result.content == "Finally!"
        assert result.auto_continues == 2  # Two session recoveries
        assert result.retries == 1  # Final retry count when succeeded

    def test_long_context_is_truncated(self):
        """Very long conversation history is truncated in recovery prompt."""
        long_content = "A" * 2000  # Longer than the 1000 char limit

        first_session = ScriptedSession(
            script=[
                [build_event(EventType.SESSION_ERROR.value, message="500")],
                [build_event(EventType.SESSION_ERROR.value, message="500")],
            ],
            messages=[
                SimpleNamespace(
                    type=EventType.ASSISTANT_MESSAGE.value,
                    data=SimpleNamespace(content=long_content)
                ),
            ]
        )

        recovery_prompt_seen = []

        class RecoverySession:
            def __init__(self):
                self._handler = None

            def on(self, handler):
                self._handler = handler
                return lambda: None

            async def send(self, options):
                recovery_prompt_seen.append(options.get("prompt"))
                if self._handler:
                    self._handler(build_event(EventType.ASSISTANT_MESSAGE.value, content="OK"))
                    self._handler(build_event(EventType.SESSION_IDLE.value))

            async def get_messages(self):
                return []

            async def destroy(self):
                pass

        mock_client = MockClient(lambda: RecoverySession())
        config = CopexConfig(
            auto_continue=True,
            retry=RetryConfig(max_retries=1, max_auto_continues=2, base_delay=0.1)
        )
        client = Copex(config)
        client._started = True
        client._client = mock_client
        client._session = first_session

        result = run(client.send("Hi"))

        # Context should be truncated (1000 chars + "...")
        prompt = recovery_prompt_seen[0]
        assert "..." in prompt
        # Should not contain the full 2000 char string
        assert len(prompt) < 2000 + 500  # Some buffer for prompt structure
