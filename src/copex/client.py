"""Core Copex client with retry logic and stuck detection."""

from __future__ import annotations

import asyncio
import logging
import random
import time
import warnings
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from copilot import CopilotClient

from copex.backoff import AdaptiveRetry, ErrorCategory, categorize_error
from copex.circuit_breaker import (
    CB_COOLDOWN_SECONDS as _CB_COOLDOWN_SECONDS,
)
from copex.circuit_breaker import (
    CB_FAILURE_THRESHOLD as _CB_FAILURE_THRESHOLD,
)
from copex.circuit_breaker import (
    DEFAULT_FALLBACK_CHAINS,
    ModelAwareBreaker,
    SlidingWindowBreaker,  # noqa: F401
)
from copex.config import CopexConfig
from copex.metrics import MetricsCollector, get_collector
from copex.models import EventType, Model, ReasoningEffort, parse_reasoning_effort
from copex.sdk_patch import patch_copilot_client  # noqa: F401
from copex.session_pool import SessionPool  # noqa: F401
from copex.streaming import ChunkBatcher, Response, StreamChunk, StreamingMetrics  # noqa: F401

logger = logging.getLogger(__name__)

MAX_RAW_EVENTS = 10_000

# Pre-cached EventType values to avoid repeated .value attribute access in the
# hot path (on_event is called for every single streaming token).
_ET_MSG_DELTA = EventType.ASSISTANT_MESSAGE_DELTA.value
_ET_REASON_DELTA = EventType.ASSISTANT_REASONING_DELTA.value
_ET_MSG = EventType.ASSISTANT_MESSAGE.value
_ET_REASON = EventType.ASSISTANT_REASONING.value
_ET_TOOL_START = EventType.TOOL_EXECUTION_START.value
_ET_TOOL_PARTIAL = EventType.TOOL_EXECUTION_PARTIAL_RESULT.value
_ET_TOOL_COMPLETE = EventType.TOOL_EXECUTION_COMPLETE.value
_ET_ERROR = EventType.ERROR.value
_ET_SESSION_ERROR = EventType.SESSION_ERROR.value
_ET_TOOL_CALL = EventType.TOOL_CALL.value
_ET_TURN_END = EventType.ASSISTANT_TURN_END.value
_ET_SESSION_IDLE = EventType.SESSION_IDLE.value
_ET_USAGE = "assistant.usage"


@dataclass
class _SendState:
    """State for handling a single send call."""

    done: asyncio.Event
    error_holder: list[Exception] = field(default_factory=list)
    content_parts: list[str] = field(default_factory=list)
    reasoning_parts: list[str] = field(default_factory=list)
    final_content: str | None = None
    final_reasoning: str | None = None
    raw_events: list[dict[str, Any]] = field(default_factory=list)
    last_activity: float = 0.0
    received_content: bool = False
    pending_tools: int = 0
    awaiting_post_tool_response: bool = False
    tool_execution_seen: bool = False

    # Usage/cost (from "assistant.usage" events, when present)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost: float | None = None
    server_model: str | None = None

    # Streaming performance tracking
    streaming_metrics: StreamingMetrics = field(default_factory=StreamingMetrics)


@dataclass(frozen=True)
class _ToolExcludePattern:
    name: str
    arg: str | None = None


class Copex:
    """Copilot Extended - Resilient wrapper with automatic retry and stuck detection."""

    # Shared model-aware circuit breaker for fallback support (v1.9.0)
    _model_breaker: ModelAwareBreaker | None = None

    @classmethod
    def get_model_breaker(cls) -> ModelAwareBreaker:
        """Get the shared model-aware circuit breaker instance."""
        if cls._model_breaker is None:
            cls._model_breaker = ModelAwareBreaker()
        return cls._model_breaker

    def __init__(
        self,
        config: CopexConfig | None = None,
        *,
        fallback_chain: list[str] | None = None,
    ):
        self.config = config or CopexConfig()
        self._client: CopilotClient | None = None
        self._session: Any = None
        self._started = False
        # Circuit breaker state (legacy per-instance breaker)
        self._cb_failures = 0
        self._cb_opened_at: float | None = None
        self._destroy_tasks: set[asyncio.Task[None]] = set()
        # Model fallback chain (v1.9.0)
        self._fallback_chain = fallback_chain
        # Track current model (may differ from config if fallback is active)
        self._current_model: str | None = None

    async def start(self) -> None:
        """Start the Copilot client."""
        if self._started:
            return
        self._client = CopilotClient(self.config.to_client_options())
        await self._client.start()
        self._started = True

    async def stop(self) -> None:
        """Stop the Copilot client."""
        # Await any pending destroy tasks from new_session() before tearing down
        if self._destroy_tasks:
            await asyncio.gather(*self._destroy_tasks, return_exceptions=True)
            self._destroy_tasks.clear()
        if self._session:
            try:
                await self._session.destroy()
            except Exception:  # Cleanup: best-effort session teardown
                logger.debug("Failed to destroy session during stop", exc_info=True)
            self._session = None
        if self._client:
            await self._client.stop()
            self._client = None
        self._started = False

    async def abort(self) -> None:
        """Abort the currently processing message (best-effort)."""
        try:
            session = await self._ensure_session()
        except Exception:  # Abort is best-effort; ignore session failures
            return
        try:
            await session.abort()
        except Exception:
            # Aborting is best-effort; ignore failures.
            return

    async def __aenter__(self) -> Copex:
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.stop()

    def _should_retry(self, error: str | Exception) -> bool:
        """Check if error should trigger a retry using AdaptiveRetry categorization.

        Uses the AdaptiveRetry error categorization system for consistent
        retry behavior across the codebase (v1.9.0).
        """
        if self.config.retry.retry_on_any_error:
            return True

        # Use AdaptiveRetry's error categorization
        if isinstance(error, Exception):
            category = categorize_error(error)
            # Non-retryable categories
            if category in (ErrorCategory.AUTH, ErrorCategory.CLIENT):
                return False
            # Retryable categories (rate limit, network, server, transient)
            if category in (
                ErrorCategory.RATE_LIMIT,
                ErrorCategory.NETWORK,
                ErrorCategory.SERVER,
                ErrorCategory.TRANSIENT,
            ):
                return True

        # Fallback to pattern matching for string errors
        error_str = str(error).lower()
        return any(pattern.lower() in error_str for pattern in self.config.retry.retry_on_errors)

    def _is_tool_state_error(self, error: str | Exception) -> bool:
        """Detect tool-state mismatch errors that require session recovery."""
        error_str = str(error).lower()
        return "tool_use_id" in error_str and "tool_result" in error_str

    @staticmethod
    def _parse_tool_exclude(value: str) -> _ToolExcludePattern | None:
        trimmed = value.strip()
        if not trimmed:
            return None
        if "(" in trimmed and trimmed.endswith(")"):
            name, arg = trimmed.split("(", 1)
            name = name.strip()
            arg = arg[:-1].strip()
            if name:
                return _ToolExcludePattern(name=name.lower(), arg=arg.lower() or None)
        return _ToolExcludePattern(name=trimmed.lower())

    @staticmethod
    def _tool_name(tool: Any) -> str | None:
        if isinstance(tool, dict):
            name = tool.get("name") or tool.get("tool") or tool.get("id")
            return str(name) if name else None
        name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
        return str(name) if name else None

    @staticmethod
    def _tool_metadata(tool: Any) -> str:
        parts: list[str] = []
        if isinstance(tool, dict):
            for key in ("name", "description", "command", "args", "arguments"):
                value = tool.get(key)
                if isinstance(value, (list, tuple)):
                    parts.append(" ".join(str(item) for item in value))
                elif value is not None:
                    parts.append(str(value))
        else:
            for attr in ("name", "description", "__doc__"):
                value = getattr(tool, attr, None)
                if isinstance(value, str):
                    parts.append(value)
        return " ".join(parts).lower()

    def _filter_tools(self, tools: list[Any] | None) -> list[Any] | None:
        if not tools:
            return tools
        if not self.config.excluded_tools:
            return tools
        patterns = [
            pattern
            for raw in self.config.excluded_tools
            if (pattern := self._parse_tool_exclude(raw)) is not None
        ]
        if not patterns:
            return tools
        filtered: list[Any] = []
        for tool in tools:
            name = self._tool_name(tool)
            if not name:
                filtered.append(tool)
                continue
            name_lower = name.lower()
            metadata: str | None = None
            excluded = False
            for pattern in patterns:
                if name_lower != pattern.name:
                    continue
                if pattern.arg is None:
                    excluded = True
                    break
                if metadata is None:
                    metadata = self._tool_metadata(tool)
                if pattern.arg in metadata:
                    excluded = True
                    break
            if not excluded:
                filtered.append(tool)
        return filtered

    def _calculate_delay(self, attempt: int, error: Exception | None = None) -> float:
        """Calculate delay with exponential backoff and jitter using AdaptiveRetry.

        Uses the AdaptiveRetry BackoffStrategy for consistent delay calculation
        across the codebase. If an error is provided, uses error-category-specific
        strategy for smarter backoff (e.g., longer delays for rate limits).
        """
        # Get error-specific strategy if available
        if error is not None:
            category = categorize_error(error)
            retry = AdaptiveRetry()
            strategy = retry.get_strategy(category)
            return strategy.compute_delay(attempt + 1)  # +1 because BackoffStrategy is 1-indexed

        # Fallback to config-based calculation with jitter
        delay = self.config.retry.base_delay * (self.config.retry.exponential_base**attempt)
        delay = min(delay, self.config.retry.max_delay)
        # Add jitter (±25%)
        jitter = delay * 0.25 * (2 * random.random() - 1)
        return delay + jitter

    def _cb_check(self) -> None:
        """Check circuit breaker state; raise if circuit is open."""
        if self._cb_opened_at is not None:
            elapsed = time.monotonic() - self._cb_opened_at
            if elapsed < _CB_COOLDOWN_SECONDS:
                raise RuntimeError(
                    f"Circuit breaker open: too many consecutive failures. "
                    f"Retry in {_CB_COOLDOWN_SECONDS - elapsed:.0f}s."
                )
            # Cooldown elapsed — half-open: allow one attempt
            self._cb_opened_at = None
            self._cb_failures = 0

    def _cb_record_success(self) -> None:
        """Record a successful request, resetting circuit breaker."""
        self._cb_failures = 0
        self._cb_opened_at = None
        # Also record to model-aware breaker (v1.9.0)
        if self._current_model:
            self.get_model_breaker().record_success(self._current_model)

    def _cb_record_failure(self) -> None:
        """Record a failed request; open circuit if threshold exceeded."""
        self._cb_failures += 1
        if self._cb_failures >= _CB_FAILURE_THRESHOLD:
            self._cb_opened_at = time.monotonic()
            logger.warning(
                "Circuit breaker opened after %d consecutive failures", self._cb_failures
            )
        # Also record to model-aware breaker (v1.9.0)
        if self._current_model:
            self.get_model_breaker().record_failure(self._current_model)

    def _check_post_tool_complete(self, state: _SendState) -> None:
        if (
            state.awaiting_post_tool_response
            and state.tool_execution_seen
            and state.pending_tools == 0
        ):
            state.awaiting_post_tool_response = False

    def _handle_content_delta(
        self,
        state: _SendState,
        delta: str,
        content_type: str,
        on_chunk: Callable[[StreamChunk], None] | None,
    ) -> None:
        if content_type == "message":
            if delta:
                state.received_content = True
                state.content_parts.append(delta)
            self._check_post_tool_complete(state)
        else:
            if delta:
                state.reasoning_parts.append(delta)
        if on_chunk:
            chunk = StreamChunk(type=content_type, delta=delta)
            state.streaming_metrics.record_chunk(chunk)
            on_chunk(chunk)

    def _handle_content_final(
        self,
        state: _SendState,
        content: str,
        content_type: str,
        on_chunk: Callable[[StreamChunk], None] | None,
    ) -> None:
        if content_type == "message":
            state.final_content = content
            if content:
                state.received_content = True
            self._check_post_tool_complete(state)
        else:
            state.final_reasoning = content
        if on_chunk:
            final = state.final_content if content_type == "message" else state.final_reasoning
            on_chunk(StreamChunk(type=content_type, delta="", is_final=True, content=final))

    def _handle_message_delta(
        self,
        event: Any,
        state: _SendState,
        on_chunk: Callable[[StreamChunk], None] | None,
    ) -> None:
        data = event.data
        delta = getattr(data, "delta_content", None) or getattr(data, "transformed_content", None) or ""
        self._handle_content_delta(state, delta, "message", on_chunk)

    def _handle_reasoning_delta(
        self,
        event: Any,
        state: _SendState,
        on_chunk: Callable[[StreamChunk], None] | None,
    ) -> None:
        delta = getattr(event.data, "delta_content", None) or ""
        self._handle_content_delta(state, delta, "reasoning", on_chunk)

    def _handle_message(
        self,
        event: Any,
        state: _SendState,
        on_chunk: Callable[[StreamChunk], None] | None,
    ) -> None:
        content = getattr(event.data, "content", "") or ""
        if not content:
            content = getattr(event.data, "transformed_content", "") or ""
        self._handle_content_final(state, content, "message", on_chunk)

    def _handle_reasoning(
        self,
        event: Any,
        state: _SendState,
        on_chunk: Callable[[StreamChunk], None] | None,
    ) -> None:
        content = getattr(event.data, "content", "") or ""
        self._handle_content_final(state, content, "reasoning", on_chunk)

    def _extract_tool_id(self, data: Any) -> str | None:
        """Extract tool ID from event data using common fields."""
        return (
            getattr(data, "tool_use_id", None)
            or getattr(data, "id", None)
            or getattr(data, "tool_id", None)
            or None
        )

    def _handle_tool_execution_start(
        self,
        event: Any,
        state: _SendState,
        on_chunk: Callable[[StreamChunk], None] | None,
    ) -> None:
        tool_name = getattr(event.data, "tool_name", None) or getattr(event.data, "name", None)
        tool_args = getattr(event.data, "arguments", None)
        tool_id = self._extract_tool_id(event.data)
        state.pending_tools += 1
        state.awaiting_post_tool_response = True
        state.tool_execution_seen = True
        if on_chunk:
            on_chunk(
                StreamChunk(
                    type="tool_call",
                    tool_id=str(tool_id) if tool_id else None,
                    tool_name=str(tool_name) if tool_name else "unknown",
                    tool_args=tool_args if isinstance(tool_args, dict) else {},
                )
            )

    def _handle_tool_execution_partial_result(
        self,
        event: Any,
        state: _SendState,
        on_chunk: Callable[[StreamChunk], None] | None,
    ) -> None:
        tool_name = getattr(event.data, "tool_name", None) or getattr(event.data, "name", None)
        partial = getattr(event.data, "partial_output", None)
        tool_id = self._extract_tool_id(event.data)
        state.awaiting_post_tool_response = True
        state.tool_execution_seen = True
        if on_chunk and partial:
            on_chunk(
                StreamChunk(
                    type="tool_result",
                    tool_id=str(tool_id) if tool_id else None,
                    tool_name=str(tool_name) if tool_name else "unknown",
                    tool_result=str(partial),
                )
            )

    def _handle_tool_execution_complete(
        self,
        event: Any,
        state: _SendState,
        on_chunk: Callable[[StreamChunk], None] | None,
    ) -> None:
        tool_name = getattr(event.data, "tool_name", None) or getattr(event.data, "name", None)
        result_obj = getattr(event.data, "result", None)
        result_text = ""
        if result_obj is not None:
            result_text = getattr(result_obj, "content", "") or str(result_obj)
        success = getattr(event.data, "success", None)
        duration = getattr(event.data, "duration", None)
        tool_id = self._extract_tool_id(event.data)
        state.pending_tools = max(0, state.pending_tools - 1)
        state.awaiting_post_tool_response = True
        state.tool_execution_seen = True
        if on_chunk:
            on_chunk(
                StreamChunk(
                    type="tool_result",
                    tool_id=str(tool_id) if tool_id else None,
                    tool_name=str(tool_name) if tool_name else "unknown",
                    tool_result=result_text,
                    tool_success=success,
                    tool_duration=duration,
                )
            )

    def _handle_error_event(self, event: Any, state: _SendState) -> None:
        error_msg = str(getattr(event.data, "message", event.data))
        state.error_holder.append(RuntimeError(error_msg))
        state.done.set()

    def _handle_tool_call(
        self,
        event: Any,
        state: _SendState,
        on_chunk: Callable[[StreamChunk], None] | None,
    ) -> None:
        data = event.data
        tool_name = getattr(data, "name", None) or getattr(data, "tool", None) or "unknown"
        tool_args = getattr(data, "arguments", None) or getattr(data, "args", {})
        tool_id = self._extract_tool_id(data)
        state.awaiting_post_tool_response = True
        state.tool_execution_seen = True
        if isinstance(tool_args, str):
            import json

            try:
                tool_args = json.loads(tool_args)
            except (json.JSONDecodeError, ValueError):
                tool_args = {"raw": tool_args}
        if on_chunk:
            on_chunk(
                StreamChunk(
                    type="tool_call",
                    tool_id=str(tool_id) if tool_id else None,
                    tool_name=str(tool_name),
                    tool_args=tool_args if isinstance(tool_args, dict) else {},
                )
            )

    def _handle_assistant_turn_end(self, state: _SendState) -> None:
        if not state.awaiting_post_tool_response:
            state.done.set()

    def _handle_session_idle(self, state: _SendState) -> None:
        state.done.set()

    async def _ensure_session(self) -> Any:
        """Ensure a session exists, creating one if needed."""
        from copex.models import model_supports_reasoning
        if not self._started:
            await self.start()
        if self._session is None:
            # github-copilot-sdk >= 0.1.21 supports reasoning_effort natively
            # Use current model (may be fallback) if set (v1.9.0)
            session_options = self.config.to_session_options()
            if self._current_model and self._current_model != self.config.model.value:
                session_options["model"] = self._current_model
                # If the fallback model doesn't support reasoning, drop it
                if not model_supports_reasoning(self._current_model):
                    session_options.pop("reasoning_effort", None)
            self._session = await self._client.create_session(session_options)
        return self._session

    async def _get_session_context(self, session: Any) -> str | None:
        """Extract conversation context from session for recovery."""
        try:
            messages = await session.get_messages()
            if not messages:
                return None

            # Build a summary of the conversation
            context_parts = []
            for msg in messages:
                msg_type = getattr(msg, "type", None)
                msg_value = msg_type.value if hasattr(msg_type, "value") else str(msg_type)
                data = getattr(msg, "data", None)

                if msg_value == EventType.USER_MESSAGE.value:
                    content = getattr(data, "content", "") or getattr(data, "prompt", "")
                    if content:
                        context_parts.append(f"User: {content[:500]}")
                elif msg_value == EventType.ASSISTANT_MESSAGE.value:
                    content = getattr(data, "content", "") or ""
                    if content:
                        # Truncate long responses
                        truncated = content[:1000] + "..." if len(content) > 1000 else content
                        context_parts.append(f"Assistant: {truncated}")

            if not context_parts:
                return None

            return "\n\n".join(context_parts[-10:])  # Last 10 messages max
        except Exception:
            logger.debug("Failed to extract session context", exc_info=True)
            return None

    async def _recover_session(
        self, on_chunk: Callable[[StreamChunk], None] | None
    ) -> tuple[Any, str]:
        """Destroy bad session and create new one, preserving context."""
        context = None
        if self._session:
            context = await self._get_session_context(self._session)
            try:
                await self._session.destroy()
            except Exception:  # Cleanup: must not fail recovery
                logger.debug("Failed to destroy session during recovery", exc_info=True)
            self._session = None

        # Create fresh session
        try:
            session = await self._ensure_session()
        except Exception:  # Propagated: session creation is critical
            logger.error("Failed to create fresh session during recovery", exc_info=True)
            raise

        # Build recovery prompt with context
        if context:
            recovery_prompt = (
                f"[Session recovered. Previous conversation context:]\n\n"
                f"{context}\n\n"
                f"[End of context. {self.config.continue_prompt}]"
            )
        else:
            recovery_prompt = self.config.continue_prompt

        if on_chunk:
            on_chunk(
                StreamChunk(
                    type="system",
                    delta="\n[Session recovered with fresh connection]\n",
                )
            )

        return session, recovery_prompt

    async def send(
        self,
        prompt: str,
        *,
        tools: list[Any] | None = None,
        on_chunk: Callable[[StreamChunk], None] | None = None,
        metrics: MetricsCollector | None = None,
    ) -> Response:
        """
        Send a prompt with automatic retry on errors.

        Args:
            prompt: The prompt to send
            tools: Optional list of tools to make available
            on_chunk: Optional callback for streaming chunks

        Returns:
            Response object with content and metadata
        """
        # Check model fallback (v1.9.0)
        model_breaker = self.get_model_breaker()
        original_model = self.config.model.value
        fallback_chain = self._fallback_chain or DEFAULT_FALLBACK_CHAINS.get(original_model)

        available_model = model_breaker.get_available_model(original_model, fallback_chain)
        if available_model is None:
            raise RuntimeError(
                f"All models in fallback chain are unavailable. "
                f"Primary: {original_model}, Fallback: {fallback_chain}"
            )

        # Switch to fallback model if needed
        if available_model != original_model:
            self._current_model = available_model
            if on_chunk:
                on_chunk(
                    StreamChunk(
                        type="system",
                        delta=f"\n[Model {original_model} unavailable, using fallback {available_model}]\n",
                    )
                )
            # Create new session with fallback model
            if self._session:
                try:
                    await self._session.destroy()
                except Exception:  # Cleanup: best-effort session teardown
                    logger.debug("Failed to destroy session for model fallback", exc_info=True)
                self._session = None
        else:
            self._current_model = original_model

        session = await self._ensure_session()
        filtered_tools = self._filter_tools(tools)
        retries = 0
        auto_continues = 0
        last_error: Exception | None = None
        collector = metrics or get_collector()
        request = collector.start_request(
            model=self._current_model or self.config.model.value,
            reasoning_effort=self.config.reasoning_effort.value,
            prompt=prompt,
        )

        # Circuit breaker gate (legacy per-instance check)
        self._cb_check()

        while True:
            try:
                result = await self._send_once(session, prompt, filtered_tools, on_chunk)
                result.retries = retries
                result.auto_continues = auto_continues
                tokens = None
                if result.prompt_tokens is not None or result.completion_tokens is not None:
                    tokens = {
                        "prompt": int(result.prompt_tokens or 0),
                        "completion": int(result.completion_tokens or 0),
                    }

                collector.complete_request(
                    request.request_id,
                    success=True,
                    response=result.content,
                    retries=retries,
                    tokens=tokens,
                )
                self._cb_record_success()
                return result

            except Exception as e:  # Catch-all: retry logic must handle any SDK error
                last_error = e
                error_str = str(e)

                if self._is_tool_state_error(e) and self.config.auto_continue:
                    auto_continues += 1
                    if auto_continues > self.config.retry.max_auto_continues:
                        collector.complete_request(
                            request.request_id,
                            success=False,
                            error=str(last_error),
                            retries=retries,
                        )
                        self._cb_record_failure()
                        raise last_error from e
                    retries = 0
                    session, prompt = await self._recover_session(on_chunk)
                    if on_chunk:
                        on_chunk(
                            StreamChunk(
                                type="system",
                                delta="\n[Tool state mismatch detected; recovered session]\n",
                            )
                        )
                    delay = self._calculate_delay(0, error=e)
                    await asyncio.sleep(delay)
                    continue

                if not self._should_retry(e):
                    collector.complete_request(
                        request.request_id,
                        success=False,
                        error=error_str,
                        retries=retries,
                    )
                    self._cb_record_failure()
                    raise

                retries += 1
                if retries <= self.config.retry.max_retries:
                    # Normal retry with exponential backoff (same session)
                    # Use AdaptiveRetry's error-aware delay calculation
                    delay = self._calculate_delay(retries - 1, error=e)
                    if on_chunk:
                        on_chunk(
                            StreamChunk(
                                type="system",
                                delta=f"\n[Retry {retries}/{self.config.retry.max_retries} after error: {error_str[:50]}...]\n",
                            )
                        )
                    await asyncio.sleep(delay)
                elif (
                    self.config.auto_continue
                    and auto_continues < self.config.retry.max_auto_continues
                ):
                    # Retries exhausted - session may be in bad state
                    # Recover with fresh session, preserving context
                    auto_continues += 1
                    retries = 0
                    session, prompt = await self._recover_session(on_chunk)
                    delay = self._calculate_delay(0, error=e)
                    if on_chunk:
                        on_chunk(
                            StreamChunk(
                                type="system",
                                delta=f"\n[Auto-continue #{auto_continues}/{self.config.retry.max_auto_continues} with fresh session]\n",
                            )
                        )
                    await asyncio.sleep(delay)
                else:
                    collector.complete_request(
                        request.request_id,
                        success=False,
                        error=str(last_error) if last_error else "Max retries exceeded",
                        retries=retries,
                    )
                    self._cb_record_failure()
                    raise (last_error or RuntimeError("Max retries exceeded")) from e

    async def _send_once(
        self,
        session: Any,
        prompt: str,
        tools: list[Any] | None,
        on_chunk: Callable[[StreamChunk], None] | None,
    ) -> Response:
        """Send a single prompt and collect the response."""
        state = _SendState(done=asyncio.Event())
        loop = asyncio.get_running_loop()
        state.last_activity = loop.time()
        state.streaming_metrics._start_time = time.monotonic()

        def _handle_usage(event: Any, st: _SendState, _oc: Any) -> None:
            data = event.data
            inp = getattr(data, "input_tokens", None)
            out = getattr(data, "output_tokens", None)
            cost = getattr(data, "cost", None)
            server_model = getattr(data, "model", None)
            if inp is not None:
                try:
                    st.prompt_tokens = int(inp)
                except (TypeError, ValueError):
                    pass
            if out is not None:
                try:
                    st.completion_tokens = int(out)
                except (TypeError, ValueError):
                    pass
            if cost is not None:
                try:
                    st.cost = float(cost)
                except (TypeError, ValueError):
                    pass
            if server_model is not None:
                st.server_model = str(server_model)

        def _handle_turn_end(_e: Any, st: _SendState, _oc: Any) -> None:
            self._handle_assistant_turn_end(st)

        def _handle_idle(_e: Any, st: _SendState, _oc: Any) -> None:
            self._handle_session_idle(st)

        def _handle_error(ev: Any, st: _SendState, _oc: Any) -> None:
            self._handle_error_event(ev, st)

        # O(1) dispatch table — avoids the if/elif chain on the hot path.
        dispatch: dict[str, Callable[..., None]] = {
            _ET_MSG_DELTA: self._handle_message_delta,
            _ET_REASON_DELTA: self._handle_reasoning_delta,
            _ET_MSG: self._handle_message,
            _ET_REASON: self._handle_reasoning,
            _ET_TOOL_START: self._handle_tool_execution_start,
            _ET_TOOL_PARTIAL: self._handle_tool_execution_partial_result,
            _ET_TOOL_COMPLETE: self._handle_tool_execution_complete,
            _ET_ERROR: _handle_error,
            _ET_SESSION_ERROR: _handle_error,
            _ET_TOOL_CALL: self._handle_tool_call,
            _ET_USAGE: _handle_usage,
            _ET_TURN_END: _handle_turn_end,
            _ET_SESSION_IDLE: _handle_idle,
        }

        raw_events = state.raw_events
        raw_events_len = 0

        def on_event(event: Any) -> None:
            nonlocal raw_events_len
            state.last_activity = loop.time()
            try:
                etype = event.type
                event_type = etype.value if hasattr(etype, "value") else str(etype)

                if raw_events_len < MAX_RAW_EVENTS:
                    raw_events.append({"type": event_type, "data": getattr(event, "data", None)})
                    raw_events_len += 1
                elif raw_events_len == MAX_RAW_EVENTS:
                    warnings.warn(
                        f"raw_events limit ({MAX_RAW_EVENTS}) reached. "
                        "Additional events will not be captured. "
                        "Consider streaming or processing events incrementally.",
                        ResourceWarning,
                        stacklevel=2,
                    )
                    raw_events.append({"type": "_limit_warning", "data": None})
                    raw_events_len += 1

                handler = dispatch.get(event_type)
                if handler is not None:
                    handler(event, state, on_chunk)

            except Exception as e:  # Catch-all: on_event must not crash the event loop
                logger.warning("Unhandled exception in on_event callback: %s", e, exc_info=True)
                state.error_holder.append(e)
                state.done.set()

        unsubscribe = session.on(on_event)

        try:
            payload: dict[str, Any] = {"prompt": prompt}
            if tools is not None:
                payload["tools"] = tools
            await session.send(payload)
            # Activity-based timeout: only timeout if no events received for timeout period
            while not state.done.is_set():
                try:
                    await asyncio.wait_for(state.done.wait(), timeout=self.config.timeout)
                except asyncio.TimeoutError:
                    # Check if we've had activity within the timeout window
                    idle_time = loop.time() - state.last_activity
                    if idle_time >= self.config.timeout:
                        raise TimeoutError(
                            f"Response timed out after {idle_time:.1f}s of inactivity"
                        ) from None
                    # Had recent activity, keep waiting
        finally:
            # Remove event handler to avoid duplicates
            try:
                unsubscribe()
            except Exception:  # Cleanup: unsubscribe is best-effort
                logger.debug("Failed to unsubscribe event handler", exc_info=True)

        # If we never got explicit content events, try to extract from history.
        # This also covers streaming mode (on_chunk provided) where events may be lost.
        if not state.received_content:
            try:
                messages = await session.get_messages()
                for message in reversed(messages):
                    message_type = getattr(message, "type", None)
                    message_value = (
                        message_type.value if hasattr(message_type, "value") else str(message_type)
                    )
                    if message_value == _ET_MSG:
                        state.final_content = (
                            getattr(message.data, "content", "") or state.final_content
                        )
                        if state.final_content:
                            break
            except Exception:  # Cleanup: fallback extraction is best-effort
                logger.debug("Failed to extract messages for history fallback", exc_info=True)

        if state.error_holder:
            raise state.error_holder[0]

        return Response(
            content=state.final_content or "".join(state.content_parts),
            reasoning=state.final_reasoning
            or ("".join(state.reasoning_parts) if state.reasoning_parts else None),
            raw_events=raw_events,
            prompt_tokens=state.prompt_tokens,
            completion_tokens=state.completion_tokens,
            cost=state.cost,
            server_model=state.server_model,
            streaming_metrics=state.streaming_metrics,
        )

    async def stream(
        self,
        prompt: str,
        *,
        tools: list[Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a response with automatic retry.

        Yields StreamChunk objects as they arrive.
        """
        queue: asyncio.Queue[StreamChunk | None | BaseException] = asyncio.Queue()

        def on_chunk(chunk: StreamChunk) -> None:
            queue.put_nowait(chunk)

        async def sender() -> None:
            try:
                await self.send(prompt, tools=tools, on_chunk=on_chunk)
                queue.put_nowait(None)  # Signal completion
            except BaseException as e:
                queue.put_nowait(e)

        task = asyncio.create_task(sender())

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def chat(self, prompt: str) -> str:
        """Simple interface - send prompt, get response content."""
        response = await self.send(prompt)
        return response.content

    def new_session(self) -> None:
        """Start a fresh session (clears conversation history)."""
        if self._session:
            session = self._session
            self._session = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                try:
                    asyncio.run(session.destroy())
                except Exception:  # Cleanup: sync destroy is best-effort
                    logger.debug("Failed to destroy session in new_session (sync)", exc_info=True)
            else:

                async def _destroy_with_logging() -> None:
                    try:
                        await session.destroy()
                    except Exception as e:
                        logger.debug("Failed to destroy session in new_session: %s", e)

                task = loop.create_task(_destroy_with_logging())
                self._destroy_tasks.add(task)
                task.add_done_callback(self._destroy_tasks.discard)


@asynccontextmanager
async def copex(
    model: Model | str = Model.GPT_5_2_CODEX,
    reasoning: ReasoningEffort | str = ReasoningEffort.XHIGH,
    **kwargs: Any,
) -> AsyncIterator[Copex]:
    """
    Context manager for quick Copex access.

    Example:
        async with copex() as c:
            response = await c.chat("Hello!")
            print(response)
    """
    config = CopexConfig(
        model=Model(model) if isinstance(model, str) else model,
        reasoning_effort=parse_reasoning_effort(reasoning)
        if isinstance(reasoning, str)
        else reasoning,
        **kwargs,
    )
    client = Copex(config)
    try:
        await client.start()
        yield client
    finally:
        await client.stop()
