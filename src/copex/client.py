"""Core Copex client with retry logic and stuck detection."""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from copilot import CopilotClient
from copilot.session import CopilotSession

from copex.config import CopexConfig
from copex.metrics import MetricsCollector, get_collector
from copex.models import EventType, Model, ReasoningEffort


@dataclass
class Response:
    """Response from a Copilot prompt."""

    content: str
    reasoning: str | None = None
    raw_events: list[dict[str, Any]] = field(default_factory=list)
    retries: int = 0
    auto_continues: int = 0


@dataclass
class StreamChunk:
    """A streaming chunk from Copilot."""

    type: str  # "message", "reasoning", "tool_call", "tool_result", "system"
    delta: str = ""
    is_final: bool = False
    content: str | None = None  # Full content when is_final=True
    # Tool call info
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: str | None = None
    tool_success: bool | None = None
    tool_duration: float | None = None


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


class Copex:
    """Copilot Extended - Resilient wrapper with automatic retry and stuck detection."""

    def __init__(self, config: CopexConfig | None = None):
        self.config = config or CopexConfig()
        self._client: CopilotClient | None = None
        self._session: Any = None
        self._started = False

    async def start(self) -> None:
        """Start the Copilot client."""
        if self._started:
            return
        self._client = CopilotClient(self.config.to_client_options())
        await self._client.start()
        self._started = True

    async def stop(self) -> None:
        """Stop the Copilot client."""
        if self._session:
            try:
                await self._session.destroy()
            except Exception:
                pass
            self._session = None
        if self._client:
            await self._client.stop()
            self._client = None
        self._started = False

    async def __aenter__(self) -> "Copex":
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.stop()

    def _should_retry(self, error: str | Exception) -> bool:
        """Check if error should trigger a retry."""
        if self.config.retry.retry_on_any_error:
            return True
        error_str = str(error).lower()
        return any(
            pattern.lower() in error_str for pattern in self.config.retry.retry_on_errors
        )

    def _is_tool_state_error(self, error: str | Exception) -> bool:
        """Detect tool-state mismatch errors that require session recovery."""
        error_str = str(error).lower()
        return "tool_use_id" in error_str and "tool_result" in error_str

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        delay = self.config.retry.base_delay * (self.config.retry.exponential_base ** attempt)
        delay = min(delay, self.config.retry.max_delay)
        # Add jitter (±25%)
        jitter = delay * 0.25 * (2 * random.random() - 1)
        return delay + jitter

    def _handle_message_delta(
        self,
        event: Any,
        state: _SendState,
        on_chunk: Callable[[StreamChunk], None] | None,
    ) -> None:
        delta = getattr(event.data, "delta_content", "") or ""
        if not delta:
            delta = getattr(event.data, "transformed_content", "") or ""
        if delta:
            state.received_content = True
        state.content_parts.append(delta)
        if state.awaiting_post_tool_response and state.tool_execution_seen and state.pending_tools == 0:
            state.awaiting_post_tool_response = False
        if on_chunk:
            on_chunk(StreamChunk(type="message", delta=delta))

    def _handle_reasoning_delta(
        self,
        event: Any,
        state: _SendState,
        on_chunk: Callable[[StreamChunk], None] | None,
    ) -> None:
        delta = getattr(event.data, "delta_content", "") or ""
        state.reasoning_parts.append(delta)
        if on_chunk:
            on_chunk(StreamChunk(type="reasoning", delta=delta))

    def _handle_message(
        self,
        event: Any,
        state: _SendState,
        on_chunk: Callable[[StreamChunk], None] | None,
    ) -> None:
        content = getattr(event.data, "content", "") or ""
        if not content:
            content = getattr(event.data, "transformed_content", "") or ""
        state.final_content = content
        if content:
            state.received_content = True
        if state.awaiting_post_tool_response and state.tool_execution_seen and state.pending_tools == 0:
            state.awaiting_post_tool_response = False
        if on_chunk:
            on_chunk(StreamChunk(
                type="message",
                delta="",
                is_final=True,
                content=state.final_content,
            ))

    def _handle_reasoning(
        self,
        event: Any,
        state: _SendState,
        on_chunk: Callable[[StreamChunk], None] | None,
    ) -> None:
        state.final_reasoning = getattr(event.data, "content", "") or ""
        if on_chunk:
            on_chunk(StreamChunk(
                type="reasoning",
                delta="",
                is_final=True,
                content=state.final_reasoning,
            ))

    def _handle_tool_execution_start(
        self,
        event: Any,
        state: _SendState,
        on_chunk: Callable[[StreamChunk], None] | None,
    ) -> None:
        tool_name = getattr(event.data, "tool_name", None) or getattr(event.data, "name", None)
        tool_args = getattr(event.data, "arguments", None)
        state.pending_tools += 1
        state.awaiting_post_tool_response = True
        state.tool_execution_seen = True
        if on_chunk:
            on_chunk(StreamChunk(
                type="tool_call",
                tool_name=str(tool_name) if tool_name else "unknown",
                tool_args=tool_args if isinstance(tool_args, dict) else {},
            ))

    def _handle_tool_execution_partial_result(
        self,
        event: Any,
        state: _SendState,
        on_chunk: Callable[[StreamChunk], None] | None,
    ) -> None:
        tool_name = getattr(event.data, "tool_name", None) or getattr(event.data, "name", None)
        partial = getattr(event.data, "partial_output", None)
        state.awaiting_post_tool_response = True
        state.tool_execution_seen = True
        if on_chunk and partial:
            on_chunk(StreamChunk(
                type="tool_result",
                tool_name=str(tool_name) if tool_name else "unknown",
                tool_result=str(partial),
            ))

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
        state.pending_tools = max(0, state.pending_tools - 1)
        state.awaiting_post_tool_response = True
        state.tool_execution_seen = True
        if on_chunk:
            on_chunk(StreamChunk(
                type="tool_result",
                tool_name=str(tool_name) if tool_name else "unknown",
                tool_result=result_text,
                tool_success=success,
                tool_duration=duration,
            ))

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
        state.awaiting_post_tool_response = True
        if isinstance(tool_args, str):
            import json
            try:
                tool_args = json.loads(tool_args)
            except Exception:
                tool_args = {"raw": tool_args}
        if on_chunk:
            on_chunk(StreamChunk(
                type="tool_call",
                tool_name=str(tool_name),
                tool_args=tool_args if isinstance(tool_args, dict) else {},
            ))

    def _handle_assistant_turn_end(self, state: _SendState) -> None:
        if not state.awaiting_post_tool_response:
            state.done.set()

    def _handle_session_idle(self, state: _SendState) -> None:
        state.done.set()

    async def _ensure_session(self) -> Any:
        """Ensure a session exists, creating one if needed."""
        if not self._started:
            await self.start()
        if self._session is None:
            self._session = await self._create_session_with_reasoning()
        return self._session

    async def _create_session_with_reasoning(self) -> CopilotSession:
        """Create a session with reasoning effort support.
        
        The GitHub Copilot SDK's create_session() ignores model_reasoning_effort,
        so we bypass it and call the JSON-RPC directly to inject this parameter.
        
        Falls back to SDK's create_session() in test environments where the
        internal JSON-RPC client isn't accessible.
        """
        opts = self.config.to_session_options()
        
        # Check if we can access the internal JSON-RPC client
        # If not (e.g., in tests with mocked clients), fall back to SDK's create_session
        if not hasattr(self._client, '_client') or self._client._client is None:
            return await self._client.create_session(opts)
        
        # Build the wire payload with proper camelCase keys
        payload: dict[str, Any] = {}
        
        if opts.get("model"):
            payload["model"] = opts["model"]
        if opts.get("streaming") is not None:
            payload["streaming"] = opts["streaming"]
        
        # The key fix: inject modelReasoningEffort directly into the wire payload
        # The SDK's create_session() drops this, but the server accepts it!
        reasoning_effort = opts.get("model_reasoning_effort")
        if reasoning_effort and reasoning_effort != "none":
            payload["modelReasoningEffort"] = reasoning_effort
        
        # Map other session options
        if opts.get("system_message"):
            payload["systemMessage"] = opts["system_message"]
        if opts.get("available_tools"):
            payload["availableTools"] = opts["available_tools"]
        if opts.get("excluded_tools"):
            payload["excludedTools"] = opts["excluded_tools"]
        if opts.get("working_directory"):
            payload["workingDirectory"] = opts["working_directory"]
        if opts.get("mcp_servers"):
            payload["mcpServers"] = opts["mcp_servers"]
        if opts.get("skill_directories"):
            payload["skillDirectories"] = opts["skill_directories"]
        if opts.get("disabled_skills"):
            payload["disabledSkills"] = opts["disabled_skills"]
        if opts.get("instructions"):
            # Instructions go into system message
            if "systemMessage" not in payload:
                payload["systemMessage"] = {"mode": "append", "content": opts["instructions"]}
            elif isinstance(payload["systemMessage"], dict):
                existing = payload["systemMessage"].get("content", "")
                payload["systemMessage"]["content"] = f"{existing}\n\n{opts['instructions']}" if existing else opts["instructions"]
        
        # Call the JSON-RPC directly, bypassing the SDK's create_session
        response = await self._client._client.request("session.create", payload)
        
        session_id = response["sessionId"]
        workspace_path = response.get("workspacePath")
        
        # Create a CopilotSession using the SDK's class
        session = CopilotSession(session_id, self._client._client, workspace_path)
        
        # Register the session with the client for event dispatch
        # Note: we access the internal _sessions dict since we bypassed create_session
        with self._client._sessions_lock:
            self._client._sessions[session_id] = session
        
        return session

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
            return None

    async def _recover_session(self, on_chunk: Callable[[StreamChunk], None] | None) -> tuple[Any, str]:
        """Destroy bad session and create new one, preserving context."""
        context = None
        if self._session:
            context = await self._get_session_context(self._session)
            try:
                await self._session.destroy()
            except Exception:
                pass
            self._session = None

        # Create fresh session
        session = await self._ensure_session()

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
            on_chunk(StreamChunk(
                type="system",
                delta="\n[Session recovered with fresh connection]\n",
            ))

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
        session = await self._ensure_session()
        retries = 0
        auto_continues = 0
        last_error: Exception | None = None
        collector = metrics or get_collector()
        request = collector.start_request(
            model=self.config.model.value,
            reasoning_effort=self.config.reasoning_effort.value,
            prompt=prompt,
        )

        while True:
            try:
                result = await self._send_once(session, prompt, tools, on_chunk)
                result.retries = retries
                result.auto_continues = auto_continues
                collector.complete_request(
                    request.request_id,
                    success=True,
                    response=result.content,
                    retries=retries,
                )
                return result

            except Exception as e:
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
                        raise last_error
                    retries = 0
                    session, prompt = await self._recover_session(on_chunk)
                    if on_chunk:
                        on_chunk(StreamChunk(
                            type="system",
                            delta="\n[Tool state mismatch detected; recovered session]\n",
                        ))
                    delay = self._calculate_delay(0)
                    await asyncio.sleep(delay)
                    continue

                if not self._should_retry(e):
                    collector.complete_request(
                        request.request_id,
                        success=False,
                        error=error_str,
                        retries=retries,
                    )
                    raise

                retries += 1
                if retries <= self.config.retry.max_retries:
                    # Normal retry with exponential backoff (same session)
                    delay = self._calculate_delay(retries - 1)
                    if on_chunk:
                        on_chunk(StreamChunk(
                            type="system",
                            delta=f"\n[Retry {retries}/{self.config.retry.max_retries} after error: {error_str[:50]}...]\n",
                        ))
                    await asyncio.sleep(delay)
                elif self.config.auto_continue and auto_continues < self.config.retry.max_auto_continues:
                    # Retries exhausted - session may be in bad state
                    # Recover with fresh session, preserving context
                    auto_continues += 1
                    retries = 0
                    session, prompt = await self._recover_session(on_chunk)
                    delay = self._calculate_delay(0)
                    if on_chunk:
                        on_chunk(StreamChunk(
                            type="system",
                            delta=f"\n[Auto-continue #{auto_continues}/{self.config.retry.max_auto_continues} with fresh session]\n",
                        ))
                    await asyncio.sleep(delay)
                else:
                    collector.complete_request(
                        request.request_id,
                        success=False,
                        error=str(last_error) if last_error else "Max retries exceeded",
                        retries=retries,
                    )
                    raise last_error or RuntimeError("Max retries exceeded")

    async def _send_once(
        self,
        session: Any,
        prompt: str,
        tools: list[Any] | None,
        on_chunk: Callable[[StreamChunk], None] | None,
    ) -> Response:
        """Send a single prompt and collect the response."""
        state = _SendState(done=asyncio.Event())
        state.last_activity = asyncio.get_running_loop().time()

        def on_event(event: Any) -> None:
            state.last_activity = asyncio.get_running_loop().time()
            try:
                event_type = event.type.value if hasattr(event.type, "value") else str(event.type)
                state.raw_events.append({"type": event_type, "data": getattr(event, "data", None)})

                if event_type == EventType.ASSISTANT_MESSAGE_DELTA.value:
                    self._handle_message_delta(event, state, on_chunk)

                elif event_type == EventType.ASSISTANT_REASONING_DELTA.value:
                    self._handle_reasoning_delta(event, state, on_chunk)

                elif event_type == EventType.ASSISTANT_MESSAGE.value:
                    self._handle_message(event, state, on_chunk)

                elif event_type == EventType.ASSISTANT_REASONING.value:
                    self._handle_reasoning(event, state, on_chunk)

                elif event_type == EventType.TOOL_EXECUTION_START.value:
                    self._handle_tool_execution_start(event, state, on_chunk)

                elif event_type == EventType.TOOL_EXECUTION_PARTIAL_RESULT.value:
                    self._handle_tool_execution_partial_result(event, state, on_chunk)

                elif event_type == EventType.TOOL_EXECUTION_COMPLETE.value:
                    self._handle_tool_execution_complete(event, state, on_chunk)

                elif event_type == EventType.ERROR.value:
                    self._handle_error_event(event, state)

                elif event_type == EventType.SESSION_ERROR.value:
                    self._handle_error_event(event, state)

                elif event_type == EventType.TOOL_CALL.value:
                    self._handle_tool_call(event, state, on_chunk)

                elif event_type == EventType.ASSISTANT_TURN_END.value:
                    self._handle_assistant_turn_end(state)

                elif event_type == EventType.SESSION_IDLE.value:
                    self._handle_session_idle(state)

            except Exception as e:
                state.error_holder.append(e)
                state.done.set()

        unsubscribe = session.on(on_event)

        try:
            await session.send({"prompt": prompt})
            # Activity-based timeout: only timeout if no events received for timeout period
            while not state.done.is_set():
                try:
                    await asyncio.wait_for(state.done.wait(), timeout=self.config.timeout)
                except asyncio.TimeoutError:
                    # Check if we've had activity within the timeout window
                    idle_time = asyncio.get_running_loop().time() - state.last_activity
                    if idle_time >= self.config.timeout:
                        raise TimeoutError(
                            f"Response timed out after {idle_time:.1f}s of inactivity"
                        )
                    # Had recent activity, keep waiting
        finally:
            # Remove event handler to avoid duplicates
            try:
                unsubscribe()
            except Exception:
                pass

        # If we never got explicit content events and NOT streaming, try to extract from history
        # When streaming (on_chunk provided), we trust the streamed chunks and don't use history
        # fallback which could return stale content from previous turns
        if not state.received_content and on_chunk is None:
            try:
                messages = await session.get_messages()
                for message in reversed(messages):
                    message_type = getattr(message, "type", None)
                    message_value = (
                        message_type.value if hasattr(message_type, "value") else str(message_type)
                    )
                    if message_value == EventType.ASSISTANT_MESSAGE.value:
                        state.final_content = getattr(message.data, "content", "") or state.final_content
                        if state.final_content:
                            break
            except Exception:
                pass

        if state.error_holder:
            raise state.error_holder[0]

        return Response(
            content=state.final_content or "".join(state.content_parts),
            reasoning=state.final_reasoning or (
                "".join(state.reasoning_parts) if state.reasoning_parts else None
            ),
            raw_events=state.raw_events,
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
        queue: asyncio.Queue[StreamChunk | None | Exception] = asyncio.Queue()

        def on_chunk(chunk: StreamChunk) -> None:
            queue.put_nowait(chunk)

        async def sender() -> None:
            try:
                await self.send(prompt, tools=tools, on_chunk=on_chunk)
                queue.put_nowait(None)  # Signal completion
            except Exception as e:
                queue.put_nowait(e)

        task = asyncio.create_task(sender())

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
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
                asyncio.run(session.destroy())
            else:
                loop.create_task(session.destroy())


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
        reasoning_effort=ReasoningEffort(reasoning) if isinstance(reasoning, str) else reasoning,
        **kwargs,
    )
    client = Copex(config)
    try:
        await client.start()
        yield client
    finally:
        await client.stop()
