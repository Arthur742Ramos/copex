"""Core Copex client with retry logic and stuck detection."""

from __future__ import annotations

import asyncio
import random
import re
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from copilot import CopilotClient

from copex.config import CopexConfig
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

    type: str  # "message" or "reasoning"
    delta: str
    is_final: bool = False
    content: str | None = None  # Full content when is_final=True


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

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        delay = self.config.retry.base_delay * (self.config.retry.exponential_base ** attempt)
        delay = min(delay, self.config.retry.max_delay)
        # Add jitter (±25%)
        jitter = delay * 0.25 * (2 * random.random() - 1)
        return delay + jitter

    async def _ensure_session(self) -> Any:
        """Ensure a session exists, creating one if needed."""
        if not self._started:
            await self.start()
        if self._session is None:
            self._session = await self._client.create_session(self.config.to_session_options())
        return self._session

    async def send(
        self,
        prompt: str,
        *,
        tools: list[Any] | None = None,
        on_chunk: Callable[[StreamChunk], None] | None = None,
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

        while retries <= self.config.retry.max_retries:
            try:
                result = await self._send_once(session, prompt, tools, on_chunk)
                result.retries = retries
                result.auto_continues = auto_continues
                return result

            except Exception as e:
                last_error = e
                error_str = str(e)

                if self._should_retry(e):
                    retries += 1
                    if retries <= self.config.retry.max_retries:
                        delay = self._calculate_delay(retries - 1)
                        if on_chunk:
                            on_chunk(StreamChunk(
                                type="system",
                                delta=f"\n[Retry {retries}/{self.config.retry.max_retries} after error: {error_str[:50]}...]\n",
                            ))
                        await asyncio.sleep(delay)

                        # Try auto-continue if enabled
                        if self.config.auto_continue:
                            auto_continues += 1
                            prompt = self.config.continue_prompt
                        continue
                raise

        raise last_error or RuntimeError("Max retries exceeded")

    async def _send_once(
        self,
        session: Any,
        prompt: str,
        tools: list[Any] | None,
        on_chunk: Callable[[StreamChunk], None] | None,
    ) -> Response:
        """Send a single prompt and collect the response."""
        done = asyncio.Event()
        error_holder: list[Exception] = []
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        final_content: str | None = None
        final_reasoning: str | None = None
        raw_events: list[dict[str, Any]] = []

        def on_event(event: Any) -> None:
            try:
                event_type = event.type.value if hasattr(event.type, "value") else str(event.type)
                raw_events.append({"type": event_type, "data": getattr(event, "data", None)})

                if event_type == EventType.ASSISTANT_MESSAGE_DELTA.value:
                    delta = getattr(event.data, "delta_content", "") or ""
                    content_parts.append(delta)
                    if on_chunk:
                        on_chunk(StreamChunk(type="message", delta=delta))

                elif event_type == EventType.ASSISTANT_REASONING_DELTA.value:
                    delta = getattr(event.data, "delta_content", "") or ""
                    reasoning_parts.append(delta)
                    if on_chunk:
                        on_chunk(StreamChunk(type="reasoning", delta=delta))

                elif event_type == EventType.ASSISTANT_MESSAGE.value:
                    nonlocal final_content
                    final_content = getattr(event.data, "content", "")
                    if on_chunk:
                        on_chunk(StreamChunk(
                            type="message",
                            delta="",
                            is_final=True,
                            content=final_content,
                        ))

                elif event_type == EventType.ASSISTANT_REASONING.value:
                    nonlocal final_reasoning
                    final_reasoning = getattr(event.data, "content", "")
                    if on_chunk:
                        on_chunk(StreamChunk(
                            type="reasoning",
                            delta="",
                            is_final=True,
                            content=final_reasoning,
                        ))

                elif event_type == EventType.ERROR.value:
                    error_msg = str(getattr(event.data, "message", event.data))
                    error_holder.append(RuntimeError(error_msg))

                elif event_type == EventType.SESSION_IDLE.value:
                    done.set()

            except Exception as e:
                error_holder.append(e)
                done.set()

        session.on(on_event)

        try:
            await session.send({"prompt": prompt})
            await asyncio.wait_for(done.wait(), timeout=self.config.timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Response timed out after {self.config.timeout}s")
        finally:
            # Remove event handler to avoid duplicates
            try:
                session.off(on_event)
            except Exception:
                pass

        if error_holder:
            raise error_holder[0]

        return Response(
            content=final_content or "".join(content_parts),
            reasoning=final_reasoning or ("".join(reasoning_parts) if reasoning_parts else None),
            raw_events=raw_events,
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
            asyncio.create_task(self._session.destroy())
            self._session = None


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
