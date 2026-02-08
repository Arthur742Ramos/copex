"""Streaming data types and utilities for Copex."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StreamChunk:
    """A streaming chunk from Copilot."""

    type: str  # "message", "reasoning", "tool_call", "tool_result", "system"
    delta: str = ""
    is_final: bool = False
    content: str | None = None  # Full content when is_final=True
    # Tool call info
    tool_id: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: str | None = None
    tool_success: bool | None = None
    tool_duration: float | None = None


@dataclass
class StreamingMetrics:
    """Lightweight metrics captured during a single streaming response."""

    first_chunk_time: float | None = None
    last_chunk_time: float | None = None
    total_chunks: int = 0
    total_bytes: int = 0
    message_chunks: int = 0
    reasoning_chunks: int = 0
    tool_chunks: int = 0
    _start_time: float = 0.0

    @property
    def time_to_first_chunk_ms(self) -> float | None:
        if self._start_time and self.first_chunk_time is not None:
            return (self.first_chunk_time - self._start_time) * 1000
        return None

    @property
    def chunks_per_second(self) -> float:
        if self.first_chunk_time is None or self.last_chunk_time is None:
            return 0.0
        elapsed = self.last_chunk_time - self.first_chunk_time
        if elapsed <= 0:
            return float(self.total_chunks)
        return self.total_chunks / elapsed

    @property
    def throughput_bytes_per_second(self) -> float:
        if self.first_chunk_time is None or self.last_chunk_time is None:
            return 0.0
        elapsed = self.last_chunk_time - self.first_chunk_time
        if elapsed <= 0:
            return float(self.total_bytes)
        return self.total_bytes / elapsed

    def record_chunk(self, chunk: StreamChunk) -> None:
        now = time.monotonic()
        if self.first_chunk_time is None:
            self.first_chunk_time = now
        self.last_chunk_time = now
        self.total_chunks += 1
        self.total_bytes += len(chunk.delta)
        if chunk.type == "message":
            self.message_chunks += 1
        elif chunk.type == "reasoning":
            self.reasoning_chunks += 1
        elif chunk.type in ("tool_call", "tool_result"):
            self.tool_chunks += 1


class ChunkBatcher:
    """Batches rapid-fire delta chunks to reduce callback overhead.

    For high-throughput streaming, invoking the on_chunk callback for
    every single token is expensive. ChunkBatcher accumulates delta
    chunks of the same type and flushes them when the type changes,
    a non-delta chunk arrives, or a size threshold is reached.
    """

    __slots__ = ("_callback", "_max_bytes", "_pending_type", "_pending_parts", "_pending_bytes")

    def __init__(
        self,
        callback: Callable[[StreamChunk], None],
        max_bytes: int = 4096,
    ) -> None:
        self._callback = callback
        self._max_bytes = max_bytes
        self._pending_type: str | None = None
        self._pending_parts: list[str] = []
        self._pending_bytes = 0

    def push(self, chunk: StreamChunk) -> None:
        is_delta = chunk.delta and not chunk.is_final and chunk.type in ("message", "reasoning")
        if is_delta:
            if self._pending_type == chunk.type:
                self._pending_parts.append(chunk.delta)
                self._pending_bytes += len(chunk.delta)
                if self._pending_bytes >= self._max_bytes:
                    self.flush()
            else:
                self.flush()
                self._pending_type = chunk.type
                self._pending_parts.append(chunk.delta)
                self._pending_bytes = len(chunk.delta)
        else:
            self.flush()
            self._callback(chunk)

    def flush(self) -> None:
        if self._pending_parts:
            merged = "".join(self._pending_parts)
            self._callback(StreamChunk(type=self._pending_type or "message", delta=merged))
            self._pending_parts.clear()
            self._pending_bytes = 0
            self._pending_type = None


@dataclass
class Response:
    """Response from a Copilot prompt."""

    content: str
    reasoning: str | None = None
    raw_events: list[dict[str, Any]] = field(default_factory=list)

    # Request accounting (when available from the SDK)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost: float | None = None
    server_model: str | None = None  # Actual model used (from assistant.usage event)

    retries: int = 0
    auto_continues: int = 0
    streaming_metrics: StreamingMetrics | None = None

    @property
    def usage(self) -> dict[str, int] | None:
        """Return token usage in an OpenAI-compatible shape when available."""
        if self.prompt_tokens is None and self.completion_tokens is None:
            return None
        return {
            "prompt_tokens": int(self.prompt_tokens or 0),
            "completion_tokens": int(self.completion_tokens or 0),
        }
