"""Core Copex client with retry logic and stuck detection."""

from __future__ import annotations

import asyncio
import logging
import random
import time
import warnings
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from copilot import CopilotClient

from copex.backoff import AdaptiveRetry, ErrorCategory, categorize_error
from copex.config import CopexConfig
from copex.metrics import MetricsCollector, get_collector
from copex.models import EventType, Model, ReasoningEffort, parse_reasoning_effort

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patch: Remove ``--no-auto-update`` from the CLI startup args.
#
# The Copilot SDK passes ``--no-auto-update`` when spawning the CLI server in
# headless mode.  This prevents the binary from fetching the latest model list
# from the Copilot backend, causing newer models (e.g. ``claude-opus-4.6``,
# ``claude-opus-4.6-fast``) to silently fall back to ``claude-sonnet-4.5``.
#
# We monkey-patch ``CopilotClient._start_cli_server`` to strip that flag so
# the CLI always operates with the up-to-date model catalogue.
# ---------------------------------------------------------------------------
_original_start_cli_server = CopilotClient._start_cli_server  # type: ignore[attr-defined]


async def _patched_start_cli_server(self: CopilotClient) -> None:
    """Wrapper that removes ``--no-auto-update`` before starting the CLI."""
    import os
    import re
    import subprocess

    cli_path = self.options["cli_path"]

    if not os.path.exists(cli_path):
        raise RuntimeError(f"Copilot CLI not found at {cli_path}")

    # Build args WITHOUT --no-auto-update
    args = ["--headless", "--log-level", self.options["log_level"]]

    if self.options.get("github_token"):
        args.extend(["--auth-token-env", "COPILOT_SDK_AUTH_TOKEN"])
    if not self.options.get("use_logged_in_user", True):
        args.append("--no-auto-login")

    if cli_path.endswith(".js"):
        args = ["node", cli_path] + args
    else:
        args = [cli_path] + args

    env = self.options.get("env")
    if env is None:
        env = dict(os.environ)
    else:
        env = dict(env)

    if self.options.get("github_token"):
        env["COPILOT_SDK_AUTH_TOKEN"] = self.options["github_token"]

    if self.options["use_stdio"]:
        args.append("--stdio")
        self._process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            cwd=self.options["cwd"],
            env=env,
        )
    else:
        if self.options["port"] > 0:
            args.extend(["--port", str(self.options["port"])])
        self._process = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.options["cwd"],
            env=env,
        )

    if self.options["use_stdio"]:
        return

    loop = asyncio.get_event_loop()
    process = self._process

    async def read_port() -> None:
        if not process or not process.stdout:
            raise RuntimeError("Process not started or stdout not available")
        while True:
            line = await loop.run_in_executor(None, process.stdout.readline)
            if not line:
                raise RuntimeError("CLI process exited before announcing port")
            line_str = line.decode() if isinstance(line, bytes) else line
            match = re.search(r"listening on port (\d+)", line_str, re.IGNORECASE)
            if match:
                self._actual_port = int(match.group(1))
                return

    try:
        await asyncio.wait_for(read_port(), timeout=10.0)
    except asyncio.TimeoutError:
        raise RuntimeError("Timeout waiting for CLI server to start")


CopilotClient._start_cli_server = _patched_start_cli_server  # type: ignore[attr-defined]
logger.debug("Patched CopilotClient._start_cli_server to remove --no-auto-update")

# Circuit breaker defaults
_CB_FAILURE_THRESHOLD = 5
_CB_COOLDOWN_SECONDS = 60.0

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


class SlidingWindowBreaker:
    """Circuit breaker using a sliding window of recent request outcomes.

    Unlike simple consecutive failure counting, this tracks a window of
    the last N requests and opens the circuit when the failure rate
    exceeds a threshold.

    Args:
        window_size: Number of recent requests to track (default: 10)
        threshold: Failure rate threshold to open circuit (default: 0.5 = 50%)
        cooldown_seconds: Seconds to wait before half-open state (default: 60)

    Example:
        breaker = SlidingWindowBreaker(window_size=10, threshold=0.5)
        breaker.check()  # Raises if circuit is open
        breaker.record_success()
        breaker.record_failure()
    """

    def __init__(
        self,
        window_size: int = 10,
        threshold: float = 0.5,
        cooldown_seconds: float = 60.0,
    ) -> None:
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if window_size < 1:
            raise ValueError("window_size must be >= 1")

        self.window_size = window_size
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds

        # Sliding window: True = success, False = failure
        self._window: deque[bool] = deque(maxlen=window_size)
        self._opened_at: float | None = None
        self._lock = asyncio.Lock()

    @property
    def failure_rate(self) -> float:
        """Current failure rate in the sliding window."""
        if not self._window:
            return 0.0
        failures = sum(1 for success in self._window if not success)
        return failures / len(self._window)

    @property
    def is_open(self) -> bool:
        """Check if circuit is currently open."""
        if self._opened_at is None:
            return False
        elapsed = time.monotonic() - self._opened_at
        return elapsed < self.cooldown_seconds

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is in half-open state (cooldown elapsed)."""
        if self._opened_at is None:
            return False
        elapsed = time.monotonic() - self._opened_at
        return elapsed >= self.cooldown_seconds

    def check(self) -> None:
        """Check circuit state; raise RuntimeError if open."""
        if self._opened_at is not None:
            elapsed = time.monotonic() - self._opened_at
            if elapsed < self.cooldown_seconds:
                remaining = self.cooldown_seconds - elapsed
                raise RuntimeError(
                    f"Circuit breaker open (failure rate {self.failure_rate:.0%}). "
                    f"Retry in {remaining:.0f}s."
                )
            # Cooldown elapsed - half-open: reset for retry
            self._opened_at = None
            self._window.clear()

    def record_success(self) -> None:
        """Record a successful request."""
        self._window.append(True)
        if self._opened_at is not None:
            # Successful request in half-open state closes the circuit
            self._opened_at = None

    def record_failure(self) -> None:
        """Record a failed request, potentially opening the circuit."""
        self._window.append(False)
        # Only evaluate after window has enough samples
        if len(self._window) >= self.window_size // 2:
            if self.failure_rate >= self.threshold:
                if self._opened_at is None:
                    self._opened_at = time.monotonic()
                    logger.warning(
                        "Circuit breaker opened: failure rate %.0f%% "
                        "exceeds threshold %.0f%%",
                        self.failure_rate * 100,
                        self.threshold * 100,
                    )

    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        self._window.clear()
        self._opened_at = None


class ModelAwareBreaker:
    """Per-model circuit breakers for granular failure isolation.

    Different models may have different reliability characteristics.
    This class maintains separate circuit breakers per model.

    Args:
        window_size: Window size for each breaker
        threshold: Failure threshold for each breaker
        cooldown_seconds: Cooldown for each breaker

    Example:
        breakers = ModelAwareBreaker()
        breakers.check("gpt-5.2-codex")
        breakers.record_success("gpt-5.2-codex")
        breakers.record_failure("claude-opus-4.5")
    """

    def __init__(
        self,
        window_size: int = 10,
        threshold: float = 0.5,
        cooldown_seconds: float = 60.0,
    ) -> None:
        self.window_size = window_size
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self._breakers: dict[str, SlidingWindowBreaker] = {}

    def _get_breaker(self, model: str) -> SlidingWindowBreaker:
        """Get or create a circuit breaker for a model."""
        if model not in self._breakers:
            self._breakers[model] = SlidingWindowBreaker(
                window_size=self.window_size,
                threshold=self.threshold,
                cooldown_seconds=self.cooldown_seconds,
            )
        return self._breakers[model]

    def check(self, model: str) -> None:
        """Check if circuit is open for a model."""
        self._get_breaker(model).check()

    def record_success(self, model: str) -> None:
        """Record a successful request for a model."""
        self._get_breaker(model).record_success()

    def record_failure(self, model: str) -> None:
        """Record a failed request for a model."""
        self._get_breaker(model).record_failure()

    def reset(self, model: str | None = None) -> None:
        """Reset breaker(s). If model is None, reset all."""
        if model is None:
            self._breakers.clear()
        elif model in self._breakers:
            self._breakers[model].reset()

    def get_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all breakers."""
        return {
            model: {
                "failure_rate": breaker.failure_rate,
                "is_open": breaker.is_open,
                "is_half_open": breaker.is_half_open,
                "window_size": len(breaker._window),
            }
            for model, breaker in self._breakers.items()
        }

    def is_open(self, model: str) -> bool:
        """Check if circuit is open for a model (without raising)."""
        breaker = self._breakers.get(model)
        return breaker.is_open if breaker else False

    def get_available_model(
        self,
        preferred: str,
        fallback_chain: list[str] | None = None,
    ) -> str | None:
        """Get the first available model from the fallback chain.

        If the preferred model's circuit is open, returns the first
        model in the fallback chain whose circuit is closed or
        half-open. Returns None if all circuits are open.

        Args:
            preferred: The preferred model to use
            fallback_chain: Optional list of fallback models in order

        Returns:
            The model to use, or None if all are unavailable

        Example:
            model = breaker.get_available_model(
                "claude-opus-4.5",
                fallback_chain=["claude-sonnet-4.5", "claude-haiku-4.5"]
            )
        """
        # Check preferred model first
        if not self.is_open(preferred):
            return preferred

        # Try fallback chain
        if fallback_chain:
            for fallback in fallback_chain:
                if not self.is_open(fallback):
                    logger.info(
                        "Circuit open for %s, falling back to %s",
                        preferred,
                        fallback,
                    )
                    return fallback

        return None


# Default fallback chains for common model families
DEFAULT_FALLBACK_CHAINS: dict[str, list[str]] = {
    # Claude family: opus -> sonnet -> haiku
    "claude-opus-4.6": ["claude-opus-4.6-fast", "claude-opus-4.5", "claude-sonnet-4.5", "claude-haiku-4.5"],
    "claude-opus-4.6-fast": ["claude-opus-4.6", "claude-opus-4.5", "claude-sonnet-4.5", "claude-haiku-4.5"],
    "claude-opus-4.5": ["claude-sonnet-4.5", "claude-haiku-4.5"],
    "claude-sonnet-4.5": ["claude-sonnet-4", "claude-haiku-4.5"],
    "claude-sonnet-4": ["claude-haiku-4.5"],
    # GPT family: codex -> codex-max -> regular
    "gpt-5.2-codex": ["gpt-5.1-codex", "gpt-5.2", "gpt-5.1"],
    "gpt-5.1-codex": ["gpt-5.1-codex-max", "gpt-5.1", "gpt-5"],
    "gpt-5.2": ["gpt-5.1", "gpt-5"],
    "gpt-5.1": ["gpt-5", "gpt-5-mini"],
    "gpt-5": ["gpt-5-mini", "gpt-4.1"],
}


class SessionPool:
    """Connection pool for Copex sessions.

    Maintains a pool of reusable sessions to reduce connection overhead.
    Sessions are model-specific since different models require different
    session configurations.  Uses LRU eviction when pool is full and
    tracks hit/miss metrics.

    Args:
        max_sessions: Maximum sessions per model (default: 5)
        max_idle_time: Seconds before idle session is evicted (default: 300)
        pre_warm: Number of sessions to pre-create per model on warm-up (default: 0)

    Example:
        pool = SessionPool()
        async with pool.acquire(client, config) as session:
            await session.send({"prompt": "Hello"})
    """

    @dataclass
    class _PooledSession:
        session: Any
        model: str
        created_at: float
        last_used: float
        in_use: bool = False

    def __init__(
        self,
        max_sessions: int = 5,
        max_idle_time: float = 300.0,
        pre_warm: int = 0,
    ) -> None:
        self.max_sessions = max_sessions
        self.max_idle_time = max_idle_time
        self.pre_warm = pre_warm
        # Per-model pools keyed by model name
        self._pools: dict[str, list[SessionPool._PooledSession]] = {}
        # Per-model locks to reduce contention
        self._model_locks: dict[str, asyncio.Lock] = {}
        # Global lock only for creating new model entries
        self._global_lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[None] | None = None
        # Pool hit/miss metrics
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

    async def _get_model_lock(self, model: str) -> asyncio.Lock:
        """Get or create a per-model lock."""
        if model not in self._model_locks:
            async with self._global_lock:
                if model not in self._model_locks:
                    self._model_locks[model] = asyncio.Lock()
        return self._model_locks[model]

    async def warm(
        self,
        client: Any,
        configs: list[CopexConfig],
    ) -> None:
        """Pre-warm the pool by creating sessions for the given configs.

        Args:
            client: CopilotClient instance
            configs: List of configs whose models should be pre-warmed
        """
        if self.pre_warm <= 0:
            return
        for config in configs:
            model = config.model.value
            lock = await self._get_model_lock(model)
            async with lock:
                pool = self._pools.setdefault(model, [])
                needed = min(self.pre_warm, self.max_sessions) - len(pool)
                for _ in range(needed):
                    try:
                        session = await client.create_session(
                            config.to_session_options()
                        )
                        pool.append(
                            SessionPool._PooledSession(
                                session=session,
                                model=model,
                                created_at=time.monotonic(),
                                last_used=time.monotonic(),
                            )
                        )
                    except Exception:
                        logger.debug("Failed to pre-warm session", exc_info=True)
                        break

    async def start(self) -> None:
        """Start the pool cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop the pool and destroy all sessions."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        # Destroy all pooled sessions
        async with self._global_lock:
            for pool in self._pools.values():
                for ps in pool:
                    try:
                        await ps.session.destroy()
                    except Exception:
                        logger.debug("Failed to destroy pooled session", exc_info=True)
            self._pools.clear()
            self._model_locks.clear()

    async def _cleanup_loop(self) -> None:
        """Periodically clean up idle sessions."""
        while True:
            await asyncio.sleep(60)  # Check every minute
            await self._evict_idle()

    async def _evict_idle(self) -> None:
        """Evict sessions that have been idle too long."""
        now = time.monotonic()
        for model in list(self._pools.keys()):
            lock = await self._get_model_lock(model)
            async with lock:
                pool = self._pools.get(model)
                if pool is None:
                    continue
                to_remove = []
                for ps in pool:
                    if not ps.in_use and (now - ps.last_used) > self.max_idle_time:
                        to_remove.append(ps)

                for ps in to_remove:
                    pool.remove(ps)
                    self._evictions += 1
                    try:
                        await ps.session.destroy()
                    except Exception:
                        logger.debug("Failed to destroy idle session", exc_info=True)

                if not pool:
                    del self._pools[model]

    def _evict_lru(self, pool: list[SessionPool._PooledSession]) -> _PooledSession | None:
        """Find the least-recently-used idle session for eviction."""
        lru: SessionPool._PooledSession | None = None
        for ps in pool:
            if not ps.in_use:
                if lru is None or ps.last_used < lru.last_used:
                    lru = ps
        return lru

    @asynccontextmanager
    async def acquire(
        self,
        client: Any,
        config: CopexConfig,
    ) -> AsyncIterator[Any]:
        """Acquire a session from the pool.

        Args:
            client: CopilotClient instance
            config: CopexConfig for session creation

        Yields:
            A session to use

        Example:
            async with pool.acquire(client, config) as session:
                await session.send({"prompt": "Hello"})
        """
        model = config.model.value
        session = None
        pooled: SessionPool._PooledSession | None = None

        lock = await self._get_model_lock(model)
        async with lock:
            pool = self._pools.setdefault(model, [])

            # LRU: pick the most-recently-used idle session for best reuse
            best: SessionPool._PooledSession | None = None
            for ps in pool:
                if not ps.in_use:
                    if best is None or ps.last_used > best.last_used:
                        best = ps
            if best is not None:
                best.in_use = True
                best.last_used = time.monotonic()
                session = best.session
                pooled = best
                self._hits += 1

        # No available session - create new one
        is_pooled = False
        if session is None:
            self._misses += 1
            session = await client.create_session(config.to_session_options())
            pooled = SessionPool._PooledSession(
                session=session,
                model=model,
                created_at=time.monotonic(),
                last_used=time.monotonic(),
                in_use=True,
            )
            async with lock:
                pool = self._pools.setdefault(model, [])
                if len(pool) < self.max_sessions:
                    pool.append(pooled)
                    is_pooled = True
                else:
                    # LRU eviction: replace least-recently-used idle session
                    lru = self._evict_lru(pool)
                    if lru is not None:
                        pool.remove(lru)
                        self._evictions += 1
                        pool.append(pooled)
                        is_pooled = True
                        try:
                            await lru.session.destroy()
                        except Exception:
                            logger.debug(
                                "Failed to destroy evicted session", exc_info=True
                            )
        else:
            is_pooled = True

        try:
            yield session
        finally:
            # Return session to pool or destroy if not pooled
            if pooled is not None:
                pooled.in_use = False
                pooled.last_used = time.monotonic()
            if not is_pooled and session is not None:
                try:
                    await session.destroy()
                except Exception:
                    pass

    def stats(self) -> dict[str, Any]:
        """Get pool statistics including hit/miss metrics."""
        result: dict[str, Any] = {
            "_pool_metrics": {
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": (
                    self._hits / (self._hits + self._misses)
                    if (self._hits + self._misses) > 0
                    else 0.0
                ),
            },
        }
        for model, pool in self._pools.items():
            result[model] = {
                "total": len(pool),
                "in_use": sum(1 for ps in pool if ps.in_use),
                "available": sum(1 for ps in pool if not ps.in_use),
            }
        return result


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
            except Exception:
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
        except Exception:
            return
        try:
            await session.abort()
        except Exception:
            # Aborting is best-effort; ignore failures.
            return

    async def __aenter__(self) -> "Copex":
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

    def _handle_message_delta(
        self,
        event: Any,
        state: _SendState,
        on_chunk: Callable[[StreamChunk], None] | None,
    ) -> None:
        data = event.data
        delta = getattr(data, "delta_content", None) or getattr(data, "transformed_content", None) or ""
        if delta:
            state.received_content = True
            state.content_parts.append(delta)
        if (
            state.awaiting_post_tool_response
            and state.tool_execution_seen
            and state.pending_tools == 0
        ):
            state.awaiting_post_tool_response = False
        if on_chunk:
            chunk = StreamChunk(type="message", delta=delta)
            state.streaming_metrics.record_chunk(chunk)
            on_chunk(chunk)

    def _handle_reasoning_delta(
        self,
        event: Any,
        state: _SendState,
        on_chunk: Callable[[StreamChunk], None] | None,
    ) -> None:
        delta = getattr(event.data, "delta_content", None) or ""
        if delta:
            state.reasoning_parts.append(delta)
        if on_chunk:
            chunk = StreamChunk(type="reasoning", delta=delta)
            state.streaming_metrics.record_chunk(chunk)
            on_chunk(chunk)

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
        if (
            state.awaiting_post_tool_response
            and state.tool_execution_seen
            and state.pending_tools == 0
        ):
            state.awaiting_post_tool_response = False
        if on_chunk:
            on_chunk(
                StreamChunk(
                    type="message",
                    delta="",
                    is_final=True,
                    content=state.final_content,
                )
            )

    def _handle_reasoning(
        self,
        event: Any,
        state: _SendState,
        on_chunk: Callable[[StreamChunk], None] | None,
    ) -> None:
        state.final_reasoning = getattr(event.data, "content", "") or ""
        if on_chunk:
            on_chunk(
                StreamChunk(
                    type="reasoning",
                    delta="",
                    is_final=True,
                    content=state.final_reasoning,
                )
            )

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
            except Exception:
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
        from copex.models import _NO_REASONING_MODELS
        if not self._started:
            await self.start()
        if self._session is None:
            # github-copilot-sdk >= 0.1.21 supports reasoning_effort natively
            # Use current model (may be fallback) if set (v1.9.0)
            session_options = self.config.to_session_options()
            if self._current_model and self._current_model != self.config.model.value:
                session_options["model"] = self._current_model
                # If the fallback model doesn't support reasoning, drop it
                if self._current_model in _NO_REASONING_MODELS:
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
            except Exception:
                logger.debug("Failed to destroy session during recovery", exc_info=True)
            self._session = None

        # Create fresh session
        try:
            session = await self._ensure_session()
        except Exception:
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
                except Exception:
                    pass
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
                        self._cb_record_failure()
                        raise last_error
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
        loop = asyncio.get_running_loop()
        state.last_activity = loop.time()
        state.streaming_metrics._start_time = time.monotonic()

        def _handle_usage(event: Any, st: _SendState, _oc: Any) -> None:
            data = event.data
            inp = getattr(data, "input_tokens", None)
            out = getattr(data, "output_tokens", None)
            cost = getattr(data, "cost", None)
            if inp is not None:
                try:
                    st.prompt_tokens = int(inp)
                except Exception:
                    pass
            if out is not None:
                try:
                    st.completion_tokens = int(out)
                except Exception:
                    pass
            if cost is not None:
                try:
                    st.cost = float(cost)
                except Exception:
                    pass

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

            except Exception as e:
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
                        )
                    # Had recent activity, keep waiting
        finally:
            # Remove event handler to avoid duplicates
            try:
                unsubscribe()
            except Exception:
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
            except Exception:
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
                except Exception:
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
