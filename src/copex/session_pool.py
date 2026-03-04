"""Connection pool for reusable Copex sessions."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from copex.config import CopexConfig

logger = logging.getLogger(__name__)


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
        async with self._global_lock:
            return self._model_locks.setdefault(model, asyncio.Lock())

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
                    except Exception:  # Cleanup: pre-warm failure is non-fatal
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
                    except Exception:  # Cleanup: best-effort session teardown
                        logger.debug("Failed to destroy pooled session", exc_info=True)
            self._pools.clear()
            self._model_locks.clear()

    async def _cleanup_loop(self) -> None:
        """Periodically clean up idle sessions."""
        try:
            while True:
                await asyncio.sleep(60)  # Check every minute
                await self._evict_idle()
        except asyncio.CancelledError:
            return

    async def _evict_idle(self) -> None:
        """Evict sessions that have been idle too long.

        Collects eviction candidates under per-model locks (no global lock held)
        to avoid lock-ordering inversion with ``acquire()`` which takes model
        lock → global lock.
        """
        now = time.monotonic()
        to_destroy: list[Any] = []

        # Snapshot model keys under the global lock, then release it
        # before acquiring per-model locks.
        async with self._global_lock:
            model_keys = list(self._pools.keys())

        for model in model_keys:
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
                    to_destroy.append(ps.session)

                if not pool:
                    async with self._global_lock:
                        # Re-check under lock before deleting
                        if model in self._pools and not self._pools[model]:
                            del self._pools[model]

        for session in to_destroy:
            try:
                await session.destroy()
            except Exception:  # Cleanup: best-effort idle session teardown
                logger.debug("Failed to destroy idle session", exc_info=True)

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

            # MRU: pick the most-recently-used idle session for best reuse
            # (warm sessions are more likely to be healthy)
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
        evicted_session: Any | None = None
        if session is None:
            session = await client.create_session(config.to_session_options())
            pooled = SessionPool._PooledSession(
                session=session,
                model=model,
                created_at=time.monotonic(),
                last_used=time.monotonic(),
                in_use=True,
            )
            async with lock:
                self._misses += 1
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
                        evicted_session = lru.session
                    else:
                        # All sessions in use and pool full — cannot pool this session
                        logger.debug(
                            "Pool full for model %s (all %d sessions in use), "
                            "session will be destroyed after use",
                            model,
                            self.max_sessions,
                        )
            if evicted_session is not None:
                try:
                    await evicted_session.destroy()
                except Exception:  # Cleanup: best-effort evicted session teardown
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
                async with lock:
                    pooled.in_use = False
                    pooled.last_used = time.monotonic()
            if not is_pooled and session is not None:
                try:
                    await session.destroy()
                except Exception:  # Cleanup: best-effort unpooled session teardown
                    logger.debug("Failed to destroy unpooled session", exc_info=True)

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
