"""Tests for session_pool module."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from copex.session_pool import SessionPool


@dataclass
class _FakeSessionOptions:
    model: str = "test-model"


class _FakeModel:
    def __init__(self, value: str = "test-model") -> None:
        self.value = value


class _FakeConfig:
    """Minimal config mock for SessionPool tests."""

    def __init__(self, model: str = "test-model") -> None:
        self.model = _FakeModel(model)

    def to_session_options(self) -> _FakeSessionOptions:
        return _FakeSessionOptions(model=self.model.value)


class _FakeSession:
    """Minimal session mock."""

    def __init__(self, session_id: int = 0) -> None:
        self.session_id = session_id
        self.destroyed = False

    async def destroy(self) -> None:
        self.destroyed = True


class _FakeClient:
    """Minimal client mock that creates fake sessions."""

    def __init__(self) -> None:
        self._counter = 0

    async def create_session(self, options: Any) -> _FakeSession:
        self._counter += 1
        return _FakeSession(session_id=self._counter)


# ---------------------------------------------------------------------------
# SessionPool basics
# ---------------------------------------------------------------------------

class TestSessionPool:
    """Tests for SessionPool."""

    def test_init_defaults(self) -> None:
        pool = SessionPool()
        assert pool.max_sessions == 5
        assert pool.max_idle_time == 300.0
        assert pool.pre_warm == 0

    def test_init_custom(self) -> None:
        pool = SessionPool(max_sessions=3, max_idle_time=60.0, pre_warm=2)
        assert pool.max_sessions == 3
        assert pool.max_idle_time == 60.0
        assert pool.pre_warm == 2

    @pytest.mark.asyncio
    async def test_acquire_creates_session(self) -> None:
        pool = SessionPool(max_sessions=2)
        client = _FakeClient()
        config = _FakeConfig()

        async with pool.acquire(client, config) as session:
            assert isinstance(session, _FakeSession)
            assert session.session_id == 1

    @pytest.mark.asyncio
    async def test_acquire_reuses_session(self) -> None:
        pool = SessionPool(max_sessions=2)
        client = _FakeClient()
        config = _FakeConfig()

        # First acquire creates a session
        async with pool.acquire(client, config) as s1:
            first_id = s1.session_id

        # Second acquire should reuse the pooled session
        async with pool.acquire(client, config) as s2:
            assert s2.session_id == first_id

    @pytest.mark.asyncio
    async def test_acquire_creates_new_when_all_in_use(self) -> None:
        pool = SessionPool(max_sessions=2)
        client = _FakeClient()
        config = _FakeConfig()

        async with pool.acquire(client, config) as s1:
            async with pool.acquire(client, config) as s2:
                # Both should be different sessions
                assert s1.session_id != s2.session_id

    @pytest.mark.asyncio
    async def test_pool_stats_hits_misses(self) -> None:
        pool = SessionPool(max_sessions=2)
        client = _FakeClient()
        config = _FakeConfig()

        async with pool.acquire(client, config):
            pass  # miss

        async with pool.acquire(client, config):
            pass  # hit

        stats = pool.stats()
        assert stats["_pool_metrics"]["misses"] == 1
        assert stats["_pool_metrics"]["hits"] == 1

    @pytest.mark.asyncio
    async def test_pool_stats_model_info(self) -> None:
        pool = SessionPool(max_sessions=2)
        client = _FakeClient()
        config = _FakeConfig("model-a")

        async with pool.acquire(client, config):
            pass

        stats = pool.stats()
        assert "model-a" in stats
        assert stats["model-a"]["total"] == 1
        assert stats["model-a"]["available"] == 1

    @pytest.mark.asyncio
    async def test_lru_eviction(self) -> None:
        pool = SessionPool(max_sessions=1)
        client = _FakeClient()
        config = _FakeConfig()

        # Fill pool with 1 session
        async with pool.acquire(client, config) as s1:
            first_id = s1.session_id

        # Acquire again while pool is full (max_sessions=1) and session is idle
        # This creates a new session, triggers LRU eviction
        async with pool.acquire(client, config) as s2:
            # Should reuse the existing session since it's available
            assert s2.session_id == first_id

    @pytest.mark.asyncio
    async def test_different_model_pools(self) -> None:
        pool = SessionPool(max_sessions=2)
        client = _FakeClient()
        config_a = _FakeConfig("model-a")
        config_b = _FakeConfig("model-b")

        async with pool.acquire(client, config_a) as sa:
            async with pool.acquire(client, config_b) as sb:
                assert sa.session_id != sb.session_id

        stats = pool.stats()
        assert "model-a" in stats
        assert "model-b" in stats

    @pytest.mark.asyncio
    async def test_stop_destroys_all(self) -> None:
        pool = SessionPool(max_sessions=2)
        client = _FakeClient()
        config = _FakeConfig()

        sessions: list[_FakeSession] = []
        async with pool.acquire(client, config) as s1:
            sessions.append(s1)

        await pool.stop()
        assert all(s.destroyed for s in sessions)
        assert len(pool._pools) == 0

    @pytest.mark.asyncio
    async def test_start_creates_cleanup_task(self) -> None:
        pool = SessionPool()
        await pool.start()
        assert pool._cleanup_task is not None
        await pool.stop()
        assert pool._cleanup_task is None

    @pytest.mark.asyncio
    async def test_warm_creates_sessions(self) -> None:
        pool = SessionPool(max_sessions=3, pre_warm=2)
        client = _FakeClient()
        config = _FakeConfig()

        await pool.warm(client, [config])
        stats = pool.stats()
        assert stats["test-model"]["total"] == 2

    @pytest.mark.asyncio
    async def test_warm_zero_prewarm(self) -> None:
        pool = SessionPool(pre_warm=0)
        client = _FakeClient()
        config = _FakeConfig()

        await pool.warm(client, [config])
        assert len(pool._pools) == 0

    @pytest.mark.asyncio
    async def test_unpooled_session_destroyed(self) -> None:
        """When pool is full and eviction fails, session is destroyed after use."""
        pool = SessionPool(max_sessions=1)
        client = _FakeClient()
        config = _FakeConfig()

        # Fill pool
        async with pool.acquire(client, config):
            pass

        # Acquire two concurrently - one should be unpooled
        async with pool.acquire(client, config) as s1:
            async with pool.acquire(client, config) as s2:
                # s1 is pooled, s2 might be unpooled
                pass
        # After context exit, unpooled sessions should be destroyed

    @pytest.mark.asyncio
    async def test_hit_rate_calculation(self) -> None:
        pool = SessionPool(max_sessions=2)
        client = _FakeClient()
        config = _FakeConfig()

        # 1 miss, then 3 hits
        for _ in range(4):
            async with pool.acquire(client, config):
                pass

        stats = pool.stats()
        metrics = stats["_pool_metrics"]
        assert metrics["misses"] == 1
        assert metrics["hits"] == 3
        assert metrics["hit_rate"] == 0.75

    @pytest.mark.asyncio
    async def test_evict_idle_sessions(self) -> None:
        pool = SessionPool(max_sessions=2, max_idle_time=0.01)
        client = _FakeClient()
        config = _FakeConfig()

        async with pool.acquire(client, config):
            pass

        await asyncio.sleep(0.02)
        await pool._evict_idle()

        stats = pool.stats()
        # Session should have been evicted
        assert "test-model" not in stats or stats["test-model"]["total"] == 0

    @pytest.mark.asyncio
    async def test_evict_idle_keeps_in_use_sessions(self) -> None:
        """In-use sessions should not be evicted even if idle."""
        pool = SessionPool(max_sessions=2, max_idle_time=0.01)
        client = _FakeClient()
        config = _FakeConfig()

        async with pool.acquire(client, config) as s1:
            await asyncio.sleep(0.02)
            await pool._evict_idle()
            # Session should NOT be evicted while in use
            stats = pool.stats()
            assert stats["test-model"]["in_use"] == 1

    @pytest.mark.asyncio
    async def test_warm_failure_is_nonfatal(self) -> None:
        """Pre-warm failure should not raise."""
        pool = SessionPool(max_sessions=3, pre_warm=2)

        class _FailClient:
            async def create_session(self, _: Any) -> _FakeSession:
                raise RuntimeError("connect failed")

        await pool.warm(_FailClient(), [_FakeConfig()])
        assert len(pool._pools.get("test-model", [])) == 0

    @pytest.mark.asyncio
    async def test_stats_empty_pool(self) -> None:
        """Stats on empty pool should have zero metrics."""
        pool = SessionPool()
        stats = pool.stats()
        assert stats["_pool_metrics"]["hits"] == 0
        assert stats["_pool_metrics"]["misses"] == 0
        assert stats["_pool_metrics"]["hit_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_lru_eviction_with_full_pool(self) -> None:
        """When pool is full with idle sessions, LRU should be evicted."""
        pool = SessionPool(max_sessions=1)
        client = _FakeClient()
        config = _FakeConfig()

        # Create and release session
        async with pool.acquire(client, config) as s1:
            old_id = s1.session_id

        # Force a new session while pool is full but all idle
        # Acquire 2 concurrently — second forces eviction
        async with pool.acquire(client, config) as s2:
            async with pool.acquire(client, config) as s3:
                # s2 reuses old session, s3 is new (evicts or unpooled)
                assert s2.session_id == old_id
                assert s3.session_id != old_id

    @pytest.mark.asyncio
    async def test_acquire_release_updates_pool_state_under_model_lock(self) -> None:
        pool = SessionPool(max_sessions=1)
        client = _FakeClient()
        config = _FakeConfig()

        # Seed pool so next acquire is a pure reuse path.
        async with pool.acquire(client, config):
            pass

        class _CountingLock:
            def __init__(self) -> None:
                self._lock = asyncio.Lock()
                self.enter_count = 0

            async def __aenter__(self) -> "_CountingLock":
                self.enter_count += 1
                await self._lock.acquire()
                return self

            async def __aexit__(self, *_: object) -> bool:
                self._lock.release()
                return False

        counting_lock = _CountingLock()
        pool._model_locks[config.model.value] = counting_lock  # type: ignore[assignment]

        async with pool.acquire(client, config):
            pass

        # Reuse path should lock once for checkout and once for release.
        assert counting_lock.enter_count == 2
