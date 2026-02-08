"""Advanced fleet tests: DynamicSemaphore, AdaptiveConcurrency, retry logic,
dependency timeout, validation, and helper functions."""

from __future__ import annotations

import asyncio

import pytest

from copex.client import Response
from copex.config import CopexConfig
from copex.fleet import (
    AdaptiveConcurrency,
    DynamicSemaphore,
    Fleet,
    FleetConfig,
    FleetResult,
    FleetTask,
    _is_rate_limit_error,
    _normalize_mcp_servers,
    _prepend_shared_context,
    _run_task_with_retry,
)
from unittest.mock import AsyncMock, patch


# ---------------------------------------------------------------------------
# TestDynamicSemaphore
# ---------------------------------------------------------------------------


class TestDynamicSemaphore:
    @pytest.mark.asyncio
    async def test_acquire_release_basic(self):
        sem = DynamicSemaphore(2)
        await sem.acquire()
        assert sem._active == 1
        await sem.release()
        assert sem._active == 0

    @pytest.mark.asyncio
    async def test_limit_property(self):
        sem = DynamicSemaphore(5)
        assert sem.limit == 5

    @pytest.mark.asyncio
    async def test_minimum_limit_is_one(self):
        sem = DynamicSemaphore(0)
        assert sem.limit == 1
        sem2 = DynamicSemaphore(-5)
        assert sem2.limit == 1

    @pytest.mark.asyncio
    async def test_context_manager(self):
        sem = DynamicSemaphore(2)
        async with sem as s:
            assert s is sem
            assert sem._active == 1
        assert sem._active == 0

    @pytest.mark.asyncio
    async def test_concurrent_acquire_respects_limit(self):
        sem = DynamicSemaphore(2)
        acquired = []

        async def _acquire_and_hold(idx: int, hold: float = 0.05):
            async with sem:
                acquired.append(idx)
                await asyncio.sleep(hold)

        # Launch 3 tasks with limit=2: third must wait
        tasks = [
            asyncio.create_task(_acquire_and_hold(i)) for i in range(3)
        ]
        await asyncio.sleep(0.01)  # Let first two acquire
        assert sem._active <= 2
        await asyncio.gather(*tasks)
        assert len(acquired) == 3

    @pytest.mark.asyncio
    async def test_resize_down_blocks_new(self):
        sem = DynamicSemaphore(3)
        await sem.acquire()
        await sem.acquire()
        assert sem._active == 2
        # Resize down to 2 — no room for more
        await sem.resize(2)
        assert sem.limit == 2

        acquired_third = asyncio.Event()

        async def _try_acquire():
            await sem.acquire()
            acquired_third.set()

        task = asyncio.create_task(_try_acquire())
        await asyncio.sleep(0.02)
        # Third acquire should be blocked (active == limit)
        assert not acquired_third.is_set()
        # Release one to unblock
        await sem.release()
        await asyncio.sleep(0.02)
        assert acquired_third.is_set()
        await sem.release()  # cleanup
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_resize_up_allows_more(self):
        sem = DynamicSemaphore(1)
        await sem.acquire()
        assert sem._active == 1

        acquired_second = asyncio.Event()

        async def _try_acquire():
            await sem.acquire()
            acquired_second.set()

        task = asyncio.create_task(_try_acquire())
        await asyncio.sleep(0.02)
        assert not acquired_second.is_set()  # blocked by limit=1

        await sem.resize(2)  # expand
        await asyncio.sleep(0.02)
        assert acquired_second.is_set()
        await sem.release()
        await sem.release()
        await task


# ---------------------------------------------------------------------------
# TestAdaptiveConcurrency
# ---------------------------------------------------------------------------


class TestAdaptiveConcurrency:
    @pytest.mark.asyncio
    async def test_on_rate_limit_reduces_concurrency(self):
        ac = AdaptiveConcurrency(initial=8, minimum=1)
        new = await ac.on_rate_limit()
        assert new == 4
        assert ac.current == 4
        assert ac.semaphore.limit == 4

    @pytest.mark.asyncio
    async def test_on_rate_limit_respects_minimum(self):
        ac = AdaptiveConcurrency(initial=4, minimum=2)
        await ac.on_rate_limit()  # 4 -> 2
        new = await ac.on_rate_limit()  # 2 -> max(2, 1) = 2
        assert new == 2
        assert ac.current == 2

    @pytest.mark.asyncio
    async def test_on_success_increments_streak(self):
        ac = AdaptiveConcurrency(initial=4, restore_after=3)
        await ac.on_success()
        assert ac._success_streak == 1
        await ac.on_success()
        assert ac._success_streak == 2

    @pytest.mark.asyncio
    async def test_on_success_restores_concurrency(self):
        ac = AdaptiveConcurrency(initial=4, minimum=1, restore_after=3)
        await ac.on_rate_limit()  # 4 -> 2
        assert ac.current == 2
        # 3 successes should restore by 1
        await ac.on_success()
        await ac.on_success()
        new = await ac.on_success()
        assert new == 3
        assert ac.semaphore.limit == 3

    @pytest.mark.asyncio
    async def test_on_success_does_not_exceed_initial(self):
        ac = AdaptiveConcurrency(initial=2, minimum=1, restore_after=1)
        # Already at initial — success shouldn't increase
        new = await ac.on_success()
        assert new == 2
        assert ac.current == 2

    @pytest.mark.asyncio
    async def test_on_failure_resets_streak(self):
        ac = AdaptiveConcurrency(initial=4, restore_after=3)
        await ac.on_success()
        await ac.on_success()
        assert ac._success_streak == 2
        await ac.on_failure()
        assert ac._success_streak == 0
        # Concurrency unchanged
        assert ac.current == 4

    @pytest.mark.asyncio
    async def test_multiple_rate_limits_reduce_further(self):
        ac = AdaptiveConcurrency(initial=16, minimum=2)
        await ac.on_rate_limit()  # 16 -> 8
        assert ac.current == 8
        await ac.on_rate_limit()  # 8 -> 4
        assert ac.current == 4
        await ac.on_rate_limit()  # 4 -> 2
        assert ac.current == 2
        await ac.on_rate_limit()  # 2 -> max(2, 1) = 2
        assert ac.current == 2


# ---------------------------------------------------------------------------
# TestRunTaskWithRetry
# ---------------------------------------------------------------------------


class TestRunTaskWithRetry:
    @pytest.mark.asyncio
    async def test_no_retry_on_success(self):
        task = FleetTask(id="t1", prompt="do it")
        config = FleetConfig(default_retries=3)

        run_once = AsyncMock(return_value=FleetResult(
            task_id="t1", success=True, response=Response(content="ok"),
        ))

        result = await _run_task_with_retry(task, config, run_once)
        assert result.success
        run_once.assert_called_once()

    @pytest.mark.asyncio
    async def test_retries_on_failure(self):
        task = FleetTask(id="t1", prompt="do it")
        config = FleetConfig(default_retries=2, default_retry_delay=0.01)

        call_count = 0

        async def _run_once(t, attempt):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return FleetResult(task_id="t1", success=False, error=RuntimeError("fail"))
            return FleetResult(task_id="t1", success=True, response=Response(content="ok"))

        result = await _run_task_with_retry(task, config, _run_once)
        assert result.success
        assert call_count == 3  # initial + 2 retries

    @pytest.mark.asyncio
    async def test_exhausted_retries_returns_failure(self):
        task = FleetTask(id="t1", prompt="do it")
        config = FleetConfig(default_retries=1, default_retry_delay=0.01)

        run_once = AsyncMock(return_value=FleetResult(
            task_id="t1", success=False, error=RuntimeError("always fails"),
        ))

        result = await _run_task_with_retry(task, config, run_once)
        assert not result.success
        assert run_once.call_count == 2  # initial + 1 retry

    @pytest.mark.asyncio
    async def test_rate_limit_always_retries(self):
        """Rate limit errors continue retrying even after normal retries."""
        task = FleetTask(id="t1", prompt="do it")
        config = FleetConfig(default_retries=2, default_retry_delay=0.01)

        call_count = 0

        async def _run_once(t, attempt):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return FleetResult(
                    task_id="t1", success=False,
                    error=RuntimeError("429 rate limit exceeded"),
                )
            return FleetResult(task_id="t1", success=True, response=Response(content="ok"))

        result = await _run_task_with_retry(task, config, _run_once)
        assert result.success
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_per_task_retry_overrides_config(self):
        task = FleetTask(id="t1", prompt="do it", retries=0, retry_delay=0.01)
        config = FleetConfig(default_retries=5, default_retry_delay=0.01)

        run_once = AsyncMock(return_value=FleetResult(
            task_id="t1", success=False, error=RuntimeError("fail"),
        ))

        result = await _run_task_with_retry(task, config, run_once)
        assert not result.success
        run_once.assert_called_once()  # No retries (retries=0)

    @pytest.mark.asyncio
    async def test_adaptive_on_success_called(self):
        task = FleetTask(id="t1", prompt="do it")
        config = FleetConfig(default_retries=0)
        adaptive = AdaptiveConcurrency(initial=4)

        run_once = AsyncMock(return_value=FleetResult(
            task_id="t1", success=True, response=Response(content="ok"),
        ))

        with patch.object(adaptive, "on_success", new_callable=AsyncMock) as mock_success:
            await _run_task_with_retry(task, config, run_once, adaptive=adaptive)
            mock_success.assert_called_once()

    @pytest.mark.asyncio
    async def test_adaptive_on_rate_limit_called(self):
        task = FleetTask(id="t1", prompt="do it")
        config = FleetConfig(default_retries=0, default_retry_delay=0.01)
        adaptive = AdaptiveConcurrency(initial=4)

        run_once = AsyncMock(return_value=FleetResult(
            task_id="t1", success=False,
            error=RuntimeError("rate limit 429"),
        ))

        with patch.object(adaptive, "on_rate_limit", new_callable=AsyncMock, return_value=2):
            await _run_task_with_retry(task, config, run_once, adaptive=adaptive)
            adaptive.on_rate_limit.assert_called_once()


# ---------------------------------------------------------------------------
# TestDependencyTimeout
# ---------------------------------------------------------------------------


def _make_mock_copex(response: Response | None = None, error: Exception | None = None):
    """Return a mock Copex that acts as an async context manager."""
    mock = AsyncMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=False)
    if error:
        mock.send = AsyncMock(side_effect=error)
    else:
        mock.send = AsyncMock(return_value=response or Response(content="done"))
    return mock


class TestDependencyTimeout:
    @pytest.mark.asyncio
    async def test_dep_timeout_triggers_failure(self):
        """Task fails with clear error when dependency times out."""
        async def _slow_a(prompt, **kwargs):
            if "A" in prompt:
                await asyncio.sleep(10)  # Very slow — will be cancelled
            return Response(content="ok")

        mock_copex = _make_mock_copex()
        mock_copex.send = AsyncMock(side_effect=_slow_a)

        config = FleetConfig(dep_timeout=0.05, timeout=600.0)
        fleet = Fleet(fleet_config=config)
        fleet.add("A", task_id="a")
        fleet.add("B", task_id="b", depends_on=["a"])

        with patch("copex.fleet.CopilotClient", None), \
             patch("copex.fleet.Copex", return_value=mock_copex):
            results = await fleet.run()

        b_result = next(r for r in results if r.task_id == "b")
        assert not b_result.success
        err_msg = str(b_result.error)
        assert "timed out" in err_msg
        assert "'b'" in err_msg
        assert "'a'" in err_msg

    @pytest.mark.asyncio
    async def test_dep_timeout_zero_falls_back_to_task_timeout(self):
        """dep_timeout=0 should use the task or fleet timeout instead."""
        config = FleetConfig(dep_timeout=0.0, timeout=30.0)
        # dep_timeout=0 is falsy, so fallback is task timeout or fleet timeout
        effective = config.dep_timeout or config.timeout
        assert effective == 30.0

    @pytest.mark.asyncio
    async def test_dep_completes_before_timeout(self):
        """Task succeeds when dependency completes before timeout."""
        mock_copex = _make_mock_copex(Response(content="ok"))

        config = FleetConfig(dep_timeout=10.0)
        fleet = Fleet(fleet_config=config)
        fleet.add("A", task_id="a")
        fleet.add("B", task_id="b", depends_on=["a"])

        with patch("copex.fleet.CopilotClient", None), \
             patch("copex.fleet.Copex", return_value=mock_copex):
            results = await fleet.run()

        assert all(r.success for r in results)


# ---------------------------------------------------------------------------
# TestPrependSharedContext
# ---------------------------------------------------------------------------


class TestPrependSharedContext:
    def test_no_shared_context_returns_same(self):
        tasks = [FleetTask(id="t1", prompt="hello")]
        result = _prepend_shared_context(tasks, None)
        assert result is tasks

    def test_empty_shared_context_returns_same(self):
        tasks = [FleetTask(id="t1", prompt="hello")]
        result = _prepend_shared_context(tasks, "")
        assert result is tasks

    def test_prepends_context(self):
        tasks = [FleetTask(id="t1", prompt="hello")]
        result = _prepend_shared_context(tasks, "PREFIX")
        assert len(result) == 1
        assert result[0].prompt == "PREFIX\n\nhello"

    def test_original_not_mutated(self):
        tasks = [FleetTask(id="t1", prompt="hello")]
        _prepend_shared_context(tasks, "PREFIX")
        assert tasks[0].prompt == "hello"

    def test_preserves_all_fields(self):
        task = FleetTask(
            id="t1", prompt="hello", depends_on=["a"],
            model=None, cwd="/tmp",
        )
        result = _prepend_shared_context([task], "CTX")
        assert result[0].depends_on == ["a"]
        assert result[0].cwd == "/tmp"
        assert result[0].id == "t1"


# ---------------------------------------------------------------------------
# TestIsRateLimitError
# ---------------------------------------------------------------------------


class TestIsRateLimitError:
    def test_rate_limit_in_message(self):
        assert _is_rate_limit_error(RuntimeError("rate limit exceeded"))

    def test_429_in_message(self):
        assert _is_rate_limit_error(RuntimeError("HTTP 429 too many requests"))

    def test_too_many_requests(self):
        assert _is_rate_limit_error(RuntimeError("too many requests"))

    def test_normal_error_not_rate_limit(self):
        assert not _is_rate_limit_error(RuntimeError("connection timeout"))

    def test_status_code_attribute(self):
        exc = RuntimeError("error")
        exc.status_code = 429  # type: ignore[attr-defined]
        assert _is_rate_limit_error(exc)


# ---------------------------------------------------------------------------
# TestNormalizeMcpServers
# ---------------------------------------------------------------------------


class TestNormalizeMcpServers:
    def test_none_returns_none(self):
        assert _normalize_mcp_servers(None) is None

    def test_list_passthrough(self):
        servers = [{"name": "s1", "command": "cmd"}]
        assert _normalize_mcp_servers(servers) is servers

    def test_single_server_dict(self):
        server = {"command": "npx", "args": ["server"]}
        result = _normalize_mcp_servers(server)
        assert result == [server]

    def test_named_servers_dict(self):
        servers = {"myserver": {"command": "cmd", "args": []}}
        result = _normalize_mcp_servers(servers)
        assert len(result) == 1
        assert result[0]["name"] == "myserver"

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="list or mapping"):
            _normalize_mcp_servers("invalid")  # type: ignore[arg-type]

    def test_invalid_entry_raises(self):
        with pytest.raises(ValueError, match="mappings"):
            _normalize_mcp_servers({"bad": "not_a_dict"})


# ---------------------------------------------------------------------------
# TestFleetConfigDefaults
# ---------------------------------------------------------------------------


class TestFleetConfigDefaults:
    def test_dep_timeout_default_zero(self):
        config = FleetConfig()
        assert config.dep_timeout == 0.0

    def test_adaptive_concurrency_defaults(self):
        config = FleetConfig()
        assert config.adaptive_concurrency is True
        assert config.min_concurrent == 1
        assert config.concurrency_restore_after == 3

    def test_retry_defaults(self):
        config = FleetConfig()
        assert config.default_retries == 3
        assert config.default_retry_delay == 1.0
