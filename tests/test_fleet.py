from __future__ import annotations

import asyncio

import pytest

from copex.client import Response
from copex.config import CopexConfig
from copex.fleet import (
    Fleet,
    FleetConfig,
    FleetCoordinator,
    FleetResult,
    FleetTask,
    _slugify,
)
from unittest.mock import AsyncMock, patch


# ---------------------------------------------------------------------------
# TestSlugify
# ---------------------------------------------------------------------------


class TestSlugify:
    def test_basic_slugification(self):
        assert _slugify("Write auth tests") == "write-auth-tests"

    def test_special_characters_removed(self):
        assert _slugify("Hello, World! @#$%") == "hello-world"

    def test_truncation_at_64_chars(self):
        long = "a" * 100
        result = _slugify(long)
        assert len(result) == 64

    def test_empty_input(self):
        assert _slugify("") == "task"
        assert _slugify("   ") == "task"
        assert _slugify("!!!") == "task"


# ---------------------------------------------------------------------------
# TestFleetDAGValidation
# ---------------------------------------------------------------------------


class TestFleetDAGValidation:
    def test_valid_dag_passes(self):
        tasks = [
            FleetTask(id="a", prompt="A"),
            FleetTask(id="b", prompt="B", depends_on=["a"]),
            FleetTask(id="c", prompt="C", depends_on=["b"]),
        ]
        FleetCoordinator._validate_dag(tasks)  # should not raise

    def test_cycle_detection_raises(self):
        tasks = [
            FleetTask(id="a", prompt="A", depends_on=["b"]),
            FleetTask(id="b", prompt="B", depends_on=["a"]),
        ]
        with pytest.raises(ValueError, match="Cycle detected"):
            FleetCoordinator._validate_dag(tasks)

    def test_unknown_dependency_raises(self):
        tasks = [
            FleetTask(id="a", prompt="A", depends_on=["nonexistent"]),
        ]
        with pytest.raises(ValueError, match="unknown task 'nonexistent'"):
            FleetCoordinator._validate_dag(tasks)

    def test_self_dependency_detected_as_cycle(self):
        tasks = [
            FleetTask(id="a", prompt="A", depends_on=["a"]),
        ]
        with pytest.raises(ValueError, match="Cycle detected"):
            FleetCoordinator._validate_dag(tasks)


# ---------------------------------------------------------------------------
# Helpers for mocking Copex
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


# ---------------------------------------------------------------------------
# TestFleetCoordinator
# ---------------------------------------------------------------------------


class TestFleetCoordinator:
    @pytest.mark.asyncio
    async def test_independent_tasks_run_in_parallel(self):
        mock_copex = _make_mock_copex()
        coord = FleetCoordinator(CopexConfig())
        tasks = [
            FleetTask(id="a", prompt="A"),
            FleetTask(id="b", prompt="B"),
            FleetTask(id="c", prompt="C"),
        ]

        with patch("copex.fleet.Copex", return_value=mock_copex):
            results = await coord.run(tasks)

        assert len(results) == 3
        assert all(r.success for r in results)
        assert mock_copex.send.call_count == 3

    @pytest.mark.asyncio
    async def test_dependency_ordering(self):
        execution_order: list[str] = []
        original_response = Response(content="ok")

        async def _tracking_send(prompt, **kwargs):
            execution_order.append(prompt)
            return original_response

        mock_copex = _make_mock_copex()
        mock_copex.send = AsyncMock(side_effect=_tracking_send)

        tasks = [
            FleetTask(id="first", prompt="first"),
            FleetTask(id="second", prompt="second", depends_on=["first"]),
        ]
        coord = FleetCoordinator(CopexConfig())

        with patch("copex.fleet.Copex", return_value=mock_copex):
            results = await coord.run(tasks)

        assert results[0].success
        assert results[1].success
        assert execution_order.index("first") < execution_order.index("second")

    @pytest.mark.asyncio
    async def test_fail_fast_cancels_remaining(self):
        call_count = 0

        async def _fail_then_succeed(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            if prompt == "fail-me":
                raise RuntimeError("boom")
            return Response(content="ok")

        mock_copex = _make_mock_copex()
        mock_copex.send = AsyncMock(side_effect=_fail_then_succeed)

        tasks = [
            FleetTask(id="fail-task", prompt="fail-me"),
            FleetTask(id="dep-task", prompt="depends", depends_on=["fail-task"]),
        ]
        coord = FleetCoordinator(CopexConfig())
        config = FleetConfig(fail_fast=True)

        with patch("copex.fleet.Copex", return_value=mock_copex):
            results = await coord.run(tasks, config=config)

        assert not results[0].success
        assert not results[1].success

    @pytest.mark.asyncio
    async def test_blocked_tasks_get_error(self):
        async def _fail_on_a(prompt, **kwargs):
            if prompt == "A":
                raise RuntimeError("A failed")
            return Response(content="ok")

        mock_copex = _make_mock_copex()
        mock_copex.send = AsyncMock(side_effect=_fail_on_a)

        tasks = [
            FleetTask(id="a", prompt="A"),
            FleetTask(id="b", prompt="B", depends_on=["a"]),
        ]
        coord = FleetCoordinator(CopexConfig())

        with patch("copex.fleet.Copex", return_value=mock_copex):
            results = await coord.run(tasks)

        assert not results[0].success
        assert not results[1].success
        err_msg = str(results[1].error)
        assert "Dependency failed" in err_msg
        assert "'a'" in err_msg

    @pytest.mark.asyncio
    async def test_on_status_callback(self):
        mock_copex = _make_mock_copex()
        statuses: list[tuple[str, str]] = []

        def _on_status(task_id: str, status: str):
            statuses.append((task_id, status))

        tasks = [FleetTask(id="t1", prompt="hello")]
        coord = FleetCoordinator(CopexConfig())

        with patch("copex.fleet.Copex", return_value=mock_copex):
            await coord.run(tasks, on_status=_on_status)

        task_statuses = [s for tid, s in statuses if tid == "t1"]
        assert "queued" in task_statuses
        assert "running" in task_statuses
        assert "done" in task_statuses


# ---------------------------------------------------------------------------
# TestFleet
# ---------------------------------------------------------------------------


class TestFleet:
    def test_add_auto_generates_task_id(self):
        fleet = Fleet()
        tid = fleet.add("Write auth tests")
        assert tid == "write-auth-tests"

    def test_add_deduplicates_ids(self):
        fleet = Fleet()
        tid1 = fleet.add("Write tests")
        tid2 = fleet.add("Write tests")
        assert tid1 != tid2
        assert tid2 == "write-tests-2"

    def test_add_with_explicit_task_id(self):
        fleet = Fleet()
        tid = fleet.add("Do something", task_id="custom-id")
        assert tid == "custom-id"

    @pytest.mark.asyncio
    async def test_shared_context_prepended(self):
        shared = "You are a Python expert."
        cfg = FleetConfig(shared_context=shared)
        mock_copex = _make_mock_copex()

        fleet = Fleet(fleet_config=cfg)
        fleet.add("Write tests", task_id="t1")

        with patch("copex.fleet.Copex", return_value=mock_copex):
            await fleet.run()

        sent_prompt = mock_copex.send.call_args[0][0]
        assert sent_prompt.startswith(shared)
        assert "Write tests" in sent_prompt

    @pytest.mark.asyncio
    async def test_run_empty_returns_empty_list(self):
        fleet = Fleet()
        results = await fleet.run()
        assert results == []
