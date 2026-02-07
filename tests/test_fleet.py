from __future__ import annotations

import asyncio

import pytest

from copex.client import Response
from copex.config import CopexConfig
from copex.fleet import (
    Fleet,
    FleetConfig,
    FleetContext,
    FleetCoordinator,
    FleetEvent,
    FleetEventType,
    FleetMailbox,
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

        with patch("copex.fleet.CopilotClient", None), \
             patch("copex.fleet.Copex", return_value=mock_copex):
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

        with patch("copex.fleet.CopilotClient", None), \
             patch("copex.fleet.Copex", return_value=mock_copex):
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

        with patch("copex.fleet.CopilotClient", None), \
             patch("copex.fleet.Copex", return_value=mock_copex):
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

        with patch("copex.fleet.CopilotClient", None), \
             patch("copex.fleet.Copex", return_value=mock_copex):
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

        with patch("copex.fleet.CopilotClient", None), \
             patch("copex.fleet.Copex", return_value=mock_copex):
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

        with patch("copex.fleet.CopilotClient", None), \
             patch("copex.fleet.Copex", return_value=mock_copex):
            await fleet.run()

        sent_prompt = mock_copex.send.call_args[0][0]
        assert sent_prompt.startswith(shared)
        assert "Write tests" in sent_prompt

    @pytest.mark.asyncio
    async def test_run_empty_returns_empty_list(self):
        fleet = Fleet()
        results = await fleet.run()
        assert results == []


# ---------------------------------------------------------------------------
# TestFleetMailbox
# ---------------------------------------------------------------------------


class TestFleetMailbox:
    @pytest.mark.asyncio
    async def test_send_and_receive(self):
        mbox = FleetMailbox()
        mbox.create_inbox("task-a")

        ok = await mbox.send("task-a", {"key": "value"}, from_task="task-b")
        assert ok is True

        msg = await mbox.receive("task-a", timeout=1.0)
        assert msg is not None
        assert msg["payload"] == {"key": "value"}
        assert msg["from"] == "task-b"
        assert msg["to"] == "task-a"

    @pytest.mark.asyncio
    async def test_send_to_missing_inbox_returns_false(self):
        mbox = FleetMailbox()
        ok = await mbox.send("nonexistent", {"data": 1})
        assert ok is False

    @pytest.mark.asyncio
    async def test_receive_timeout_returns_none(self):
        mbox = FleetMailbox()
        mbox.create_inbox("task-a")

        msg = await mbox.receive("task-a", timeout=0.05)
        assert msg is None

    @pytest.mark.asyncio
    async def test_broadcast_sends_to_all_except_excluded(self):
        mbox = FleetMailbox()
        mbox.create_inbox("t1")
        mbox.create_inbox("t2")
        mbox.create_inbox("t3")

        count = await mbox.broadcast(
            {"alert": True},
            from_task="t1",
            exclude=["t1"],
        )
        assert count == 2

        # t2 and t3 should have messages, t1 should not
        assert mbox.pending_count("t1") == 0
        assert mbox.pending_count("t2") == 1
        assert mbox.pending_count("t3") == 1

        msg_t2 = await mbox.receive("t2", timeout=0.1)
        assert msg_t2 is not None
        assert msg_t2["payload"] == {"alert": True}

    def test_try_receive_empty_returns_none(self):
        mbox = FleetMailbox()
        mbox.create_inbox("t1")
        assert mbox.try_receive("t1") is None

    def test_remove_inbox(self):
        mbox = FleetMailbox()
        mbox.create_inbox("t1")
        mbox.remove_inbox("t1")
        assert mbox.pending_count("t1") == 0

    @pytest.mark.asyncio
    async def test_receive_from_missing_inbox_returns_none(self):
        mbox = FleetMailbox()
        msg = await mbox.receive("ghost", timeout=0.05)
        assert msg is None


# ---------------------------------------------------------------------------
# TestFleetContext
# ---------------------------------------------------------------------------


class TestFleetContext:
    @pytest.mark.asyncio
    async def test_set_and_get(self):
        ctx = FleetContext()
        await ctx.set("key", 42)
        assert await ctx.get("key") == 42

    @pytest.mark.asyncio
    async def test_get_default(self):
        ctx = FleetContext()
        assert await ctx.get("missing", "fallback") == "fallback"

    @pytest.mark.asyncio
    async def test_initial_state(self):
        ctx = FleetContext(initial_state={"x": 1})
        assert await ctx.get("x") == 1

    @pytest.mark.asyncio
    async def test_add_and_get_result(self):
        ctx = FleetContext()
        await ctx.add_result("task-1", {"status": "ok"})
        r = await ctx.get_result("task-1")
        assert r == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_get_results_returns_copy(self):
        ctx = FleetContext()
        await ctx.add_result("a", 1)
        await ctx.add_result("b", 2)
        results = await ctx.get_results()
        assert results == {"a": 1, "b": 2}
        # Mutating the copy doesn't affect the context
        results["c"] = 3
        assert await ctx.get_result("c") is None

    @pytest.mark.asyncio
    async def test_delete_key(self):
        ctx = FleetContext()
        await ctx.set("temp", "value")
        deleted = await ctx.delete("temp")
        assert deleted is True
        assert await ctx.get("temp") is None
        assert await ctx.delete("temp") is False

    @pytest.mark.asyncio
    async def test_update_multiple(self):
        ctx = FleetContext()
        await ctx.update({"a": 1, "b": 2})
        assert await ctx.get("a") == 1
        assert await ctx.get("b") == 2

    @pytest.mark.asyncio
    async def test_aggregate_results(self):
        ctx = FleetContext()
        await ctx.add_result("t1", 10)
        await ctx.add_result("t2", 20)
        total = await ctx.aggregate_results(lambda r: sum(r.values()))
        assert total == 30

    def test_clear(self):
        ctx = FleetContext(initial_state={"k": "v"})
        ctx.add_result_sync("t1", "r1")
        ctx.clear()
        assert ctx.state == {}
        assert ctx.results == {}

    def test_sync_accessors(self):
        ctx = FleetContext()
        ctx.set_sync("key", "val")
        assert ctx.get_sync("key") == "val"
        ctx.add_result_sync("t1", "done")
        assert ctx.get_result_sync("t1") == "done"
        assert ctx.get_results_sync() == {"t1": "done"}


# ---------------------------------------------------------------------------
# TestRunStreaming
# ---------------------------------------------------------------------------


class TestRunStreaming:
    @pytest.mark.asyncio
    async def test_empty_fleet_yields_fleet_complete(self):
        fleet = Fleet()
        events = [e async for e in fleet.run_streaming()]
        assert len(events) == 1
        assert events[0].event_type == FleetEventType.FLEET_COMPLETE
        assert events[0].data["total_tasks"] == 0

    @pytest.mark.asyncio
    async def test_single_task_event_ordering(self):
        mock_copex = _make_mock_copex(Response(content="result"))
        fleet = Fleet()
        fleet.add("Do something", task_id="t1")

        with patch("copex.fleet.Copex", return_value=mock_copex):
            events = [e async for e in fleet.run_streaming()]

        event_types = [e.event_type for e in events]
        # Must start with FLEET_START and end with FLEET_COMPLETE
        assert event_types[0] == FleetEventType.FLEET_START
        assert event_types[-1] == FleetEventType.FLEET_COMPLETE

        # Task lifecycle events must appear in order
        task_events = [e for e in events if e.task_id == "t1"]
        task_types = [e.event_type for e in task_events]
        assert task_types.index(FleetEventType.TASK_QUEUED) < task_types.index(FleetEventType.TASK_RUNNING)
        assert task_types.index(FleetEventType.TASK_RUNNING) < task_types.index(FleetEventType.TASK_DONE)

    @pytest.mark.asyncio
    async def test_streaming_failure_emits_task_failed(self):
        mock_copex = _make_mock_copex(error=RuntimeError("kaboom"))
        fleet = Fleet()
        fleet.add("Fail task", task_id="f1")

        with patch("copex.fleet.Copex", return_value=mock_copex):
            events = [e async for e in fleet.run_streaming()]

        failed = [e for e in events if e.event_type == FleetEventType.TASK_FAILED]
        assert len(failed) == 1
        assert failed[0].task_id == "f1"
        assert "kaboom" in (failed[0].error or "")

        # Fleet complete should report 0 succeeded, 1 failed
        complete = [e for e in events if e.event_type == FleetEventType.FLEET_COMPLETE]
        assert complete[0].data["succeeded"] == 0
        assert complete[0].data["failed"] == 1

    @pytest.mark.asyncio
    async def test_streaming_blocked_dependency(self):
        async def _fail_on_a(prompt, **kwargs):
            if "A" in prompt:
                raise RuntimeError("A failed")
            return Response(content="ok")

        mock_copex = _make_mock_copex()
        mock_copex.send = AsyncMock(side_effect=_fail_on_a)

        fleet = Fleet()
        fleet.add("A", task_id="a")
        fleet.add("B", task_id="b", depends_on=["a"])

        with patch("copex.fleet.Copex", return_value=mock_copex):
            events = [e async for e in fleet.run_streaming()]

        blocked = [e for e in events if e.event_type == FleetEventType.TASK_BLOCKED]
        assert len(blocked) == 1
        assert blocked[0].task_id == "b"

    @pytest.mark.asyncio
    async def test_streaming_stores_results_in_context(self):
        mock_copex = _make_mock_copex(Response(content="hello world"))
        ctx = FleetContext()

        fleet = Fleet()
        fleet.add("Task one", task_id="t1")

        with patch("copex.fleet.Copex", return_value=mock_copex):
            events = [e async for e in fleet.run_streaming(context=ctx)]

        # Context should have the task result
        result = await ctx.get_result("t1")
        assert result is not None
        assert result["success"] is True
        assert result["content"] == "hello world"

    @pytest.mark.asyncio
    async def test_streaming_creates_mailbox_inboxes(self):
        mock_copex = _make_mock_copex()
        mbox = FleetMailbox()

        fleet = Fleet()
        fleet.add("T1", task_id="t1")
        fleet.add("T2", task_id="t2")

        with patch("copex.fleet.Copex", return_value=mock_copex):
            _ = [e async for e in fleet.run_streaming(mailbox=mbox)]

        # Mailbox inboxes should have been created for both tasks
        assert mbox.pending_count("t1") == 0  # exists but empty
        assert mbox.pending_count("t2") == 0


# ---------------------------------------------------------------------------
# TestFleetPromptImmutability
# ---------------------------------------------------------------------------


class TestFleetPromptImmutability:
    @pytest.mark.asyncio
    async def test_shared_context_does_not_mutate_tasks(self):
        """Shared context should be prepended without modifying original tasks."""
        shared = "Context prefix."
        cfg = FleetConfig(shared_context=shared)
        mock_copex = _make_mock_copex()

        fleet = Fleet(fleet_config=cfg)
        fleet.add("Original prompt", task_id="t1")

        # Save original prompt
        original_prompt = fleet._tasks[0].prompt
        assert original_prompt == "Original prompt"

        with patch("copex.fleet.CopilotClient", None), \
             patch("copex.fleet.Copex", return_value=mock_copex):
            await fleet.run()

        # Task prompt should NOT have been modified
        assert fleet._tasks[0].prompt == "Original prompt"

        # But the sent prompt should have the context
        sent_prompt = mock_copex.send.call_args[0][0]
        assert sent_prompt.startswith(shared)

    @pytest.mark.asyncio
    async def test_double_run_no_double_prepend(self):
        """Calling run() twice should not double-prepend shared context."""
        shared = "PREFIX"
        cfg = FleetConfig(shared_context=shared)
        mock_copex = _make_mock_copex()

        fleet = Fleet(fleet_config=cfg)
        fleet.add("My prompt", task_id="t1")

        with patch("copex.fleet.CopilotClient", None), \
             patch("copex.fleet.Copex", return_value=mock_copex):
            await fleet.run()
            # Reset mock for second call
            mock_copex.send.reset_mock()
            await fleet.run()

        sent_prompt = mock_copex.send.call_args[0][0]
        # Should start with PREFIX exactly once, not "PREFIX\n\nPREFIX\n\n..."
        assert sent_prompt == "PREFIX\n\nMy prompt"
        # Count occurrences of PREFIX
        assert sent_prompt.count("PREFIX") == 1

    @pytest.mark.asyncio
    async def test_streaming_shared_context_does_not_mutate_tasks(self):
        """Streaming run should also not mutate original tasks."""
        shared = "Stream context."
        cfg = FleetConfig(shared_context=shared)
        mock_copex = _make_mock_copex()

        fleet = Fleet(fleet_config=cfg)
        fleet.add("Stream prompt", task_id="s1")

        original_prompt = fleet._tasks[0].prompt

        with patch("copex.fleet.Copex", return_value=mock_copex):
            _ = [e async for e in fleet.run_streaming()]

        assert fleet._tasks[0].prompt == original_prompt


# ---------------------------------------------------------------------------
# TestFleetDepErrorMessage
# ---------------------------------------------------------------------------


class TestFleetDepErrorMessage:
    @pytest.mark.asyncio
    async def test_streaming_dep_error_matches_format(self):
        """Streaming path should use same error format as non-streaming."""
        async def _fail_on_a(prompt, **kwargs):
            if "A" in prompt:
                raise RuntimeError("A failed")
            return Response(content="ok")

        mock_copex = _make_mock_copex()
        mock_copex.send = AsyncMock(side_effect=_fail_on_a)

        fleet = Fleet()
        fleet.add("A", task_id="a")
        fleet.add("B", task_id="b", depends_on=["a"])

        with patch("copex.fleet.Copex", return_value=mock_copex):
            events = [e async for e in fleet.run_streaming()]

        blocked = [e for e in events if e.event_type == FleetEventType.TASK_BLOCKED]
        assert len(blocked) == 1

        # Error message should include task ID and upstream info
        error_msg = blocked[0].error
        assert "b" in error_msg

        # The FleetResult for the blocked task should have the
        # "Dependency failed" error with upstream details
        failed_events = [e for e in events if e.event_type == FleetEventType.TASK_FAILED and e.task_id == "a"]
        assert len(failed_events) == 1
        assert "A failed" in (failed_events[0].error or "")
