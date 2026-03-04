"""Tests for copex.progress — ProgressReporter and PlanProgressReporter."""

from __future__ import annotations

import io
import json
import time

import pytest

from copex.progress import (
    PlanProgressReporter,
    ProgressItem,
    ProgressReporter,
    ProgressState,
    ProgressStatus,
)


# ---------------------------------------------------------------------------
# ProgressItem
# ---------------------------------------------------------------------------


class TestProgressItem:
    def test_duration_not_started(self) -> None:
        item = ProgressItem(id="a", description="test")
        assert item.duration is None

    def test_duration_running(self) -> None:
        item = ProgressItem(id="a", description="test", started_at=time.time() - 2.0)
        dur = item.duration
        assert dur is not None and dur >= 1.5

    def test_duration_completed(self) -> None:
        now = time.time()
        item = ProgressItem(
            id="a", description="test", started_at=now - 5.0, completed_at=now - 1.0
        )
        assert item.duration == pytest.approx(4.0, abs=0.1)


# ---------------------------------------------------------------------------
# ProgressState
# ---------------------------------------------------------------------------


class TestProgressState:
    def test_percent_zero_total(self) -> None:
        state = ProgressState(total=0)
        assert state.percent == 0.0

    def test_percent_partial(self) -> None:
        state = ProgressState(total=4, completed=1)
        assert state.percent == 25.0

    def test_eta_no_completed(self) -> None:
        state = ProgressState(total=5, completed=0)
        assert state.eta_seconds is None

    def test_eta_with_progress(self) -> None:
        state = ProgressState(total=10, completed=5, started_at=time.time() - 10.0)
        eta = state.eta_seconds
        assert eta is not None and eta > 0

    def test_to_dict_keys(self) -> None:
        state = ProgressState(total=2)
        d = state.to_dict()
        assert set(d.keys()) == {
            "total",
            "completed",
            "failed",
            "running",
            "percent",
            "elapsed_seconds",
            "eta_seconds",
            "message",
            "items",
        }


# ---------------------------------------------------------------------------
# ProgressReporter — lifecycle
# ---------------------------------------------------------------------------


class TestProgressReporterLifecycle:
    def test_start_complete_item(self) -> None:
        r = ProgressReporter(total=3, format="quiet")
        r.start_item(1, "step-1")
        assert r.state.running == 1

        r.complete_item(1)
        assert r.state.running == 0
        assert r.state.completed == 1

    def test_start_fail_item(self) -> None:
        r = ProgressReporter(total=3, format="quiet")
        r.start_item(1, "step-1")
        r.fail_item(1, "boom")
        assert r.state.running == 0
        assert r.state.failed == 1
        item = r._get_item(1)
        assert item is not None and item.error == "boom"

    def test_skip_running_item_decrements_running(self) -> None:
        """Regression: skip_item must decrement running for RUNNING items."""
        r = ProgressReporter(total=3, format="quiet")
        r.start_item(1, "step-1")
        assert r.state.running == 1

        r.skip_item(1, reason="not needed")
        assert r.state.running == 0
        assert r.state.completed == 1
        item = r._get_item(1)
        assert item is not None and item.status == ProgressStatus.SKIPPED
        assert item.metadata.get("skip_reason") == "not needed"

    def test_skip_pending_item_does_not_touch_running(self) -> None:
        r = ProgressReporter(total=3, format="quiet")
        r.add_item(1, "step-1")
        r.skip_item(1)
        assert r.state.running == 0
        assert r.state.completed == 1

    def test_skip_already_completed_is_noop(self) -> None:
        r = ProgressReporter(total=3, format="quiet")
        r.start_item(1, "step-1")
        r.complete_item(1)
        r.skip_item(1)
        # completed count stays 1 (not incremented again)
        assert r.state.completed == 1

    def test_complete_already_completed_is_noop(self) -> None:
        r = ProgressReporter(total=2, format="quiet")
        r.start_item(1, "step-1")
        r.complete_item(1)
        r.complete_item(1)
        assert r.state.completed == 1

    def test_fail_already_failed_is_noop(self) -> None:
        r = ProgressReporter(total=2, format="quiet")
        r.start_item(1, "step-1")
        r.fail_item(1, "err")
        r.fail_item(1, "err again")
        assert r.state.failed == 1


# ---------------------------------------------------------------------------
# ProgressReporter — output formats
# ---------------------------------------------------------------------------


class TestProgressReporterOutput:
    def test_terminal_output(self) -> None:
        buf = io.StringIO()
        r = ProgressReporter(total=1, output=buf, format="terminal")
        r.start_item(0, "step-0")
        r.complete_item(0)
        r.finish("done")
        text = buf.getvalue()
        assert "Progress" in text
        assert "100%" in text

    def test_json_output(self) -> None:
        buf = io.StringIO()
        r = ProgressReporter(total=2, output=buf, format="json")
        r.start_item(0, "step-0")
        r.complete_item(0)
        lines = [l for l in buf.getvalue().strip().split("\n") if l]
        for line in lines:
            obj = json.loads(line)
            assert obj["type"] == "progress"

    def test_quiet_produces_no_output(self) -> None:
        buf = io.StringIO()
        r = ProgressReporter(total=1, output=buf, format="quiet")
        r.start_item(0, "step-0")
        r.complete_item(0)
        assert buf.getvalue() == ""

    def test_callback_on_update(self) -> None:
        updates: list[ProgressState] = []
        r = ProgressReporter(total=2, format="quiet", on_update=lambda s: updates.append(s))
        r.start_item(0, "step-0")
        r.complete_item(0)
        assert len(updates) >= 2


# ---------------------------------------------------------------------------
# ProgressReporter — misc
# ---------------------------------------------------------------------------


class TestProgressReporterMisc:
    def test_set_total(self) -> None:
        r = ProgressReporter(format="quiet")
        r.set_total(10)
        assert r.state.total == 10

    def test_set_message(self) -> None:
        r = ProgressReporter(format="quiet")
        r.set_message("hello")
        assert r.state.message == "hello"

    def test_get_or_create_item(self) -> None:
        r = ProgressReporter(total=1, format="quiet")
        r.start_item("auto", "auto-created")
        assert r._get_item("auto") is not None

    def test_format_time_seconds(self) -> None:
        assert ProgressReporter._format_time(30) == "30s"

    def test_format_time_minutes(self) -> None:
        assert ProgressReporter._format_time(90) == "1m 30s"

    def test_format_time_hours(self) -> None:
        assert ProgressReporter._format_time(3700) == "1h 1m"

    def test_format_time_none(self) -> None:
        assert ProgressReporter._format_time(None) == "?"


# ---------------------------------------------------------------------------
# PlanProgressReporter
# ---------------------------------------------------------------------------


class _FakePlan:
    """Minimal plan stub for PlanProgressReporter."""

    def __init__(self, n: int) -> None:
        self.steps = [_FakeStep(i + 1) for i in range(n)]

    def __len__(self) -> int:
        return len(self.steps)


class _FakeStep:
    def __init__(self, number: int) -> None:
        self.number = number
        self.description = f"Step {number}"
        self.result: str | None = None


class TestPlanProgressReporter:
    def test_pre_populates_items(self) -> None:
        plan = _FakePlan(3)
        r = PlanProgressReporter(plan, format="quiet")
        assert r.state.total == 3
        assert len(r.state.items) == 3

    def test_step_callbacks(self) -> None:
        plan = _FakePlan(2)
        r = PlanProgressReporter(plan, format="quiet")
        r.on_step_start(plan.steps[0])
        assert r.state.running == 1

        plan.steps[0].result = "OK"
        r.on_step_complete(plan.steps[0])
        assert r.state.completed == 1
        assert r.state.running == 0

    def test_on_error_callback(self) -> None:
        plan = _FakePlan(1)
        r = PlanProgressReporter(plan, format="quiet")
        r.on_step_start(plan.steps[0])
        cont = r.on_error(plan.steps[0], RuntimeError("oops"))
        assert cont is True  # continues by default
        assert r.state.failed == 1
