"""Tests for copex.stats — StatsTracker, RunStats, and state helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from copex.stats import (
    RunStats,
    StatsTracker,
    _save_state,
    load_start_commit,
    load_state,
    save_start_commit,
)


@pytest.fixture()
def stats_dir(tmp_path: Path) -> Path:
    """Provide a temp directory for stats files."""
    return tmp_path


@pytest.fixture()
def tracker(stats_dir: Path) -> StatsTracker:
    return StatsTracker(path=stats_dir / "stats.jsonl")


# ---------------------------------------------------------------------------
# RunStats
# ---------------------------------------------------------------------------


class TestRunStats:
    def test_to_dict_roundtrip(self) -> None:
        rs = RunStats(
            timestamp="2025-01-01T00:00:00Z",
            model="gpt-5.2",
            reasoning_effort="high",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )
        d = rs.to_dict()
        assert d["model"] == "gpt-5.2"
        assert d["total_tokens"] == 30
        assert d["success"] is True

    def test_defaults(self) -> None:
        rs = RunStats(timestamp="t", model="m", reasoning_effort="none")
        assert rs.prompt_tokens == 0
        assert rs.error is None
        assert rs.command == ""


# ---------------------------------------------------------------------------
# StatsTracker — write/read
# ---------------------------------------------------------------------------


class TestStatsTracker:
    def test_log_and_read(self, tracker: StatsTracker) -> None:
        rs = RunStats(timestamp="2025-01-01T00:00:00Z", model="m", reasoning_effort="h")
        tracker.log(rs)
        runs = tracker.all_runs()
        assert len(runs) == 1
        assert runs[0]["model"] == "m"

    def test_all_runs_empty(self, tracker: StatsTracker) -> None:
        assert tracker.all_runs() == []

    def test_last_run_none(self, tracker: StatsTracker) -> None:
        assert tracker.last_run() is None

    def test_last_run_returns_latest(self, tracker: StatsTracker) -> None:
        for i in range(3):
            tracker.log(
                RunStats(timestamp=f"2025-01-0{i+1}T00:00:00Z", model=f"m{i}", reasoning_effort="h")
            )
        last = tracker.last_run()
        assert last is not None and last["model"] == "m2"

    def test_runs_today_filters(self, tracker: StatsTracker) -> None:
        tracker.log(RunStats(timestamp="2020-01-01T00:00:00Z", model="old", reasoning_effort="h"))
        # Won't match today unless today is 2020-01-01
        today = tracker.runs_today()
        assert all(r["model"] != "old" for r in today)

    def test_malformed_jsonl_skipped(self, tracker: StatsTracker) -> None:
        tracker.path.write_text("not-json\n")
        assert tracker.all_runs() == []

    def test_log_updates_state(self, tracker: StatsTracker, stats_dir: Path) -> None:
        # Patch _state_path to use our temp dir
        import copex.stats as stats_mod

        original = stats_mod._state_path
        stats_mod._state_path = lambda: stats_dir / "state.json"
        try:
            rs = RunStats(timestamp="t", model="m", reasoning_effort="h")
            tracker.log(rs)
            state = load_state()
        finally:
            stats_mod._state_path = original


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------


class TestStateHelpers:
    def test_save_and_load_start_commit(self, stats_dir: Path) -> None:
        import copex.stats as stats_mod

        original = stats_mod._state_path
        stats_mod._state_path = lambda: stats_dir / "state.json"
        try:
            save_start_commit("abc123")
            assert load_start_commit() == "abc123"
        finally:
            stats_mod._state_path = original

    def test_load_start_commit_missing_file(self, stats_dir: Path) -> None:
        import copex.stats as stats_mod

        original = stats_mod._state_path
        stats_mod._state_path = lambda: stats_dir / "nonexistent" / "state.json"
        try:
            assert load_start_commit() is None
        finally:
            stats_mod._state_path = original

    def test_load_state_corrupt_file(self, stats_dir: Path) -> None:
        import copex.stats as stats_mod

        original = stats_mod._state_path
        state_path = stats_dir / "state.json"
        state_path.write_text("{bad json")
        stats_mod._state_path = lambda: state_path
        try:
            assert load_state() == {}
        finally:
            stats_mod._state_path = original

    def test_save_state_creates_parent_dirs(self, tmp_path: Path) -> None:
        import copex.stats as stats_mod

        original = stats_mod._state_path
        state_path = tmp_path / "nested" / "deep" / "state.json"
        stats_mod._state_path = lambda: state_path
        try:
            _save_state({"key": "value"})
            assert state_path.exists()
            data = json.loads(state_path.read_text())
            assert data["key"] == "value"
        finally:
            stats_mod._state_path = original
