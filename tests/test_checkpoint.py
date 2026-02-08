from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from copex.checkpoint import Checkpoint, CheckpointedRalph, CheckpointStore


def _save_checkpoint(store: CheckpointStore, checkpoint_id: str, **overrides) -> Checkpoint:
    """Helper to save a Checkpoint with a deterministic ID."""
    now = datetime.now().isoformat()
    defaults = dict(
        checkpoint_id=checkpoint_id,
        loop_id="loop",
        prompt="p",
        iteration=0,
        max_iterations=None,
        completion_promise=None,
        created_at=now,
        updated_at=now,
        started_at=now,
    )
    defaults.update(overrides)
    cp = Checkpoint(**defaults)
    store._save(cp)
    return cp


# ---------------------------------------------------------------------------
# Checkpoint dataclass
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_to_dict_roundtrip(self):
        cp = Checkpoint(
            checkpoint_id="id1",
            loop_id="loop1",
            prompt="do stuff",
            iteration=3,
            max_iterations=10,
            completion_promise="DONE",
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:01:00",
            started_at="2024-01-01T00:00:00",
            history=["resp1", "resp2"],
            metadata={"key": "val"},
        )
        d = cp.to_dict()
        restored = Checkpoint.from_dict(d)
        assert restored.checkpoint_id == cp.checkpoint_id
        assert restored.history == cp.history
        assert restored.metadata == cp.metadata

    def test_defaults(self):
        cp = Checkpoint(
            checkpoint_id="id",
            loop_id="l",
            prompt="p",
            iteration=0,
            max_iterations=None,
            completion_promise=None,
            created_at="t",
            updated_at="t",
            started_at="t",
        )
        assert cp.completed is False
        assert cp.history == []
        assert cp.metadata == {}
        assert cp.model == "gpt-5.2-codex"
        assert cp.reasoning_effort == "xhigh"


# ---------------------------------------------------------------------------
# CheckpointStore
# ---------------------------------------------------------------------------


class TestCheckpointStore:
    def test_creates_base_dir(self, tmp_path: Path):
        target = tmp_path / "sub" / "checkpoints"
        store = CheckpointStore(base_dir=target)
        assert target.is_dir()
        assert store.base_dir == target

    def test_create_saves_file(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        cp = store.create("myloop", "prompt text", max_iterations=5)
        assert cp.loop_id == "myloop"
        assert cp.prompt == "prompt text"
        assert cp.iteration == 0
        assert cp.max_iterations == 5
        path = store._checkpoint_path(cp.checkpoint_id)
        assert path.exists()

    def test_load_existing(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        cp = store.create("loop", "p")
        loaded = store.load(cp.checkpoint_id)
        assert loaded is not None
        assert loaded.checkpoint_id == cp.checkpoint_id
        assert loaded.prompt == "p"

    def test_load_nonexistent_returns_none(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        assert store.load("does-not-exist") is None

    def test_update_checkpoint(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        cp = store.create("loop", "p")
        updated = store.update(
            cp.checkpoint_id,
            iteration=3,
            history=["a", "b", "c"],
            completed=True,
            completion_reason="promise",
        )
        assert updated is not None
        assert updated.iteration == 3
        assert updated.history == ["a", "b", "c"]
        assert updated.completed is True
        assert updated.completion_reason == "promise"

    def test_update_nonexistent_returns_none(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        assert store.update("nope", iteration=1) is None

    def test_update_metadata_merges(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        cp = store.create("loop", "p", metadata={"a": 1})
        store.update(cp.checkpoint_id, metadata={"b": 2})
        loaded = store.load(cp.checkpoint_id)
        assert loaded.metadata == {"a": 1, "b": 2}

    def test_overwrite_existing_checkpoint(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        cp = store.create("loop", "p")
        store.update(cp.checkpoint_id, iteration=5)
        store.update(cp.checkpoint_id, iteration=10)
        loaded = store.load(cp.checkpoint_id)
        assert loaded.iteration == 10

    def test_list_checkpoints_empty(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        assert store.list_checkpoints() == []

    def test_list_checkpoints(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        _save_checkpoint(store, "cp-a", loop_id="loop1")
        _save_checkpoint(store, "cp-b", loop_id="loop2")
        items = store.list_checkpoints()
        assert len(items) == 2

    def test_list_checkpoints_filter_by_loop_id(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        _save_checkpoint(store, "cp-a", loop_id="loop1")
        _save_checkpoint(store, "cp-b", loop_id="loop2")
        items = store.list_checkpoints(loop_id="loop1")
        assert len(items) == 1
        assert items[0]["loop_id"] == "loop1"

    def test_corrupt_json_skipped_in_list(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        store.create("loop", "p")
        (tmp_path / "corrupt.json").write_text("{invalid json", encoding="utf-8")
        items = store.list_checkpoints()
        assert len(items) == 1

    def test_corrupt_json_skipped_in_get_incomplete(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        store.create("loop", "p")
        (tmp_path / "bad.json").write_text("not json", encoding="utf-8")
        incomplete = store.get_incomplete()
        assert len(incomplete) == 1

    def test_delete_existing(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        cp = store.create("loop", "p")
        assert store.delete(cp.checkpoint_id) is True
        assert store.load(cp.checkpoint_id) is None

    def test_delete_nonexistent(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        assert store.delete("nope") is False

    def test_get_latest(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        _save_checkpoint(store, "cp-old", loop_id="loop", updated_at="2024-01-01T00:00:00")
        _save_checkpoint(store, "cp-new", loop_id="loop", updated_at="2024-06-01T00:00:00")
        latest = store.get_latest("loop")
        assert latest is not None
        assert latest.checkpoint_id == "cp-new"

    def test_get_latest_returns_none_when_empty(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        assert store.get_latest("noloop") is None

    def test_get_incomplete_filters_completed(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        _save_checkpoint(store, "cp-done", loop_id="loop", completed=True)
        _save_checkpoint(store, "cp-wip", loop_id="loop", completed=False)
        incomplete = store.get_incomplete()
        assert len(incomplete) == 1
        assert incomplete[0].checkpoint_id == "cp-wip"

    def test_get_incomplete_filter_by_loop_id(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        _save_checkpoint(store, "cp-a", loop_id="loop1")
        _save_checkpoint(store, "cp-b", loop_id="loop2")
        incomplete = store.get_incomplete(loop_id="loop1")
        assert len(incomplete) == 1
        assert incomplete[0].loop_id == "loop1"

    def test_cleanup_keeps_latest(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        for i in range(7):
            _save_checkpoint(
                store,
                f"cp-{i}",
                loop_id="loop",
                iteration=i,
                updated_at=f"2024-01-0{i + 1}T00:00:00",
            )
        deleted = store.cleanup(loop_id="loop", keep_latest=3)
        assert deleted == 4
        remaining = store.list_checkpoints(loop_id="loop")
        assert len(remaining) == 3

    def test_checkpoint_id_sanitizes_special_chars(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        path = store._checkpoint_path("my/loop:test")
        assert "/" not in path.name.replace(".json", "").replace("/", "")
        assert ":" not in path.name

    def test_create_with_metadata(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        cp = store.create("loop", "p", metadata={"env": "test", "version": 2})
        loaded = store.load(cp.checkpoint_id)
        assert loaded.metadata == {"env": "test", "version": 2}

    def test_create_with_all_params(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        cp = store.create(
            "loop",
            "prompt",
            max_iterations=20,
            completion_promise="ALL DONE",
            model="claude-sonnet",
            reasoning_effort="high",
        )
        assert cp.max_iterations == 20
        assert cp.completion_promise == "ALL DONE"
        assert cp.model == "claude-sonnet"
        assert cp.reasoning_effort == "high"


# ---------------------------------------------------------------------------
# CheckpointedRalph
# ---------------------------------------------------------------------------


class TestCheckpointedRalph:
    def _make_client(self):
        client = AsyncMock()
        client.config = SimpleNamespace(
            model=SimpleNamespace(value="gpt-5.2-codex"),
            reasoning_effort=SimpleNamespace(value="xhigh"),
        )
        return client

    @pytest.mark.asyncio
    async def test_creates_checkpoint_on_loop(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        client = self._make_client()

        with patch("copex.ralph.RalphWiggum") as MockRalph:
            instance = MockRalph.return_value
            instance.loop = AsyncMock()
            ralph = CheckpointedRalph(client, store, loop_id="test-loop")
            result = await ralph.loop("do stuff", max_iterations=5)

        assert result is not None
        assert result.loop_id == "test-loop"
        assert result.prompt == "do stuff"
        items = store.list_checkpoints(loop_id="test-loop")
        assert len(items) == 1

    @pytest.mark.asyncio
    async def test_on_iteration_updates_checkpoint(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        client = self._make_client()

        async def fake_loop(prompt, *, max_iterations=None, completion_promise=None, on_iteration=None, on_complete=None):
            if on_iteration:
                on_iteration(1, "response 1")
                on_iteration(2, "response 2")

        with patch("copex.ralph.RalphWiggum") as MockRalph:
            instance = MockRalph.return_value
            instance.loop = AsyncMock(side_effect=fake_loop)
            ralph = CheckpointedRalph(client, store, loop_id="iter-loop")
            result = await ralph.loop("prompt")

        assert result.iteration == 2
        assert result.history == ["response 1", "response 2"]

    @pytest.mark.asyncio
    async def test_on_complete_marks_completed(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        client = self._make_client()

        async def fake_loop(prompt, *, max_iterations=None, completion_promise=None, on_iteration=None, on_complete=None):
            if on_iteration:
                on_iteration(1, "done")
            if on_complete:
                on_complete(SimpleNamespace(iteration=1, completion_reason="promise: DONE"))

        with patch("copex.ralph.RalphWiggum") as MockRalph:
            instance = MockRalph.return_value
            instance.loop = AsyncMock(side_effect=fake_loop)
            ralph = CheckpointedRalph(client, store, loop_id="complete-loop")
            result = await ralph.loop("prompt", completion_promise="DONE")

        assert result.completed is True
        assert result.completion_reason == "promise: DONE"

    @pytest.mark.asyncio
    async def test_resume_from_existing_checkpoint(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        client = self._make_client()

        cp = _save_checkpoint(
            store, "resume-loop_20240101_000000",
            loop_id="resume-loop", prompt="original prompt",
            max_iterations=10, iteration=3, history=["r1", "r2", "r3"],
        )

        with patch("copex.ralph.RalphWiggum") as MockRalph:
            instance = MockRalph.return_value
            instance.loop = AsyncMock()
            ralph = CheckpointedRalph(client, store, loop_id="resume-loop")
            result = await ralph.loop("original prompt", resume=True)

        items = store.list_checkpoints(loop_id="resume-loop")
        assert len(items) == 1
        assert result.checkpoint_id == cp.checkpoint_id

    @pytest.mark.asyncio
    async def test_no_resume_creates_new_checkpoint(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        client = self._make_client()

        _save_checkpoint(store, "loop_old", loop_id="loop")

        with patch("copex.ralph.RalphWiggum") as MockRalph:
            instance = MockRalph.return_value
            instance.loop = AsyncMock()
            ralph = CheckpointedRalph(client, store, loop_id="loop")
            await ralph.loop("new prompt", resume=False)

        items = store.list_checkpoints(loop_id="loop")
        assert len(items) == 2

    @pytest.mark.asyncio
    async def test_resume_skips_completed_checkpoint(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        client = self._make_client()

        _save_checkpoint(
            store, "loop_done", loop_id="loop",
            completed=True, completion_reason="done",
        )

        with patch("copex.ralph.RalphWiggum") as MockRalph:
            instance = MockRalph.return_value
            instance.loop = AsyncMock()
            ralph = CheckpointedRalph(client, store, loop_id="loop")
            await ralph.loop("prompt", resume=True)

        items = store.list_checkpoints(loop_id="loop")
        assert len(items) == 2

    @pytest.mark.asyncio
    async def test_passes_config_to_ralph(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        client = self._make_client()

        with patch("copex.ralph.RalphWiggum") as MockRalph, \
             patch("copex.ralph.RalphConfig") as MockConfig:
            instance = MockRalph.return_value
            instance.loop = AsyncMock()
            ralph = CheckpointedRalph(client, store, loop_id="loop")
            await ralph.loop("prompt", max_iterations=10, completion_promise="DONE")

        MockConfig.assert_called_once_with(max_iterations=10, completion_promise="DONE")
        MockRalph.assert_called_once_with(client, MockConfig.return_value)
