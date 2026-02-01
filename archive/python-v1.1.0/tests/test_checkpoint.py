from __future__ import annotations

from pathlib import Path

from copex.checkpoint import CheckpointStore


def test_checkpoint_store_create_and_load(tmp_path: Path) -> None:
    store = CheckpointStore(base_dir=tmp_path)
    checkpoint = store.create("loop-1", "Prompt", max_iterations=3, completion_promise="DONE")

    loaded = store.load(checkpoint.checkpoint_id)
    assert loaded is not None
    assert loaded.loop_id == "loop-1"
    assert loaded.prompt == "Prompt"
    assert loaded.max_iterations == 3


def test_checkpoint_store_update_and_get_latest(tmp_path: Path) -> None:
    store = CheckpointStore(base_dir=tmp_path)
    first = store.create("loop-1", "Prompt")
    store.update(first.checkpoint_id, iteration=2)

    latest = store.get_latest("loop-1")
    assert latest is not None
    assert latest.iteration == 2


def test_checkpoint_store_incomplete_and_cleanup(tmp_path: Path) -> None:
    store = CheckpointStore(base_dir=tmp_path)
    first = store.create("loop-1", "Prompt")
    second = store.create("loop-2", "Prompt")
    store.update(second.checkpoint_id, completed=True)

    incomplete = store.get_incomplete("loop-1")
    assert len(incomplete) == 1
    assert incomplete[0].checkpoint_id == first.checkpoint_id

    deleted = store.cleanup("loop-1", keep_latest=0)
    assert deleted >= 1


def test_checkpoint_store_path_sanitization(tmp_path: Path) -> None:
    store = CheckpointStore(base_dir=tmp_path)
    path = store._checkpoint_path("loop/with spaces")
    assert path.name == "loop_with_spaces.json"
