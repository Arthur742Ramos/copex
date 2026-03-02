from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from copex.campaign import (
    CampaignState,
    WaveResult,
    WaveStatus,
    batch_targets,
    create_campaign,
    generate_wave_tasks,
    get_pending_wave_indices,
    load_campaign_state,
    run_discover_command,
    save_campaign_state,
)


class TestCampaignStatePersistence:
    def test_save_and_load_campaign_state_round_trip(self, tmp_path):
        state = create_campaign(
            goal="add typing",
            discover_command="find . -name '*.py'",
            batch_size=2,
            targets=["a.py", "b.py", "c.py"],
        )
        path = tmp_path / ".copex" / "campaign.json"

        saved_path = save_campaign_state(state, path)
        loaded = load_campaign_state(saved_path)

        assert saved_path.is_file()
        assert loaded is not None
        assert loaded.goal == "add typing"
        assert loaded.batch_size == 2
        assert loaded.all_targets == ["a.py", "b.py", "c.py"]
        assert len(loaded.waves) == 2

    def test_load_campaign_state_missing_returns_none(self, tmp_path):
        assert load_campaign_state(tmp_path / "missing.json") is None

    def test_load_campaign_state_corrupt_returns_none(self, tmp_path):
        path = tmp_path / ".copex" / "campaign.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not-json", encoding="utf-8")

        assert load_campaign_state(path) is None

    def test_campaign_state_to_dict_round_trip(self):
        state = CampaignState(
            goal="goal",
            discover_command="discover",
            batch_size=3,
            all_targets=["a", "b"],
            waves=[WaveResult(wave_index=0, targets=["a"], status=WaveStatus.COMPLETED, succeeded=1)],
            created_at="t1",
            updated_at="t2",
            total_duration_seconds=12.5,
            status="completed",
        )

        restored = CampaignState.from_dict(json.loads(json.dumps(state.to_dict())))

        assert restored.goal == "goal"
        assert restored.waves[0].status == WaveStatus.COMPLETED
        assert restored.total_duration_seconds == 12.5


class TestCampaignDiscovery:
    def test_run_discover_command_success(self):
        result = SimpleNamespace(returncode=0, stdout="a.py\n\nb.py\n", stderr="")
        with patch("copex.campaign.subprocess.run", return_value=result) as mock_run:
            targets = run_discover_command("fake discover", cwd="/tmp/demo")

        assert targets == ["a.py", "b.py"]
        mock_run.assert_called_once()
        assert mock_run.call_args.args[0] == ["fake", "discover"]
        assert mock_run.call_args.kwargs["shell"] is False
        assert mock_run.call_args.kwargs["cwd"] == "/tmp/demo"

    def test_run_discover_command_timeout_raises(self):
        with patch(
            "copex.campaign.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="discover", timeout=120),
        ):
            with pytest.raises(RuntimeError, match="timed out"):
                run_discover_command("discover")

    def test_run_discover_command_nonzero_exit_raises(self):
        result = SimpleNamespace(returncode=2, stdout="", stderr="bad command")
        with patch("copex.campaign.subprocess.run", return_value=result):
            with pytest.raises(RuntimeError, match="exit 2"):
                run_discover_command("discover")

    def test_run_discover_command_invalid_command_raises(self):
        with pytest.raises(RuntimeError, match="invalid"):
            run_discover_command('echo "unterminated')


class TestCampaignOrchestrationHelpers:
    def test_batch_targets_respects_invalid_batch_size(self):
        assert batch_targets(["a", "b", "c"], batch_size=0) == [["a"], ["b"], ["c"]]
        assert batch_targets(["a", "b", "c"], batch_size=2) == [["a", "b"], ["c"]]

    def test_generate_wave_tasks_builds_ids_and_prompts(self):
        tasks = generate_wave_tasks("Refactor modules", ["src/a.py", "src/b.py"], wave_index=3)

        assert [task.id for task in tasks] == ["wave-3-task-1", "wave-3-task-2"]
        assert "Goal: Refactor modules" in tasks[0].prompt
        assert "Target: src/a.py" in tasks[0].prompt
        assert "Target: src/b.py" in tasks[1].prompt

    def test_get_pending_wave_indices_includes_pending_and_failed(self):
        state = CampaignState(
            goal="g",
            discover_command="d",
            batch_size=2,
            all_targets=[],
            waves=[
                WaveResult(wave_index=0, targets=["a"], status=WaveStatus.COMPLETED),
                WaveResult(wave_index=1, targets=["b"], status=WaveStatus.PENDING),
                WaveResult(wave_index=2, targets=["c"], status=WaveStatus.FAILED),
                WaveResult(wave_index=3, targets=["d"], status=WaveStatus.SKIPPED),
            ],
        )

        assert get_pending_wave_indices(state) == [1, 2]

    def test_create_campaign_builds_waves(self):
        state = create_campaign(
            goal="apply change",
            discover_command="discover things",
            batch_size=2,
            targets=["a", "b", "c", "d", "e"],
        )

        assert state.goal == "apply change"
        assert state.discover_command == "discover things"
        assert len(state.waves) == 3
        assert state.waves[0].targets == ["a", "b"]
        assert state.waves[1].targets == ["c", "d"]
        assert state.waves[2].targets == ["e"]
        assert state.created_at
