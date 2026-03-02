from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from copex.config import CopexConfig
from copex.squad import SquadAgent
from copex.squad_team import SquadTeam
from copex.squad_team_ai import build_repo_analysis_prompt


def run(coro):
    return asyncio.run(coro)


def _fake_cli_class(response_content: str | None = None, error: Exception | None = None):
    class _FakeCLI:
        def __init__(self, _config):
            self._response_content = response_content or "[]"
            self._error = error

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def send(self, _prompt: str):
            if self._error is not None:
                raise self._error
            return SimpleNamespace(content=self._response_content)

    return _FakeCLI


class TestSquadTeamFromRepo:
    def test_from_repo_detects_role_signals(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("print('ok')", encoding="utf-8")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_app.py").write_text("def test_x(): pass", encoding="utf-8")
        (tmp_path / "docs").mkdir()
        (tmp_path / "Dockerfile").write_text("FROM python:3.11", encoding="utf-8")
        (tmp_path / "components").mkdir()
        (tmp_path / "api").mkdir()

        team = SquadTeam.from_repo(tmp_path)
        assert {"lead", "developer", "tester", "docs", "devops", "frontend", "backend"}.issubset(
            set(team.roles)
        )

    def test_from_repo_empty_dir_falls_back_to_minimum(self, tmp_path):
        team = SquadTeam.from_repo(tmp_path)
        assert team.roles == ["lead", "developer"]


class TestSquadTeamPersistenceAndNormalization:
    def test_save_load_toml_round_trip(self, tmp_path):
        team = SquadTeam(
            agents=[
                SquadAgent(
                    name="Architect",
                    role="lead",
                    emoji="A",
                    system_prompt="You are Architect.",
                    phase=1,
                ),
                SquadAgent(
                    name="Security Engineer",
                    role="security_engineer",
                    emoji="S",
                    system_prompt="You are the Security Engineer.",
                    phase=2,
                    depends_on=["lead"],
                    subtasks=["threat model"],
                    retries=2,
                    retry_delay=1.5,
                ),
            ],
            phase_gates={1: True, 2: False},
        )

        team.save_squad_file(tmp_path)
        loaded = SquadTeam.load_squad_file(tmp_path)

        assert loaded is not None
        assert loaded.phase_gates == {1: True, 2: False}
        loaded_agent = loaded.get_agent("security_engineer")
        assert loaded_agent is not None
        assert loaded_agent.depends_on == ["lead"]
        assert loaded_agent.subtasks == ["threat model"]
        assert loaded_agent.retries == 2
        assert loaded_agent.retry_delay == 1.5

    def test_from_squad_payload_normalizes_role_phase_and_retries(self):
        payload = {
            "squad": {
                "lead": "Lead",
                "agents": [
                    {
                        "name": "Security Engineer",
                        "role": "Threat modeling",
                        "phase": "0",
                        "depends_on": ["Lead", "lead"],
                        "retries": "-2",
                        "retry_delay": "-5",
                    }
                ],
            }
        }

        team = SquadTeam._from_squad_payload(payload)

        assert team is not None
        agent = team.get_agent("security_engineer")
        assert agent is not None
        assert agent.phase == 1
        assert agent.depends_on == ["lead"]
        assert agent.retries == 0
        assert agent.retry_delay == 0.0

    def test_load_squad_file_migrates_legacy_json(self, tmp_path):
        legacy_path = tmp_path / ".copex" / "squad.json"
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_path.write_text(
            json.dumps(
                [
                    {"role": "lead", "name": "Architect", "phase": 1},
                    {"role": "developer", "name": "Dev", "phase": 2, "retries": 3},
                ]
            ),
            encoding="utf-8",
        )

        with pytest.warns(DeprecationWarning):
            team = SquadTeam.load_squad_file(tmp_path)

        assert team is not None
        assert team.get_agent("lead") is not None
        assert team.get_agent("developer") is not None
        assert team.get_agent("developer").retries == 3
        assert (tmp_path / ".squad" / "team.toml").is_file()


class TestSquadTeamAIModes:
    def test_from_repo_ai_with_mocked_ai(self, tmp_path):
        ai_response = json.dumps(
            [
                {
                    "role": "lead",
                    "name": "Lead Architect",
                    "emoji": "L",
                    "prompt": "Lead planning",
                    "phase": 1,
                },
                {
                    "role": "developer",
                    "name": "Developer",
                    "emoji": "D",
                    "prompt": "Build features",
                    "phase": 2,
                    "depends_on": ["lead"],
                    "retries": 3,
                    "retry_delay": 2.0,
                },
                {
                    "role": "tester",
                    "name": "Tester",
                    "emoji": "T",
                    "prompt": "Test changes",
                    "phase": 3,
                },
            ]
        )
        repo_context = {"project_name": "demo", "source_extensions": [".py"]}

        with patch("copex.cli_client.CopilotCLI", new=_fake_cli_class(ai_response)):
            team = run(
                SquadTeam.from_repo_ai(
                    config=CopexConfig(),
                    path=tmp_path,
                    repo_context=repo_context,
                )
            )

        assert team.get_agent("lead").name == "Lead Architect"
        assert team.get_agent("developer").phase == 2
        assert team.get_agent("developer").depends_on == ["lead"]
        assert team.get_agent("developer").retries == 3
        assert team.get_agent("developer").retry_delay == 2.0
        assert team.get_agent("tester").phase == 3

    def test_build_repo_analysis_prompt_mentions_retry_fields(self):
        prompt = build_repo_analysis_prompt(
            {"project_name": "demo"},
            existing_agents=[("🏗️", "Lead", "lead", 1)],
        )
        assert '"retries"' in prompt
        assert '"retry_delay"' in prompt

    def test_from_repo_ai_falls_back_to_pattern_matching_on_error(self, tmp_path):
        (tmp_path / "main.py").write_text("print('ok')", encoding="utf-8")
        repo_context = {"project_name": "demo"}

        with patch("copex.cli_client.CopilotCLI", new=_fake_cli_class(error=RuntimeError("boom"))):
            team = run(
                SquadTeam.from_repo_ai(
                    config=CopexConfig(),
                    path=tmp_path,
                    repo_context=repo_context,
                )
            )

        assert "lead" in team.roles
        assert "developer" in team.roles

    def test_update_from_request_normalizes_roles_and_phase(self, tmp_path):
        SquadTeam.default().save_squad_file(tmp_path)
        ai_response = json.dumps(
            [
                {
                    "name": "Chief Architect",
                    "role": "Owns architecture",
                    "phase": 1,
                    "lead": True,
                },
                {
                    "name": "Developer",
                    "id": "dev-role",
                    "role": "Implements features",
                    "phase": 5,
                    "depends_on": ["Lead"],
                    "retries": "3",
                    "retry_delay": "1.25",
                    "subtasks": ["api", ""],
                },
                {
                    "name": "Developer",
                    "role": "Duplicate should be dropped",
                    "phase": 2,
                },
                {
                    "name": "QA",
                    "role": "",
                    "phase": 3,
                },
            ]
        )

        with patch("copex.cli_client.CopilotCLI", new=_fake_cli_class(ai_response)):
            team = run(
                SquadTeam.update_from_request(
                    "add QA coverage",
                    config=CopexConfig(),
                    path=tmp_path,
                )
            )

        assert team.get_agent("lead").name == "Chief Architect"
        dev = team.get_agent("dev-role")
        assert dev is not None
        assert dev.phase == 4
        assert dev.depends_on == ["lead"]
        assert dev.retries == 3
        assert dev.retry_delay == 1.25
        assert dev.subtasks == ["api"]
        assert team.get_agent("qa") is not None
