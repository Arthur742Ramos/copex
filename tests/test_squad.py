"""Tests for squad functionality.

Tests the SquadCoordinator, SquadTeam, SquadAgent, SquadResult,
and CLI integration for the built-in multi-agent orchestration.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from copex.config import CopexConfig
from copex.fleet import Fleet, FleetConfig, FleetResult
from copex.squad import (
    _KNOWN_ROLE_PHASES,
    _ROLE_EMOJIS,
    _ROLE_PROMPTS,
    SquadAgent,
    SquadAgentResult,
    SquadCoordinator,
    SquadResult,
    SquadRole,
    SquadTeam,
)
from copex.streaming import Response


def run(coro):
    return asyncio.run(coro)


# ===========================================================================
# 1. SquadRole enum
# ===========================================================================


class TestSquadRole:

    def test_roles_exist(self):
        assert SquadRole.LEAD == "lead"
        assert SquadRole.DEVELOPER == "developer"
        assert SquadRole.TESTER == "tester"

    def test_role_from_string(self):
        assert SquadRole("lead") == SquadRole.LEAD
        assert SquadRole("developer") == SquadRole.DEVELOPER
        assert SquadRole("tester") == SquadRole.TESTER

    def test_invalid_role(self):
        with pytest.raises(ValueError):
            SquadRole("invalid")

    def test_all_roles_have_prompts(self):
        for role in SquadRole:
            assert role in _ROLE_PROMPTS
            assert len(_ROLE_PROMPTS[role]) > 0

    def test_all_roles_have_emojis(self):
        for role in SquadRole:
            assert role in _ROLE_EMOJIS
            assert len(_ROLE_EMOJIS[role]) > 0


# ===========================================================================
# 2. SquadAgent
# ===========================================================================


class TestSquadAgent:

    def test_fields(self):
        agent = SquadAgent(
            name="Lead",
            role=SquadRole.LEAD,
            emoji="🏗️",
            system_prompt="You are the lead.",
        )
        assert agent.name == "Lead"
        assert agent.role == SquadRole.LEAD
        assert agent.emoji == "🏗️"
        assert agent.system_prompt == "You are the lead."

    def test_default_for_role(self):
        for role in SquadRole:
            agent = SquadAgent.default_for_role(role)
            assert agent.role == role
            assert agent.name == role.value.title()
            assert agent.emoji == _ROLE_EMOJIS[role]
            assert agent.system_prompt == _ROLE_PROMPTS[role]

    def test_default_lead(self):
        agent = SquadAgent.default_for_role(SquadRole.LEAD)
        assert agent.name == "Lead"
        assert "Architect" in agent.system_prompt

    def test_default_developer(self):
        agent = SquadAgent.default_for_role(SquadRole.DEVELOPER)
        assert agent.name == "Developer"
        assert "Implement" in agent.system_prompt

    def test_default_tester(self):
        agent = SquadAgent.default_for_role(SquadRole.TESTER)
        assert agent.name == "Tester"
        assert "tests" in agent.system_prompt.lower()


# ===========================================================================
# 3. SquadTeam
# ===========================================================================


class TestSquadTeam:

    def test_default_team(self):
        team = SquadTeam.default()
        assert len(team.agents) == 4
        roles = team.roles
        assert SquadRole.LEAD in roles
        assert SquadRole.DEVELOPER in roles
        assert SquadRole.TESTER in roles
        assert SquadRole.DOCS in roles

    def test_get_agent(self):
        team = SquadTeam.default()
        lead = team.get_agent(SquadRole.LEAD)
        assert lead is not None
        assert lead.role == SquadRole.LEAD

    def test_get_agent_missing(self):
        team = SquadTeam(agents=[])
        assert team.get_agent(SquadRole.LEAD) is None

    def test_custom_team(self):
        agents = [SquadAgent.default_for_role(SquadRole.DEVELOPER)]
        team = SquadTeam(agents=agents)
        assert len(team.agents) == 1
        assert team.roles == [SquadRole.DEVELOPER]

    def test_roles_property(self):
        team = SquadTeam.default()
        assert team.roles == [SquadRole.LEAD, SquadRole.DEVELOPER, SquadRole.TESTER, SquadRole.DOCS]


# ===========================================================================
# 4. SquadAgentResult
# ===========================================================================


class TestSquadAgentResult:

    def test_fields(self):
        agent = SquadAgent.default_for_role(SquadRole.LEAD)
        ar = SquadAgentResult(
            agent=agent,
            content="Analysis complete.",
            success=True,
            duration_ms=1500.0,
        )
        assert ar.agent.role == SquadRole.LEAD
        assert ar.content == "Analysis complete."
        assert ar.success is True
        assert ar.duration_ms == 1500.0
        assert ar.error is None

    def test_fields_with_error(self):
        agent = SquadAgent.default_for_role(SquadRole.DEVELOPER)
        ar = SquadAgentResult(
            agent=agent,
            content="",
            success=False,
            error="Connection failed",
        )
        assert ar.success is False
        assert ar.error == "Connection failed"

    def test_default_tokens(self):
        agent = SquadAgent.default_for_role(SquadRole.TESTER)
        ar = SquadAgentResult(agent=agent, content="ok", success=True)
        assert ar.prompt_tokens == 0
        assert ar.completion_tokens == 0


# ===========================================================================
# 5. SquadResult
# ===========================================================================


class TestSquadResult:

    def _make_result(self, success=True) -> SquadResult:
        lead = SquadAgent.default_for_role(SquadRole.LEAD)
        dev = SquadAgent.default_for_role(SquadRole.DEVELOPER)
        return SquadResult(
            agent_results=[
                SquadAgentResult(agent=lead, content="Plan: build API", success=True, duration_ms=500),
                SquadAgentResult(agent=dev, content="Built API", success=success, duration_ms=1000),
            ],
            total_duration_ms=1500.0,
            success=success,
        )

    def test_final_content(self):
        result = self._make_result()
        content = result.final_content
        assert "Lead" in content
        assert "Developer" in content
        assert "Plan: build API" in content
        assert "Built API" in content

    def test_final_content_empty(self):
        result = SquadResult()
        assert result.final_content == ""

    def test_to_dict(self):
        result = self._make_result()
        d = result.to_dict()
        assert d["success"] is True
        assert d["total_duration_ms"] == 1500.0
        assert len(d["agents"]) == 2
        assert d["agents"][0]["role"] == "lead"
        assert d["agents"][1]["role"] == "developer"

    def test_to_json(self):
        result = self._make_result()
        j = result.to_json()
        parsed = json.loads(j)
        assert parsed["success"] is True
        assert len(parsed["agents"]) == 2

    def test_to_json_with_indent(self):
        result = self._make_result()
        j = result.to_json(indent=2)
        assert "\n" in j  # Indented
        parsed = json.loads(j)
        assert parsed["success"] is True

    def test_failed_result(self):
        result = self._make_result(success=False)
        assert result.success is False
        d = result.to_dict()
        assert d["success"] is False


# ===========================================================================
# 6. SquadCoordinator — task building
# ===========================================================================


class TestSquadCoordinatorTaskBuilding:

    def test_build_agent_prompt_lead(self):
        config = CopexConfig()
        coord = SquadCoordinator(config)
        agent = coord.team.get_agent(SquadRole.LEAD)
        prompt = coord._build_agent_prompt(agent, "Build X", {})
        assert "Lead Architect" in prompt
        assert "Build X" in prompt
        assert "{{task:" not in prompt  # Lead has no deps

    def test_build_agent_prompt_developer(self):
        config = CopexConfig()
        coord = SquadCoordinator(config)
        agent = coord.team.get_agent(SquadRole.DEVELOPER)
        task_ids = {"lead": "lead"}
        prompt = coord._build_agent_prompt(agent, "Build X", task_ids)
        assert "Developer" in prompt
        assert "Build X" in prompt
        assert "{{task:lead.content}}" in prompt

    def test_build_agent_prompt_tester(self):
        config = CopexConfig()
        coord = SquadCoordinator(config)
        agent = coord.team.get_agent(SquadRole.TESTER)
        task_ids = {"lead": "lead"}
        prompt = coord._build_agent_prompt(agent, "Build X", task_ids)
        assert "Tester" in prompt
        assert "{{task:lead.content}}" in prompt

    def test_get_dependencies_lead(self):
        config = CopexConfig()
        coord = SquadCoordinator(config)
        agent = SquadAgent.default_for_role(SquadRole.LEAD)
        deps = coord._get_dependencies(agent, {})
        assert deps == []

    def test_get_dependencies_developer(self):
        config = CopexConfig()
        coord = SquadCoordinator(config)
        agent = SquadAgent.default_for_role(SquadRole.DEVELOPER)
        deps = coord._get_dependencies(agent, {"lead": "lead"})
        assert deps == ["lead"]

    def test_get_dependencies_tester(self):
        config = CopexConfig()
        coord = SquadCoordinator(config)
        agent = SquadAgent.default_for_role(SquadRole.TESTER)
        deps = coord._get_dependencies(agent, {"lead": "lead"})
        assert deps == ["lead"]

    def test_get_dependencies_no_lead(self):
        config = CopexConfig()
        coord = SquadCoordinator(config)
        agent = SquadAgent.default_for_role(SquadRole.DEVELOPER)
        deps = coord._get_dependencies(agent, {})
        assert deps == []


# ===========================================================================
# 7. SquadCoordinator — result building
# ===========================================================================


class TestSquadCoordinatorResultBuilding:

    def test_build_result_success(self):
        config = CopexConfig()
        coord = SquadCoordinator(config)
        task_ids = {
            "lead": "lead",
            "developer": "developer",
            "tester": "tester",
        }
        fleet_results = [
            FleetResult(task_id="lead", success=True, response=Response(content="Plan"), duration_ms=100),
            FleetResult(task_id="developer", success=True, response=Response(content="Code"), duration_ms=200),
            FleetResult(task_id="tester", success=True, response=Response(content="Tests"), duration_ms=150),
        ]

        result = coord._build_result(fleet_results, task_ids)
        assert result.success is True
        assert len(result.agent_results) == 3
        assert result.agent_results[0].content == "Plan"
        assert result.agent_results[1].content == "Code"
        assert result.agent_results[2].content == "Tests"

    def test_build_result_partial_failure(self):
        config = CopexConfig()
        coord = SquadCoordinator(config)
        task_ids = {"lead": "lead", "developer": "developer"}
        fleet_results = [
            FleetResult(task_id="lead", success=True, response=Response(content="Plan"), duration_ms=100),
            FleetResult(
                task_id="developer",
                success=False,
                error=RuntimeError("Failed"),
                duration_ms=50,
            ),
        ]

        result = coord._build_result(fleet_results, task_ids)
        assert result.success is False
        assert result.agent_results[1].error == "Failed"

    def test_build_result_unknown_task_id(self):
        config = CopexConfig()
        coord = SquadCoordinator(config)
        task_ids = {"lead": "lead"}
        fleet_results = [
            FleetResult(task_id="lead", success=True, response=Response(content="ok"), duration_ms=100),
            FleetResult(task_id="unknown", success=True, response=Response(content="?"), duration_ms=50),
        ]

        result = coord._build_result(fleet_results, task_ids)
        assert len(result.agent_results) == 1  # Unknown ignored

    def test_build_result_no_response(self):
        config = CopexConfig()
        coord = SquadCoordinator(config)
        task_ids = {"lead": "lead"}
        fleet_results = [
            FleetResult(task_id="lead", success=False, response=None, duration_ms=50),
        ]

        result = coord._build_result(fleet_results, task_ids)
        assert result.agent_results[0].content == ""


# ===========================================================================
# 8. SquadCoordinator — custom team
# ===========================================================================


class TestSquadCoordinatorCustomTeam:

    def test_custom_team(self):
        config = CopexConfig()
        team = SquadTeam(agents=[SquadAgent.default_for_role(SquadRole.DEVELOPER)])
        coord = SquadCoordinator(config, team=team)
        assert len(coord.team.agents) == 1
        assert coord.team.roles == [SquadRole.DEVELOPER]

    def test_custom_fleet_config(self):
        config = CopexConfig()
        fc = FleetConfig(max_concurrent=10, timeout=300.0)
        coord = SquadCoordinator(config, fleet_config=fc)
        assert coord._fleet_config.max_concurrent == 10
        assert coord._fleet_config.timeout == 300.0

    def test_default_fleet_config(self):
        config = CopexConfig()
        coord = SquadCoordinator(config)
        assert coord._fleet_config.max_concurrent == 3
        assert coord._fleet_config.timeout == 600.0


# ===========================================================================
# 9. SquadCoordinator — context manager
# ===========================================================================


class TestSquadCoordinatorContextManager:

    def test_async_context_manager(self):
        config = CopexConfig()
        coord = SquadCoordinator(config)

        async def _test():
            async with coord as c:
                assert c is coord

        run(_test())


# ===========================================================================
# 10. SquadCoordinator — add_tasks integration
# ===========================================================================


class TestSquadCoordinatorAddTasks:

    def test_add_tasks_creates_fleet_tasks(self):
        from copex.fleet import Fleet

        config = CopexConfig()
        coord = SquadCoordinator(config, team=SquadTeam.default())
        fleet = Fleet(config)
        task_ids = coord._add_tasks(fleet, "Build an API")

        assert "lead" in task_ids
        assert "developer" in task_ids
        assert "tester" in task_ids
        assert "docs" in task_ids
        assert len(fleet._tasks) == 4

    def test_add_tasks_dependencies(self):
        from copex.fleet import Fleet

        config = CopexConfig()
        coord = SquadCoordinator(config, team=SquadTeam.default())
        fleet = Fleet(config)
        task_ids = coord._add_tasks(fleet, "Build an API")

        lead_task = next(t for t in fleet._tasks if t.id == task_ids["lead"])
        dev_task = next(t for t in fleet._tasks if t.id == task_ids["developer"])
        tester_task = next(t for t in fleet._tasks if t.id == task_ids["tester"])

        assert lead_task.depends_on == []
        assert dev_task.depends_on == [task_ids["lead"]]
        # Tester (phase 3) depends on all phase 1+2 agents
        assert task_ids["lead"] in tester_task.depends_on
        assert task_ids["developer"] in tester_task.depends_on

    def test_add_tasks_prompts_include_role(self):
        from copex.fleet import Fleet

        config = CopexConfig()
        coord = SquadCoordinator(config)
        fleet = Fleet(config)
        coord._add_tasks(fleet, "Fix the bug")

        lead_task = next(t for t in fleet._tasks if t.id == "lead")
        dev_task = next(t for t in fleet._tasks if t.id == "developer")
        tester_task = next(t for t in fleet._tasks if t.id == "tester")

        assert "Lead Architect" in lead_task.prompt
        assert "Developer" in dev_task.prompt
        assert "Tester" in tester_task.prompt

    def test_add_tasks_prompts_include_user_prompt(self):
        from copex.fleet import Fleet

        config = CopexConfig()
        coord = SquadCoordinator(config)
        fleet = Fleet(config)
        coord._add_tasks(fleet, "Build a calculator")

        for task in fleet._tasks:
            assert "Build a calculator" in task.prompt


# ===========================================================================
# 11. SquadResult edge cases
# ===========================================================================


class TestSquadResultEdgeCases:

    def test_empty_result(self):
        result = SquadResult()
        assert result.success is True
        assert result.final_content == ""
        assert result.total_duration_ms == 0.0
        d = result.to_dict()
        assert d["agents"] == []

    def test_result_with_empty_content(self):
        agent = SquadAgent.default_for_role(SquadRole.LEAD)
        ar = SquadAgentResult(agent=agent, content="", success=True)
        result = SquadResult(agent_results=[ar])
        assert result.final_content == ""  # Empty content filtered out

    def test_duration_rounding(self):
        result = SquadResult(total_duration_ms=1234.5678)
        d = result.to_dict()
        assert d["total_duration_ms"] == 1234.6

    def test_agent_result_duration_rounding(self):
        agent = SquadAgent.default_for_role(SquadRole.DEVELOPER)
        ar = SquadAgentResult(agent=agent, content="done", success=True, duration_ms=999.999)
        result = SquadResult(agent_results=[ar])
        d = result.to_dict()
        assert d["agents"][0]["duration_ms"] == 1000.0


# ===========================================================================
# 12. JSON serialization
# ===========================================================================


class TestSquadJSON:

    def test_json_roundtrip(self):
        agent = SquadAgent.default_for_role(SquadRole.LEAD)
        ar = SquadAgentResult(
            agent=agent,
            content="Analysis done",
            success=True,
            duration_ms=500.0,
            prompt_tokens=100,
            completion_tokens=50,
        )
        result = SquadResult(
            agent_results=[ar],
            total_duration_ms=500.0,
            success=True,
        )
        j = result.to_json()
        parsed = json.loads(j)
        assert parsed["success"] is True
        assert parsed["agents"][0]["name"] == "Lead"
        assert parsed["agents"][0]["prompt_tokens"] == 100

    def test_json_with_error(self):
        agent = SquadAgent.default_for_role(SquadRole.DEVELOPER)
        ar = SquadAgentResult(
            agent=agent,
            content="",
            success=False,
            error="Timeout",
        )
        result = SquadResult(agent_results=[ar], success=False)
        parsed = json.loads(result.to_json())
        assert parsed["success"] is False
        assert parsed["agents"][0]["error"] == "Timeout"

    def test_json_unicode(self):
        agent = SquadAgent.default_for_role(SquadRole.TESTER)
        ar = SquadAgentResult(
            agent=agent,
            content="Tests für Ärger with emojis 🎉",
            success=True,
        )
        result = SquadResult(agent_results=[ar])
        j = result.to_json()
        parsed = json.loads(j)
        assert "für" in parsed["agents"][0]["content"]
        assert "🎉" in parsed["agents"][0]["content"]


# ===========================================================================
# 13. DOCS role
# ===========================================================================


class TestDocsRole:

    def test_docs_role_exists(self):
        assert SquadRole.DOCS == "docs"
        assert SquadRole("docs") == SquadRole.DOCS
        assert SquadRole.DOCS in _ROLE_PROMPTS
        assert SquadRole.DOCS in _ROLE_EMOJIS
        assert _ROLE_EMOJIS[SquadRole.DOCS] == "📝"
        assert "Documentation" in _ROLE_PROMPTS[SquadRole.DOCS]

    def test_default_team_has_docs(self):
        team = SquadTeam.default()
        assert len(team.agents) == 4
        docs = team.get_agent(SquadRole.DOCS)
        assert docs is not None
        assert docs.name == "Docs"
        assert docs.emoji == "📝"

    def test_docs_dependencies(self):
        config = CopexConfig()
        coord = SquadCoordinator(config)
        task_ids = {
            "lead": "lead",
            "developer": "developer",
            "tester": "tester",
        }
        agent = SquadAgent.default_for_role(SquadRole.DOCS)
        deps = coord._get_dependencies(agent, task_ids)
        assert "developer" in deps
        assert "tester" in deps
        assert "lead" in deps  # phase 4 depends on all lower phases


# ===========================================================================
# 14. Project context discovery
# ===========================================================================


class TestProjectContextDiscovery:

    def test_discover_project_context_with_readme(self, tmp_path, monkeypatch):
        (tmp_path / "README.md").write_text("# My Project\nSome description", encoding="utf-8")
        (tmp_path / "src").mkdir()
        (tmp_path / "tests").mkdir()
        monkeypatch.chdir(tmp_path)

        config = CopexConfig()
        coord = SquadCoordinator(config)
        ctx = coord._discover_project_context()

        assert "## Project Context" in ctx
        assert "My Project" in ctx
        assert "src/" in ctx

    def test_discover_project_context_without_readme(self, tmp_path, monkeypatch):
        (tmp_path / "somefile.txt").write_text("hello", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        config = CopexConfig()
        coord = SquadCoordinator(config)
        ctx = coord._discover_project_context()

        # Should still return structure even without README
        assert "somefile.txt" in ctx

    def test_discover_project_context_empty_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        config = CopexConfig()
        coord = SquadCoordinator(config)
        ctx = coord._discover_project_context()

        assert ctx == ""

    def test_project_context_in_prompts(self, tmp_path, monkeypatch):
        (tmp_path / "README.md").write_text("# TestProject\nA test project", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        config = CopexConfig()
        coord = SquadCoordinator(config)
        coord._project_context = coord._discover_project_context()

        agent = coord.team.get_agent(SquadRole.LEAD)
        prompt = coord._build_agent_prompt(agent, "Do something", {})
        assert "## Project Context" in prompt
        assert "TestProject" in prompt

    def test_project_context_cached(self, tmp_path, monkeypatch):
        (tmp_path / "README.md").write_text("# Cached", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        config = CopexConfig()
        coord = SquadCoordinator(config)
        assert coord._project_context is None

        ctx1 = coord._discover_project_context()
        coord._project_context = ctx1

        # Overwrite README — cached value should remain unchanged
        (tmp_path / "README.md").write_text("# Changed", encoding="utf-8")
        assert coord._project_context == ctx1


# ===========================================================================
# 15. Project context — pyproject.toml reading
# ===========================================================================


class TestProjectContextPyproject:

    def test_discover_with_pyproject(self, tmp_path, monkeypatch):
        toml_content = b'[project]\nname = "my-tool"\ndescription = "A great tool"\n'
        (tmp_path / "pyproject.toml").write_bytes(toml_content)
        monkeypatch.chdir(tmp_path)

        config = CopexConfig()
        coord = SquadCoordinator(config)
        ctx = coord._discover_project_context()

        assert "Project: my-tool" in ctx
        assert "Description: A great tool" in ctx

    def test_discover_with_pyproject_name_only(self, tmp_path, monkeypatch):
        toml_content = b'[project]\nname = "bare-project"\n'
        (tmp_path / "pyproject.toml").write_bytes(toml_content)
        monkeypatch.chdir(tmp_path)

        config = CopexConfig()
        coord = SquadCoordinator(config)
        ctx = coord._discover_project_context()

        assert "Project: bare-project" in ctx
        assert "Description:" not in ctx

    def test_discover_with_pyproject_and_readme(self, tmp_path, monkeypatch):
        toml_content = b'[project]\nname = "combo"\n'
        (tmp_path / "pyproject.toml").write_bytes(toml_content)
        (tmp_path / "README.md").write_text("# Combo\nCombined", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        config = CopexConfig()
        coord = SquadCoordinator(config)
        ctx = coord._discover_project_context()

        assert "Project: combo" in ctx
        assert "Combo" in ctx
        assert "Structure:" in ctx


# ===========================================================================
# 16. _build_result edge case — agent not in team
# ===========================================================================


class TestBuildResultEdgeCases:

    def test_build_result_skips_role_not_in_team(self):
        """When task_ids reference a role the team doesn't have, skip it."""
        config = CopexConfig()
        team = SquadTeam(agents=[SquadAgent.default_for_role(SquadRole.LEAD)])
        coord = SquadCoordinator(config, team=team)

        task_ids = {
            "lead": "lead",
            "developer": "developer",  # not in team
        }
        fleet_results = [
            FleetResult(task_id="lead", success=True, response=Response(content="Plan"), duration_ms=100),
            FleetResult(task_id="developer", success=True, response=Response(content="Code"), duration_ms=200),
        ]

        result = coord._build_result(fleet_results, task_ids)
        # Developer result should be skipped since the team only has Lead
        assert len(result.agent_results) == 1
        assert result.agent_results[0].agent.role == SquadRole.LEAD


# ===========================================================================
# 17. SquadCoordinator.run() integration
# ===========================================================================


class TestSquadCoordinatorRun:

    def test_run_calls_fleet(self):
        """Test that run() wires Fleet and returns a SquadResult."""
        config = CopexConfig()
        coord = SquadCoordinator(config, team=SquadTeam.default())

        mock_fleet_results = [
            FleetResult(task_id="lead", success=True, response=Response(content="Plan"), duration_ms=100),
            FleetResult(task_id="developer", success=True, response=Response(content="Code"), duration_ms=200),
            FleetResult(task_id="tester", success=True, response=Response(content="Tests"), duration_ms=150),
            FleetResult(task_id="docs", success=True, response=Response(content="Docs"), duration_ms=80),
        ]

        async def _test():
            with patch.object(Fleet, "run", new_callable=AsyncMock, return_value=mock_fleet_results):
                return await coord.run("Build something")

        result = run(_test())
        assert result.success is True
        assert len(result.agent_results) == 4
        assert result.total_duration_ms >= 0

    def test_run_with_failure(self):
        """Test that run() reports failure when an agent fails."""
        config = CopexConfig()
        coord = SquadCoordinator(config, team=SquadTeam.default())

        mock_fleet_results = [
            FleetResult(task_id="lead", success=True, response=Response(content="Plan"), duration_ms=100),
            FleetResult(task_id="developer", success=False, error=RuntimeError("compile error"), duration_ms=50),
            FleetResult(task_id="tester", success=True, response=Response(content="Tests"), duration_ms=150),
            FleetResult(task_id="docs", success=True, response=Response(content="Docs"), duration_ms=80),
        ]

        async def _test():
            with patch.object(Fleet, "run", new_callable=AsyncMock, return_value=mock_fleet_results):
                return await coord.run("Build something")

        result = run(_test())
        assert result.success is False
        assert result.agent_results[1].error == "compile error"


# ===========================================================================
# 18. Squad CLI integration
# ===========================================================================


class TestSquadCLIIntegration:

    def test_squad_command_exists(self):
        from typer.testing import CliRunner

        from copex.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["squad", "--help"])
        assert result.exit_code == 0, f"squad --help failed: {result.output}"

    def test_squad_help_shows_options(self):
        import re

        from typer.testing import CliRunner

        from copex.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["squad", "--help"])
        plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--json" in plain
        assert "--model" in plain
        assert "--reasoning" in plain

    def test_squad_no_prompt_exits_nonzero(self):
        from typer.testing import CliRunner

        from copex.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["squad"], input="")
        assert result.exit_code != 0


# ===========================================================================
# 19. SquadTeam.from_repo — dynamic team creation
# ===========================================================================


class TestFromRepoPythonProject:

    def test_from_repo_python_project(self, tmp_path):
        """Repo with src/, tests/, docs/ → Lead + Developer + Tester + Docs."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("x = 1")
        (tmp_path / "tests").mkdir()
        (tmp_path / "docs").mkdir()

        team = SquadTeam.from_repo(tmp_path)
        roles = team.roles
        assert SquadRole.LEAD in roles
        assert SquadRole.DEVELOPER in roles
        assert SquadRole.TESTER in roles
        assert SquadRole.DOCS in roles
        assert SquadRole.DEVOPS not in roles
        assert SquadRole.FRONTEND not in roles
        assert SquadRole.BACKEND not in roles


class TestFromRepoMinimal:

    def test_from_repo_minimal(self, tmp_path):
        """Just source files → Lead + Developer."""
        (tmp_path / "app.js").write_text("console.log('hi')")

        team = SquadTeam.from_repo(tmp_path)
        roles = team.roles
        assert roles == [SquadRole.LEAD, SquadRole.DEVELOPER]


class TestFromRepoWithDocker:

    def test_from_repo_with_docker(self, tmp_path):
        """Dockerfile present → includes DevOps."""
        (tmp_path / "main.go").write_text("package main")
        (tmp_path / "Dockerfile").write_text("FROM alpine")

        team = SquadTeam.from_repo(tmp_path)
        roles = team.roles
        assert SquadRole.DEVOPS in roles
        assert SquadRole.LEAD in roles
        assert SquadRole.DEVELOPER in roles

    def test_from_repo_with_docker_compose(self, tmp_path):
        (tmp_path / "main.py").write_text("pass")
        (tmp_path / "docker-compose.yml").write_text("version: '3'")

        team = SquadTeam.from_repo(tmp_path)
        assert SquadRole.DEVOPS in team.roles

    def test_from_repo_with_github_workflows(self, tmp_path):
        (tmp_path / "main.py").write_text("pass")
        (tmp_path / ".github" / "workflows").mkdir(parents=True)

        team = SquadTeam.from_repo(tmp_path)
        assert SquadRole.DEVOPS in team.roles

    def test_from_repo_with_makefile(self, tmp_path):
        (tmp_path / "main.py").write_text("pass")
        (tmp_path / "Makefile").write_text("all:")

        team = SquadTeam.from_repo(tmp_path)
        assert SquadRole.DEVOPS in team.roles


class TestFromRepoFrontendBackend:

    def test_from_repo_frontend_backend(self, tmp_path):
        """Has components/ + api/ + src/ → includes Frontend + Backend."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.ts").write_text("export default {}")
        (tmp_path / "components").mkdir()
        (tmp_path / "api").mkdir()

        team = SquadTeam.from_repo(tmp_path)
        roles = team.roles
        assert SquadRole.FRONTEND in roles
        assert SquadRole.BACKEND in roles

    def test_from_repo_frontend_only(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "index.js").write_text("")
        (tmp_path / "pages").mkdir()

        team = SquadTeam.from_repo(tmp_path)
        assert SquadRole.FRONTEND in team.roles
        assert SquadRole.BACKEND not in team.roles

    def test_from_repo_backend_only(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "server.py").write_text("")
        (tmp_path / "services").mkdir()

        team = SquadTeam.from_repo(tmp_path)
        assert SquadRole.BACKEND in team.roles
        assert SquadRole.FRONTEND not in team.roles

    def test_from_repo_no_src_dir_no_frontend_backend(self, tmp_path):
        """Frontend/backend detection requires src/ dir."""
        (tmp_path / "app.py").write_text("")
        (tmp_path / "components").mkdir()
        (tmp_path / "api").mkdir()

        team = SquadTeam.from_repo(tmp_path)
        assert SquadRole.FRONTEND not in team.roles
        assert SquadRole.BACKEND not in team.roles


class TestFromRepoEmptyDir:

    def test_from_repo_empty_dir(self, tmp_path):
        """Nothing → Lead + Developer minimum."""
        team = SquadTeam.from_repo(tmp_path)
        roles = team.roles
        assert SquadRole.LEAD in roles
        assert SquadRole.DEVELOPER in roles
        assert len(roles) == 2


class TestFromRepoCustomPath:

    def test_from_repo_custom_path(self, tmp_path):
        project = tmp_path / "myproject"
        project.mkdir()
        (project / "lib.rs").write_text("fn main() {}")
        (project / "tests").mkdir()

        team = SquadTeam.from_repo(project)
        assert SquadRole.DEVELOPER in team.roles
        assert SquadRole.TESTER in team.roles


class TestDefaultStillWorks:

    def test_default_still_works(self):
        """Backward compat: default() returns static 4-agent team."""
        team = SquadTeam.default()
        assert len(team.agents) == 4
        assert team.roles == [
            SquadRole.LEAD, SquadRole.DEVELOPER,
            SquadRole.TESTER, SquadRole.DOCS,
        ]


class TestNewRoleDependencies:

    def test_devops_depends_on_lead(self):
        config = CopexConfig()
        team = SquadTeam(agents=[
            SquadAgent.default_for_role(SquadRole.LEAD),
            SquadAgent.default_for_role(SquadRole.DEVOPS),
        ])
        coord = SquadCoordinator(config, team=team)
        agent = SquadAgent.default_for_role(SquadRole.DEVOPS)
        deps = coord._get_dependencies(agent, {"lead": "lead"})
        assert deps == ["lead"]

    def test_frontend_depends_on_lead(self):
        config = CopexConfig()
        coord = SquadCoordinator(config, team=SquadTeam.default())
        agent = SquadAgent.default_for_role(SquadRole.FRONTEND)
        deps = coord._get_dependencies(agent, {"lead": "lead"})
        assert deps == ["lead"]

    def test_backend_depends_on_lead(self):
        config = CopexConfig()
        coord = SquadCoordinator(config, team=SquadTeam.default())
        agent = SquadAgent.default_for_role(SquadRole.BACKEND)
        deps = coord._get_dependencies(agent, {"lead": "lead"})
        assert deps == ["lead"]

    def test_tester_depends_on_lower_phases(self):
        config = CopexConfig()
        team = SquadTeam(agents=[
            SquadAgent.default_for_role(SquadRole.LEAD),
            SquadAgent.default_for_role(SquadRole.DEVELOPER),
            SquadAgent.default_for_role(SquadRole.FRONTEND),
            SquadAgent.default_for_role(SquadRole.TESTER),
        ])
        coord = SquadCoordinator(config, team=team)
        agent = SquadAgent.default_for_role(SquadRole.TESTER)
        task_ids = {
            "lead": "lead",
            "developer": "developer",
            "frontend": "frontend",
        }
        deps = coord._get_dependencies(agent, task_ids)
        assert "developer" in deps
        assert "frontend" in deps
        assert "lead" in deps  # phase 3 depends on phase 1+2

    def test_tester_depends_on_lead_only(self):
        config = CopexConfig()
        coord = SquadCoordinator(config, team=SquadTeam.default())
        agent = SquadAgent.default_for_role(SquadRole.TESTER)
        deps = coord._get_dependencies(agent, {"lead": "lead"})
        assert deps == ["lead"]

    def test_docs_depends_on_all_lower_phases(self):
        config = CopexConfig()
        team = SquadTeam(agents=[
            SquadAgent.default_for_role(SquadRole.LEAD),
            SquadAgent.default_for_role(SquadRole.DEVELOPER),
            SquadAgent.default_for_role(SquadRole.DEVOPS),
            SquadAgent.default_for_role(SquadRole.TESTER),
            SquadAgent.default_for_role(SquadRole.DOCS),
        ])
        coord = SquadCoordinator(config, team=team)
        agent = SquadAgent.default_for_role(SquadRole.DOCS)
        task_ids = {
            "lead": "lead",
            "developer": "developer",
            "tester": "tester",
            "devops": "devops",
        }
        deps = coord._get_dependencies(agent, task_ids)
        assert "developer" in deps
        assert "tester" in deps
        assert "devops" in deps
        assert "lead" in deps  # phase 4 depends on all lower phases


class TestFromRepoTestPatterns:

    def test_test_prefix_pattern(self, tmp_path):
        (tmp_path / "test_main.py").write_text("pass")
        team = SquadTeam.from_repo(tmp_path)
        assert SquadRole.TESTER in team.roles

    def test_test_suffix_pattern(self, tmp_path):
        (tmp_path / "main_test.go").write_text("package main")
        team = SquadTeam.from_repo(tmp_path)
        assert SquadRole.TESTER in team.roles

    def test_spec_pattern(self, tmp_path):
        (tmp_path / "main.spec.ts").write_text("")
        team = SquadTeam.from_repo(tmp_path)
        assert SquadRole.TESTER in team.roles

    def test_multiple_md_docs_signal(self, tmp_path):
        (tmp_path / "README.md").write_text("# Readme")
        (tmp_path / "CONTRIBUTING.md").write_text("# Contributing")
        team = SquadTeam.from_repo(tmp_path)
        assert SquadRole.DOCS in team.roles


# ===========================================================================
# Freeform roles, phases, save/load
# ===========================================================================


class TestFreeformRoles:

    def test_custom_role_string(self):
        agent = SquadAgent.default_for_role("security_engineer")
        assert agent.role == "security_engineer"
        assert agent.name == "Security Engineer"
        assert agent.emoji == "🔹"  # fallback emoji
        assert agent.phase == 2  # default phase
        assert "Security Engineer" in agent.system_prompt

    def test_custom_role_phase(self):
        agent = SquadAgent(
            name="Data Scientist",
            role="data_scientist",
            emoji="📊",
            system_prompt="You analyze data.",
            phase=2,
        )
        assert agent.phase == 2
        assert agent.role == "data_scientist"

    def test_known_role_gets_correct_phase(self):
        assert SquadAgent.default_for_role("lead").phase == 1
        assert SquadAgent.default_for_role("developer").phase == 2
        assert SquadAgent.default_for_role("tester").phase == 3
        assert SquadAgent.default_for_role("docs").phase == 4

    def test_freeform_role_dependencies(self):
        team = SquadTeam(agents=[
            SquadAgent(name="Lead", role="lead", emoji="🏗️",
                       system_prompt="Lead.", phase=1),
            SquadAgent(name="ML Engineer", role="ml_engineer", emoji="🤖",
                       system_prompt="ML.", phase=2),
            SquadAgent(name="QA", role="qa_specialist", emoji="✅",
                       system_prompt="QA.", phase=3),
        ])
        config = CopexConfig()
        coord = SquadCoordinator(config, team=team)

        # ML (phase 2) depends on Lead (phase 1)
        ml = team.get_agent("ml_engineer")
        deps = coord._get_dependencies(ml, {"lead": "lead"})
        assert deps == ["lead"]

        # QA (phase 3) depends on Lead (phase 1) and ML (phase 2)
        qa = team.get_agent("qa_specialist")
        deps = coord._get_dependencies(qa, {"lead": "lead", "ml_engineer": "ml_engineer"})
        assert "lead" in deps
        assert "ml_engineer" in deps

    def test_mixed_known_and_custom_roles(self):
        team = SquadTeam(agents=[
            SquadAgent.default_for_role(SquadRole.LEAD),
            SquadAgent.default_for_role("api_designer"),
            SquadAgent.default_for_role(SquadRole.TESTER),
        ])
        assert team.roles == ["lead", "api_designer", "tester"]
        assert team.get_agent("api_designer") is not None


class TestSquadTeamPersistence:

    def test_save_and_load(self, tmp_path):
        team = SquadTeam(agents=[
            SquadAgent(name="Lead", role="lead", emoji="🏗️",
                       system_prompt="Lead.", phase=1),
            SquadAgent(name="ML Eng", role="ml_engineer", emoji="🤖",
                       system_prompt="Do ML.", phase=2),
        ])
        config_path = tmp_path / "squad.json"
        team.save(config_path)

        loaded = SquadTeam.load(config_path)
        assert loaded is not None
        assert len(loaded.agents) == 2
        assert loaded.agents[0].role == "lead"
        assert loaded.agents[1].role == "ml_engineer"
        assert loaded.agents[1].name == "ML Eng"
        assert loaded.agents[1].emoji == "🤖"
        assert loaded.agents[1].system_prompt == "Do ML."
        assert loaded.agents[1].phase == 2

    def test_load_nonexistent(self, tmp_path):
        result = SquadTeam.load(tmp_path / "nope.json")
        assert result is None

    def test_load_invalid_json(self, tmp_path):
        config_path = tmp_path / "squad.json"
        config_path.write_text("not json", encoding="utf-8")
        result = SquadTeam.load(config_path)
        assert result is None

    def test_save_creates_directory(self, tmp_path):
        team = SquadTeam.default()
        config_path = tmp_path / "subdir" / "squad.json"
        team.save(config_path)
        assert config_path.is_file()
        loaded = SquadTeam.load(config_path)
        assert loaded is not None
        assert len(loaded.agents) == 4


# ===========================================================================
# Subtask parallelism
# ===========================================================================


class TestSubtasks:
    """Tests for SquadAgent subtask parallelism within a phase."""

    def test_agent_default_has_no_subtasks(self):
        agent = SquadAgent.default_for_role("developer")
        assert agent.subtasks == []

    def test_agent_with_subtasks(self):
        agent = SquadAgent(
            name="Dev", role="developer", emoji="🔧",
            system_prompt="Build it.", phase=2,
            subtasks=["API routes", "Database models", "Middleware"],
        )
        assert len(agent.subtasks) == 3
        assert agent.subtasks[0] == "API routes"

    def test_add_tasks_fans_out_subtasks(self):
        """Agent with subtasks creates multiple fleet tasks."""
        config = CopexConfig()
        team = SquadTeam(agents=[
            SquadAgent(name="Lead", role="lead", emoji="🏗️",
                       system_prompt="Lead.", phase=1),
            SquadAgent(name="Dev", role="developer", emoji="🔧",
                       system_prompt="Build.", phase=2,
                       subtasks=["Module A", "Module B"]),
        ])
        coord = SquadCoordinator(config, team=team)
        fleet = Fleet(config, fleet_config=FleetConfig(max_concurrent=3))
        task_ids = coord._add_tasks(fleet, "Do work")

        # Lead gets one task
        assert task_ids["lead"] == "lead"
        # Developer gets pipe-joined subtask IDs
        sub_ids = task_ids["developer"].split("|")
        assert len(sub_ids) == 2
        assert sub_ids[0] == "developer__sub1"
        assert sub_ids[1] == "developer__sub2"

    def test_subtask_dependencies_expand(self):
        """When a dep agent has subtasks, all subtask IDs are included."""
        config = CopexConfig()
        lead_agent = SquadAgent(
            name="Lead", role="lead", emoji="🏗️",
            system_prompt="Lead.", phase=1,
            subtasks=["Architecture review", "Security review"],
        )
        dev_agent = SquadAgent(
            name="Dev", role="developer", emoji="🔧",
            system_prompt="Build.", phase=2,
        )
        team = SquadTeam(agents=[lead_agent, dev_agent])
        coord = SquadCoordinator(config, team=team)

        # Simulate lead having pipe-joined subtask IDs
        task_ids = {"lead": "lead__sub1|lead__sub2"}
        deps = coord._get_dependencies(dev_agent, task_ids)
        assert "lead__sub1" in deps
        assert "lead__sub2" in deps
        assert len(deps) == 2

    def test_build_result_merges_subtasks(self):
        """Multiple fleet results for subtasks are merged into one agent result."""
        config = CopexConfig()
        dev_agent = SquadAgent(
            name="Dev", role="developer", emoji="🔧",
            system_prompt="Build.", phase=2,
            subtasks=["API routes", "Database models"],
        )
        team = SquadTeam(agents=[dev_agent])
        coord = SquadCoordinator(config, team=team)

        task_ids = {"developer": "developer__sub1|developer__sub2"}
        fleet_results = [
            FleetResult(
                task_id="developer__sub1", success=True,
                response=Response(content="Routes done"), duration_ms=100,
                prompt_tokens=50, completion_tokens=30,
            ),
            FleetResult(
                task_id="developer__sub2", success=True,
                response=Response(content="Models done"), duration_ms=80,
                prompt_tokens=40, completion_tokens=20,
            ),
        ]

        result = coord._build_result(fleet_results, task_ids)
        assert len(result.agent_results) == 1
        ar = result.agent_results[0]
        assert ar.success is True
        assert "API routes" in ar.content
        assert "Routes done" in ar.content
        assert "Database models" in ar.content
        assert "Models done" in ar.content
        assert ar.duration_ms == 180  # sum
        assert ar.prompt_tokens == 90
        assert ar.completion_tokens == 50

    def test_build_result_subtask_partial_failure(self):
        """If one subtask fails, the merged result reports failure."""
        config = CopexConfig()
        dev_agent = SquadAgent(
            name="Dev", role="developer", emoji="🔧",
            system_prompt="Build.", phase=2,
            subtasks=["Module A", "Module B"],
        )
        team = SquadTeam(agents=[dev_agent])
        coord = SquadCoordinator(config, team=team)

        task_ids = {"developer": "developer__sub1|developer__sub2"}
        fleet_results = [
            FleetResult(
                task_id="developer__sub1", success=True,
                response=Response(content="A done"), duration_ms=100,
            ),
            FleetResult(
                task_id="developer__sub2", success=False,
                error=RuntimeError("compile error"), duration_ms=50,
            ),
        ]

        result = coord._build_result(fleet_results, task_ids)
        assert result.success is False
        ar = result.agent_results[0]
        assert ar.success is False
        assert "compile error" in ar.error

    def test_subtasks_persist_save_load(self, tmp_path):
        """Subtasks survive save/load roundtrip."""
        team = SquadTeam(agents=[
            SquadAgent(
                name="Dev", role="developer", emoji="🔧",
                system_prompt="Build.", phase=2,
                subtasks=["API", "DB", "Auth"],
            ),
        ])
        config_path = tmp_path / "squad.json"
        team.save(config_path)

        loaded = SquadTeam.load(config_path)
        assert loaded is not None
        assert loaded.agents[0].subtasks == ["API", "DB", "Auth"]

    def test_subtasks_not_saved_when_empty(self, tmp_path):
        """Empty subtasks list is omitted from JSON."""
        team = SquadTeam(agents=[
            SquadAgent.default_for_role("lead"),
        ])
        config_path = tmp_path / "squad.json"
        team.save(config_path)

        data = json.loads(config_path.read_text(encoding="utf-8"))
        assert "subtasks" not in data[0]

    def test_mixed_subtask_and_normal_agents(self):
        """Team with both subtask and normal agents works correctly."""
        config = CopexConfig()
        team = SquadTeam(agents=[
            SquadAgent(name="Lead", role="lead", emoji="🏗️",
                       system_prompt="Lead.", phase=1),
            SquadAgent(name="Dev", role="developer", emoji="🔧",
                       system_prompt="Build.", phase=2,
                       subtasks=["Frontend", "Backend"]),
            SquadAgent(name="Tester", role="tester", emoji="🧪",
                       system_prompt="Test.", phase=3),
        ])
        coord = SquadCoordinator(config, team=team)
        fleet = Fleet(config, fleet_config=FleetConfig(max_concurrent=3))
        task_ids = coord._add_tasks(fleet, "Build app")

        # Lead: single task
        assert task_ids["lead"] == "lead"
        # Dev: two subtasks
        assert "developer__sub1" in task_ids["developer"]
        assert "developer__sub2" in task_ids["developer"]
        # Tester depends on lead AND both dev subtasks
        tester = team.get_agent("tester")
        deps = coord._get_dependencies(tester, task_ids)
        assert "lead" in deps
        assert "developer__sub1" in deps
        assert "developer__sub2" in deps
