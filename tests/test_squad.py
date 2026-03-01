"""Tests for squad functionality.

Tests the SquadCoordinator, SquadTeam, SquadAgent, SquadResult,
and CLI integration for the built-in multi-agent orchestration.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from copex.config import CopexConfig
from copex.fleet import Fleet, FleetConfig, FleetResult
from copex.squad import (
    SquadAgent,
    SquadAgentResult,
    SquadCoordinator,
    SquadResult,
    SquadRole,
    SquadTeam,
    _ROLE_EMOJIS,
    _ROLE_PROMPTS,
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
        task_ids = {SquadRole.LEAD: "lead"}
        prompt = coord._build_agent_prompt(agent, "Build X", task_ids)
        assert "Developer" in prompt
        assert "Build X" in prompt
        assert "{{task:lead.content}}" in prompt

    def test_build_agent_prompt_tester(self):
        config = CopexConfig()
        coord = SquadCoordinator(config)
        agent = coord.team.get_agent(SquadRole.TESTER)
        task_ids = {SquadRole.LEAD: "lead"}
        prompt = coord._build_agent_prompt(agent, "Build X", task_ids)
        assert "Tester" in prompt
        assert "{{task:lead.content}}" in prompt

    def test_get_dependencies_lead(self):
        config = CopexConfig()
        coord = SquadCoordinator(config)
        deps = coord._get_dependencies(SquadRole.LEAD, {})
        assert deps == []

    def test_get_dependencies_developer(self):
        config = CopexConfig()
        coord = SquadCoordinator(config)
        deps = coord._get_dependencies(SquadRole.DEVELOPER, {SquadRole.LEAD: "lead"})
        assert deps == ["lead"]

    def test_get_dependencies_tester(self):
        config = CopexConfig()
        coord = SquadCoordinator(config)
        deps = coord._get_dependencies(SquadRole.TESTER, {SquadRole.LEAD: "lead"})
        assert deps == ["lead"]

    def test_get_dependencies_no_lead(self):
        config = CopexConfig()
        coord = SquadCoordinator(config)
        deps = coord._get_dependencies(SquadRole.DEVELOPER, {})
        assert deps == []


# ===========================================================================
# 7. SquadCoordinator — result building
# ===========================================================================


class TestSquadCoordinatorResultBuilding:

    def test_build_result_success(self):
        config = CopexConfig()
        coord = SquadCoordinator(config)
        task_ids = {
            SquadRole.LEAD: "lead",
            SquadRole.DEVELOPER: "developer",
            SquadRole.TESTER: "tester",
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
        task_ids = {SquadRole.LEAD: "lead", SquadRole.DEVELOPER: "developer"}
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
        task_ids = {SquadRole.LEAD: "lead"}
        fleet_results = [
            FleetResult(task_id="lead", success=True, response=Response(content="ok"), duration_ms=100),
            FleetResult(task_id="unknown", success=True, response=Response(content="?"), duration_ms=50),
        ]

        result = coord._build_result(fleet_results, task_ids)
        assert len(result.agent_results) == 1  # Unknown ignored

    def test_build_result_no_response(self):
        config = CopexConfig()
        coord = SquadCoordinator(config)
        task_ids = {SquadRole.LEAD: "lead"}
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
        coord = SquadCoordinator(config)
        fleet = Fleet(config)
        task_ids = coord._add_tasks(fleet, "Build an API")

        assert SquadRole.LEAD in task_ids
        assert SquadRole.DEVELOPER in task_ids
        assert SquadRole.TESTER in task_ids
        assert SquadRole.DOCS in task_ids
        assert len(fleet._tasks) == 4

    def test_add_tasks_dependencies(self):
        from copex.fleet import Fleet

        config = CopexConfig()
        coord = SquadCoordinator(config)
        fleet = Fleet(config)
        task_ids = coord._add_tasks(fleet, "Build an API")

        lead_task = next(t for t in fleet._tasks if t.id == task_ids[SquadRole.LEAD])
        dev_task = next(t for t in fleet._tasks if t.id == task_ids[SquadRole.DEVELOPER])
        tester_task = next(t for t in fleet._tasks if t.id == task_ids[SquadRole.TESTER])

        assert lead_task.depends_on == []
        assert dev_task.depends_on == [task_ids[SquadRole.LEAD]]
        assert tester_task.depends_on == [task_ids[SquadRole.LEAD]]

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
            SquadRole.LEAD: "lead",
            SquadRole.DEVELOPER: "developer",
            SquadRole.TESTER: "tester",
        }
        deps = coord._get_dependencies(SquadRole.DOCS, task_ids)
        assert "developer" in deps
        assert "tester" in deps
        assert "lead" not in deps


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
            SquadRole.LEAD: "lead",
            SquadRole.DEVELOPER: "developer",  # not in team
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
        coord = SquadCoordinator(config)

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
        coord = SquadCoordinator(config)

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
        from copex.cli import app
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["squad", "--help"])
        assert result.exit_code == 0, f"squad --help failed: {result.output}"

    def test_squad_help_shows_options(self):
        from copex.cli import app
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["squad", "--help"])
        assert "--json" in result.output
        assert "--model" in result.output
        assert "--reasoning" in result.output

    def test_squad_no_prompt_exits_nonzero(self):
        from copex.cli import app
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["squad"], input="")
        assert result.exit_code != 0
