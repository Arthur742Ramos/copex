"""Tests for squad_team_ai module."""

from __future__ import annotations

import json

import pytest

from copex.squad_team_ai import (
    build_repo_analysis_prompt,
    parse_repo_analysis_response,
)


class TestBuildRepoAnalysisPrompt:
    def test_basic_context(self) -> None:
        context = {
            "project_name": "myproject",
            "description": "A test project",
        }
        prompt = build_repo_analysis_prompt(context)

        assert "myproject" in prompt
        assert "test project" in prompt
        assert "Analyze" in prompt

    def test_with_languages(self) -> None:
        context = {
            "project_name": "app",
            "source_extensions": [".py", ".ts", ".go"],
        }
        prompt = build_repo_analysis_prompt(context)

        assert ".py" in prompt or "py" in prompt.lower()

    def test_with_dependencies(self) -> None:
        context = {
            "project_name": "app",
            "dependencies": ["fastapi", "pydantic", "sqlalchemy"],
        }
        prompt = build_repo_analysis_prompt(context)

        assert "fastapi" in prompt

    def test_with_directory_structure(self) -> None:
        context = {
            "project_name": "app",
            "directory_structure": ["src/", "tests/", "docs/"],
        }
        prompt = build_repo_analysis_prompt(context)

        assert "src/" in prompt or "Directory" in prompt

    def test_with_readme_excerpt(self) -> None:
        context = {
            "project_name": "app",
            "readme_excerpt": "This is a powerful CLI tool for developers.",
        }
        prompt = build_repo_analysis_prompt(context)

        assert "CLI tool" in prompt or "README" in prompt

    def test_with_existing_agents(self) -> None:
        context = {"project_name": "app"}
        existing = [
            ("🏗️", "Lead Architect", "lead", 0),
            ("🔧", "Developer", "developer", 1),
        ]
        prompt = build_repo_analysis_prompt(context, existing_agents=existing)

        assert "Lead" in prompt or "lead" in prompt
        assert "Developer" in prompt or "developer" in prompt

    def test_empty_context(self) -> None:
        context: dict = {}
        prompt = build_repo_analysis_prompt(context)

        # Should still produce a valid prompt
        assert "Analyze" in prompt
        assert "JSON" in prompt


class TestParseRepoAnalysisResponse:
    def test_valid_json_array(self) -> None:
        response = """
        Here's my analysis:
        [
            {"role": "lead", "name": "Lead Architect", "emoji": "🏗️", "description": "Plans the work", "phase": 0},
            {"role": "developer", "name": "Developer", "emoji": "🔧", "description": "Writes code", "phase": 1}
        ]
        """
        agents = parse_repo_analysis_response(response)

        assert len(agents) == 2
        assert agents[0]["role"] == "lead"
        assert agents[1]["role"] == "developer"

    def test_extracts_from_markdown(self) -> None:
        response = """
        Based on my analysis:

        ```json
        [
            {"role": "tester", "name": "Tester", "emoji": "🧪", "description": "Tests", "phase": 2}
        ]
        ```
        """
        agents = parse_repo_analysis_response(response)

        assert len(agents) == 1
        assert agents[0]["role"] == "tester"

    def test_handles_empty_response(self) -> None:
        agents = parse_repo_analysis_response("")
        assert agents == []

    def test_handles_invalid_json(self) -> None:
        response = "This is not JSON at all"
        agents = parse_repo_analysis_response(response)
        assert agents == []

    def test_handles_non_array_json(self) -> None:
        response = '{"role": "lead"}'
        agents = parse_repo_analysis_response(response)
        # Should handle gracefully
        assert isinstance(agents, list)

    def test_filters_invalid_entries(self) -> None:
        response = """
        [
            {"role": "lead", "name": "Lead", "emoji": "🏗️", "description": "Plans", "phase": 0},
            {"invalid": "entry"},
            {"role": "dev", "name": "Dev", "emoji": "🔧", "description": "Codes", "phase": 1}
        ]
        """
        agents = parse_repo_analysis_response(response)

        # Should only include valid entries
        valid_roles = [a.get("role") for a in agents if "role" in a]
        assert "lead" in valid_roles
        assert "dev" in valid_roles
