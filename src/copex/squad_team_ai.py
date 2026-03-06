"""Helpers for SquadTeam AI repo analysis prompt construction and parsing."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from typing import Any

from copex.json_utils import extract_json_array

logger = logging.getLogger(__name__)


def build_repo_analysis_prompt(
    context: Mapping[str, Any],
    *,
    existing_agents: Iterable[tuple[str, str, str, int]] | None = None,
) -> str:
    """Build the repo-analysis prompt used by ``SquadTeam.from_repo_ai``."""
    prompt_parts = [
        "Analyze this repository and determine what development team roles are needed.",
        "",
        "# Repository Context",
    ]

    project_name = context.get("project_name")
    if isinstance(project_name, str) and project_name.strip():
        prompt_parts.append(f"Project: {project_name}")

    description = context.get("description")
    if isinstance(description, str) and description.strip():
        prompt_parts.append(f"Description: {description}")

    source_extensions = context.get("source_extensions")
    if isinstance(source_extensions, list):
        languages = [str(ext).strip() for ext in source_extensions if str(ext).strip()]
        if languages:
            prompt_parts.append(f"Languages: {', '.join(languages)}")

    dependencies = context.get("dependencies")
    if isinstance(dependencies, list):
        deps = [str(dep).strip() for dep in dependencies if str(dep).strip()]
        if deps:
            prompt_parts.append(f"Dependencies: {', '.join(deps[:5])}")

    directory_structure = context.get("directory_structure")
    if isinstance(directory_structure, list):
        entries = [str(entry) for entry in directory_structure[:30]]
        if entries:
            struct = "\n".join(entries)
            prompt_parts.append(f"\nDirectory structure:\n{struct}")

    readme_excerpt = context.get("readme_excerpt")
    if isinstance(readme_excerpt, str) and readme_excerpt.strip():
        prompt_parts.append(f"\nREADME excerpt:\n{readme_excerpt[:1000]}")

    agents = list(existing_agents or [])
    if agents:
        prompt_parts.append("")
        prompt_parts.append("# Current Team Configuration")
        prompt_parts.append("")
        prompt_parts.append(
            "This repository already has a team configured. "
            "Review it and decide whether to keep it as-is, "
            "modify roles, or add/remove agents:"
        )
        for emoji, name, role, phase in agents:
            prompt_parts.append(f"- {emoji} {name} (role: {role}, phase: {phase})")

    prompt_parts.extend([
        "",
        "# Instructions",
        "",
        "Determine the ideal team composition for this repository.",
        "You may create ANY roles that make sense — you are NOT limited to a predefined list.",
        "Common roles include lead, developer, tester, docs, devops, frontend, backend,",
        "but you can invent specialized roles like security_engineer, data_scientist,",
        "api_designer, performance_engineer, etc.",
        "",
        "Respond ONLY with a JSON array of objects, each with:",
        '- "role": short snake_case identifier (e.g. "security_engineer")',
        '- "name": human-readable name (e.g. "Security Engineer")',
        '- "emoji": single emoji for display',
        '- "prompt": system prompt describing the agent\'s expertise and focus',
        '- "phase": execution order (1=analyze first, 2=build, 3=verify, 4=document)',
        '- "depends_on": (optional) list of role IDs this agent depends on',
        '- "subtasks": (optional) list of strings describing parallel work items.',
        '  When provided, the agent\'s work is split into parallel Fleet tasks —',
        '  one per subtask — all running within the same phase.',
        '  Use subtasks when an agent has clearly separable concerns',
        '  (e.g., a developer handling 3 independent modules).',
        '- "retries": (optional) non-negative integer for squad-level retries of this role.',
        '- "retry_delay": (optional) non-negative float base delay in seconds for retries.',
        "",
        "Rules:",
        "- Always include a 'lead' role with phase 1",
        "- Keep the team small (3-5 roles typical)",
        "- Only include roles clearly needed for THIS specific repository",
        "- Use depends_on for explicit DAG dependencies when useful.",
        "- If depends_on is omitted, phase ordering determines dependencies.",
        "- Subtasks are optional — only add them when work is naturally parallelizable",
        "- retries defaults to 1 if omitted; retry_delay defaults to immediate retry.",
        "- If a current team is shown above, use it as a starting point —"
        " keep roles that still make sense, remove unnecessary ones, and add new ones if needed",
        "",
        "Example response:",
        "[",
        '  {"role": "lead", "name": "Lead Architect", "emoji": "🏗️", '
        '"prompt": "You are the Lead Architect. Analyze the task...", "phase": 1},',
        '  {"role": "developer", "name": "Developer", "emoji": "🔧", '
        '"prompt": "You are the Developer. Implement the task...", "phase": 2,',
        '   "subtasks": ["Core module implementation", "API endpoint handlers"],',
        '   "retries": 2, "retry_delay": 1.5},',
        '  {"role": "tester", "name": "Tester", "emoji": "🧪", '
        '"prompt": "You are the Tester. Write comprehensive tests...", "phase": 3}',
        "]",
    ])

    return "\n".join(prompt_parts)


def parse_repo_analysis_response(content: str) -> list[Any]:
    """Extract the JSON array payload from an AI repo-analysis response."""
    if not content or not content.strip():
        return []
    try:
        return extract_json_array(content.strip())
    except (ValueError, json.JSONDecodeError):
        logger.debug("Failed to parse repo analysis response as JSON array")
        return []
