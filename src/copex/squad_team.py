"""SquadTeam - team composition and persistence utilities."""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore

from copex.config import CopexConfig
from copex.json_utils import extract_json_array
from copex.squad import (
    SquadAgent,
    SquadConfigError,
    SquadRole,
    _KNOWN_ROLE_PHASES,
    _ROLE_DESCRIPTIONS,
    _ROLE_EMOJIS,
    _ROLE_PROMPTS,
    _SQUAD_DIR_NAME,
    _SQUAD_TEAM_FILE_NAME,
    _clamp_phase,
    _ensure_squad_workspace,
    _gather_repo_context,
    _has_backend,
    _has_devops,
    _has_docs,
    _has_frontend,
    _has_source_files,
    _has_test_files,
    _normalize_depends_on_roles,
    _normalize_role_identifier,
)

logger = logging.getLogger(__name__)

_SQUAD_LEGACY_DIR_NAME = ".copex"
_SQUAD_LEGACY_JSON_FILE_NAME = "squad.json"


def _clamp_retries(value: Any, fallback: int = 1) -> int:
    try:
        retries = int(value)
    except (TypeError, ValueError):
        retries = fallback
    return max(0, retries)


@dataclass
class SquadTeam:
    """A team of squad agents."""

    agents: list[SquadAgent] = field(default_factory=list)
    phase_gates: dict[int, bool] = field(default_factory=dict)
    SQUAD_FILE = Path(_SQUAD_DIR_NAME)
    SQUAD_TEAM_FILE = Path(_SQUAD_DIR_NAME) / _SQUAD_TEAM_FILE_NAME

    @classmethod
    def default(cls) -> SquadTeam:
        """Create the default team: Lead, Developer, Tester, Docs."""
        return cls(
            agents=[
                SquadAgent.default_for_role(SquadRole.LEAD),
                SquadAgent.default_for_role(SquadRole.DEVELOPER),
                SquadAgent.default_for_role(SquadRole.TESTER),
                SquadAgent.default_for_role(SquadRole.DOCS),
            ]
        )

    @classmethod
    def from_repo(cls, path: Path | None = None) -> SquadTeam:
        """Create a team adapted to the repo structure at path.

        Scans the directory for signals (source files, tests, docs, Docker,
        frontend/backend dirs) and builds a team with matching roles.
        Always includes LEAD. Falls back to Lead + Developer minimum.
        """
        root = path or Path.cwd()
        roles: list[SquadRole] = [SquadRole.LEAD]

        has_source = _has_source_files(root)
        has_tests = _has_test_files(root)
        has_docs = _has_docs(root)
        has_devops = _has_devops(root)
        has_src = (root / "src").is_dir()
        has_frontend = has_src and _has_frontend(root)
        has_backend = has_src and _has_backend(root)

        if has_source:
            roles.append(SquadRole.DEVELOPER)
        if has_tests:
            roles.append(SquadRole.TESTER)
        if has_docs:
            roles.append(SquadRole.DOCS)
        if has_devops:
            roles.append(SquadRole.DEVOPS)
        if has_frontend:
            roles.append(SquadRole.FRONTEND)
        if has_backend:
            roles.append(SquadRole.BACKEND)

        # Minimum viable team: Lead + Developer
        if SquadRole.DEVELOPER not in roles:
            roles.append(SquadRole.DEVELOPER)

        return cls(agents=[SquadAgent.default_for_role(r) for r in roles])

    @classmethod
    def _resolve_workspace_root(cls, path: Path | None = None) -> Path:
        if path is None:
            return Path.cwd()
        if path.suffix == ".toml":
            if path.parent.name == _SQUAD_DIR_NAME:
                return path.parent.parent
            return path.parent
        if path.suffix == ".json":
            if path.parent.name == _SQUAD_LEGACY_DIR_NAME:
                return path.parent.parent
            return path.parent
        if path.name in {_SQUAD_DIR_NAME, _SQUAD_LEGACY_DIR_NAME}:
            return path.parent
        return path

    @classmethod
    def _resolve_squad_file_path(cls, path: Path | None = None) -> Path:
        if path is None:
            return cls.SQUAD_TEAM_FILE
        if path.suffix == ".toml":
            return path
        if path.suffix == ".json":
            if path.parent.name == _SQUAD_LEGACY_DIR_NAME:
                return path.parent.parent / cls.SQUAD_TEAM_FILE
            return path.parent / cls.SQUAD_TEAM_FILE
        if path.name == _SQUAD_DIR_NAME:
            if path.is_file():
                return path
            return path / _SQUAD_TEAM_FILE_NAME
        return path / cls.SQUAD_TEAM_FILE

    @classmethod
    def _resolve_legacy_squad_file_path(cls, path: Path | None = None) -> Path:
        if path is None:
            return cls.SQUAD_FILE
        if path.name == _SQUAD_DIR_NAME:
            return path
        if path.suffix == ".json":
            if path.parent.name == _SQUAD_LEGACY_DIR_NAME:
                return path.parent.parent / _SQUAD_DIR_NAME
            return path.parent / _SQUAD_DIR_NAME
        if path.suffix == ".toml":
            if path.parent.name == _SQUAD_DIR_NAME:
                return path.parent.parent / _SQUAD_DIR_NAME
            return path.with_name(_SQUAD_DIR_NAME)
        return path / _SQUAD_DIR_NAME

    @classmethod
    def _resolve_legacy_json_file_path(cls, path: Path | None = None) -> Path:
        if path is not None and path.suffix == ".json":
            return path
        return cls._resolve_workspace_root(path) / _SQUAD_LEGACY_DIR_NAME / _SQUAD_LEGACY_JSON_FILE_NAME

    @staticmethod
    def _squad_role_description(agent: SquadAgent) -> str:
        known = _ROLE_DESCRIPTIONS.get(agent.role)
        if known:
            return known
        prefix = f"You are the {agent.name}. "
        if agent.system_prompt.startswith(prefix):
            description = agent.system_prompt[len(prefix):].strip()
            if description:
                return description
        text = agent.system_prompt.strip()
        if text:
            return text
        return f"Handles {agent.name} responsibilities"

    @classmethod
    def _from_squad_payload(cls, data: dict[str, Any]) -> SquadTeam | None:
        squad = data.get("squad")
        if not isinstance(squad, dict):
            return None
        raw_phases = squad.get("phases", [])
        phase_gates: dict[int, bool] = {}
        if isinstance(raw_phases, list):
            for item in raw_phases:
                if not isinstance(item, dict):
                    continue
                raw_phase = item.get("phase", item.get("id", item.get("number")))
                if raw_phase is None:
                    continue
                phase = _clamp_phase(raw_phase, fallback=2)
                phase_gates[phase] = bool(item.get("gate", False))

        lead_name_raw = squad.get("lead", "Lead")
        lead_name = str(lead_name_raw).strip() if lead_name_raw is not None else "Lead"
        if not lead_name:
            lead_name = "Lead"

        lead_prompt = f"You are {lead_name}. {_ROLE_PROMPTS['lead']}"
        agents: list[SquadAgent] = [
            SquadAgent(
                name=lead_name,
                role=SquadRole.LEAD.value,
                emoji=_ROLE_EMOJIS.get(SquadRole.LEAD.value, "🏗️"),
                system_prompt=lead_prompt,
                phase=1,
                depends_on=[],
            )
        ]

        raw_agents = squad.get("agents", [])
        if not isinstance(raw_agents, list):
            raw_agents = []

        seen_roles: set[str] = {SquadRole.LEAD.value}
        for item in raw_agents:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue

            role_id_raw = str(item.get("id", "")).strip().lower()
            role_id = role_id_raw or _normalize_role_identifier(name)
            if not role_id or role_id in seen_roles:
                continue
            seen_roles.add(role_id)

            role_description = str(item.get("role", "")).strip()
            prompt_override = str(item.get("prompt", "")).strip()
            if prompt_override:
                system_prompt = prompt_override
            elif role_description:
                system_prompt = f"You are the {name}. {role_description}"
            else:
                system_prompt = _ROLE_PROMPTS.get(
                    role_id,
                    f"You are the {name}. Focus on your area of expertise for this task.",
                )

            phase = _clamp_phase(
                item.get("phase", _KNOWN_ROLE_PHASES.get(role_id, 2)),
                fallback=_KNOWN_ROLE_PHASES.get(role_id, 2),
            )
            depends_on = _normalize_depends_on_roles(item.get("depends_on", []))
            subtasks_raw = item.get("subtasks", [])
            subtasks = (
                [str(s).strip() for s in subtasks_raw if str(s).strip()]
                if isinstance(subtasks_raw, list)
                else []
            )
            retries = _clamp_retries(item.get("retries", 1), fallback=1)

            emoji_raw = str(item.get("emoji", "")).strip()
            agents.append(
                SquadAgent(
                    name=name,
                    role=role_id,
                    emoji=emoji_raw or _ROLE_EMOJIS.get(role_id, "🔹"),
                    system_prompt=system_prompt,
                    phase=phase,
                    depends_on=depends_on,
                    subtasks=subtasks,
                    retries=retries,
                )
            )

        return cls(agents=agents, phase_gates=phase_gates)

    @classmethod
    def _from_legacy_json_payload(cls, data: Any) -> SquadTeam | None:
        if not isinstance(data, list):
            return None

        agents: list[SquadAgent] = []
        seen_roles: set[str] = set()
        for item in data:
            if not isinstance(item, dict) or not item.get("role"):
                continue
            role_str = str(item["role"]).strip().lower()
            if not role_str or role_str in seen_roles:
                continue
            seen_roles.add(role_str)
            phase = _clamp_phase(item.get("phase", _KNOWN_ROLE_PHASES.get(role_str, 2)), fallback=2)
            depends_on = _normalize_depends_on_roles(item.get("depends_on", []))
            subtasks_raw = item.get("subtasks", [])
            subtasks = (
                [str(s).strip() for s in subtasks_raw if str(s).strip()]
                if isinstance(subtasks_raw, list)
                else []
            )
            retries = _clamp_retries(item.get("retries", 1), fallback=1)
            agents.append(SquadAgent(
                name=item.get("name", role_str.replace("_", " ").title()),
                role=role_str,
                emoji=item.get("emoji", _ROLE_EMOJIS.get(role_str, "🔹")),
                system_prompt=item.get("prompt", _ROLE_PROMPTS.get(
                    role_str,
                    f"You are the {role_str.replace('_', ' ').title()}. "
                    "Focus on your area of expertise for this task.",
                )),
                phase=max(1, min(4, phase)),
                depends_on=depends_on,
                subtasks=subtasks,
                retries=retries,
            ))

        if not agents:
            return None
        if "lead" not in seen_roles:
            agents.insert(0, SquadAgent.default_for_role("lead"))
        return cls(agents=agents, phase_gates={})

    @classmethod
    def _load_legacy_json_file(
        cls,
        path: Path,
        *,
        warn_deprecated: bool = False,
    ) -> SquadTeam | None:
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to load legacy squad config from %s: %s", path, exc)
            return None

        team = cls._from_legacy_json_payload(data)
        if team is None:
            return None

        if warn_deprecated:
            msg = (
                f"Loaded deprecated squad config at {path}; "
                f"migrating to .squad/{_SQUAD_TEAM_FILE_NAME}"
            )
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            logger.warning(msg)
        else:
            logger.info("Loaded legacy squad team from %s: %s", path, [a.role for a in team.agents])
        return team

    def _to_squad_payload(self) -> dict[str, Any]:
        lead_agent = self.get_agent(SquadRole.LEAD.value)
        lead_name = lead_agent.name if lead_agent is not None else "Lead"
        serialized_agents: list[dict[str, Any]] = []
        for agent in self.agents:
            if agent.role == SquadRole.LEAD.value:
                continue
            item: dict[str, Any] = {
                "name": agent.name,
                "role": self._squad_role_description(agent),
                "phase": agent.phase,
            }
            normalized_name_role = _normalize_role_identifier(agent.name)
            if normalized_name_role != agent.role:
                item["id"] = agent.role
            if agent.depends_on:
                item["depends_on"] = list(agent.depends_on)
            if agent.subtasks:
                item["subtasks"] = agent.subtasks
            if agent.retries != 1:
                item["retries"] = _clamp_retries(agent.retries, fallback=1)
            serialized_agents.append(item)
        payload: dict[str, Any] = {
            "lead": lead_name,
            "agents": serialized_agents,
        }
        if self.phase_gates:
            payload["phases"] = [
                {"phase": phase, "gate": gate}
                for phase, gate in sorted(self.phase_gates.items())
            ]
        return {"squad": payload}

    @classmethod
    def load_squad_file(cls, path: Path | None = None) -> SquadTeam | None:
        """Load team composition from .squad TOML."""
        primary = cls._resolve_squad_file_path(path)
        legacy = cls._resolve_legacy_squad_file_path(path)
        candidates = [primary]
        if legacy not in candidates:
            candidates.append(legacy)
        for config_path in candidates:
            if not config_path.is_file():
                continue
            try:
                with open(config_path, "rb") as f:
                    data = tomllib.load(f)
                if not isinstance(data, dict):
                    continue
                team = cls._from_squad_payload(data)
                if team and team.agents:
                    logger.info(f"Loaded squad team from {config_path}: {[a.role for a in team.agents]}")
                    return team
            except Exception as e:
                logger.warning(f"Failed to load .squad file from {config_path}: {e}")

        legacy_json = cls._resolve_legacy_json_file_path(path)
        if not primary.is_file() and legacy_json.is_file():
            team = cls._load_legacy_json_file(legacy_json, warn_deprecated=True)
            if team is None:
                return None
            try:
                team.save_squad_file(primary)
                logger.info("Migrated legacy squad JSON %s -> %s", legacy_json, primary)
            except OSError as exc:
                logger.warning(
                    "Loaded legacy squad JSON from %s but failed to persist TOML to %s: %s",
                    legacy_json,
                    primary,
                    exc,
                )
            return team
        return None

    def save_squad_file(self, path: Path | None = None) -> None:
        """Save team composition to .squad TOML."""
        workspace_root: Path | None = None
        if path is None:
            workspace_root = Path.cwd()
        elif path.suffix == ".toml":
            if path.parent.name == _SQUAD_DIR_NAME:
                workspace_root = path.parent.parent
        elif path.name == _SQUAD_DIR_NAME:
            workspace_root = path.parent
        else:
            workspace_root = path

        if workspace_root is not None:
            _ensure_squad_workspace(workspace_root)

        config_path = self._resolve_squad_file_path(path or workspace_root)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._to_squad_payload().get("squad", {})
        lead_name = str(payload.get("lead", "Lead"))
        agents = payload.get("agents", [])
        phases = payload.get("phases", [])

        lines = [
            "[squad]",
            f"lead = {json.dumps(lead_name, ensure_ascii=False)}",
        ]
        if isinstance(phases, list):
            for item in phases:
                if not isinstance(item, dict):
                    continue
                phase = _clamp_phase(item.get("phase", 2), fallback=2)
                gate = bool(item.get("gate", False))
                lines.append("")
                lines.append("[[squad.phases]]")
                lines.append(f"phase = {phase}")
                lines.append(f"gate = {str(gate).lower()}")
        if isinstance(agents, list):
            for item in agents:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                role_desc = str(item.get("role", "")).strip()
                if not name or not role_desc:
                    continue
                lines.append("")
                lines.append("[[squad.agents]]")
                lines.append(f"name = {json.dumps(name, ensure_ascii=False)}")
                lines.append(f"role = {json.dumps(role_desc, ensure_ascii=False)}")
                lines.append(f"phase = {_clamp_phase(item.get('phase', 2), fallback=2)}")

                role_id = str(item.get("id", "")).strip().lower()
                if role_id:
                    lines.append(f"id = {json.dumps(role_id, ensure_ascii=False)}")
                depends_on = _normalize_depends_on_roles(item.get("depends_on", []))
                if depends_on:
                    depends_on_str = ", ".join(
                        json.dumps(dep, ensure_ascii=False) for dep in depends_on
                    )
                    lines.append(f"depends_on = [{depends_on_str}]")
                if "retries" in item:
                    lines.append(f"retries = {_clamp_retries(item.get('retries', 1), fallback=1)}")

                subtasks = item.get("subtasks")
                if isinstance(subtasks, list):
                    normalized = [str(subtask).strip() for subtask in subtasks if str(subtask).strip()]
                    if normalized:
                        subtasks_str = ", ".join(
                            json.dumps(subtask, ensure_ascii=False) for subtask in normalized
                        )
                        lines.append(f"subtasks = [{subtasks_str}]")

        config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info(f"Squad team saved to {config_path}")

    @classmethod
    async def from_repo_or_file(
        cls,
        config: CopexConfig | None = None,
        path: Path | None = None,
        *,
        use_ai: bool = True,
        repo_context: dict[str, Any] | None = None,
    ) -> SquadTeam:
        """Prefer .squad file, then AI analysis (or pattern matching fallback)."""
        root = path or Path.cwd()
        from_file = cls.load_squad_file(root)
        if from_file is not None:
            return from_file
        if not use_ai:
            return cls.from_repo(root)
        return await cls.from_repo_ai(config=config, path=root, repo_context=repo_context)

    @classmethod
    async def update_from_request(
        cls,
        request: str,
        *,
        config: CopexConfig | None = None,
        path: Path | None = None,
    ) -> SquadTeam:
        """Use AI to apply a natural-language edit request to a squad definition."""
        # Lazy import to avoid circular dependency
        from copex.cli_client import CopilotCLI
        from copex.models import Model, ReasoningEffort

        root = path or Path.cwd()
        current_team = cls.load_squad_file(root)
        if current_team is None:
            current_team = await cls.from_repo_ai(config=config, path=root)

        current_payload = current_team._to_squad_payload()
        squad_payload = current_payload.get("squad", {})
        current_agents = squad_payload.get("agents", [])
        lead_name = squad_payload.get("lead", "Lead")

        prompt_parts = [
            "You update a squad configuration from a natural-language request.",
            "",
            "Current configuration:",
            f"Lead: {lead_name}",
            json.dumps(current_agents, indent=2, ensure_ascii=False),
            "",
            f"Request: {request}",
            "",
            "Return ONLY a JSON array of objects with:",
            '- "name": agent display name',
            '- "role": short responsibility sentence',
            '- "phase": integer 1-4',
            '- "depends_on": optional list of role IDs this agent depends on',
            '- "id": optional snake_case role identifier',
            '- "lead": optional boolean (true for lead entry)',
            "",
            "Rules:",
            "- Include exactly one lead entry (lead=true or phase=1).",
            "- Keep unaffected agents unless the request removes them.",
            "- Use unique names.",
            "- Keep team compact (typically 3-6 agents).",
        ]
        prompt = "\n".join(prompt_parts)

        ai_config = (config or CopexConfig()).model_copy()
        ai_config.model = Model.CLAUDE_OPUS_4_6_FAST
        ai_config.reasoning_effort = ReasoningEffort.LOW
        ai_config.streaming = False
        ai_config.use_cli = True

        async with CopilotCLI(ai_config) as cli:
            response = await cli.send(prompt)

        parsed_agents = extract_json_array(response.content.strip())
        if not parsed_agents:
            raise SquadConfigError("AI returned an empty squad configuration")

        updated_lead = str(lead_name).strip() or "Lead"
        updated_agents: list[dict[str, Any]] = []
        seen_names: set[str] = set()

        for item in parsed_agents:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            name_key = name.casefold()
            if name_key in seen_names:
                continue
            seen_names.add(name_key)

            phase = _clamp_phase(item.get("phase", 2), fallback=2)
            if bool(item.get("lead")) or phase == 1:
                updated_lead = name
                continue

            role_description = str(item.get("role", "")).strip()
            role_id = str(item.get("id", "")).strip().lower()
            agent_entry: dict[str, Any] = {
                "name": name,
                "role": role_description or f"Handles {name} responsibilities",
                "phase": phase,
            }
            if role_id:
                agent_entry["id"] = role_id
            depends_on = _normalize_depends_on_roles(item.get("depends_on", []))
            if depends_on:
                agent_entry["depends_on"] = depends_on
            retries = item.get("retries")
            if retries is not None:
                agent_entry["retries"] = _clamp_retries(retries, fallback=1)
            subtasks = item.get("subtasks")
            if isinstance(subtasks, list):
                normalized_subtasks = [
                    str(subtask).strip() for subtask in subtasks if str(subtask).strip()
                ]
                if normalized_subtasks:
                    agent_entry["subtasks"] = normalized_subtasks
            updated_agents.append(agent_entry)

        updated_payload: dict[str, Any] = {"squad": {"lead": updated_lead, "agents": updated_agents}}
        if current_team.phase_gates:
            updated_payload["squad"]["phases"] = [
                {"phase": phase, "gate": gate}
                for phase, gate in sorted(current_team.phase_gates.items())
            ]
        team = cls._from_squad_payload(updated_payload)
        if team is None:
            raise SquadConfigError("AI returned an invalid squad definition")
        return team

    @classmethod
    async def from_repo_ai(
        cls,
        config: CopexConfig | None = None,
        path: Path | None = None,
        existing_team: SquadTeam | None = None,
        repo_context: dict[str, Any] | None = None,
    ) -> SquadTeam:
        """Create a team by using AI to analyze the repository.

        Uses CopilotCLI with claude-opus-4.6-fast to intelligently analyze
        the repo structure, README, and config files. Falls back to from_repo()
        (pattern matching) on any failure.

        If an existing_team is provided (e.g., from .squad/team.toml), the AI
        sees the current team and can keep, modify, or expand it.

        Args:
            config: Optional CopexConfig (creates lightweight config if None).
            path: Optional path to repository (defaults to cwd).
            existing_team: Optional existing team for the AI to consider.

        Returns:
            SquadTeam with AI-determined roles, or fallback to from_repo().
        """
        try:
            # Lazy import to avoid circular dependency
            from copex.cli_client import CopilotCLI
            from copex.models import Model, ReasoningEffort

            # Gather repo context
            context = repo_context if repo_context is not None else _gather_repo_context(path)
            if not context:
                logger.warning("No repo context found, falling back to pattern matching")
                return cls.from_repo(path)

            # Check for existing team from canonical squad config
            if existing_team is None:
                existing_team = cls.load_squad_file(path or Path.cwd())

            # Build prompt for AI
            prompt_parts = [
                "Analyze this repository and determine what development team roles are needed.",
                "",
                "# Repository Context",
            ]

            if context.get("project_name"):
                prompt_parts.append(f"Project: {context['project_name']}")
            if context.get("description"):
                prompt_parts.append(f"Description: {context['description']}")
            if context.get("source_extensions"):
                prompt_parts.append(f"Languages: {', '.join(context['source_extensions'])}")
            if context.get("dependencies"):
                deps = context["dependencies"]
                prompt_parts.append(f"Dependencies: {', '.join(deps[:5])}")
            if context.get("directory_structure"):
                struct = "\n".join(context["directory_structure"][:30])
                prompt_parts.append(f"\nDirectory structure:\n{struct}")
            if context.get("readme_excerpt"):
                excerpt = context["readme_excerpt"][:1000]
                prompt_parts.append(f"\nREADME excerpt:\n{excerpt}")

            # Include existing team context if available
            if existing_team and existing_team.agents:
                prompt_parts.append("")
                prompt_parts.append("# Current Team Configuration")
                prompt_parts.append("")
                prompt_parts.append(
                    "This repository already has a team configured. "
                    "Review it and decide whether to keep it as-is, "
                    "modify roles, or add/remove agents:"
                )
                for agent in existing_team.agents:
                    prompt_parts.append(
                        f"- {agent.emoji} {agent.name} (role: {agent.role}, "
                        f"phase: {agent.phase})"
                    )

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
                "",
                "Rules:",
                "- Always include a 'lead' role with phase 1",
                "- Keep the team small (3-5 roles typical)",
                "- Only include roles clearly needed for THIS specific repository",
                "- Use depends_on for explicit DAG dependencies when useful.",
                "- If depends_on is omitted, phase ordering determines dependencies.",
                "- Subtasks are optional — only add them when work is naturally parallelizable",
                "- If a current team is shown above, use it as a starting point —"
                " keep roles that still make sense, remove unnecessary ones, and add new ones if needed",
                "",
                "Example response:",
                '[',
                '  {"role": "lead", "name": "Lead Architect", "emoji": "🏗️", '
                '"prompt": "You are the Lead Architect. Analyze the task...", "phase": 1},',
                '  {"role": "developer", "name": "Developer", "emoji": "🔧", '
                '"prompt": "You are the Developer. Implement the task...", "phase": 2,',
                '   "subtasks": ["Core module implementation", "API endpoint handlers"]},',
                '  {"role": "tester", "name": "Tester", "emoji": "🧪", '
                '"prompt": "You are the Tester. Write comprehensive tests...", "phase": 3}',
                ']',
            ])

            prompt = "\n".join(prompt_parts)

            # Create lightweight config for AI analysis (copy to avoid mutation)
            ai_config = (config or CopexConfig()).model_copy()
            ai_config.model = Model.CLAUDE_OPUS_4_6_FAST
            ai_config.reasoning_effort = ReasoningEffort.LOW
            ai_config.streaming = False
            ai_config.use_cli = True

            # Call AI with timeout
            async with CopilotCLI(ai_config) as cli:
                response = await cli.send(prompt)
                content = response.content.strip()

                # Robustly extract JSON array from LLM response
                roles_list = extract_json_array(content)

                # Build agents from AI response
                agents: list[SquadAgent] = []
                seen_roles: set[str] = set()
                for item in roles_list:
                    if isinstance(item, str):
                        # Legacy format: plain role string
                        role_str = item.lower()
                        if role_str not in seen_roles:
                            seen_roles.add(role_str)
                            agents.append(SquadAgent.default_for_role(role_str))
                    elif isinstance(item, dict):
                        role_str = item.get("role", "").lower()
                        if not role_str or role_str in seen_roles:
                            continue
                        seen_roles.add(role_str)
                        phase = item.get("phase", _KNOWN_ROLE_PHASES.get(role_str, 2))
                        try:
                            phase = int(phase)
                        except (TypeError, ValueError):
                            phase = 2
                        subtasks_raw = item.get("subtasks", [])
                        subtasks = (
                            [str(s) for s in subtasks_raw]
                            if isinstance(subtasks_raw, list)
                            else []
                        )
                        depends_on = _normalize_depends_on_roles(item.get("depends_on", []))
                        retries = _clamp_retries(item.get("retries", 1), fallback=1)
                        agents.append(SquadAgent(
                            name=item.get("name", role_str.replace("_", " ").title()),
                            role=role_str,
                            emoji=item.get("emoji", _ROLE_EMOJIS.get(role_str, "🔹")),
                            system_prompt=item.get("prompt", _ROLE_PROMPTS.get(
                                role_str,
                                f"You are the {role_str.replace('_', ' ').title()}. "
                                f"Focus on your area of expertise for this task.",
                            )),
                            phase=max(1, min(4, phase)),
                            depends_on=depends_on,
                            subtasks=subtasks,
                            retries=retries,
                        ))

                # Ensure lead exists
                if "lead" not in seen_roles:
                    agents.insert(0, SquadAgent.default_for_role("lead"))

                logger.info(f"AI analysis completed: {[a.role for a in agents]}")
                team = cls(agents=agents)
                # Persist for future runs
                try:
                    team.save_squad_file(path or Path.cwd())
                except Exception:
                    pass  # Non-critical
                return team

        except Exception as e:
            logger.warning(f"AI repo analysis failed ({e}), falling back to pattern matching")
            return cls.from_repo(path)

    SQUAD_CONFIG = Path(_SQUAD_DIR_NAME) / _SQUAD_TEAM_FILE_NAME
    LEGACY_SQUAD_CONFIG = Path(_SQUAD_LEGACY_DIR_NAME) / _SQUAD_LEGACY_JSON_FILE_NAME

    def save(self, path: Path | None = None) -> None:
        """Deprecated compatibility shim; persists canonical TOML config."""
        if path is not None and path.suffix == ".json":
            msg = (
                f"Saving squad config to JSON ({path}) is deprecated; "
                f"writing TOML to {self._resolve_squad_file_path(path)} instead"
            )
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            logger.warning(msg)
        target = self._resolve_squad_file_path(path)
        self.save_squad_file(target)

    @classmethod
    def load(cls, path: Path | None = None) -> SquadTeam | None:
        """Deprecated compatibility shim around canonical TOML loader."""
        config_path = path or cls.SQUAD_CONFIG
        if config_path.suffix == ".json":
            team = cls._load_legacy_json_file(config_path, warn_deprecated=True)
            if team is not None:
                toml_path = cls._resolve_squad_file_path(config_path)
                if not toml_path.is_file():
                    try:
                        team.save_squad_file(toml_path)
                        logger.info("Migrated legacy squad JSON %s -> %s", config_path, toml_path)
                    except OSError as exc:
                        logger.warning(
                            "Loaded legacy squad JSON from %s but failed to persist TOML to %s: %s",
                            config_path,
                            toml_path,
                            exc,
                        )
                return team
            return cls.load_squad_file(cls._resolve_squad_file_path(config_path))
        return cls.load_squad_file(config_path)

    def get_agent(self, role: SquadRole | str) -> SquadAgent | None:
        """Get agent by role."""
        role_str = role.value if isinstance(role, SquadRole) else role.lower()
        for a in self.agents:
            if a.role == role_str:
                return a
        return None

    @property
    def roles(self) -> list[str]:
        """List roles present in the team."""
        return [a.role for a in self.agents]
