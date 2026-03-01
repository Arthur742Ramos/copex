"""Squad - Built-in multi-agent team orchestration for Copex.

Creates a default team (Lead, Developer, Tester) and orchestrates work
through Fleet with automatic task decomposition and parallel execution.

Usage::

    from copex import SquadCoordinator

    async with SquadCoordinator(config) as squad:
        result = await squad.run("Build a REST API with auth")
        print(result.final_content)

    # Or via CLI:
    # copex squad "Build a REST API" --model claude-sonnet-4.5 --reasoning medium
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore

from copex.config import CopexConfig
from copex.fleet import Fleet, FleetConfig, FleetResult
from copex.json_utils import extract_json_array

logger = logging.getLogger(__name__)


class SquadRole(str, Enum):
    """Well-known roles for backward compatibility.

    Squad teams can use any string role — these are just convenient constants
    for common roles. AI analysis may return completely custom roles.
    """

    LEAD = "lead"
    DEVELOPER = "developer"
    TESTER = "tester"
    DOCS = "docs"
    DEVOPS = "devops"
    FRONTEND = "frontend"
    BACKEND = "backend"


_ROLE_PROMPTS: dict[str, str] = {
    "lead": (
        "You are the Lead Architect. Analyze the task, break it down into "
        "a clear implementation plan, identify key files and patterns to "
        "follow, and flag any risks or decisions needed. Be specific about "
        "what needs to be built and how."
    ),
    "developer": (
        "You are the Developer. Implement the task according to the plan. "
        "Write clean, well-structured code following the project's existing "
        "patterns and conventions. Make the smallest changes necessary."
    ),
    "tester": (
        "You are the Tester. Write comprehensive tests for the task. "
        "Cover happy paths, edge cases, and error handling. Follow the "
        "project's existing test patterns and conventions."
    ),
    "docs": (
        "You are the Documentation Expert. Review and update documentation, "
        "README files, docstrings, and examples. Ensure documentation stays "
        "in sync with code changes."
    ),
    "devops": (
        "You are the DevOps Engineer. Manage CI/CD, Docker, infrastructure, "
        "and deployment configurations."
    ),
    "frontend": (
        "You are the Frontend Developer. Build UI components, handle "
        "client-side logic, and ensure good UX."
    ),
    "backend": (
        "You are the Backend Developer. Build APIs, services, database "
        "interactions, and server-side logic."
    ),
}

_ROLE_EMOJIS: dict[str, str] = {
    "lead": "🏗️",
    "developer": "🔧",
    "tester": "🧪",
    "docs": "📝",
    "devops": "⚙️",
    "frontend": "⚛️",
    "backend": "🔧",
}

_SOURCE_EXTENSIONS = frozenset({
    ".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp",
})

_FRONTEND_DIRS = frozenset({
    "frontend", "web", "ui", "app", "pages", "components",
})

_BACKEND_DIRS = frozenset({
    "api", "server", "backend", "services",
})

_TASK_OUTPUT_REF_RE = re.compile(
    r"\{\{\s*task:(?P<task_id>[a-zA-Z0-9_.-]+)\.(?P<field>"
    r"content|reasoning|success|error|duration_ms|prompt_tokens|completion_tokens"
    r")\s*\}\}"
)

_REASONING_TIMEOUTS: dict[str, float] = {
    "xhigh": 1800.0,
    "high": 900.0,
    "medium": 600.0,
    "low": 300.0,
    "none": 300.0,
}

_SQUAD_DIR_NAME = ".squad"
_SQUAD_TEAM_FILE_NAME = "team.toml"
_SQUAD_KNOWLEDGE_DIR_NAME = "knowledge"
_SQUAD_LOG_DIR_NAME = "log"
_SQUAD_DECISIONS_FILE_NAME = "decisions.md"

_KNOWLEDGE_SECTION_HEADING = "What I Learned About This Codebase"
_DECISIONS_SECTION_HEADING = "Key Decisions and Trade-offs"
_MAX_PERSISTED_CONTEXT_CHARS = 4000
_MAX_PERSISTED_ITEM_CHARS = 280
_MAX_PERSISTED_ITEMS = 8


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _session_log_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M")


def _clip_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _extract_markdown_section(content: str, headings: tuple[str, ...]) -> str:
    if not content.strip():
        return ""
    normalized = content.replace("\r\n", "\n")
    for heading in headings:
        pattern = re.compile(
            rf"(?ims)^##\s*{re.escape(heading)}\s*:?\s*$\n(?P<body>.*?)(?=^\s*##\s+|\Z)"
        )
        match = pattern.search(normalized)
        if match:
            return match.group("body").strip()
    return ""


def _extract_persisted_items(
    content: str,
    *,
    headings: tuple[str, ...],
) -> list[str]:
    section = _extract_markdown_section(content, headings)
    if not section:
        return []

    entries: list[str] = []
    seen: set[str] = set()
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = re.match(r"^[-*+]\s+(.+)$", stripped)
        if match is None:
            match = re.match(r"^\d+[.)]\s+(.+)$", stripped)
        item = match.group(1).strip() if match else stripped
        item = " ".join(item.split())
        if not item:
            continue
        item = _clip_text(item, _MAX_PERSISTED_ITEM_CHARS)
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        entries.append(item)
        if len(entries) >= _MAX_PERSISTED_ITEMS:
            break

    if entries:
        return entries

    fallback = " ".join(section.split())
    if not fallback:
        return []
    return [_clip_text(fallback, _MAX_PERSISTED_ITEM_CHARS)]


def _append_markdown_items(path: Path, *, title: str, items: list[str]) -> None:
    if not items:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        path.write_text(f"# {title}\n\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"## {_now_iso_utc()}\n")
        for item in items:
            handle.write(f"- {item}\n")
        handle.write("\n")


def _read_text_tail(path: Path, *, max_chars: int = _MAX_PERSISTED_CONTEXT_CHARS) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8").strip()
    if len(text) <= max_chars:
        return text
    return "[Older entries truncated]\n" + text[-max_chars:]


def _has_source_files(root: Path) -> bool:
    """Check for source code files (shallow scan, fast exit)."""
    for ext in _SOURCE_EXTENSIONS:
        try:
            if next(root.rglob(f"*{ext}"), None) is not None:
                return True
        except OSError:
            pass
    return False


def _has_test_files(root: Path) -> bool:
    """Check for test directories or test file patterns."""
    for name in ("tests", "test"):
        if (root / name).is_dir():
            return True
    test_patterns = ("*_test.*", "test_*.*", "*.spec.*")
    for pat in test_patterns:
        try:
            if next(root.rglob(pat), None) is not None:
                return True
        except OSError:
            pass
    return False


def _has_docs(root: Path) -> bool:
    """Check for documentation signals."""
    if (root / "docs").is_dir():
        return True
    if (root / "README.md").is_file():
        return True
    try:
        md_files = list(root.glob("*.md"))
        return len(md_files) >= 2
    except OSError:
        return False


def _has_devops(root: Path) -> bool:
    """Check for DevOps-related files."""
    signals = [
        root / "Dockerfile",
        root / "Makefile",
        root / "Jenkinsfile",
        root / ".github" / "workflows",
    ]
    for p in signals:
        if p.exists():
            return True
    try:
        if next(root.glob("docker-compose*"), None) is not None:
            return True
    except OSError:
        pass
    return False


def _has_frontend(root: Path) -> bool:
    """Check for frontend directory signals under root."""
    return any((root / d).is_dir() for d in _FRONTEND_DIRS)


def _has_backend(root: Path) -> bool:
    """Check for backend directory signals under root."""
    return any((root / d).is_dir() for d in _BACKEND_DIRS)


def _gather_repo_context(path: Path | None = None) -> dict[str, Any]:
    """Gather repository context for AI analysis.

    Reads pyproject.toml, README.md, directory structure, and source file
    extensions. Returns a dictionary with all available context.
    """
    root = path or Path.cwd()
    context: dict[str, Any] = {}

    # Read pyproject.toml
    pyproject_path = root / "pyproject.toml"
    if pyproject_path.is_file():
        try:
            try:
                import tomllib  # Python 3.11+
            except ImportError:
                import tomli as tomllib  # type: ignore

            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)
            project = data.get("project", {})
            context["project_name"] = project.get("name", "")
            context["description"] = project.get("description", "")
            deps = project.get("dependencies", [])
            if deps:
                context["dependencies"] = deps[:10]  # First 10 deps
        except Exception:
            pass

    # Read README.md
    readme_path = root / "README.md"
    if readme_path.is_file():
        try:
            text = readme_path.read_text(encoding="utf-8")[:2000]
            context["readme_excerpt"] = text
        except Exception:
            pass

    # Directory structure (2 levels deep for AI)
    try:
        structure: list[str] = []
        for entry in sorted(root.iterdir()):
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                structure.append(entry.name + "/")
                # One level deeper for directories
                try:
                    for subentry in sorted(entry.iterdir()):
                        if not subentry.name.startswith("."):
                            suffix = "/" if subentry.is_dir() else ""
                            structure.append(f"  {subentry.name}{suffix}")
                except (OSError, PermissionError):
                    pass
            else:
                structure.append(entry.name)
        if structure:
            context["directory_structure"] = structure[:50]  # Limit to 50 items
    except Exception:
        pass

    # Source file extensions found
    extensions: set[str] = set()
    for ext in _SOURCE_EXTENSIONS:
        try:
            if next(root.rglob(f"*{ext}"), None) is not None:
                extensions.add(ext)
        except OSError:
            pass
    if extensions:
        context["source_extensions"] = sorted(extensions)

    return context


_KNOWN_ROLE_PHASES: dict[str, int] = {
    "lead": 1,
    "developer": 2,
    "frontend": 2,
    "backend": 2,
    "devops": 2,
    "tester": 3,
    "docs": 4,
}

_ROLE_DESCRIPTIONS: dict[str, str] = {
    "lead": "Leads architecture and task planning",
    "developer": "Writes implementation code",
    "tester": "Writes and runs tests",
    "docs": "Updates documentation",
    "devops": "Maintains CI/CD and infrastructure",
    "frontend": "Builds frontend UI and client behavior",
    "backend": "Builds backend services and APIs",
}


def _normalize_role_identifier(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return normalized or "agent"


def _squad_dir_path(path: Path | None = None) -> Path:
    return (path or Path.cwd()) / _SQUAD_DIR_NAME


def _squad_team_path(path: Path | None = None) -> Path:
    return _squad_dir_path(path) / _SQUAD_TEAM_FILE_NAME


def _squad_legacy_file_path(path: Path | None = None) -> Path:
    return (path or Path.cwd()) / _SQUAD_DIR_NAME


def _squad_knowledge_path(role: str, path: Path | None = None) -> Path:
    safe_role = _normalize_role_identifier(role).replace("_", "-")
    return _squad_dir_path(path) / _SQUAD_KNOWLEDGE_DIR_NAME / f"{safe_role}.md"


def _squad_decisions_path(path: Path | None = None) -> Path:
    return _squad_dir_path(path) / _SQUAD_DECISIONS_FILE_NAME


def _squad_log_dir_path(path: Path | None = None) -> Path:
    return _squad_dir_path(path) / _SQUAD_LOG_DIR_NAME


def _ensure_squad_workspace(path: Path | None = None) -> Path:
    root = path or Path.cwd()
    squad_path = _squad_dir_path(root)
    if squad_path.is_file():
        legacy_content = squad_path.read_text(encoding="utf-8")
        squad_path.unlink()
        squad_path.mkdir(parents=True, exist_ok=True)
        _squad_team_path(root).write_text(legacy_content, encoding="utf-8")
        logger.info("Migrated legacy .squad file to %s", _squad_team_path(root))
    else:
        squad_path.mkdir(parents=True, exist_ok=True)
    (squad_path / _SQUAD_KNOWLEDGE_DIR_NAME).mkdir(parents=True, exist_ok=True)
    (squad_path / _SQUAD_LOG_DIR_NAME).mkdir(parents=True, exist_ok=True)
    return squad_path


def _clamp_phase(value: Any, fallback: int = 2) -> int:
    try:
        phase = int(value)
    except (TypeError, ValueError):
        phase = fallback
    return max(1, min(4, phase))


@dataclass
class SquadAgent:
    """An agent in the squad team."""

    name: str
    role: str
    emoji: str
    system_prompt: str
    phase: int = 2
    subtasks: list[str] = field(default_factory=list)

    @classmethod
    def default_for_role(cls, role: SquadRole | str) -> SquadAgent:
        """Create a default agent for a role.

        Accepts SquadRole enum values or arbitrary role strings.
        Known roles get predefined prompts and emojis; custom roles
        get sensible defaults.
        """
        role_str = role.value if isinstance(role, SquadRole) else role.lower()
        return cls(
            name=role_str.replace("_", " ").title(),
            role=role_str,
            emoji=_ROLE_EMOJIS.get(role_str, "🔹"),
            system_prompt=_ROLE_PROMPTS.get(
                role_str,
                f"You are the {role_str.replace('_', ' ').title()}. "
                f"Focus on your area of expertise for this task.",
            ),
            phase=_KNOWN_ROLE_PHASES.get(role_str, 2),
        )


@dataclass
class SquadTeam:
    """A team of squad agents."""

    agents: list[SquadAgent] = field(default_factory=list)
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
    def _resolve_squad_file_path(cls, path: Path | None = None) -> Path:
        if path is None:
            return cls.SQUAD_TEAM_FILE
        if path.suffix == ".toml":
            return path
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
        if path.suffix == ".toml":
            if path.parent.name == _SQUAD_DIR_NAME:
                return path.parent.parent / _SQUAD_DIR_NAME
            return path.with_name(_SQUAD_DIR_NAME)
        return path / _SQUAD_DIR_NAME

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
            subtasks_raw = item.get("subtasks", [])
            subtasks = (
                [str(s).strip() for s in subtasks_raw if str(s).strip()]
                if isinstance(subtasks_raw, list)
                else []
            )

            emoji_raw = str(item.get("emoji", "")).strip()
            agents.append(
                SquadAgent(
                    name=name,
                    role=role_id,
                    emoji=emoji_raw or _ROLE_EMOJIS.get(role_id, "🔹"),
                    system_prompt=system_prompt,
                    phase=phase,
                    subtasks=subtasks,
                )
            )

        return cls(agents=agents)

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
            if agent.subtasks:
                item["subtasks"] = agent.subtasks
            serialized_agents.append(item)
        return {
            "squad": {
                "lead": lead_name,
                "agents": serialized_agents,
            }
        }

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

        lines = [
            "[squad]",
            f"lead = {json.dumps(lead_name, ensure_ascii=False)}",
        ]
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
    ) -> SquadTeam:
        """Prefer .squad file, then AI analysis (or pattern matching fallback)."""
        root = path or Path.cwd()
        from_file = cls.load_squad_file(root)
        if from_file is not None:
            return from_file
        if not use_ai:
            return cls.from_repo(root)
        return await cls.from_repo_ai(config=config, path=root)

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
            raise ValueError("AI returned an empty squad configuration")

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
            subtasks = item.get("subtasks")
            if isinstance(subtasks, list):
                normalized_subtasks = [
                    str(subtask).strip() for subtask in subtasks if str(subtask).strip()
                ]
                if normalized_subtasks:
                    agent_entry["subtasks"] = normalized_subtasks
            updated_agents.append(agent_entry)

        updated_payload = {"squad": {"lead": updated_lead, "agents": updated_agents}}
        team = cls._from_squad_payload(updated_payload)
        if team is None:
            raise ValueError("AI returned an invalid squad definition")
        return team

    @classmethod
    async def from_repo_ai(
        cls,
        config: CopexConfig | None = None,
        path: Path | None = None,
        existing_team: SquadTeam | None = None,
    ) -> SquadTeam:
        """Create a team by using AI to analyze the repository.

        Uses CopilotCLI with claude-opus-4.6-fast to intelligently analyze
        the repo structure, README, and config files. Falls back to from_repo()
        (pattern matching) on any failure.

        If an existing_team is provided (e.g., from .copex/squad.json), the AI
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
            context = _gather_repo_context(path)
            if not context:
                logger.warning("No repo context found, falling back to pattern matching")
                return cls.from_repo(path)

            # Check for existing team (from config or .copex/squad.json)
            if existing_team is None:
                existing_team = cls.load(
                    (path or Path.cwd()) / ".copex" / "squad.json"
                )

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
                "- Phase ordering determines dependencies (phase 2 waits for phase 1, etc.)",
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
                            subtasks=subtasks,
                        ))

                # Ensure lead exists
                if "lead" not in seen_roles:
                    agents.insert(0, SquadAgent.default_for_role("lead"))

                logger.info(f"AI analysis completed: {[a.role for a in agents]}")
                team = cls(agents=agents)
                # Persist for future runs
                try:
                    team.save(
                        (path or Path.cwd()) / ".copex" / "squad.json"
                    )
                except Exception:
                    pass  # Non-critical
                return team

        except Exception as e:
            logger.warning(f"AI repo analysis failed ({e}), falling back to pattern matching")
            return cls.from_repo(path)

    SQUAD_CONFIG = Path(".copex") / "squad.json"

    def save(self, path: Path | None = None) -> None:
        """Save team configuration to .copex/squad.json."""
        config_path = path or self.SQUAD_CONFIG
        config_path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "role": a.role,
                "name": a.name,
                "emoji": a.emoji,
                "prompt": a.system_prompt,
                "phase": a.phase,
                **({"subtasks": a.subtasks} if a.subtasks else {}),
            }
            for a in self.agents
        ]
        config_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(f"Squad team saved to {config_path}")

    @classmethod
    def load(cls, path: Path | None = None) -> SquadTeam | None:
        """Load team configuration from .copex/squad.json.

        Returns None if file doesn't exist.
        """
        config_path = path or cls.SQUAD_CONFIG
        if not config_path.is_file():
            return None
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                return None
            agents = []
            for item in data:
                if not isinstance(item, dict) or not item.get("role"):
                    continue
                role_str = item["role"].lower()
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
                    subtasks=subtasks,
                ))
            if agents:
                logger.info(f"Loaded squad team from {config_path}: {[a.role for a in agents]}")
                return cls(agents=agents)
        except Exception as e:
            logger.warning(f"Failed to load squad config from {config_path}: {e}")
        return None

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


@dataclass
class SquadAgentResult:
    """Result from a single squad agent."""

    agent: SquadAgent
    content: str
    success: bool
    duration_ms: float = 0.0
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class SquadResult:
    """Aggregated result from the squad."""

    agent_results: list[SquadAgentResult] = field(default_factory=list)
    total_duration_ms: float = 0.0
    success: bool = True

    @property
    def final_content(self) -> str:
        """Get the combined output from all agents."""
        parts = []
        for ar in self.agent_results:
            if ar.content:
                parts.append(f"## {ar.agent.emoji} {ar.agent.name}\n\n{ar.content}")
        return "\n\n---\n\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "success": self.success,
            "total_duration_ms": round(self.total_duration_ms, 1),
            "agents": [
                {
                    "name": ar.agent.name,
                    "role": ar.agent.role,
                    "success": ar.success,
                    "content": ar.content,
                    "duration_ms": round(ar.duration_ms, 1),
                    "error": ar.error,
                    "prompt_tokens": ar.prompt_tokens,
                    "completion_tokens": ar.completion_tokens,
                }
                for ar in self.agent_results
            ],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class SquadCoordinator:
    """Orchestrates a team of agents using Fleet for parallel execution.

    Creates a default team (Lead → Developer + Tester) and runs them
    through Fleet with dependency ordering:

    - Lead analyzes first (no dependencies)
    - Developer and Tester run in parallel (both depend on Lead's output)

    Uses Fleet features automatically:
    - Parallel execution with dependency ordering
    - Adaptive concurrency and rate limit handling
    - Session pooling for efficiency
    - Retry with exponential backoff
    - Cost tracking

    Usage::

        coordinator = SquadCoordinator(config)
        result = await coordinator.run("Build a REST API")
        print(result.final_content)
    """

    def __init__(
        self,
        config: CopexConfig,
        *,
        team: SquadTeam | None = None,
        fleet_config: FleetConfig | None = None,
    ) -> None:
        self._config = config
        self._team = team  # Can be None for lazy init
        self._fleet_config = fleet_config or FleetConfig(
            max_concurrent=3,
            timeout=self._default_timeout_from_reasoning(config.reasoning_effort),
        )
        self._project_context: str | None = None
        self._repo_map: Any | None = None

    @property
    def team(self) -> SquadTeam:
        """The squad team."""
        if self._team is None:
            # Deterministic sync fallback for direct usage and tests.
            # run() performs .squad/AI discovery when execution begins.
            self._team = SquadTeam.default()
        return self._team

    async def __aenter__(self) -> SquadCoordinator:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    async def run(
        self,
        prompt: str,
        *,
        on_status: Any | None = None,
    ) -> SquadResult:
        """Run the squad on a task.

        Args:
            prompt: The task to accomplish.
            on_status: Optional callback(task_id, status) for progress.

        Returns:
            SquadResult with results from all agents.
        """
        start_time = time.monotonic()

        # Lazy init: prefer .squad, then AI repo analysis
        if self._team is None:
            self._team = await SquadTeam.from_repo_or_file(self._config)
            team_path = SquadTeam._resolve_squad_file_path()
            if not team_path.is_file():
                self._team.save_squad_file()
                logger.info("Squad team auto-saved to %s", team_path)

        if self._project_context is None:
            self._project_context = self._discover_project_context()
        if self._repo_map is None:
            try:
                from copex.repo_map import RepoMap

                self._repo_map = RepoMap(Path.cwd())
                self._repo_map.refresh(force=False)
            except Exception as exc:
                logger.info("Repo map unavailable for squad context: %s", exc)
                self._repo_map = None

        max_feedback_iterations = 3
        feedback_iteration = 0
        current_prompt = prompt

        final_result: SquadResult | None = None
        while True:
            fleet = Fleet(self._config, fleet_config=self._fleet_config)
            task_prompts: dict[str, str] = {}
            task_ids = self._add_tasks(fleet, current_prompt, task_prompts=task_prompts)
            fleet_results = await fleet.run(on_status=on_status)
            fleet_results = await self._retry_failed_agents(
                fleet_results,
                task_prompts,
                on_status=on_status,
            )

            iteration_result = self._build_result(fleet_results, task_ids)
            tester_results = [
                ar
                for ar in iteration_result.agent_results
                if ar.agent.role == SquadRole.TESTER.value
            ]
            tester_needs_feedback = any(
                (
                    (not ar.success and not self._is_timeout_error(ar.error))
                    or self._tester_reported_issues(ar.content)
                )
                for ar in tester_results
            )

            if (
                not tester_results
                or not tester_needs_feedback
                or feedback_iteration >= max_feedback_iterations
            ):
                final_result = iteration_result
                break

            feedback_iteration += 1
            current_prompt = self._build_feedback_iteration_prompt(
                prompt,
                tester_results,
                feedback_iteration,
            )

        assert final_result is not None  # noqa: S101
        final_result.total_duration_ms = (time.monotonic() - start_time) * 1000
        try:
            self._persist_run_artifacts(prompt, final_result)
        except OSError as exc:
            logger.warning("Failed to persist squad knowledge artifacts: %s", exc)
        return final_result

    def _add_tasks(
        self,
        fleet: Fleet,
        prompt: str,
        *,
        task_prompts: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Add fleet tasks for each agent.

        When an agent has subtasks, each subtask becomes a separate Fleet task
        within the same phase, running in parallel. Results are later merged
        back per agent in ``_build_result()``.
        """
        task_ids: dict[str, str] = {}

        for agent in self.team.agents:  # Use property for lazy init
            deps = self._get_dependencies(agent, task_ids)

            if agent.subtasks:
                # Fan out: one Fleet task per subtask, all in same phase
                sub_ids: list[str] = []
                for i, subtask_desc in enumerate(agent.subtasks):
                    sub_prompt = self._build_agent_prompt(
                        agent,
                        f"{prompt}\n\n## Subtask ({i + 1}/{len(agent.subtasks)})"
                        f"\n\nFocus on: {subtask_desc}",
                        task_ids,
                    )
                    sub_id = f"{agent.role}__sub{i + 1}"
                    tid = fleet.add(
                        sub_prompt,
                        task_id=sub_id,
                        depends_on=deps,
                        model=self._config.model,
                        retries=0,
                    )
                    if task_prompts is not None:
                        task_prompts[tid] = sub_prompt
                    sub_ids.append(tid)
                # Store pipe-joined subtask IDs so deps and result mapping work
                task_ids[agent.role] = "|".join(sub_ids)
            else:
                agent_prompt = self._build_agent_prompt(agent, prompt, task_ids)
                tid = fleet.add(
                    agent_prompt,
                    task_id=agent.role,
                    depends_on=deps,
                    model=self._config.model,
                    retries=0,
                )
                if task_prompts is not None:
                    task_prompts[tid] = agent_prompt
                task_ids[agent.role] = tid

        return task_ids

    def _get_dependencies(
        self, agent: SquadAgent, task_ids: dict[str, str]
    ) -> list[str]:
        """Get dependency task IDs for an agent based on phase ordering.

        Phase 1 (analyze): no dependencies
        Phase 2 (build): depends on all phase 1 agents
        Phase 3 (verify): depends on all phase 2 agents
        Phase 4 (document): depends on all phase 2 and 3 agents

        When a dependency agent has subtasks, all its subtask IDs are
        included (the dependent agent waits for all subtasks to finish).
        """
        if agent.phase <= 1:
            return []
        deps = []
        for other in self.team.agents:
            if other.role == agent.role:
                continue
            tid = task_ids.get(other.role)
            if tid and other.phase < agent.phase:
                # Expand pipe-joined subtask IDs
                deps.extend(tid.split("|"))
        return deps

    def _build_agent_prompt(
        self,
        agent: SquadAgent,
        user_prompt: str,
        task_ids: dict[str, str],
    ) -> str:
        """Build the full prompt for an agent including role context."""
        parts = [agent.system_prompt]

        # Inject project context if available
        if self._project_context:
            parts.append("")
            parts.append(self._project_context)

        if self._repo_map is not None:
            try:
                repo_context = self._repo_map.relevant_context(
                    f"{agent.role} {user_prompt}",
                    max_files=6,
                    max_symbols_per_file=6,
                )
                if repo_context:
                    parts.append("")
                    parts.append(repo_context)
            except Exception as exc:
                logger.debug("Failed to build repo map context for %s: %s", agent.role, exc)

        shared_decisions = self._load_shared_decisions()
        if shared_decisions:
            parts.append("")
            parts.append("## Shared Squad Decisions")
            parts.append(
                "Treat these as settled decisions unless you have concrete new evidence."
            )
            parts.append(shared_decisions)

        role_knowledge = self._load_agent_knowledge(agent)
        if role_knowledge:
            parts.append("")
            parts.append("## Your Persistent Knowledge")
            parts.append(role_knowledge)

        parts.append("")
        parts.append(f"## Task\n\n{user_prompt}")

        # Add references to all completed prior-phase outputs
        prior_phase_refs: list[tuple[SquadAgent, list[str]]] = []
        for other in self.team.agents:
            if other.role == agent.role or other.phase >= agent.phase:
                continue
            other_task_ids = task_ids.get(other.role)
            if not other_task_ids:
                continue
            prior_phase_refs.append((other, other_task_ids.split("|")))

        if prior_phase_refs:
            parts.append("")
            parts.append("## Prior Phase Outputs")
            for other, ref_ids in prior_phase_refs:
                parts.append("")
                parts.append(f"### {other.name}")
                if len(ref_ids) == 1:
                    parts.append(f"{{{{task:{ref_ids[0]}.content}}}}")
                    continue
                for i, ref_id in enumerate(ref_ids):
                    label = (
                        other.subtasks[i]
                        if i < len(other.subtasks)
                        else f"Subtask {i + 1}"
                    )
                    parts.append(f"#### {label}")
                    parts.append(f"{{{{task:{ref_id}.content}}}}")

        parts.append("")
        parts.append("## Knowledge Capture")
        parts.append(
            f"At the end of your response, include a section titled "
            f"'## {_KNOWLEDGE_SECTION_HEADING}' with concise bullet points "
            "about patterns, conventions, gotchas, and decisions you learned."
        )
        if agent.role == SquadRole.LEAD.value:
            parts.append(
                f"Also include a section titled '## {_DECISIONS_SECTION_HEADING}' "
                "with concise bullet points that capture final decisions and trade-offs."
            )

        return "\n".join(parts)

    def _load_agent_knowledge(self, agent: SquadAgent) -> str:
        return _read_text_tail(_squad_knowledge_path(agent.role))

    def _load_shared_decisions(self) -> str:
        return _read_text_tail(_squad_decisions_path())

    @staticmethod
    def _extract_knowledge_items(content: str) -> list[str]:
        return _extract_persisted_items(
            content,
            headings=(
                _KNOWLEDGE_SECTION_HEADING,
                "Learned Knowledge",
                "What did you learn about this codebase?",
                "What I Learned",
            ),
        )

    @staticmethod
    def _extract_decision_items(content: str) -> list[str]:
        return _extract_persisted_items(
            content,
            headings=(
                _DECISIONS_SECTION_HEADING,
                "Decisions and Trade-offs",
                "Decisions",
            ),
        )

    @staticmethod
    def _next_session_log_path(log_dir: Path) -> Path:
        stamp = _session_log_stamp()
        candidate = log_dir / f"{stamp}.md"
        index = 2
        while candidate.exists():
            candidate = log_dir / f"{stamp}-{index}.md"
            index += 1
        return candidate

    @staticmethod
    def _summarize_output_for_log(content: str, *, limit: int = 1200) -> str:
        text = content.strip()
        if not text:
            return ""
        return _clip_text(text, limit)

    def _persist_run_artifacts(self, task_prompt: str, result: SquadResult) -> None:
        _ensure_squad_workspace()

        for agent_result in result.agent_results:
            knowledge_items = self._extract_knowledge_items(agent_result.content)
            if knowledge_items:
                _append_markdown_items(
                    _squad_knowledge_path(agent_result.agent.role),
                    title=f"{agent_result.agent.name} Knowledge",
                    items=knowledge_items,
                )

            if agent_result.agent.role == SquadRole.LEAD.value:
                decision_items = self._extract_decision_items(agent_result.content)
                if decision_items:
                    _append_markdown_items(
                        _squad_decisions_path(),
                        title="Shared Squad Decisions",
                        items=decision_items,
                    )

        self._write_session_log(task_prompt, result)

    def _write_session_log(self, task_prompt: str, result: SquadResult) -> None:
        log_dir = _squad_log_dir_path()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._next_session_log_path(log_dir)

        lines = [
            "# Squad Session Log",
            "",
            f"- Timestamp: {_now_iso_utc()}",
            f"- Task: {task_prompt}",
            f"- Success: {'yes' if result.success else 'no'}",
            f"- Duration: {result.total_duration_ms / 1000:.1f}s",
            "",
            "## Agents",
        ]
        for agent_result in result.agent_results:
            status = "success" if agent_result.success else "failed"
            lines.append(
                f"- {agent_result.agent.emoji} {agent_result.agent.name} "
                f"({agent_result.agent.role}) — {status} ({agent_result.duration_ms / 1000:.1f}s)"
            )

        lines.append("")
        lines.append("## Work Summary")
        for agent_result in result.agent_results:
            lines.append("")
            lines.append(f"### {agent_result.agent.emoji} {agent_result.agent.name}")
            lines.append(f"- Outcome: {'success' if agent_result.success else 'failed'}")
            if agent_result.error:
                lines.append(f"- Error: {agent_result.error}")
            excerpt = self._summarize_output_for_log(agent_result.content)
            if excerpt:
                lines.append("- Output excerpt:")
                lines.append("")
                lines.append(excerpt)

        log_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    async def _retry_failed_agents(
        self,
        fleet_results: list[FleetResult],
        task_prompts: dict[str, str],
        *,
        on_status: Any | None = None,
    ) -> list[FleetResult]:
        """Retry each failed non-timeout agent task once."""
        if not fleet_results:
            return fleet_results

        results_by_id = {fr.task_id: fr for fr in fleet_results}
        ordered_ids = [fr.task_id for fr in fleet_results]

        for task_id in ordered_ids:
            current = results_by_id.get(task_id)
            if current is None or current.success or current.error is None:
                continue
            if self._is_timeout_error(current.error):
                continue
            prompt_template = task_prompts.get(task_id)
            if not prompt_template:
                continue

            retry_prompt = self._materialize_prompt(prompt_template, results_by_id)
            retry_task_id = f"{task_id}__retry"
            retry_fleet = Fleet(self._config, fleet_config=self._fleet_config)
            retry_fleet.add(
                retry_prompt,
                task_id=retry_task_id,
                model=self._config.model,
                retries=0,
            )
            retry_results = await retry_fleet.run(on_status=on_status)
            retry_result = next(
                (result for result in retry_results if result.task_id == retry_task_id),
                None,
            )
            if retry_result is None:
                continue

            results_by_id[task_id] = FleetResult(
                task_id=task_id,
                success=retry_result.success,
                response=retry_result.response,
                error=retry_result.error,
                duration_ms=current.duration_ms + retry_result.duration_ms,
                prompt_tokens=current.prompt_tokens + retry_result.prompt_tokens,
                completion_tokens=(
                    current.completion_tokens + retry_result.completion_tokens
                ),
                total_cost=current.total_cost + retry_result.total_cost,
                retries_used=current.retries_used + 1,
            )

        return [results_by_id[task_id] for task_id in ordered_ids]

    def _build_feedback_iteration_prompt(
        self,
        base_prompt: str,
        tester_results: list[SquadAgentResult],
        iteration: int,
    ) -> str:
        """Append tester findings so Developer can fix and Tester can re-verify."""
        feedback_parts: list[str] = []
        for tester_result in tester_results:
            if tester_result.content.strip():
                feedback_parts.append(
                    f"### {tester_result.agent.name} Findings\n\n"
                    f"{tester_result.content.strip()}"
                )
            if tester_result.error:
                feedback_parts.append(
                    f"### {tester_result.agent.name} Error\n\n"
                    f"{tester_result.error}"
                )

        feedback_body = (
            "\n\n".join(feedback_parts)
            if feedback_parts
            else "Tester reported failures or issues that must be fixed."
        )
        return (
            f"{base_prompt}\n\n"
            f"## Tester Feedback Loop (Iteration {iteration})\n\n"
            "Developer: fix every failing test and issue reported below.\n"
            "Tester: re-run tests against the updated implementation and "
            "explicitly state whether all tests pass.\n\n"
            f"{feedback_body}"
        )

    @staticmethod
    def _tester_reported_issues(content: str) -> bool:
        """Best-effort signal for tester-reported issues in successful output."""
        if not content:
            return False
        text = content.lower()
        pass_markers = (
            "all tests passed",
            "tests passed",
            "all checks passed",
            "no issues found",
            "no issues detected",
            "no failing tests",
            "no failed tests",
            "no errors found",
        )
        if any(marker in text for marker in pass_markers):
            return False
        issue_markers = (
            "test failed",
            "tests failed",
            "failing test",
            "failing tests",
            "failure",
            "errors found",
            "issue found",
            "issues found",
            "bug found",
            "regression",
            "does not pass",
            "did not pass",
        )
        return any(marker in text for marker in issue_markers)

    @staticmethod
    def _default_timeout_from_reasoning(reasoning_effort: Any) -> float:
        value = (
            reasoning_effort.value
            if hasattr(reasoning_effort, "value")
            else str(reasoning_effort)
        )
        return _REASONING_TIMEOUTS.get(str(value).lower(), 600.0)

    @staticmethod
    def _is_timeout_error(error: Any) -> bool:
        if error is None:
            return False
        if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
            return True
        text = str(error).lower()
        return "timed out" in text or "timeout" in text

    @staticmethod
    def _materialize_prompt(
        prompt: str,
        results: dict[str, FleetResult],
    ) -> str:
        """Resolve {{task:...}} templates using known results."""
        if "{{task:" not in prompt:
            return prompt

        def _replace(match: re.Match[str]) -> str:
            task_id = match.group("task_id")
            field = match.group("field")
            result = results.get(task_id)
            if result is None:
                return ""
            if field == "success":
                return "true" if result.success else "false"
            if field == "error":
                return str(result.error) if result.error else ""
            if field == "duration_ms":
                return str(round(result.duration_ms, 3))
            if field == "prompt_tokens":
                return str(result.prompt_tokens)
            if field == "completion_tokens":
                return str(result.completion_tokens)
            response = result.response
            if response is None:
                return ""
            if field == "content":
                return response.content
            if field == "reasoning":
                return response.reasoning or ""
            return ""

        return _TASK_OUTPUT_REF_RE.sub(_replace, prompt)

    def _discover_project_context(self) -> str:
        """Auto-discover project context from the current working directory.

        Reads README.md (first 2000 chars), pyproject.toml name/description,
        and top-level directory listing. Returns empty string if nothing found.
        """
        context = _gather_repo_context()
        if not context:
            return ""

        sections: list[str] = []

        if context.get("project_name"):
            sections.append(f"Project: {context['project_name']}")
        if context.get("description"):
            sections.append(f"Description: {context['description']}")
        if context.get("readme_excerpt"):
            sections.append(f"README (excerpt):\n{context['readme_excerpt']}")
        if context.get("directory_structure"):
            # Use simpler structure for project context (just top-level)
            top_level = [s for s in context["directory_structure"] if not s.startswith("  ")]
            if top_level:
                sections.append(f"Structure: {', '.join(top_level[:20])}")

        if not sections:
            return ""

        return "## Project Context\n\n" + "\n".join(sections)

    def _build_result(
        self,
        fleet_results: list[FleetResult],
        task_ids: dict[str, str],
    ) -> SquadResult:
        """Convert fleet results to SquadResult.

        When an agent has subtasks, its multiple FleetResults are merged
        into a single SquadAgentResult with combined content and totals.
        """
        # Build reverse map: fleet task_id → role
        id_to_role: dict[str, str] = {}
        for role, tid_value in task_ids.items():
            for tid in tid_value.split("|"):
                id_to_role[tid] = role

        # Group fleet results by role
        role_results: dict[str, list[FleetResult]] = {}
        for fr in fleet_results:
            role = id_to_role.get(fr.task_id)
            if role is not None:
                role_results.setdefault(role, []).append(fr)

        agent_results = []
        all_success = True

        for agent in self.team.agents:
            results = role_results.get(agent.role, [])
            if not results:
                continue

            if len(results) == 1:
                # Single task (no subtasks) — same as before
                fr = results[0]
                content = fr.response.content if fr.response else ""
                ar = SquadAgentResult(
                    agent=agent,
                    content=content,
                    success=fr.success,
                    duration_ms=fr.duration_ms,
                    error=str(fr.error) if fr.error else None,
                    prompt_tokens=fr.prompt_tokens,
                    completion_tokens=fr.completion_tokens,
                )
            else:
                # Merge multiple subtask results
                parts: list[str] = []
                total_duration = 0.0
                total_prompt = 0
                total_completion = 0
                sub_success = True
                errors: list[str] = []

                for j, fr in enumerate(results):
                    content = fr.response.content if fr.response else ""
                    if content:
                        label = (
                            agent.subtasks[j]
                            if j < len(agent.subtasks)
                            else f"Subtask {j + 1}"
                        )
                        parts.append(f"### {label}\n\n{content}")
                    total_duration += fr.duration_ms
                    total_prompt += fr.prompt_tokens
                    total_completion += fr.completion_tokens
                    if not fr.success:
                        sub_success = False
                        if fr.error:
                            errors.append(str(fr.error))

                ar = SquadAgentResult(
                    agent=agent,
                    content="\n\n".join(parts),
                    success=sub_success,
                    duration_ms=total_duration,
                    error="; ".join(errors) if errors else None,
                    prompt_tokens=total_prompt,
                    completion_tokens=total_completion,
                )

            agent_results.append(ar)
            if not ar.success:
                all_success = False

        return SquadResult(
            agent_results=agent_results,
            success=all_success,
        )
