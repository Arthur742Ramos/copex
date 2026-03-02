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
import hashlib
import json
import logging
import os
import re
import time
from collections.abc import AsyncIterator
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from copex.config import CopexConfig
from copex.fleet import (
    _TASK_OUTPUT_REF_RE,
    Fleet,
    FleetConfig,
    FleetEventType,
    FleetMailbox,
    FleetResult,
)
from copex.streaming import Response

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


class SquadError(Exception):
    """Base class for squad-specific failures."""


class SquadConfigError(SquadError):
    """Raised when squad configuration is invalid."""


class SquadExecutionError(SquadError):
    """Raised when squad execution fails."""


class SquadTimeoutError(SquadExecutionError):
    """Raised when a squad task times out."""


class SquadDependencyError(SquadExecutionError):
    """Raised when a squad dependency fails."""


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
_SCAN_SKIP_DIRS = frozenset({"node_modules", "vendor", ".git", "__pycache__"})

_FRONTEND_DIRS = frozenset({
    "frontend", "web", "ui", "app", "pages", "components",
})

_BACKEND_DIRS = frozenset({
    "api", "server", "backend", "services",
})

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
_SQUAD_STATE_FILE_NAME = "state.json"

_KNOWLEDGE_SECTION_HEADING = "What I Learned About This Codebase"
_DECISIONS_SECTION_HEADING = "Key Decisions and Trade-offs"
_MAX_PERSISTED_CONTEXT_CHARS = 4000
_MAX_PERSISTED_ITEM_CHARS = 280
_MAX_PERSISTED_ITEMS = 8
_TAIL_READ_CHUNK_BYTES = 4096
_MAIL_REF_RE = re.compile(r"\{\{\s*mail:(?P<role>[a-zA-Z0-9_.-]+)\s*\}\}")


@dataclass(frozen=True)
class _RepoScanSummary:
    has_source: bool
    has_tests: bool
    source_extensions: tuple[str, ...]


_REPO_CONTEXT_CACHE: dict[Path, dict[str, Any]] = {}
_REPO_SCAN_CACHE: dict[Path, _RepoScanSummary] = {}


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
    if max_chars <= 0:
        return ""

    max_bytes = max_chars * 4
    truncated = False
    chunks: list[bytes] = []
    collected = 0
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        while position > 0 and collected < max_bytes:
            read_size = min(_TAIL_READ_CHUNK_BYTES, position, max_bytes - collected)
            position -= read_size
            handle.seek(position)
            chunk = handle.read(read_size)
            if not chunk:
                break
            chunks.append(chunk)
            collected += len(chunk)
        truncated = position > 0

    text = b"".join(reversed(chunks)).decode("utf-8", errors="ignore").strip()
    if not truncated and len(text) <= max_chars:
        return text
    return "[Older entries truncated]\n" + text[-max_chars:]


def _scan_repo_files(root: Path) -> _RepoScanSummary:
    resolved_root = root.resolve()
    cached = _REPO_SCAN_CACHE.get(resolved_root)
    if cached is not None:
        return cached

    has_source = False
    has_tests = any((resolved_root / name).is_dir() for name in ("tests", "test"))
    extensions: set[str] = set()

    try:
        for _, dirs, files in os.walk(resolved_root, topdown=True):
            dirs[:] = [
                d for d in dirs
                if not d.startswith(".") and d not in _SCAN_SKIP_DIRS
            ]
            for file_name in files:
                ext = Path(file_name).suffix.lower()
                if ext in _SOURCE_EXTENSIONS:
                    has_source = True
                    extensions.add(ext)
                if not has_tests:
                    lower_name = file_name.lower()
                    if lower_name.startswith("test_") and "." in lower_name:
                        has_tests = True
                    elif "." in lower_name and lower_name.rsplit(".", 1)[0].endswith("_test"):
                        has_tests = True
                    elif ".spec." in lower_name:
                        has_tests = True
            if has_source and has_tests and len(extensions) == len(_SOURCE_EXTENSIONS):
                break
    except OSError:
        pass

    summary = _RepoScanSummary(
        has_source=has_source,
        has_tests=has_tests,
        source_extensions=tuple(sorted(extensions)),
    )
    _REPO_SCAN_CACHE[resolved_root] = summary
    return summary


def _has_source_files(root: Path) -> bool:
    """Check for source code files."""
    return _scan_repo_files(root).has_source


def _has_test_files(root: Path) -> bool:
    """Check for test directories or test file patterns."""
    return _scan_repo_files(root).has_tests


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


def _clear_repo_context_cache() -> None:
    _REPO_CONTEXT_CACHE.clear()
    _REPO_SCAN_CACHE.clear()


def _gather_repo_context(path: Path | None = None) -> dict[str, Any]:
    """Gather repository context for AI analysis.

    Reads pyproject.toml, README.md, directory structure, and source file
    extensions. Returns a dictionary with all available context.
    """
    root = (path or Path.cwd()).resolve()
    cached = _REPO_CONTEXT_CACHE.get(root)
    if cached is not None:
        return deepcopy(cached)

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

    scan_summary = _scan_repo_files(root)
    if scan_summary.source_extensions:
        context["source_extensions"] = list(scan_summary.source_extensions)

    _REPO_CONTEXT_CACHE[root] = deepcopy(context)
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


def _normalize_depends_on_roles(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        role = _normalize_role_identifier(str(item))
        if role and role not in seen:
            seen.add(role)
            normalized.append(role)
    return normalized


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


def _squad_state_path(path: Path | None = None) -> Path:
    return _squad_dir_path(path) / _SQUAD_STATE_FILE_NAME


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
    depends_on: list[str] = field(default_factory=list)
    subtasks: list[str] = field(default_factory=list)
    retries: int = 1
    retry_delay: float | None = None

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


from .squad_team import SquadTeam  # noqa: E402


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


class SquadAggregationStrategy(str, Enum):
    """Strategies for aggregating squad agent content."""

    CONCAT = "concat"
    SUCCESS_ONLY = "success_only"
    FAILURES_ONLY = "failures_only"


@dataclass
class SquadResult:
    """Aggregated result from the squad."""

    agent_results: list[SquadAgentResult] = field(default_factory=list)
    total_duration_ms: float = 0.0
    success: bool = True
    role_dependencies: dict[str, list[str]] = field(default_factory=dict)
    task_dependencies: dict[str, list[str]] = field(default_factory=dict)

    @property
    def final_content(self) -> str:
        """Get the combined output from all agents."""
        return self.aggregate_content()

    @staticmethod
    def _coerce_aggregation_strategy(
        strategy: SquadAggregationStrategy | str,
    ) -> SquadAggregationStrategy:
        if isinstance(strategy, SquadAggregationStrategy):
            return strategy
        try:
            return SquadAggregationStrategy(str(strategy).strip().lower())
        except ValueError as exc:
            raise ValueError(f"Unsupported squad aggregation strategy: {strategy}") from exc

    def aggregate_content(
        self,
        *,
        strategy: SquadAggregationStrategy | str = SquadAggregationStrategy.CONCAT,
        roles: list[str] | None = None,
        separator: str = "\n\n---\n\n",
    ) -> str:
        """Aggregate agent content using a selectable strategy."""
        selected_strategy = self._coerce_aggregation_strategy(strategy)
        role_filter = (
            {_normalize_role_identifier(role) for role in roles}
            if roles is not None
            else None
        )

        parts = []
        for ar in self.agent_results:
            if role_filter is not None and ar.agent.role not in role_filter:
                continue
            if selected_strategy == SquadAggregationStrategy.SUCCESS_ONLY and not ar.success:
                continue
            if selected_strategy == SquadAggregationStrategy.FAILURES_ONLY and ar.success:
                continue
            if ar.content:
                parts.append(f"## {ar.agent.emoji} {ar.agent.name}\n\n{ar.content}")
        return separator.join(parts)

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
            "dependencies": {
                "roles": {
                    role: list(dependencies)
                    for role, dependencies in self.role_dependencies.items()
                },
                "tasks": {
                    task_id: list(dependencies)
                    for task_id, dependencies in self.task_dependencies.items()
                },
            },
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class SquadEventType(str, Enum):
    """Types of squad-level progress events."""

    PHASE_STARTED = "phase_started"
    PHASE_COMPLETED = "phase_completed"
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    SQUAD_COMPLETED = "squad_completed"


@dataclass
class SquadEvent:
    """Progress event emitted by ``SquadCoordinator.run_streaming``."""

    event_type: SquadEventType
    role: str | None = None
    phase: int | None = None
    agent: SquadAgent | None = None
    status: str | None = None
    success: bool | None = None
    iteration: int = 1
    error: str | None = None
    result: SquadResult | None = None


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
        max_cost: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self._config = config
        self._team = team  # Can be None for lazy init
        self._fleet_config = fleet_config or FleetConfig(
            max_concurrent=3,
            timeout=self._default_timeout_from_reasoning(config.reasoning_effort),
        )
        self._max_cost = max_cost
        self._max_tokens = max_tokens
        self._project_context: str | None = None
        self._repo_map: Any | None = None
        self._has_local_state = False

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
        auto_approve_gates: bool | None = None,
        force: bool = False,
        interactive: bool | None = None,
        mailbox: FleetMailbox | None = None,
    ) -> SquadResult:
        """Run the squad on a task.

        Args:
            prompt: The task to accomplish.
            on_status: Optional callback(task_id, status) for progress.
            auto_approve_gates: Skip phase gate prompts when True.
            force: Force rerun of all tasks and ignore incremental cache.
            interactive: Override whether gate prompts should be interactive.
            mailbox: Optional FleetMailbox for inter-agent communication.

        Returns:
            SquadResult with results from all agents.
        """
        start_time = time.monotonic()

        # Lazy init: prefer .squad, then AI repo analysis
        repo_context: dict[str, Any] | None = None
        if self._project_context is None:
            repo_context = _gather_repo_context()

        if self._team is None:
            self._team = await SquadTeam.from_repo_or_file(
                self._config,
                repo_context=repo_context,
            )
            team_path = SquadTeam._resolve_squad_file_path()
            if not team_path.is_file():
                self._team.save_squad_file()
                logger.info("Squad team auto-saved to %s", team_path)

        if self._project_context is None:
            self._project_context = self._discover_project_context(
                repo_context=repo_context
            )
        if self._repo_map is None:
            try:
                from copex.repo_map import RepoMap

                self._repo_map = RepoMap(Path.cwd())
                self._repo_map.refresh(force=False)
            except Exception as exc:
                logger.info("Repo map unavailable for squad context: %s", exc)
                self._repo_map = None

        self._validate_team_dependency_graph()
        interactive_mode = self._is_interactive_tty() if interactive is None else interactive
        gate_auto_approve = (not interactive_mode) if auto_approve_gates is None else (
            auto_approve_gates or not interactive_mode
        )
        use_controlled_execution = (
            bool(self.team.phase_gates)
            or force
            or (self._has_local_state and _squad_state_path().is_file())
        )

        max_feedback_iterations = 3
        feedback_iteration = 0
        current_prompt = prompt
        cumulative_cost = 0.0
        cumulative_tokens = 0
        shared_mailbox = mailbox or FleetMailbox()

        final_result: SquadResult | None = None
        while True:
            task_prompts: dict[str, str] = {}
            if use_controlled_execution:
                fleet_results, task_ids = await self._run_iteration_with_controls(
                    current_prompt,
                    on_status=on_status,
                    auto_approve_gates=gate_auto_approve,
                    interactive=interactive_mode,
                    force=force,
                    mailbox=shared_mailbox,
                )
            else:
                fleet = Fleet(self._config, fleet_config=self._fleet_config)
                task_ids = self._add_tasks(
                    fleet,
                    current_prompt,
                    task_prompts=task_prompts,
                    mailbox=shared_mailbox,
                )
                fleet_results = await self._run_fleet(
                    fleet,
                    on_status=on_status,
                    mailbox=shared_mailbox,
                )
                fleet_results = await self._retry_failed_agents(
                    fleet_results,
                    task_prompts,
                    on_status=on_status,
                    mailbox=shared_mailbox,
                )
                self._persist_state_from_results(
                    task_ids=task_ids,
                    task_prompts=task_prompts,
                    fleet_results=fleet_results,
                )
            cumulative_cost += sum(fr.total_cost for fr in fleet_results)
            cumulative_tokens += sum(
                fr.prompt_tokens + fr.completion_tokens
                for fr in fleet_results
            )
            self._enforce_budget(cumulative_cost, cumulative_tokens)
            iteration_result = self._build_result(fleet_results, task_ids)
            role_dependencies, task_dependencies = self._dependency_graphs(task_ids)
            iteration_result.role_dependencies = role_dependencies
            iteration_result.task_dependencies = task_dependencies
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

    async def run_streaming(
        self,
        prompt: str,
        *,
        on_status: Any | None = None,
        mailbox: FleetMailbox | None = None,
    ) -> AsyncIterator[SquadEvent]:
        """Run the squad while emitting real-time squad-level progress events."""
        start_time = time.monotonic()

        repo_context: dict[str, Any] | None = None
        if self._project_context is None:
            repo_context = _gather_repo_context()

        if self._team is None:
            self._team = await SquadTeam.from_repo_or_file(
                self._config,
                repo_context=repo_context,
            )
            team_path = SquadTeam._resolve_squad_file_path()
            if not team_path.is_file():
                self._team.save_squad_file()
                logger.info("Squad team auto-saved to %s", team_path)

        if self._project_context is None:
            self._project_context = self._discover_project_context(
                repo_context=repo_context
            )
        if self._repo_map is None:
            try:
                from copex.repo_map import RepoMap

                self._repo_map = RepoMap(Path.cwd())
                self._repo_map.refresh(force=False)
            except Exception as exc:
                logger.info("Repo map unavailable for squad context: %s", exc)
                self._repo_map = None

        self._validate_team_dependency_graph()

        max_feedback_iterations = 3
        feedback_iteration = 0
        current_prompt = prompt
        cumulative_cost = 0.0
        cumulative_tokens = 0
        shared_mailbox = mailbox or FleetMailbox()

        final_result: SquadResult | None = None
        while True:
            fleet = Fleet(self._config, fleet_config=self._fleet_config)
            task_prompts: dict[str, str] = {}
            task_ids = self._add_tasks(
                fleet,
                current_prompt,
                task_prompts=task_prompts,
                mailbox=shared_mailbox,
            )
            task_to_role = self._task_to_role_map(task_ids)
            role_to_task_ids = {
                role: [tid for tid in tid_value.split("|") if tid]
                for role, tid_value in task_ids.items()
            }
            role_to_agent = {agent.role: agent for agent in self.team.agents}
            phase_to_task_ids: dict[int, set[str]] = {}
            for role, role_task_ids in role_to_task_ids.items():
                agent = role_to_agent.get(role)
                phase = agent.phase if agent is not None else 2
                phase_to_task_ids.setdefault(phase, set()).update(role_task_ids)

            started_phases: set[int] = set()
            completed_phases: set[int] = set()
            started_roles: set[str] = set()
            completed_roles: set[str] = set()
            completed_tasks: set[str] = set()
            task_success: dict[str, bool] = {}

            async with fleet:
                async for event in fleet.run_streaming(mailbox=shared_mailbox):
                    task_id = event.task_id
                    if task_id is None:
                        continue
                    role = task_to_role.get(task_id)
                    if role is None:
                        continue
                    agent = role_to_agent.get(role)
                    phase = agent.phase if agent is not None else 2

                    if event.event_type == FleetEventType.TASK_RUNNING:
                        if phase not in started_phases:
                            started_phases.add(phase)
                            yield SquadEvent(
                                event_type=SquadEventType.PHASE_STARTED,
                                phase=phase,
                                status="running",
                                iteration=feedback_iteration + 1,
                            )
                        if role not in started_roles:
                            started_roles.add(role)
                            if on_status:
                                on_status(role, "running")
                            yield SquadEvent(
                                event_type=SquadEventType.AGENT_STARTED,
                                role=role,
                                phase=phase,
                                agent=agent,
                                status="running",
                                iteration=feedback_iteration + 1,
                            )
                        continue

                    if event.event_type not in {
                        FleetEventType.TASK_DONE,
                        FleetEventType.TASK_FAILED,
                        FleetEventType.TASK_BLOCKED,
                        FleetEventType.TASK_CANCELLED,
                    }:
                        continue

                    completed_tasks.add(task_id)
                    task_success[task_id] = event.event_type == FleetEventType.TASK_DONE

                    role_task_ids = role_to_task_ids.get(role, [])
                    if role not in completed_roles and all(
                        task in completed_tasks for task in role_task_ids
                    ):
                        completed_roles.add(role)
                        role_success = all(task_success.get(task, False) for task in role_task_ids)
                        status = "done" if role_success else "failed"
                        if on_status:
                            on_status(role, status)
                        yield SquadEvent(
                            event_type=SquadEventType.AGENT_COMPLETED,
                            role=role,
                            phase=phase,
                            agent=agent,
                            status=status,
                            success=role_success,
                            error=event.error,
                            iteration=feedback_iteration + 1,
                        )

                    phase_task_ids = phase_to_task_ids.get(phase, set())
                    if phase in completed_phases or not phase_task_ids:
                        continue
                    if not phase_task_ids.issubset(completed_tasks):
                        continue
                    completed_phases.add(phase)
                    phase_success = all(task_success.get(task, False) for task in phase_task_ids)
                    yield SquadEvent(
                        event_type=SquadEventType.PHASE_COMPLETED,
                        phase=phase,
                        status="done" if phase_success else "failed",
                        success=phase_success,
                        iteration=feedback_iteration + 1,
                    )

            fleet_results = fleet.last_results
            if not fleet_results:
                raise SquadExecutionError("Squad execution produced no fleet results")

            fleet_results = await self._retry_failed_agents(
                fleet_results,
                task_prompts,
                on_status=on_status,
                mailbox=shared_mailbox,
            )
            cumulative_cost += sum(fr.total_cost for fr in fleet_results)
            cumulative_tokens += sum(
                fr.prompt_tokens + fr.completion_tokens
                for fr in fleet_results
            )
            self._enforce_budget(cumulative_cost, cumulative_tokens)
            self._persist_state_from_results(
                task_ids=task_ids,
                task_prompts=task_prompts,
                fleet_results=fleet_results,
            )

            iteration_result = self._build_result(fleet_results, task_ids)
            role_dependencies, task_dependencies = self._dependency_graphs(task_ids)
            iteration_result.role_dependencies = role_dependencies
            iteration_result.task_dependencies = task_dependencies
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
        yield SquadEvent(
            event_type=SquadEventType.SQUAD_COMPLETED,
            status="done" if final_result.success else "failed",
            success=final_result.success,
            result=final_result,
            iteration=feedback_iteration + 1,
        )

    @staticmethod
    def _task_to_role_map(task_ids: dict[str, str]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for role, tid_value in task_ids.items():
            for task_id in tid_value.split("|"):
                if task_id:
                    mapping[task_id] = role
        return mapping

    @staticmethod
    async def _run_fleet(
        fleet: Fleet,
        *,
        on_status: Any | None,
        mailbox: FleetMailbox | None = None,
    ) -> list[FleetResult]:
        if mailbox is not None:
            try:
                return await fleet.run(on_status=on_status, mailbox=mailbox)
            except TypeError as exc:
                if "mailbox" not in str(exc):
                    raise
        return await fleet.run(on_status=on_status)

    def _dependency_graphs(
        self,
        task_ids: dict[str, str],
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        role_dependencies: dict[str, list[str]] = {}
        task_dependencies: dict[str, list[str]] = {}

        for agent in self.team.agents:
            role_dependencies[agent.role] = self._dependency_roles(agent)
            deps = self._get_dependencies(agent, task_ids)
            for task_id in [tid for tid in task_ids.get(agent.role, "").split("|") if tid]:
                task_dependencies[task_id] = list(deps)

        return role_dependencies, task_dependencies

    def _validate_team_dependency_graph(self) -> None:
        role_ids = [agent.role for agent in self.team.agents]
        role_set = set(role_ids)
        missing: list[str] = []
        for agent in self.team.agents:
            for dep_role in self._dependency_roles(agent):
                if dep_role not in role_set:
                    missing.append(f"{agent.role} -> {dep_role}")
        if missing:
            raise SquadConfigError(
                "Squad dependency graph references unknown roles: " + ", ".join(sorted(missing))
            )

        in_degree: dict[str, int] = {role: 0 for role in role_ids}
        adjacency: dict[str, list[str]] = {role: [] for role in role_ids}
        for agent in self.team.agents:
            for dep_role in self._dependency_roles(agent):
                adjacency[dep_role].append(agent.role)
                in_degree[agent.role] += 1

        queue: list[str] = [role for role, degree in in_degree.items() if degree == 0]
        visited = 0
        while queue:
            role = queue.pop(0)
            visited += 1
            for dependent in adjacency[role]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if visited != len(role_ids):
            cycle_roles = sorted(role for role, degree in in_degree.items() if degree > 0)
            raise SquadConfigError(
                "Cycle detected in squad dependencies: " + ", ".join(cycle_roles)
            )

    @staticmethod
    def _stable_json(payload: Any) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def _build_task_id_map(self) -> dict[str, str]:
        task_ids: dict[str, str] = {}
        for agent in self.team.agents:
            if agent.subtasks:
                task_ids[agent.role] = "|".join(
                    f"{agent.role}__sub{i + 1}" for i in range(len(agent.subtasks))
                )
            else:
                task_ids[agent.role] = agent.role
        return task_ids

    def _compute_execution_levels(self) -> dict[str, int]:
        role_ids = [agent.role for agent in self.team.agents]
        in_degree: dict[str, int] = {role: 0 for role in role_ids}
        adjacency: dict[str, list[str]] = {role: [] for role in role_ids}
        levels: dict[str, int] = {role: 1 for role in role_ids}
        order_index = {role: idx for idx, role in enumerate(role_ids)}

        for agent in self.team.agents:
            for dep_role in self._dependency_roles(agent):
                adjacency[dep_role].append(agent.role)
                in_degree[agent.role] += 1

        queue = sorted(
            [role for role, degree in in_degree.items() if degree == 0],
            key=lambda role: order_index[role],
        )
        while queue:
            role = queue.pop(0)
            for dependent in sorted(adjacency[role], key=lambda dep: order_index[dep]):
                levels[dependent] = max(levels.get(dependent, 1), levels[role] + 1)
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        return levels

    def _load_squad_state(self) -> dict[str, Any]:
        path = _squad_state_path()
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_squad_state(self, state: dict[str, Any]) -> None:
        _ensure_squad_workspace()
        path = _squad_state_path()
        path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    def _task_input_hash(
        self,
        *,
        task_id: str,
        rendered_prompt: str,
        dependency_hashes: dict[str, str],
    ) -> str:
        payload = {
            "task_id": task_id,
            "prompt": rendered_prompt,
            "dependency_hashes": dependency_hashes,
        }
        return hashlib.sha256(self._stable_json(payload).encode("utf-8")).hexdigest()

    def _render_mailbox_refs(
        self,
        prompt: str,
        messages_by_sender: dict[str, list[str]],
        *,
        known_roles: set[str] | None = None,
    ) -> str:
        if "{{mail:" not in prompt:
            return prompt

        normalized_known_roles = {
            _normalize_role_identifier(role)
            for role in (known_roles or {agent.role for agent in self.team.agents})
        }

        def _replace(match: re.Match[str]) -> str:
            sender_role = _normalize_role_identifier(match.group("role"))
            if sender_role not in normalized_known_roles and sender_role not in messages_by_sender:
                return ""
            messages = messages_by_sender.get(sender_role, [])
            if not messages:
                return ""
            return "\n\n".join(messages)

        return _MAIL_REF_RE.sub(_replace, prompt)

    def _collect_mail_for_role(
        self,
        role: str,
        *,
        task_ids: dict[str, str],
        task_to_role: dict[str, str],
        mailbox: FleetMailbox,
    ) -> dict[str, list[str]]:
        messages_by_sender: dict[str, list[str]] = {}
        role_task_ids = [tid for tid in task_ids.get(role, "").split("|") if tid]
        for task_id in role_task_ids:
            while True:
                envelope = mailbox.try_receive(task_id)
                if envelope is None:
                    break
                from_task = str(envelope.get("from") or "")
                sender_role = task_to_role.get(from_task, _normalize_role_identifier(from_task))
                payload = envelope.get("payload")
                text = ""
                if isinstance(payload, dict):
                    text = str(payload.get("content", "")).strip()
                elif payload is not None:
                    text = str(payload).strip()
                if text:
                    messages_by_sender.setdefault(sender_role, []).append(text)
        return messages_by_sender

    async def _relay_mailbox_messages(
        self,
        *,
        mailbox: FleetMailbox,
        task_ids: dict[str, str],
        task_to_role: dict[str, str],
        phase_results: list[FleetResult],
    ) -> None:
        dependents_by_role: dict[str, list[str]] = {agent.role: [] for agent in self.team.agents}
        for agent in self.team.agents:
            for dep_role in self._dependency_roles(agent):
                dependents_by_role.setdefault(dep_role, []).append(agent.role)

        for result in phase_results:
            sender_role = task_to_role.get(result.task_id)
            if sender_role is None:
                continue
            content = (result.response.content if result.response else "").strip()
            if not content:
                continue
            targets = dependents_by_role.get(sender_role, [])
            for target_role in targets:
                for target_task_id in [tid for tid in task_ids.get(target_role, "").split("|") if tid]:
                    await mailbox.send(
                        target_task_id,
                        {"content": content},
                        from_task=result.task_id,
                    )

    def _is_interactive_tty(self) -> bool:
        try:
            return os.isatty(0) and os.isatty(1)
        except Exception:
            return False

    def _should_gate_phase(self, phase: int) -> bool:
        return bool(getattr(self.team, "phase_gates", {}).get(phase, False))

    def _approve_phase_gate(
        self,
        *,
        phase: int,
        phase_result: SquadResult,
        auto_approve: bool,
        interactive: bool,
    ) -> bool:
        if not self._should_gate_phase(phase):
            return True
        if auto_approve or not interactive:
            return True

        print(f"\nPhase {phase} completed. Review results before continuing:")
        for agent_result in phase_result.agent_results:
            status = "ok" if agent_result.success else "failed"
            print(f"- {agent_result.agent.name} ({agent_result.agent.role}): {status}")
        try:
            decision = input("Approve next phase? [y/N]: ").strip().lower()
        except EOFError:
            return False
        return decision in {"y", "yes"}

    async def _run_iteration_with_controls(
        self,
        prompt: str,
        *,
        on_status: Any | None,
        auto_approve_gates: bool,
        interactive: bool,
        force: bool,
        mailbox: FleetMailbox,
    ) -> tuple[list[FleetResult], dict[str, str]]:
        task_ids = self._build_task_id_map()
        known_roles = set(task_ids)
        task_to_role = self._task_to_role_map(task_ids)
        levels = self._compute_execution_levels()
        agents_by_level: dict[int, list[SquadAgent]] = {}
        for agent in self.team.agents:
            level = levels.get(agent.role, max(1, agent.phase))
            agents_by_level.setdefault(level, []).append(agent)

        for tid_value in task_ids.values():
            for task_id in tid_value.split("|"):
                if task_id:
                    mailbox.create_inbox(task_id)

        state = {} if force else self._load_squad_state()
        state_tasks = state.get("tasks", {}) if isinstance(state.get("tasks"), dict) else {}
        next_state_tasks: dict[str, Any] = {}
        results_by_task_id: dict[str, FleetResult] = {}
        input_hashes: dict[str, str] = {}

        for level in sorted(agents_by_level):
            phase_agents = agents_by_level[level]
            fleet = Fleet(self._config, fleet_config=self._fleet_config)
            task_prompts: dict[str, str] = {}
            phase_hashes: dict[str, str] = {}
            cached_results: dict[str, FleetResult] = {}

            for agent in phase_agents:
                dep_task_ids = self._get_dependencies(agent, task_ids)
                dep_hashes = {
                    dep_id: input_hashes.get(dep_id, "")
                    for dep_id in dep_task_ids
                }
                mail_messages = self._collect_mail_for_role(
                    agent.role,
                    task_ids=task_ids,
                    task_to_role=task_to_role,
                    mailbox=mailbox,
                )

                if agent.subtasks:
                    for i, subtask_desc in enumerate(agent.subtasks):
                        task_id = f"{agent.role}__sub{i + 1}"
                        sub_prompt = self._build_agent_prompt(
                            agent,
                            f"{prompt}\n\n## Subtask ({i + 1}/{len(agent.subtasks)})"
                            f"\n\nFocus on: {subtask_desc}",
                            task_ids,
                        )
                        prompt_with_mail = self._render_mailbox_refs(
                            sub_prompt,
                            mail_messages,
                            known_roles=known_roles,
                        )
                        rendered_prompt = self._materialize_prompt(prompt_with_mail, results_by_task_id)
                        input_hash = self._task_input_hash(
                            task_id=task_id,
                            rendered_prompt=rendered_prompt,
                            dependency_hashes=dep_hashes,
                        )
                        phase_hashes[task_id] = input_hash
                        prior = state_tasks.get(task_id) if isinstance(state_tasks, dict) else None
                        if (
                            not force
                            and isinstance(prior, dict)
                            and prior.get("input_hash") == input_hash
                            and bool(prior.get("success", False))
                        ):
                            content = str(prior.get("content", ""))
                            cached = FleetResult(
                                task_id=task_id,
                                success=True,
                                response=Response(content=content),
                                duration_ms=float(prior.get("duration_ms", 0.0)),
                                prompt_tokens=int(prior.get("prompt_tokens", 0)),
                                completion_tokens=int(prior.get("completion_tokens", 0)),
                                total_cost=float(prior.get("total_cost", 0.0)),
                            )
                            cached_results[task_id] = cached
                            results_by_task_id[task_id] = cached
                            input_hashes[task_id] = input_hash
                            continue

                        tid = fleet.add(
                            rendered_prompt,
                            task_id=task_id,
                            depends_on=[],
                            model=self._config.model,
                            retries=0,
                        )
                        task_prompts[tid] = rendered_prompt
                else:
                    task_id = agent.role
                    agent_prompt = self._build_agent_prompt(agent, prompt, task_ids)
                    prompt_with_mail = self._render_mailbox_refs(
                        agent_prompt,
                        mail_messages,
                        known_roles=known_roles,
                    )
                    rendered_prompt = self._materialize_prompt(prompt_with_mail, results_by_task_id)
                    input_hash = self._task_input_hash(
                        task_id=task_id,
                        rendered_prompt=rendered_prompt,
                        dependency_hashes=dep_hashes,
                    )
                    phase_hashes[task_id] = input_hash
                    prior = state_tasks.get(task_id) if isinstance(state_tasks, dict) else None
                    if (
                        not force
                        and isinstance(prior, dict)
                        and prior.get("input_hash") == input_hash
                        and bool(prior.get("success", False))
                    ):
                        content = str(prior.get("content", ""))
                        cached = FleetResult(
                            task_id=task_id,
                            success=True,
                            response=Response(content=content),
                            duration_ms=float(prior.get("duration_ms", 0.0)),
                            prompt_tokens=int(prior.get("prompt_tokens", 0)),
                            completion_tokens=int(prior.get("completion_tokens", 0)),
                            total_cost=float(prior.get("total_cost", 0.0)),
                        )
                        cached_results[task_id] = cached
                        results_by_task_id[task_id] = cached
                        input_hashes[task_id] = input_hash
                        continue

                    tid = fleet.add(
                        rendered_prompt,
                        task_id=task_id,
                        depends_on=[],
                        model=self._config.model,
                        retries=0,
                    )
                    task_prompts[tid] = rendered_prompt

            run_results = (
                await self._run_fleet(fleet, on_status=on_status, mailbox=mailbox)
                if fleet.tasks
                else []
            )
            run_results = await self._retry_failed_agents(
                run_results,
                task_prompts,
                on_status=on_status,
                mailbox=mailbox,
            )
            run_results_by_id = {result.task_id: result for result in run_results}

            phase_task_order = [
                tid
                for agent in phase_agents
                for tid in task_ids.get(agent.role, "").split("|")
                if tid
            ]
            phase_results: list[FleetResult] = []
            for task_id in phase_task_order:
                result = run_results_by_id.get(task_id) or cached_results.get(task_id)
                if result is None:
                    continue
                phase_results.append(result)
                results_by_task_id[task_id] = result
                input_hashes[task_id] = phase_hashes.get(task_id, input_hashes.get(task_id, ""))
                next_state_tasks[task_id] = {
                    "role": task_to_role.get(task_id, ""),
                    "input_hash": input_hashes.get(task_id, ""),
                    "success": result.success,
                    "content": result.response.content if result.response else "",
                    "error": str(result.error) if result.error else None,
                    "duration_ms": result.duration_ms,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_cost": result.total_cost,
                    "updated_at": _now_iso_utc(),
                }

            await self._relay_mailbox_messages(
                mailbox=mailbox,
                task_ids=task_ids,
                task_to_role=task_to_role,
                phase_results=phase_results,
            )

            if self._should_gate_phase(level):
                phase_task_ids = {agent.role: task_ids[agent.role] for agent in phase_agents}
                phase_result = self._build_result(phase_results, phase_task_ids)
                approved = self._approve_phase_gate(
                    phase=level,
                    phase_result=phase_result,
                    auto_approve=auto_approve_gates,
                    interactive=interactive,
                )
                if not approved:
                    raise SquadExecutionError(f"Execution stopped: phase {level} was not approved.")

        self._save_squad_state(
            {
                "version": 1,
                "updated_at": _now_iso_utc(),
                "tasks": next_state_tasks,
            }
        )
        self._has_local_state = True

        ordered_task_ids = [
            tid
            for agent in self.team.agents
            for tid in task_ids.get(agent.role, "").split("|")
            if tid
        ]
        ordered_results = [
            results_by_task_id[task_id]
            for task_id in ordered_task_ids
            if task_id in results_by_task_id
        ]
        return ordered_results, task_ids

    def _persist_state_from_results(
        self,
        *,
        task_ids: dict[str, str],
        task_prompts: dict[str, str],
        fleet_results: list[FleetResult],
    ) -> None:
        if not fleet_results:
            return
        known_roles = set(task_ids)
        result_map = {result.task_id: result for result in fleet_results}
        task_to_role = self._task_to_role_map(task_ids)
        input_hashes: dict[str, str] = {}
        state_tasks: dict[str, Any] = {}

        for agent in self.team.agents:
            dep_task_ids = self._get_dependencies(agent, task_ids)
            dep_hashes = {dep: input_hashes.get(dep, "") for dep in dep_task_ids}
            for task_id in [tid for tid in task_ids.get(agent.role, "").split("|") if tid]:
                prompt_template = task_prompts.get(task_id, "")
                mail_messages: dict[str, list[str]] = {}
                for dep_task_id in dep_task_ids:
                    dep_result = result_map.get(dep_task_id)
                    if dep_result is None or dep_result.response is None:
                        continue
                    dep_content = dep_result.response.content.strip()
                    if not dep_content:
                        continue
                    dep_role = task_to_role.get(dep_task_id, "")
                    if dep_role:
                        mail_messages.setdefault(dep_role, []).append(dep_content)
                prompt_with_mail = self._render_mailbox_refs(
                    prompt_template,
                    mail_messages,
                    known_roles=known_roles,
                )
                rendered_prompt = self._materialize_prompt(prompt_with_mail, result_map)
                input_hash = self._task_input_hash(
                    task_id=task_id,
                    rendered_prompt=rendered_prompt,
                    dependency_hashes=dep_hashes,
                )
                input_hashes[task_id] = input_hash
                result = result_map.get(task_id)
                if result is None:
                    continue
                role = task_to_role.get(task_id, agent.role)
                state_tasks[task_id] = {
                    "role": role,
                    "input_hash": input_hash,
                    "success": result.success,
                    "content": result.response.content if result.response else "",
                    "error": str(result.error) if result.error else None,
                    "duration_ms": result.duration_ms,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_cost": result.total_cost,
                    "updated_at": _now_iso_utc(),
                }

        self._save_squad_state(
            {
                "version": 1,
                "updated_at": _now_iso_utc(),
                "tasks": state_tasks,
            }
        )
        self._has_local_state = True

    def _add_tasks(
        self,
        fleet: Fleet,
        prompt: str,
        *,
        task_prompts: dict[str, str] | None = None,
        mailbox: FleetMailbox | None = None,
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
                    if mailbox is not None:
                        mailbox.create_inbox(tid)
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
                if mailbox is not None:
                    mailbox.create_inbox(tid)
                task_ids[agent.role] = tid

        return task_ids

    def _get_dependencies(
        self, agent: SquadAgent, task_ids: dict[str, str]
    ) -> list[str]:
        """Get dependency task IDs from explicit DAG deps or phase fallback."""
        deps: list[str] = []
        for role in self._dependency_roles(agent):
            tid = task_ids.get(role)
            if tid:
                deps.extend(tid.split("|"))
        return deps

    def _dependency_roles(self, agent: SquadAgent) -> list[str]:
        """Get dependency roles for an agent from DAG config or phase fallback."""
        if agent.depends_on:
            deps: list[str] = []
            seen: set[str] = set()
            for dep in agent.depends_on:
                dep_role = _normalize_role_identifier(dep)
                if dep_role and dep_role != agent.role and dep_role not in seen:
                    seen.add(dep_role)
                    deps.append(dep_role)
            return deps

        if agent.phase <= 1:
            return []

        deps = []
        for other in self.team.agents:
            if other.role == agent.role:
                continue
            if other.phase < agent.phase:
                deps.append(other.role)
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

        parts.append("")
        parts.append("## mailbox")
        parts.append(
            "Read inter-agent messages with {{mail:role}} (for example {{mail:lead}}). "
            "Messages are delivered after dependency tasks complete. "
            "If a role has no messages yet, {{mail:role}} resolves to an empty string."
        )

        # Add references to dependency outputs
        dependency_roles = set(self._dependency_roles(agent))
        dependency_refs: list[tuple[SquadAgent, list[str]]] = []
        for other in self.team.agents:
            if other.role == agent.role or other.role not in dependency_roles:
                continue
            other_task_ids = task_ids.get(other.role)
            if not other_task_ids:
                continue
            dependency_refs.append((other, other_task_ids.split("|")))

        if dependency_refs:
            parts.append("")
            parts.append("## Dependency Outputs")
            for other, ref_ids in dependency_refs:
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
        mailbox: FleetMailbox | None = None,
    ) -> list[FleetResult]:
        """Retry each failed non-timeout agent task based on per-role limits."""
        if not fleet_results:
            return fleet_results

        results_by_id = {fr.task_id: fr for fr in fleet_results}
        ordered_ids = [fr.task_id for fr in fleet_results]

        for task_id in ordered_ids:
            prompt_template = task_prompts.get(task_id)
            if not prompt_template:
                continue
            retry_limit = self._retry_limit_for_task(task_id)
            if retry_limit <= 0:
                continue

            for attempt in range(retry_limit):
                current = results_by_id.get(task_id)
                if current is None or current.success or current.error is None:
                    break
                if self._is_timeout_error(current.error):
                    break
                retry_delay = self._retry_delay_for_task(task_id, attempt)
                if retry_delay > 0:
                    await asyncio.sleep(retry_delay)

                retry_prompt = self._materialize_prompt(prompt_template, results_by_id)
                retry_task_id = f"{task_id}__retry" if attempt == 0 else f"{task_id}__retry{attempt + 1}"
                async with Fleet(self._config, fleet_config=self._fleet_config) as retry_fleet:
                    if mailbox is not None:
                        mailbox.create_inbox(retry_task_id)
                    retry_fleet.add(
                        retry_prompt,
                        task_id=retry_task_id,
                        model=self._config.model,
                        retries=0,
                    )
                    retry_results = await self._run_fleet(
                        retry_fleet,
                        on_status=on_status,
                        mailbox=mailbox,
                    )
                retry_result = next(
                    (result for result in retry_results if result.task_id == retry_task_id),
                    None,
                )
                if retry_result is None:
                    break

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

    def _retry_limit_for_task(self, task_id: str) -> int:
        role = task_id.split("__", 1)[0]
        agent = self.team.get_agent(role)
        if agent is None:
            return 1
        try:
            return max(0, int(agent.retries))
        except (TypeError, ValueError):
            return 1

    def _retry_delay_for_task(self, task_id: str, attempt: int) -> float:
        role = task_id.split("__", 1)[0]
        agent = self.team.get_agent(role)
        if agent is None or agent.retry_delay is None:
            return 0.0
        try:
            base_delay = max(0.0, float(agent.retry_delay))
        except (TypeError, ValueError):
            return 0.0
        return base_delay * (2 ** max(0, int(attempt)))

    def _enforce_budget(self, cumulative_cost: float, cumulative_tokens: int) -> None:
        if self._max_cost is not None and cumulative_cost > self._max_cost:
            raise SquadExecutionError(
                "Squad cost budget exceeded: "
                f"spent ${cumulative_cost:.4f} so far (limit ${self._max_cost:.4f})."
            )
        if self._max_tokens is not None and cumulative_tokens > self._max_tokens:
            raise SquadExecutionError(
                "Squad token budget exceeded: "
                f"used {cumulative_tokens:,} tokens so far (limit {self._max_tokens:,})."
            )

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
    def _as_squad_execution_error(error: Any) -> SquadExecutionError:
        if isinstance(error, SquadExecutionError):
            return error
        if error is None:
            return SquadExecutionError("Unknown squad execution error")
        text = str(error)
        lowered = text.lower()
        if "dependency" in lowered or "upstream" in lowered or "blocked" in lowered:
            return SquadDependencyError(text)
        if "timed out" in lowered or "timeout" in lowered:
            return SquadTimeoutError(text)
        return SquadExecutionError(text)

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

    def _discover_project_context(
        self,
        repo_context: dict[str, Any] | None = None,
    ) -> str:
        """Auto-discover project context from the current working directory.

        Reads README.md (first 2000 chars), pyproject.toml name/description,
        and top-level directory listing. Returns empty string if nothing found.
        """
        context = repo_context if repo_context is not None else _gather_repo_context()
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

            if len(results) > 1:
                def _subtask_sort_key(result: FleetResult) -> tuple[int, int]:
                    match = re.search(r"__sub(?P<index>\d+)$", result.task_id)
                    if match is None:
                        return (1, 0)
                    return (0, int(match.group("index")))

                results = sorted(results, key=_subtask_sort_key)

            if len(results) == 1:
                # Single task (no subtasks) — same as before
                fr = results[0]
                content = fr.response.content if fr.response else ""
                ar = SquadAgentResult(
                    agent=agent,
                    content=content,
                    success=fr.success,
                    duration_ms=fr.duration_ms,
                    error=str(self._as_squad_execution_error(fr.error)) if fr.error else None,
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
                            errors.append(str(self._as_squad_execution_error(fr.error)))

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
