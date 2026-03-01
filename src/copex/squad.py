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

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from copex.config import CopexConfig
from copex.fleet import Fleet, FleetConfig, FleetResult

logger = logging.getLogger(__name__)


class SquadRole(str, Enum):
    """Roles available in a squad team."""

    LEAD = "lead"
    DEVELOPER = "developer"
    TESTER = "tester"
    DOCS = "docs"
    DEVOPS = "devops"
    FRONTEND = "frontend"
    BACKEND = "backend"


_ROLE_PROMPTS: dict[SquadRole, str] = {
    SquadRole.LEAD: (
        "You are the Lead Architect. Analyze the task, break it down into "
        "a clear implementation plan, identify key files and patterns to "
        "follow, and flag any risks or decisions needed. Be specific about "
        "what needs to be built and how."
    ),
    SquadRole.DEVELOPER: (
        "You are the Developer. Implement the task according to the plan. "
        "Write clean, well-structured code following the project's existing "
        "patterns and conventions. Make the smallest changes necessary."
    ),
    SquadRole.TESTER: (
        "You are the Tester. Write comprehensive tests for the task. "
        "Cover happy paths, edge cases, and error handling. Follow the "
        "project's existing test patterns and conventions."
    ),
    SquadRole.DOCS: (
        "You are the Documentation Expert. Review and update documentation, "
        "README files, docstrings, and examples. Ensure documentation stays "
        "in sync with code changes."
    ),
    SquadRole.DEVOPS: (
        "You are the DevOps Engineer. Manage CI/CD, Docker, infrastructure, "
        "and deployment configurations."
    ),
    SquadRole.FRONTEND: (
        "You are the Frontend Developer. Build UI components, handle "
        "client-side logic, and ensure good UX."
    ),
    SquadRole.BACKEND: (
        "You are the Backend Developer. Build APIs, services, database "
        "interactions, and server-side logic."
    ),
}

_ROLE_EMOJIS: dict[SquadRole, str] = {
    SquadRole.LEAD: "🏗️",
    SquadRole.DEVELOPER: "🔧",
    SquadRole.TESTER: "🧪",
    SquadRole.DOCS: "📝",
    SquadRole.DEVOPS: "⚙️",
    SquadRole.FRONTEND: "⚛️",
    SquadRole.BACKEND: "🔧",
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


@dataclass
class SquadAgent:
    """An agent in the squad team."""

    name: str
    role: SquadRole
    emoji: str
    system_prompt: str

    @classmethod
    def default_for_role(cls, role: SquadRole) -> SquadAgent:
        """Create a default agent for a role."""
        return cls(
            name=role.value.title(),
            role=role,
            emoji=_ROLE_EMOJIS[role],
            system_prompt=_ROLE_PROMPTS[role],
        )


@dataclass
class SquadTeam:
    """A team of squad agents."""

    agents: list[SquadAgent] = field(default_factory=list)

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
    async def from_repo_ai(
        cls,
        config: CopexConfig | None = None,
        path: Path | None = None,
    ) -> SquadTeam:
        """Create a team by using AI to analyze the repository.

        Uses CopilotCLI with claude-opus-4.6-fast to intelligently analyze
        the repo structure, README, and config files. Falls back to from_repo()
        (pattern matching) on any failure.

        Args:
            config: Optional CopexConfig (creates lightweight config if None).
            path: Optional path to repository (defaults to cwd).

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

            prompt_parts.extend([
                "",
                "# Instructions",
                "",
                "Respond ONLY with a JSON array of role strings. Valid roles:",
                "- lead (always required)",
                "- developer (always required)",
                "- tester (if has tests or should have tests)",
                "- docs (if has documentation to maintain)",
                "- devops (if has CI/CD, Docker, or infrastructure)",
                "- frontend (if has UI components)",
                "- backend (if has API/server components)",
                "",
                "Rules:",
                "- Always include 'lead' and 'developer'",
                "- Only include roles that are clearly needed",
                "- Keep the team small (3-5 roles typical)",
                "",
                'Example response: ["lead", "developer", "tester", "docs"]',
            ])

            prompt = "\n".join(prompt_parts)

            # Create lightweight config for AI analysis
            ai_config = config or CopexConfig()
            ai_config.model = Model.CLAUDE_OPUS_4_6_FAST
            ai_config.reasoning_effort = ReasoningEffort.LOW
            ai_config.streaming = False
            ai_config.use_cli = True

            # Call AI with timeout
            async with CopilotCLI(ai_config) as cli:
                response = await cli.send(prompt)
                content = response.content.strip()

                # Parse JSON (handle markdown code fences)
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                # Extract JSON array
                roles_list = json.loads(content)
                if not isinstance(roles_list, list):
                    raise ValueError("AI response is not a list")

                # Validate and convert to SquadRole
                roles: list[SquadRole] = []
                for role_str in roles_list:
                    try:
                        role = SquadRole(role_str.lower())
                        if role not in roles:
                            roles.append(role)
                    except ValueError:
                        logger.warning(f"Unknown role from AI: {role_str}")

                # Ensure minimum viable team
                if SquadRole.LEAD not in roles:
                    roles.insert(0, SquadRole.LEAD)
                if SquadRole.DEVELOPER not in roles:
                    roles.append(SquadRole.DEVELOPER)

                logger.info(f"AI analysis completed: {[r.value for r in roles]}")
                return cls(agents=[SquadAgent.default_for_role(r) for r in roles])

        except Exception as e:
            logger.warning(f"AI repo analysis failed ({e}), falling back to pattern matching")
            return cls.from_repo(path)

    def get_agent(self, role: SquadRole) -> SquadAgent | None:
        """Get agent by role."""
        for a in self.agents:
            if a.role == role:
                return a
        return None

    @property
    def roles(self) -> list[SquadRole]:
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
                    "role": ar.agent.role.value,
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
            timeout=600.0,
        )
        self._project_context: str | None = None

    @property
    def team(self) -> SquadTeam:
        """The squad team."""
        if self._team is None:
            # Sync fallback if team requested before async init
            self._team = SquadTeam.from_repo()
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

        # Lazy init: analyze repo with AI if team not provided
        if self._team is None:
            self._team = await SquadTeam.from_repo_ai(self._config)

        if self._project_context is None:
            self._project_context = self._discover_project_context()

        fleet = Fleet(self._config, fleet_config=self._fleet_config)
        task_ids = self._add_tasks(fleet, prompt)

        fleet_results = await fleet.run(on_status=on_status)

        result = self._build_result(fleet_results, task_ids)
        result.total_duration_ms = (time.monotonic() - start_time) * 1000
        return result

    def _add_tasks(self, fleet: Fleet, prompt: str) -> dict[SquadRole, str]:
        """Add fleet tasks for each agent."""
        task_ids: dict[SquadRole, str] = {}

        for agent in self.team.agents:  # Use property for lazy init
            deps = self._get_dependencies(agent.role, task_ids)
            agent_prompt = self._build_agent_prompt(agent, prompt, task_ids)

            tid = fleet.add(
                agent_prompt,
                task_id=agent.role.value,
                depends_on=deps,
                model=self._config.model,
            )
            task_ids[agent.role] = tid

        return task_ids

    def _get_dependencies(
        self, role: SquadRole, task_ids: dict[SquadRole, str]
    ) -> list[str]:
        """Get dependency task IDs for a role."""
        if role == SquadRole.LEAD:
            return []
        if role == SquadRole.DOCS:
            # Docs runs last — depends on all other non-lead roles
            deps = []
            for r, tid in task_ids.items():
                if r != SquadRole.LEAD and r != SquadRole.DOCS:
                    deps.append(tid)
            return deps
        if role == SquadRole.TESTER:
            # Tester depends on implementation agents
            deps = []
            for dep_role in (
                SquadRole.DEVELOPER, SquadRole.FRONTEND, SquadRole.BACKEND,
            ):
                dep_id = task_ids.get(dep_role)
                if dep_id:
                    deps.append(dep_id)
            if not deps:
                lead_id = task_ids.get(SquadRole.LEAD)
                if lead_id:
                    deps.append(lead_id)
            return deps
        # DEVELOPER, FRONTEND, BACKEND, DEVOPS all depend on Lead
        lead_id = task_ids.get(SquadRole.LEAD)
        return [lead_id] if lead_id else []

    def _build_agent_prompt(
        self,
        agent: SquadAgent,
        user_prompt: str,
        task_ids: dict[SquadRole, str],
    ) -> str:
        """Build the full prompt for an agent including role context."""
        parts = [agent.system_prompt]

        # Inject project context if available
        if self._project_context:
            parts.append("")
            parts.append(self._project_context)

        parts.append("")
        parts.append(f"## Task\n\n{user_prompt}")

        # Add reference to Lead's output for dependent agents
        lead_id = task_ids.get(SquadRole.LEAD)
        if lead_id and agent.role != SquadRole.LEAD:
            parts.append(
                f"\n## Lead's Analysis\n\n"
                f"{{{{task:{lead_id}.content}}}}"
            )

        return "\n".join(parts)

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
        task_ids: dict[SquadRole, str],
    ) -> SquadResult:
        """Convert fleet results to SquadResult."""
        id_to_role = {tid: role for role, tid in task_ids.items()}

        agent_results = []
        all_success = True

        for fr in fleet_results:
            role = id_to_role.get(fr.task_id)
            if role is None:
                continue
            agent = self.team.get_agent(role)  # Use property for lazy init
            if agent is None:
                continue

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
            agent_results.append(ar)

            if not fr.success:
                all_success = False

        return SquadResult(
            agent_results=agent_results,
            success=all_success,
        )
