"""Fleet and council CLI commands."""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from copex.cli_utils import apply_approval_flags as _apply_approval_flags
from copex.config import CopexConfig
from copex.fleet import FleetTask
from copex.models import (
    Model,
    ReasoningEffort,
    normalize_reasoning_effort,
    parse_reasoning_effort,
)

if TYPE_CHECKING:
    pass

console = Console()
_DEFAULT_MODEL = Model.CLAUDE_OPUS_4_5
_resolve_cli_model: Callable[[str], Model | str] | None = None
_parse_exclude_tools: Callable[[str | None], list[str]] | None = None
_format_duration: Callable[[float], str] | None = None
_apply_approval_flags: Callable[..., None] | None = None


def configure_fleet_cli(
    *,
    shared_console: Console,
    default_model: Model,
    resolve_cli_model: Callable[[str], Model | str],
    parse_exclude_tools: Callable[[str | None], list[str]],
    format_duration: Callable[[float], str],
    apply_approval_flags: Callable[..., None],
) -> None:
    """Configure shared dependencies used by fleet/council CLI handlers.

    Args:
        shared_console: Console instance used for command output.
        default_model: Default model used when no CLI model override is provided.
        resolve_cli_model: Resolver for CLI model strings.
        parse_exclude_tools: Parser for ``--exclude-tools`` option values.
        format_duration: Formatter for elapsed durations displayed in output.
        apply_approval_flags: Callback to apply approval-mode flags to config.

    Returns:
        None: Module-level configuration is updated in place.
    """
    global console, _DEFAULT_MODEL, _resolve_cli_model, _parse_exclude_tools, _format_duration
    global _apply_approval_flags
    console = shared_console
    _DEFAULT_MODEL = default_model
    _resolve_cli_model = resolve_cli_model
    _parse_exclude_tools = parse_exclude_tools
    _format_duration = format_duration
    _apply_approval_flags = apply_approval_flags


def register_fleet_commands(app: typer.Typer) -> None:
    """Register fleet and council commands on the provided Typer app.

    Args:
        app: Typer application to register commands on.

    Returns:
        None: Commands are attached directly to ``app``.
    """
    app.command("fleet")(fleet_command)
    app.command("council")(council_command)


class FleetTaskSpec(BaseModel):
    """Shared task schema for fleet JSON and TOML configs."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    prompt: str
    depends_on: list[str] = Field(default_factory=list)
    model: Model | None = None
    reasoning_effort: ReasoningEffort | None = None
    cwd: str | None = None
    on_dependency_failure: str = "block"
    exclude_tools: list[str] = Field(default_factory=list)
    skills: list[str] | None = None
    skills_dirs: list[str] = Field(default_factory=list)
    mcp_servers: dict[str, Any] | list[dict[str, Any]] | None = None
    timeout_sec: float | None = None

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError("must be a non-empty string")
        return value.strip()

    @field_validator("prompt")
    @classmethod
    def _validate_prompt(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be a non-empty string")
        return value

    @field_validator("depends_on")
    @classmethod
    def _validate_depends_on(cls, value: list[str]) -> list[str]:
        if any(not isinstance(dep, str) or not dep.strip() for dep in value):
            raise ValueError("must be a list of non-empty strings")
        return value

    @field_validator("reasoning_effort", mode="before")
    @classmethod
    def _parse_reasoning_effort(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, ReasoningEffort):
            return value
        if not isinstance(value, str):
            return value
        parsed = parse_reasoning_effort(value)
        if parsed is None:
            valid = ", ".join(r.value for r in ReasoningEffort)
            raise ValueError(f"invalid value '{value}'. Valid: {valid}")
        return parsed

    @field_validator("on_dependency_failure", mode="before")
    @classmethod
    def _validate_on_dependency_failure(cls, value: Any) -> str:
        if value is None:
            return "block"
        if not isinstance(value, str):
            raise ValueError("must be a string ('block' or 'continue')")
        normalized = value.strip().lower()
        if normalized not in {"block", "continue"}:
            raise ValueError("must be 'block' or 'continue'")
        return normalized

    @field_validator("exclude_tools", mode="before")
    @classmethod
    def _parse_exclude_tools(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("exclude_tools")
    @classmethod
    def _validate_exclude_tools(cls, value: list[str]) -> list[str]:
        if any(not isinstance(tool, str) or not tool.strip() for tool in value):
            raise ValueError("must be a list of non-empty strings")
        return [tool.strip() for tool in value if tool.strip()]

    @field_validator("skills")
    @classmethod
    def _validate_skills(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if any(not isinstance(skill, str) or not skill.strip() for skill in value):
            raise ValueError("must be a list of non-empty strings")
        return value

    @field_validator("skills_dirs", mode="before")
    @classmethod
    def _parse_skills_dirs(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("skills_dirs")
    @classmethod
    def _validate_skills_dirs(cls, value: list[str]) -> list[str]:
        if any(not isinstance(path, str) or not path.strip() for path in value):
            raise ValueError("must be a list of non-empty strings")
        return [path.strip() for path in value if path.strip()]

    @field_validator("mcp_servers")
    @classmethod
    def _validate_mcp_servers(
        cls,
        value: dict[str, Any] | list[dict[str, Any]] | None,
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        if value is None:
            return None
        if isinstance(value, list):
            if any(not isinstance(server, dict) for server in value):
                raise ValueError("must be a list of objects")
            return value
        if not isinstance(value, dict):
            raise ValueError("must be an object or list of objects")
        return value

    @field_validator("timeout_sec")
    @classmethod
    def _validate_timeout(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("must be greater than zero")
        return float(value)


def _format_fleet_spec_error(exc: ValidationError, idx: int) -> str:
    """Convert pydantic errors into existing CLI-style task errors."""
    err = exc.errors(include_url=False)[0]
    loc = ".".join(str(part) for part in err.get("loc", ()))
    loc_suffix = f".{loc}" if loc else ""
    return f"tasks[{idx}]{loc_suffix} {err.get('msg', 'is invalid')}"


def _parse_fleet_task_specs(raw_tasks: Any, *, key_name: str = "tasks") -> list[FleetTask]:
    """Parse/validate task rows from JSON or TOML using one shared schema."""
    from copex.fleet import DependencyFailurePolicy, FleetTask

    if raw_tasks is None:
        raise ValueError(f"Fleet config missing required '{key_name}' list")
    if not isinstance(raw_tasks, list):
        raise ValueError(f"Fleet config '{key_name}' must be a list")
    if not raw_tasks:
        raise ValueError(f"Fleet config '{key_name}' list is empty")

    tasks: list[FleetTask] = []
    seen_ids: set[str] = set()
    for idx, task_data in enumerate(raw_tasks):
        if not isinstance(task_data, dict):
            raise ValueError(f"tasks[{idx}] must be an object")

        try:
            spec = FleetTaskSpec.model_validate(task_data)
        except ValidationError as exc:
            raise ValueError(_format_fleet_spec_error(exc, idx)) from exc

        task_id = spec.id or f"task-{idx + 1}"
        if task_id in seen_ids:
            raise ValueError(f"Duplicate task id in fleet config: '{task_id}'")
        seen_ids.add(task_id)

        tasks.append(
            FleetTask(
                id=task_id,
                prompt=spec.prompt,
                depends_on=spec.depends_on,
                model=spec.model,
                reasoning_effort=spec.reasoning_effort,
                cwd=spec.cwd,
                skills=spec.skills,
                exclude_tools=spec.exclude_tools,
                mcp_servers=spec.mcp_servers,
                timeout_sec=spec.timeout_sec,
                skills_dirs=spec.skills_dirs,
                on_dependency_failure=DependencyFailurePolicy(spec.on_dependency_failure),
            )
        )

    return tasks


def fleet_command(
    prompts: Annotated[
        list[str] | None, typer.Argument(help="Task prompts to run in parallel")
    ] = None,
    file: Annotated[
        Path | None, typer.Option("--file", "-f", help="TOML or JSONL file with task definitions")
    ] = None,
    config_file: Annotated[
        Path | None, typer.Option("--config", help="JSON config file with task definitions")
    ] = None,
    max_concurrent: Annotated[
        int, typer.Option("--max-concurrent", help="Max concurrent tasks")
    ] = 5,
    parallel: Annotated[
        int | None, typer.Option("--parallel", help="Max parallel tasks (alias for --max-concurrent)")
    ] = None,
    fail_fast: Annotated[
        bool, typer.Option("--fail-fast", help="Stop all tasks on first failure")
    ] = False,
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model to use")] = None,
    context_budget: Annotated[
        int | None,
        typer.Option("--context-budget", help="Override context window budget in tokens"),
    ] = None,
    reasoning: Annotated[
        ReasoningEffort, typer.Option("--reasoning", "-r", help="Reasoning effort level")
    ] = ReasoningEffort.HIGH,
    mcp_config: Annotated[
        Path | None, typer.Option("--mcp-config", help="Path to MCP config JSON file")
    ] = None,
    shared_context: Annotated[
        str | None, typer.Option("--shared-context", help="Context prepended to all tasks")
    ] = None,
    timeout: Annotated[
        float, typer.Option("--timeout", help="Per-task timeout in seconds")
    ] = 600.0,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Print full task outputs after completion")
    ] = False,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Directory to save each task result as a file"),
    ] = None,
    artifact: Annotated[
        Path | None,
        typer.Option("--artifact", help="Write run artifact JSON"),
    ] = None,
    git_finalize: Annotated[
        bool,
        typer.Option(
            "--git-finalize/--no-git-finalize",
            help="Stage and commit all changes after fleet completes (auto-detected, enabled by default)",
        ),
    ] = True,
    git_message: Annotated[
        str | None,
        typer.Option("--git-message", help="Commit message for git finalize"),
    ] = None,
    skills: Annotated[
        list[str] | None,
        typer.Option("--skills", help="Skill directories (.md files) to prepend to system prompt"),
    ] = None,
    exclude_tools: Annotated[
        str | None,
        typer.Option(
            "--exclude-tools",
            help="Comma-separated tool patterns to exclude per task (e.g. shell(rm))",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            is_flag=True,
            help="Preview fleet actions without executing",
        ),
    ] = False,
    approval_dry_run: Annotated[
        bool,
        typer.Option(
            "--approval-dry-run",
            is_flag=True,
            help="Show file change diffs without applying them during execution",
        ),
    ] = False,
    auto_approve: Annotated[
        bool, typer.Option("--auto-approve", help="Apply file changes without prompts")
    ] = False,
    approve: Annotated[
        bool, typer.Option("--approve", help="Prompt before each file change")
    ] = False,
    audit: Annotated[
        bool, typer.Option("--audit", help="Log file change decisions to .copex/audit.log")
    ] = False,
    commit_msg: Annotated[
        str | None,
        typer.Option(
            "--commit-msg",
            "-cm",
            help="Git commit message (used after successful build; implies --git-finalize)",
        ),
    ] = None,
    retry: Annotated[
        int,
        typer.Option("--retry", "-R", help="Retry N times on build failure after fleet completes"),
    ] = 0,
    progress: Annotated[
        bool,
        typer.Option("--progress", "-P", help="Show real-time file change progress"),
    ] = False,
    worktree: Annotated[
        bool,
        typer.Option("--worktree", "-w", is_flag=True, help="Run each task in an isolated git worktree"),
    ] = False,
) -> None:
    """
        Run multiple tasks in parallel with fleet execution.

        Examples:
            copex fleet "Write tests" "Fix linting" "Update docs"
            copex fleet --file tasks.toml
            copex fleet -f tasks.jsonl --parallel 3
            copex fleet --config tasks.json
            copex fleet "Task A" "Task B" --max-concurrent 3 --fail-fast

    Args:
        prompts: CLI argument or option value.
        file: CLI argument or option value.
        config_file: CLI argument or option value.
        max_concurrent: CLI argument or option value.
        parallel: CLI argument or option value.
        fail_fast: CLI argument or option value.
        model: CLI argument or option value.
        context_budget: CLI argument or option value.
        reasoning: CLI argument or option value.
        mcp_config: CLI argument or option value.
        shared_context: CLI argument or option value.
        timeout: CLI argument or option value.
        verbose: CLI argument or option value.
        output_dir: CLI argument or option value.
        artifact: CLI argument or option value.
        git_finalize: CLI argument or option value.
        git_message: CLI argument or option value.
        skills: CLI argument or option value.
        exclude_tools: CLI argument or option value.
        dry_run: CLI argument or option value.
        approval_dry_run: CLI argument or option value.
        auto_approve: CLI argument or option value.
        approve: CLI argument or option value.
        audit: CLI argument or option value.
        commit_msg: CLI argument or option value.
        retry: CLI argument or option value.
        progress: CLI argument or option value.
        worktree: CLI argument or option value.

    Returns:
        None: Command result.
    """
    if config_file and (prompts or file):
        console.print("[red]Error: Use --config without prompts or --file[/red]")
        raise typer.Exit(1) from None
    if not prompts and not file and not config_file:
        console.print("[red]Error: Provide task prompts, --file, or --config[/red]")
        raise typer.Exit(1) from None

    # --parallel is an alias for --max-concurrent
    if parallel is not None:
        max_concurrent = parallel

    effective_model = model or _DEFAULT_MODEL.value
    try:
        if _resolve_cli_model is None or _apply_approval_flags is None:
            raise RuntimeError("Fleet CLI not configured")
        model_enum = _resolve_cli_model(effective_model)
        requested_effort = parse_reasoning_effort(reasoning) or ReasoningEffort.HIGH
        normalized_effort, warning = normalize_reasoning_effort(model_enum, requested_effort)
        if warning:
            console.print(f"[yellow]{warning}[/yellow]")

        config = CopexConfig(model=model_enum, reasoning_effort=normalized_effort)
        if context_budget is not None:
            config.context_budget = context_budget
        if mcp_config:
            if not mcp_config.exists():
                console.print(f"[red]MCP config file not found: {mcp_config}[/red]")
                raise typer.Exit(1) from None
            config.mcp_config_file = str(mcp_config)
        _apply_approval_flags(
            config,
            auto_approve=auto_approve,
            approve=approve,
            dry_run=approval_dry_run,
            audit=audit,
            default_auto=True,
        )
    except typer.BadParameter as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None

    # --commit-msg implies git finalize with the given message
    if commit_msg:
        git_finalize = True
        git_message = commit_msg

    # --dry-run: preview what fleet would do without executing
    if dry_run:
        from rich.table import Table

        from copex.fleet import FleetTask as _FT

        # Build task list for preview (same logic as _run_fleet)
        preview_tasks: list[_FT] = []
        if config_file:
            try:
                preview_tasks = _parse_fleet_task_specs(
                    json.loads(config_file.read_text()), key_name="tasks"
                )
            except Exception:
                try:
                    preview_tasks = _load_fleet_json_config(config_file)
                except Exception as exc:
                    console.print(f"[red]Error loading config: {exc}[/red]")
                    raise typer.Exit(1) from None
        elif file:
            if file.suffix == ".jsonl":
                try:
                    from copex.multi_fleet import load_jsonl_tasks

                    preview_tasks = load_jsonl_tasks(file)
                except ValueError as exc:
                    console.print(f"[red]Error loading JSONL: {exc}[/red]")
                    raise typer.Exit(1) from None
            else:
                try:
                    _, preview_tasks = _load_fleet_toml_config(file)
                except ValueError as exc:
                    console.print(f"[red]Error loading TOML: {exc}[/red]")
                    raise typer.Exit(1) from None
        for i, prompt in enumerate(prompts or []):
            preview_tasks.append(_FT(id=f"task-{i + 1}", prompt=prompt))

        console.print(
            Panel(
                f"[bold]Model:[/bold] {model_enum.value}\n"
                f"[bold]Reasoning effort:[/bold] {normalized_effort.value}\n"
                f"[bold]Max concurrent:[/bold] {max_concurrent}\n"
                f"[bold]Fail fast:[/bold] {fail_fast}\n"
                f"[bold]Timeout:[/bold] {timeout:.0f}s\n"
                f"[bold]Git finalize:[/bold] {git_finalize}"
                + (f"  →  [cyan]{git_message}[/cyan]" if git_message else ""),
                title="🔍 Dry Run — Fleet Configuration",
                border_style="yellow",
            )
        )

        tbl = Table(title="Tasks", expand=True)
        tbl.add_column("ID", style="cyan", no_wrap=True)
        tbl.add_column("Prompt", max_width=70)
        tbl.add_column("Model Override", style="dim")
        tbl.add_column("Depends On", style="dim")
        for t in preview_tasks:
            tbl.add_row(
                t.id,
                t.prompt[:70],
                t.model or "—",
                ", ".join(t.depends_on) if t.depends_on else "—",
            )
        console.print(tbl)
        console.print("\n[yellow bold]DRY RUN: No changes made[/yellow bold]")
        raise typer.Exit(0)

    asyncio.run(
        _run_fleet(
            config=config,
            prompts=prompts or [],
            file=file,
            config_file=config_file,
            max_concurrent=max_concurrent,
            fail_fast=fail_fast,
            shared_context=shared_context,
            timeout=timeout,
            verbose=verbose,
            output_dir=output_dir,
            artifact_path=artifact,
            git_finalize=git_finalize,
            git_message=git_message,
            skills=skills or [],
            exclude_tools=(_parse_exclude_tools(exclude_tools) if _parse_exclude_tools is not None else []),
            retry=retry,
            progress=progress,
            worktree=worktree,
        )
    )


def _build_council_tasks(
    task: str,
    *,
    investigator_model: Model | None = None,
    codex_model: Model | None = None,
    gemini_model: Model | None = None,
    opus_model: Model | None = None,
    chair_model: Model | None = None,
    reasoning_effort: ReasoningEffort = ReasoningEffort.HIGH,
    debate: bool = False,
    preset: str | None = None,
    escalate: bool = False,
) -> list[FleetTask]:
    """Create the council workflow task graph with enhancements.

    Args:
        task: The task/problem for the council
        investigator_model: Override model for all 3 investigators
        codex_model: Override model for Codex investigator
        gemini_model: Override model for Gemini investigator
        opus_model: Override model for Opus investigator
        chair_model: Model for chair (default: claude-opus-4.6)
        reasoning_effort: Reasoning effort for all tasks
        debate: Enable debate rounds (investigators revise after seeing others)
        preset: Specialist preset (security/architecture/refactor/review)
        escalate: Enable tie-breaker escalation on uncertainty
    """
    from copex.council import CouncilConfig, CouncilPreset, build_council_tasks

    # Parse preset
    council_preset = None
    if preset:
        try:
            council_preset = CouncilPreset(preset.lower())
        except ValueError:
            valid = ", ".join(p.value for p in CouncilPreset)
            console.print(f"[yellow]Warning: Invalid preset '{preset}'. Valid: {valid}[/yellow]")

    config = CouncilConfig(
        investigator_model=investigator_model,
        codex_model=codex_model,
        gemini_model=gemini_model,
        opus_model=opus_model,
        chair_model=chair_model or Model.CLAUDE_OPUS_4_6,
        reasoning_effort=reasoning_effort,
        debate=debate,
        preset=council_preset,
        escalate=escalate,
    )

    return build_council_tasks(task, config)


def council_command(
    task: Annotated[str, typer.Argument(help="Task/problem for the model council")],
    max_concurrent: Annotated[
        int, typer.Option("--max-concurrent", help="Max concurrent tasks")
    ] = 3,
    timeout: Annotated[
        float, typer.Option("--timeout", help="Per-task timeout in seconds")
    ] = 900.0,
    shared_context: Annotated[
        str | None, typer.Option("--shared-context", help="Context prepended to all tasks")
    ] = None,
    mcp_config: Annotated[
        Path | None, typer.Option("--mcp-config", help="Path to MCP config JSON file")
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Print full task outputs after completion")
    ] = False,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Directory to save each task result as a file"),
    ] = None,
    artifact: Annotated[
        Path | None,
        typer.Option("--artifact", help="Write run artifact JSON"),
    ] = None,
    git_finalize: Annotated[
        bool,
        typer.Option(
            "--git-finalize/--no-git-finalize",
            help="Stage and commit all changes after council completes",
        ),
    ] = True,
    git_message: Annotated[
        str | None,
        typer.Option("--git-message", help="Commit message for git finalize"),
    ] = None,
    skills: Annotated[
        list[str] | None,
        typer.Option("--skills", help="Skill directories (.md files) to prepend to system prompt"),
    ] = None,
    exclude_tools: Annotated[
        str | None,
        typer.Option(
            "--exclude-tools",
            help="Comma-separated tool patterns to exclude per task (e.g. shell(rm))",
        ),
    ] = None,
    # New council enhancement options
    investigator_model: Annotated[
        str | None,
        typer.Option(
            "--investigator-model",
            help="Model for all 3 investigators (overrides defaults)",
        ),
    ] = None,
    codex_model: Annotated[
        str | None,
        typer.Option("--codex-model", help="Model for Codex investigator"),
    ] = None,
    gemini_model: Annotated[
        str | None,
        typer.Option("--gemini-model", help="Model for Gemini investigator"),
    ] = None,
    opus_model: Annotated[
        str | None,
        typer.Option("--opus-model", help="Model for Opus investigator (not chair)"),
    ] = None,
    chair_model: Annotated[
        str | None,
        typer.Option("--chair-model", help="Model for chair (default: claude-opus-4.6)"),
    ] = None,
    reasoning: Annotated[
        ReasoningEffort | None,
        typer.Option(
            "-r", "--reasoning", help="Reasoning effort for all tasks (none/low/medium/high/xhigh)"
        ),
    ] = None,
    debate: Annotated[
        bool,
        typer.Option(
            "--debate/--no-debate",
            help="Enable debate round (investigators revise after seeing others)",
        ),
    ] = False,
    preset: Annotated[
        str | None,
        typer.Option(
            "--preset",
            help="Specialist preset: security, architecture, refactor, review",
        ),
    ] = None,
    escalate: Annotated[
        bool,
        typer.Option(
            "--escalate/--no-escalate", help="Enable tie-breaker escalation on uncertainty"
        ),
    ] = False,
) -> None:
    """Run a council workflow: Codex + Gemini + Opus, with Opus as chair.

        Enhanced council features:

        
        MODEL SELECTION:
          --investigator-model  Override model for all 3 investigators
          --codex-model         Override just Codex
          --gemini-model        Override just Gemini
          --opus-model          Override just Opus investigator
          --chair-model         Override chair model (default: claude-opus-4.6)

        
        DEBATE ROUNDS:
          --debate              After initial opinions, each investigator sees others'
                                responses and can revise their position

        
        SPECIALIST PRESETS:
          --preset security      Focus on vulnerabilities, auth, injection
          --preset architecture  Focus on patterns, scaling, modularity
          --preset refactor      Focus on code quality, DRY, naming
          --preset review        Balanced code review

        
        TIE-BREAKER:
          --escalate            If chair is uncertain (confidence < 0.7), re-run
                                with xhigh reasoning for a definitive answer

    Args:
        task: CLI argument or option value.
        max_concurrent: CLI argument or option value.
        timeout: CLI argument or option value.
        shared_context: CLI argument or option value.
        mcp_config: CLI argument or option value.
        verbose: CLI argument or option value.
        output_dir: CLI argument or option value.
        artifact: CLI argument or option value.
        git_finalize: CLI argument or option value.
        git_message: CLI argument or option value.
        skills: CLI argument or option value.
        exclude_tools: CLI argument or option value.
        investigator_model: CLI argument or option value.
        codex_model: CLI argument or option value.
        gemini_model: CLI argument or option value.
        opus_model: CLI argument or option value.
        chair_model: CLI argument or option value.
        reasoning: CLI argument or option value.
        debate: CLI argument or option value.
        preset: CLI argument or option value.
        escalate: CLI argument or option value.

    Returns:
        None: Command result.
    """
    if _resolve_cli_model is None:
        console.print("[red]Error: Fleet CLI not configured[/red]")
        raise typer.Exit(1) from None

    # Parse model options
    def parse_model(m: str | None) -> Model | str | None:
        if m is None:
            return None
        try:
            return _resolve_cli_model(m)
        except ValueError:
            console.print(f"[red]Invalid model: {m}[/red]")
            raise typer.Exit(1) from None

    inv_model = parse_model(investigator_model)
    cdx_model = parse_model(codex_model)
    gem_model = parse_model(gemini_model)
    op_model = parse_model(opus_model)
    chr_model = parse_model(chair_model)

    # Parse reasoning effort
    effort = ReasoningEffort.HIGH
    if reasoning:
        try:
            effort = parse_reasoning_effort(reasoning) or ReasoningEffort.HIGH
        except ValueError:
            console.print(f"[red]Invalid reasoning effort: {reasoning}[/red]")
            raise typer.Exit(1) from None

    config = CopexConfig(
        model=chr_model or Model.CLAUDE_OPUS_4_6,
        reasoning_effort=effort,
    )
    if mcp_config:
        if not mcp_config.exists():
            console.print(f"[red]MCP config file not found: {mcp_config}[/red]")
            raise typer.Exit(1) from None
        config.mcp_config_file = str(mcp_config)

    council_tasks = _build_council_tasks(
        task,
        investigator_model=inv_model,
        codex_model=cdx_model,
        gemini_model=gem_model,
        opus_model=op_model,
        chair_model=chr_model,
        reasoning_effort=effort,
        debate=debate,
        preset=preset,
        escalate=escalate,
    )
    asyncio.run(
        _run_fleet(
            config=config,
            prompts=[],
            file=None,
            config_file=None,
            max_concurrent=max_concurrent,
            fail_fast=False,
            shared_context=shared_context,
            timeout=timeout,
            verbose=verbose,
            output_dir=output_dir,
            artifact_path=artifact,
            git_finalize=git_finalize,
            git_message=git_message,
            skills=skills or [],
            exclude_tools=(_parse_exclude_tools(exclude_tools) if _parse_exclude_tools is not None else []),
            tasks_override=council_tasks,
        )
    )


def _load_fleet_json_config(path: Path) -> list[FleetTask]:
    """Load and validate JSON fleet config."""
    if not path.exists():
        raise ValueError(f"Config file not found: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {path}: {exc.msg} (line {exc.lineno} column {exc.colno})"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError("Fleet config must be a JSON object with a top-level 'tasks' list")

    raw_tasks = data.get("tasks")
    return _parse_fleet_task_specs(raw_tasks, key_name="tasks")


def _load_fleet_toml_config(path: Path) -> tuple[dict[str, Any], list[FleetTask]]:
    """Load TOML fleet config and return (fleet_section, tasks)."""
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        import tomli as tomllib  # type: ignore

    if not path.exists():
        raise ValueError(f"File not found: {path}")

    with open(path, "rb") as f:
        data = tomllib.load(f)

    if not isinstance(data, dict):
        raise ValueError("Fleet TOML must contain a top-level table/object")

    fleet_section = data.get("fleet", {})
    if not isinstance(fleet_section, dict):
        raise ValueError("fleet section must be a table/object")

    raw_tasks = data.get("task")
    if raw_tasks is None:
        return fleet_section, []
    if isinstance(raw_tasks, list) and not raw_tasks:
        return fleet_section, []

    tasks = _parse_fleet_task_specs(raw_tasks, key_name="task")
    return fleet_section, tasks


async def _run_fleet(
    config: CopexConfig,
    prompts: list[str],
    file: Path | None,
    config_file: Path | None,
    max_concurrent: int,
    fail_fast: bool,
    shared_context: str | None,
    timeout: float,
    verbose: bool = False,
    output_dir: Path | None = None,
    artifact_path: Path | None = None,
    git_finalize: bool = True,
    git_message: str | None = None,
    skills: list[str] | None = None,
    exclude_tools: list[str] | None = None,
    tasks_override: list[FleetTask] | None = None,
    retry: int = 0,
    progress: bool = False,
    worktree: bool = False,
) -> None:
    """Run fleet tasks with live progress display."""
    import threading

    from rich.table import Table

    from copex.fleet import Fleet, FleetConfig, FleetTask

    fleet_config = FleetConfig(
        max_concurrent=max_concurrent,
        timeout=timeout,
        fail_fast=fail_fast,
        shared_context=shared_context,
        git_auto_finalize=git_finalize,
    )

    tasks: list[FleetTask] = []
    fleet_skills = skills or []
    fleet_excluded = exclude_tools or []

    def _merge_skills(base: list[str], extra: list[str]) -> list[str]:
        merged: list[str] = []
        for path in [*base, *extra]:
            if path not in merged:
                merged.append(path)
        return merged

    def _merge_excludes(base: list[str], extra: list[str]) -> list[str]:
        merged: list[str] = []
        for tool in [*base, *extra]:
            if tool not in merged:
                merged.append(tool)
        return merged

    # Direct task graph (used by preset modes like council)
    if tasks_override is not None:
        tasks = list(tasks_override)
        for task in tasks:
            task.skills_dirs = _merge_skills(fleet_skills, task.skills_dirs)
            task.exclude_tools = _merge_excludes(
                fleet_excluded,
                task.exclude_tools or [],
            )

    # Load tasks from JSON config file
    elif config_file:
        try:
            tasks = _load_fleet_json_config(config_file)
        except ValueError as exc:
            console.print(f"[red]Error: {exc}[/red]")
            raise typer.Exit(1) from None
        for task in tasks:
            task.skills_dirs = _merge_skills(fleet_skills, task.skills_dirs)
            task.exclude_tools = _merge_excludes(
                fleet_excluded,
                task.exclude_tools or [],
            )

    # Load tasks from TOML or JSONL file
    elif file:
        if file.suffix == ".jsonl":
            # JSONL multi-task file
            try:
                from copex.multi_fleet import load_jsonl_tasks

                parsed_tasks = load_jsonl_tasks(file)
            except ValueError as exc:
                console.print(f"[red]Error: {exc}[/red]")
                raise typer.Exit(1) from None

            for task in parsed_tasks:
                task.skills_dirs = _merge_skills(fleet_skills, task.skills_dirs)
                task.exclude_tools = _merge_excludes(
                    fleet_excluded,
                    task.exclude_tools or [],
                )
                tasks.append(task)
        else:
            # TOML file
            try:
                fleet_section, parsed_tasks = _load_fleet_toml_config(file)
            except ValueError as exc:
                console.print(f"[red]Error: {exc}[/red]")
                raise typer.Exit(1) from None

            # Apply fleet-level config from file
            if "max_concurrent" in fleet_section:
                fleet_config.max_concurrent = fleet_section["max_concurrent"]
            if "shared_context" in fleet_section:
                fleet_config.shared_context = fleet_section["shared_context"]
            if "timeout" in fleet_section:
                fleet_config.timeout = fleet_section["timeout"]
            if "fail_fast" in fleet_section:
                fleet_config.fail_fast = fleet_section["fail_fast"]

            for task in parsed_tasks:
                task.skills_dirs = _merge_skills(fleet_skills, task.skills_dirs)
                task.exclude_tools = _merge_excludes(
                    fleet_excluded,
                    task.exclude_tools or [],
                )
                tasks.append(task)

    # Add CLI prompt tasks
    if tasks_override is None and not config_file:
        for i, prompt in enumerate(prompts):
            tasks.append(
                FleetTask(
                    id=f"task-{i + 1}",
                    prompt=prompt,
                    skills_dirs=fleet_skills,
                    exclude_tools=fleet_excluded,
                )
            )

    if not tasks:
        console.print("[red]No tasks to run[/red]")
        raise typer.Exit(1) from None

    try:
        from copex.repo_map import RepoMap

        repo_root = config.working_dir
        repo_map = RepoMap(repo_root)
        repo_map.refresh(force=False)
        for task in tasks:
            base_prompt = task.prompt
            relevant_context = repo_map.relevant_context(
                base_prompt,
                max_files=6,
                max_symbols_per_file=6,
            )
            if relevant_context:
                task.prompt = f"{relevant_context}\n\n{base_prompt}"
    except Exception as exc:
        console.print(f"[dim]Repo map unavailable for fleet: {exc}[/dim]")

    # Track statuses for live display
    statuses: dict[str, str] = {t.id: "pending" for t in tasks}

    def on_status(task_id: str, status: str) -> None:
        statuses[task_id] = status

    def build_table() -> Table:
        table = Table(title="Fleet Progress", expand=True)
        table.add_column("Task", style="cyan", no_wrap=True)
        table.add_column("Status", justify="center")
        table.add_column("Prompt", max_width=60)

        task_map = {t.id: t for t in tasks}
        for task_id, status in statuses.items():
            if status == "pending":
                badge = "[dim]⏳ pending[/dim]"
            elif status == "running":
                badge = "[yellow]⚡ running[/yellow]"
            elif status == "done":
                badge = "[green]✅ done[/green]"
            elif status == "failed":
                badge = "[red]❌ failed[/red]"
            elif status == "skipped":
                badge = "[dim]⏭️  skipped[/dim]"
            else:
                badge = f"[dim]{status}[/dim]"
            prompt_text = task_map[task_id].prompt[:60]
            table.add_row(task_id, badge, prompt_text)
        return table

    console.print(
        Panel(
            f"[bold]Running {len(tasks)} task(s)[/bold] • "
            f"max concurrent: {fleet_config.max_concurrent} • "
            f"timeout: {fleet_config.timeout:.0f}s",
            title="🚀 Fleet",
            border_style="blue",
        )
    )

    start_time = time.time()

    # --progress: background thread that watches for file changes
    _progress_stop = threading.Event()
    _progress_thread: threading.Thread | None = None

    if progress:
        cwd = config.working_dir

        def _count_lines(path: Path) -> int:
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    return sum(1 for _ in f)
            except (OSError, UnicodeDecodeError):
                return 0

        def _snapshot_files(
            root: Path,
            *,
            include_line_counts: bool,
        ) -> dict[Path, tuple[float, int, int | None]]:
            """Return {path: (mtime, size, line_count?)} for tracked files."""
            snap: dict[Path, tuple[float, int, int | None]] = {}
            for p in root.rglob("*"):
                if p.is_file() and ".git" not in p.parts and not p.name.startswith("."):
                    try:
                        st = p.stat()
                        line_count = _count_lines(p) if include_line_counts else None
                        snap[p] = (st.st_mtime, st.st_size, line_count)
                    except (OSError, UnicodeDecodeError):
                        pass
            return snap

        _baseline = _snapshot_files(cwd, include_line_counts=True)

        def _progress_worker() -> None:
            while not _progress_stop.wait(timeout=3.0):
                elapsed = time.time() - start_time
                mins, secs = divmod(int(elapsed), 60)
                current = _snapshot_files(cwd, include_line_counts=False)
                for p, (mtime, size, _) in current.items():
                    base = _baseline.get(p)
                    if base is None:
                        lines = _count_lines(p)
                        rel = p.relative_to(cwd)
                        console.print(
                            f"  [dim][{mins}m {secs:02d}s] Created: {rel} (+{lines} lines)[/dim]"
                        )
                        _baseline[p] = (mtime, size, lines)
                    else:
                        base_mtime, base_size, base_lines = base
                        if mtime <= base_mtime and size == base_size:
                            continue
                        lines = _count_lines(p)
                        previous_lines = base_lines or 0
                        diff = lines - previous_lines
                        sign = "+" if diff >= 0 else ""
                        rel = p.relative_to(cwd)
                        console.print(
                            f"  [dim][{mins}m {secs:02d}s] Modified: {rel} ({sign}{diff} lines)[/dim]"
                        )
                        _baseline[p] = (mtime, size, lines)

        _progress_thread = threading.Thread(target=_progress_worker, daemon=True)
        _progress_thread.start()

    # ── Worktree isolation setup ───────────────────────────────────
    worktree_managers: dict[str, WorktreeManager] = {}  # task_id -> manager  # noqa: F821
    if worktree:
        from copex.worktree import WorktreeManager

        repo_root = WorktreeManager.get_repo_root(config.working_dir)
        if repo_root is None:
            console.print("[red]Error: --worktree requires a git repository[/red]")
            raise typer.Exit(1) from None
        if WorktreeManager.has_uncommitted_changes(repo_root):
            console.print(
                "[yellow]Warning: uncommitted changes in working tree. "
                "Worktrees will be based on HEAD (uncommitted changes won't be included).[/yellow]"
            )
        for task in tasks:
            mgr = WorktreeManager(repo_root=repo_root)
            create_result = mgr.create_worktree()
            if not create_result.success:
                console.print(
                    f"[red]Failed to create worktree for task '{task.id}': "
                    f"{create_result.error}[/red]"
                )
                # Clean up already-created worktrees
                for prev_mgr in worktree_managers.values():
                    prev_mgr.cleanup_worktree()
                raise typer.Exit(1) from None
            worktree_managers[task.id] = mgr
            # Override task cwd to the worktree path
            task.cwd = str(mgr.worktree_path)
            console.print(
                f"  [dim]🌲 Task '{task.id}' → worktree {mgr.worktree_path}[/dim]"
            )

    async with Fleet(config, fleet_config) as fleet:
        for task in tasks:
            fleet.add(
                task.prompt,
                task_id=task.id,
                depends_on=task.depends_on if task.depends_on else None,
                model=task.model,
                reasoning_effort=task.reasoning_effort,
                cwd=task.cwd,
                skills=task.skills,
                exclude_tools=task.exclude_tools,
                mcp_servers=task.mcp_servers,
                timeout_sec=task.timeout_sec,
                skills_dirs=task.skills_dirs,
                on_dependency_failure=task.on_dependency_failure,
            )

        with Live(build_table(), console=console, refresh_per_second=4) as live:

            async def _run_with_updates() -> list:

                results = await fleet.run(on_status=on_status)
                # Mark final statuses
                for r in results:
                    statuses[r.task_id] = "done" if r.success else "failed"
                live.update(build_table())
                return results

            results = await _run_with_updates()

    # ── Worktree merge-back & cleanup ────────────────────────────────
    if worktree and worktree_managers:
        merge_failures: list[str] = []
        for r in results:
            mgr = worktree_managers.get(r.task_id)
            if mgr is None:
                continue
            if r.success:
                commit_res = mgr.commit_in_worktree(
                    f"fleet({r.task_id}): apply changes"
                )
                if commit_res.success and commit_res.commit_hash:
                    merge_res = mgr.merge_back()
                    if merge_res.success:
                        console.print(
                            f"  [green]✅ Task '{r.task_id}' merged back "
                            f"({commit_res.commit_hash[:8]})[/green]"
                        )
                    else:
                        merge_failures.append(
                            f"Task '{r.task_id}': {merge_res.error}"
                        )
                        console.print(
                            f"  [red]❌ Task '{r.task_id}' merge failed: "
                            f"{merge_res.error}[/red]"
                        )
                elif commit_res.error == "No changes to commit":
                    console.print(
                        f"  [dim]Task '{r.task_id}': no changes to merge[/dim]"
                    )
                else:
                    console.print(
                        f"  [red]❌ Task '{r.task_id}' commit failed: "
                        f"{commit_res.error}[/red]"
                    )
            else:
                console.print(
                    f"  [dim]Task '{r.task_id}' failed — skipping merge[/dim]"
                )

            # Always clean up
            mgr.cleanup_worktree()

        if merge_failures:
            console.print(
                Panel(
                    "\n".join(merge_failures),
                    title="⚠️  Worktree Merge Failures",
                    border_style="red",
                )
            )

    # Stop progress watcher
    if _progress_thread is not None:
        _progress_stop.set()
        _progress_thread.join(timeout=5.0)

    # --retry: detect build commands and retry on failure
    if retry > 0:
        _BUILD_CMDS = [
            "lake build", "cargo build", "npm run build", "yarn build",
            "make", "go build", "dotnet build", "gradle build",
            "mvn compile", "tsc", "pnpm build",
        ]

        def _detect_build_cmd() -> str | None:
            """Try to detect a build command from task prompts or common project files."""
            cwd_path = config.working_dir
            # Check for lakefile / Cargo.toml / package.json etc.
            detectors: list[tuple[str, str]] = [
                ("lakefile.lean", "lake build"),
                ("Cargo.toml", "cargo build"),
                ("package.json", "npm run build"),
                ("Makefile", "make"),
                ("go.mod", "go build ./..."),
                ("build.gradle", "gradle build"),
                ("pom.xml", "mvn compile"),
                ("tsconfig.json", "tsc --noEmit"),
            ]
            for marker, cmd in detectors:
                if (cwd_path / marker).exists():
                    return cmd
            return None

        build_cmd = _detect_build_cmd()
        if build_cmd:
            console.print(f"[blue]🔨 Detected build command: {build_cmd}[/blue]")
            for attempt in range(1, retry + 1):
                console.print(f"[yellow]Running build check ({attempt}/{retry})…[/yellow]")
                build_result = subprocess.run(
                    build_cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=str(config.working_dir),
                )
                if build_result.returncode == 0:
                    console.print("[green]✅ Build succeeded![/green]")
                    break
                else:
                    errors_text = (build_result.stderr or build_result.stdout or "unknown error").strip()
                    # Truncate very long error output
                    if len(errors_text) > 4000:
                        errors_text = errors_text[:4000] + "\n... (truncated)"
                    console.print(
                        f"[red]Build failed (attempt {attempt}/{retry}).[/red]\n"
                        f"[dim]{errors_text[:500]}[/dim]"
                    )
                    if attempt >= retry:
                        console.print("[red]❌ All retry attempts exhausted. Build still failing.[/red]")
                        break
                    # Use fleet to fix the errors
                    fix_prompt = (
                        f"Build failed with errors:\n{errors_text}\n\n"
                        f"Please fix these errors. The build command is: {build_cmd}"
                    )
                    console.print(f"[yellow]Retry {attempt}/{retry}: fixing build errors…[/yellow]")
                    async with Fleet(config, fleet_config) as retry_fleet:
                        retry_fleet.add(fix_prompt, task_id=f"retry-{attempt}")
                        retry_results = await retry_fleet.run()
                        for r in retry_results:
                            if not r.success:
                                console.print(f"[red]Retry task failed: {r.error}[/red]")
        else:
            console.print("[dim]--retry: no build system detected, skipping build check[/dim]")

    total_time = time.time() - start_time
    successes = sum(1 for r in results if r.success)
    failures = len(results) - successes
    git_artifact: dict[str, Any] | None = None
    duration_display = (
        _format_duration(total_time) if _format_duration is not None else f"{total_time:.0f}s"
    )

    # Summary
    summary_lines = [
        f"[green]✅ Succeeded: {successes}[/green]",
        f"[red]❌ Failed:    {failures}[/red]",
        f"[blue]⏱  Duration:  {duration_display}[/blue]",
    ]

    if failures > 0:
        summary_lines.append("")
        for r in results:
            if not r.success:
                err = str(r.error) if r.error else "unknown error"
                summary_lines.append(f"  [red]• {r.task_id}: {err}[/red]")

    console.print(
        Panel(
            "\n".join(summary_lines),
            title="📊 Fleet Summary",
            border_style="green" if failures == 0 else "red",
        )
    )

    # Print full task outputs when --verbose is set
    if verbose:
        for r in results:
            content = r.response.content if r.response else None
            if content:
                console.print(
                    Panel(
                        content,
                        title=f"📝 {r.task_id}",
                        border_style="green" if r.success else "red",
                    )
                )

    # Save each task result to a file when --output-dir is set
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        for r in results:
            content = r.response.content if r.response else ""
            out_file = output_dir / f"{r.task_id}.md"
            out_file.write_text(content or f"Error: {r.error}")
        console.print(f"[blue]Results saved to {output_dir}/[/blue]")

    # Git finalize: stage and commit all changes
    if git_finalize:
        from copex.worktree import GitFinalizer

        if not GitFinalizer.is_git_repo():
            console.print("[dim]Not a git repository — skipping git finalize[/dim]")
            git_artifact = {"enabled": True, "success": False, "error": "not a git repository"}
        else:
            # Auto-generate commit message from task descriptions if not provided
            if git_message:
                message = git_message
            else:
                task_summaries = [t.prompt[:50] for t in tasks[:5]]
                desc = "; ".join(task_summaries)
                if len(tasks) > 5:
                    desc += f" (+{len(tasks) - 5} more)"
                message = f"fleet: {desc}"

            console.print("[dim]Detecting changes…[/dim]")
            finalizer = GitFinalizer(message=message)
            git_result = await finalizer.finalize()
            git_artifact = {
                "enabled": True,
                "success": bool(git_result.success),
                "message": message,
                "commit_hash": git_result.commit_hash,
                "files_staged": git_result.files_staged,
                "error": git_result.error,
            }

            if git_result.success and git_result.commit_hash:
                console.print(
                    Panel(
                        f"[green]Committed {git_result.files_staged} file(s)[/green]\n"
                        f"[dim]{git_result.commit_hash}[/dim] {message}",
                        title="🔒 Git Finalize",
                        border_style="green",
                    )
                )
            elif git_result.success:
                console.print("[dim]No changes to commit[/dim]")
            else:
                console.print(f"[red]Git finalize failed: {git_result.error}[/red]")
    else:
        git_artifact = {"enabled": False}

    if artifact_path:
        result_map = {r.task_id: r for r in results}
        task_artifacts: list[dict[str, Any]] = []

        total_prompt_tokens = 0
        total_completion_tokens = 0
        for task in tasks:
            result = result_map.get(task.id)
            response = result.response if result else None
            prompt_tokens = int(response.prompt_tokens or 0) if response else 0
            completion_tokens = int(response.completion_tokens or 0) if response else 0
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            task_artifacts.append(
                {
                    "id": task.id,
                    "depends_on": task.depends_on,
                    "on_dependency_failure": (
                        task.on_dependency_failure.value
                        if hasattr(task.on_dependency_failure, "value")
                        else str(task.on_dependency_failure)
                    ),
                    "model": (task.model.value if task.model else config.model.value),
                    "reasoning_effort": (
                        task.reasoning_effort.value
                        if task.reasoning_effort
                        else config.reasoning_effort.value
                    ),
                    "status": statuses.get(task.id, "unknown"),
                    "success": result.success if result else None,
                    "duration_ms": result.duration_ms if result else None,
                    "error": str(result.error) if result and result.error else None,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "usage": response.usage if response else None,
                    "output_file": (
                        str((output_dir / f"{task.id}.md").resolve()) if output_dir else None
                    ),
                    "content_preview": (
                        response.content[:500] if response and response.content else None
                    ),
                }
            )

        artifact = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_tasks": len(tasks),
                "succeeded": successes,
                "failed": failures,
                "duration_seconds": total_time,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
            },
            "run_config": {
                "max_concurrent": fleet_config.max_concurrent,
                "timeout_seconds": fleet_config.timeout,
                "fail_fast": fleet_config.fail_fast,
                "shared_context": fleet_config.shared_context,
            },
            "git_finalize": git_artifact,
            "tasks": task_artifacts,
        }

        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        console.print(f"[blue]Run artifact saved to {artifact_path}[/blue]")

    if failures > 0:
        raise typer.Exit(1) from None
