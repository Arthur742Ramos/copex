"""Squad CLI commands and helpers."""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel

from copex.config import CopexConfig
from copex.models import Model, ReasoningEffort, normalize_reasoning_effort, parse_reasoning_effort

console = Console()
_DEFAULT_MODEL = Model.CLAUDE_OPUS_4_5
_resolve_cli_model: Callable[[str], Model | str] | None = None
_apply_approval_flags: Callable[..., None] | None = None


def configure_squad_cli(
    *,
    shared_console: Console,
    default_model: Model,
    resolve_cli_model: Callable[[str], Model | str],
    apply_approval_flags: Callable[..., None],
) -> None:
    """Configure shared dependencies used by squad CLI handlers.

    Args:
        shared_console: Console instance used for command output.
        default_model: Default model used when no CLI model override is provided.
        resolve_cli_model: Resolver for CLI model strings.
        apply_approval_flags: Callback to apply approval-mode flags to config.

    Returns:
        None: Module-level configuration is updated in place.
    """
    global console, _DEFAULT_MODEL, _resolve_cli_model, _apply_approval_flags
    console = shared_console
    _DEFAULT_MODEL = default_model
    _resolve_cli_model = resolve_cli_model
    _apply_approval_flags = apply_approval_flags


def register_squad_commands(app: typer.Typer) -> None:
    """Register squad commands on the provided Typer app.

    Args:
        app: Typer application to register commands on.

    Returns:
        None: Commands are attached directly to ``app``.
    """
    app.command("squad")(squad_command)


def squad_command(
    args: Annotated[
        list[str] | None,
        typer.Argument(help="Task for the squad, or subcommand (status/show/init/add/remove/reset/knowledge)"),
    ] = None,
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model to use")] = None,
    reasoning: Annotated[
        ReasoningEffort, typer.Option("--reasoning", "-r", help="Reasoning effort level")
    ] = ReasoningEffort.HIGH,
    json_output: Annotated[
        bool, typer.Option("--json", help="Output JSON result")
    ] = False,
    use_cli: Annotated[
        bool,
        typer.Option("--use-cli", help="Use CLI subprocess instead of SDK"),
    ] = False,
    no_ai: Annotated[
        bool,
        typer.Option("--no-ai", help="Skip AI repo analysis, use pattern matching"),
    ] = False,
    max_cost: Annotated[
        float | None,
        typer.Option("--max-cost", min=0.0, help="Abort if cumulative squad cost exceeds USD limit"),
    ] = None,
    max_tokens: Annotated[
        int | None,
        typer.Option("--max-tokens", min=1, help="Abort if cumulative squad tokens exceed limit"),
    ] = None,
    config_file: Annotated[
        Path | None, typer.Option("--config", "-c", help="Config file path")
    ] = None,
    stdin: Annotated[bool, typer.Option("--stdin", "-i", help="Read prompt from stdin")] = False,
    auto_approve: Annotated[
        bool, typer.Option("--auto-approve", help="Apply file changes without prompts")
    ] = False,
    approve: Annotated[
        bool, typer.Option("--approve", help="Prompt before each file change")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show file change diffs without applying them")
    ] = False,
    audit: Annotated[
        bool, typer.Option("--audit", help="Log file change decisions to .copex/audit.log")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Force rerun all squad agents (ignore .squad/state.json)")
    ] = False,
) -> None:
    """Run a squad or manage the repo .squad file.

        Management subcommands:
          - status: display parsed squad configuration
          - show: display .squad content
          - init: create .squad from AI repo analysis
          - add/remove: apply natural-language edits to .squad using AI
          - reset: delete .squad and fall back to dynamic AI analysis
          - knowledge: show/reset persistent squad knowledge store

        Without a management subcommand, this runs the squad workflow.

        Examples:
            copex squad "Build a REST API with auth"
            copex squad status
            copex squad show
            copex squad init
            copex squad add "Add a Security Auditor agent that reviews code for vulnerabilities"
            copex squad remove "Remove the Docs agent"
            copex squad reset
            copex squad knowledge
            copex squad knowledge reset

    Args:
        args: CLI argument or option value.
        model: CLI argument or option value.
        reasoning: CLI argument or option value.
        json_output: CLI argument or option value.
        use_cli: CLI argument or option value.
        no_ai: CLI argument or option value.
        max_cost: CLI argument or option value.
        max_tokens: CLI argument or option value.
        config_file: CLI argument or option value.
        stdin: CLI argument or option value.
        auto_approve: CLI argument or option value.
        approve: CLI argument or option value.
        dry_run: CLI argument or option value.
        audit: CLI argument or option value.
        force: CLI argument or option value.

    Returns:
        None: Command result.
    """
    raw_args = args or []
    subcommand = raw_args[0].lower() if raw_args else None
    if subcommand == "status":
        if len(raw_args) != 1:
            console.print("[red]Usage: copex squad status[/red]")
            raise typer.Exit(1) from None
        _show_squad_status(json_output=json_output)
        return
    if subcommand == "show":
        if len(raw_args) != 1:
            console.print("[red]Usage: copex squad show[/red]")
            raise typer.Exit(1) from None
        _show_squad_file(json_output=json_output)
        return
    if subcommand == "reset":
        if len(raw_args) != 1:
            console.print("[red]Usage: copex squad reset[/red]")
            raise typer.Exit(1) from None
        _reset_squad_file(json_output=json_output)
        return
    if subcommand == "knowledge":
        action = raw_args[1].lower() if len(raw_args) > 1 else "show"
        if action == "show":
            if len(raw_args) not in {1, 2}:
                console.print("[red]Usage: copex squad knowledge [show][/red]")
                raise typer.Exit(1) from None
            _show_squad_knowledge(json_output=json_output)
            return
        if action == "reset":
            if len(raw_args) != 2:
                console.print("[red]Usage: copex squad knowledge reset[/red]")
                raise typer.Exit(1) from None
            _reset_squad_knowledge(json_output=json_output)
            return
        console.print("[red]Usage: copex squad knowledge [reset][/red]")
        raise typer.Exit(1) from None

    # Load config
    if config_file and config_file.exists():
        config = CopexConfig.from_file(config_file)
    else:
        default_path = CopexConfig.default_path()
        config = CopexConfig.from_file(default_path) if default_path.exists() else CopexConfig()

    # Override model
    effective_model = model or _DEFAULT_MODEL.value
    try:
        if _resolve_cli_model is None:
            raise RuntimeError("Squad CLI not configured")
        config.model = _resolve_cli_model(effective_model)
    except ValueError:
        console.print(f"[red]Invalid model: {effective_model}[/red]")
        raise typer.Exit(1) from None

    # Reasoning effort
    try:
        requested_effort = parse_reasoning_effort(reasoning)
        if requested_effort is None:
            raise ValueError(reasoning)
        normalized_effort, warning = normalize_reasoning_effort(config.model, requested_effort)
        if warning and not json_output:
            console.print(f"[yellow]{warning}[/yellow]")
        config.reasoning_effort = normalized_effort
    except ValueError:
        console.print(f"[red]Invalid reasoning effort: {reasoning}[/red]")
        raise typer.Exit(1) from None

    if use_cli:
        config.use_cli = True
    try:
        if _apply_approval_flags is None:
            raise RuntimeError("Squad CLI not configured")
        _apply_approval_flags(
            config,
            auto_approve=auto_approve,
            approve=approve,
            dry_run=dry_run,
            audit=audit,
            default_auto=True,
        )
    except typer.BadParameter as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None

    if subcommand == "init":
        if len(raw_args) != 1:
            console.print("[red]Usage: copex squad init[/red]")
            raise typer.Exit(1) from None
        asyncio.run(_init_squad_file(config, json_output=json_output))
        return

    if subcommand in {"add", "remove"}:
        request = " ".join(raw_args[1:]).strip()
        if not request:
            console.print(f"[red]Usage: copex squad {subcommand} \"<request>\"[/red]")
            raise typer.Exit(1) from None
        asyncio.run(
            _update_squad_file(config, request, action=subcommand, json_output=json_output)
        )
        return

    prompt = " ".join(raw_args).strip() if raw_args else None

    # Read prompt
    if stdin:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
        else:
            console.print("[yellow]Enter prompt (Ctrl+D to submit):[/yellow]")
            prompt = sys.stdin.read().strip()

    if prompt is None and not stdin:
        if sys.stdin.isatty():
            console.print("[yellow]Enter prompt (Ctrl+D to submit):[/yellow]")
        prompt = sys.stdin.read().strip()

    if not prompt:
        console.print("[red]No prompt provided[/red]")
        raise typer.Exit(1) from None

    asyncio.run(
        _run_squad(
            config,
            prompt,
            json_output=json_output,
            no_ai=no_ai,
            max_cost=max_cost,
            max_tokens=max_tokens,
            auto_approve_gates=auto_approve,
            force=force,
        )
    )


def _squad_dir_path(path: Path | None = None) -> Path:
    root = path or Path.cwd()
    return root / ".squad"


def _squad_file_path(path: Path | None = None) -> Path:
    return _squad_dir_path(path) / "team.toml"


def _legacy_squad_toml_path(path: Path | None = None) -> Path:
    root = path or Path.cwd()
    return root / ".squad"


def _legacy_squad_file_path(path: Path | None = None) -> Path:
    """Backward-compatible alias for legacy single-file .squad path."""
    return _legacy_squad_toml_path(path)


def _legacy_squad_json_path(path: Path | None = None) -> Path:
    root = path or Path.cwd()
    return root / ".copex" / "squad.json"


def _show_squad_file(*, json_output: bool = False) -> None:
    squad_path = _squad_file_path()
    legacy_toml_path = _legacy_squad_toml_path()
    legacy_json_path = _legacy_squad_json_path()
    selected_path: Path | None = None
    if squad_path.is_file():
        selected_path = squad_path
    elif legacy_toml_path.is_file():
        selected_path = legacy_toml_path
    elif legacy_json_path.is_file():
        selected_path = legacy_json_path

    if selected_path is None:
        if json_output:
            print(json.dumps({"exists": False, "path": str(squad_path)}), flush=True)
        else:
            console.print("[yellow]No .squad file found.[/yellow]")
        return

    content = selected_path.read_text(encoding="utf-8")
    if json_output:
        print(
            json.dumps({"exists": True, "path": str(selected_path), "content": content}, ensure_ascii=False),
            flush=True,
        )
    else:
        console.print(f"[cyan]{selected_path}[/cyan]")
        console.print(content)


def _show_squad_status(*, json_output: bool = False) -> None:
    from rich.table import Table

    from copex.squad_team import SquadTeam

    team = SquadTeam.load_squad_file(Path.cwd())
    squad_path = _squad_file_path()
    if team is None:
        if json_output:
            print(json.dumps({"exists": False, "path": str(squad_path)}), flush=True)
        else:
            console.print("[yellow]No .squad/team.toml found.[/yellow]")
        return

    lead = team.get_agent("lead")
    agents_payload = []
    for agent in team.agents:
        agents_payload.append(
            {
                "name": agent.name,
                "role": agent.role,
                "phase": agent.phase,
                "retries": int(agent.retries),
                "subtasks": list(agent.subtasks),
            }
        )

    if json_output:
        print(
            json.dumps(
                {
                    "exists": True,
                    "path": str(squad_path),
                    "lead": lead.name if lead else "Lead",
                    "agents": agents_payload,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return

    table = Table(title="Squad Status", expand=True)
    table.add_column("Role", style="cyan", no_wrap=True)
    table.add_column("Name", style="white")
    table.add_column("Phase", justify="center")
    table.add_column("Retries", justify="center")
    table.add_column("Subtasks", style="dim")
    for agent in team.agents:
        subtasks = ", ".join(agent.subtasks) if agent.subtasks else "—"
        table.add_row(
            agent.role,
            agent.name,
            str(agent.phase),
            str(int(agent.retries)),
            subtasks,
        )

    console.print(f"[cyan]{squad_path}[/cyan]")
    if lead is not None:
        console.print(f"[bold]Lead:[/bold] {lead.name}")
    console.print(table)


def _reset_squad_file(*, json_output: bool = False) -> None:
    squad_dir = _squad_dir_path()
    legacy_json = _legacy_squad_json_path()
    deleted = False
    if squad_dir.is_dir():
        shutil.rmtree(squad_dir)
        deleted = True
    elif squad_dir.is_file():
        squad_dir.unlink()
        deleted = True
    if legacy_json.is_file():
        legacy_json.unlink()
        deleted = True
    if json_output:
        print(json.dumps({"deleted": deleted, "path": str(squad_dir)}), flush=True)
    elif deleted:
        console.print(f"[green]Deleted {squad_dir}[/green]")
    else:
        console.print("[yellow]No .squad file found.[/yellow]")


def _show_squad_knowledge(*, json_output: bool = False) -> None:
    squad_dir = _squad_dir_path()
    knowledge_dir = squad_dir / "knowledge"
    decisions_path = squad_dir / "decisions.md"
    knowledge: dict[str, str] = {}

    if knowledge_dir.is_dir():
        for file_path in sorted(knowledge_dir.glob("*.md")):
            knowledge[file_path.stem] = file_path.read_text(encoding="utf-8")

    decisions = decisions_path.read_text(encoding="utf-8") if decisions_path.is_file() else ""

    if json_output:
        print(
            json.dumps(
                {
                    "path": str(squad_dir),
                    "knowledge": knowledge,
                    "decisions": decisions,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return

    if not knowledge and not decisions:
        console.print("[yellow]No squad knowledge found.[/yellow]")
        return

    console.print(f"[cyan]{squad_dir}[/cyan]")
    if decisions:
        console.print("[bold]Shared decisions[/bold]")
        console.print(decisions)
    for role, content in knowledge.items():
        console.print(f"[bold]{role} knowledge[/bold]")
        console.print(content)


def _reset_squad_knowledge(*, json_output: bool = False) -> None:
    squad_dir = _squad_dir_path()
    knowledge_dir = squad_dir / "knowledge"
    decisions_path = squad_dir / "decisions.md"

    deleted_paths: list[str] = []
    if knowledge_dir.is_dir():
        shutil.rmtree(knowledge_dir)
        deleted_paths.append(str(knowledge_dir))
    if decisions_path.is_file():
        decisions_path.unlink()
        deleted_paths.append(str(decisions_path))

    if json_output:
        print(
            json.dumps(
                {
                    "deleted": bool(deleted_paths),
                    "paths": deleted_paths,
                }
            ),
            flush=True,
        )
    elif deleted_paths:
        for path in deleted_paths:
            console.print(f"[green]Deleted {path}[/green]")
    else:
        console.print("[yellow]No squad knowledge found.[/yellow]")


def _serialize_squad_file(team: Any, squad_path: Path) -> dict[str, Any]:
    lead = team.get_agent("lead")
    return {
        "path": str(squad_path),
        "lead": lead.name if lead else "Lead",
        "agents": [
            {
                "name": agent.name,
                "role": agent.role,
                "phase": agent.phase,
                "depends_on": list(agent.depends_on),
            }
            for agent in team.agents
            if agent.role != "lead"
        ],
        "phases": [
            {"phase": phase, "gate": gate}
            for phase, gate in sorted(getattr(team, "phase_gates", {}).items())
        ],
    }


async def _init_squad_file(config: CopexConfig, *, json_output: bool = False) -> None:
    from copex.squad_team import SquadTeam

    team = await SquadTeam.from_repo_ai(config=config, path=Path.cwd())
    squad_path = _squad_file_path()
    team.save_squad_file(squad_path)
    if json_output:
        print(json.dumps(_serialize_squad_file(team, squad_path), ensure_ascii=False), flush=True)
    else:
        console.print(f"[green]Created {squad_path}[/green]")


async def _update_squad_file(
    config: CopexConfig,
    request: str,
    *,
    action: str,
    json_output: bool = False,
) -> None:
    from copex.squad_team import SquadTeam

    team = await SquadTeam.update_from_request(
        f"{action}: {request}",
        config=config,
        path=Path.cwd(),
    )
    squad_path = _squad_file_path()
    team.save_squad_file(squad_path)
    if json_output:
        print(json.dumps(_serialize_squad_file(team, squad_path), ensure_ascii=False), flush=True)
    else:
        console.print(f"[green]Updated {squad_path}[/green]")


async def _run_squad(
    config: CopexConfig,
    prompt: str,
    *,
    json_output: bool = False,
    no_ai: bool = False,
    max_cost: float | None = None,
    max_tokens: int | None = None,
    auto_approve_gates: bool = False,
    force: bool = False,
) -> None:
    """Run the squad coordinator."""
    from copex.squad import SquadCoordinator
    from copex.squad_team import SquadTeam

    try:
        if not json_output:
            console.print(
                Panel(
                    "[bold cyan]Squad[/bold cyan] — AI-assembled team for your repo",
                    border_style="cyan",
                    expand=False,
                )
            )

        # Create team based on --no-ai flag
        team = None
        if no_ai:
            team = await SquadTeam.from_repo_or_file(config, use_ai=False)
        else:
            team = SquadTeam.load_squad_file()
            if not json_output:
                if team is not None:
                    console.print("📄 Using .squad configuration")
                else:
                    console.print("🔍 Analyzing repository...")

        coordinator = SquadCoordinator(
            config,
            team=team,
            max_cost=max_cost,
            max_tokens=max_tokens,
        )
        result = None
        stream_progress = (
            not json_output
            and not force
            and not auto_approve_gates
            and not (team is not None and bool(team.phase_gates))
        )
        async with coordinator:
            if stream_progress:
                from copex.squad import _ROLE_EMOJIS, SquadEventType

                async for event in coordinator.run_streaming(prompt):
                    if event.event_type == SquadEventType.PHASE_STARTED and event.phase is not None:
                        console.print(f"[cyan]▶ Phase {event.phase} started[/cyan]")
                        continue
                    if event.event_type == SquadEventType.PHASE_COMPLETED and event.phase is not None:
                        style = "green" if event.success else "red"
                        status = "completed" if event.success else "failed"
                        console.print(f"[{style}]✓ Phase {event.phase} {status}[/{style}]")
                        continue
                    if event.event_type == SquadEventType.AGENT_STARTED:
                        role = (event.role or "").split("__", 1)[0]
                        agent = event.agent
                        emoji = agent.emoji if agent is not None else _ROLE_EMOJIS.get(role, "🔹")
                        name = agent.name if agent is not None else role.replace("_", " ").title()
                        console.print(f"  {emoji} {name}: [yellow]running[/yellow]")
                        continue
                    if event.event_type == SquadEventType.AGENT_COMPLETED:
                        role = (event.role or "").split("__", 1)[0]
                        agent = event.agent
                        emoji = agent.emoji if agent is not None else _ROLE_EMOJIS.get(role, "🔹")
                        name = agent.name if agent is not None else role.replace("_", " ").title()
                        success = bool(event.success)
                        style = "green" if success else "red"
                        status = "done" if success else "failed"
                        line = f"  {emoji} {name}: [{style}]{status}[/{style}]"
                        if event.error and not success:
                            line += f" [dim]({event.error})[/dim]"
                        console.print(line)
                        continue
                    if event.event_type == SquadEventType.SQUAD_COMPLETED:
                        result = event.result
            else:
                def on_status(task_id: str, status: str) -> None:
                    if json_output:
                        return
                    from copex.squad import _ROLE_EMOJIS

                    role = task_id.split("__", 1)[0]
                    emoji = _ROLE_EMOJIS.get(role, "🔹")
                    name = role.replace("_", " ").title()
                    console.print(f"  {emoji} {name}: [dim]{status}[/dim]")

                result = await coordinator.run(
                    prompt,
                    on_status=on_status,
                    auto_approve_gates=auto_approve_gates,
                    force=force,
                    interactive=(sys.stdin.isatty() and sys.stdout.isatty() and not json_output),
                )

            if result is None:
                raise typer.Exit(1) from None

            # Show team composition after analysis
            if not json_output and coordinator.team.agents:
                team_desc = "  ".join(
                    f"{a.emoji} {a.name}" for a in coordinator.team.agents
                )
                console.print(f"  Team: {team_desc}")

        if json_output:
            print(result.to_json(indent=2), flush=True)
        else:
            for ar in result.agent_results:
                style = "green" if ar.success else "red"
                title = f"{ar.agent.emoji} {ar.agent.name}"
                content = ar.content[:3000] if ar.content else "[dim]No output[/dim]"
                if ar.content and len(ar.content) > 3000:
                    content += f"\n... ({len(ar.content)} chars total)"
                if ar.error:
                    content += f"\n[red]Error: {ar.error}[/red]"
                console.print(
                    Panel(
                        content,
                        title=f"[bold {style}]{title}[/bold {style}]",
                        subtitle=f"[dim]{ar.duration_ms / 1000:.1f}s[/dim]",
                        border_style=style,
                    )
                )

            console.print()
            stop_style = "green" if result.success else "red"
            agents_ok = sum(1 for ar in result.agent_results if ar.success)
            agents_total = len(result.agent_results)
            console.print(
                Panel(
                    f"[bold]{agents_ok}/{agents_total}[/bold] agents succeeded · "
                    f"[bold]{result.total_duration_ms / 1000:.1f}s[/bold] total",
                    title=f"[bold {stop_style}]Squad Complete[/bold {stop_style}]",
                    border_style=stop_style,
                    expand=False,
                )
            )
    except KeyboardInterrupt:
        console.print("\n[yellow]Squad interrupted[/yellow]")
    except Exception as e:
        if json_output:
            error_obj = {"success": False, "error": str(e), "agents": []}
            print(json.dumps(error_obj), flush=True)
        else:
            console.print(f"[red]Squad failed ({type(e).__name__}): {e}[/red]")
            console.print("[dim]Tip: run with --json for structured error output.[/dim]")
        raise typer.Exit(1) from None
