"""Plan CLI command and helpers."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from copex.config import CopexConfig, make_client
from copex.models import Model, ReasoningEffort, normalize_reasoning_effort, parse_reasoning_effort
from copex.plan import Plan, PlanExecutor, PlanState, PlanStep, StepStatus
from copex.ralph import RalphWiggum

console = Console()
_DEFAULT_MODEL = Model.CLAUDE_OPUS_4_5
_resolve_cli_model: Callable[[str], Model | str] | None = None
_apply_approval_flags: Callable[..., None] | None = None


def configure_plan_cli(
    *,
    shared_console: Console,
    default_model: Model,
    resolve_cli_model: Callable[[str], Model | str],
    apply_approval_flags: Callable[..., None],
) -> None:
    """Configure shared dependencies used by plan CLI handlers.

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


def register_plan_commands(app: typer.Typer) -> None:
    """Register plan-related commands on the provided Typer app.

    Args:
        app: Typer application to register commands on.

    Returns:
        None: Commands are attached directly to ``app``.
    """
    app.command("plan")(plan_command)


def plan_command(
    task: Annotated[
        str | None, typer.Argument(help="Task to plan (optional with --resume)")
    ] = None,
    execute: Annotated[
        bool, typer.Option("--execute", "-e", help="Execute the plan after generating")
    ] = False,
    review: Annotated[
        bool, typer.Option("--review", "-R", help="Show plan and confirm before executing")
    ] = False,
    resume: Annotated[
        bool, typer.Option("--resume", help="Resume from last checkpoint (.copex-state.json)")
    ] = False,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Save plan to file")
    ] = None,
    from_step: Annotated[
        int, typer.Option("--from-step", "-f", help="Resume execution from step number")
    ] = 1,
    load_plan: Annotated[
        Path | None,
        typer.Option("--load", "-l", help="Load plan from file instead of generating"),
    ] = None,
    max_iterations: Annotated[
        int, typer.Option("--max-iterations", "-n", help="Max iterations per step (Ralph loop)")
    ] = 10,
    visualize: Annotated[
        str | None,
        typer.Option("--visualize", "-V", help="Show plan visualization (ascii, mermaid, tree)"),
    ] = None,
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model to use")] = None,
    context_budget: Annotated[
        int | None,
        typer.Option("--context-budget", help="Override context window budget in tokens"),
    ] = None,
    reasoning: Annotated[
        ReasoningEffort, typer.Option("--reasoning", "-r", help="Reasoning effort level")
    ] = ReasoningEffort.HIGH,
    skill_dir: Annotated[
        list[str] | None, typer.Option("--skill-dir", "-S", help="Add skill directory")
    ] = None,
    disable_skill: Annotated[
        list[str] | None, typer.Option("--disable-skill", help="Disable specific skill")
    ] = None,
    no_auto_skills: Annotated[
        bool, typer.Option("--no-auto-skills", help="Disable skill auto-discovery")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Generate and display the plan without executing")
    ] = False,
    auto_approve: Annotated[
        bool, typer.Option("--auto-approve", help="Apply file changes without prompts")
    ] = False,
    approve: Annotated[
        bool, typer.Option("--approve", help="Prompt before each file change")
    ] = False,
    approval_dry_run: Annotated[
        bool,
        typer.Option(
            "--approval-dry-run",
            help="Show file change diffs without applying them during execution",
        ),
    ] = False,
    audit: Annotated[
        bool, typer.Option("--audit", help="Log file change decisions to .copex/audit.log")
    ] = False,
) -> None:
    """
        Generate and optionally execute a step-by-step plan.

        Examples:
            copex plan "Build a REST API"              # Generate plan only
            copex plan "Build a REST API" --execute    # Generate and execute
            copex plan "Build a REST API" --review     # Generate, review, then execute
            copex plan "Build a REST API" --dry-run    # Generate plan, show it, don't execute
            copex plan "Build a REST API" --visualize ascii  # Show ASCII plan graph
            copex plan --resume                        # Resume from .copex-state.json
            copex plan "Continue" --load plan.json -f3 # Resume from step 3

    Args:
        task: CLI argument or option value.
        execute: CLI argument or option value.
        review: CLI argument or option value.
        resume: CLI argument or option value.
        output: CLI argument or option value.
        from_step: CLI argument or option value.
        load_plan: CLI argument or option value.
        max_iterations: CLI argument or option value.
        visualize: CLI argument or option value.
        model: CLI argument or option value.
        context_budget: CLI argument or option value.
        reasoning: CLI argument or option value.
        skill_dir: CLI argument or option value.
        disable_skill: CLI argument or option value.
        no_auto_skills: CLI argument or option value.
        dry_run: CLI argument or option value.
        auto_approve: CLI argument or option value.
        approve: CLI argument or option value.
        approval_dry_run: CLI argument or option value.
        audit: CLI argument or option value.

    Returns:
        None: Command result.
    """
    # Validate: need task OR resume OR load_plan
    if not task and not resume and not load_plan:
        console.print("[red]Error: Provide a task, --resume, or --load[/red]")
        raise typer.Exit(1) from None

    effective_model = model or _DEFAULT_MODEL.value
    try:
        if _resolve_cli_model is None:
            raise RuntimeError("Plan CLI not configured")
        model_enum = _resolve_cli_model(effective_model)
        requested_effort = parse_reasoning_effort(reasoning) or ReasoningEffort.HIGH
        normalized_effort, warning = normalize_reasoning_effort(model_enum, requested_effort)
        if warning:
            console.print(f"[yellow]{warning}[/yellow]")

        config = CopexConfig(model=model_enum, reasoning_effort=normalized_effort)
        if context_budget is not None:
            config.context_budget = context_budget

        # Skills options
        if skill_dir:
            config.skill_directories.extend(skill_dir)
        if disable_skill:
            config.disabled_skills.extend(disable_skill)
        if no_auto_skills:
            config.auto_discover_skills = False
        if _apply_approval_flags is None:
            raise RuntimeError("Plan CLI not configured")
        _apply_approval_flags(
            config,
            auto_approve=auto_approve,
            approve=approve,
            dry_run=approval_dry_run,
            audit=audit,
        )
    except typer.BadParameter as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None

    # --dry-run overrides --execute/--review/--resume to prevent execution
    should_execute = (execute or review or resume) and not dry_run

    asyncio.run(
        _run_plan(
            config=config,
            task=task or "",
            execute=should_execute,
            review=review and not dry_run,
            resume=resume,
            output=output,
            from_step=from_step,
            load_plan=load_plan,
            max_iterations=max_iterations,
            visualize=visualize,
        )
    )


def _format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours}h {mins}m"


async def _run_plan(
    config: CopexConfig,
    task: str,
    execute: bool,
    review: bool,
    resume: bool,
    output: Path | None,
    from_step: int,
    load_plan: Path | None,
    max_iterations: int = 10,
    visualize: str | None = None,
) -> None:
    """Run plan generation and optional execution with beautiful UI."""
    from copex.ui import PlanUI
    from copex.visualization import visualize_plan

    client = make_client(config)
    await client.start()

    plan_ui = PlanUI(console)

    try:
        # Create Ralph instance for iterative step execution
        ralph = RalphWiggum(client)
        executor = PlanExecutor(client, ralph=ralph)
        executor.max_iterations_per_step = max_iterations
        repo_map = None
        try:
            from copex.repo_map import RepoMap

            repo_map = RepoMap(Path.cwd())
            repo_map.refresh(force=False)
        except Exception as exc:
            console.print(
                f"[dim]Repo map unavailable for plan ({type(exc).__name__}): {exc}[/dim]"
            )
            repo_map = None

        # Check for resume from checkpoint
        if resume:
            state = PlanState.load()
            if state is None:
                console.print("[red]No checkpoint found (.copex-state.json)[/red]")
                console.print("[dim]Run a plan with --execute first to create a checkpoint[/dim]")
                raise typer.Exit(1) from None

            plan = state.plan
            from_step = state.current_step
            if repo_map is not None:
                executor.repo_context = repo_map.relevant_context(state.task)
            console.print(
                Panel(
                    f"[bold]Resuming plan:[/bold] {state.task}\n"
                    f"[dim]Started:[/dim] {state.started_at}\n"
                    f"[dim]Completed steps:[/dim] {len(state.completed)}/{len(plan.steps)}\n"
                    f"[dim]Resuming from step:[/dim] {from_step}",
                    title="🔄 Resume from Checkpoint",
                    border_style="yellow",
                )
            )
        elif load_plan:
            # Load from plan file
            if not load_plan.exists():
                console.print(f"[red]Plan file not found: {load_plan}[/red]")
                raise typer.Exit(1) from None
            plan = Plan.load(load_plan)
            if repo_map is not None:
                executor.repo_context = repo_map.relevant_context(plan.task)
            console.print(f"[green]✓ Loaded plan from {load_plan}[/green]\n")
        else:
            # Generate new plan
            if repo_map is not None:
                executor.repo_context = repo_map.relevant_context(task)
            console.print(
                Panel(
                    f"[bold]Generating plan for:[/bold]\n{task}",
                    title="📋 Plan Mode",
                    border_style="blue",
                )
            )

            plan = await executor.generate_plan(task)
            console.print(f"\n[green]✓ Generated {len(plan.steps)} steps[/green]\n")

        # Display plan overview with new UI
        steps_info = [(s.number, s.description, s.status.value) for s in plan.steps]
        plan_ui.print_plan_overview(steps_info)

        # Show visualization if requested
        if visualize:
            try:
                viz_output = visualize_plan(plan, format=visualize)
                console.print(f"\n[bold]Plan Visualization ({visualize}):[/bold]")
                console.print(viz_output)
                console.print()
            except ValueError as e:
                console.print(f"[yellow]Visualization error: {e}[/yellow]")

        # Save plan if requested
        if output:
            plan.save(output)
            console.print(f"\n[green]✓ Saved plan to {output}[/green]")

        # Execute if requested
        if execute:
            if review:
                if not typer.confirm("\nProceed with execution?"):
                    console.print("[yellow]Execution cancelled[/yellow]")
                    return

            # Print plan execution header
            plan_ui.print_plan_header(
                task=plan.task,
                step_count=len(plan.steps),
                model=config.model.value,
                reasoning=config.reasoning_effort.value,
            )

            # Track execution timing
            plan_start_time = time.time()

            def on_step_start(step: PlanStep) -> None:
                plan_ui.print_step_start(
                    step_number=step.number,
                    total_steps=len(plan.steps),
                    description=step.description,
                )

            def on_step_complete(step: PlanStep) -> None:
                duration = step.duration_seconds or 0
                preview = (step.result or "")[:150]
                if len(step.result or "") > 150:
                    preview += "..."

                # Calculate ETA
                eta = None
                if plan.completed_count >= 2:
                    eta = plan.estimate_remaining_seconds()

                plan_ui.print_step_complete(
                    step_number=step.number,
                    total_steps=len(plan.steps),
                    duration=duration,
                    result_preview=preview if preview else None,
                    eta_remaining=eta,
                )

            def on_error(step: PlanStep, error: Exception) -> bool:
                duration = step.duration_seconds or 0
                plan_ui.print_step_failed(
                    step_number=step.number,
                    total_steps=len(plan.steps),
                    error=str(error),
                    duration=duration,
                )
                return typer.confirm("Continue with next step?", default=False)

            await executor.execute_plan(
                plan,
                from_step=from_step,
                on_step_start=on_step_start,
                on_step_complete=on_step_complete,
                on_error=on_error,
                save_checkpoints=True,
            )

            # Calculate total time
            total_time = time.time() - plan_start_time

            # Show enhanced summary with new UI
            plan_ui.print_plan_complete(
                completed_steps=plan.completed_count,
                failed_steps=plan.failed_count,
                total_steps=len(plan.steps),
                total_time=total_time,
                total_tokens=plan.total_tokens,
            )

            # Save updated plan
            if output:
                plan.save(output)
                console.print(f"\n[green]✓ Updated plan saved to {output}[/green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled[/yellow]")
        console.print("[dim]Checkpoint saved. Resume with: copex plan --resume[/dim]")
    except Exception as e:  # Catch-all: top-level CLI error handler
        console.print(f"[red]Plan failed ({type(e).__name__}): {e}[/red]")
        console.print(
            "[dim]Run without --execute to inspect steps, or use --resume after fixing the issue.[/dim]"
        )
        raise typer.Exit(1) from None
    finally:
        await client.stop()


def _display_plan(plan: Plan) -> None:
    """Display plan steps."""
    for step in plan.steps:
        status_icon = {
            StepStatus.PENDING: "⬜",
            StepStatus.RUNNING: "🔄",
            StepStatus.COMPLETED: "✅",
            StepStatus.FAILED: "❌",
            StepStatus.SKIPPED: "⏭️",
        }.get(step.status, "⬜")
        console.print(f"{status_icon} [bold]Step {step.number}:[/bold] {step.description}")


def _display_plan_summary(plan: Plan) -> None:
    """Display plan execution summary."""
    completed = plan.completed_count
    failed = plan.failed_count
    total = len(plan.steps)

    if plan.is_complete and failed == 0:
        console.print(
            Panel(
                f"[green]All {total} steps completed successfully![/green]",
                title="✅ Plan Complete",
                border_style="green",
            )
        )
    elif failed > 0:
        console.print(
            Panel(
                f"Completed: {completed}/{total}\nFailed: {failed}",
                title="⚠️ Plan Incomplete",
                border_style="yellow",
            )
        )
    else:
        console.print(
            Panel(
                f"Completed: {completed}/{total}",
                title="📋 Progress",
                border_style="blue",
            )
        )


def _display_plan_summary_enhanced(plan: Plan, total_time: float) -> None:
    """Display enhanced plan execution summary with timing and tokens."""
    completed = plan.completed_count
    failed = plan.failed_count
    total = len(plan.steps)

    # Build summary lines
    lines = []

    if plan.is_complete and failed == 0:
        lines.append(f"[green]✅ {completed}/{total} steps completed successfully![/green]")
    elif failed > 0:
        lines.append(f"[yellow]⚠️ {completed}/{total} steps completed, {failed} failed[/yellow]")
    else:
        lines.append(f"[blue]📋 {completed}/{total} steps completed[/blue]")

    # Timing
    lines.append("")
    lines.append(f"[bold]Total time:[/bold] {_format_duration(total_time)}")

    # Per-step breakdown
    if completed > 0:
        avg = plan.avg_step_duration
        if avg:
            lines.append(f"[dim]Avg per step: {_format_duration(avg)}[/dim]")

    # Token usage (if tracked)
    if plan.total_tokens > 0:
        lines.append(f"[bold]Tokens used:[/bold] {plan.total_tokens:,}")

    # Determine panel style
    if plan.is_complete and failed == 0:
        title = "✅ Plan Complete"
        border = "green"
    elif failed > 0:
        title = "⚠️ Plan Incomplete"
        border = "yellow"
    else:
        title = "📋 Progress"
        border = "blue"

    console.print(
        Panel(
            "\n".join(lines),
            title=title,
            border_style=border,
        )
    )
