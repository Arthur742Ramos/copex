"""CLI interface for Copex."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

if TYPE_CHECKING:
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from copex import __version__
from copex.approval import AuditLogger
from copex.cli_fleet import (
    FleetTaskSpec,
    _build_council_tasks,
    _load_fleet_json_config,
    _load_fleet_toml_config,
    _parse_fleet_task_specs,
    _run_fleet,
    configure_fleet_cli,
    council_command,
    fleet_command,
    register_fleet_commands,
)  # noqa: F401
from copex.cli_plan import (
    _display_plan,
    _display_plan_summary,
    _display_plan_summary_enhanced,
    _format_duration,
    _run_plan,
    configure_plan_cli,
    plan_command,
    register_plan_commands,
)  # noqa: F401
from copex.cli_squad import (
    _init_squad_file,
    _legacy_squad_file_path,
    _reset_squad_file,
    _reset_squad_knowledge,
    _run_squad,
    _serialize_squad_file,
    _show_squad_file,
    _show_squad_knowledge,
    _show_squad_status,
    _squad_dir_path,
    _squad_file_path,
    _update_squad_file,
    configure_squad_cli,
    register_squad_commands,
    squad_command,
)  # noqa: F401
from copex.cli_stream import (
    _stream_response,
    _stream_response_interactive,
    configure_stream_console,
)
from copex.cli_utils import apply_approval_flags as _apply_approval_flags
from copex.config import (
    COPILOT_CLI_NOT_FOUND_MESSAGE,
    CopexConfig,
    load_last_model,
    make_client,
    save_last_model,
)
from copex.edits import apply_edit_text, list_undo_history, undo_last_edit_batch
from copex.exceptions import ValidationError as CopexValidationError
from copex.log_render import render_jsonl
from copex.memory import ProjectMemory
from copex.model_router import option_was_explicit, route_model_for_prompt
from copex.models import (
    Model,
    ReasoningEffort,
    normalize_reasoning_effort,
    parse_reasoning_effort,
    resolve_model,
)
from copex.ralph import RalphState, RalphWiggum
from copex.ui import (
    CopexUI,
    Icons,
    Theme,
    print_error,
    print_user_prompt,
    print_welcome,
)

logger = logging.getLogger(__name__)

__all__ = [
    "FleetTaskSpec",
    "_build_council_tasks",
    "_display_plan",
    "_display_plan_summary",
    "_display_plan_summary_enhanced",
    "_format_duration",
    "_init_squad_file",
    "_legacy_squad_file_path",
    "_load_fleet_json_config",
    "_load_fleet_toml_config",
    "_parse_fleet_task_specs",
    "_reset_squad_file",
    "_reset_squad_knowledge",
    "_run_fleet",
    "_run_plan",
    "_run_squad",
    "_serialize_squad_file",
    "_show_squad_file",
    "_show_squad_knowledge",
    "_show_squad_status",
    "_squad_dir_path",
    "_squad_file_path",
    "_update_squad_file",
    "council_command",
    "fleet_command",
    "plan_command",
    "squad_command",
]

# Default model for CLI configure calls (constant to avoid import-time side effects).
# Actual model resolution uses _get_default_model() which lazy-loads the last used model.
_DEFAULT_MODEL: Model = Model.CLAUDE_OPUS_4_6
_DEFAULT_MODEL_CACHE: Model | None = None


def _get_default_model() -> Model:
    """Lazy-load default model (last used or claude-opus-4.6)."""
    global _DEFAULT_MODEL_CACHE
    if _DEFAULT_MODEL_CACHE is None:
        _DEFAULT_MODEL_CACHE = load_last_model() or Model.CLAUDE_OPUS_4_6
    return _DEFAULT_MODEL_CACHE


class _ResolvedModel(str):
    """String subclass that provides a .value property for CLI model resolution.

    When resolve_model returns a model ID not in the Model enum (e.g., a newly
    added model), we wrap it in _ResolvedModel so it can still be used with
    code expecting a .value attribute.
    """

    @property
    def value(self) -> str:
        return str(self)


def _resolve_cli_model(model: str) -> Model | _ResolvedModel:
    try:
        resolved = resolve_model(model)
    except CopexValidationError as exc:
        raise ValueError(str(exc)) from exc
    try:
        return Model(resolved)
    except ValueError:
        return _ResolvedModel(resolved)


def _record_start_commit() -> None:
    """Record current git HEAD so `copex diff` can reference it later."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        from copex.stats import save_start_commit
        save_start_commit(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass  # Not a git repo – nothing to record


app = typer.Typer(
    name="copex",
    help="Copilot Extended - Resilient wrapper with auto-retry and Ralph Wiggum loops.",
    no_args_is_help=False,
    invoke_without_command=True,
)
memory_app = typer.Typer(help="Manage persistent project memory.")
app.add_typer(memory_app, name="memory")
console = Console()


# Shell completion scripts
BASH_COMPLETION = """
_copex_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    opts="chat plan ralph fleet council interactive map render models skills memory status config init login logout tui completions stats diff squad audit agent edit undo"

    case "${prev}" in
        -m|--model)
            local models="claude-opus-4.6 claude-opus-4.6-fast claude-opus-4.6-1m claude-opus-4.5 claude-sonnet-4.6 claude-sonnet-4.5 claude-sonnet-4 claude-haiku-4.5 gpt-5.4 gpt-5.3-codex gpt-5.2-codex gpt-5.1-codex gpt-5.1-codex-max gpt-5.1-codex-mini gpt-5.2 gpt-5.1 gpt-5 gpt-5-mini gpt-4.1 gemini-3-pro-preview"
            COMPREPLY=( $(compgen -W "${models}" -- ${cur}) )
            return 0
            ;;
        -r|--reasoning)
            COMPREPLY=( $(compgen -W "none low medium high xhigh" -- ${cur}) )
            return 0
            ;;
        copex)
            COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
            return 0
            ;;
    esac

    if [[ ${cur} == -* ]]; then
        COMPREPLY=( $(compgen -W "--model -m --reasoning -r --help -h" -- ${cur}) )
        return 0
    fi

    COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
}
complete -F _copex_completion copex
"""

ZSH_COMPLETION = """
#compdef copex

_copex() {
    local -a commands models reasoning_levels
    commands=(
        'chat:Send a prompt to Copilot'
        'plan:Generate and execute step-by-step plans'
        'ralph:Start a Ralph Wiggum loop'
        'fleet:Run multiple tasks in parallel'
        'council:Run a model council with Opus as chair'
        'interactive:Start interactive chat session'
        'map:Show repository map and relevant files'
        'render:Render JSONL session logs'
        'models:List available models'
        'skills:Manage skills'
        'memory:Manage persistent project memory'
        'status:Check Copilot status'
        'config:Show configuration'
        'init:Create default config'
        'login:Login to GitHub'
        'logout:Logout from GitHub'
        'tui:Start the TUI'
        'completions:Generate shell completions'
        'stats:Show run statistics'
        'diff:Show changes since last run'
        'audit:Show approval audit log'
        'edit:Apply structured edits to files'
        'undo:Undo the latest edit batch'
    )
    models=(
        'claude-opus-4.6'
        'claude-opus-4.6-fast'
        'claude-opus-4.6-1m'
        'claude-opus-4.5'
        'claude-sonnet-4.6'
        'claude-sonnet-4.5'
        'claude-sonnet-4'
        'claude-haiku-4.5'
        'gpt-5.4'
        'gpt-5.3-codex'
        'gpt-5.2-codex'
        'gpt-5.1-codex'
        'gpt-5.1-codex-max'
        'gpt-5.1-codex-mini'
        'gpt-5.2'
        'gpt-5.1'
        'gpt-5'
        'gpt-5-mini'
        'gpt-4.1'
        'gemini-3-pro-preview'
    )
    reasoning_levels=('none' 'low' 'medium' 'high' 'xhigh')

    _arguments -C \\
        '1:command:->command' \\
        '*::arg:->args'

    case "$state" in
        command)
            _describe 'command' commands
            ;;
        args)
            case "$words[1]" in
                chat|plan|ralph|fleet|council|interactive)
                    _arguments \\
                        '(-m --model)'{-m,--model}'[Model to use]:model:($models)' \\
                        '(-r --reasoning)'{-r,--reasoning}'[Reasoning effort]:level:($reasoning_levels)' \\
                        '*:prompt:'
                    ;;
                completions)
                    _arguments '1:shell:(bash zsh fish)'
                    ;;
            esac
            ;;
    esac
}

_copex "$@"
"""

FISH_COMPLETION = """
# copex fish completion

set -l commands chat plan ralph fleet council interactive map render models skills memory status config init login logout tui completions stats diff squad audit agent edit undo
set -l models claude-opus-4.6 claude-opus-4.6-fast claude-opus-4.6-1m claude-opus-4.5 claude-sonnet-4.6 claude-sonnet-4.5 claude-sonnet-4 claude-haiku-4.5 gpt-5.4 gpt-5.3-codex gpt-5.2-codex gpt-5.1-codex gpt-5.1-codex-max gpt-5.1-codex-mini gpt-5.2 gpt-5.1 gpt-5 gpt-5-mini gpt-4.1 gemini-3-pro-preview
set -l reasoning none low medium high xhigh

complete -c copex -f
complete -c copex -n "not __fish_seen_subcommand_from $commands" -a "$commands"

# Model option
complete -c copex -s m -l model -d "Model to use" -xa "$models"
# Reasoning option
complete -c copex -s r -l reasoning -d "Reasoning effort" -xa "$reasoning"

# Completions subcommand
complete -c copex -n "__fish_seen_subcommand_from completions" -a "bash zsh fish"
"""


def version_callback(value: bool) -> None:
    """Print version and exit.

    Args:
        value: CLI argument or option value.

    Returns:
        None: Command result.
    """
    if value:
        console.print(f"copex version {__version__}")
        raise typer.Exit()


def model_callback(value: str | None) -> Model | _ResolvedModel | None:
    """Validate model name.

    Args:
        value: CLI argument or option value.

    Returns:
        Model | _ResolvedModel | None: Command result.
    """
    if value is None:
        return None
    return _resolve_cli_model(value)


def reasoning_callback(value: str | None) -> ReasoningEffort | None:
    """Validate reasoning effort.

    Args:
        value: CLI argument or option value.

    Returns:
        ReasoningEffort | None: Command result.
    """
    if value is None:
        return None
    try:
        return ReasoningEffort(value)
    except ValueError:
        valid = ", ".join(r.value for r in ReasoningEffort)
        raise typer.BadParameter(f"Invalid reasoning effort. Valid: {valid}") from None


def _parse_exclude_tools(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


configure_stream_console(console)
configure_plan_cli(
    shared_console=console,
    default_model=_DEFAULT_MODEL,
    resolve_cli_model=_resolve_cli_model,
    apply_approval_flags=_apply_approval_flags,
)
configure_fleet_cli(
    shared_console=console,
    default_model=_DEFAULT_MODEL,
    resolve_cli_model=_resolve_cli_model,
    parse_exclude_tools=_parse_exclude_tools,
    format_duration=_format_duration,
    apply_approval_flags=_apply_approval_flags,
)
configure_squad_cli(
    shared_console=console,
    default_model=_DEFAULT_MODEL,
    resolve_cli_model=_resolve_cli_model,
    apply_approval_flags=_apply_approval_flags,
)
register_plan_commands(app)
register_fleet_commands(app)
register_squad_commands(app)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit",
        ),
    ] = False,
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model to use")] = None,
    reasoning: Annotated[
        ReasoningEffort, typer.Option("--reasoning", "-r", help="Reasoning effort level")
    ] = ReasoningEffort.HIGH,
    prompt: Annotated[
        str | None, typer.Option("--prompt", "-p", help="Execute a prompt in non-interactive mode")
    ] = None,
    non_interactive: Annotated[
        bool,
        typer.Option(
            "--non-interactive", "-s", help="Non-interactive mode (exit after completion)"
        ),
    ] = False,
    allow_all: Annotated[
        bool, typer.Option("--allow-all", help="Enable all permissions (tools, paths, URLs)")
    ] = False,
    use_cli: Annotated[
        bool,
        typer.Option("--use-cli", help="Use CLI subprocess instead of SDK (supports all models)"),
    ] = False,
    no_squad: Annotated[
        bool,
        typer.Option("--no-squad", help="Disable squad mode; use single-agent chat instead"),
    ] = False,
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
    js_repl: Annotated[
        bool, typer.Option("--js-repl", help="Enable persistent JavaScript REPL tool (requires Node.js)")
    ] = False,
    js_repl_node: Annotated[
        str | None,
        typer.Option("--js-repl-node", help="Path to the Node.js executable for --js-repl"),
    ] = None,
    pdf_analyze: Annotated[
        bool, typer.Option("--pdf-analyze", help="Enable PDF analysis tools (requires PyMuPDF)")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Force rerun all squad agents (ignore .squad/state.json)")
    ] = False,
) -> None:
    """Copilot Extended - Resilient wrapper with auto-retry and Ralph Wiggum loops.

        By default, prompts run through the squad (Lead → Developer + Tester in
        parallel).  Pass --no-squad to use single-agent chat mode instead.

    Args:
        ctx: Typer context for command dispatch.
        version: CLI argument or option value.
        model: CLI argument or option value.
        reasoning: CLI argument or option value.
        prompt: CLI argument or option value.
        non_interactive: CLI argument or option value.
        allow_all: CLI argument or option value.
        use_cli: CLI argument or option value.
        no_squad: CLI argument or option value.
        auto_approve: CLI argument or option value.
        approve: CLI argument or option value.
        dry_run: CLI argument or option value.
        audit: CLI argument or option value.
        js_repl: CLI argument or option value.
        js_repl_node: CLI argument or option value.
        pdf_analyze: CLI argument or option value.
        force: CLI argument or option value.

    Returns:
        None: Command result.
    """
    # Record HEAD commit before any command runs (for `copex diff`)
    _record_start_commit()

    if ctx.invoked_subcommand is None:
        effective_model = model or _get_default_model().value

        # If -p/--prompt provided, run squad (default) or chat
        if prompt is not None:
            try:
                config = CopexConfig()
                if use_cli:
                    config.use_cli = True
                model_explicit = option_was_explicit("model")
                reasoning_explicit = option_was_explicit("reasoning")
                route = None
                if not model_explicit:
                    route = route_model_for_prompt(
                        prompt,
                        client_options=config.to_client_options(),
                    )
                    effective_model = route.model
                config.model = _resolve_cli_model(effective_model)
                requested_effort = parse_reasoning_effort(reasoning)
                if requested_effort is None:
                    requested_effort = ReasoningEffort.HIGH
                if route is not None and not reasoning_explicit:
                    requested_effort = route.reasoning_effort
                normalized_effort, warning = normalize_reasoning_effort(
                    config.model, requested_effort
                )
                if warning:
                    console.print(f"[yellow]{warning}[/yellow]")
                config.reasoning_effort = normalized_effort
                if js_repl or js_repl_node:
                    config.js_repl = True
                if js_repl_node:
                    config.js_repl_node_path = js_repl_node
                if pdf_analyze:
                    config.pdf_analyze = True
                _apply_approval_flags(
                    config,
                    auto_approve=auto_approve,
                    approve=approve,
                    dry_run=dry_run,
                    audit=audit,
                    default_auto=not no_squad,
                )

                if no_squad:
                    # Single-agent chat mode
                    asyncio.run(_run_chat(config, prompt, show_reasoning=True, raw=non_interactive))
                else:
                    # Squad mode (default): Lead → Developer + Tester
                    asyncio.run(
                        _run_squad(
                            config,
                            prompt,
                            json_output=non_interactive,
                            auto_approve_gates=auto_approve,
                            force=force,
                        )
                    )
            except typer.BadParameter as e:
                console.print(f"[red]Error: {e}[/red]")
                raise typer.Exit(1) from None
            except ValueError:
                console.print(f"[red]Invalid model: {effective_model}[/red]")
                raise typer.Exit(1) from None
        else:
            # No prompt provided - launch interactive mode
            interactive(
                model=effective_model,
                reasoning=reasoning,
                js_repl=js_repl,
                js_repl_node=js_repl_node,
                pdf_analyze=pdf_analyze,
            )


@memory_app.command("show")
def memory_show() -> None:
    """Display current project memory.

    Args:
        None.

    Returns:
        None: Command result.
    """
    memory = ProjectMemory(Path.cwd())
    content = memory.read_memory().strip()
    if not content:
        console.print("[yellow]No project memory found at .copex/memory.md[/yellow]")
        return
    console.print(Markdown(content))


@memory_app.command("add")
def memory_add(
    entry: Annotated[str, typer.Argument(help="Memory entry to persist.")],
) -> None:
    """Add a manual memory entry.

    Args:
        entry: CLI argument or option value.

    Returns:
        None: Command result.
    """
    memory = ProjectMemory(Path.cwd())
    if memory.add_entry(entry, kind="manual"):
        console.print("[green]Added memory entry.[/green]")
    else:
        console.print("[yellow]Entry already exists or was empty.[/yellow]")


@memory_app.command("clear")
def memory_clear() -> None:
    """Reset project memory.

    Args:
        None.

    Returns:
        None: Command result.
    """
    memory = ProjectMemory(Path.cwd())
    memory.clear()
    console.print("[green]Project memory reset.[/green]")


@memory_app.command("import")
def memory_import() -> None:
    """Import guidance from common assistant memory files.

    Args:
        None.

    Returns:
        None: Command result.
    """
    memory = ProjectMemory(Path.cwd())
    imported = memory.import_external_guidance()
    if imported:
        console.print("[green]Imported guidance from:[/green]")
        for path in imported:
            console.print(f"  - {path.name}")
        return

    detected = memory.detect_external_guidance_files()
    if detected:
        console.print("[yellow]No new guidance entries to import.[/yellow]")
    else:
        console.print("[yellow]No CLAUDE.md/.cursorrules/.aider.conf.yml/AGENTS.md found.[/yellow]")


class SlashCompleter(Completer):
    """Completer that only triggers on slash commands."""

    def __init__(self, commands: list[str]) -> None:
        self.commands = commands

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterator[Completion]:
        text = document.text_before_cursor.lstrip()
        if not text.startswith("/"):
            return
        for cmd in self.commands:
            if cmd.lower().startswith(text.lower()):
                yield Completion(cmd, start_position=-len(text))


def _build_prompt_session() -> PromptSession:
    history_path = Path.home() / ".copex" / "history"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    bindings = KeyBindings()
    commands = ["/model", "/reasoning", "/models", "/new", "/status", "/tools", "/help"]
    completer = SlashCompleter(commands)

    @bindings.add("enter")
    def _(event) -> None:
        buffer = event.app.current_buffer
        if buffer.document.text.strip():
            buffer.validate_and_handle()
        else:
            buffer.reset()

    @bindings.add("escape", "enter")
    def _(event) -> None:
        event.app.current_buffer.insert_text("\n")

    return PromptSession(
        message="copilot> ",
        history=FileHistory(str(history_path)),
        key_bindings=bindings,
        completer=completer,
        complete_while_typing=True,
        multiline=True,
        prompt_continuation=lambda width, line_number, is_soft_wrap: "... ",
    )


async def _model_picker(current: Model) -> Model | None:
    """Interactive model picker using arrow keys."""
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import Window
    from prompt_toolkit.layout.controls import FormattedTextControl

    models = list(Model)
    selected_idx = models.index(current) if current in models else 0

    kb = KeyBindings()

    @kb.add("up")
    def move_up(event) -> None:
        nonlocal selected_idx
        selected_idx = (selected_idx - 1) % len(models)

    @kb.add("down")
    def move_down(event) -> None:
        nonlocal selected_idx
        selected_idx = (selected_idx + 1) % len(models)

    @kb.add("enter")
    def select(event) -> None:
        event.app.exit(result=models[selected_idx])

    @kb.add("escape")
    @kb.add("c-c")
    def cancel(event) -> None:
        event.app.exit(result=None)

    def get_text():
        lines = [("bold", "Select a model (↑/↓ to navigate, Enter to select, Esc to cancel):\n\n")]
        for i, m in enumerate(models):
            if i == selected_idx:
                lines.append(("class:selected", f"  ▸ {m.value}"))
            else:
                lines.append(("", f"    {m.value}"))
            if m == current:
                lines.append(("class:current", " ← current"))
            lines.append(("", "\n"))
        return lines

    from prompt_toolkit.styles import Style

    style = Style.from_dict(
        {
            "selected": "fg:ansicyan bold",
            "current": "fg:ansiyellow italic",
        }
    )

    app: Application[Model | None] = Application(
        layout=Layout(Window(FormattedTextControl(get_text))),
        key_bindings=kb,
        style=style,
        full_screen=False,
    )

    return await app.run_async()


async def _reasoning_picker(current: ReasoningEffort) -> ReasoningEffort | None:
    """Interactive reasoning effort picker."""
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import Window
    from prompt_toolkit.layout.controls import FormattedTextControl

    efforts = list(ReasoningEffort)
    selected_idx = efforts.index(current) if current in efforts else 0

    kb = KeyBindings()

    @kb.add("up")
    def move_up(event) -> None:
        nonlocal selected_idx
        selected_idx = (selected_idx - 1) % len(efforts)

    @kb.add("down")
    def move_down(event) -> None:
        nonlocal selected_idx
        selected_idx = (selected_idx + 1) % len(efforts)

    @kb.add("enter")
    def select(event) -> None:
        event.app.exit(result=efforts[selected_idx])

    @kb.add("escape")
    @kb.add("c-c")
    def cancel(event) -> None:
        event.app.exit(result=None)

    def get_text():
        lines = [
            (
                "bold",
                "Select reasoning effort (↑/↓ to navigate, Enter to select, Esc to cancel):\n\n",
            )
        ]
        for i, r in enumerate(efforts):
            if i == selected_idx:
                lines.append(("class:selected", f"  ▸ {r.value}"))
            else:
                lines.append(("", f"    {r.value}"))
            if r == current:
                lines.append(("class:current", " ← current"))
            lines.append(("", "\n"))
        return lines

    from prompt_toolkit.styles import Style

    style = Style.from_dict(
        {
            "selected": "fg:ansicyan bold",
            "current": "fg:ansiyellow italic",
        }
    )

    app: Application[ReasoningEffort | None] = Application(
        layout=Layout(Window(FormattedTextControl(get_text))),
        key_bindings=kb,
        style=style,
        full_screen=False,
    )

    return await app.run_async()


@app.command()
def chat(
    prompt: Annotated[
        str | None, typer.Argument(help="Prompt to send (or read from stdin)")
    ] = None,
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model to use")] = None,
    context_budget: Annotated[
        int | None,
        typer.Option("--context-budget", help="Override context window budget in tokens"),
    ] = None,
    reasoning: Annotated[
        ReasoningEffort, typer.Option("--reasoning", "-r", help="Reasoning effort level")
    ] = ReasoningEffort.HIGH,
    max_retries: Annotated[int, typer.Option("--max-retries", help="Maximum retry attempts")] = 5,
    no_stream: Annotated[
        bool, typer.Option("--no-stream", help="Disable streaming output")
    ] = False,
    show_reasoning: Annotated[
        bool, typer.Option("--show-reasoning/--no-reasoning", help="Show model reasoning")
    ] = True,
    config_file: Annotated[
        Path | None, typer.Option("--config", "-c", help="Config file path")
    ] = None,
    raw: Annotated[bool, typer.Option("--raw", help="Output raw text without formatting")] = False,
    ui_theme: Annotated[
        str | None,
        typer.Option("--ui-theme", help="UI theme (default, midnight, mono, sunset, tokyo)"),
    ] = None,
    ui_density: Annotated[
        str | None, typer.Option("--ui-density", help="UI density (compact or extended)")
    ] = None,
    skill_dir: Annotated[
        list[str] | None, typer.Option("--skill-dir", "-S", help="Add skill directory")
    ] = None,
    disable_skill: Annotated[
        list[str] | None, typer.Option("--disable-skill", help="Disable specific skill")
    ] = None,
    no_auto_skills: Annotated[
        bool, typer.Option("--no-auto-skills", help="Disable skill auto-discovery")
    ] = False,
    json_output: Annotated[
        bool, typer.Option("--json", help="Output response as machine-readable JSON")
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet", "-q", help="Minimal output (content only, no panels or formatting)"
        ),
    ] = False,
    # New options
    stdin: Annotated[bool, typer.Option("--stdin", "-i", help="Read prompt from stdin")] = False,
    context: Annotated[
        list[Path] | None,
        typer.Option("--context", "-C", help="Include file(s) as context (can be repeated)"),
    ] = None,
    template: Annotated[
        str | None,
        typer.Option("--template", "-T", help="Jinja2 template for output formatting"),
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write response to file")
    ] = None,
    use_cli: Annotated[
        bool,
        typer.Option("--use-cli", help="Use CLI subprocess instead of SDK (supports all models)"),
    ] = False,
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
    js_repl: Annotated[
        bool, typer.Option("--js-repl", help="Enable persistent JavaScript REPL tool (requires Node.js)")
    ] = False,
    js_repl_node: Annotated[
        str | None,
        typer.Option("--js-repl-node", help="Path to the Node.js executable for --js-repl"),
    ] = None,
    pdf_analyze: Annotated[
        bool, typer.Option("--pdf-analyze", help="Enable PDF analysis tools (requires PyMuPDF)")
    ] = False,
) -> None:
    """Send a prompt to Copilot with automatic retry on errors.

    Args:
        prompt: CLI argument or option value.
        model: CLI argument or option value.
        context_budget: CLI argument or option value.
        reasoning: CLI argument or option value.
        max_retries: CLI argument or option value.
        no_stream: CLI argument or option value.
        show_reasoning: CLI argument or option value.
        config_file: CLI argument or option value.
        raw: CLI argument or option value.
        ui_theme: CLI argument or option value.
        ui_density: CLI argument or option value.
        skill_dir: CLI argument or option value.
        disable_skill: CLI argument or option value.
        no_auto_skills: CLI argument or option value.
        json_output: CLI argument or option value.
        quiet: CLI argument or option value.
        stdin: CLI argument or option value.
        context: CLI argument or option value.
        template: CLI argument or option value.
        output: CLI argument or option value.
        use_cli: CLI argument or option value.
        auto_approve: CLI argument or option value.
        approve: CLI argument or option value.
        dry_run: CLI argument or option value.
        audit: CLI argument or option value.
        js_repl: CLI argument or option value.
        js_repl_node: CLI argument or option value.
        pdf_analyze: CLI argument or option value.

    Returns:
        None: Command result.
    """
    # Load config: explicit flag wins; otherwise auto-load from ~/.config/copex/config.toml if present
    if config_file and config_file.exists():
        config = CopexConfig.from_file(config_file)
    else:
        default_path = CopexConfig.default_path()
        if default_path.exists():
            config = CopexConfig.from_file(default_path)
        else:
            config = CopexConfig()

    model_explicit = option_was_explicit("model")
    reasoning_explicit = option_was_explicit("reasoning")
    effective_model = model or _get_default_model().value

    config.retry.max_retries = max_retries
    config.streaming = not no_stream
    if context_budget is not None:
        config.context_budget = context_budget
    if use_cli:
        config.use_cli = True
    if js_repl or js_repl_node:
        config.js_repl = True
    if js_repl_node:
        config.js_repl_node_path = js_repl_node
    if pdf_analyze:
        config.pdf_analyze = True
    if ui_theme:
        config.ui_theme = ui_theme
    if ui_density:
        config.ui_density = ui_density

    # Skills options
    if skill_dir:
        config.skill_directories.extend(skill_dir)
    if disable_skill:
        config.disabled_skills.extend(disable_skill)
    if no_auto_skills:
        config.auto_discover_skills = False
    try:
        _apply_approval_flags(
            config,
            auto_approve=auto_approve,
            approve=approve,
            dry_run=dry_run,
            audit=audit,
        )
    except typer.BadParameter as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None

    # Handle stdin flag
    if stdin:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
        else:
            console.print("[yellow]Enter prompt (Ctrl+D to submit):[/yellow]")
            prompt = sys.stdin.read().strip()

    # Get prompt from stdin if not provided and -i not used
    if prompt is None and not stdin:
        if sys.stdin.isatty():
            console.print("[yellow]Enter prompt (Ctrl+D to submit):[/yellow]")
        prompt = sys.stdin.read().strip()

    if not prompt:
        console.print("[red]No prompt provided[/red]")
        raise typer.Exit(1) from None

    route = None
    if not model_explicit:
        route = route_model_for_prompt(prompt, client_options=config.to_client_options())
        effective_model = route.model
    try:
        config.model = _resolve_cli_model(effective_model)
    except ValueError:
        console.print(f"[red]Invalid model: {effective_model}[/red]")
        raise typer.Exit(1) from None

    try:
        requested_effort = parse_reasoning_effort(reasoning)
        if requested_effort is None:
            raise ValueError(reasoning)
        if route is not None and not reasoning_explicit:
            requested_effort = route.reasoning_effort
        normalized_effort, warning = normalize_reasoning_effort(config.model, requested_effort)
        if warning:
            console.print(f"[yellow]{warning}[/yellow]")
        config.reasoning_effort = normalized_effort
    except ValueError:
        console.print(f"[red]Invalid reasoning effort: {reasoning}[/red]")
        raise typer.Exit(1) from None

    # Handle context files
    context_content = ""
    if context:
        context_parts = []
        for ctx_path in context:
            if not ctx_path.exists():
                console.print(f"[red]Context file not found: {ctx_path}[/red]")
                raise typer.Exit(1) from None
            try:
                content = ctx_path.read_text()
                # Format with filename header
                context_parts.append(f'<file path="{ctx_path}">\n{content}\n</file>')
            except OSError as e:
                console.print(f"[red]Error reading {ctx_path}: {e}[/red]")
                raise typer.Exit(1) from None

        if context_parts:
            context_content = "\n\n".join(context_parts)
            prompt = f"Context files:\n{context_content}\n\nPrompt: {prompt}"

    asyncio.run(
        _run_chat(
            config,
            prompt,
            show_reasoning,
            raw,
            json_output=json_output,
            quiet=quiet,
            template=template,
            output_path=output,
        )
    )


async def _run_chat(
    config: CopexConfig,
    prompt: str,
    show_reasoning: bool,
    raw: bool,
    *,
    json_output: bool = False,
    quiet: bool = False,
    template: str | None = None,
    output_path: Path | None = None,
) -> None:
    """Run the chat command."""
    client = make_client(config)

    try:
        await client.start()

        if json_output:
            # Machine-readable JSON output (no streaming UI)
            response = await client.send(prompt)
            result: dict[str, Any] = {
                "content": response.content,
                "model": config.model.value,
                "server_model": response.server_model,  # Actual model from assistant.usage
                "retries": response.retries,
            }
            if show_reasoning and response.reasoning:
                result["reasoning"] = response.reasoning
            if response.usage:
                result["usage"] = response.usage
            if response.context_usage:
                result["context_usage"] = response.context_usage
            if response.cost is not None:
                result["cost"] = response.cost
            output_text = json.dumps(result, indent=2)

            if output_path:
                output_path.write_text(output_text)
                console.print(f"[green]✓ Output saved to {output_path}[/green]")
            else:
                print(output_text)

        elif template:
            # Jinja2 template formatting
            try:
                from jinja2 import Template, TemplateError
            except ImportError:
                console.print("[red]Jinja2 not installed. Run: pip install jinja2[/red]")
                raise typer.Exit(1) from None

            response = await client.send(prompt)

            # Build template context
            template_ctx = {
                "content": response.content,
                "reasoning": response.reasoning,
                "model": config.model.value,
                "retries": response.retries,
                "usage": response.usage,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "context_usage": response.context_usage,
                "context_used_tokens": response.context_used_tokens,
                "context_budget_tokens": response.context_budget_tokens,
            }

            try:
                tmpl = Template(template)
                output_text = tmpl.render(**template_ctx)
            except TemplateError as e:
                console.print(f"[red]Template error: {e}[/red]")
                raise typer.Exit(1) from None

            if output_path:
                output_path.write_text(output_text)
                console.print(f"[green]✓ Output saved to {output_path}[/green]")
            else:
                print(output_text)

        elif quiet:
            # Minimal output: content only, no panels
            response = await client.send(prompt)
            output_text = response.content

            if output_path:
                output_path.write_text(output_text)
                console.print(f"[green]✓ Output saved to {output_path}[/green]")
            else:
                print(output_text)

        elif config.streaming and not raw:
            response_content = await _stream_response(client, prompt, show_reasoning)

            # Save to file if requested
            if output_path and response_content:
                output_path.write_text(response_content)
                console.print(f"[green]✓ Output saved to {output_path}[/green]")

        else:
            response = await client.send(prompt)

            if output_path:
                output_path.write_text(response.content)
                console.print(f"[green]✓ Output saved to {output_path}[/green]")
            elif raw:
                print(response.content)
            else:
                if show_reasoning and response.reasoning:
                    console.print(
                        Panel(
                            Markdown(response.reasoning),
                            title="[dim]Reasoning[/dim]",
                            border_style="dim",
                        )
                    )
                console.print(Markdown(response.content))

                if response.retries > 0:
                    console.print(f"\n[dim]Completed with {response.retries} retries[/dim]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
    except Exception as e:  # Catch-all: top-level CLI error handler
        console.print(f"[red]Chat failed ({type(e).__name__}): {e}[/red]")
        console.print("[dim]Tip: retry with --json or --quiet for easier automation.[/dim]")
        raise typer.Exit(1) from None
    finally:
        await client.stop()




@app.command()
def models() -> None:
    """List available models.

    Args:
        None.

    Returns:
        None: Command result.
    """
    console.print("[bold]Available Models:[/bold]\n")
    for model in Model:
        console.print(f"  • {model.value}")


@app.command("render")
def render_command(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Path to JSONL session log file (use - for stdin)",
        ),
    ] = ...,
) -> None:
    """Render JSONL session logs with readable formatting.

    Args:
        input_path: CLI argument or option value.

    Returns:
        None: Command result.
    """
    if str(input_path) == "-":
        render_jsonl(sys.stdin, console)
        return

    if not input_path.exists():
        console.print(f"[red]File not found: {input_path}[/red]")
        raise typer.Exit(1) from None

    with input_path.open("r", encoding="utf-8") as handle:
        render_jsonl(handle, console)


# Skills subcommand group
skills_app = typer.Typer(
    name="skills",
    help="Manage and discover skills.",
    no_args_is_help=True,
)
app.add_typer(skills_app, name="skills")


@skills_app.command("list")
def skills_list(
    skill_dir: Annotated[
        list[str] | None, typer.Option("--skill-dir", "-S", help="Add skill directory")
    ] = None,
    no_auto: Annotated[bool, typer.Option("--no-auto", help="Disable auto-discovery")] = False,
) -> None:
    """List all available skills.

    Args:
        skill_dir: CLI argument or option value.
        no_auto: CLI argument or option value.

    Returns:
        None: Command result.
    """
    from copex.skills import list_skills

    skills = list_skills(
        skill_directories=skill_dir,
        auto_discover=not no_auto,
    )

    if not skills:
        console.print("[yellow]No skills found.[/yellow]")
        console.print("\nSkills are auto-discovered from:")
        console.print("  • .github/skills/ (in repo)")
        console.print("  • .claude/skills/ (in repo)")
        console.print("  • .copex/skills/ (in repo)")
        console.print("  • ~/.config/copex/skills/ (user)")
        return

    console.print("[bold]Available Skills:[/bold]\n")
    for skill in skills:
        source_label = f"[dim]({skill.source})[/dim]"
        desc = f" - {skill.description}" if skill.description else ""
        console.print(f"  • [cyan]{skill.name}[/cyan]{desc} {source_label}")
        console.print(f"    [dim]{skill.path}[/dim]")


@skills_app.command("show")
def skills_show(
    name: Annotated[str, typer.Argument(help="Skill name to show")],
    skill_dir: Annotated[
        list[str] | None, typer.Option("--skill-dir", "-S", help="Add skill directory")
    ] = None,
) -> None:
    """Show the content of a specific skill.

    Args:
        name: CLI argument or option value.
        skill_dir: CLI argument or option value.

    Returns:
        None: Command result.
    """
    from copex.skills import get_skill_content

    content = get_skill_content(name, skill_directories=skill_dir)

    if content is None:
        console.print(f"[red]Skill not found: {name}[/red]")
        raise typer.Exit(1) from None

    console.print(Markdown(content))


@app.command("tui")
def tui(
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model to use")] = None,
    reasoning: Annotated[
        ReasoningEffort, typer.Option("--reasoning", "-r", help="Reasoning effort level")
    ] = ReasoningEffort.HIGH,
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
) -> None:
    """Start the Copex TUI.

    Args:
        model: CLI argument or option value.
        reasoning: CLI argument or option value.
        auto_approve: CLI argument or option value.
        approve: CLI argument or option value.
        dry_run: CLI argument or option value.
        audit: CLI argument or option value.

    Returns:
        None: Command result.
    """
    from copex.tui import run_tui

    effective_model = model or _get_default_model().value
    try:
        model_enum = _resolve_cli_model(effective_model)
        requested_effort = parse_reasoning_effort(reasoning) or ReasoningEffort.HIGH
        normalized_effort, warning = normalize_reasoning_effort(model_enum, requested_effort)
        if warning:
            console.print(f"[yellow]{warning}[/yellow]")

        config = CopexConfig(model=model_enum, reasoning_effort=normalized_effort)
        _apply_approval_flags(
            config,
            auto_approve=auto_approve,
            approve=approve,
            dry_run=dry_run,
            audit=audit,
        )
    except typer.BadParameter as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None

    asyncio.run(run_tui(config))


@app.command()
def init(
    path: Annotated[
        Path, typer.Option("--path", "-p", help="Config file path")
    ] = None,
) -> None:
    """Create a default config file.

    Args:
        path: CLI argument or option value.

    Returns:
        None: Command result.
    """
    import tomli_w

    if path is None:
        path = CopexConfig.default_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "model": Model.CLAUDE_OPUS_4_5.value,
        "reasoning_effort": ReasoningEffort.XHIGH.value,
        "streaming": True,
        "timeout": 300.0,
        "auto_continue": True,
        "continue_prompt": "Keep going",
        "retry": {
            "max_retries": 5,
            "base_delay": 1.0,
            "max_delay": 30.0,
            "exponential_base": 2.0,
            "retry_on_errors": ["500", "502", "503", "504", "Internal Server Error", "rate limit"],
        },
        "approval_mode": "auto-approve",
        "audit": False,
        "js_repl": False,
        "pdf_analyze": False,
        "ui_theme": "default",
        "ui_density": "extended",
    }

    with open(path, "wb") as f:
        tomli_w.dump(config, f)

    console.print(f"[green]Created config at:[/green] {path}")


@app.command()
def interactive(
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model to use")] = None,
    reasoning: Annotated[
        ReasoningEffort, typer.Option("--reasoning", "-r", help="Reasoning effort level")
    ] = ReasoningEffort.HIGH,
    ui_theme: Annotated[
        str | None,
        typer.Option("--ui-theme", help="UI theme (default, midnight, mono, sunset, tokyo)"),
    ] = None,
    ui_density: Annotated[
        str | None, typer.Option("--ui-density", help="UI density (compact or extended)")
    ] = None,
    classic: Annotated[
        bool, typer.Option("--classic", help="Use classic interactive mode (legacy)")
    ] = False,
    skill_dir: Annotated[
        list[str] | None, typer.Option("--skill-dir", "-S", help="Add skill directory")
    ] = None,
    disable_skill: Annotated[
        list[str] | None, typer.Option("--disable-skill", help="Disable specific skill")
    ] = None,
    no_auto_skills: Annotated[
        bool, typer.Option("--no-auto-skills", help="Disable skill auto-discovery")
    ] = False,
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
    js_repl: Annotated[
        bool, typer.Option("--js-repl", help="Enable persistent JavaScript REPL tool (requires Node.js)")
    ] = False,
    js_repl_node: Annotated[
        str | None,
        typer.Option("--js-repl-node", help="Path to the Node.js executable for --js-repl"),
    ] = None,
    pdf_analyze: Annotated[
        bool, typer.Option("--pdf-analyze", help="Enable PDF analysis tools (requires PyMuPDF)")
    ] = False,
) -> None:
    """Start an interactive chat session.

    Args:
        model: CLI argument or option value.
        reasoning: CLI argument or option value.
        ui_theme: CLI argument or option value.
        ui_density: CLI argument or option value.
        classic: CLI argument or option value.
        skill_dir: CLI argument or option value.
        disable_skill: CLI argument or option value.
        no_auto_skills: CLI argument or option value.
        auto_approve: CLI argument or option value.
        approve: CLI argument or option value.
        dry_run: CLI argument or option value.
        audit: CLI argument or option value.
        js_repl: CLI argument or option value.
        js_repl_node: CLI argument or option value.
        pdf_analyze: CLI argument or option value.

    Returns:
        None: Command result.
    """
    effective_model = model or _get_default_model().value
    try:
        model_enum = _resolve_cli_model(effective_model)
        requested_effort = parse_reasoning_effort(reasoning) or ReasoningEffort.HIGH
        normalized_effort, warning = normalize_reasoning_effort(model_enum, requested_effort)
        if warning:
            console.print(f"[yellow]{warning}[/yellow]")

        config = CopexConfig(model=model_enum, reasoning_effort=normalized_effort)
        if ui_theme:
            config.ui_theme = ui_theme
        if ui_density:
            config.ui_density = ui_density
        if js_repl or js_repl_node:
            config.js_repl = True
        if js_repl_node:
            config.js_repl_node_path = js_repl_node
        if pdf_analyze:
            config.pdf_analyze = True

        # Skills options
        if skill_dir:
            config.skill_directories.extend(skill_dir)
        if disable_skill:
            config.disabled_skills.extend(disable_skill)
        if no_auto_skills:
            config.auto_discover_skills = False
        _apply_approval_flags(
            config,
            auto_approve=auto_approve,
            approve=approve,
            dry_run=dry_run,
            audit=audit,
        )
    except typer.BadParameter as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None

    if classic:
        # Use legacy interactive mode
        print_welcome(
            console,
            config.model.value,
            config.reasoning_effort.value,
            theme=config.ui_theme,
            density=config.ui_density,
        )
        asyncio.run(_interactive_loop(config))
    else:
        # Use new beautiful interactive mode
        from copex.interactive import run_interactive

        asyncio.run(run_interactive(config))


async def _interactive_loop(config: CopexConfig) -> None:
    """Run interactive chat loop."""
    client = make_client(config)
    await client.start()
    session = _build_prompt_session()
    show_all_tools = False

    # Create persistent UI for conversation history
    ui = CopexUI(
        console,
        theme=config.ui_theme,
        density=config.ui_density,
        show_all_tools=show_all_tools,
    )

    def show_help() -> None:
        console.print(f"\n[{Theme.MUTED}]Commands:[/{Theme.MUTED}]")
        console.print(
            f"  [{Theme.PRIMARY}]/model <name>[/{Theme.PRIMARY}]     - Change model (e.g., /model gpt-5.1-codex)"
        )
        console.print(
            f"  [{Theme.PRIMARY}]/reasoning <level>[/{Theme.PRIMARY}] - Change reasoning (low, medium, high, xhigh)"
        )
        console.print(
            f"  [{Theme.PRIMARY}]/models[/{Theme.PRIMARY}]            - List available models"
        )
        console.print(
            f"  [{Theme.PRIMARY}]/new[/{Theme.PRIMARY}]               - Start new session"
        )
        console.print(
            f"  [{Theme.PRIMARY}]/status[/{Theme.PRIMARY}]            - Show current settings"
        )
        console.print(
            f"  [{Theme.PRIMARY}]/tools[/{Theme.PRIMARY}]             - Toggle full tool call list"
        )
        console.print(f"  [{Theme.PRIMARY}]/help[/{Theme.PRIMARY}]              - Show this help")
        console.print(f"  [{Theme.PRIMARY}]exit[/{Theme.PRIMARY}]               - Exit\n")

    def show_status() -> None:
        console.print(f"\n[{Theme.MUTED}]Current settings:[/{Theme.MUTED}]")
        console.print(
            f"  Model:     [{Theme.PRIMARY}]{client.config.model.value}[/{Theme.PRIMARY}]"
        )
        console.print(
            f"  Reasoning: [{Theme.PRIMARY}]{client.config.reasoning_effort.value}[/{Theme.PRIMARY}]\n"
        )

    try:
        while True:
            try:
                prompt = await session.prompt_async()
            except EOFError:
                break
            except KeyboardInterrupt:
                break

            prompt = prompt.strip()
            if not prompt:
                continue

            command = prompt.lower()

            if command in {"exit", "quit"}:
                break

            if command in {"new", "/new"}:
                client.new_session()
                # Clear UI history for new session
                ui.state.history = []
                console.print(
                    f"\n[{Theme.SUCCESS}]{Icons.DONE} Started new session[/{Theme.SUCCESS}]\n"
                )
                continue

            if command in {"help", "/help"}:
                show_help()
                continue

            if command in {"status", "/status"}:
                show_status()
                continue

            if command in {"models", "/models"}:
                selected = await _model_picker(client.config.model)
                if selected and selected != client.config.model:
                    client.config.model = selected
                    save_last_model(selected)  # Persist for next run

                    desired_effort = client.config.reasoning_effort
                    # Prompt for reasoning effort if GPT model
                    if selected.value.startswith("gpt-"):
                        new_reasoning = await _reasoning_picker(client.config.reasoning_effort)
                        if new_reasoning:
                            desired_effort = new_reasoning

                    normalized_effort, warning = normalize_reasoning_effort(
                        selected, desired_effort
                    )
                    if warning:
                        console.print(f"[{Theme.WARNING}]{warning}[/{Theme.WARNING}]")
                    client.config.reasoning_effort = normalized_effort

                    client.new_session()
                    # Clear UI history for new session
                    ui.state.history = []
                    console.print(
                        f"\n[{Theme.SUCCESS}]{Icons.DONE} Switched to {selected.value} (new session started)[/{Theme.SUCCESS}]\n"
                    )
                continue

            if command in {"tools", "/tools"}:
                show_all_tools = not show_all_tools
                ui.show_all_tools = show_all_tools
                mode = "all tools" if show_all_tools else "recent tools"
                console.print(f"\n[{Theme.SUCCESS}]{Icons.DONE} Showing {mode}[/{Theme.SUCCESS}]\n")
                continue

            if command.startswith("/model ") or command.startswith("model "):
                parts = prompt.split(maxsplit=1)
                if len(parts) < 2:
                    console.print(f"[{Theme.ERROR}]Usage: /model <model-name>[/{Theme.ERROR}]")
                    continue
                model_name = parts[1].strip()
                try:
                    new_model = _resolve_cli_model(model_name)
                    client.config.model = new_model
                    save_last_model(new_model)  # Persist for next run

                    normalized_effort, warning = normalize_reasoning_effort(
                        new_model, client.config.reasoning_effort
                    )
                    if warning:
                        console.print(f"[{Theme.WARNING}]{warning}[/{Theme.WARNING}]")
                    client.config.reasoning_effort = normalized_effort

                    client.new_session()  # Need new session for model change
                    # Clear UI history for new session
                    ui.state.history = []
                    console.print(
                        f"\n[{Theme.SUCCESS}]{Icons.DONE} Switched to {new_model.value} (new session started)[/{Theme.SUCCESS}]\n"
                    )
                except ValueError:
                    console.print(f"[{Theme.ERROR}]Unknown model: {model_name}[/{Theme.ERROR}]")
                    console.print(
                        f"[{Theme.MUTED}]Use /models to see available models[/{Theme.MUTED}]"
                    )
                continue

            if command.startswith("/reasoning ") or command.startswith("reasoning "):
                parts = prompt.split(maxsplit=1)
                if len(parts) < 2:
                    console.print(f"[{Theme.ERROR}]Usage: /reasoning <level>[/{Theme.ERROR}]")
                    continue
                level = parts[1].strip()
                try:
                    requested = parse_reasoning_effort(level)
                    if requested is None:
                        raise ValueError(level)

                    normalized_effort, warning = normalize_reasoning_effort(
                        client.config.model, requested
                    )
                    if warning:
                        console.print(f"[{Theme.WARNING}]{warning}[/{Theme.WARNING}]")

                    client.config.reasoning_effort = normalized_effort
                    client.new_session()  # Need new session for reasoning change
                    # Clear UI history for new session
                    ui.state.history = []
                    console.print(
                        f"\n[{Theme.SUCCESS}]{Icons.DONE} Switched to {normalized_effort.value} reasoning (new session started)[/{Theme.SUCCESS}]\n"
                    )
                except ValueError:
                    valid = ", ".join(r.value for r in ReasoningEffort)
                    console.print(
                        f"[{Theme.ERROR}]Invalid reasoning level. Valid: {valid}[/{Theme.ERROR}]"
                    )
                continue

            try:
                print_user_prompt(console, prompt)
                await _stream_response_interactive(client, prompt, ui)
            except Exception as e:  # Catch-all: show error and continue interactive loop
                print_error(console, str(e))

    except KeyboardInterrupt:
        console.print(f"\n[{Theme.WARNING}]{Icons.INFO} Goodbye![/{Theme.WARNING}]")
    finally:
        await client.stop()




@app.command("ralph")
def ralph_command(
    prompt: Annotated[str, typer.Argument(help="Task prompt for the Ralph loop")],
    max_iterations: Annotated[
        int, typer.Option("--max-iterations", "-n", help="Maximum iterations")
    ] = 30,
    completion_promise: Annotated[
        str | None, typer.Option("--promise", "-p", help="Completion promise text")
    ] = None,
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model to use")] = None,
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
) -> None:
    """
        Start a Ralph Wiggum loop - iterative AI development.

        The same prompt is fed to the AI repeatedly. The AI sees its previous
        work in conversation history and iteratively improves until complete.

        Example:
            copex ralph "Build a REST API with CRUD and tests" --promise "ALL TESTS PASSING" -n 20

    Args:
        prompt: CLI argument or option value.
        max_iterations: CLI argument or option value.
        completion_promise: CLI argument or option value.
        model: CLI argument or option value.
        reasoning: CLI argument or option value.
        skill_dir: CLI argument or option value.
        disable_skill: CLI argument or option value.
        no_auto_skills: CLI argument or option value.

    Returns:
        None: Command result.
    """
    model_explicit = option_was_explicit("model")
    reasoning_explicit = option_was_explicit("reasoning")
    effective_model = model or _get_default_model().value
    route = None
    if not model_explicit:
        route = route_model_for_prompt(prompt)
        effective_model = route.model
    try:
        model_enum = _resolve_cli_model(effective_model)
        requested_effort = parse_reasoning_effort(reasoning) or ReasoningEffort.HIGH
        if route is not None and not reasoning_explicit:
            requested_effort = route.reasoning_effort
        normalized_effort, warning = normalize_reasoning_effort(model_enum, requested_effort)
        if warning:
            console.print(f"[yellow]{warning}[/yellow]")

        config = CopexConfig(model=model_enum, reasoning_effort=normalized_effort)

        # Skills options
        if skill_dir:
            config.skill_directories.extend(skill_dir)
        if disable_skill:
            config.disabled_skills.extend(disable_skill)
        if no_auto_skills:
            config.auto_discover_skills = False
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None

    # Use new RalphUI header - displayed in _run_ralph
    asyncio.run(_run_ralph(config, prompt, max_iterations, completion_promise))


async def _run_ralph(
    config: CopexConfig,
    prompt: str,
    max_iterations: int,
    completion_promise: str | None,
) -> None:
    """Run Ralph loop with beautiful UI."""
    from copex.ui import RalphUI

    client = make_client(config)
    await client.start()

    ralph_ui = RalphUI(console)
    start_time = time.time()

    # Print header
    ralph_ui.print_header(
        prompt=prompt,
        max_iterations=max_iterations,
        completion_promise=completion_promise,
        model=config.model.value,
        reasoning=config.reasoning_effort.value,
    )

    def on_iteration(iteration: int, response: str) -> None:
        ralph_ui.print_iteration(
            iteration=iteration,
            max_iterations=max_iterations,
            response_preview=response,
            is_complete=False,
        )

    def on_complete(state: RalphState) -> None:
        total_time = time.time() - start_time
        ralph_ui.print_complete(
            iteration=state.iteration,
            completion_reason=state.completion_reason or "unknown",
            total_time=total_time,
        )

    try:
        ralph = RalphWiggum(client)
        await ralph.loop(
            prompt,
            max_iterations=max_iterations,
            completion_promise=completion_promise,
            on_iteration=on_iteration,
            on_complete=on_complete,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Loop cancelled[/yellow]")
    except Exception as e:  # Catch-all: top-level CLI error handler
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None
    finally:
        await client.stop()


@app.command("login")
def login() -> None:
    """Login to GitHub (uses GitHub CLI for authentication).

    Args:
        None.

    Returns:
        None: Command result.
    """
    import subprocess

    # Check for gh CLI
    gh_path = shutil.which("gh")
    if not gh_path:
        console.print("[red]Error: GitHub CLI (gh) not found.[/red]")
        console.print("Install it from: [bold]https://cli.github.com/[/bold]")
        console.print("\nOr with:")
        console.print("  Windows: [bold]winget install GitHub.cli[/bold]")
        console.print("  macOS:   [bold]brew install gh[/bold]")
        console.print("  Linux:   [bold]sudo apt install gh[/bold]")
        raise typer.Exit(1) from None

    console.print("[blue]Opening browser for GitHub authentication...[/blue]\n")

    try:
        result = subprocess.run([gh_path, "auth", "login"], check=False)
        if result.returncode == 0:
            console.print("\n[green]✓ Successfully logged in![/green]")
            console.print("You can now use [bold]copex chat[/bold]")
        else:
            console.print("\n[yellow]Login may have failed. Check status with:[/yellow]")
            console.print("  [bold]copex status[/bold]")
    except (OSError, subprocess.SubprocessError) as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None


@app.command("logout")
def logout() -> None:
    """Logout from GitHub.

    Args:
        None.

    Returns:
        None: Command result.
    """
    import subprocess

    gh_path = shutil.which("gh")
    if not gh_path:
        console.print("[red]Error: GitHub CLI (gh) not found.[/red]")
        raise typer.Exit(1) from None

    try:
        result = subprocess.run([gh_path, "auth", "logout"], check=False)
        if result.returncode == 0:
            console.print("[green]✓ Logged out[/green]")
    except (OSError, subprocess.SubprocessError) as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None


@app.command("status")
def status() -> None:
    """Check Copilot CLI and GitHub authentication status.

    Args:
        None.

    Returns:
        None: Command result.
    """
    import subprocess

    from copex.config import find_copilot_cli

    try:
        cli_path = find_copilot_cli()
    except (AttributeError, TypeError):
        cli_path = None
    gh_path = shutil.which("gh")

    # Get copilot version
    copilot_version = "N/A"
    if cli_path:
        try:
            result = subprocess.run(
                [cli_path, "--version"], capture_output=True, text=True, timeout=5
            )
            copilot_version = result.stdout.strip() or result.stderr.strip()
        except (OSError, subprocess.SubprocessError):
            pass

    console.print(
        Panel(
            f"[bold]Copex Version:[/bold] {__version__}\n"
            f"[bold]Copilot CLI:[/bold] {cli_path or '[red]Not found[/red]'}\n"
            f"[bold]Copilot Version:[/bold] {copilot_version}\n"
            f"[bold]GitHub CLI:[/bold] {gh_path or '[red]Not found[/red]'}",
            title="Copex Status",
            border_style="blue",
        )
    )

    if not cli_path:
        console.print(f"\n[red]{COPILOT_CLI_NOT_FOUND_MESSAGE}[/red]")

    if gh_path:
        console.print("\n[bold]GitHub Auth Status:[/bold]")
        try:
            subprocess.run([gh_path, "auth", "status"], check=False)
        except (OSError, subprocess.SubprocessError) as e:
            console.print(f"[red]Error checking status: {e}[/red]")
    else:
        console.print("\n[yellow]GitHub CLI not found - cannot check auth status[/yellow]")
        console.print("Install: [bold]https://cli.github.com/[/bold]")


@app.command("config")
def config_cmd(
    config_file: Annotated[
        Path | None, typer.Option("--config", "-c", help="Config file path")
    ] = None,
) -> None:
    """Validate and display the current Copex configuration.

    Args:
        config_file: CLI argument or option value.

    Returns:
        None: Command result.
    """
    import warnings as _warnings

    config_path = Path(config_file) if config_file else CopexConfig.default_path()
    source = str(config_path) if config_path.exists() else "defaults"

    with _warnings.catch_warnings(record=True) as caught_warnings:
        _warnings.simplefilter("always")
        try:
            if config_path.exists():
                config = CopexConfig.from_file(config_path)
            else:
                config = CopexConfig()
        except Exception as e:
            console.print(f"[red]Configuration error:[/red] {e}")
            raise typer.Exit(1) from None

    # Display warnings
    if caught_warnings:
        for w in caught_warnings:
            console.print(f"[yellow]⚠ {w.message}[/yellow]")
        console.print()

    # Env var overrides
    env_overrides: list[str] = []
    import os as _os

    if _os.environ.get("COPEX_MODEL"):
        env_overrides.append(f"COPEX_MODEL={_os.environ['COPEX_MODEL']}")
    if _os.environ.get("COPEX_REASONING"):
        env_overrides.append(f"COPEX_REASONING={_os.environ['COPEX_REASONING']}")

    env_line = ""
    if env_overrides:
        env_line = f"\n[bold]Env Overrides:[/bold] {', '.join(env_overrides)}"

    console.print(
        Panel(
            f"[bold]Source:[/bold] {source}\n"
            f"[bold]Model:[/bold] {config.model.value}\n"
            f"[bold]Reasoning:[/bold] {config.reasoning_effort.value}\n"
            f"[bold]Streaming:[/bold] {config.streaming}\n"
            f"[bold]Timeout:[/bold] {config.timeout}s\n"
            f"[bold]Auto-continue:[/bold] {config.auto_continue}\n"
            f"[bold]Max Retries:[/bold] {config.retry.max_retries}\n"
            f"[bold]UI Theme:[/bold] {config.ui_theme}\n"
            f"[bold]UI Density:[/bold] {config.ui_density}"
            f"{env_line}",
            title="Copex Configuration",
            border_style="green",
        )
    )

    if config.skill_directories:
        console.print("\n[bold]Skill Directories:[/bold]")
        for d in config.skill_directories:
            exists = Path(d).exists()
            icon = "[green]✓[/green]" if exists else "[red]✗[/red]"
            console.print(f"  {icon} {d}")

    if not caught_warnings:
        console.print("\n[green]✓ Configuration is valid[/green]")


@app.command("map")
def map_command(
    refresh: Annotated[
        bool,
        typer.Option("--refresh", help="Force a full repo map rebuild"),
    ] = False,
    relevant: Annotated[
        str | None,
        typer.Option("--relevant", help="Show files most relevant to this description"),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", help="Max relevant files to show"),
    ] = 10,
) -> None:
    """Display the repository map or task-relevant files.

    Args:
        refresh: CLI argument or option value.
        relevant: CLI argument or option value.
        limit: CLI argument or option value.

    Returns:
        None: Command result.
    """
    from copex.repo_map import RepoMap

    repo_map = RepoMap(Path.cwd())
    try:
        repo_map.refresh(force=refresh)
    except Exception as exc:
        console.print(f"[red]Error building repo map:[/red] {exc}")
        raise typer.Exit(1) from None

    if relevant:
        console.print(repo_map.render_relevant(relevant, limit=max(1, limit)))
    else:
        console.print(repo_map.render_map())


@app.command("campaign")

@app.command("campaign")
def campaign_command(
    goal: Annotated[str, typer.Option("--goal", "-g", help="Campaign goal description")] = "",
    discover: Annotated[
        str, typer.Option("--discover", "-d", help="Shell command to discover targets")
    ] = "",
    batch_size: Annotated[
        int, typer.Option("--batch-size", "-b", help="Targets per wave/batch")
    ] = 5,
    max_concurrent: Annotated[
        int, typer.Option("--max-concurrent", help="Max concurrent tasks per wave")
    ] = 3,
    parallel: Annotated[
        int | None, typer.Option("--parallel", help="Max parallel tasks per wave (alias)")
    ] = None,
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model to use")] = None,
    reasoning: Annotated[
        ReasoningEffort, typer.Option("--reasoning", "-r", help="Reasoning effort level")
    ] = ReasoningEffort.HIGH,
    timeout: Annotated[
        float, typer.Option("--timeout", help="Per-task timeout in seconds")
    ] = 600.0,
    resume: Annotated[
        bool, typer.Option("--resume", help="Resume a previously interrupted campaign")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Print full task outputs")
    ] = False,
    git_finalize: Annotated[
        bool,
        typer.Option(
            "--git-finalize/--no-git-finalize",
            help="Stage and commit changes after each wave",
        ),
    ] = True,
    git_message: Annotated[
        str | None,
        typer.Option("--git-message", help="Commit message template (wave number appended)"),
    ] = None,
    skills: Annotated[
        list[str] | None,
        typer.Option("--skills", help="Skill directories (.md files) to prepend to system prompt"),
    ] = None,
    exclude_tools: Annotated[
        str | None,
        typer.Option(
            "--exclude-tools",
            help="Comma-separated tool patterns to exclude per task",
        ),
    ] = None,
    state_file: Annotated[
        Path | None,
        typer.Option("--state-file", help="Custom campaign state file path"),
    ] = None,
) -> None:
    """Run a campaign: discover targets, batch them, run fleet waves.

        A campaign orchestrates multi-wave fleet execution:
          1. Run --discover command to find targets
          2. Batch targets into groups of --batch-size
          3. Run each wave as a parallel fleet batch
          4. Report results; state saved to .copex/campaign.json for resume

        
        Examples:
            copex campaign --goal "Add type annotations" --discover "find . -name '*.py'" --batch-size 5
            copex campaign --resume   # resume interrupted campaign

    Args:
        goal: CLI argument or option value.
        discover: CLI argument or option value.
        batch_size: CLI argument or option value.
        max_concurrent: CLI argument or option value.
        parallel: CLI argument or option value.
        model: CLI argument or option value.
        reasoning: CLI argument or option value.
        timeout: CLI argument or option value.
        resume: CLI argument or option value.
        verbose: CLI argument or option value.
        git_finalize: CLI argument or option value.
        git_message: CLI argument or option value.
        skills: CLI argument or option value.
        exclude_tools: CLI argument or option value.
        state_file: CLI argument or option value.

    Returns:
        None: Command result.
    """
    from copex.campaign import (
        WaveStatus,
        create_campaign,
        generate_wave_tasks,
        get_pending_wave_indices,
        load_campaign_state,
        run_discover_command,
        save_campaign_state,
    )

    if parallel is not None:
        max_concurrent = parallel

    # Handle resume
    if resume:
        state = load_campaign_state(state_file)
        if state is None:
            console.print("[red]No campaign state found to resume.[/red]")
            console.print("[dim]Run a campaign first, or specify --state-file[/dim]")
            raise typer.Exit(1) from None
        console.print(
            Panel(
                f"[bold]Resuming campaign[/bold]\n"
                f"Goal: {state.goal}\n"
                f"Targets: {len(state.all_targets)} • "
                f"Waves: {len(state.waves)} • "
                f"Pending: {len(get_pending_wave_indices(state))}",
                title="🔄 Campaign Resume",
                border_style="yellow",
            )
        )
    else:
        if not goal:
            console.print("[red]Error: --goal is required for a new campaign[/red]")
            raise typer.Exit(1) from None
        if not discover:
            console.print("[red]Error: --discover is required for a new campaign[/red]")
            raise typer.Exit(1) from None

        # Step 1: Discovery
        console.print(f"[blue]🔍 Running discovery:[/blue] {discover}")
        try:
            targets = run_discover_command(discover)
        except RuntimeError as exc:
            console.print(f"[red]Discovery failed: {exc}[/red]")
            raise typer.Exit(1) from None

        if not targets:
            console.print("[yellow]No targets discovered. Nothing to do.[/yellow]")
            raise typer.Exit(0)

        console.print(f"[green]Found {len(targets)} target(s)[/green]")

        # Step 2: Create campaign
        state = create_campaign(goal, discover, batch_size, targets)
        save_campaign_state(state, state_file)
        console.print(
            Panel(
                f"[bold]Campaign created[/bold]\n"
                f"Goal: {goal}\n"
                f"Targets: {len(targets)} • "
                f"Waves: {len(state.waves)} • "
                f"Batch size: {batch_size}",
                title="🚀 Campaign",
                border_style="blue",
            )
        )

    # Step 3: Configure model
    effective_model = model or _get_default_model().value
    try:
        model_enum = _resolve_cli_model(effective_model)
        requested_effort = parse_reasoning_effort(reasoning) or ReasoningEffort.HIGH
        normalized_effort, warning = normalize_reasoning_effort(model_enum, requested_effort)
        if warning:
            console.print(f"[yellow]{warning}[/yellow]")
        config = CopexConfig(model=model_enum, reasoning_effort=normalized_effort)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None

    # Step 4: Run waves sequentially (single asyncio.run to avoid nested loop issues)
    async def _run_campaign_waves() -> tuple[int, int]:
        """Run all pending waves and return (succeeded, failed) counts."""
        _total_succeeded = 0
        _total_failed = 0
        for wave_idx in pending:
            wave = state.waves[wave_idx]
            wave.status = WaveStatus.RUNNING
            save_campaign_state(state, state_file)

            console.print(
                f"\n[bold blue]━━━ Wave {wave_idx + 1}/{len(state.waves)} "
                f"({len(wave.targets)} targets) ━━━[/bold blue]"
            )

            # Generate fleet tasks for this wave
            wave_tasks = generate_wave_tasks(state.goal, wave.targets, wave_idx)

            wave_git_msg = git_message or f"campaign wave {wave_idx + 1}: {state.goal[:50]}"

            try:
                await _run_fleet(
                    config=config,
                    prompts=[],
                    file=None,
                    config_file=None,
                    max_concurrent=max_concurrent,
                    fail_fast=False,
                    shared_context=None,
                    timeout=timeout,
                    verbose=verbose,
                    output_dir=None,
                    artifact_path=None,
                    git_finalize=git_finalize,
                    git_message=wave_git_msg,
                    skills=skills or [],
                    exclude_tools=_parse_exclude_tools(exclude_tools),
                    tasks_override=wave_tasks,
                )
                wave.status = WaveStatus.COMPLETED
                wave.succeeded = len(wave.targets)
                _total_succeeded += wave.succeeded
            except (SystemExit, typer.Exit):
                # Fleet exits with code 1 on failures - mark wave but continue
                wave.status = WaveStatus.COMPLETED
                wave.succeeded = len(wave.targets)  # approximate
                _total_succeeded += wave.succeeded
            except Exception as exc:
                wave.status = WaveStatus.FAILED
                wave.error = str(exc)
                wave.failed = len(wave.targets)
                _total_failed += wave.failed
                console.print(f"[red]Wave {wave_idx + 1} failed: {exc}[/red]")
                logger.debug("Campaign wave %d failed", wave_idx + 1, exc_info=True)

            save_campaign_state(state, state_file)

        return _total_succeeded, _total_failed

    pending = get_pending_wave_indices(state)
    campaign_start = time.time()
    total_succeeded, total_failed = asyncio.run(_run_campaign_waves())

    # Step 5: Final summary
    campaign_duration = time.time() - campaign_start
    state.total_duration_seconds = campaign_duration
    remaining = get_pending_wave_indices(state)
    state.status = "completed" if not remaining else "failed"
    save_campaign_state(state, state_file)

    completed_waves = sum(1 for w in state.waves if w.status == WaveStatus.COMPLETED)
    failed_waves = sum(1 for w in state.waves if w.status == WaveStatus.FAILED)

    summary_lines = [
        f"[bold]Waves:[/bold] {completed_waves} completed, {failed_waves} failed, "
        f"{len(remaining)} remaining",
        f"[bold]Targets:[/bold] {len(state.all_targets)} total",
        f"[bold]Duration:[/bold] {_format_duration(campaign_duration)}",
        f"[bold]State:[/bold] {state_file or '.copex/campaign.json'}",
    ]

    if remaining:
        summary_lines.append("\n[yellow]Campaign incomplete — use --resume to continue[/yellow]")

    console.print(
        Panel(
            "\n".join(summary_lines),
            title="📊 Campaign Summary",
            border_style="green" if not remaining else "yellow",
        )
    )

    if failed_waves > 0 or remaining:
        raise typer.Exit(1) from None


# ──────────────────────────────────────────────────────────────────────
# copex agent
# ──────────────────────────────────────────────────────────────────────


@app.command("agent")
def agent_command(
    prompt: Annotated[
        str | None, typer.Argument(help="Prompt for the agent")
    ] = None,
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model to use")] = None,
    reasoning: Annotated[
        ReasoningEffort, typer.Option("--reasoning", "-r", help="Reasoning effort level")
    ] = ReasoningEffort.HIGH,
    max_turns: Annotated[
        int, typer.Option("--max-turns", "-t", help="Maximum agent turns")
    ] = 10,
    json_output: Annotated[
        bool, typer.Option("--json", help="Output JSON Lines (one JSON object per turn)")
    ] = False,
    use_cli: Annotated[
        bool,
        typer.Option("--use-cli", help="Use CLI subprocess instead of SDK (supports all models)"),
    ] = False,
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
) -> None:
    """Run an agent loop: prompt → tool calls → respond → repeat.

        The agent sends the prompt, observes tool calls, and continues
        until the model stops calling tools or max-turns is reached.
        Designed for machine consumption via --json (JSON Lines output).

        Examples:
            copex agent "Fix the failing test in src/auth.py" --max-turns 5
            copex agent "Refactor the logger module" --json | jq .
            echo "Add type hints" | copex agent --stdin --json

    Args:
        prompt: CLI argument or option value.
        model: CLI argument or option value.
        reasoning: CLI argument or option value.
        max_turns: CLI argument or option value.
        json_output: CLI argument or option value.
        use_cli: CLI argument or option value.
        config_file: CLI argument or option value.
        stdin: CLI argument or option value.
        auto_approve: CLI argument or option value.
        approve: CLI argument or option value.
        dry_run: CLI argument or option value.
        audit: CLI argument or option value.

    Returns:
        None: Command result.
    """
    # Load config
    if config_file and config_file.exists():
        config = CopexConfig.from_file(config_file)
    else:
        default_path = CopexConfig.default_path()
        config = CopexConfig.from_file(default_path) if default_path.exists() else CopexConfig()

    model_explicit = option_was_explicit("model")
    reasoning_explicit = option_was_explicit("reasoning")
    effective_model = model or _get_default_model().value

    if use_cli:
        config.use_cli = True

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

    route = None
    if not model_explicit:
        route = route_model_for_prompt(prompt, client_options=config.to_client_options())
        effective_model = route.model
    try:
        config.model = _resolve_cli_model(effective_model)
    except ValueError:
        console.print(f"[red]Invalid model: {effective_model}[/red]")
        raise typer.Exit(1) from None

    try:
        requested_effort = parse_reasoning_effort(reasoning)
        if requested_effort is None:
            raise ValueError(reasoning)
        if route is not None and not reasoning_explicit:
            requested_effort = route.reasoning_effort
        normalized_effort, warning = normalize_reasoning_effort(config.model, requested_effort)
        if warning and not json_output:
            console.print(f"[yellow]{warning}[/yellow]")
        config.reasoning_effort = normalized_effort
    except ValueError:
        console.print(f"[red]Invalid reasoning effort: {reasoning}[/red]")
        raise typer.Exit(1) from None

    asyncio.run(_run_agent(config, prompt, max_turns=max_turns, json_output=json_output))


def _is_squad_worker_json_mode(*, json_output: bool) -> bool:
    if not json_output:
        return False
    return os.environ.get("COPEX_SQUAD_WORKER", "").strip().lower() in {"1", "true", "yes", "on"}


def _configure_worker_logging_to_stderr() -> None:
    for logger_name in ("", "copex"):
        logger_obj = logging.getLogger(logger_name)
        for handler in logger_obj.handlers:
            set_stream = getattr(handler, "setStream", None)
            if callable(set_stream):
                try:
                    set_stream(sys.stderr)
                except (AttributeError, ValueError):
                    continue


def _format_framed_json(payload: dict[str, Any]) -> str:
    body = json.dumps(payload, ensure_ascii=False)
    body_len = len(body.encode("utf-8"))
    return f"Content-Length: {body_len}\r\n\r\n{body}"


def _emit_json_payload(payload: dict[str, Any], *, framed: bool) -> None:
    if framed:
        sys.stdout.write(_format_framed_json(payload))
        sys.stdout.flush()
        return
    print(json.dumps(payload, ensure_ascii=False), flush=True)


async def _run_agent(
    config: CopexConfig,
    prompt: str,
    *,
    max_turns: int = 10,
    json_output: bool = False,
) -> None:
    """Run the agent loop."""
    from copex.agent import AgentSession

    client = make_client(config)
    worker_json_mode = _is_squad_worker_json_mode(json_output=json_output)
    if worker_json_mode:
        _configure_worker_logging_to_stderr()

    try:
        session = AgentSession(client, max_turns=max_turns)
        async with session:
            if json_output:
                async for turn in session.run_streaming(prompt):
                    _emit_json_payload(turn.to_dict(), framed=worker_json_mode)
            else:
                result = await session.run(prompt)
                for turn in result.turns:
                    _print_agent_turn(turn)
                # Summary
                console.print()
                stop_style = "green" if result.stop_reason == "end_turn" else "yellow"
                console.print(
                    Panel(
                        f"[bold]{result.total_turns}[/bold] turns · "
                        f"[bold]{result.total_duration_ms / 1000:.1f}s[/bold] · "
                        f"stop: [{stop_style}]{result.stop_reason}[/{stop_style}]",
                        title="[bold cyan]Agent Complete[/bold cyan]",
                        border_style="cyan",
                        expand=False,
                    )
                )
                if result.error:
                    console.print(f"[red]Error: {result.error}[/red]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Agent interrupted[/yellow]")
    except Exception as e:
        if json_output:
            error_obj = {"turn": 0, "content": "", "tool_calls": [], "stop_reason": "error", "error": str(e)}
            _emit_json_payload(error_obj, framed=worker_json_mode)
        else:
            console.print(f"[red]Agent error: {e}[/red]")
        raise typer.Exit(1) from None


def _print_agent_turn(turn: Any) -> None:
    """Render a single agent turn with Rich panels."""
    title_style = "red" if turn.stop_reason == "error" else "cyan"
    title = f"Turn {turn.turn}"

    parts: list[str] = []

    if turn.tool_calls:
        tool_lines = []
        for tc in turn.tool_calls:
            name = tc.get("name", "unknown") if isinstance(tc, dict) else getattr(tc, "name", "unknown")
            success = tc.get("success") if isinstance(tc, dict) else getattr(tc, "success", None)
            duration = tc.get("duration") if isinstance(tc, dict) else getattr(tc, "duration", None)
            status = "✓" if success is not False else "✗"
            dur = f" ({duration:.0f}ms)" if duration else ""
            tool_lines.append(f"  {status} {name}{dur}")
        parts.append("[bold]Tools:[/bold]\n" + "\n".join(tool_lines))

    if turn.content:
        # Truncate long content for display
        display_content = turn.content[:2000]
        if len(turn.content) > 2000:
            display_content += f"\n... ({len(turn.content)} chars total)"
        parts.append(display_content)

    if turn.error:
        parts.append(f"[red]Error: {turn.error}[/red]")

    body = "\n\n".join(parts) if parts else "[dim]No content[/dim]"

    console.print(
        Panel(
            body,
            title=f"[bold {title_style}]{title}[/bold {title_style}]",
            subtitle=f"[dim]{turn.duration_ms / 1000:.1f}s[/dim]",
            border_style=title_style,
        )
    )




@app.command("edit")
def edit_command(
    edit_file: Annotated[
        Path | None,
        typer.Argument(help="Optional file containing structured edit blocks"),
    ] = None,
    verify: Annotated[
        bool,
        typer.Option("--verify/--no-verify", help="Run syntax/lint/type verification"),
    ] = True,
) -> None:
    """Apply structured edits (unified diff, SEARCH/REPLACE, whole-file blocks).

    Args:
        edit_file: CLI argument or option value.
        verify: CLI argument or option value.

    Returns:
        None: Command result.
    """
    if edit_file is not None:
        if not edit_file.exists():
            console.print(f"[red]Edit file not found: {edit_file}[/red]")
            raise typer.Exit(1) from None
        edit_text = edit_file.read_text(encoding="utf-8")
    elif not sys.stdin.isatty():
        edit_text = sys.stdin.read()
    else:
        console.print("[red]Provide edit content via stdin or a file path.[/red]")
        raise typer.Exit(1) from None

    if not edit_text.strip():
        console.print("[red]No edit content provided.[/red]")
        raise typer.Exit(1) from None

    try:
        result = apply_edit_text(edit_text, root=Path.cwd(), verify=verify)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None

    if result.applied_files:
        console.print(f"[green]Applied edits to {len(result.applied_files)} file(s).[/green]")
        for file_path in result.applied_files:
            console.print(f"  [green]✓[/green] {file_path}")
    else:
        console.print("[yellow]No file changes were applied.[/yellow]")

    if result.undo_batch_id:
        console.print(f"[dim]Undo batch: {result.undo_batch_id}[/dim]")

    if result.verification is not None:
        if result.verification.ok:
            console.print("[green]Verification passed.[/green]")
        else:
            console.print("[red]Verification failed.[/red]")
            for check in result.verification.checks:
                if check.ran and not check.success:
                    if check.output:
                        console.print(check.output)

    if result.failed_files:
        for file_path, error in result.failed_files.items():
            console.print(f"[red]{file_path}: {error}[/red]")
        raise typer.Exit(1) from None

    if result.verification is not None and not result.verification.ok:
        raise typer.Exit(1) from None


@app.command("undo")
def undo_command(
    list_history: Annotated[bool, typer.Option("--list", help="List undo history")] = False,
) -> None:
    """Undo structured edits from the latest (or listed) undo batch.

    Args:
        list_history: CLI argument or option value.

    Returns:
        None: Command result.
    """
    if list_history:
        history = list_undo_history(Path.cwd())
        if not history:
            console.print("[yellow]No undo history found.[/yellow]")
            return
        for item in history:
            created = item.created_at or "unknown time"
            console.print(f"{item.batch_id}  ({item.file_count} files)  {created}")
        return

    try:
        result = undo_last_edit_batch(Path.cwd())
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None

    console.print(f"[green]Restored {len(result.restored_files)} file(s).[/green]")
    for file_path in result.restored_files:
        console.print(f"  [green]↺[/green] {file_path}")
    console.print(f"[dim]From undo batch: {result.batch_id}[/dim]")


@app.command("completions")
def completions_command(
    shell: Annotated[
        str, typer.Argument(help="Shell to generate completions for (bash, zsh, fish)")
    ],
) -> None:
    """Generate shell completion scripts.

        Examples:
            copex completions bash >> ~/.bashrc
            copex completions zsh >> ~/.zshrc
            copex completions fish > ~/.config/fish/completions/copex.fish

    Args:
        shell: CLI argument or option value.

    Returns:
        None: Command result.
    """
    shell = shell.lower()

    if shell == "bash":
        print(BASH_COMPLETION)
    elif shell == "zsh":
        print(ZSH_COMPLETION)
    elif shell == "fish":
        print(FISH_COMPLETION)
    else:
        console.print(f"[red]Unknown shell: {shell}[/red]")
        console.print("Supported shells: bash, zsh, fish")
        raise typer.Exit(1) from None


# ──────────────────────────────────────────────────────────────────────
# copex stats
# ──────────────────────────────────────────────────────────────────────

@app.command("stats")
def stats_command() -> None:
    """Show statistics for recent copex runs.

        Displays: last model, reasoning effort, token usage, estimated cost,
        total runs & tokens today.

    Args:
        None.

    Returns:
        None: Command result.
    """
    from copex.stats import StatsTracker, load_state

    tracker = StatsTracker()
    state = load_state()
    last = tracker.last_run()

    if not last and not state.get("last_run"):
        console.print("[yellow]No copex runs recorded yet.[/yellow]")
        console.print("Run a command first, then check back here.")
        return

    last = last or state.get("last_run", {})

    # ── Last run ──
    console.print()
    console.print(Panel("[bold]Last Copex Run[/bold]", style="cyan", expand=False))

    def _val(label: str, value: str, color: str = "white") -> None:
        console.print(f"  [dim]{label:<22}[/dim] [{color}]{value}[/{color}]")

    _val("Model", last.get("model", "unknown"), "green")
    _val("Reasoning effort", last.get("reasoning_effort", "—"), "green")
    _val("Command", last.get("command", "—"), "blue")

    prompt_tok = last.get("prompt_tokens", 0)
    comp_tok = last.get("completion_tokens", 0)
    total_tok = last.get("total_tokens", 0)
    _val("Prompt tokens", f"{prompt_tok:,}")
    _val("Completion tokens", f"{comp_tok:,}")
    _val("Total tokens", f"{total_tok:,}", "yellow")

    cost = last.get("estimated_cost_usd", 0.0)
    _val("Estimated cost", f"${cost:.4f}", "yellow")

    dur = last.get("duration_ms", 0)
    if dur:
        if dur > 60_000:
            _val("Duration", f"{dur / 1000:.1f}s ({dur / 60_000:.1f}m)")
        else:
            _val("Duration", f"{dur / 1000:.1f}s")

    ts = last.get("timestamp", "")
    if ts:
        _val("Timestamp", ts, "dim")

    success = last.get("success", True)
    _val("Status", "✓ success" if success else "✗ failed", "green" if success else "red")

    # ── Today's summary ──
    today_runs = tracker.runs_today()
    console.print()
    console.print(Panel("[bold]Today's Summary[/bold]", style="cyan", expand=False))
    _val("Total runs", str(len(today_runs)), "blue")
    today_tokens = sum(r.get("total_tokens", 0) for r in today_runs)
    _val("Total tokens", f"{today_tokens:,}", "yellow")
    today_cost = sum(r.get("estimated_cost_usd", 0.0) for r in today_runs)
    _val("Total est. cost", f"${today_cost:.4f}", "yellow")
    successes = sum(1 for r in today_runs if r.get("success", True))
    _val("Success rate", f"{successes}/{len(today_runs)}", "green" if successes == len(today_runs) else "red")
    console.print()


# ──────────────────────────────────────────────────────────────────────
# copex audit
# ──────────────────────────────────────────────────────────────────────

@app.command("audit")
def audit_command(
    action: Annotated[
        str, typer.Argument(help="Action to perform (currently only: show)")
    ] = "show",
    last: Annotated[
        int, typer.Option("--last", help="Show the last N audit entries")
    ] = 10,
    file: Annotated[
        str | None, typer.Option("--file", help="Filter by file path")
    ] = None,
) -> None:
    """Show approval audit log entries.

    Args:
        action: CLI argument or option value.
        last: CLI argument or option value.
        file: CLI argument or option value.

    Returns:
        None: Command result.
    """
    if action.lower() != "show":
        console.print(f"[red]Unknown audit action: {action}[/red]")
        console.print("[red]Usage: copex audit show [--last N] [--file PATH][/red]")
        raise typer.Exit(1) from None

    entries = AuditLogger().query(last=max(0, last), file=file)
    if not entries:
        console.print("[yellow]No audit entries found.[/yellow]")
        return

    from rich.table import Table

    table = Table(title="Approval Audit Log", expand=True)
    table.add_column("Timestamp", style="dim", no_wrap=True)
    table.add_column("Mode", style="cyan", no_wrap=True)
    table.add_column("Model", style="magenta", no_wrap=True)
    table.add_column("File", style="white")
    table.add_column("Summary", style="green")
    table.add_column("Risk", style="yellow")
    for entry in entries:
        table.add_row(
            entry.timestamp,
            entry.mode,
            entry.model,
            entry.file,
            entry.diff_summary,
            ", ".join(entry.risk_flags) if entry.risk_flags else "low",
        )
    console.print(table)


# ──────────────────────────────────────────────────────────────────────
# copex diff
# ──────────────────────────────────────────────────────────────────────

@app.command("diff")
def diff_command(
    full: Annotated[bool, typer.Option("--full", help="Show full diff instead of stat summary")] = False,
) -> None:
    """Show what changed since the last copex run started.

        Uses git to compare HEAD against the commit recorded when copex last ran.

    Args:
        full: CLI argument or option value.

    Returns:
        None: Command result.
    """

    from copex.stats import load_start_commit, load_state

    state = load_state()
    start_commit = load_start_commit()

    # Check we're in a git repo
    try:
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        console.print("[red]Not inside a git repository.[/red]")
        raise typer.Exit(1) from None

    if not start_commit:
        console.print("[yellow]No start commit recorded.[/yellow]")
        console.print("copex records HEAD at the start of each run.")
        console.print("Run a copex command first, then use [bold]copex diff[/bold].")
        return

    # Current HEAD
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()

    if head == start_commit:
        console.print("[green]No changes since last copex run.[/green]")
        return

    console.print()
    start_short = start_commit[:8]
    head_short = head[:8]
    console.print(
        Panel(
            f"[bold]Changes since copex run[/bold]  [dim]{start_short}..{head_short}[/dim]",
            style="cyan",
            expand=False,
        )
    )

    start_time = state.get("last_start_time", "")
    if start_time:
        console.print(f"  [dim]Run started:[/dim] {start_time}")

    # --stat summary
    diff_range = f"{start_commit}..HEAD"
    stat_result = subprocess.run(
        ["git", "diff", "--stat", diff_range],
        capture_output=True, text=True, check=False,
    )
    if stat_result.stdout.strip():
        console.print()
        console.print("[bold]Files changed:[/bold]")
        console.print(stat_result.stdout.rstrip())

    # Detailed numbers
    numstat = subprocess.run(
        ["git", "diff", "--numstat", diff_range],
        capture_output=True, text=True, check=False,
    )
    added_lines = 0
    removed_lines = 0
    files_modified = 0
    for line in numstat.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            files_modified += 1
            try:
                added_lines += int(parts[0])
            except ValueError:
                pass
            try:
                removed_lines += int(parts[1])
            except ValueError:
                pass

    # New files (added since start)
    new_files = subprocess.run(
        ["git", "diff", "--diff-filter=A", "--name-only", diff_range],
        capture_output=True, text=True, check=False,
    )
    new_file_list = [f for f in new_files.stdout.strip().splitlines() if f]

    console.print()
    console.print(f"  [dim]Files modified:[/dim]  [white]{files_modified}[/white]")
    console.print(f"  [dim]Files added:[/dim]    [green]{len(new_file_list)}[/green]")
    console.print(f"  [dim]Lines added:[/dim]    [green]+{added_lines:,}[/green]")
    console.print(f"  [dim]Lines removed:[/dim]  [red]-{removed_lines:,}[/red]")

    if new_file_list:
        console.print()
        console.print("[bold]New files:[/bold]")
        for nf in new_file_list:
            console.print(f"  [green]+[/green] {nf}")

    # Full diff
    if full:
        console.print()
        console.print(Panel("[bold]Full diff[/bold]", style="cyan", expand=False))
        full_diff = subprocess.run(
            ["git", "diff", diff_range],
            capture_output=True, text=True, check=False,
        )
        if full_diff.stdout:
            console.print(full_diff.stdout)
        else:
            console.print("[dim]No diff output.[/dim]")

    console.print()


if __name__ == "__main__":
    app()
