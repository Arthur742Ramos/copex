"""CLI interface for Copex."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

from copex import __version__
from copex.client import Copex, StreamChunk
from copex.config import CopexConfig, load_last_model, save_last_model
from copex.models import Model, ReasoningEffort, normalize_reasoning_effort, parse_reasoning_effort
from copex.plan import Plan, PlanExecutor, PlanState, PlanStep, StepStatus
from copex.ralph import RalphState, RalphWiggum
from copex.ui import (
    ActivityType,
    CopexUI,
    Icons,
    Theme,
    ToolCallInfo,
    print_error,
    print_retry,
    print_user_prompt,
    print_welcome,
)

# Effective default: last used model or claude-opus-4.5
_DEFAULT_MODEL = load_last_model() or Model.CLAUDE_OPUS_4_5

app = typer.Typer(
    name="copex",
    help="Copilot Extended - Resilient wrapper with auto-retry and Ralph Wiggum loops.",
    no_args_is_help=False,
    invoke_without_command=True,
)
console = Console()


# Shell completion scripts
BASH_COMPLETION = '''
_copex_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    opts="chat plan ralph fleet interactive models skills status config init login logout tui completions"

    case "${prev}" in
        -m|--model)
            local models="claude-opus-4.5 claude-sonnet-4.1 gpt-5.2-codex gpt-5.1-codex o3 o3-mini o1 o1-mini"
            COMPREPLY=( $(compgen -W "${models}" -- ${cur}) )
            return 0
            ;;
        -r|--reasoning)
            COMPREPLY=( $(compgen -W "low medium high xhigh" -- ${cur}) )
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
'''

ZSH_COMPLETION = '''
#compdef copex

_copex() {
    local -a commands models reasoning_levels
    commands=(
        'chat:Send a prompt to Copilot'
        'plan:Generate and execute step-by-step plans'
        'ralph:Start a Ralph Wiggum loop'
        'fleet:Run multiple tasks in parallel'
        'interactive:Start interactive chat session'
        'models:List available models'
        'skills:Manage skills'
        'status:Check Copilot status'
        'config:Show configuration'
        'init:Create default config'
        'login:Login to GitHub'
        'logout:Logout from GitHub'
        'tui:Start the TUI'
        'completions:Generate shell completions'
    )
    models=(
        'claude-opus-4.5'
        'claude-sonnet-4.1'
        'gpt-5.2-codex'
        'gpt-5.1-codex'
        'o3'
        'o3-mini'
        'o1'
        'o1-mini'
    )
    reasoning_levels=('low' 'medium' 'high' 'xhigh')

    _arguments -C \\
        '1:command:->command' \\
        '*::arg:->args'

    case "$state" in
        command)
            _describe 'command' commands
            ;;
        args)
            case "$words[1]" in
                chat|plan|ralph|fleet|interactive)
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
'''

FISH_COMPLETION = '''
# copex fish completion

set -l commands chat plan ralph fleet interactive models skills status config init login logout tui completions
set -l models claude-opus-4.5 claude-sonnet-4.1 gpt-5.2-codex gpt-5.1-codex o3 o3-mini o1 o1-mini
set -l reasoning low medium high xhigh

complete -c copex -f
complete -c copex -n "not __fish_seen_subcommand_from $commands" -a "$commands"

# Model option
complete -c copex -s m -l model -d "Model to use" -xa "$models"
# Reasoning option
complete -c copex -s r -l reasoning -d "Reasoning effort" -xa "$reasoning"

# Completions subcommand
complete -c copex -n "__fish_seen_subcommand_from completions" -a "bash zsh fish"
'''


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"copex version {__version__}")
        raise typer.Exit()


def model_callback(value: str | None) -> Model | None:
    """Validate model name."""
    if value is None:
        return None
    try:
        return Model(value)
    except ValueError:
        valid = ", ".join(m.value for m in Model)
        raise typer.BadParameter(f"Invalid model. Valid: {valid}")


def reasoning_callback(value: str | None) -> ReasoningEffort | None:
    """Validate reasoning effort."""
    if value is None:
        return None
    try:
        return ReasoningEffort(value)
    except ValueError:
        valid = ", ".join(r.value for r in ReasoningEffort)
        raise typer.BadParameter(f"Invalid reasoning effort. Valid: {valid}")


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
        str, typer.Option("--reasoning", "-r", help="Reasoning effort level")
    ] = ReasoningEffort.HIGH.value,
    prompt: Annotated[
        str | None, typer.Option("--prompt", "-p", help="Execute a prompt in non-interactive mode")
    ] = None,
    non_interactive: Annotated[
        bool, typer.Option("--non-interactive", "-s", help="Non-interactive mode (exit after completion)")
    ] = False,
    allow_all: Annotated[
        bool, typer.Option("--allow-all", help="Enable all permissions (tools, paths, URLs)")
    ] = False,
) -> None:
    """Copilot Extended - Resilient wrapper with auto-retry and Ralph Wiggum loops."""
    if ctx.invoked_subcommand is None:
        effective_model = model or _DEFAULT_MODEL.value
        
        # If -p/--prompt provided, run in non-interactive/chat mode
        if prompt is not None:
            try:
                config = CopexConfig()
                config.model = Model(effective_model)
                requested_effort = parse_reasoning_effort(reasoning)
                if requested_effort is None:
                    requested_effort = ReasoningEffort.HIGH
                normalized_effort, warning = normalize_reasoning_effort(config.model, requested_effort)
                if warning:
                    console.print(f"[yellow]{warning}[/yellow]")
                config.reasoning_effort = normalized_effort
                
                # Run chat with the prompt
                asyncio.run(_run_chat(config, prompt, show_reasoning=True, raw=non_interactive))
            except ValueError as e:
                console.print(f"[red]Invalid model: {effective_model}[/red]")
                raise typer.Exit(1)
        else:
            # No prompt provided - launch interactive mode
            interactive(model=effective_model, reasoning=reasoning)


class SlashCompleter(Completer):
    """Completer that only triggers on slash commands."""

    def __init__(self, commands: list[str]) -> None:
        self.commands = commands

    def get_completions(self, document, complete_event):
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
        Optional[str], typer.Argument(help="Prompt to send (or read from stdin)")
    ] = None,
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model to use")] = None,
    reasoning: Annotated[
        str, typer.Option("--reasoning", "-r", help="Reasoning effort level")
    ] = ReasoningEffort.HIGH.value,
    max_retries: Annotated[int, typer.Option("--max-retries", help="Maximum retry attempts")] = 5,
    no_stream: Annotated[
        bool, typer.Option("--no-stream", help="Disable streaming output")
    ] = False,
    show_reasoning: Annotated[
        bool, typer.Option("--show-reasoning/--no-reasoning", help="Show model reasoning")
    ] = True,
    config_file: Annotated[
        Optional[Path], typer.Option("--config", "-c", help="Config file path")
    ] = None,
    raw: Annotated[bool, typer.Option("--raw", help="Output raw text without formatting")] = False,
    ui_theme: Annotated[
        Optional[str], typer.Option("--ui-theme", help="UI theme (default, midnight, mono, sunset, tokyo)")
    ] = None,
    ui_density: Annotated[
        Optional[str], typer.Option("--ui-density", help="UI density (compact or extended)")
    ] = None,
    skill_dir: Annotated[
        Optional[list[str]], typer.Option("--skill-dir", "-S", help="Add skill directory")
    ] = None,
    disable_skill: Annotated[
        Optional[list[str]], typer.Option("--disable-skill", help="Disable specific skill")
    ] = None,
    no_auto_skills: Annotated[
        bool, typer.Option("--no-auto-skills", help="Disable skill auto-discovery")
    ] = False,
    json_output: Annotated[
        bool, typer.Option("--json", help="Output response as machine-readable JSON")
    ] = False,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Minimal output (content only, no panels or formatting)")
    ] = False,
    # New options
    stdin: Annotated[
        bool, typer.Option("--stdin", "-i", help="Read prompt from stdin")
    ] = False,
    context: Annotated[
        Optional[list[Path]], typer.Option("--context", "-C", help="Include file(s) as context (can be repeated)")
    ] = None,
    template: Annotated[
        Optional[str], typer.Option("--template", "-T", help="Jinja2 template for output formatting")
    ] = None,
    output: Annotated[
        Optional[Path], typer.Option("--output", "-o", help="Write response to file")
    ] = None,
) -> None:
    """Send a prompt to Copilot with automatic retry on errors."""
    # Load config: explicit flag wins; otherwise auto-load from ~/.config/copex/config.toml if present
    if config_file and config_file.exists():
        config = CopexConfig.from_file(config_file)
    else:
        default_path = CopexConfig.default_path()
        if default_path.exists():
            config = CopexConfig.from_file(default_path)
        else:
            config = CopexConfig()

    # Override with CLI options
    effective_model = model or _DEFAULT_MODEL.value
    try:
        config.model = Model(effective_model)
    except ValueError:
        console.print(f"[red]Invalid model: {effective_model}[/red]")
        raise typer.Exit(1)

    try:
        requested_effort = parse_reasoning_effort(reasoning)
        if requested_effort is None:
            raise ValueError(reasoning)
        normalized_effort, warning = normalize_reasoning_effort(config.model, requested_effort)
        if warning:
            console.print(f"[yellow]{warning}[/yellow]")
        config.reasoning_effort = normalized_effort
    except ValueError:
        console.print(f"[red]Invalid reasoning effort: {reasoning}[/red]")
        raise typer.Exit(1)

    config.retry.max_retries = max_retries
    config.streaming = not no_stream
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
        raise typer.Exit(1)

    # Handle context files
    context_content = ""
    if context:
        context_parts = []
        for ctx_path in context:
            if not ctx_path.exists():
                console.print(f"[red]Context file not found: {ctx_path}[/red]")
                raise typer.Exit(1)
            try:
                content = ctx_path.read_text()
                # Format with filename header
                context_parts.append(f"<file path=\"{ctx_path}\">\n{content}\n</file>")
            except Exception as e:
                console.print(f"[red]Error reading {ctx_path}: {e}[/red]")
                raise typer.Exit(1)

        if context_parts:
            context_content = "\n\n".join(context_parts)
            prompt = f"Context files:\n{context_content}\n\nPrompt: {prompt}"

    asyncio.run(_run_chat(
        config, prompt, show_reasoning, raw,
        json_output=json_output, quiet=quiet,
        template=template, output_path=output,
    ))


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
    client = Copex(config)

    try:
        await client.start()

        if json_output:
            # Machine-readable JSON output (no streaming UI)
            response = await client.send(prompt)
            result: dict[str, Any] = {
                "content": response.content,
                "model": config.model.value,
                "retries": response.retries,
            }
            if show_reasoning and response.reasoning:
                result["reasoning"] = response.reasoning
            if response.usage:
                result["usage"] = response.usage
            output_text = json.dumps(result, indent=2)

            if output_path:
                output_path.write_text(output_text)
                console.print(f"[green]✓ Output saved to {output_path}[/green]")
            else:
                print(output_text)

        elif template:
            # Jinja2 template formatting
            try:
                from jinja2 import Template
            except ImportError:
                console.print("[red]Jinja2 not installed. Run: pip install jinja2[/red]")
                raise typer.Exit(1)

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
            }

            try:
                tmpl = Template(template)
                output_text = tmpl.render(**template_ctx)
            except Exception as e:
                console.print(f"[red]Template error: {e}[/red]")
                raise typer.Exit(1)

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
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    finally:
        await client.stop()


async def _stream_response(client: Copex, prompt: str, show_reasoning: bool) -> str:
    """Stream response with beautiful live updates. Returns the final content."""
    ui = CopexUI(
        console, theme=client.config.ui_theme, density=client.config.ui_density, show_all_tools=True
    )
    ui.reset(model=client.config.model.value)
    ui.set_activity(ActivityType.THINKING)

    live_display: Live | None = None
    refresh_stop = asyncio.Event()
    final_content = ""

    def on_chunk(chunk: StreamChunk) -> None:
        if chunk.type == "message":
            if chunk.is_final:
                ui.set_final_content(chunk.content or ui.state.message, ui.state.reasoning)
            else:
                ui.add_message(chunk.delta)
        elif chunk.type == "reasoning":
            if show_reasoning:
                if chunk.is_final:
                    pass  # Already captured incrementally
                else:
                    ui.add_reasoning(chunk.delta)
        elif chunk.type == "tool_call":
            tool = ToolCallInfo(
                tool_id=chunk.tool_id or "",
                name=chunk.tool_name or "unknown",
                arguments=chunk.tool_args or {},
                status="running",
            )
            ui.add_tool_call(tool)
        elif chunk.type == "tool_result":
            status = "success" if chunk.tool_success is not False else "error"
            ui.update_tool_call(
                chunk.tool_id,
                chunk.tool_name or "unknown",
                status,
                result=chunk.tool_result,
                duration=chunk.tool_duration,
            )
        elif chunk.type == "system":
            # Retry notification
            ui.increment_retries()
            print_retry(console, ui.state.retries, client.config.retry.max_retries, chunk.delta)

        # Update the live display
        if live_display:
            live_display.update(ui.build_live_display())

    async def refresh_loop() -> None:
        while not refresh_stop.is_set():
            if live_display:
                live_display.update(ui.build_live_display())
            await asyncio.sleep(0.1)

    with Live(console=console, refresh_per_second=10, transient=True) as live:
        live_display = live
        live.update(ui.build_live_display())
        refresh_task = asyncio.create_task(refresh_loop())
        try:
            response = await client.send(prompt, on_chunk=on_chunk)
            # Prefer streamed content over response object (which may have stale fallback)
            final_message = ui.state.message if ui.state.message else response.content
            final_reasoning = (
                (ui.state.reasoning if ui.state.reasoning else response.reasoning)
                if show_reasoning
                else None
            )
            ui.set_final_content(final_message, final_reasoning)
            ui.state.retries = response.retries
            ui.set_usage(response.prompt_tokens, response.completion_tokens)
            final_content = final_message
        finally:
            refresh_stop.set()
            try:
                await refresh_task
            except asyncio.CancelledError:
                pass

    # Print final beautiful output
    console.print(ui.build_final_display())
    return final_content


async def _stream_response_plain(client: Copex, prompt: str) -> None:
    """Stream response as plain text."""
    content = ""
    retries = 0

    def on_chunk(chunk: StreamChunk) -> None:
        nonlocal content
        if chunk.type == "message":
            if chunk.is_final:
                if chunk.content:
                    content = chunk.content
                return
            if chunk.delta:
                content += chunk.delta
                sys.stdout.write(chunk.delta)
                sys.stdout.flush()
        elif chunk.type == "system":
            console.print(f"[yellow]{chunk.delta.strip()}[/yellow]")

    response = await client.send(prompt, on_chunk=on_chunk)
    retries = response.retries
    if response.content and response.content != content:
        if response.content.startswith(content):
            sys.stdout.write(response.content[len(content) :])
        else:
            sys.stdout.write(response.content)
    sys.stdout.write("\n")
    sys.stdout.flush()

    if retries > 0:
        console.print(f"[dim]Completed with {retries} retries[/dim]")


@app.command()
def models() -> None:
    """List available models."""
    console.print("[bold]Available Models:[/bold]\n")
    for model in Model:
        console.print(f"  • {model.value}")


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
        Optional[list[str]], typer.Option("--skill-dir", "-S", help="Add skill directory")
    ] = None,
    no_auto: Annotated[
        bool, typer.Option("--no-auto", help="Disable auto-discovery")
    ] = False,
) -> None:
    """List all available skills."""
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
        Optional[list[str]], typer.Option("--skill-dir", "-S", help="Add skill directory")
    ] = None,
) -> None:
    """Show the content of a specific skill."""
    from copex.skills import get_skill_content

    content = get_skill_content(name, skill_directories=skill_dir)

    if content is None:
        console.print(f"[red]Skill not found: {name}[/red]")
        raise typer.Exit(1)

    console.print(Markdown(content))


@app.command("tui")
def tui(
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model to use")] = None,
    reasoning: Annotated[
        str, typer.Option("--reasoning", "-r", help="Reasoning effort level")
    ] = ReasoningEffort.HIGH.value,
) -> None:
    """Start the Copex TUI."""
    from copex.tui import run_tui

    effective_model = model or _DEFAULT_MODEL.value
    try:
        model_enum = Model(effective_model)
        requested_effort = parse_reasoning_effort(reasoning) or ReasoningEffort.HIGH
        normalized_effort, warning = normalize_reasoning_effort(model_enum, requested_effort)
        if warning:
            console.print(f"[yellow]{warning}[/yellow]")

        config = CopexConfig(
            model=model_enum,
            reasoning_effort=normalized_effort,
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    asyncio.run(run_tui(config))


@app.command()
def init(
    path: Annotated[
        Path, typer.Option("--path", "-p", help="Config file path")
    ] = CopexConfig.default_path(),
) -> None:
    """Create a default config file."""
    import tomli_w

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
        str, typer.Option("--reasoning", "-r", help="Reasoning effort level")
    ] = ReasoningEffort.HIGH.value,
    ui_theme: Annotated[
        Optional[str], typer.Option("--ui-theme", help="UI theme (default, midnight, mono, sunset, tokyo)")
    ] = None,
    ui_density: Annotated[
        Optional[str], typer.Option("--ui-density", help="UI density (compact or extended)")
    ] = None,
    classic: Annotated[
        bool, typer.Option("--classic", help="Use classic interactive mode (legacy)")
    ] = False,
    skill_dir: Annotated[
        Optional[list[str]], typer.Option("--skill-dir", "-S", help="Add skill directory")
    ] = None,
    disable_skill: Annotated[
        Optional[list[str]], typer.Option("--disable-skill", help="Disable specific skill")
    ] = None,
    no_auto_skills: Annotated[
        bool, typer.Option("--no-auto-skills", help="Disable skill auto-discovery")
    ] = False,
) -> None:
    """Start an interactive chat session."""
    effective_model = model or _DEFAULT_MODEL.value
    try:
        model_enum = Model(effective_model)
        requested_effort = parse_reasoning_effort(reasoning) or ReasoningEffort.HIGH
        normalized_effort, warning = normalize_reasoning_effort(model_enum, requested_effort)
        if warning:
            console.print(f"[yellow]{warning}[/yellow]")

        config = CopexConfig(
            model=model_enum,
            reasoning_effort=normalized_effort,
        )
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
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

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
    client = Copex(config)
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
                    new_model = Model(model_name)
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
            except Exception as e:
                print_error(console, str(e))

    except KeyboardInterrupt:
        console.print(f"\n[{Theme.WARNING}]{Icons.INFO} Goodbye![/{Theme.WARNING}]")
    finally:
        await client.stop()


async def _stream_response_interactive(
    client: Copex,
    prompt: str,
    ui: CopexUI,
) -> None:
    """Stream response with beautiful UI in interactive mode."""
    # Add user message to history
    ui.add_user_message(prompt)

    # Reset for new turn but preserve history
    ui.reset(model=client.config.model.value, preserve_history=True)
    ui.set_activity(ActivityType.THINKING)

    live_display: Live | None = None
    refresh_stop = asyncio.Event()

    def on_chunk(chunk: StreamChunk) -> None:
        if chunk.type == "message":
            if chunk.is_final:
                ui.set_final_content(chunk.content or ui.state.message, ui.state.reasoning)
            else:
                ui.add_message(chunk.delta)
        elif chunk.type == "reasoning":
            if chunk.is_final:
                pass
            else:
                ui.add_reasoning(chunk.delta)
        elif chunk.type == "tool_call":
            tool = ToolCallInfo(
                tool_id=chunk.tool_id or "",
                name=chunk.tool_name or "unknown",
                arguments=chunk.tool_args or {},
                status="running",
            )
            ui.add_tool_call(tool)
        elif chunk.type == "tool_result":
            status = "success" if chunk.tool_success is not False else "error"
            ui.update_tool_call(
                chunk.tool_id,
                chunk.tool_name or "unknown",
                status,
                result=chunk.tool_result,
                duration=chunk.tool_duration,
            )
        elif chunk.type == "system":
            ui.increment_retries()

        if live_display:
            live_display.update(ui.build_live_display())

    async def refresh_loop() -> None:
        while not refresh_stop.is_set():
            if live_display:
                live_display.update(ui.build_live_display())
            await asyncio.sleep(0.1)

    with Live(console=console, refresh_per_second=10, transient=True) as live:
        live_display = live
        live.update(ui.build_live_display())
        refresh_task = asyncio.create_task(refresh_loop())
        try:
            response = await client.send(prompt, on_chunk=on_chunk)
            # Prefer streamed content over response object (which may have stale fallback)
            final_message = ui.state.message if ui.state.message else response.content
            final_reasoning = ui.state.reasoning if ui.state.reasoning else response.reasoning
            ui.set_final_content(final_message, final_reasoning)
            ui.state.retries = response.retries
            ui.set_usage(response.prompt_tokens, response.completion_tokens)
        finally:
            refresh_stop.set()
            try:
                await refresh_task
            except asyncio.CancelledError:
                pass

    # Finalize the assistant response to history
    ui.finalize_assistant_response()

    console.print(ui.build_final_display())
    console.print()  # Extra spacing


@app.command("ralph")
def ralph_command(
    prompt: Annotated[str, typer.Argument(help="Task prompt for the Ralph loop")],
    max_iterations: Annotated[
        int, typer.Option("--max-iterations", "-n", help="Maximum iterations")
    ] = 30,
    completion_promise: Annotated[
        Optional[str], typer.Option("--promise", "-p", help="Completion promise text")
    ] = None,
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model to use")] = None,
    reasoning: Annotated[
        str, typer.Option("--reasoning", "-r", help="Reasoning effort level")
    ] = ReasoningEffort.HIGH.value,
    skill_dir: Annotated[
        Optional[list[str]], typer.Option("--skill-dir", "-S", help="Add skill directory")
    ] = None,
    disable_skill: Annotated[
        Optional[list[str]], typer.Option("--disable-skill", help="Disable specific skill")
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
    """
    effective_model = model or _DEFAULT_MODEL.value
    try:
        model_enum = Model(effective_model)
        requested_effort = parse_reasoning_effort(reasoning) or ReasoningEffort.HIGH
        normalized_effort, warning = normalize_reasoning_effort(model_enum, requested_effort)
        if warning:
            console.print(f"[yellow]{warning}[/yellow]")

        config = CopexConfig(
            model=model_enum,
            reasoning_effort=normalized_effort,
        )

        # Skills options
        if skill_dir:
            config.skill_directories.extend(skill_dir)
        if disable_skill:
            config.disabled_skills.extend(disable_skill)
        if no_auto_skills:
            config.auto_discover_skills = False
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

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

    client = Copex(config)
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
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    finally:
        await client.stop()


@app.command("login")
def login() -> None:
    """Login to GitHub (uses GitHub CLI for authentication)."""
    import shutil
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
        raise typer.Exit(1)

    console.print("[blue]Opening browser for GitHub authentication...[/blue]\n")

    try:
        result = subprocess.run([gh_path, "auth", "login"], check=False)
        if result.returncode == 0:
            console.print("\n[green]✓ Successfully logged in![/green]")
            console.print("You can now use [bold]copex chat[/bold]")
        else:
            console.print("\n[yellow]Login may have failed. Check status with:[/yellow]")
            console.print("  [bold]copex status[/bold]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("logout")
def logout() -> None:
    """Logout from GitHub."""
    import shutil
    import subprocess

    gh_path = shutil.which("gh")
    if not gh_path:
        console.print("[red]Error: GitHub CLI (gh) not found.[/red]")
        raise typer.Exit(1)

    try:
        result = subprocess.run([gh_path, "auth", "logout"], check=False)
        if result.returncode == 0:
            console.print("[green]✓ Logged out[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("status")
def status() -> None:
    """Check Copilot CLI and GitHub authentication status."""
    import shutil
    import subprocess

    from copex.config import find_copilot_cli

    cli_path = find_copilot_cli()
    gh_path = shutil.which("gh")

    # Get copilot version
    copilot_version = "N/A"
    if cli_path:
        try:
            result = subprocess.run(
                [cli_path, "--version"], capture_output=True, text=True, timeout=5
            )
            copilot_version = result.stdout.strip() or result.stderr.strip()
        except Exception:
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
        console.print("\n[red]Copilot CLI not found.[/red]")
        console.print("Install: [bold]npm install -g @github/copilot[/bold]")

    if gh_path:
        console.print("\n[bold]GitHub Auth Status:[/bold]")
        try:
            subprocess.run([gh_path, "auth", "status"], check=False)
        except Exception as e:
            console.print(f"[red]Error checking status: {e}[/red]")
    else:
        console.print("\n[yellow]GitHub CLI not found - cannot check auth status[/yellow]")
        console.print("Install: [bold]https://cli.github.com/[/bold]")


@app.command("config")
def config_cmd(
    config_file: Annotated[
        Optional[Path], typer.Option("--config", "-c", help="Config file path")
    ] = None,
) -> None:
    """Validate and display the current Copex configuration."""
    import warnings as _warnings

    config_path = Path(config_file) if config_file else CopexConfig.default_path()
    source = str(config_path) if config_path.exists() else "defaults"

    caught_warnings: list[_warnings.WarningMessage] = []
    with _warnings.catch_warnings(record=True) as caught_warnings:
        _warnings.simplefilter("always")
        try:
            if config_path.exists():
                config = CopexConfig.from_file(config_path)
            else:
                config = CopexConfig()
        except Exception as e:
            console.print(f"[red]Configuration error:[/red] {e}")
            raise typer.Exit(1)

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


@app.command("plan")
def plan_command(
    task: Annotated[
        Optional[str], typer.Argument(help="Task to plan (optional with --resume)")
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
        Optional[Path], typer.Option("--output", "-o", help="Save plan to file")
    ] = None,
    from_step: Annotated[
        int, typer.Option("--from-step", "-f", help="Resume execution from step number")
    ] = 1,
    load_plan: Annotated[
        Optional[Path],
        typer.Option("--load", "-l", help="Load plan from file instead of generating"),
    ] = None,
    max_iterations: Annotated[
        int, typer.Option("--max-iterations", "-n", help="Max iterations per step (Ralph loop)")
    ] = 10,
    visualize: Annotated[
        Optional[str],
        typer.Option("--visualize", "-V", help="Show plan visualization (ascii, mermaid, tree)")
    ] = None,
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model to use")] = None,
    reasoning: Annotated[
        str, typer.Option("--reasoning", "-r", help="Reasoning effort level")
    ] = ReasoningEffort.HIGH.value,
    skill_dir: Annotated[
        Optional[list[str]], typer.Option("--skill-dir", "-S", help="Add skill directory")
    ] = None,
    disable_skill: Annotated[
        Optional[list[str]], typer.Option("--disable-skill", help="Disable specific skill")
    ] = None,
    no_auto_skills: Annotated[
        bool, typer.Option("--no-auto-skills", help="Disable skill auto-discovery")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Generate and display the plan without executing")
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
    """
    # Validate: need task OR resume OR load_plan
    if not task and not resume and not load_plan:
        console.print("[red]Error: Provide a task, --resume, or --load[/red]")
        raise typer.Exit(1)

    effective_model = model or _DEFAULT_MODEL.value
    try:
        model_enum = Model(effective_model)
        requested_effort = parse_reasoning_effort(reasoning) or ReasoningEffort.HIGH
        normalized_effort, warning = normalize_reasoning_effort(model_enum, requested_effort)
        if warning:
            console.print(f"[yellow]{warning}[/yellow]")

        config = CopexConfig(
            model=model_enum,
            reasoning_effort=normalized_effort,
        )

        # Skills options
        if skill_dir:
            config.skill_directories.extend(skill_dir)
        if disable_skill:
            config.disabled_skills.extend(disable_skill)
        if no_auto_skills:
            config.auto_discover_skills = False
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

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

    client = Copex(config)
    await client.start()

    plan_ui = PlanUI(console)

    try:
        # Create Ralph instance for iterative step execution
        ralph = RalphWiggum(client)
        executor = PlanExecutor(client, ralph=ralph)
        executor.max_iterations_per_step = max_iterations

        # Check for resume from checkpoint
        if resume:
            state = PlanState.load()
            if state is None:
                console.print("[red]No checkpoint found (.copex-state.json)[/red]")
                console.print("[dim]Run a plan with --execute first to create a checkpoint[/dim]")
                raise typer.Exit(1)

            plan = state.plan
            from_step = state.current_step
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
                raise typer.Exit(1)
            plan = Plan.load(load_plan)
            console.print(f"[green]✓ Loaded plan from {load_plan}[/green]\n")
        else:
            # Generate new plan
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
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
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


@app.command("fleet")
def fleet_command(
    prompts: Annotated[
        Optional[list[str]], typer.Argument(help="Task prompts to run in parallel")
    ] = None,
    file: Annotated[
        Optional[Path], typer.Option("--file", "-f", help="TOML file with task definitions")
    ] = None,
    max_concurrent: Annotated[
        int, typer.Option("--max-concurrent", help="Max concurrent tasks")
    ] = 5,
    fail_fast: Annotated[
        bool, typer.Option("--fail-fast", help="Stop all tasks on first failure")
    ] = False,
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model to use")] = None,
    reasoning: Annotated[
        str, typer.Option("--reasoning", "-r", help="Reasoning effort level")
    ] = ReasoningEffort.HIGH.value,
    shared_context: Annotated[
        Optional[str], typer.Option("--shared-context", help="Context prepended to all tasks")
    ] = None,
    timeout: Annotated[
        float, typer.Option("--timeout", help="Per-task timeout in seconds")
    ] = 600.0,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Print full task outputs after completion")
    ] = False,
    output_dir: Annotated[
        Optional[Path],
        typer.Option("--output-dir", "-o", help="Directory to save each task result as a file"),
    ] = None,
    git_finalize: Annotated[
        bool,
        typer.Option(
            "--git-finalize/--no-git-finalize",
            help="Stage and commit all changes after fleet completes (auto-detected, enabled by default)",
        ),
    ] = True,
    git_message: Annotated[
        Optional[str],
        typer.Option("--git-message", help="Commit message for git finalize"),
    ] = None,
) -> None:
    """
    Run multiple tasks in parallel with fleet execution.

    Examples:
        copex fleet "Write tests" "Fix linting" "Update docs"
        copex fleet --file tasks.toml
        copex fleet "Task A" "Task B" --max-concurrent 3 --fail-fast
    """
    if not prompts and not file:
        console.print("[red]Error: Provide task prompts or --file[/red]")
        raise typer.Exit(1)

    effective_model = model or _DEFAULT_MODEL.value
    try:
        model_enum = Model(effective_model)
        requested_effort = parse_reasoning_effort(reasoning) or ReasoningEffort.HIGH
        normalized_effort, warning = normalize_reasoning_effort(model_enum, requested_effort)
        if warning:
            console.print(f"[yellow]{warning}[/yellow]")

        config = CopexConfig(
            model=model_enum,
            reasoning_effort=normalized_effort,
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    asyncio.run(
        _run_fleet(
            config=config,
            prompts=prompts or [],
            file=file,
            max_concurrent=max_concurrent,
            fail_fast=fail_fast,
            shared_context=shared_context,
            timeout=timeout,
            verbose=verbose,
            output_dir=output_dir,
            git_finalize=git_finalize,
            git_message=git_message,
        )
    )


async def _run_fleet(
    config: CopexConfig,
    prompts: list[str],
    file: Path | None,
    max_concurrent: int,
    fail_fast: bool,
    shared_context: str | None,
    timeout: float,
    verbose: bool = False,
    output_dir: Path | None = None,
    git_finalize: bool = True,
    git_message: str | None = None,
) -> None:
    """Run fleet tasks with live progress display."""
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

    # Load tasks from TOML file
    if file:
        try:
            import tomllib  # Python 3.11+
        except Exception:
            import tomli as tomllib  # type: ignore

        if not file.exists():
            console.print(f"[red]File not found: {file}[/red]")
            raise typer.Exit(1)

        with open(file, "rb") as f:
            data = tomllib.load(f)

        # Apply fleet-level config from file
        fleet_section = data.get("fleet", {})
        if "max_concurrent" in fleet_section:
            fleet_config.max_concurrent = fleet_section["max_concurrent"]
        if "shared_context" in fleet_section:
            fleet_config.shared_context = fleet_section["shared_context"]
        if "timeout" in fleet_section:
            fleet_config.timeout = fleet_section["timeout"]
        if "fail_fast" in fleet_section:
            fleet_config.fail_fast = fleet_section["fail_fast"]

        for task_data in data.get("task", []):
            tasks.append(
                FleetTask(
                    id=task_data["id"],
                    prompt=task_data["prompt"],
                    depends_on=task_data.get("depends_on", []),
                )
            )

    # Add CLI prompt tasks
    for i, prompt in enumerate(prompts):
        tasks.append(FleetTask(id=f"task-{i + 1}", prompt=prompt))

    if not tasks:
        console.print("[red]No tasks to run[/red]")
        raise typer.Exit(1)

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

    async with Fleet(config, fleet_config) as fleet:
        for task in tasks:
            fleet.add(
                task.prompt,
                task_id=task.id,
                depends_on=task.depends_on if task.depends_on else None,
            )

        with Live(build_table(), console=console, refresh_per_second=4) as live:

            async def _run_with_updates() -> list:
                from copex.fleet import FleetResult

                results = await fleet.run(on_status=on_status)
                # Mark final statuses
                for r in results:
                    statuses[r.task_id] = "done" if r.success else "failed"
                live.update(build_table())
                return results

            results = await _run_with_updates()

    total_time = time.time() - start_time
    successes = sum(1 for r in results if r.success)
    failures = len(results) - successes

    # Summary
    summary_lines = [
        f"[green]✅ Succeeded: {successes}[/green]",
        f"[red]❌ Failed:    {failures}[/red]",
        f"[blue]⏱  Duration:  {_format_duration(total_time)}[/blue]",
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
        from copex.git import GitFinalizer

        if not GitFinalizer.is_git_repo():
            console.print("[dim]Not a git repository — skipping git finalize[/dim]")
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
                console.print(
                    f"[red]Git finalize failed: {git_result.error}[/red]"
                )

    if failures > 0:
        raise typer.Exit(1)


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
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
