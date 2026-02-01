"""CLI interface for Copex."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Annotated, Optional

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

from copex.client import Copex, StreamChunk
from copex.config import (
    CopexConfig,
    load_last_model,
    load_last_reasoning_effort,
    save_last_model,
    save_last_reasoning_effort,
)
from copex.models import Model, ReasoningEffort
from copex.plan import Plan, PlanExecutor, PlanState, PlanStep, StepStatus
from copex.ralph import RalphState, RalphWiggum
from copex.ui import (
    ActivityType,
    ASCIIIcons,
    CopexUI,
    Icons,
    Theme,
    THEME_PRESETS,
    ToolCallInfo,
    apply_theme,
    auto_apply_theme,
    get_theme_for_terminal,
    is_light_theme,
    list_themes,
    print_error,
    print_retry,
    print_user_prompt,
    print_welcome,
)
from copex.ui_components import (
    CodeBlock,
    DiffDisplay,
    PlainTextRenderer,
    StepProgress,
    TokenUsageDisplay,
    ToolCallGroup,
    ToolCallPanel,
    get_plain_console,
    is_terminal,
)

# Effective default: last used model/reasoning or defaults
_DEFAULT_MODEL = load_last_model() or Model.CLAUDE_OPUS_4_5
_DEFAULT_REASONING = load_last_reasoning_effort() or ReasoningEffort.XHIGH

# Global state for color/plain mode (set by callbacks)
_FORCE_NO_COLOR = False
_FORCE_PLAIN = False

app = typer.Typer(
    name="copex",
    help="Copilot Extended - Resilient wrapper with auto-retry and Ralph Wiggum loops.",
    no_args_is_help=False,
    invoke_without_command=True,
)


def _get_console() -> Console:
    """Get a console with appropriate color settings."""
    global _FORCE_NO_COLOR, _FORCE_PLAIN
    
    if _FORCE_PLAIN or _FORCE_NO_COLOR:
        return get_plain_console()
    
    # Check environment for NO_COLOR
    if os.environ.get("NO_COLOR"):
        return get_plain_console()
    
    # Check if not a TTY
    if not is_terminal():
        return get_plain_console()
    
    return Console()


console = Console()  # Default, may be replaced by callbacks

# Version for --version flag
__version__ = "1.1.0"


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"copex version {__version__}")
        raise typer.Exit()


def no_color_callback(value: bool) -> None:
    """Disable colors."""
    global _FORCE_NO_COLOR, console
    if value:
        _FORCE_NO_COLOR = True
        os.environ["NO_COLOR"] = "1"
        console = get_plain_console()


def plain_callback(value: bool) -> None:
    """Enable plain text mode (no colors, no unicode, no animations)."""
    global _FORCE_PLAIN, console
    if value:
        _FORCE_PLAIN = True
        os.environ["NO_COLOR"] = "1"
        console = get_plain_console()


def theme_callback(value: str | None) -> str | None:
    """Apply theme immediately."""
    if value:
        if value == "auto":
            auto_apply_theme()
        elif value in THEME_PRESETS:
            apply_theme(value)
        else:
            valid = ", ".join(list_themes())
            raise typer.BadParameter(f"Invalid theme. Valid: {valid}, auto")
    return value


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
        bool, typer.Option("--version", "-V", callback=version_callback, is_eager=True, help="Show version and exit")
    ] = False,
    no_color: Annotated[
        bool, typer.Option("--no-color", callback=no_color_callback, is_eager=True, help="Disable colors")
    ] = False,
    plain: Annotated[
        bool, typer.Option("--plain", callback=plain_callback, is_eager=True, help="Plain text mode (no colors, unicode, animations)")
    ] = False,
    theme: Annotated[
        Optional[str], typer.Option("--theme", "-t", callback=theme_callback, help="Color theme (auto, default, light, dark-256, etc.)")
    ] = None,
    log_level: Annotated[
        Optional[str], typer.Option("--log-level", help="Log level (debug, info, warning, error)")
    ] = None,
    log_file: Annotated[
        Optional[Path], typer.Option("--log-file", help="Write structured logs to file")
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Model to use")
    ] = None,
    reasoning: Annotated[
        str, typer.Option("--reasoning", "-r", help="Reasoning effort level")
    ] = _DEFAULT_REASONING.value,
) -> None:
    """Copilot Extended - Resilient wrapper with auto-retry and Ralph Wiggum loops."""
    global console
    
    # Auto-detect theme if not set
    if not theme and not no_color and not plain:
        auto_apply_theme()
    
    # Update console based on settings
    console = _get_console()
    
    if ctx.invoked_subcommand is None:
        # No command provided - launch interactive mode
        effective_model = model or _DEFAULT_MODEL.value
        interactive(
            model=effective_model,
            reasoning=reasoning,
            log_level=log_level,
            log_file=log_file,
            no_color=no_color,
            plain=plain,
            theme=theme,
        )


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
    style = Style.from_dict({
        "selected": "fg:ansicyan bold",
        "current": "fg:ansiyellow italic",
    })

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
        lines = [("bold", "Select reasoning effort (↑/↓ to navigate, Enter to select, Esc to cancel):\n\n")]
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
    style = Style.from_dict({
        "selected": "fg:ansicyan bold",
        "current": "fg:ansiyellow italic",
    })

    app: Application[ReasoningEffort | None] = Application(
        layout=Layout(Window(FormattedTextControl(get_text))),
        key_bindings=kb,
        style=style,
        full_screen=False,
    )

    return await app.run_async()


@app.command()
def chat(
    prompt: Annotated[Optional[str], typer.Argument(help="Prompt to send (or read from stdin)")] = None,
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Model to use")
    ] = None,
    reasoning: Annotated[
        str, typer.Option("--reasoning", "-r", help="Reasoning effort level")
    ] = _DEFAULT_REASONING.value,
    max_retries: Annotated[
        int, typer.Option("--max-retries", help="Maximum retry attempts")
    ] = 5,
    no_stream: Annotated[
        bool, typer.Option("--no-stream", help="Disable streaming output")
    ] = False,
    show_reasoning: Annotated[
        bool, typer.Option("--show-reasoning/--no-reasoning", help="Show model reasoning")
    ] = True,
    config_file: Annotated[
        Optional[Path], typer.Option("--config", "-c", help="Config file path")
    ] = None,
    raw: Annotated[
        bool, typer.Option("--raw", help="Output raw text without formatting")
    ] = False,
    log_level: Annotated[
        Optional[str], typer.Option("--log-level", help="Log level (debug, info, warning, error)")
    ] = None,
    log_file: Annotated[
        Optional[Path], typer.Option("--log-file", help="Write structured logs to file")
    ] = None,
    ui_theme: Annotated[
        Optional[str], typer.Option("--ui-theme", help="UI theme (default, midnight, mono, sunset)")
    ] = None,
    ui_density: Annotated[
        Optional[str], typer.Option("--ui-density", help="UI density (compact or extended)")
    ] = None,
    no_color: Annotated[
        bool, typer.Option("--no-color", help="Disable colors")
    ] = False,
    plain: Annotated[
        bool, typer.Option("--plain", help="Plain text mode (no colors, unicode, animations)")
    ] = False,
) -> None:
    """Send a prompt to Copilot with automatic retry on errors."""
    global console, _FORCE_NO_COLOR, _FORCE_PLAIN
    
    # Handle accessibility flags
    if no_color:
        _FORCE_NO_COLOR = True
        os.environ["NO_COLOR"] = "1"
    if plain:
        _FORCE_PLAIN = True
        os.environ["NO_COLOR"] = "1"
    
    console = _get_console()
    
    # Load config
    if config_file and config_file.exists():
        config = CopexConfig.from_file(config_file)
    else:
        config = CopexConfig()
    
    # Apply accessibility settings
    if plain or raw:
        config.ui_ascii_icons = True

    # Override with CLI options
    effective_model = model or _DEFAULT_MODEL.value
    try:
        config.model = Model(effective_model)
    except ValueError:
        console.print(f"[red]Invalid model: {effective_model}[/red]")
        raise typer.Exit(1)

    try:
        config.reasoning_effort = ReasoningEffort(reasoning)
    except ValueError:
        console.print(f"[red]Invalid reasoning effort: {reasoning}[/red]")
        raise typer.Exit(1)

    save_last_reasoning_effort(config.reasoning_effort)

    config.retry.max_retries = max_retries
    config.streaming = not no_stream
    if log_level:
        config.log_level = log_level
    if log_file:
        config.log_file = str(log_file)
    if ui_theme:
        config.ui_theme = ui_theme
    if ui_density:
        config.ui_density = ui_density

    from copex.config import configure_logging
    configure_logging(config)

    # Get prompt from stdin if not provided
    if prompt is None:
        if sys.stdin.isatty():
            console.print("[yellow]Enter prompt (Ctrl+D to submit):[/yellow]")
        prompt = sys.stdin.read().strip()
        if not prompt:
            console.print("[red]No prompt provided[/red]")
            raise typer.Exit(1)

    asyncio.run(_run_chat(config, prompt, show_reasoning, raw))


async def _run_chat(
    config: CopexConfig, prompt: str, show_reasoning: bool, raw: bool
) -> None:
    """Run the chat command."""
    client = Copex(config)
    logger = logging.getLogger(__name__)

    try:
        await client.start()
        logger.info(
            "chat.start",
            extra={
                "model": config.model.value,
                "reasoning_effort": config.reasoning_effort.value,
                "streaming": config.streaming,
            },
        )

        if config.streaming and not raw:
            await _stream_response(client, prompt, show_reasoning)
        else:
            response = await client.send(prompt)
            if raw:
                print(response.content)
            else:
                if show_reasoning and response.reasoning:
                    console.print(Panel(
                        Markdown(response.reasoning),
                        title="[dim]Reasoning[/dim]",
                        border_style="dim",
                    ))
                console.print(Markdown(response.content))

                if response.retries > 0:
                    console.print(
                        f"\n[dim]Completed with {response.retries} retries[/dim]"
                    )

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    finally:
        logger.info("chat.stop")
        await client.stop()


async def _stream_response(
    client: Copex, prompt: str, show_reasoning: bool
) -> None:
    """Stream response with beautiful live updates."""
    ui = CopexUI(
        console,
        theme=client.config.ui_theme,
        density=client.config.ui_density,
        show_all_tools=True,
        show_reasoning=show_reasoning,
        ascii_icons=client.config.ui_ascii_icons,
    )
    ui.reset(model=client.config.model.value)
    ui.set_activity(ActivityType.THINKING)
    await _stream_with_ui(
        client,
        prompt,
        ui,
        show_reasoning=show_reasoning,
        show_retry_notifications=True,
    )


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
            sys.stdout.write(response.content[len(content):])
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


@app.command()
def themes() -> None:
    """List available UI themes."""
    from copex.ui import get_theme_preview, is_light_theme
    
    console.print("[bold]Available Themes:[/bold]\n")
    
    # Group themes
    dark_themes = []
    light_themes = []
    special_themes = []
    
    for name in list_themes():
        if "light" in name:
            light_themes.append(name)
        elif "dark" in name or name in ("default", "midnight", "codex"):
            dark_themes.append(name)
        else:
            special_themes.append(name)
    
    # Show current detection
    is_light = is_light_theme()
    current = get_theme_for_terminal()
    console.print(f"  [dim]Detected:[/dim] {'light' if is_light else 'dark'} terminal → [cyan]{current}[/cyan]\n")
    
    console.print("[bold]Dark themes:[/bold]")
    for name in sorted(dark_themes):
        marker = " ← detected" if name == current else ""
        console.print(f"  • {name}{marker}")
    
    console.print("\n[bold]Light themes:[/bold]")
    for name in sorted(light_themes):
        marker = " ← detected" if name == current else ""
        console.print(f"  • {name}{marker}")
    
    if special_themes:
        console.print("\n[bold]Special themes:[/bold]")
        for name in sorted(special_themes):
            marker = " ← detected" if name == current else ""
            console.print(f"  • {name}{marker}")
    
    console.print("\n[dim]Use --theme <name> or --theme auto[/dim]")


@app.command(name="ui-demo")
def ui_demo(
    theme: Annotated[
        Optional[str], typer.Option("--theme", "-t", help="Theme to use for demo")
    ] = None,
    plain: Annotated[
        bool, typer.Option("--plain", help="Show plain text versions")
    ] = False,
    component: Annotated[
        Optional[str], typer.Option("--component", "-c", help="Show only specific component")
    ] = None,
) -> None:
    """Showcase all UI components with sample data."""
    import time
    from rich.rule import Rule
    
    global console, _FORCE_PLAIN
    
    if plain:
        _FORCE_PLAIN = True
        console = get_plain_console()
    else:
        console = _get_console()
    
    # Apply theme
    if theme:
        if theme in THEME_PRESETS:
            apply_theme(theme)
        else:
            console.print(f"[red]Unknown theme: {theme}[/red]")
            console.print(f"[dim]Available: {', '.join(list_themes())}[/dim]")
            raise typer.Exit(1)
    else:
        auto_apply_theme()
    
    valid_components = ["code", "diff", "tokens", "tools", "progress", "collapsible", "streaming", "all"]
    if component and component not in valid_components:
        console.print(f"[red]Unknown component: {component}[/red]")
        console.print(f"[dim]Available: {', '.join(valid_components)}[/dim]")
        raise typer.Exit(1)
    
    show_all = component is None or component == "all"
    
    console.print()
    console.print(Panel(
        "[bold]Copex UI Components Demo[/bold]\n\n"
        "This demo showcases all the UI components available in Copex.\n"
        f"Theme: [cyan]{theme or 'auto'}[/cyan] | Plain mode: [cyan]{plain}[/cyan]",
        title="🎨 UI Demo",
        border_style=Theme.PRIMARY,
    ))
    console.print()
    
    # 1. Code Block with Syntax Highlighting
    if show_all or component == "code":
        console.print(Rule("1. Code Block with Syntax Highlighting"))
        console.print()
        
        python_code = '''def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Example usage
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")'''
        
        code_block = CodeBlock(python_code, language="python", filename="fibonacci.py")
        
        if plain:
            console.print(code_block.to_plain_text())
        else:
            console.print(code_block.to_panel())
        console.print()
        
        # Also show JavaScript
        js_code = '''const fetchData = async (url) => {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return response.json();
};'''
        
        js_block = CodeBlock(js_code, language="javascript")
        if plain:
            console.print(js_block.to_plain_text())
        else:
            console.print(js_block.to_panel())
        console.print()
    
    # 2. Diff Display
    if show_all or component == "diff":
        console.print(Rule("2. Diff Display"))
        console.print()
        
        old_code = '''def greet(name):
    print("Hello, " + name)
    return None'''
        
        new_code = '''def greet(name: str) -> str:
    """Greet someone by name."""
    message = f"Hello, {name}!"
    print(message)
    return message'''
        
        diff = DiffDisplay.from_strings(old_code, new_code, filename="greet.py")
        
        if plain:
            console.print(diff.to_plain_text())
        else:
            console.print(diff.to_panel())
        
        stats = diff.stats
        console.print(f"\n[dim]Stats: +{stats['additions']} additions, -{stats['deletions']} deletions[/dim]")
        console.print()
    
    # 3. Token Usage Display
    if show_all or component == "tokens":
        console.print(Rule("3. Token Usage Display"))
        console.print()
        
        # Small usage
        usage1 = TokenUsageDisplay(
            input_tokens=1234,
            output_tokens=567,
            model="gpt-5",
            show_cost=True,
        )
        console.print("[bold]Small usage:[/bold]")
        if plain:
            console.print(usage1.to_plain_text())
        else:
            console.print(usage1.to_text())
        console.print()
        
        # Large usage
        usage2 = TokenUsageDisplay(
            input_tokens=125_000,
            output_tokens=45_000,
            model="claude-opus-4.5",
            show_cost=True,
        )
        console.print("[bold]Large usage (panel view):[/bold]")
        if plain:
            console.print(usage2.to_plain_text())
        else:
            console.print(usage2.to_panel())
        console.print()
    
    # 4. Tool Call Panels
    if show_all or component == "tools":
        console.print(Rule("4. Tool Call Panels"))
        console.print()
        
        # Create a group of tool calls
        group = ToolCallGroup()
        
        # Bash command - success
        group.add("bash", {"command": "ls -la /tmp"})
        group.complete(0, result="total 16\ndrwxrwxrwt  5 root  wheel  160 Jan 15 10:30 .\ndrwxr-xr-x  6 root  wheel  192 Jan 10 09:00 ..", duration=0.15)
        
        # File read - success
        group.add("read", {"path": "config.json", "lines": "1-20"})
        group.complete(1, result='{\n  "name": "copex",\n  "version": "1.1.0"\n}', duration=0.02)
        
        # Search - success
        group.add("grep", {"pattern": "def.*:", "path": "src/"})
        group.complete(2, result="src/main.py:5: def main():\nsrc/utils.py:12: def helper():", duration=0.35)
        
        # Failed command
        group.add("bash", {"command": "rm -rf /protected"})
        group.complete(3, error="Permission denied: /protected", duration=0.01)
        
        if plain:
            for call in group.calls:
                console.print(call.to_plain_text())
                console.print()
        else:
            panel = group.to_panel()
            if panel:
                console.print(panel)
        
        console.print()
        stats = group.get_stats()
        console.print(f"[dim]Stats: {stats['success']} success, {stats['error']} error, {stats['running']} running[/dim]")
        console.print()
    
    # 5. Step Progress
    if show_all or component == "progress":
        console.print(Rule("5. Step Progress"))
        console.print()
        
        progress = StepProgress(total=5, title="Implementation Plan")
        progress.add_step(1, "Analyze requirements")
        progress.add_step(2, "Design architecture")
        progress.add_step(3, "Implement core features")
        progress.add_step(4, "Write tests")
        progress.add_step(5, "Deploy to production")
        
        # Simulate progress
        progress.complete_step(1)
        progress.complete_step(2)
        progress.start_step(3)
        
        if plain:
            console.print(progress.to_plain_text())
        else:
            console.print(progress.to_panel())
        console.print()
        
        # Also show a completed progress
        progress2 = StepProgress(total=3, title="Quick Task")
        progress2.add_step(1, "Download data")
        progress2.add_step(2, "Process files")
        progress2.add_step(3, "Generate report")
        progress2.complete_step(1)
        progress2.complete_step(2)
        progress2.complete_step(3)
        
        console.print("[bold]Completed progress:[/bold]")
        if plain:
            console.print(progress2.to_plain_text())
        else:
            console.print(progress2.to_text())
        console.print()
    
    # 6. Collapsible Sections
    if show_all or component == "collapsible":
        console.print(Rule("6. Collapsible Sections"))
        console.print()
        
        from copex.ui_components import CollapsibleSection, CollapsibleGroup
        
        # Create sections
        section1 = CollapsibleSection(
            "System Information",
            "CPU: Apple M1 Pro\nMemory: 32GB\nOS: macOS 14.0",
            expanded=False,
        )
        
        section2 = CollapsibleSection(
            "Environment Variables",
            "PATH=/usr/local/bin:/usr/bin\nHOME=/Users/demo\nSHELL=/bin/zsh",
            expanded=True,
        )
        
        section3 = CollapsibleSection(
            "Long Output (collapsed with preview)",
            "This is line 1 of a very long output...\n" * 20,
            expanded=False,
            preview_length=50,
        )
        
        console.print("[bold]Collapsed section:[/bold]")
        console.print(section1)
        console.print()
        
        console.print("[bold]Expanded section:[/bold]")
        console.print(section2)
        console.print()
        
        console.print("[bold]Collapsed with preview:[/bold]")
        console.print(section3)
        console.print()
    
    # 7. Status Icons
    if show_all or component == "streaming":
        console.print(Rule("7. Status Icons"))
        console.print()
        
        icon_set = ASCIIIcons if plain else Icons
        
        console.print(f"  {icon_set.DONE} Success - Operation completed")
        console.print(f"  {icon_set.ERROR} Error - Something went wrong")
        console.print(f"  {icon_set.WARNING} Warning - Proceed with caution")
        console.print(f"  {icon_set.INFO} Info - Just FYI")
        console.print(f"  {icon_set.THINKING} Thinking - Processing...")
        console.print(f"  {icon_set.TOOL} Tool - Executing tool")
        console.print()
        
        if not plain:
            console.print("[bold]Spinner animation (Braille):[/bold]")
            frames = Icons.BRAILLE_SPINNER
            console.print(f"  Frames: {' '.join(frames)}")
            console.print()
    
    # Summary
    console.print(Rule("Summary"))
    console.print()
    console.print(Panel(
        "[bold green]✓[/bold green] All UI components demonstrated successfully!\n\n"
        "Components shown:\n"
        "  • CodeBlock - Syntax highlighting with 40+ languages\n"
        "  • DiffDisplay - Git-style diff with colors\n"
        "  • TokenUsageDisplay - Token counts and cost\n"
        "  • ToolCallPanel - Tool execution display\n"
        "  • StepProgress - Progress bars with ETA\n"
        "  • CollapsibleSection - Expandable content\n"
        "  • Icons - Status and spinner icons\n\n"
        f"[dim]Run with --plain to see ASCII fallback[/dim]",
        title="Demo Complete",
        border_style="green",
    ))


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
        "recovery_prompt_max_chars": 8000,
        "auth_refresh_interval": 3300.0,
        "auth_refresh_buffer": 300.0,
        "auth_refresh_on_error": True,
        "retry": {
            "max_retries": 5,
            "base_delay": 1.0,
            "max_delay": 30.0,
            "exponential_base": 2.0,
            "retry_on_errors": ["500", "502", "503", "504", "Internal Server Error", "rate limit"],
        },
        "log_level": "warning",
        "log_levels": {},
        "stream_queue_max_size": 1000,
        "stream_drop_mode": "drop_oldest",
        "ui_theme": "default",
        "ui_density": "extended",
        "ui_ascii_icons": False,
    }

    with open(path, "wb") as f:
        tomli_w.dump(config, f)

    console.print(f"[green]Created config at:[/green] {path}")


@app.command()
def interactive(
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Model to use")
    ] = None,
    reasoning: Annotated[
        str, typer.Option("--reasoning", "-r", help="Reasoning effort level")
    ] = _DEFAULT_REASONING.value,
    ui_theme: Annotated[
        Optional[str], typer.Option("--ui-theme", help="UI theme (default, midnight, mono, sunset)")
    ] = None,
    ui_density: Annotated[
        Optional[str], typer.Option("--ui-density", help="UI density (compact or extended)")
    ] = None,
    log_level: Annotated[
        Optional[str], typer.Option("--log-level", help="Log level (debug, info, warning, error)")
    ] = None,
    log_file: Annotated[
        Optional[Path], typer.Option("--log-file", help="Write structured logs to file")
    ] = None,
    no_color: Annotated[
        bool, typer.Option("--no-color", help="Disable colors")
    ] = False,
    plain: Annotated[
        bool, typer.Option("--plain", help="Plain text mode (no colors, unicode, animations)")
    ] = False,
    theme: Annotated[
        Optional[str], typer.Option("--theme", "-t", help="Color theme (auto, default, light, dark-256, etc.)")
    ] = None,
) -> None:
    """Start an interactive chat session."""
    global console, _FORCE_NO_COLOR, _FORCE_PLAIN
    
    # Handle accessibility flags
    if no_color:
        _FORCE_NO_COLOR = True
        os.environ["NO_COLOR"] = "1"
    if plain:
        _FORCE_PLAIN = True
        os.environ["NO_COLOR"] = "1"
    
    console = _get_console()
    
    # Apply theme
    if theme:
        if theme == "auto":
            auto_apply_theme()
        elif theme in THEME_PRESETS:
            apply_theme(theme)
    elif not no_color and not plain:
        auto_apply_theme()
    
    # Determine if we should use ASCII icons
    use_ascii = plain or not is_terminal()
    
    effective_model = model or _DEFAULT_MODEL.value
    try:
        config = CopexConfig(
            model=Model(effective_model),
            reasoning_effort=ReasoningEffort(reasoning),
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    
    # Apply UI settings
    if ui_theme:
        config.ui_theme = ui_theme
    if ui_density:
        config.ui_density = ui_density
    config.ui_ascii_icons = use_ascii

    print_welcome(
        console,
        config.model.value,
        config.reasoning_effort.value,
        theme=config.ui_theme,
        density=config.ui_density,
        ascii_icons=config.ui_ascii_icons,
    )
    if log_level:
        config.log_level = log_level
    if log_file:
        config.log_file = str(log_file)
    from copex.config import configure_logging
    configure_logging(config)
    save_last_reasoning_effort(config.reasoning_effort)
    asyncio.run(_interactive_loop(config))


async def _interactive_loop(config: CopexConfig) -> None:
    """Run interactive chat loop."""
    client = Copex(config)
    await client.start()
    session = _build_prompt_session()
    show_all_tools = False
    show_reasoning = True

    # Create persistent UI for conversation history
    ui = CopexUI(
        console,
        theme=config.ui_theme,
        density=config.ui_density,
        show_all_tools=show_all_tools,
        show_reasoning=show_reasoning,
        ascii_icons=config.ui_ascii_icons,
    )

    def show_help() -> None:
        console.print(f"\n[{Theme.MUTED}]Commands:[/{Theme.MUTED}]")
        console.print(f"  [{Theme.PRIMARY}]/model <name>[/{Theme.PRIMARY}]     - Change model (e.g., /model gpt-5.1-codex)")
        console.print(f"  [{Theme.PRIMARY}]/reasoning <level>[/{Theme.PRIMARY}] - Change reasoning (low, medium, high, xhigh)")
        console.print(f"  [{Theme.PRIMARY}]/reasoning[/{Theme.PRIMARY}]        - Toggle reasoning display")
        console.print(f"  [{Theme.PRIMARY}]/models[/{Theme.PRIMARY}]            - List available models")
        console.print(f"  [{Theme.PRIMARY}]/new[/{Theme.PRIMARY}]               - Start new session")
        console.print(f"  [{Theme.PRIMARY}]/status[/{Theme.PRIMARY}]            - Show current settings")
        console.print(f"  [{Theme.PRIMARY}]/tools[/{Theme.PRIMARY}]             - Toggle full tool call list")
        console.print(f"  [{Theme.PRIMARY}]/help[/{Theme.PRIMARY}]              - Show this help")
        console.print(f"  [{Theme.PRIMARY}]exit[/{Theme.PRIMARY}]               - Exit\n")

    def show_status() -> None:
        console.print(f"\n[{Theme.MUTED}]Current settings:[/{Theme.MUTED}]")
        console.print(f"  Model:     [{Theme.PRIMARY}]{client.config.model.value}[/{Theme.PRIMARY}]")
        console.print(f"  Reasoning: [{Theme.PRIMARY}]{client.config.reasoning_effort.value}[/{Theme.PRIMARY}]\n")

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
                icon_set = ASCIIIcons if config.ui_ascii_icons else Icons
                console.print(f"\n[{Theme.SUCCESS}]{icon_set.DONE} Started new session[/{Theme.SUCCESS}]\n")
                continue

            if command in {"help", "/help"}:
                show_help()
                continue

            if command in {"status", "/status"}:
                show_status()
                continue

            if command in {"reasoning", "/reasoning"}:
                show_reasoning = not show_reasoning
                ui.show_reasoning = show_reasoning
                if not show_reasoning:
                    ui.state.reasoning = ""
                mode = "showing reasoning" if show_reasoning else "hiding reasoning"
                icon_set = ASCIIIcons if config.ui_ascii_icons else Icons
                console.print(f"\n[{Theme.SUCCESS}]{icon_set.DONE} Now {mode}[/{Theme.SUCCESS}]\n")
                continue

            if command in {"models", "/models"}:
                selected = await _model_picker(client.config.model)
                if selected and selected != client.config.model:
                    client.config.model = selected
                    save_last_model(selected)  # Persist for next run
                    # Prompt for reasoning effort if GPT model
                    if selected.value.startswith("gpt-"):
                        new_reasoning = await _reasoning_picker(client.config.reasoning_effort)
                        if new_reasoning:
                            client.config.reasoning_effort = new_reasoning
                            save_last_reasoning_effort(new_reasoning)
                    client.new_session()
                    # Clear UI history for new session
                    ui.state.history = []
                    icon_set = ASCIIIcons if config.ui_ascii_icons else Icons
                    console.print(f"\n[{Theme.SUCCESS}]{icon_set.DONE} Switched to {selected.value} (new session started)[/{Theme.SUCCESS}]\n")
                continue

            if command in {"tools", "/tools"}:
                show_all_tools = not show_all_tools
                ui.show_all_tools = show_all_tools
                mode = "all tools" if show_all_tools else "recent tools"
                icon_set = ASCIIIcons if config.ui_ascii_icons else Icons
                console.print(f"\n[{Theme.SUCCESS}]{icon_set.DONE} Showing {mode}[/{Theme.SUCCESS}]\n")
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
                    save_last_reasoning_effort(client.config.reasoning_effort)
                    client.new_session()  # Need new session for model change
                    # Clear UI history for new session
                    ui.state.history = []
                    icon_set = ASCIIIcons if config.ui_ascii_icons else Icons
                    console.print(f"\n[{Theme.SUCCESS}]{icon_set.DONE} Switched to {new_model.value} (new session started)[/{Theme.SUCCESS}]\n")
                except ValueError:
                    console.print(f"[{Theme.ERROR}]Unknown model: {model_name}[/{Theme.ERROR}]")
                    console.print(f"[{Theme.MUTED}]Use /models to see available models[/{Theme.MUTED}]")
                continue

            if command.startswith("/reasoning ") or command.startswith("reasoning "):
                parts = prompt.split(maxsplit=1)
                if len(parts) < 2:
                    console.print(f"[{Theme.ERROR}]Usage: /reasoning <level>[/{Theme.ERROR}]")
                    continue
                level = parts[1].strip()
                try:
                    new_reasoning = ReasoningEffort(level)
                    client.config.reasoning_effort = new_reasoning
                    save_last_reasoning_effort(new_reasoning)
                    client.new_session()  # Need new session for reasoning change
                    # Clear UI history for new session
                    ui.state.history = []
                    icon_set = ASCIIIcons if config.ui_ascii_icons else Icons
                    console.print(f"\n[{Theme.SUCCESS}]{icon_set.DONE} Switched to {new_reasoning.value} reasoning (new session started)[/{Theme.SUCCESS}]\n")
                except ValueError:
                    valid = ", ".join(r.value for r in ReasoningEffort)
                    console.print(f"[{Theme.ERROR}]Invalid reasoning level. Valid: {valid}[/{Theme.ERROR}]")
                continue

            try:
                print_user_prompt(console, prompt, ascii_icons=config.ui_ascii_icons)
                await _stream_response_interactive(client, prompt, ui, show_reasoning)
            except Exception as e:
                print_error(console, str(e), ascii_icons=config.ui_ascii_icons)

    except KeyboardInterrupt:
        icon_set = ASCIIIcons if config.ui_ascii_icons else Icons
        console.print(f"\n[{Theme.WARNING}]{icon_set.INFO} Goodbye![/{Theme.WARNING}]")
    finally:
        await client.stop()


async def _stream_response_interactive(
    client: Copex,
    prompt: str,
    ui: CopexUI,
    show_reasoning: bool,
) -> None:
    """Stream response with beautiful UI in interactive mode."""
    # Add user message to history
    ui.add_user_message(prompt)

    # Reset for new turn but preserve history
    ui.reset(model=client.config.model.value, preserve_history=True)
    ui.set_activity(ActivityType.THINKING)
    await _stream_with_ui(client, prompt, ui, show_reasoning=show_reasoning, render_final=False)

    ui.finalize_assistant_response()
    console.print(ui.build_final_display())
    console.print()


async def _stream_with_ui(
    client: Copex,
    prompt: str,
    ui: CopexUI,
    *,
    show_reasoning: bool = True,
    show_retry_notifications: bool = False,
    render_final: bool = True,
) -> None:
    """Stream a response using shared UI logic."""
    live_display: Live | None = None
    refresh_stop = asyncio.Event()

    def on_chunk(chunk: StreamChunk) -> None:
        if chunk.type == "message":
            if chunk.is_final:
                ui.set_final_content(chunk.content or ui.state.message, ui.state.reasoning)
            else:
                ui.add_message(chunk.delta)
        elif chunk.type == "reasoning":
            if show_reasoning:
                if chunk.is_final:
                    pass
                else:
                    ui.add_reasoning(chunk.delta)
        elif chunk.type == "tool_call":
            tool = ToolCallInfo(
                id=chunk.tool_call_id,
                name=chunk.tool_name or "unknown",
                arguments=chunk.tool_args or {},
                status="running",
            )
            ui.add_tool_call(tool)
        elif chunk.type == "tool_result":
            status = "success" if chunk.tool_success is not False else "error"
            ui.update_tool_call(
                chunk.tool_name or "unknown",
                status,
                result=chunk.tool_result,
                duration=chunk.tool_duration,
                tool_call_id=chunk.tool_call_id,
            )
        elif chunk.type == "system":
            ui.increment_retries()
            if show_retry_notifications:
                print_retry(
                    console,
                    ui.state.retries,
                    client.config.retry.max_retries,
                    chunk.delta,
                    ascii_icons=client.config.ui_ascii_icons,
                )

        if live_display:
            live_display.update(ui.build_live_display())

    async def refresh_loop() -> None:
        while not refresh_stop.is_set():
            if live_display and ui.consume_dirty():
                live_display.update(ui.build_live_display())
            await asyncio.sleep(0.1)

    with Live(console=console, refresh_per_second=10, transient=True) as live:
        live_display = live
        live.update(ui.build_live_display())
        refresh_task = asyncio.create_task(refresh_loop())
        try:
            response = await client.send(prompt, on_chunk=on_chunk)
            final_message = ui.state.message if ui.state.message else response.content
            final_reasoning = (ui.state.reasoning if ui.state.reasoning else response.reasoning) if show_reasoning else None
            ui.set_final_content(final_message, final_reasoning)
            ui.state.retries = response.retries
        finally:
            refresh_stop.set()
            try:
                await refresh_task
            except asyncio.CancelledError:
                pass

    if render_final:
        console.print(ui.build_final_display())


@app.command("ralph")
def ralph_command(
    prompt: Annotated[str, typer.Argument(help="Task prompt for the Ralph loop")],
    max_iterations: Annotated[
        int, typer.Option("--max-iterations", "-n", help="Maximum iterations")
    ] = 30,
    completion_promise: Annotated[
        Optional[str], typer.Option("--promise", "-p", help="Completion promise text")
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Model to use")
    ] = None,
    reasoning: Annotated[
        str, typer.Option("--reasoning", "-r", help="Reasoning effort level")
    ] = _DEFAULT_REASONING.value,
    log_level: Annotated[
        Optional[str], typer.Option("--log-level", help="Log level (debug, info, warning, error)")
    ] = None,
    log_file: Annotated[
        Optional[Path], typer.Option("--log-file", help="Write structured logs to file")
    ] = None,
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
        config = CopexConfig(
            model=Model(effective_model),
            reasoning_effort=ReasoningEffort(reasoning),
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    icon_set = ASCIIIcons if config.ui_ascii_icons else Icons
    console.print(Panel(
        f"[bold]Ralph Wiggum Loop[/bold]\n"
        f"Model: {config.model.value}\n"
        f"Reasoning: {config.reasoning_effort.value}\n"
        f"Max iterations: {max_iterations}\n"
        f"Completion promise: {completion_promise or '(none)'}",
        title=f"{icon_set.TOOL} Starting Loop",
        border_style="yellow",
    ))

    if completion_promise:
        console.print(
            f"\n[dim]To complete, the AI must output: "
            f"[yellow]<promise>{completion_promise}</promise>[/yellow][/dim]\n"
        )

    save_last_reasoning_effort(config.reasoning_effort)
    if log_level:
        config.log_level = log_level
    if log_file:
        config.log_file = str(log_file)
    from copex.config import configure_logging
    configure_logging(config)
    asyncio.run(_run_ralph(config, prompt, max_iterations, completion_promise))


async def _run_ralph(
    config: CopexConfig,
    prompt: str,
    max_iterations: int,
    completion_promise: str | None,
) -> None:
    """Run Ralph loop."""
    client = Copex(config)
    await client.start()
    logger = logging.getLogger(__name__)
    logger.info(
        "ralph.start",
        extra={
            "model": config.model.value,
            "reasoning_effort": config.reasoning_effort.value,
            "max_iterations": max_iterations,
            "has_promise": bool(completion_promise),
        },
    )

    def on_iteration(iteration: int, response: str) -> None:
        preview = response[:200] + "..." if len(response) > 200 else response
        console.print(Panel(
            preview,
            title=f"[bold]Iteration {iteration}[/bold]",
            border_style="blue",
        ))

    def on_complete(state: RalphState) -> None:
        console.print(Panel(
            f"Iterations: {state.iteration}\n"
            f"Reason: {state.completion_reason}",
            title="[bold green]Loop Complete[/bold green]",
            border_style="green",
        ))

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
        logger.info("ralph.stop")
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
                [cli_path, "--version"],
                capture_output=True, text=True, timeout=5
            )
            copilot_version = result.stdout.strip() or result.stderr.strip()
        except Exception:
            pass

    console.print(Panel(
        f"[bold]Copex Version:[/bold] {__version__}\n"
        f"[bold]Copilot CLI:[/bold] {cli_path or '[red]Not found[/red]'}\n"
        f"[bold]Copilot Version:[/bold] {copilot_version}\n"
        f"[bold]GitHub CLI:[/bold] {gh_path or '[red]Not found[/red]'}",
        title="Copex Status",
        border_style="blue",
    ))

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


@app.command("session")
def session_command(
    export: Annotated[
        Optional[str], typer.Option("--export", help="Export a session by ID")
    ] = None,
    format: Annotated[
        str, typer.Option("--format", help="Export format (json, md)")
    ] = "json",
    output: Annotated[
        Optional[Path], typer.Option("--output", "-o", help="Write export to file")
    ] = None,
) -> None:
    """Export saved sessions to JSON or Markdown."""
    if not export:
        console.print("[red]Provide --export <session-id>[/red]")
        raise typer.Exit(1)

    fmt = format.lower()
    if fmt not in {"json", "md"}:
        console.print("[red]Invalid format. Use json or md.[/red]")
        raise typer.Exit(1)

    from copex.persistence import SessionStore

    store = SessionStore()
    export_format = "markdown" if fmt == "md" else "json"
    try:
        content = store.export(export, format=export_format)
    except Exception as e:
        console.print(f"[red]Error exporting session: {e}[/red]")
        raise typer.Exit(1)

    if output:
        output.write_text(content, encoding="utf-8")
        console.print(f"[green]Exported session to {output}[/green]")
    else:
        console.print(content)


@app.command("plan")
def plan_command(
    task: Annotated[Optional[str], typer.Argument(help="Task to plan (optional with --resume)")] = None,
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
        Optional[Path], typer.Option("--load", "-l", help="Load plan from file instead of generating")
    ] = None,
    max_iterations: Annotated[
        int, typer.Option("--max-iterations", "-n", help="Max iterations per step (Ralph loop)")
    ] = 10,
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Model to use")
    ] = None,
    reasoning: Annotated[
        str, typer.Option("--reasoning", "-r", help="Reasoning effort level")
    ] = _DEFAULT_REASONING.value,
    progress: Annotated[
        str, typer.Option("--progress", help="Progress output format (terminal, rich, json, quiet)")
    ] = "terminal",
    log_level: Annotated[
        Optional[str], typer.Option("--log-level", help="Log level (debug, info, warning, error)")
    ] = None,
    log_file: Annotated[
        Optional[Path], typer.Option("--log-file", help="Write structured logs to file")
    ] = None,
) -> None:
    """
    Generate and optionally execute a step-by-step plan.

    Examples:
        copex plan "Build a REST API"              # Generate plan only
        copex plan "Build a REST API" --execute    # Generate and execute
        copex plan "Build a REST API" --review     # Generate, review, then execute
        copex plan --resume                        # Resume from .copex-state.json
        copex plan "Continue" --load plan.json -f3 # Resume from step 3
    """
    # Validate: need task OR resume OR load_plan
    if not task and not resume and not load_plan:
        console.print("[red]Error: Provide a task, --resume, or --load[/red]")
        raise typer.Exit(1)

    effective_model = model or _DEFAULT_MODEL.value
    try:
        config = CopexConfig(
            model=Model(effective_model),
            reasoning_effort=ReasoningEffort(reasoning),
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    save_last_reasoning_effort(config.reasoning_effort)
    if log_level:
        config.log_level = log_level
    if log_file:
        config.log_file = str(log_file)
    from copex.config import configure_logging
    configure_logging(config)
    asyncio.run(_run_plan(
        config=config,
        task=task or "",
        execute=execute or review or resume,  # --resume implies --execute
        review=review,
        resume=resume,
        output=output,
        from_step=from_step,
        load_plan=load_plan,
        max_iterations=max_iterations,
        progress=progress,
    ))


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
    progress: str = "terminal",
) -> None:
    """Run plan generation and optional execution."""
    client = Copex(config)
    await client.start()
    logger = logging.getLogger(__name__)
    logger.info(
        "plan.start",
        extra={
            "model": config.model.value,
            "reasoning_effort": config.reasoning_effort.value,
            "execute": execute,
            "review": review,
            "resume": resume,
            "from_step": from_step,
        },
    )

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
            icon_set = ASCIIIcons if config.ui_ascii_icons else Icons
            console.print(Panel(
                f"[bold]Resuming plan:[/bold] {state.task}\n"
                f"[dim]Started:[/dim] {state.started_at}\n"
                f"[dim]Completed steps:[/dim] {len(state.completed)}/{len(plan.steps)}\n"
                f"[dim]Resuming from step:[/dim] {from_step}",
                title=f"{icon_set.TOOL} Resume from Checkpoint",
                border_style="yellow",
            ))
        elif load_plan:
            # Load from plan file
            if not load_plan.exists():
                console.print(f"[red]Plan file not found: {load_plan}[/red]")
                raise typer.Exit(1)
            plan = Plan.load(load_plan)
            icon_set = ASCIIIcons if config.ui_ascii_icons else Icons
            console.print(f"[green]{icon_set.DONE} Loaded plan from {load_plan}[/green]\n")
        else:
            # Generate new plan
            icon_set = ASCIIIcons if config.ui_ascii_icons else Icons
            console.print(Panel(
                f"[bold]Generating plan for:[/bold]\n{task}",
                title=f"{icon_set.TOOL} Plan Mode",
                border_style="blue",
            ))

            plan = await executor.generate_plan(task)
            icon_set = ASCIIIcons if config.ui_ascii_icons else Icons
            console.print(f"\n[green]{icon_set.DONE} Generated {len(plan.steps)} steps[/green]\n")

        # Display plan
        _display_plan(plan, ascii_icons=config.ui_ascii_icons)

        # Save plan if requested
        if output:
            plan.save(output)
            icon_set = ASCIIIcons if config.ui_ascii_icons else Icons
            console.print(f"\n[green]{icon_set.DONE} Saved plan to {output}[/green]")

        # Execute if requested
        if execute:
            if review:
                if not typer.confirm("\nProceed with execution?"):
                    console.print("[yellow]Execution cancelled[/yellow]")
                    return

            console.print(f"\n[bold blue]Executing from step {from_step}...[/bold blue]\n")

            # Track execution timing
            plan_start_time = time.time()

            def on_step_start(step: PlanStep) -> None:
                total = len(plan.steps)
                icon_set = ASCIIIcons if config.ui_ascii_icons else Icons
                console.print(f"[blue]{icon_set.CLOCK} Step {step.number}/{total}:[/blue] {step.description}")

            def on_step_complete(step: PlanStep) -> None:
                # Format duration
                duration = step.duration_seconds or 0
                duration_str = _format_duration(duration)

                # Get result preview
                preview = (step.result or "")[:100]
                if len(step.result or "") > 100:
                    preview += "..."

                icon_set = ASCIIIcons if config.ui_ascii_icons else Icons
                console.print(f"[green]{icon_set.DONE} Step {step.number} complete ({duration_str})[/green]")
                if preview:
                    console.print(f"  [dim]— {preview}[/dim]")

                # Show ETA after 2+ steps completed
                completed_count = plan.completed_count
                if completed_count >= 2:
                    remaining_est = plan.estimate_remaining_seconds()
                    if remaining_est is not None:
                        console.print(f"  [dim cyan]Estimated remaining: ~{_format_duration(remaining_est)}[/dim cyan]")

                console.print()

            def on_error(step: PlanStep, error: Exception) -> bool:
                duration = step.duration_seconds or 0
                duration_str = _format_duration(duration)
                icon_set = ASCIIIcons if config.ui_ascii_icons else Icons
                console.print(f"[red]{icon_set.ERROR} Step {step.number} failed ({duration_str}): {error}[/red]")
                console.print("[dim]Checkpoint saved. Resume with: copex plan --resume[/dim]")
                return typer.confirm("Continue with next step?", default=False)

            from copex.progress import PlanProgressReporter
            if progress not in {"terminal", "rich", "json", "quiet"}:
                console.print("[red]Invalid progress format. Use terminal, rich, json, or quiet.[/red]")
                raise typer.Exit(1)

            use_reporter = progress not in {"terminal", "quiet"}
            reporter = PlanProgressReporter(plan, format=progress) if use_reporter else None

            await executor.execute_plan(
                plan,
                from_step=from_step,
                on_step_start=reporter.on_step_start if use_reporter else on_step_start,
                on_step_complete=reporter.on_step_complete if use_reporter else on_step_complete,
                on_error=reporter.on_error if use_reporter else on_error,
                save_checkpoints=True,
            )

            if reporter and progress != "quiet":
                reporter.finish()

            # Calculate total time
            total_time = time.time() - plan_start_time

            # Show enhanced summary
            _display_plan_summary_enhanced(plan, total_time, ascii_icons=config.ui_ascii_icons)

            # Save updated plan
            if output:
                plan.save(output)
                icon_set = ASCIIIcons if config.ui_ascii_icons else Icons
                console.print(f"\n[green]{icon_set.DONE} Updated plan saved to {output}[/green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled[/yellow]")
        console.print("[dim]Checkpoint saved. Resume with: copex plan --resume[/dim]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    finally:
        logger.info("plan.stop")
        await client.stop()


def _display_plan(plan: Plan, *, ascii_icons: bool = False) -> None:
    """Display plan steps."""
    icon_set = ASCIIIcons if ascii_icons else Icons
    for step in plan.steps:
        status_icon = {
            StepStatus.PENDING: f"{icon_set.BULLET}",
            StepStatus.RUNNING: f"{icon_set.TOOL}",
            StepStatus.COMPLETED: f"{icon_set.DONE}",
            StepStatus.FAILED: f"{icon_set.ERROR}",
            StepStatus.SKIPPED: f"{icon_set.ARROW_RIGHT}",
        }.get(step.status, icon_set.BULLET)
        console.print(f"{status_icon} [bold]Step {step.number}:[/bold] {step.description}")


def _display_plan_summary(plan: Plan, *, ascii_icons: bool = False) -> None:
    """Display plan execution summary."""
    completed = plan.completed_count
    failed = plan.failed_count
    total = len(plan.steps)

    icon_set = ASCIIIcons if ascii_icons else Icons
    if plan.is_complete and failed == 0:
        console.print(Panel(
            f"[green]All {total} steps completed successfully![/green]",
            title=f"{icon_set.DONE} Plan Complete",
            border_style="green",
        ))
    elif failed > 0:
        console.print(Panel(
            f"Completed: {completed}/{total}\nFailed: {failed}",
            title=f"{icon_set.WARNING} Plan Incomplete",
            border_style="yellow",
        ))
    else:
        console.print(Panel(
            f"Completed: {completed}/{total}",
            title=f"{icon_set.TOOL} Progress",
            border_style="blue",
        ))


def _display_plan_summary_enhanced(plan: Plan, total_time: float, *, ascii_icons: bool = False) -> None:
    """Display enhanced plan execution summary with timing and tokens."""
    completed = plan.completed_count
    failed = plan.failed_count
    total = len(plan.steps)

    # Build summary lines
    lines = []

    icon_set = ASCIIIcons if ascii_icons else Icons
    if plan.is_complete and failed == 0:
        lines.append(f"[green]{icon_set.DONE} {completed}/{total} steps completed successfully![/green]")
    elif failed > 0:
        lines.append(f"[yellow]{icon_set.WARNING} {completed}/{total} steps completed, {failed} failed[/yellow]")
    else:
        lines.append(f"[blue]{icon_set.TOOL} {completed}/{total} steps completed[/blue]")

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
        title = f"{icon_set.DONE} Plan Complete"
        border = "green"
    elif failed > 0:
        title = f"{icon_set.WARNING} Plan Incomplete"
        border = "yellow"
    else:
        title = f"{icon_set.TOOL} Progress"
        border = "blue"

    console.print(Panel(
        "\n".join(lines),
        title=title,
        border_style=border,
    ))


if __name__ == "__main__":
    app()
