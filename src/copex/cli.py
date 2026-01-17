"""CLI interface for Copex."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from copex.client import Copex, StreamChunk
from copex.config import CopexConfig
from copex.models import Model, ReasoningEffort
from copex.ralph import RalphWiggum, RalphState

app = typer.Typer(
    name="copex",
    help="Copilot Extended - Resilient wrapper with auto-retry and Ralph Wiggum loops.",
    no_args_is_help=True,
)
console = Console()


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


@app.command()
def chat(
    prompt: Annotated[Optional[str], typer.Argument(help="Prompt to send (or read from stdin)")] = None,
    model: Annotated[
        str, typer.Option("--model", "-m", help="Model to use")
    ] = Model.GPT_5_2_CODEX.value,
    reasoning: Annotated[
        str, typer.Option("--reasoning", "-r", help="Reasoning effort level")
    ] = ReasoningEffort.XHIGH.value,
    max_retries: Annotated[
        int, typer.Option("--max-retries", help="Maximum retry attempts")
    ] = 5,
    no_stream: Annotated[
        bool, typer.Option("--no-stream", help="Disable streaming output")
    ] = False,
    show_reasoning: Annotated[
        bool, typer.Option("--show-reasoning", help="Show model reasoning")
    ] = False,
    config_file: Annotated[
        Optional[Path], typer.Option("--config", "-c", help="Config file path")
    ] = None,
    raw: Annotated[
        bool, typer.Option("--raw", help="Output raw text without formatting")
    ] = False,
) -> None:
    """Send a prompt to Copilot with automatic retry on errors."""
    # Load config
    if config_file and config_file.exists():
        config = CopexConfig.from_file(config_file)
    else:
        config = CopexConfig()

    # Override with CLI options
    try:
        config.model = Model(model)
    except ValueError:
        console.print(f"[red]Invalid model: {model}[/red]")
        raise typer.Exit(1)

    try:
        config.reasoning_effort = ReasoningEffort(reasoning)
    except ValueError:
        console.print(f"[red]Invalid reasoning effort: {reasoning}[/red]")
        raise typer.Exit(1)

    config.retry.max_retries = max_retries
    config.streaming = not no_stream

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

    try:
        await client.start()

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
        await client.stop()


async def _stream_response(
    client: Copex, prompt: str, show_reasoning: bool
) -> None:
    """Stream response with live updates."""
    content = ""
    reasoning = ""
    retries = 0

    def on_chunk(chunk: StreamChunk) -> None:
        nonlocal content, reasoning
        if chunk.type == "message":
            if chunk.is_final:
                content = chunk.content or content
            else:
                content += chunk.delta
        elif chunk.type == "reasoning":
            if chunk.is_final:
                reasoning = chunk.content or reasoning
            else:
                reasoning += chunk.delta
        elif chunk.type == "system":
            # Retry notification
            console.print(f"[yellow]{chunk.delta.strip()}[/yellow]")

    with Live(console=console, refresh_per_second=10) as live:
        response = await client.send(prompt, on_chunk=on_chunk)
        retries = response.retries

        # Update live display
        output = Text()
        if show_reasoning and reasoning:
            output.append("─── Reasoning ───\n", style="dim")
            output.append(reasoning + "\n\n", style="dim italic")
            output.append("─── Response ───\n", style="dim")
        output.append(content)
        live.update(output)

    # Final formatted output
    console.print()
    if show_reasoning and reasoning:
        console.print(Panel(
            Markdown(reasoning),
            title="[dim]Reasoning[/dim]",
            border_style="dim",
        ))
    console.print(Markdown(content))

    if retries > 0:
        console.print(f"\n[dim]Completed with {retries} retries[/dim]")


@app.command()
def models() -> None:
    """List available models."""
    console.print("[bold]Available Models:[/bold]\n")
    for model in Model:
        console.print(f"  • {model.value}")


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
        "model": Model.GPT_5_2_CODEX.value,
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
    }

    with open(path, "wb") as f:
        tomli_w.dump(config, f)

    console.print(f"[green]Created config at:[/green] {path}")


@app.command()
def interactive(
    model: Annotated[
        str, typer.Option("--model", "-m", help="Model to use")
    ] = Model.GPT_5_2_CODEX.value,
    reasoning: Annotated[
        str, typer.Option("--reasoning", "-r", help="Reasoning effort level")
    ] = ReasoningEffort.XHIGH.value,
) -> None:
    """Start an interactive chat session."""
    try:
        config = CopexConfig(
            model=Model(model),
            reasoning_effort=ReasoningEffort(reasoning),
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold]Copex[/bold] - Copilot Extended\n"
        f"Model: {config.model.value}\n"
        f"Reasoning: {config.reasoning_effort.value}\n"
        f"Auto-retry: {config.retry.max_retries}x\n"
        f"Ralph Wiggum: Active 🧠",
        title="Interactive Mode",
        border_style="blue",
    ))
    console.print("[dim]Type 'exit' or Ctrl+C to quit, 'new' for fresh session[/dim]\n")

    asyncio.run(_interactive_loop(config))


async def _interactive_loop(config: CopexConfig) -> None:
    """Run interactive chat loop."""
    client = Copex(config)
    await client.start()

    try:
        while True:
            try:
                prompt = console.input("[bold blue]>[/bold blue] ")
            except EOFError:
                break

            prompt = prompt.strip()
            if not prompt:
                continue
            if prompt.lower() == "exit":
                break
            if prompt.lower() == "new":
                client.new_session()
                console.print("[dim]Started new session[/dim]\n")
                continue

            console.print()
            try:
                await _stream_response(client, prompt, show_reasoning=False)
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
            console.print()

    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye![/yellow]")
    finally:
        await client.stop()


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
        str, typer.Option("--model", "-m", help="Model to use")
    ] = Model.GPT_5_2_CODEX.value,
    reasoning: Annotated[
        str, typer.Option("--reasoning", "-r", help="Reasoning effort level")
    ] = ReasoningEffort.XHIGH.value,
) -> None:
    """
    Start a Ralph Wiggum loop - iterative AI development.

    The same prompt is fed to the AI repeatedly. The AI sees its previous
    work in conversation history and iteratively improves until complete.

    Example:
        copex ralph "Build a REST API with CRUD and tests" --promise "ALL TESTS PASSING" -n 20
    """
    try:
        config = CopexConfig(
            model=Model(model),
            reasoning_effort=ReasoningEffort(reasoning),
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold]Ralph Wiggum Loop[/bold]\n"
        f"Model: {config.model.value}\n"
        f"Reasoning: {config.reasoning_effort.value}\n"
        f"Max iterations: {max_iterations}\n"
        f"Completion promise: {completion_promise or '(none)'}",
        title="🔄 Starting Loop",
        border_style="yellow",
    ))

    if completion_promise:
        console.print(
            f"\n[dim]To complete, the AI must output: "
            f"[yellow]<promise>{completion_promise}</promise>[/yellow][/dim]\n"
        )

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


__version__ = "0.1.0"


if __name__ == "__main__":
    app()
