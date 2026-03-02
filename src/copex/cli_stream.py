"""Shared stream response handlers for CLI modes."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass

from rich.console import Console
from rich.live import Live

from copex.client import Copex, StreamChunk
from copex.ui import ActivityType, CopexUI, ToolCallInfo, print_retry

console = Console()


def configure_stream_console(shared_console: Console) -> None:
    """Bind stream rendering to CLI's shared console instance."""
    global console
    console = shared_console


@dataclass
class StreamHandler:
    """Shared stream chunk handling for rich/plain/interactive output."""

    on_message: Callable[[StreamChunk], None]
    on_reasoning: Callable[[StreamChunk], None]
    on_tool_call: Callable[[StreamChunk], None]
    on_tool_result: Callable[[StreamChunk], None]
    on_system: Callable[[StreamChunk], None]

    def handle(self, chunk: StreamChunk) -> None:
        if chunk.type == "message":
            self.on_message(chunk)
        elif chunk.type == "reasoning":
            self.on_reasoning(chunk)
        elif chunk.type == "tool_call":
            self.on_tool_call(chunk)
        elif chunk.type == "tool_result":
            self.on_tool_result(chunk)
        elif chunk.type == "system":
            self.on_system(chunk)


async def _stream_response(client: Copex, prompt: str, show_reasoning: bool) -> str:
    """Stream response with beautiful live updates. Returns the final content."""
    ui = CopexUI(
        console, theme=client.config.ui_theme, density=client.config.ui_density, show_all_tools=True
    )
    ui.reset(model=client.config.model.value)
    ui.set_activity(ActivityType.THINKING)

    live_display: Live | None = None
    final_content = ""

    handler = StreamHandler(
        on_message=lambda chunk: ui.set_final_content(
            chunk.content or ui.state.message, ui.state.reasoning
        ) if chunk.is_final else ui.add_message(chunk.delta),
        on_reasoning=lambda chunk: (
            None
            if not show_reasoning or chunk.is_final
            else ui.add_reasoning(chunk.delta)
        ),
        on_tool_call=lambda chunk: ui.add_tool_call(
            ToolCallInfo(
                tool_id=chunk.tool_id or "",
                name=chunk.tool_name or "unknown",
                arguments=chunk.tool_args or {},
                status="running",
            )
        ),
        on_tool_result=lambda chunk: ui.update_tool_call(
            chunk.tool_id,
            chunk.tool_name or "unknown",
            "success" if chunk.tool_success is not False else "error",
            result=chunk.tool_result,
            duration=chunk.tool_duration,
        ),
        on_system=lambda chunk: (
            ui.increment_retries(),
            print_retry(console, ui.state.retries, client.config.retry.max_retries, chunk.delta),
        ),
    )

    def on_chunk(chunk: StreamChunk) -> None:
        handler.handle(chunk)
        if live_display:
            live_display.update(ui.build_live_display())

    with Live(console=console, refresh_per_second=10, transient=True) as live:
        live_display = live
        live.update(ui.build_live_display())
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
        ui.set_context_usage(response.context_used_tokens, response.context_budget_tokens)
        final_content = final_message

    # Print final beautiful output
    console.print(ui.build_final_display())
    return final_content


async def _stream_response_plain(client: Copex, prompt: str) -> None:
    """Stream response as plain text."""
    content = ""
    retries = 0

    def _on_message(chunk: StreamChunk) -> None:
        nonlocal content
        if chunk.is_final:
            if chunk.content:
                content = chunk.content
            return
        if chunk.delta:
            content += chunk.delta
            sys.stdout.write(chunk.delta)
            sys.stdout.flush()

    handler = StreamHandler(
        on_message=_on_message,
        on_reasoning=lambda chunk: None,
        on_tool_call=lambda chunk: None,
        on_tool_result=lambda chunk: None,
        on_system=lambda chunk: console.print(f"[yellow]{chunk.delta.strip()}[/yellow]"),
    )

    def on_chunk(chunk: StreamChunk) -> None:
        handler.handle(chunk)

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

    handler = StreamHandler(
        on_message=lambda chunk: ui.set_final_content(
            chunk.content or ui.state.message, ui.state.reasoning
        ) if chunk.is_final else ui.add_message(chunk.delta),
        on_reasoning=lambda chunk: None if chunk.is_final else ui.add_reasoning(chunk.delta),
        on_tool_call=lambda chunk: ui.add_tool_call(
            ToolCallInfo(
                tool_id=chunk.tool_id or "",
                name=chunk.tool_name or "unknown",
                arguments=chunk.tool_args or {},
                status="running",
            )
        ),
        on_tool_result=lambda chunk: ui.update_tool_call(
            chunk.tool_id,
            chunk.tool_name or "unknown",
            "success" if chunk.tool_success is not False else "error",
            result=chunk.tool_result,
            duration=chunk.tool_duration,
        ),
        on_system=lambda chunk: ui.increment_retries(),
    )

    def on_chunk(chunk: StreamChunk) -> None:
        handler.handle(chunk)
        if live_display:
            live_display.update(ui.build_live_display())

    with Live(console=console, refresh_per_second=10, transient=True) as live:
        live_display = live
        live.update(ui.build_live_display())
        response = await client.send(prompt, on_chunk=on_chunk)
        # Prefer streamed content over response object (which may have stale fallback)
        final_message = ui.state.message if ui.state.message else response.content
        final_reasoning = ui.state.reasoning if ui.state.reasoning else response.reasoning
        ui.set_final_content(final_message, final_reasoning)
        ui.state.retries = response.retries
        ui.set_usage(response.prompt_tokens, response.completion_tokens)
        ui.set_context_usage(response.context_used_tokens, response.context_budget_tokens)

    # Finalize the assistant response to history
    ui.finalize_assistant_response()

    console.print(ui.build_final_display())
    console.print()  # Extra spacing
