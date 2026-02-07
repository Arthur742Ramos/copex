"""Render JSONL session logs with readable, colorized output."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TextIO

from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from copex.ui import Icons, Theme


@dataclass
class RenderState:
    """Track buffers while replaying JSONL logs."""

    message_buffer: str = ""
    reasoning_buffer: str = ""
    tool_names: dict[str, str] = field(default_factory=dict)


def render_jsonl(stream: TextIO, console: Console) -> None:
    """Render JSONL log entries from a stream."""
    state = RenderState()
    for line_no, line in enumerate(stream, 1):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            console.print(Text(f"Invalid JSON on line {line_no}: {exc}", style=Theme.ERROR))
            continue
        _render_event(event, state, console)

    _flush_buffers(state, console, None)


def _render_event(event: Any, state: RenderState, console: Console) -> None:
    if not isinstance(event, dict):
        console.print(Text(json.dumps(event, ensure_ascii=False), style=Theme.MUTED))
        return

    event_type = _extract_event_type(event)
    data = _extract_data(event)
    timestamp = _extract_timestamp(event, data)
    tokens, cost = _extract_token_usage(event, data)
    category = _classify_event(event_type, event, data)

    if category == "user":
        _flush_buffers(state, console, timestamp)
        content = _extract_content(data or event)
        _print_section(
            console,
            "User",
            content,
            timestamp,
            Theme.SUCCESS,
            markdown=False,
            content_style=Theme.MESSAGE,
        )
        _print_tokens(console, tokens, cost)
        console.print()
        return

    if category == "assistant_message_delta":
        state.message_buffer += _extract_delta_content(data or event)
        return

    if category == "assistant_reasoning_delta":
        state.reasoning_buffer += _extract_delta_content(data or event)
        return

    if category == "assistant_message":
        content = _extract_content(data or event) or state.message_buffer
        state.message_buffer = ""
        _print_section(console, "Assistant", content, timestamp, Theme.PRIMARY, markdown=True)
        _print_tokens(console, tokens, cost)
        console.print()
        return

    if category == "assistant_reasoning":
        content = _extract_content(data or event) or state.reasoning_buffer
        state.reasoning_buffer = ""
        _print_section(
            console,
            "Reasoning",
            content,
            timestamp,
            Theme.REASONING,
            markdown=False,
            content_style=Theme.REASONING,
        )
        _print_tokens(console, tokens, cost)
        console.print()
        return

    if category == "tool_call":
        tool_calls = _extract_tool_calls(data or event)
        if tool_calls:
            for call in tool_calls:
                _render_tool_call(call, timestamp, state, console)
        else:
            _render_tool_call(data or event, timestamp, state, console)
        _print_tokens(console, tokens, cost)
        return

    if category == "tool_partial":
        name = _extract_tool_name(data or event, state)
        output = _extract_tool_output(data or event)
        _print_tool_partial(console, name, output, timestamp)
        _print_tokens(console, tokens, cost)
        return

    if category == "tool_result":
        results = _extract_tool_results(data or event)
        if results:
            for result in results:
                _render_tool_result(result, timestamp, state, console)
        else:
            _render_tool_result(data or event, timestamp, state, console)
        _print_tokens(console, tokens, cost)
        return

    if category == "usage":
        _print_tokens(console, tokens, cost, timestamp=timestamp)
        return

    if category == "progress":
        _print_progress(console, data or event, timestamp)
        return

    if category == "turn_end":
        _flush_buffers(state, console, timestamp)
        _print_tokens(console, tokens, cost)
        return

    if category == "error":
        message = _extract_error_message(data or event)
        _print_error(console, message, timestamp)
        _print_tokens(console, tokens, cost)
        return

    if category == "system":
        message = _extract_content(data or event)
        _print_system(console, message, timestamp)
        _print_tokens(console, tokens, cost)
        return

    console.print(Text(json.dumps(event, ensure_ascii=False, indent=2), style=Theme.MUTED))


def _extract_event_type(event: dict[str, Any]) -> str | None:
    value = event.get("type") or event.get("event") or event.get("event_type")
    if value is None:
        return None
    return str(value)


def _extract_data(event: dict[str, Any]) -> dict[str, Any] | None:
    data = event.get("data") or event.get("payload")
    if isinstance(data, dict):
        return data
    return None


def _classify_event(
    event_type: str | None, event: dict[str, Any], data: dict[str, Any] | None
) -> str:
    if event_type:
        t = event_type.lower()
    else:
        t = ""

    if t == "progress":
        return "progress"

    if "usage" in t:
        return "usage"

    if "tool.execution_partial_result" in t or "partial_result" in t or t.endswith("tool_partial"):
        return "tool_partial"

    if "tool.execution_complete" in t or "tool_result" in t or t.endswith("tool_result"):
        return "tool_result"

    if "tool.execution_start" in t or "tool.call" in t or "tool_call" in t:
        return "tool_call"

    if "assistant.reasoning_delta" in t or "reasoning_delta" in t:
        return "assistant_reasoning_delta"

    if "assistant.reasoning" in t or t.endswith("reasoning") or t == "reasoning":
        return "assistant_reasoning"

    if "assistant.message_delta" in t or "message_delta" in t:
        return "assistant_message_delta"

    if "user.message" in t or ("user" in t and "message" in t):
        return "user"

    if "assistant.message" in t or t.endswith("message") or t == "message":
        return "assistant_message"

    if "turn_end" in t or "session.idle" in t or t == "idle":
        return "turn_end"

    if "error" in t:
        return "error"

    if "system" in t:
        return "system"

    role = event.get("role")
    if role and event.get("content"):
        if str(role).lower() == "user":
            return "user"
        if str(role).lower() == "assistant":
            return "assistant_message"

    return "unknown"


def _extract_timestamp(event: dict[str, Any], data: dict[str, Any] | None) -> str | None:
    for payload in (event, data or {}):
        if not isinstance(payload, dict):
            continue
        value = payload.get("timestamp") or payload.get("time") or payload.get("created_at")
        if value is None:
            continue
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value).isoformat()
        return str(value)
    return None


def _extract_content(payload: dict[str, Any]) -> str:
    for key in ("content", "text", "message", "prompt", "response", "transformed_content"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    value = payload.get("content")
    if value is not None:
        return str(value)
    return ""


def _extract_delta_content(payload: dict[str, Any]) -> str:
    for key in ("delta_content", "delta", "content_delta", "text_delta"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    value = payload.get("delta")
    return str(value) if value is not None else ""


def _extract_tool_calls(payload: dict[str, Any]) -> list[dict[str, Any]]:
    calls = payload.get("tool_calls")
    if isinstance(calls, list):
        return [c for c in calls if isinstance(c, dict)]
    return []


def _extract_tool_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = payload.get("tool_results")
    if isinstance(results, list):
        return [r for r in results if isinstance(r, dict)]
    return []


def _extract_tool_name(payload: dict[str, Any], state: RenderState | None = None) -> str:
    for key in ("tool_name", "name", "tool"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    tool_id = _extract_tool_id(payload)
    if tool_id and state:
        return state.tool_names.get(tool_id, "unknown")
    return "unknown"


def _extract_tool_id(payload: dict[str, Any]) -> str | None:
    for key in ("tool_call_id", "tool_id", "id"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _extract_tool_args(payload: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("arguments", "args", "tool_args"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return None


def _extract_tool_output(payload: dict[str, Any]) -> str | None:
    for key in ("partial_output", "output", "result", "content"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict) and "content" in value:
            inner = value.get("content")
            if isinstance(inner, str) and inner:
                return inner
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
    return None


def _extract_error_message(payload: dict[str, Any]) -> str:
    for key in ("message", "error", "detail"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return json.dumps(payload, ensure_ascii=False)


def _extract_token_usage(
    event: dict[str, Any], data: dict[str, Any] | None
) -> tuple[dict[str, int] | None, float | None]:
    tokens: dict[str, int] = {}
    cost: float | None = None

    for payload in (event, data or {}):
        if not isinstance(payload, dict):
            continue
        usage = payload.get("usage")
        if isinstance(usage, dict):
            _update_tokens(tokens, usage)
        _update_tokens(tokens, payload)
        if cost is None:
            cost = _coerce_float(payload.get("cost"))

    if tokens:
        if "total" not in tokens and ("input" in tokens or "output" in tokens):
            tokens["total"] = tokens.get("input", 0) + tokens.get("output", 0)
        return tokens, cost
    return None, cost


def _update_tokens(target: dict[str, int], payload: dict[str, Any]) -> None:
    input_tokens = _coerce_int(payload.get("prompt_tokens") or payload.get("input_tokens"))
    output_tokens = _coerce_int(payload.get("completion_tokens") or payload.get("output_tokens"))
    total_tokens = _coerce_int(payload.get("total_tokens"))
    if input_tokens is not None:
        target["input"] = input_tokens
    if output_tokens is not None:
        target["output"] = output_tokens
    if total_tokens is not None:
        target["total"] = total_tokens
    tokens_payload = payload.get("tokens")
    if isinstance(tokens_payload, dict):
        _update_tokens(target, tokens_payload)


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _print_section(
    console: Console,
    label: str,
    content: str,
    timestamp: str | None,
    style: str,
    *,
    markdown: bool,
    content_style: str | None = None,
) -> None:
    header = Text()
    if timestamp:
        header.append(f"{timestamp} ", style=Theme.MUTED)
    header.append(label, style=f"bold {style}")
    console.print(header)
    if content:
        if markdown:
            console.print(Markdown(content))
        else:
            console.print(Text(content, style=content_style or style))


def _print_tool_partial(
    console: Console, name: str, output: str | None, timestamp: str | None
) -> None:
    header = Text()
    if timestamp:
        header.append(f"{timestamp} ", style=Theme.MUTED)
    header.append(f"{Icons.TOOL} {name}", style=Theme.WARNING)
    header.append(" (partial)", style=Theme.MUTED)
    console.print(header)
    if output:
        console.print(Text(output, style=Theme.MUTED))


def _render_tool_call(
    payload: dict[str, Any], timestamp: str | None, state: RenderState, console: Console
) -> None:
    name = _extract_tool_name(payload, state)
    args = _extract_tool_args(payload)
    tool_id = _extract_tool_id(payload)
    if tool_id and name:
        state.tool_names[tool_id] = name
    header = Text()
    if timestamp:
        header.append(f"{timestamp} ", style=Theme.MUTED)
    header.append(f"{Icons.TOOL} {name}", style=f"bold {Theme.WARNING}")
    preview = _format_arg_preview(args)
    if preview:
        header.append(f" {preview}", style=Theme.MUTED)
    console.print(header)
    if args:
        console.print(Text(json.dumps(args, ensure_ascii=False, indent=2), style=Theme.MUTED))


def _render_tool_result(
    payload: dict[str, Any], timestamp: str | None, state: RenderState, console: Console
) -> None:
    name = _extract_tool_name(payload, state)
    success = payload.get("success")
    if isinstance(success, bool):
        success_flag = success
    else:
        success_flag = payload.get("error") is None
    duration = _coerce_float(payload.get("duration"))
    output = _extract_tool_output(payload)
    icon = Icons.DONE if success_flag else Icons.ERROR
    style = Theme.SUCCESS if success_flag else Theme.ERROR

    header = Text()
    if timestamp:
        header.append(f"{timestamp} ", style=Theme.MUTED)
    header.append(f"{icon} {name}", style=f"bold {style}")
    if duration is not None:
        header.append(f" ({duration:.1f}s)", style=Theme.MUTED)
    console.print(header)
    if output:
        console.print(Text(output, style=Theme.MESSAGE))


def _print_tokens(
    console: Console,
    tokens: dict[str, int] | None,
    cost: float | None,
    *,
    timestamp: str | None = None,
) -> None:
    if not tokens and cost is None:
        return
    parts: list[str] = []
    if tokens:
        if "input" in tokens:
            parts.append(f"in={tokens['input']:,}")
        if "output" in tokens:
            parts.append(f"out={tokens['output']:,}")
        if "total" in tokens:
            parts.append(f"total={tokens['total']:,}")
    if cost is not None:
        parts.append(f"cost=${cost:.4f}")
    summary = "Tokens: " + ", ".join(parts) if parts else "Tokens"
    line = Text()
    if timestamp:
        line.append(f"{timestamp} ", style=Theme.MUTED)
    line.append(summary, style=Theme.MUTED)
    console.print(line)


def _print_progress(console: Console, payload: dict[str, Any], timestamp: str | None) -> None:
    completed = payload.get("completed")
    total = payload.get("total")
    percent = payload.get("percent")
    message = payload.get("message") or ""

    header = Text()
    if timestamp:
        header.append(f"{timestamp} ", style=Theme.MUTED)
    header.append(f"{Icons.INFO} Progress", style=Theme.INFO)
    details = []
    if completed is not None and total is not None:
        details.append(f"{completed}/{total}")
    if percent is not None:
        details.append(f"{percent}%")
    if details:
        header.append(f" ({', '.join(details)})", style=Theme.MUTED)
    console.print(header)
    if message:
        console.print(Text(str(message), style=Theme.MUTED))


def _print_error(console: Console, message: str, timestamp: str | None) -> None:
    header = Text()
    if timestamp:
        header.append(f"{timestamp} ", style=Theme.MUTED)
    header.append(f"{Icons.ERROR} Error", style=f"bold {Theme.ERROR}")
    console.print(header)
    console.print(Text(message, style=Theme.ERROR))


def _print_system(console: Console, message: str, timestamp: str | None) -> None:
    header = Text()
    if timestamp:
        header.append(f"{timestamp} ", style=Theme.MUTED)
    header.append(f"{Icons.INFO} System", style=Theme.INFO)
    console.print(header)
    if message:
        console.print(Text(message, style=Theme.MUTED))


def _flush_buffers(state: RenderState, console: Console, timestamp: str | None) -> None:
    if state.reasoning_buffer:
        _print_section(
            console,
            "Reasoning",
            state.reasoning_buffer,
            timestamp,
            Theme.REASONING,
            markdown=False,
            content_style=Theme.REASONING,
        )
        state.reasoning_buffer = ""
        console.print()
    if state.message_buffer:
        _print_section(
            console, "Assistant", state.message_buffer, timestamp, Theme.PRIMARY, markdown=True
        )
        state.message_buffer = ""
        console.print()


def _format_arg_preview(args: dict[str, Any] | None) -> str:
    if not args:
        return ""
    if "path" in args:
        return f"path={_truncate(str(args['path']))}"
    if "command" in args:
        return f"cmd={_truncate(str(args['command']))}"
    if "pattern" in args:
        return f"pattern={_truncate(str(args['pattern']))}"
    return ""


def _truncate(value: str, max_len: int = 60) -> str:
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."
