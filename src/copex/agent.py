"""Agent loop for Copex — iterative prompt/tool-call/respond cycle.

Wraps either the SDK client (Copex) or CLI client (CopilotCLI) in a
structured agent loop that yields JSON-serializable results per turn.
Designed for subprocess consumption by orchestrators like Squad.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from copex.streaming import StreamChunk

logger = logging.getLogger(__name__)

DEFAULT_MAX_TURNS = 10


@runtime_checkable
class AgentClient(Protocol):
    """Minimal interface that both Copex and CopilotCLI satisfy."""

    async def start(self) -> None: ...
    async def stop(self) -> None: ...

    async def send(
        self,
        prompt: str,
        *,
        tools: list[Any] | None = None,
        on_chunk: Any | None = None,
        metrics: Any | None = None,
    ) -> Any: ...


@dataclass
class AgentTurn:
    """Result of a single agent turn."""

    turn: int
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str | None = "end_turn"  # end_turn | max_turns | error | None
    error: str | None = None
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "content": self.content,
            "tool_calls": self.tool_calls,
            "stop_reason": self.stop_reason,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 1),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class AgentResult:
    """Final result after the agent loop completes."""

    turns: list[AgentTurn] = field(default_factory=list)
    final_content: str = ""
    total_turns: int = 0
    total_duration_ms: float = 0.0
    stop_reason: str = "end_turn"  # end_turn | max_turns | error
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "turns": [t.to_dict() for t in self.turns],
            "total_duration_ms": round(self.total_duration_ms, 1),
            "total_turns": self.total_turns,
            "final_content": self.final_content,
            "stop_reason": self.stop_reason,
            "error": self.error,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def _extract_tool_calls(response: Any) -> list[dict[str, Any]]:
    """Extract tool calls from on_chunk captures and response raw_events."""
    tool_calls: list[dict[str, Any]] = []
    raw_events = getattr(response, "raw_events", None) or []
    for event in raw_events:
        if not isinstance(event, dict):
            continue
        if event.get("type") == "tool.call":
            data = event.get("data")
            if isinstance(data, dict):
                tc: dict[str, Any] = {"name": data.get("name", "unknown")}
                if "arguments" in data:
                    tc["arguments"] = data["arguments"]
                tool_calls.append(tc)
    return tool_calls


class AgentSession:
    """Iterative agent loop over a Copex or CopilotCLI client.

    The loop sends a prompt, collects the response (including any tool calls
    the model makes autonomously via the SDK / CLI), and decides whether to
    continue for another turn.  The agent continues when tool calls were
    observed, indicating the model wants to keep working, up to *max_turns*.

    Usage::

        session = AgentSession(client, max_turns=10)
        result = await session.run("Fix the failing test")
        for turn in result.turns:
            print(turn.to_json())
    """

    def __init__(
        self,
        client: Any,
        *,
        max_turns: int = DEFAULT_MAX_TURNS,
        model: str | None = None,
        continue_prompt: str = "Continue.",
    ) -> None:
        self._client = client
        self.max_turns = max_turns
        self.model = model
        self._continue_prompt = continue_prompt
        self._started = False

    async def start(self) -> None:
        if not self._started:
            await self._client.start()
            self._started = True

    async def stop(self) -> None:
        if self._started:
            await self._client.stop()
            self._started = False

    async def __aenter__(self) -> AgentSession:
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.stop()

    async def run(self, prompt: str) -> AgentResult:
        """Execute the agent loop and return the full result."""
        result = AgentResult()
        start_time = time.monotonic()

        for turn_num in range(1, self.max_turns + 1):
            turn_prompt = prompt if turn_num == 1 else self._continue_prompt
            turn = await self._execute_turn(turn_num, turn_prompt)
            result.turns.append(turn)

            if turn.stop_reason == "error":
                result.stop_reason = "error"
                result.error = turn.error
                result.final_content = turn.content
                break

            if turn.tool_calls:
                # Intermediate turn — model is still working.
                continue

            # No tool calls — the model is done.
            result.final_content = turn.content
            result.stop_reason = "end_turn"
            break
        else:
            # Exhausted max_turns
            result.stop_reason = "max_turns"
            if result.turns:
                result.final_content = result.turns[-1].content

        result.total_turns = len(result.turns)
        result.total_duration_ms = (time.monotonic() - start_time) * 1000
        return result

    async def run_streaming(self, prompt: str) -> AsyncIterator[AgentTurn]:
        """Execute the agent loop, yielding each AgentTurn as it completes."""
        for turn_num in range(1, self.max_turns + 1):
            turn_prompt = prompt if turn_num == 1 else self._continue_prompt
            turn = await self._execute_turn(turn_num, turn_prompt)
            yield turn

            if turn.stop_reason == "error":
                break

            if not turn.tool_calls:
                break

    async def _execute_turn(self, turn_num: int, prompt: str) -> AgentTurn:
        """Execute a single turn: send prompt, collect response and tool calls."""
        chunk_tool_calls: list[dict[str, Any]] = []
        _active: dict[str | None, dict[str, Any]] = {}

        def on_chunk(chunk: StreamChunk) -> None:
            if chunk.type == "tool_call":
                tc: dict[str, Any] = {
                    "name": chunk.tool_name or "unknown",
                    "arguments": chunk.tool_args or {},
                }
                chunk_tool_calls.append(tc)
                _active[chunk.tool_id] = tc
            elif chunk.type == "tool_result":
                tc = _active.get(chunk.tool_id)
                if tc is not None:
                    tc["result"] = chunk.tool_result
                    tc["success"] = chunk.tool_success
                    if chunk.tool_duration is not None:
                        tc["duration"] = chunk.tool_duration

        turn_start = time.monotonic()
        try:
            response = await self._client.send(prompt, on_chunk=on_chunk)
            content = response.content if hasattr(response, "content") else str(response)
            duration_ms = (time.monotonic() - turn_start) * 1000

            # Merge tool calls: prefer on_chunk captures, fall back to raw_events.
            tool_calls = chunk_tool_calls or _extract_tool_calls(response)

            # Determine stop reason for this turn.
            if tool_calls:
                if turn_num >= self.max_turns:
                    stop_reason: str | None = "max_turns"
                else:
                    stop_reason = None  # Intermediate — loop continues.
            else:
                stop_reason = "end_turn"

            return AgentTurn(
                turn=turn_num,
                content=content,
                tool_calls=tool_calls,
                stop_reason=stop_reason,
                duration_ms=duration_ms,
            )
        except Exception as exc:
            duration_ms = (time.monotonic() - turn_start) * 1000
            logger.error("Agent turn %d failed: %s", turn_num, exc)
            return AgentTurn(
                turn=turn_num,
                content="",
                tool_calls=chunk_tool_calls,
                stop_reason="error",
                error=str(exc),
                duration_ms=duration_ms,
            )
