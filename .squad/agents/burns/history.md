# Project Context

- **Owner:** Arthur
- **Project:** Copex — Resilient Python wrapper for GitHub Copilot SDK with auto-retry, exponential backoff, Ralph Wiggum loops, streaming, session persistence, metrics, parallel tools, and MCP integration.
- **Stack:** Python 3.10+, asyncio, Typer CLI, Rich, Pydantic, pytest, GitHub Copilot SDK
- **Created:** 2026-03-01

## Learnings

<!-- Append new learnings below. Each entry is something lasting about the project. -->

### 2026-03-01: Agent Module Review

**Context:** Reviewed Frink's `agent.py` implementation for subprocess consumption. Module provides `AgentSession`, `AgentTurn`, `AgentResult` with JSON-serializable output for orchestrators.

**Public API Exports:**
- `AgentSession` — Wrapper for multi-turn agent loops
- `AgentTurn` — Single turn result (content, tool_calls, stop_reason, error, duration)
- `AgentResult` — Final aggregated result across all turns

**Architecture:**
- Minimal dependency surface: only imports `StreamChunk` from `copex.streaming`
- `AgentClient` protocol allows both `Copex` and `CopilotCLI` without circular deps
- Clean separation: no UI, no config, pure agent loop logic
- CLI integration in `cli.py` handles presentation and config

**Design Quality:**
- Dataclasses with `to_dict()` / `to_json()` for machine consumption
- Stop reasons: `end_turn` | `max_turns` | `error` | `None` (for intermediate turns)
- Duration tracking per turn and total
- Error handling preserves partial results
- Both `run()` and `run_streaming()` APIs

**Test Coverage:** 474 tests pass, 57 agent-specific tests covering:
- Turn limiting, graceful completion, tool call handling
- JSON output format validation
- Error paths and edge cases
- CLI integration

**Verdict:** APPROVED. Clean architecture, minimal surface, well-tested. Fits the existing module structure perfectly. No changes required before shipping.
