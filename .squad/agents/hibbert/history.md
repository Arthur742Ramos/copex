# Project Context

- **Owner:** Arthur
- **Project:** Copex — Resilient Python wrapper for GitHub Copilot SDK with auto-retry, exponential backoff, Ralph Wiggum loops, streaming, session persistence, metrics, parallel tools, and MCP integration.
- **Stack:** Python 3.10+, asyncio, Typer CLI, Rich, Pydantic, pytest, GitHub Copilot SDK
- **Created:** 2026-03-01

## Learnings

<!-- Append new learnings below. Each entry is something lasting about the project. -->

📌 Team update (2026-03-01T01:15:00Z): Comprehensive agent test suite (57 tests, 11 classes) complete and all passing. Tests validate AgentSession, dataclasses, CLI integration, tool calling, error handling, streaming. — Hibbert

- **Agent module API (src/copex/agent.py):** `AgentSession(client, *, max_turns, model, continue_prompt)` — prompt is passed to `run(prompt)` / `run_streaming(prompt)`, not the constructor. Tool calls are plain dicts (not a `ToolCall` class). Intermediate turns with tool calls get `stop_reason=None`; final turns get `end_turn`, `max_turns`, or `error`.
- **Test pattern for agent:** FakeClient delivers tool calls via `on_chunk` callbacks using `StreamChunk` objects (type="tool_call" and type="tool_result") — this mirrors how the real SDK flows tool information. The agent's `_execute_turn` captures these via the on_chunk callback.
- **CLI agent command:** Registered as `@app.command("agent")` in cli.py. Accepts `--json` for JSON Lines output, `--max-turns`, `--model`, `--use-cli`, `--stdin`. Without prompt, exits nonzero.
- **Test file:** `tests/test_agent.py` — 57 tests across 11 test classes covering dataclasses, session basics, turn limiting, graceful completion, tool calls, error handling, JSON output, stop reasons, model param, edge cases, CLI integration, and run_streaming.
- **Full test suite runs in ~4 minutes (474 tests)** via `python -m pytest tests/ -v`.
