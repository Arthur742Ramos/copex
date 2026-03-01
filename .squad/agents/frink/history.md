# Project Context

- **Owner:** Arthur
- **Project:** Copex — Resilient Python wrapper for GitHub Copilot SDK with auto-retry, exponential backoff, Ralph Wiggum loops, streaming, session persistence, metrics, parallel tools, and MCP integration.
- **Stack:** Python 3.10+, asyncio, Typer CLI, Rich, Pydantic, pytest, GitHub Copilot SDK
- **Created:** 2026-03-01

## Learnings

<!-- Append new learnings below. Each entry is something lasting about the project. -->

### Agent Support (2026-03-01)
- **New module:** `src/copex/agent.py` — `AgentSession`, `AgentTurn`, `AgentResult`
- **API pattern:** `AgentSession(client, max_turns=N)` then `session.run(prompt)`. Prompt goes to `run()`, not constructor.
- **Tool call detection:** Dual path — on_chunk callbacks for real SDK/CLI clients, raw_events fallback for test mocks (FakeClient doesn't fire on_chunk).
- **JSON Lines protocol:** Each turn emits one JSON line via `turn.to_json()`. Fields: `turn`, `content`, `tool_calls`, `stop_reason`, `error`, `duration_ms`.
- **Stop reasons:** `end_turn` (no tool calls), `max_turns` (exhausted), `error` (exception). Intermediate turns use `None`.
- **CLI command:** `copex agent "prompt" --json --max-turns 10 --model X --use-cli`
- **Pre-existing tests:** Hibbert wrote 39 tests before implementation (test-first). All pass.
- **Key files:** `src/copex/agent.py`, CLI command in `cli.py` (search for `agent_command`), exports in `__init__.py`.

📌 Team update (2026-03-01T01:15:00Z): Agent module fully implemented with AgentSession, AgentTurn, AgentResult, and CLI integration. All 57 tests passing. Production-ready. — Frink
