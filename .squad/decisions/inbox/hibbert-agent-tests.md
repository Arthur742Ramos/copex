# Decision: Agent test patterns and FakeClient design

**By:** Hibbert (Tester)
**Date:** 2025-07-24
**Scope:** tests/test_agent.py

## Decision

Agent tests use a `FakeClient` that delivers tool calls via `on_chunk` callbacks with `StreamChunk` objects, matching the real SDK flow. Tool calls in the agent are plain dicts (not a separate dataclass). Tests cover all 10 requested areas: dataclasses, session basics, turn limiting, graceful completion, tool calls, error handling, JSON output, stop reasons, model param, and edge cases — plus `run_streaming()`.

## Key conventions

- `AgentSession` constructor takes `client` positionally; `prompt` goes to `run(prompt)` / `run_streaming(prompt)`.
- Intermediate turns with tool calls have `stop_reason=None`; final turns use `end_turn`, `max_turns`, or `error`.
- CLI integration tests use `typer.testing.CliRunner` against `--help` output (no network calls).
- 57 tests total, all passing alongside the full 474-test suite.

## Impact

All agents should know: if Frink changes the agent API signature, the test file needs corresponding updates. The FakeClient pattern in test_agent.py is the canonical way to test the agent loop.
