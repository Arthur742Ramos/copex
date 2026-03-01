# Decisions

> Shared decision log. All agents read this before starting work.
> Scribe merges new decisions from `.squad/decisions/inbox/`.

---

### Agent Module Design (2026-03-01)
- **Architecture:** `src/copex/agent.py` with `AgentSession`, `AgentTurn`, `AgentResult`
- **API:** `AgentSession(client, max_turns=N)` then `session.run(prompt)`. Prompt goes to `run()`, not constructor.
- **Tool Detection:** On-chunk callbacks for SDK/CLI, raw_events fallback for mocks
- **JSON Lines:** Each turn emits one line via `turn.to_json()` with fields: turn, content, tool_calls, stop_reason, error, duration_ms
- **Stop Reasons:** `end_turn` (no tools), `max_turns` (limit hit), `error` (exception), `None` (intermediate)
- **CLI Command:** `copex agent "prompt" --json --max-turns 10 --model X --use-cli`
- **Test Suite:** 57 comprehensive tests, all passing (11 test classes)
- **Status:** Fully implemented and production-ready
