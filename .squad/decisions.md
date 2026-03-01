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

---

### Agent Module Design Approved (2026-03-01)
**Reviewer:** Burns  
**Author:** Frink  

**Public API:**
- `AgentSession(client, max_turns=10, model=None, continue_prompt="Continue.")`
- `AgentTurn` dataclass (turn, content, tool_calls, stop_reason, error, duration_ms)
- `AgentResult` dataclass (turns, final_content, total_turns, total_duration_ms, stop_reason, error)

**Design Rationale:**
- Protocol allows `AgentSession` to work with both `Copex` (SDK) and `CopilotCLI` (subprocess)
- JSON output for machine consumption by orchestrators
- Separation keeps agent loop logic (agent.py) from CLI presentation (cli.py)

**Decision:** APPROVED — No changes required. Ship as-is.

---

### Agent Support Architecture (2026-03-01)
**Author:** Frink

- `AgentSession` wraps any client (Copex SDK or CopilotCLI) in turn-based loop
- Tool detection uses dual path: `on_chunk` callbacks (real clients) + `raw_events` fallback (test mocks)
- Intermediate turns get `stop_reason=None`; final turns get `end_turn`/`max_turns`/`error`
- JSON Lines protocol: one JSON object per line per turn
- `copex agent` CLI command with `--json` flag for subprocess integration
- **Impact:** New `agent` command in cli.py, exports in __init__.py, new src/copex/agent.py

---

### Agent Test Patterns and FakeClient Design (2025-07-24)
**By:** Hibbert (Tester)

- `FakeClient` delivers tool calls via `on_chunk` callbacks with `StreamChunk` objects, matching real SDK flow
- Tool calls in agent are plain dicts (not separate dataclass)
- Constructor takes `client` positionally; `prompt` goes to `run(prompt)` / `run_streaming(prompt)`
- Intermediate turns with tool calls have `stop_reason=None`; final turns use `end_turn`, `max_turns`, or `error`
- 57 tests total across 11 test classes, all passing
- FakeClient pattern is canonical for testing agent loop
