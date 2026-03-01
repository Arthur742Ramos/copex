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

---

### Code Review: Agent + Squad + CLI Modules (2026-03-01)
**Reviewer:** Burns (Lead/Architect)

**Scope:** agent.py, squad.py, cli.py commands, __init__.py exports

**Verdicts:**
- `agent.py` — APPROVED (unchanged since prior review)
- `squad.py` — APPROVED WITH NOTES
- `cli.py` (agent/squad commands) — APPROVED WITH NOTES
- `__init__.py` — APPROVED

**Architecture Notes:**
- Squad correctly delegates to Fleet for parallel execution (no reimplementation)
- Dependency DAG (Lead → Developer∥Tester → Docs) uses Fleet template interpolation correctly
- Default squad mode in CLI is good UX; `--no-squad` opt-out is clean

**Items for Follow-up (non-blocking):**
1. Tests needed for SquadCoordinator: default team creation, dependency ordering, prompt building with templates, result aggregation, error propagation
2. Minor type issue: `on_status` callback typed as `Any` → should be `Callable[[str, str], None] | None`
3. Encapsulation: `_ROLE_EMOJIS` is private but imported by cli.py → either make public (`ROLE_EMOJIS`) or use `SquadAgent.emoji`

**Decision:** APPROVED WITH NOTES — code is clean, well-integrated, follows established Copex patterns. Ship-ready.

---

### Test Coverage Audit: Agent + Squad (2026-03-01)
**By:** Hibbert (Tester)

**Results:**
- Full test suite: 547 tests, all passing (~4 min runtime)
- `agent.py`: 93% → **100%** coverage (6 new tests for `_extract_tool_calls` raw_events fallback)
- `squad.py`: 84% → **95%** coverage (10 new tests: pyproject.toml, run() integration, CLI)

**Bug Fixed:**
- `squad.py` line 332: imported `tomli` directly instead of `try: tomllib / except: tomli` pattern
- This caused pyproject.toml context discovery to fail silently on Python 3.11+ without tomli installed
- Fixed to match pattern in config.py

**Remaining Gaps (Acceptable):**
- 8 uncovered lines in squad.py (334-335, 346-347, 355-356, 367-368) — all `except Exception: pass` in `_discover_project_context`
- These are defensive I/O error handlers; testing would require filesystem monkeypatching (low value)

**Verdict:** Excellent — both modules at 95%+ coverage. All critical paths tested.

---

### Documentation Audit: Agent + Squad Features (2026-03-01)
**By:** Brockman (Docs/DevRel)

**Work Completed:**
- README.md: added Agent/Squad sections, updated Features list & CLI Commands table (+150 lines)
- CHANGELOG.md: added v2.7.0 release entry covering AgentSession, SquadCoordinator, default mode (+30 lines)
- IMPROVEMENTS.md: added feature summary and metrics (was empty, +25 lines)
- examples/agent_loop.py: basic agent loop, JSON Lines output, streaming (+60 lines)
- examples/squad_orchestration.py: squad task, context auto-discovery, JSON output, custom team (+70 lines)

**Quality Metrics:**
- ✅ All examples copy-pasteable and follow project style
- ✅ Code examples verified against agent.py and squad.py
- ✅ CLI commands documented with flags
- ✅ Python API documented with usage patterns
- ✅ Clear distinction between single-agent and multi-agent modes
- ✅ Default behavior explained (`copex chat` runs squad by default)

**No Changes Required:**
- agent.py, squad.py — comprehensive docstrings already present
- docs/ — UI design docs in place; architecture docs not needed yet
- Tests — documentation-only update

**Decision:** Documentation complete and in sync with v2.7.0 implementation.
