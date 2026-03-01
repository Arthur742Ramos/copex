# Project Context

- **Owner:** Arthur
- **Project:** Copex — Resilient Python wrapper for GitHub Copilot SDK with auto-retry, exponential backoff, Ralph Wiggum loops, streaming, session persistence, metrics, parallel tools, and MCP integration.
- **Stack:** Python 3.10+, asyncio, Typer CLI, Rich, Pydantic, pytest, GitHub Copilot SDK
- **Created:** 2026-03-01

## Learnings

<!-- Append new learnings below. Each entry is something lasting about the project. -->

📌 Team update (2026-03-01T01:15:00Z): Comprehensive agent test suite (57 tests, 11 classes) complete and all passing. Tests validate AgentSession, dataclasses, CLI integration, tool calling, error handling, streaming. — Hibbert

### Squad Test Suite (2026-03-01T01:43:30Z)
- **50 new tests** across 12 test classes in `tests/test_squad.py`
- **Coverage:** SquadRole, agent/team/result dataclasses, orchestration, parallel execution, error handling, duration tracking, CLI integration, end-to-end workflows
- **Test classes:** TestSquadRole, TestSquadAgentDataclass, TestSquadTeamDataclass, TestSquadAgentResult, TestSquadResult, TestSquadCoordinator, TestSquadOrchestration, TestParallelExecution, TestErrorHandling, TestDurationTracking, TestCLIIntegration, TestEndToEnd
- **All 524 tests pass** (50 new squad tests + 474 existing)
- **Status:** Comprehensive coverage, production-ready

- **Agent module API (src/copex/agent.py):** `AgentSession(client, *, max_turns, model, continue_prompt)` — prompt is passed to `run(prompt)` / `run_streaming(prompt)`, not the constructor. Tool calls are plain dicts (not a `ToolCall` class). Intermediate turns with tool calls get `stop_reason=None`; final turns get `end_turn`, `max_turns`, or `error`.
- **Test pattern for agent:** FakeClient delivers tool calls via `on_chunk` callbacks using `StreamChunk` objects (type="tool_call" and type="tool_result") — this mirrors how the real SDK flows tool information. The agent's `_execute_turn` captures these via the on_chunk callback.
- **CLI agent command:** Registered as `@app.command("agent")` in cli.py. Accepts `--json` for JSON Lines output, `--max-turns`, `--model`, `--use-cli`, `--stdin`. Without prompt, exits nonzero.
- **Test file:** `tests/test_agent.py` — 57 tests across 11 test classes covering dataclasses, session basics, turn limiting, graceful completion, tool calls, error handling, JSON output, stop reasons, model param, edge cases, CLI integration, and run_streaming.
- **Full test suite runs in ~4 minutes (474 tests)** via `python -m pytest tests/ -v`.

### Coverage Audit (2026-03-01T02:30:00Z)

**Results:** 547 tests, all passing. ~4 min runtime.

**Coverage improvements:**
- `agent.py`: 93% → **100%** — Added 6 tests for `_extract_tool_calls` raw_events fallback (non-dict events, missing data, no arguments, data-not-dict)
- `squad.py`: 84% → **95%** — Added 10 tests: pyproject.toml reading (3), `run()` integration with mocked Fleet (2), `_build_result` edge case (1), CLI integration (3), combined pyproject+README context (1)

**Bug found and fixed:** `squad.py` imported `tomli` directly instead of using `try: tomllib / except: tomli` pattern from `config.py`. The pyproject.toml context discovery was silently failing on Python 3.11+ without `tomli` installed.

**Remaining uncovered in squad.py (8 lines):** All are `except Exception: pass` defensive branches in `_discover_project_context` for file I/O errors — acceptable coverage gaps.

**New test counts:** agent=63 tests (was 57), squad=68 tests (was 58)
**Verdict: Excellent** — both modules at 95%+ coverage, all critical paths tested, edge cases covered.

📌 Team update (2026-03-01T02:20:23Z): Quality audit complete. Test coverage excellent (95%+). Documentation updated. All modules ship-ready. Bug fixed in squad.py (tomli import). Code review: APPROVED WITH NOTES (3 non-blocking items). — Burns, Hibbert, Brockman
