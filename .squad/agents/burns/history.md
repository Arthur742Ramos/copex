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

### 2026-03-01: Squad Module + CLI Commands Full Review

**Context:** Reviewed squad.py, cli.py squad/agent commands, and __init__.py exports as a full code audit of all new session work.

**Files Reviewed:**
- `src/copex/agent.py` — Previously approved. No changes since. Still clean.
- `src/copex/squad.py` — New multi-agent orchestration layer on top of Fleet.
- `src/copex/cli.py` — New `copex agent` and `copex squad` commands, squad-as-default in `main()`.
- `src/copex/__init__.py` — New exports for agent + squad types.

**squad.py Assessment:**
- Clean data model: SquadRole enum, SquadAgent/SquadTeam/SquadAgentResult/SquadResult dataclasses
- Good architecture: Delegates to Fleet for execution. No parallel/async reinvention.
- Correct dependency DAG: Lead→(Developer∥Tester)→Docs
- Uses Fleet's `{task:ID.content}` template interpolation correctly for cross-agent context
- `_discover_project_context()` adds repo-awareness (pyproject.toml, README, dir listing)
- `to_dict()`/`to_json()` follows established patterns from agent.py
- Note: `__aenter__`/`__aexit__` are no-ops (future-proofing, acceptable)
- Note: `_ROLE_EMOJIS` is private but imported by cli.py — minor encapsulation leak

**cli.py Assessment:**
- `copex agent`: Clean, mirrors existing command patterns (config loading, model/reasoning parsing, stdin). Both JSON Lines and Rich panel output.
- `copex squad`: Follows same pattern. Good status callback with role emojis.
- Default behavior in `main()`: Squad mode by default, `--no-squad` for single-agent. Clean opt-out design.
- Consistent error handling: JSON error objects in `--json` mode, Rich errors otherwise.
- Config loading pattern duplicated across agent/squad/main — acceptable given CLI framework constraints.
- `_print_agent_turn()` handles both dict and object tool calls (defensive, good).
- One import of private `_ROLE_EMOJIS` from squad — should be public or accessed via method.

**__init__.py Assessment:**
- All squad types exported: SquadAgent, SquadAgentResult, SquadCoordinator, SquadResult, SquadRole, SquadTeam
- Agent types already present from prior approval
- `__all__` properly updated with both Agent and Squad sections
- Clean alphabetical grouping

**Test Status:** 532 tests pass, 0 failures. No squad-specific tests yet (noted — Hibbert should write these).

**Verdict:** APPROVED WITH NOTES.
- Ship-ready. Architecture is sound, integrates cleanly with Fleet, follows existing patterns.
- Note 1: `_ROLE_EMOJIS` imported across modules as private — minor, fix when convenient.
- Note 2: `on_status` callback typed as `Any` in squad.py — should be `Callable[[str, str], None] | None`.
- Note 3: No squad-specific tests. Hibbert should add SquadCoordinator unit tests.

📌 Team update (2026-03-01T02:20:23Z): Quality audit complete. Test coverage excellent (95%+). Documentation updated. All modules ship-ready. Bug fixed in squad.py (tomli import). Code review: APPROVED WITH NOTES (3 non-blocking items). — Burns, Hibbert, Brockman
