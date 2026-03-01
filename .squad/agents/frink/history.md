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

### Squad Module (2026-03-01T01:43:30Z)
- **New module:** `src/copex/squad.py` — `SquadCoordinator`, `SquadTeam`, `SquadAgent`, `SquadResult`, `SquadAgentResult`, `SquadRole`
- **Orchestration:** Uses Fleet executor for parallel Dev + Tester execution after Lead completes
- **CLI command:** `copex squad` accepts squad config JSON, outputs SquadResult JSON with outcomes and durations
- **Public API exports:** Added to __init__.py (SquadCoordinator, SquadTeam, SquadAgent, SquadResult, SquadAgentResult, SquadRole)
- **Status:** Fully implemented, 50 tests passing, production-ready

### Repo-Aware Squad (2026-03-01)
- **Feature:** `_discover_project_context()` on `SquadCoordinator` auto-reads README.md (first 2000 chars), pyproject.toml name/description, and top-level directory structure
- **Injection:** Project context injected into all agent prompts via `_build_agent_prompt()`, between role prompt and task
- **Caching:** Lazy discovery — only runs once per `run()` call, cached in `_project_context`
- **New role:** `SquadRole.DOCS` = "docs" with 📝 emoji — Documentation Expert for README, docstrings, examples
- **Default team:** Now 4 agents: Lead → Developer + Tester (parallel) → Docs (runs last, depends on both dev + tester)
- **Graceful fallback:** Empty string returned if no README/pyproject found
- **Tests:** 8 new tests (58 total), all passing

### Dynamic Squad Teams (2026-03-01)
- **Feature:** `SquadTeam.from_repo(path)` scans repo structure to build role-appropriate teams
- **New roles:** DEVOPS (⚙️), FRONTEND (⚛️), BACKEND (🔧) with prompts and emojis
- **Detection:** Source files → Developer, tests/ or test patterns → Tester, docs/ or multiple .md → Docs, Dockerfile/Makefile/CI → DevOps, src/ + frontend dirs → Frontend, src/ + backend dirs → Backend
- **Minimum team:** Always Lead + Developer (fallback if no signals found)
- **Default change:** `SquadCoordinator.__init__` now uses `from_repo()` instead of `default()` when no team specified
- **Backward compat:** `SquadTeam.default()` unchanged — still returns static 4-agent team
- **Dependency updates:** Tester depends on implementation agents (Developer/Frontend/Backend), falls back to Lead. Docs depends on all non-lead roles. New roles depend on Lead.
- **Tests:** 23 new tests (90 total), all passing
