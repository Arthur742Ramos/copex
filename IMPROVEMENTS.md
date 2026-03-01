# Recent Improvements

## v2.7.0 — Agent & Squad Features (2026-03-01)

### Documentation Updates
- **README.md** — Added Agent and Squad sections with examples and CLI usage
- **CHANGELOG.md** — Documented Agent Support and Squad Orchestration features
- **CLI Commands Table** — Updated to include `copex agent` and `copex squad` with descriptions
- **Features List** — Added Agent loop, Squad orchestration, Squad default mode, and Repo-awareness
- **examples/** — Added `agent_loop.py` and `squad_orchestration.py` with practical examples

### Agent Loop Features
- Turn-based agent with tool calls and iterative reasoning
- JSON Lines output for machine consumption (`--json` flag)
- Configurable max turns and model selection
- Support for both SDK (Copex) and CLI (subprocess) clients
- AgentSession, AgentTurn, AgentResult classes in public API

### Squad Orchestration Features
- Multi-agent team: Lead Architect, Developer, Tester, Docs Expert
- Default mode for `copex chat` (use `--no-squad` to disable)
- Auto-discovery of project context (README, pyproject.toml, conventions)
- Parallel execution via Fleet with adaptive concurrency
- SquadCoordinator, SquadTeam, SquadAgent, SquadRole in public API

### Documentation Quality
- All new CLI commands documented with examples
- Usage patterns for both CLI and Python API
- Code examples are copy-pasteable and tested
- Clear distinction between single-agent (Agent) and multi-agent (Squad) modes
