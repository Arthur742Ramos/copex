# Changelog

## v2.12.0 (2026-03-01)

### Features
- **Subtask parallelism**: Squad agents can now have `subtasks` — a list of parallel work items that fan out as separate Fleet tasks within the same phase
- When an agent has subtasks, each becomes an independent Fleet task sharing the agent's system prompt but with a focused instruction
- All subtasks within an agent run concurrently (up to `max_concurrent` limit) and results are merged back into a single `SquadAgentResult`
- AI repo analysis (`from_repo_ai()`) can now optionally return subtasks per agent for naturally parallelizable work
- Subtasks persist through `.copex/squad.json` save/load

### Tests
- 9 new tests covering subtask fan-out, dependency expansion, result merging, partial failures, and persistence

## v2.11.0 (2026-03-01)

### Features
- **Freeform AI roles**: Squad AI analysis now freely determines custom roles (e.g., `security_engineer`, `data_scientist`, `api_designer`) instead of being limited to 7 hardcoded roles
- **Phase-based dependencies**: Agents use execution phases (1=analyze, 2=build, 3=verify, 4=document) for dependency ordering, supporting arbitrary role combinations
- **Team persistence**: Squad teams are saved to `.copex/squad.json` and loaded on subsequent runs — AI sees existing teams and decides whether to keep, modify, or expand
- **Existing team awareness**: `from_repo_ai()` shows the current team to the AI, which can add/remove roles based on the repo's needs
- **Custom role defaults**: Unknown roles get sensible default names, emojis (🔹), and system prompts

### Tests
- 9 new tests for freeform roles, phase-based dependencies, save/load persistence
- Updated all dependency tests to use phase-based model

## v2.10.1 (2026-03-01)

### Bug Fixes
- Fixed Fleet/Squad not respecting `--use-cli` flag — agents were always using the SDK client instead of CopilotCLI subprocess, causing permission handler errors
- `FleetCoordinator._task_config()` now propagates `use_cli` to per-task configs
- Skip SDK session pool creation when `use_cli` is enabled

## v2.10.0 (2026-03-01)

### Features
- **AI-powered repo analysis**: `SquadTeam.from_repo_ai()` uses CopilotCLI with `claude-opus-4.6-fast` to intelligently analyze repository structure and determine optimal team composition
- **Model passthrough**: Squad sub-agents now use the same model the user specified (e.g., `copex -m claude-opus-4.6-fast` propagates to all agents)
- **`--no-ai` flag**: Skip AI analysis and use fast pattern-matching fallback for squad team creation

### Bug Fixes
- Fixed ANSI escape code issues in CLI help output tests (Rich/Typer color codes broke `--json`/`--model`/`--reasoning` assertions)
- Strip ANSI codes before asserting in test_agent.py and test_squad.py

## v2.9.0 (2026-02-28)

### Features
- Dynamic squad team creation: `SquadTeam.from_repo()` analyzes repo structure to automatically compose the right team (Developer, Tester, Docs, DevOps, Frontend, Backend roles detected from directory structure)
- 3 new squad roles: DEVOPS (⚙️), FRONTEND (⚛️), BACKEND (🔧)
- Squad coordinator now defaults to dynamic team creation instead of static `default()` team

### Bug Fixes
- Fixed ruff lint errors (unsorted imports, unused imports) that caused CI failures
- Fixed sync-squad-labels workflow to handle HTTP 422 (already_exists) errors gracefully

## v2.7.0 (2026-03-01)

### New Features

#### Agent Support
- **Agent Loop** (`copex agent`) — Turn-based agent with tool calls, iterative reasoning, and JSON Lines output
- **AgentSession** — Python API for running agents with configurable max turns
- **AgentTurn & AgentResult** — JSON-serializable turn results for machine consumption by orchestrators
- Design: dual-client support (SDK Copex + CLI subprocess), tool detection via on_chunk callbacks
- Use cases: subprocess integration, fine-grained control, JSON-based tooling

#### Squad Orchestration
- **Squad Mode** (`copex squad`) — Built-in multi-agent team: Lead Architect, Developer, Tester, Docs Expert
- **Default Mode** — `copex chat "prompt"` now runs squad by default; use `--no-squad` for single-agent
- **Auto-discovery** — Squad discovers project context (README, pyproject.toml, directory structure, conventions)
- **Parallel Execution** — Developer and Tester work in parallel via Fleet; Lead analyzes first
- **SquadCoordinator, SquadTeam, SquadAgent, SquadRole** — Configurable multi-agent orchestration
- **Cost Tracking & Retries** — Integrated with Fleet for adaptive concurrency and error handling

### Internal
- New `src/copex/agent.py` — AgentSession loop with tool detection and JSON Lines output
- New `src/copex/squad.py` — SquadCoordinator, team composition, and role-based prompts
- Enhanced `src/copex/cli.py` — `copex agent` and `copex squad` commands with JSON output support
- Exports in `src/copex/__init__.py` — AgentSession, AgentTurn, AgentResult, SquadCoordinator, SquadTeam, SquadAgent, SquadRole, SquadResult

## 2.0.1 (2026-02-07)

### Bug Fixes

- **fix: patch SDK to remove `--no-auto-update` flag that caused silent model fallback**

  The Copilot SDK's `CopilotClient._start_cli_server` passes `--no-auto-update`
  when spawning the CLI server in headless mode. This prevents the CLI binary from
  fetching the up-to-date model catalogue from the Copilot backend, causing newer
  models (`claude-opus-4.6`, `claude-opus-4.6-fast`) to silently fall back to
  `claude-sonnet-4.5`. Copex now monkey-patches the startup method to strip
  `--no-auto-update`, so the CLI always has the current model list.

### New Models

- Added `claude-opus-4.6-fast` to the `Model` enum and fallback chains.

## v2.6.0 (2026-02-14)

### New Commands
- `copex stats` — Show last run metrics, today's totals (tokens, cost, model, reasoning effort)
- `copex diff` — Show git changes since last copex run (with `--full` for complete diff)
- `copex campaign` — High-level orchestration: discover targets, batch, run waves, resume

### New Fleet Flags
- `--dry-run` — Preview fleet configuration without executing
- `--commit-msg` / `-cm` — Set git commit message from CLI (implies --git-finalize)
- `--retry N` / `-R N` — Auto-retry on build failure, feeding errors back to model
- `--progress` / `-P` — Real-time file change streaming during fleet execution
- `--file` / `-f` — Multi-task mode via JSONL file input
- `--parallel N` — Max concurrent tasks (alias for --max-concurrent)
- `--worktree` / `-w` — Git worktree isolation per task (prevents cross-contamination)

### Internal
- New `src/copex/stats.py` — RunStats tracking and JSONL logging
- New `src/copex/campaign.py` — Campaign orchestration engine
- Enhanced `src/copex/worktree.py` — Full WorktreeManager with cherry-pick merge-back
- Enhanced `src/copex/multi_fleet.py` — JSONL task loading
