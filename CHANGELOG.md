# Changelog

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
