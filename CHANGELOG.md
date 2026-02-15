# Changelog

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
