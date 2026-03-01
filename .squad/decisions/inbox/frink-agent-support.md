# Decision: Agent Support Architecture

**Author:** Frink  
**Date:** 2026-03-01  
**Status:** Implemented

## Context
Squad needs to spawn Copex as a subprocess agent. The existing CLI had no iterative agent loop.

## Decision
- `AgentSession` wraps any client (Copex SDK or CopilotCLI) in a turn-based loop.
- Tool call detection uses **dual path**: `on_chunk` callbacks (real clients) + `raw_events` fallback (test mocks).
- Intermediate turns get `stop_reason=None`; only final turns get `end_turn`/`max_turns`/`error`.
- JSON Lines protocol: one JSON object per line per turn for machine consumption.
- `copex agent` CLI command with `--json` flag for subprocess integration.

## Impact
- **cli.py:** New `agent` command added before `completions`.
- **__init__.py:** Exports `AgentSession`, `AgentTurn`, `AgentResult`.
- **New file:** `src/copex/agent.py`.
- No changes to existing client behavior.
