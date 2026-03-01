# Project Context

- **Owner:** Arthur
- **Project:** Copex — Resilient Python wrapper for GitHub Copilot SDK with auto-retry, exponential backoff, Ralph Wiggum loops, streaming, session persistence, metrics, parallel tools, and MCP integration.
- **Stack:** Python 3.10+, asyncio, Typer CLI, Rich, Pydantic, pytest, GitHub Copilot SDK
- **Created:** 2026-03-01

## Learnings

<!-- Append new learnings below. Each entry is something lasting about the project. -->

- **Orchestration logs:** `.squad/orchestration-log/{timestamp}-{agent-name}.md` per agent. Contains spawn assignment, work summary, key decisions, outcomes, status.
- **Session logs:** `.squad/log/{timestamp}-{topic}.md` brief facts-only record of who worked, what was done, decisions, outcomes.
- **Decision merging:** Read inbox files, append to decisions.md, delete inbox files, deduplicate exact headers.
- **Cross-agent propagation:** Update affected agents' history.md with team update notices in format: `📌 Team update ({timestamp}): {summary} — {Name}`
