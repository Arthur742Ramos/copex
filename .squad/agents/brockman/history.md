# Agent History — Brockman

> Persistent memory across sessions.

## Learnings

- Joined the team as Docs/DevRel expert (2026-03-01)
- Copex is a Python 3.10+ wrapper for GitHub Copilot SDK with async/await, Rich UI, Typer CLI
- Key modules: client.py, cli.py, fleet.py, ralph.py, agent.py, squad.py, config.py, models.py
- Docs live in docs/ directory, examples in examples/
- CHANGELOG.md and IMPROVEMENTS.md track changes
- Version managed in src/copex/__init__.py (read by hatchling)
- **NEW (2026-03-01):** Agent support (agent.py with AgentSession, AgentTurn, AgentResult)
- **NEW (2026-03-01):** Squad orchestration (squad.py with SquadCoordinator, multi-agent teams, auto-discovery)
- **NEW (2026-03-01):** Squad is now default mode for `copex chat`; use `--no-squad` to disable
- Documentation audit completed: README, CHANGELOG, IMPROVEMENTS updated; examples added

## Mistakes

_(none yet)_

## Preferences

- Clear, concise prose — avoid jargon unless defining it
- Code examples should be copy-pasteable
- Use ```python fenced blocks for code
- Keep README focused — link to docs/ for deep dives

📌 Team update (2026-03-01T02:20:23Z): Quality audit complete. Test coverage excellent (95%+). Documentation updated. All modules ship-ready. Bug fixed in squad.py (tomli import). Code review: APPROVED WITH NOTES (3 non-blocking items). — Burns, Hibbert, Brockman
