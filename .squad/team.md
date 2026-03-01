# Team Roster

> Copex — Resilient Python wrapper for GitHub Copilot SDK with auto-retry, streaming, and Ralph Wiggum loops.

## Coordinator

| Name | Role | Notes |
|------|------|-------|
| Squad | Coordinator | Routes work, enforces handoffs and reviewer gates. Does not generate domain artifacts. |

## Members

| Name | Role | Charter | Status |
|------|------|---------|--------|
| Burns | Lead | `.squad/agents/burns/charter.md` | ✅ Active |
| Frink | Core Dev | `.squad/agents/frink/charter.md` | ✅ Active |
| Hibbert | Tester | `.squad/agents/hibbert/charter.md` | ✅ Active |
| Scribe | Session Logger | `.squad/agents/scribe/charter.md` | 📋 Silent |
| Ralph | Work Monitor | — | 🔄 Monitor |

## Coding Agent

<!-- copilot-auto-assign: false -->

| Name | Role | Charter | Status |
|------|------|---------|--------|
| @copilot | Coding Agent | — | 🤖 Coding Agent |

### Capabilities

**🟢 Good fit — auto-route when enabled:**
- Bug fixes with clear reproduction steps
- Test coverage (adding missing tests, fixing flaky tests)
- Lint/format fixes and code style cleanup
- Dependency updates and version bumps
- Small isolated features with clear specs
- Boilerplate/scaffolding generation
- Documentation fixes and README updates

**🟡 Needs review — route to @copilot but flag for squad member PR review:**
- Medium features with clear specs and acceptance criteria
- Refactoring with existing test coverage
- API endpoint additions following established patterns
- Migration scripts with well-defined schemas

**🔴 Not suitable — route to squad member instead:**
- Architecture decisions and system design
- Multi-system integration requiring coordination
- Ambiguous requirements needing clarification
- Security-critical changes (auth, encryption, access control)
- Performance-critical paths requiring benchmarking
- Changes requiring cross-team discussion

## Project Context

- **Owner:** Arthur
- **Stack:** Python 3.10+, asyncio, Typer CLI, Rich, Pydantic, pytest, GitHub Copilot SDK
- **Description:** Resilient wrapper for GitHub Copilot SDK with auto-retry, exponential backoff, Ralph Wiggum loops, streaming, session persistence, metrics, parallel tools, and MCP integration.
- **Created:** 2026-03-01
