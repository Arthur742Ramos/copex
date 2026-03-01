### Dynamic Squad Team Creation (2026-03-01)
**Author:** Frink (Core Dev)

**Change:** `SquadTeam.from_repo(path)` class method scans repo directory structure to build context-appropriate teams instead of always creating the same 4 agents.

**New Roles:** DEVOPS, FRONTEND, BACKEND added to SquadRole enum with prompts and emojis.

**Detection Rules:**
- Always: LEAD
- Source files (*.py, *.js, *.ts, *.go, *.rs, *.java, *.c, *.cpp): DEVELOPER
- tests/, test/, test patterns: TESTER
- docs/, README.md, multiple .md files: DOCS
- Dockerfile, docker-compose*, .github/workflows/, Makefile, Jenkinsfile: DEVOPS
- src/ + frontend dirs (frontend/, web/, ui/, app/, pages/, components/): FRONTEND
- src/ + backend dirs (api/, server/, backend/, services/): BACKEND
- Empty repo fallback: Lead + Developer minimum

**Default Behavior Change:** `SquadCoordinator` now uses `from_repo()` when no team is provided. `SquadTeam.default()` preserved for backward compatibility.

**Dependency Graph Updated:**
- DEVELOPER, FRONTEND, BACKEND, DEVOPS → depend on LEAD
- TESTER → depends on implementation agents (Developer/Frontend/Backend), falls back to LEAD
- DOCS → depends on all non-LEAD roles (runs last)

**Impact:** squad.py (new roles, detection functions, from_repo method, updated dependencies), test_squad.py (23 new tests, 2 existing tests updated)

**Tests:** 90 total, all passing.

**Decision:** Ship — backward compatible, fast detection (Path.exists/glob only), minimum viable team guarantee.
