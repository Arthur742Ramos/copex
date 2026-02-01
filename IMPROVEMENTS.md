# Copex Improvements (Prioritized)

Scoring: Impact/Effort on a 1–5 scale (5 = highest). Ordered within each section by impact ÷ effort.

## Quick Wins (low effort, high impact)

1) **Fix ProgressReporter running count decrement** (Impact 4, Effort 1)
   - **Rationale:** `complete_item`/`fail_item` currently check status after mutation, so running counts never decrement.
   - **Change:** Decrement running counters based on prior status or track running IDs.
   - **Verify:** Start + complete an item and confirm running=0, completed=1.

2) **Pass config.skills into session create payload** (Impact 4, Effort 1)
   - **Rationale:** Skills defined in config are not forwarded in `_create_session_with_reasoning` payload.
   - **Change:** Map `skills` into JSON-RPC payload when present.
   - **Verify:** Enable a skill and confirm it appears in session capabilities.

3) **Propagate tool_call_id through StreamChunk/UI** (Impact 4, Effort 2)
   - **Rationale:** Tool updates are keyed only by name; concurrent same-name tools overwrite each other.
   - **Change:** Include tool_call_id in chunks and UI state; update by id first, name as fallback.
   - **Verify:** Run two same-name tool calls; both update correctly.

4) **Log exceptions on stop/recovery paths** (Impact 3, Effort 1)
   - **Rationale:** Silent failures in `stop()`/`_recover_session()` hide root causes.
   - **Change:** Add logger warnings with context.
   - **Verify:** Inject a failure and confirm logs surface it.

5) **Persist last reasoning effort** (Impact 3, Effort 1)
   - **Rationale:** Model is persisted but reasoning effort resets each run.
   - **Change:** Store last reasoning in state.json, load on startup.
   - **Verify:** Restart interactive; last reasoning is default.

6) **Add /reasoning toggle in interactive UI** (Impact 3, Effort 2)
   - **Rationale:** Users may want to hide reasoning for privacy or screen space.
   - **Change:** Toggle to suppress reasoning panels and streaming.
   - **Verify:** Toggle hides reasoning in live and final views.

7) **Plan step parser: accept “STEP 1 - …” and “1)”** (Impact 3, Effort 1)
   - **Rationale:** Many LLMs emit alternate numbering formats.
   - **Change:** Extend regex and add parse warnings when nothing matches.
   - **Verify:** Parse sample outputs with hyphens/parentheses.

8) **Surface MCP stderr on failures** (Impact 3, Effort 2)
   - **Rationale:** Stdio MCP failures are opaque without stderr.
   - **Change:** Capture stderr lines and include in exceptions.
   - **Verify:** Run a bad server and see stderr in error output.

9) **Cache find_copilot_cli + env override** (Impact 3, Effort 2)
   - **Rationale:** Repeated filesystem scans slow startup.
   - **Change:** Memoize and respect COPEX_COPILOT_CLI.
   - **Verify:** Repeated calls do not rescan; env override works.

10) **UI ASCII icon mode** (Impact 2, Effort 1)
   - **Rationale:** Some terminals render Unicode poorly.
   - **Change:** Add ASCII fallback icon set.
   - **Verify:** ASCII mode renders cleanly in non-Unicode terminals.

## Medium-Term Improvements (moderate effort)

1) **Structured logging + configurable sink** (Impact 5, Effort 3)
   - **Files:** client.py, cli.py, config.py
   - **Rationale:** Debugging retries and tool calls is currently hard.
   - **Change:** Add structured logger, file output option, and per-component levels.
   - **Verify:** `--log-level debug` emits event logs with request IDs.

2) **Token usage from SDK events** (Impact 5, Effort 3)
   - **Files:** client.py, metrics.py, models.py
   - **Rationale:** Current token counts are estimates only.
   - **Change:** Extract token usage from session events; fallback to estimates.
   - **Verify:** Metrics show exact token counts when SDK provides them.

3) **MCP HTTP transport + retries/backoff** (Impact 5, Effort 3)
   - **Files:** mcp.py, config.py
   - **Rationale:** Stdio-only limits MCP adoption.
   - **Change:** Add HTTP transport with retry policy and health checks.
   - **Verify:** Connect to HTTP MCP server and execute tool calls successfully.

4) **Streaming backpressure + bounded queues** (Impact 4, Effort 3)
   - **Files:** client.py, ui.py
   - **Rationale:** Unbounded queues can grow on large outputs.
   - **Change:** Add max queue size and drop/flush strategy.
   - **Verify:** Stress test large output; memory remains stable.

5) **Plan execution progress outputs (json/rich/quiet)** (Impact 4, Effort 3)
   - **Files:** cli.py, plan.py, progress.py
   - **Rationale:** Plan runs are hard to automate.
   - **Change:** Add `--progress` format flag and structured updates.
   - **Verify:** `--progress=json` yields JSON lines for CI.

6) **Session replay/export command** (Impact 4, Effort 3)
   - **Files:** cli.py, persistence.py, ui.py
   - **Rationale:** Easier auditing and debugging of runs.
   - **Change:** Add `copex session export --format md/json`.
   - **Verify:** Exported transcripts match persisted data.

7) **Plugin tool registry (entry points)** (Impact 4, Effort 3)
   - **Files:** tools.py, __init__.py, pyproject.toml
   - **Rationale:** Enables ecosystem tools without core changes.
   - **Change:** Load tools from entry points.
   - **Verify:** Install dummy plugin; tools are discoverable.

8) **Auto-continue prompt truncation** (Impact 4, Effort 3)
   - **Files:** client.py, config.py
   - **Rationale:** Recovery prompts can exceed model context.
   - **Change:** Summarize/truncate context to configured limit.
   - **Verify:** Recovery prompt stays under max size.

## Future Vision Features (high effort)

1) **Multi-agent orchestration with shared memory** (Impact 5, Effort 5)
   - **Scope:** Task router, agent roles, shared session memory, orchestration UI.
   - **Value:** Enables large multi-step projects with delegation.

2) **IDE companion daemon + socket API** (Impact 5, Effort 5)
   - **Scope:** Background service for persistent low-latency sessions.
   - **Value:** Tight editor integration and fast responses.

3) **Tool sandboxing + policy engine** (Impact 5, Effort 5)
   - **Scope:** Allow/deny policies, approvals, dry-run, audit logs.
   - **Value:** Safer enterprise usage and governance.

4) **Session timeline + diff viewer UI** (Impact 4, Effort 5)
   - **Scope:** Replay view, diffs between iterations, event timeline.
   - **Value:** Debuggability and auditability.

5) **Offline record/replay of Copilot streams** (Impact 4, Effort 5)
   - **Scope:** Capture/replay SDK streams for deterministic tests.
   - **Value:** Reliable CI and regression testing.

6) **Model auto-selection and fallback** (Impact 4, Effort 5)
   - **Scope:** Policy-driven model choice based on cost/latency/quality.
   - **Value:** Optimize spend and performance automatically.

7) **Project indexer + semantic search tool** (Impact 4, Effort 5)
   - **Scope:** Background indexing, vector search, tool integration.
   - **Value:** Faster codebase understanding.

8) **Collaborative shared sessions** (Impact 4, Effort 5)
   - **Scope:** Multi-user sessions with roles and permissions.
   - **Value:** Team workflows and pair programming.
