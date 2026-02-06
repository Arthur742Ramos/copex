# Changelog

All notable changes to Copex will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.9.0] - 2026-02-06

### Added

- **SessionPool integration in Fleet**: Fleet now reuses sessions from the SessionPool instead of creating new Copex clients per task, improving performance for parallel task execution
- **AdaptiveRetry integration in client**: Core client now uses AdaptiveRetry's error categorization and backoff strategies for smarter retry behavior
- **Model fallback on circuit break**: When a model's circuit breaker opens, automatically falls back to the next model in the fallback chain (e.g., opus → sonnet → haiku)
- `DEFAULT_FALLBACK_CHAINS`: Pre-defined fallback chains for Claude and GPT model families
- `ModelAwareBreaker.get_available_model()`: Get first available model from fallback chain
- `ModelAwareBreaker.is_open()`: Check if circuit is open without raising
- `Copex(fallback_chain=[...])`: Custom fallback chain parameter

### Changed

- `categorize_error()` now uses compiled regex patterns with context-aware matching (O(1) instead of O(100) for status code checks)
- `_calculate_delay()` now accepts optional `error` parameter for error-aware backoff delays
- `_should_retry()` now uses AdaptiveRetry's error categorization for consistent behavior

### Fixed

- `categorize_error()` no longer has false positives for messages containing "500" (e.g., "Error at line 500" is now correctly categorized as UNKNOWN, not SERVER)

## [1.6.0] - 2026-02-06

### Added

- **FleetStore**: SQLite-backed persistence for fleet task orchestration
- **Fleet.resume()**: Crash recovery by reloading incomplete runs from database
- `Fleet(db_path=...)` optional parameter for persistent fleet runs
- 50 new tests (267 total): `test_mcp.py`, `test_config.py`, `test_fleet_store.py`

### Fixed

- MCP: fail pending futures immediately on EOF instead of 30-second stall
- MCP: relax argument validation to allow Windows paths and spaces
- Config: fix NVM version detection to use semantic sort (`v10 > v9`)
- Client: destroy non-pooled sessions to prevent resource leaks
- Fleet: fix prompt mutation (`shared_context` no longer mutates tasks)
- Fleet: fix streaming backpressure race condition (drop deltas on full queue)
- Fleet: standardize dependency failure error messages

## [1.5.0] - 2026-02-06

### Added

- `--verbose` and `--output-dir` flags on `copex fleet` command
- Tests for FleetMailbox, FleetContext, and `run_streaming()`

### Changed

- Default model changed to `claude-opus-4.6`

### Fixed

- CLI: client history fallback when `on_chunk` callback is provided
- Fleet: handle `QueueFull` in event streaming
- Plan: document `skip_context` mutation behavior

## [1.4.0] - 2026-02-06

### Added

- **Fleet Mailbox**: `FleetMailbox` for inter-task messaging (send/receive/broadcast)
- **Fleet Context**: `FleetContext` with shared state and result aggregation
- **Fleet Streaming**: `run_streaming()` yields `FleetEvent` objects
- **Circuit Breaker**: `SlidingWindowBreaker` (window=10, threshold=50%) and `ModelAwareBreaker` for per-model circuit breakers
- **Session Pool**: `SessionPool` with `acquire()` context manager
- **CLI Input**: `--stdin`/`-i` to read prompt from pipe, `--context`/`-C` to include files as context
- **CLI Output**: `--template`/`-T` for Jinja2 output templates, `--output`/`-o` to write response to file
- **Shell Completions**: `copex completions bash|zsh|fish`
- **Plan Editor**: `PlanEditor` for edit/remove/insert/reorder steps
- **Plan Checkpoint**: `PlanCheckpoint` for step-level resume with context
- `skip_condition` for conditional step execution
- **Visualization**: `render_dag_with_status()` with Mermaid/ASCII/DOT output, `find_parallel_groups()` to detect parallelizable steps, status indicators with colors/symbols
- Warning when `raw_events` hits 10,000 event limit

## [1.3.0] - 2026-02-06

### Added

- `copex config` command: validate and display current configuration with env var overrides, skill directory status, and warnings
- `--json` flag on `copex chat` for machine-readable JSON output
- `--quiet`/`-q` flag on `copex chat` for minimal content-only output
- `--dry-run` flag on `copex plan` to generate and display plan without executing
- Client-level circuit breaker with configurable failure threshold and cooldown
- MCP command validation with allowlist pattern for security
- MCP config file size limit (1 MB) to prevent memory exhaustion
- Session file size limit (10 MB) with validation on load
- Windows reserved device name sanitization for session file paths
- Improved fleet DAG error messages: missing dependencies now list available task IDs, cycle detection identifies specific tasks involved
- Fleet tasks now use event-based dependency waiting instead of wave-based dispatch
- Structured logging via Python `logging` module in client

### Changed

- Fleet coordinator rewritten from wave-based to event-based concurrent execution model

## [1.2.0] - 2026-02-05

### Added

- **Fleet Mode**: parallel AI sub-agent orchestration with `FleetTask`, `FleetResult`, `FleetConfig`, `FleetCoordinator`
- DAG-based dependency resolution with cycle detection
- Semaphore-limited concurrency and fail-fast support
- `copex fleet` CLI command with live Rich progress table, TOML task file support, and summary panel
- Fleet classes exported from `copex` package
- `claude-opus-4.6` model support
- Top-level `-p` (prompt) and `-s` (system) flags on CLI
- 18 fleet tests covering DAG validation, parallel execution, dependency ordering, fail-fast, and the Fleet API

## [1.1.1] - 2026-02-03

### Changed

- Upgraded to SDK v0.1.21 with native `reasoning_effort` support
- Config key changed from `model_reasoning_effort` to `reasoning_effort`
- SDK now handles `reasoningEffort` wire format directly

### Fixed

- Ruff unused import lint error after SDK update
- Visualization bugs: `plan.goal` → `plan.task` (3 locations)
- `--visualize` flag added to `copex plan` command (ascii, mermaid, tree)

### Added

- Typed exceptions and structured logging
- Security: input sanitization, environment variable filtering
- Plan visualization (ASCII/Mermaid/tree)
- Step caching with TTL
- Adaptive retry backoff
- Conditional plan steps with `skip_condition`
- Step templates
- Full API documentation in README

## [1.0.3] - 2026-02-03

### Added

- **Skills System**: full auto-discovery from `.github/skills/`, `.claude/skills/`, `.copex/skills/`, `~/.config/copex/skills/`
- `copex skills list` and `copex skills show` commands
- `--skill-dir`, `--disable-skill`, `--no-auto-skills` flags on chat, interactive, ralph, and plan commands
- `skill_directories`, `disabled_skills`, `auto_discover_skills` config options
- `list_skills`, `get_skill_content`, `SkillInfo`, `SkillDiscovery` exported from package
- Comprehensive skill test suite (11 tests)
- Skills documentation in README

### Fixed

- Resolve skill directories to absolute paths before passing to SDK

## [1.0.2] - 2026-02-01

### Added

- **Interactive UI Overhaul**: modern Tokyo Night color palette inspired by Claude Code, OpenCode, and Aider
- Clean prompt showing model/reasoning level
- Animated spinner with elapsed time during thinking
- Compact tool call display (1 line per tool with icon, args, and duration)
- Smooth streaming with blinking cursor
- Final markdown rendering with clean stats line
- New slash commands: `/model`, `/reasoning`, `/models`, `/new`, `/status`, `/help`, `/clear`
- `tokyo` theme preset
- `--classic` flag for legacy interactive mode
- 34 new tests for interactive module
- GitHub Actions CI/CD for testing and releases

### Changed

- Minimal design inspired by `gh copilot` CLI — removed box/panel clutter for clean flowing text

## [1.0.1] - 2026-02-01

### Fixed

- Prevent infinite auth refresh loop with max attempts counter and lock

## [1.0.0] - 2026-02-01

### Added

- **Core Client**: `Copex` client with automatic retry and exponential backoff with jitter
- **Streaming**: real-time streaming with `on_chunk` callbacks for message, reasoning, and tool events
- **Ralph Wiggum Loops**: iterative AI development loops with completion promise detection
- **Plan Mode**: `copex plan` command with generate, execute, review, and resume modes
- **Parallel Execution**: `--parallel`/`-P` flag for concurrent independent plan steps
- **Smart Planning**: `--smart`/`-S` for v2 planning with dependency detection
- **Progress Reporting**: `ProgressReporter` and `PlanProgressReporter` for real-time progress bars
- **Session Persistence**: `SessionStore` for saving/loading sessions to disk
- **Checkpointing**: `CheckpointStore` and `CheckpointedRalph` for resumable operations
- **Metrics**: `MetricsCollector` for usage and cost tracking
- **Parallel Tools**: `ParallelToolExecutor` for concurrent tool execution
- **MCP Integration**: `MCPManager` with stdio transport for MCP server connections
- **Interactive Mode**: default mode with conversation history, model picker, and slash commands
- **Rich UI**: beautiful terminal output with themes, spinners, panels, and icons
- **Reasoning Effort**: support for reasoning levels (low/medium/high/xhigh) via direct JSON-RPC
- **Session Recovery**: automatic context preservation and recovery after retry exhaustion
- **Activity Timeout**: timeout based on inactivity, not total elapsed time
- **Configuration**: TOML config from `~/.config/copex/config.toml` with env var overrides
- **Models**: support for Claude Opus 4.5, GPT-5, GPT-5.1, GPT-5.2, Gemini, and more
- **CLI Commands**: `chat`, `ralph`, `plan`, `interactive`, `status`, `skills`
- GitHub Actions CI/CD pipeline with PyPI publishing
- Comprehensive test suite (100+ tests) with `FakeSession` mock

[1.6.0]: https://github.com/arthurbrenno/copex/compare/v1.5.0...v1.6.0
[1.5.0]: https://github.com/arthurbrenno/copex/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/arthurbrenno/copex/compare/v1.2.0...v1.4.0
[1.3.0]: https://github.com/arthurbrenno/copex/compare/v1.2.0...v1.4.0
[1.2.0]: https://github.com/arthurbrenno/copex/compare/v1.1.1...v1.2.0
[1.1.1]: https://github.com/arthurbrenno/copex/compare/v1.0.3...v1.1.1
[1.0.3]: https://github.com/arthurbrenno/copex/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/arthurbrenno/copex/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/arthurbrenno/copex/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/arthurbrenno/copex/releases/tag/v1.0.0
