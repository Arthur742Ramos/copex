# Copex — Copilot Extended

[![PyPI](https://img.shields.io/pypi/v/copex)](https://pypi.org/project/copex/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A resilient Python wrapper for the GitHub Copilot SDK with automatic retry, Ralph Wiggum loops, fleet orchestration, and a beautiful CLI.

## Features

- **Automatic retry** with adaptive exponential backoff and jitter
- **Circuit breaker** (sliding-window) to avoid hammering a failing backend
- **Model fallback chains** — automatically try the next model when one fails
- **Session pooling** with LRU eviction and pre-warming
- **Agent loop** — turn-based agent with tool calls and iterative reasoning
- **Squad orchestration** — multi-agent team (Lead architect, Developer, Tester, Docs expert) with auto-discovery of project context and parallel execution
- **Squad as default for root prompts** — `copex -p "prompt"` runs squad mode by default; use `--no-squad` to force single-agent mode, or call `copex chat` directly
- **Ralph Wiggum loops** — iterative AI development via repeated prompts
- **Fleet mode** — run multiple tasks in parallel with dependency ordering
- **Council mode** — multi-model deliberation with a chair model
- **Plan mode** — AI-generated step-by-step plans with execution & checkpointing
- **Repo-awareness** — automatic discovery of project structure, README, pyproject.toml, and conventions
- **MCP integration** — connect external Model Context Protocol servers
- **Skill discovery** — auto-discover and load skill files from repo/user dirs
- **Persistent JavaScript REPL** — optional stateful Node.js tool for calculations, prototyping, and debugging JS
- **PDF vision analysis** — render PDF pages for the active model to inspect text, charts, and figures
- **Beautiful CLI** with Rich terminal output, themes, and streaming
- **CLI client mode** — bypass the SDK to access all models via subprocess
- **Metrics & cost tracking** — token usage, timing, success rates

## Installation

```bash
pip install copex
```

Or with [pipx](https://pipx.pypa.io/) for isolated CLI usage:

```bash
pipx install copex
```

Optional extras:

```bash
pip install "copex[pdf]"            # Enable --pdf-analyze
pip install "copex[repo-map]"       # Enhanced repository mapping
pip install "copex[scrape]"         # Built-in Scrapling MCP integration
pip install -e ".[dev]"             # Test/lint tooling for contributors
```

Extras can be combined, for example: `pip install "copex[pdf,repo-map]"`.

**Prerequisite:** The GitHub Copilot CLI must be installed and authenticated:

```bash
npm i -g @github/copilot
copilot login
```

## Quick Start

### CLI

```bash
# One-shot single-agent chat
copex chat "Explain the builder pattern in Python"

# One-shot squad run from the root command
copex -p "Build a REST API with auth and tests"

# Interactive session
copex interactive

# Pipe from stdin
echo "What is a monad?" | copex chat --stdin

# Root prompt, but disable squad and force single-agent mode
copex -p "Explain quicksort" --no-squad

# Enable the persistent JavaScript REPL tool for this session
copex chat "Check this JS array logic" --js-repl

# Choose model and reasoning
copex chat "Optimize this SQL" -m gpt-5.2-codex -r xhigh
```

### Python API

```python
import asyncio
from copex import Copex, CopexConfig, Model, ReasoningEffort

async def main():
    config = CopexConfig(
        model=Model.CLAUDE_OPUS_4_6,
        reasoning_effort=ReasoningEffort.HIGH,
    )
    async with Copex(config) as client:
        response = await client.send("Explain quicksort")
        print(response.content)

asyncio.run(main())
```

#### Streaming

```python
async with Copex(config) as client:
    async for chunk in client.stream("Write a prime sieve"):
        if chunk.type == "message":
            print(chunk.delta, end="", flush=True)
        elif chunk.type == "reasoning":
            print(f"\033[2m{chunk.delta}\033[0m", end="", flush=True)
```

## Squad Orchestration

Squad is Copex's built-in multi-agent team that auto-decomposes tasks and executes them in parallel:

```bash
# Root prompt path — orchestrate with Lead, Developer, Tester
copex -p "Build a REST API with auth and tests"

# Or explicitly:
copex squad "Build a REST API with CRUD operations"

# With specific model:
copex squad "Build a REST API" -m gpt-5.2-codex -r high
```

**How Squad works:**
1. **Lead Architect** analyzes the task, breaks it into steps, identifies patterns and conventions
2. **Developer** implements according to the plan (parallel with Tester)
3. **Tester** writes comprehensive tests (parallel with Developer)
4. **Docs Expert** updates documentation and examples

Squad auto-discovers project context (README, pyproject.toml, directory structure) and applies project-specific conventions. Results are tracked via Fleet with adaptive concurrency, retries, and cost tracking.

```python
from copex import SquadCoordinator, CopexConfig

async with SquadCoordinator(config) as squad:
    result = await squad.run("Build a REST API with auth")
    print(result.final_content)
```

Use `copex chat` for direct single-agent mode, or keep the root prompt flow and add `--no-squad`:

```bash
copex chat "Explain quicksort"
copex -p "Explain quicksort" --no-squad
```

## Agent Loops

For machine-consumption or fine-grained turn-by-turn control, use the Agent command with JSON Lines output:

```bash
# JSON Lines: one JSON object per turn (for subprocess/orchestrator consumption)
copex agent "Build a simple HTTP server" --json --max-turns 10
```

Each turn outputs JSON with: `turn`, `content`, `tool_calls`, `stop_reason`, `error`, `duration_ms`.

```python
from copex import AgentSession, CopexConfig, make_client

config = CopexConfig()
client = make_client(config)

async with AgentSession(client, max_turns=10) as agent:
    result = await agent.run("Implement a stack data structure")
    for turn in result.turns:
        print(f"Turn {turn.turn}: {turn.content[:100]}...")
```

## Configuration

Copex loads configuration from `~/.config/copex/config.toml` automatically.
Generate a starter config with:

```bash
copex init
```

### Config file example

```toml
model = "claude-opus-4.6"
reasoning_effort = "high"
streaming = true
use_cli = false
timeout = 300.0
auto_continue = true
js_repl = false
# js_repl_node_path = "/path/to/node"
pdf_analyze = false
ui_theme = "default"       # default, midnight, mono, sunset, tokyo
ui_density = "extended"    # compact, extended

[retry]
max_retries = 5
base_delay = 1.0
max_delay = 30.0
exponential_base = 2.0

# Skills
skills = ["code-review"]
# skill_directories = ["/path/to/skills"]
# disabled_skills = ["some-skill"]

# MCP servers
# mcp_config_file = "~/.config/copex/mcp.json"

# Tool filtering
# available_tools = ["bash", "view"]
# excluded_tools = ["dangerous-tool"]
```

### Environment variables

| Variable | Description |
|---|---|
| `COPEX_MODEL` | Override the default model |
| `COPEX_REASONING` | Override the reasoning effort |

## JavaScript REPL Tool

Enable the optional persistent JavaScript REPL with `--js-repl` on `copex chat`, the root `copex -p ...` flow, or `copex interactive`.

```bash
copex chat "Inspect this snippet" --js-repl
copex interactive --js-repl --js-repl-node /custom/node
```

When enabled, Copex registers two domain tools for the model:

- `js_repl` — execute JavaScript in a persistent Node.js VM context
- `js_repl_reset` — clear the persistent JS state

If Node.js is not on `PATH`, set `js_repl_node_path` in `config.toml` or pass `--js-repl-node`. Startup failures now surface the kernel's stderr output to make troubleshooting easier.

## CLI Commands

The root command also accepts `-p/--prompt` to run a prompt without a subcommand; that path uses squad by default and honors `--no-squad`.

| Command | Description |
|---|---|
| `copex plan` | Generate and execute step-by-step plans |
| `copex fleet` | Run multiple tasks in parallel |
| `copex council` | Run a council workflow: Codex + Gemini + Opus, with Opus as chair |
| `copex squad` | Run a squad or manage the repo `.squad` file |
| `copex chat` | Send a prompt through the direct single-agent chat command |
| `copex models` | List available models |
| `copex render` | Render JSONL session logs with readable formatting |
| `copex tui` | Start the Copex TUI |
| `copex init` | Create a default config file |
| `copex interactive` | Start an interactive chat session |
| `copex ralph` | Start a Ralph Wiggum iterative loop |
| `copex login` | Login to GitHub |
| `copex logout` | Logout from GitHub |
| `copex status` | Check Copilot CLI and GitHub authentication status |
| `copex config` | Validate and display the current Copex configuration |
| `copex map` | Display the repository map or task-relevant files |
| `copex campaign` | Run a campaign: discover targets, batch them, and run fleet waves |
| `copex agent` | Run an agent loop: prompt -> tool calls -> respond -> repeat |
| `copex edit` | Apply structured edits (unified diff, SEARCH/REPLACE, whole-file blocks) |
| `copex undo` | Undo structured edits from the latest (or listed) undo batch |
| `copex completions` | Generate shell completion scripts |
| `copex stats` | Show statistics for recent Copex runs |
| `copex audit` | Show approval audit log entries |
| `copex diff` | Show what changed since the last Copex run started |
| `copex memory` | Manage persistent project memory (`show`, `add`, `clear`, `import`) |
| `copex skills` | Manage and discover skills (`list`, `show`) |

### Common flags

```
-m, --model         Model to use
-r, --reasoning     Reasoning effort (none, low, medium, high, xhigh)
-c, --config        Config file path
-S, --skill-dir     Add skill directory
    --use-cli       Use CLI subprocess instead of SDK
    --json          Machine-readable JSON output
-q, --quiet         Content only, no formatting
```

## Models

Copex supports all models available through the Copilot SDK:

| Model | Reasoning | xhigh |
|---|---|---|
| `gpt-5.4` | ✅ | ✅ |
| `gpt-5.3-codex` | ✅ | ✅ |
| `gpt-5.2-codex` | ✅ | ✅ |
| `gpt-5.1-codex` | ✅ | ❌ |
| `gpt-5.1-codex-max` | ✅ | ❌ |
| `gpt-5.1-codex-mini` | ✅ | ❌ |
| `gpt-5.2` | ✅ | ✅ |
| `gpt-5.1` | ✅ | ❌ |
| `gpt-5` | ✅ | ❌ |
| `gpt-5-mini` | ✅ | ❌ |
| `gpt-4.1` | ❌ | ❌ |
| `claude-sonnet-4.5` | ❌ | ❌ |
| `claude-sonnet-4` | ❌ | ❌ |
| `claude-haiku-4.5` | ❌ | ❌ |
| `claude-opus-4.5` | ❌ | ❌ |
| `claude-opus-4.6` | ✅ | ❌ |
| `claude-opus-4.6-fast` | ✅ | ❌ |
| `claude-opus-4.6-1m` | ✅ | ❌ |
| `claude-sonnet-4.6` | ✅ | ❌ |
| `gemini-3-pro-preview` | ❌ | ❌ |

Copex also discovers models dynamically from `copilot --help` at runtime, so newly added models work automatically.

```bash
copex models  # List all available models
```

## Reasoning Effort

Five levels control how much thinking the model does:

| Level | Description |
|---|---|
| `none` | No extended reasoning |
| `low` | Minimal reasoning |
| `medium` | Balanced |
| `high` | Thorough reasoning (default for CLI commands) |
| `xhigh` | Maximum reasoning (GPT-5.2+ only) |

If you request an unsupported level, Copex automatically downgrades and warns you. The CLI defaults `--reasoning` to `high`; config and API values are still normalized per model capability.

## Advanced Features

### Retry & Backoff

Copex uses adaptive per-error-category backoff strategies:

```python
config = CopexConfig(
    retry=RetryConfig(
        max_retries=5,
        base_delay=1.0,
        max_delay=30.0,
        exponential_base=2.0,
    ),
)
```

Error categories (rate limit, network, server, auth, client) each have their own backoff curve. Rate limit errors respect `Retry-After` headers when available.

### Circuit Breaker

A sliding-window circuit breaker opens when the failure rate exceeds 50% in the last 10 requests, then enters a 60-second cooldown before retrying.

### Model Fallback Chains

```python
client = Copex(config, fallback_chain=["claude-opus-4.6", "gpt-5.2-codex", "gpt-5.1-codex"])
```

If the primary model fails, Copex transparently tries the next model in the chain.

### Session Pooling

```python
from copex.client import SessionPool

pool = SessionPool(max_sessions=5, max_idle_time=300)
async with pool.acquire(client, config) as session:
    await session.send({"prompt": "Hello"})
```

### Ralph Wiggum Loops

Iterative AI development: the same prompt is fed repeatedly, with the AI seeing its own previous work to iteratively improve.

```bash
copex ralph "Build a REST API with CRUD and tests" \
  --promise "ALL TESTS PASSING" \
  -n 20
```

```python
from copex.ralph import RalphWiggum

ralph = RalphWiggum(copex_client)
result = await ralph.loop(
    prompt="Build a REST API with CRUD operations",
    completion_promise="API COMPLETE",
    max_iterations=30,
)
```

### Plan Mode

AI-generated step-by-step execution with checkpointing and resume:

```bash
copex plan "Build a REST API" --execute
copex plan "Build a REST API" --review        # Confirm before executing
copex plan --resume                           # Resume from checkpoint
copex plan "Build a REST API" --visualize ascii
```

### Fleet Mode

Run multiple tasks in parallel with optional dependency ordering:

```bash
copex fleet "Write tests" "Fix linting" "Update docs" --max-concurrent 3
copex fleet --file tasks.toml
```

```python
from copex import Fleet, FleetConfig

async with Fleet(config) as fleet:
    fleet.add("Write auth tests")
    fleet.add("Refactor DB", depends_on=["write-auth-tests"])
    results = await fleet.run()
```

Features: adaptive concurrency, rate-limit backoff, git finalize, artifact export.

### Council Mode

Multi-model deliberation — multiple models investigate a problem, then a chair model synthesizes the best solution:

```bash
copex council "Design a caching strategy for our API" \
  --chair-model claude-opus-4.6 \
  --codex-model gpt-5.2-codex \
  --gemini-model gemini-3-pro-preview
```

### MCP Integration

Connect external Model Context Protocol servers:

```toml
# In config.toml
mcp_config_file = "~/.config/copex/mcp.json"
```

Or pass inline:

```bash
copex fleet --mcp-config mcp.json "Analyze the codebase"
```

### Checkpointing & Persistence

Ralph loops and plan execution save checkpoints to disk for crash recovery. Sessions can be saved and restored across runs.

## CLI Client Mode

Use `--use-cli` to bypass the SDK and invoke the Copilot CLI directly as a subprocess. This is useful when the SDK doesn't support a model but the CLI does:

```bash
copex chat "Hello" --use-cli -m claude-opus-4.6
```

```python
config = CopexConfig(use_cli=True, model=Model.CLAUDE_OPUS_4_6)
client = make_client(config)  # Returns CopilotCLI instead of Copex
```

## Development

```bash
# Clone and install
git clone https://github.com/Arthur742Ramos/copex.git
cd copex
pip install -e ".[dev]"

# Include optional feature extras when working on those paths
pip install -e ".[dev,pdf,repo-map,scrape]"

# Run tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=copex --cov-report=term-missing

# Lint
ruff check src/
ruff format src/
```

## License

[MIT](LICENSE)
