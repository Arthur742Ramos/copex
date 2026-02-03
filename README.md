# Copex - Copilot Extended

[![PyPI version](https://badge.fury.io/py/copex.svg)](https://badge.fury.io/py/copex)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/Arthur742Ramos/copex/actions/workflows/test.yml/badge.svg)](https://github.com/Arthur742Ramos/copex/actions/workflows/test.yml)

A resilient Python wrapper for the GitHub Copilot SDK with automatic retry, Ralph Wiggum loops, session persistence, metrics, parallel tools, and MCP integration.

## Features

- 🔄 **Automatic Retry** - Handles 500 errors, rate limits, and transient failures with exponential backoff
- 🚀 **Auto-Continue** - Automatically sends "Keep going" on any error
- 🔁 **Ralph Wiggum Loops** - Iterative AI development with completion promises
- 💾 **Session Persistence** - Save/restore conversation history to disk
- 📍 **Checkpointing** - Resume interrupted Ralph loops after crashes
- 📊 **Metrics & Logging** - Track token usage, timing, and costs
- ⚡ **Parallel Tools** - Execute multiple tool calls concurrently
- 🔌 **MCP Integration** - Connect to external MCP servers for extended capabilities
- 🎯 **Model Selection** - Easy switching between GPT-5.2-codex, Claude, Gemini, and more
- 🧠 **Reasoning Effort** - Configure reasoning depth from `none` to `xhigh`
- 💻 **Beautiful CLI** - Rich terminal output with markdown rendering
- 🖥️ **TUI** - Full-screen terminal UI with command palette (`copex tui`)

### New in v1.1.0

- 📋 **Plan Visualization** - ASCII, Mermaid, and tree views of execution plans (`--visualize`)
- 🛡️ **Security Module** - Input sanitization, path validation, and env var filtering
- ⚠️ **Typed Exceptions** - Structured error hierarchy with logging context
- 🔄 **Adaptive Retry Backoff** - Per-error-category retry strategies with jitter
- 💾 **Step Caching** - Cache step results with TTL to skip unchanged steps
- 🔀 **Conditional Steps** - Dynamic plan flows based on conditions and prior outputs
- 📦 **Step Templates** - Reusable templates for testing, building, and deployment workflows

## Installation

```bash
pip install copex
```

Or install from source:

```bash
git clone https://github.com/Arthur742Ramos/copex
cd copex
pip install -e .
```

## Prerequisites

- Python 3.10+
- [GitHub Copilot CLI](https://docs.github.com/en/copilot/how-tos/set-up/install-copilot-cli) installed
- Active Copilot subscription

**Note:** Copex automatically detects the Copilot CLI path on Windows, macOS, and Linux. If auto-detection fails, you can specify the path manually:

```python
config = CopexConfig(cli_path="/path/to/copilot")
```

Or check detection:

```python
from copex import find_copilot_cli
print(f"Found CLI at: {find_copilot_cli()}")
```

## Quick Start

### Python API

```python
import asyncio
from copex import Copex, CopexConfig, Model, ReasoningEffort

async def main():
    # Simple usage with defaults (gpt-5.2-codex, xhigh reasoning)
    async with Copex() as copex:
        response = await copex.chat("Explain async/await in Python")
        print(response)

    # Custom configuration
    config = CopexConfig(
        model=Model.GPT_5_2_CODEX,
        reasoning_effort=ReasoningEffort.XHIGH,
        retry={"max_retries": 10, "base_delay": 2.0},
        auto_continue=True,
    )
    
    async with Copex(config) as copex:
        # Get full response object with metadata
        response = await copex.send("Write a binary search function")
        print(f"Content: {response.content}")
        print(f"Reasoning: {response.reasoning}")
        print(f"Retries needed: {response.retries}")

asyncio.run(main())
```

### Ralph Wiggum Loops

The [Ralph Wiggum technique](https://ghuntley.com/ralph/) enables iterative AI development:

```python
from copex import Copex, RalphWiggum

async def main():
    async with Copex() as copex:
        ralph = RalphWiggum(copex)
        
        result = await ralph.loop(
            prompt="Build a REST API with CRUD operations and tests",
            completion_promise="ALL TESTS PASSING",
            max_iterations=30,
        )
        
        print(f"Completed in {result.iteration} iterations")
        print(f"Reason: {result.completion_reason}")
```

**How it works:**
1. The same prompt is fed to the AI repeatedly
2. The AI sees its previous work in conversation history
3. It iteratively improves until outputting `<promise>COMPLETION TEXT</promise>`
4. Loop ends when promise matches or max iterations reached

### Terminal UI (TUI)

Run the full-screen terminal UI:

```bash
copex tui --model gpt-5.2-codex --reasoning xhigh
```

Key bindings (highlights):
- Ctrl+P: command palette
- Ctrl+J: insert newline (multiline input)
- Enter: send
- Esc: close palette / go back
- Ctrl+C: cancel streaming
- Ctrl+Q: quit

### Skills, Instructions & MCP

Copex is fully compatible with Copilot SDK features:

```python
from copex import Copex, CopexConfig

config = CopexConfig(
    model=Model.GPT_5_2_CODEX,
    reasoning_effort=ReasoningEffort.XHIGH,
    
    # Enable skills
    skills=["code-review", "api-design", "security"],
    
    # Custom instructions
    instructions="Follow PEP 8. Use type hints. Prefer dataclasses.",
    # Or load from file:
    # instructions_file=".copilot/instructions.md",
    
    # MCP servers (inline or from file)
    mcp_servers=[
        {"name": "github", "url": "https://api.github.com/mcp/"},
    ],
    # mcp_config_file=".copex/mcp.json",
    
    # Tool filtering
    available_tools=["repos", "issues", "code_security"],
    excluded_tools=["delete_repo"],
)

async with Copex(config) as copex:
    response = await copex.chat("Review this code for security issues")
```

### Streaming

```python
async def stream_example():
    async with Copex() as copex:
        async for chunk in copex.stream("Write a REST API"):
            if chunk.type == "message":
                print(chunk.delta, end="", flush=True)
            elif chunk.type == "reasoning":
                print(f"[thinking: {chunk.delta}]", end="")
```

## CLI Usage

### Single prompt

```bash
# Basic usage
copex chat "Explain Docker containers"

# With options
copex chat "Write a Python web scraper" \
    --model gpt-5.2-codex \
    --reasoning xhigh \
    --max-retries 10

# From stdin (for long prompts)
cat prompt.txt | copex chat

# Show reasoning output
copex chat "Solve this algorithm" --show-reasoning

# Raw output (for piping)
copex chat "Write a bash script" --raw > script.sh
```

### Ralph Wiggum loop

```bash
# Run iterative development loop
copex ralph "Build a calculator with tests" --promise "ALL TESTS PASSING" -n 20

# Without completion promise (runs until max iterations)
copex ralph "Improve code coverage" --max-iterations 10
```

### Interactive mode

```bash
copex interactive

# With specific model
copex interactive --model claude-sonnet-4.5 --reasoning high
```

Interactive slash commands:
- `/model <name>` - Change model
- `/reasoning <level>` - Change reasoning effort
- `/models` - List available models
- `/new` - Start a new session
- `/status` - Show current settings
- `/tools` - Toggle full tool call list
- `/help` - Show commands

### Other commands

```bash
# List available models
copex models

# Create default config file
copex init

# List available skills (auto-discovered)
copex skills list

# Show skill content
copex skills show code-review
```

### Skills Management

Copex auto-discovers skills from:
- `.github/skills/` (in repo)
- `.claude/skills/` (in repo, Claude Code compatibility)
- `.copex/skills/` (in repo)
- `~/.config/copex/skills/` (personal skills)

```bash
# List all discovered skills
copex skills list

# Show a specific skill
copex skills show my-skill

# Add explicit skill directory
copex chat "Do something" --skill-dir ./my-skills

# Disable a specific skill
copex chat "Do something" --disable-skill broken-skill

# Disable auto-discovery
copex chat "Do something" --no-auto-skills
```

The same flags work on `interactive`, `ralph`, and `plan` commands.

## Configuration

Create a config file at `~/.config/copex/config.toml`:

```toml
model = "gpt-5.2-codex"
reasoning_effort = "xhigh"
streaming = true
timeout = 300.0
auto_continue = true
continue_prompt = "Keep going"

# Skills to enable (named skills)
skills = ["code-review", "api-design", "test-writer"]

# Skills auto-discovery
auto_discover_skills = true  # Auto-discover from repo and user dirs
skill_directories = []       # Explicit skill directories to add
disabled_skills = []         # Skills to disable by name

# Custom instructions (inline or file path)
instructions = "Follow our team coding standards. Prefer functional programming."
# instructions_file = ".copilot/instructions.md"

# MCP server config file
# mcp_config_file = ".copex/mcp.json"

# Tool filtering
# available_tools = ["repos", "issues", "code_security"]
excluded_tools = []

[retry]
max_retries = 5
retry_on_any_error = true
base_delay = 1.0
max_delay = 30.0
exponential_base = 2.0
```

## Available Models

| Model | Description |
|-------|-------------|
| `gpt-5.2-codex` | Latest Codex model (default) |
| `gpt-5.1-codex` | Previous Codex version |
| `gpt-5.1-codex-max` | High-capacity Codex |
| `gpt-5.1-codex-mini` | Fast, lightweight Codex |
| `claude-sonnet-4.5` | Claude Sonnet 4.5 |
| `claude-sonnet-4` | Claude Sonnet 4 |
| `claude-opus-4.5` | Claude Opus (premium) |
| `gemini-3-pro-preview` | Gemini 3 Pro |

## Reasoning Effort Levels

| Level | Description |
|-------|-------------|
| `none` | No extended reasoning |
| `low` | Minimal reasoning |
| `medium` | Balanced reasoning |
| `high` | Deep reasoning |
| `xhigh` | Maximum reasoning (best for complex tasks) |

**Note:** `xhigh` is only supported for GPT/Codex models **gpt-5.2+** (e.g. `gpt-5.2`, `gpt-5.2-codex`, and higher). If you request `xhigh` on other models (e.g. Claude), Copex will **downgrade to `high`** and emit a warning.

## Error Handling

By default, Copex retries on **any error** (`retry_on_any_error=True`).

You can also be specific:

```python
config = CopexConfig(
    retry={
        "retry_on_any_error": False,
        "max_retries": 10,
        "retry_on_errors": ["500", "timeout", "rate limit"],
    }
)
```

## Credits

- **Ralph Wiggum technique**: [Geoffrey Huntley](https://ghuntley.com/ralph/)
- **GitHub Copilot SDK**: [github/copilot-sdk](https://github.com/github/copilot-sdk)

## v1.1.0 Features

### Plan Visualization

Visualize execution plans in ASCII, Mermaid, or tree format:

```bash
# Generate plan with ASCII visualization
copex plan "Build a REST API" --visualize ascii

# Mermaid diagram (for docs/GitHub)
copex plan "Build a REST API" --visualize mermaid

# Simple tree view
copex plan "Build a REST API" --visualize tree
```

Programmatic usage:

```python
from copex import Plan, visualize_plan, render_ascii, render_mermaid

plan = Plan(task="Build API", steps=[...])

# Get ASCII visualization
print(visualize_plan(plan, format="ascii"))

# Get Mermaid diagram
print(render_mermaid(plan))
```

### Typed Exception Hierarchy

All exceptions inherit from `CopexError` for unified handling:

```python
from copex import (
    CopexError,        # Base exception
    ConfigError,       # Invalid configuration
    MCPError,          # MCP server/tool errors
    RetryError,        # All retries exhausted
    PlanExecutionError, # Plan step failed
    ValidationError,   # Input validation failed
    SecurityError,     # Security violation
    TimeoutError,      # Operation timed out
    RateLimitError,    # Rate limit exceeded
    ConnectionError,   # Network connection failed
)

try:
    result = await copex.send(prompt)
except RateLimitError as e:
    print(f"Rate limited, retry after {e.retry_after}s")
except CopexError as e:
    print(f"Copex error: {e.message}")
    print(f"Context: {e.context}")
```

### Adaptive Retry Backoff

Per-error-category retry strategies:

```python
from copex import AdaptiveRetry, BackoffStrategy, ErrorCategory

# Custom strategies per error type
retry = AdaptiveRetry(
    strategies={
        ErrorCategory.RATE_LIMIT: BackoffStrategy(
            base_delay=5.0,
            max_delay=120.0,
            max_retries=10,
        ),
        ErrorCategory.NETWORK: BackoffStrategy(
            base_delay=1.0,
            max_delay=30.0,
            max_retries=5,
        ),
    },
    max_total_time=300.0,  # 5 minute timeout
)

# Use with any async operation
result = await retry.execute(some_async_function)

# Or as decorator
@retry.wrap()
async def my_operation():
    ...
```

### Step Caching

Cache step results to skip unchanged steps:

```python
from copex import StepCache, get_cache

cache = get_cache(default_ttl=3600)  # 1 hour TTL

# Compute hash for a step
step_hash = cache.compute_hash(
    description="Run tests",
    inputs={"framework": "pytest"},
    file_paths=["tests/", "src/"],  # Content affects hash
)

# Check cache
if cached := cache.get(step_hash):
    print(f"Cache hit: {cached.result}")
else:
    result = run_step()
    cache.set(step_hash, result)

# Cache stats
print(cache.stats())
```

### Conditional Plan Steps

Dynamic plan flows based on conditions:

```python
from copex import Condition, ConditionContext, when, all_of, any_of

# Built-in conditions
run_if_success = Condition.step_completed(1)
run_if_failed = Condition.step_failed(1)
check_env = Condition.env_equals("CI", "true")

# Expression-based conditions
deploy_condition = when("${step.3.status} == 'completed'")

# Combine conditions
full_check = all_of(
    Condition.step_completed(1),
    Condition.step_completed(2),
    "${env.DEPLOY_ENABLED} == 'true'",
)

# Evaluate against context
context = ConditionContext(
    step_outputs={1: "Tests passed"},
    step_statuses={1: "completed"},
    env=os.environ,
    variables={},
)

if full_check.evaluate(context):
    run_deployment()
```

### Step Templates

Reusable templates for common workflows:

```python
from copex import (
    create_step,
    get_registry,
    test_workflow,
    build_workflow,
    deploy_workflow,
)

# Use built-in templates
test_step = create_step(
    "run_tests",
    test_framework="pytest",
    directory="tests/",
    options="-v --cov",
    command="pytest tests/ -v --cov",
)

# Pre-built workflows
steps = test_workflow(framework="pytest", directory="tests")
steps = build_workflow(project_name="myapp", build_command="python -m build")
steps = deploy_workflow(environment="prod", target="api", version="1.2.0")

# Register custom templates
from copex import StepTemplate

my_template = StepTemplate(
    name="security_scan",
    description_template="Security scan with {scanner}",
    prompt_template="Run {scanner} on {paths}...",
    tags=["security"],
)

registry = get_registry()
registry.register(my_template)
```

### Security Module

Input sanitization and validation:

```python
from copex import (
    filter_env_vars,
    sanitize_command,
    validate_path,
    validate_mcp_tool_name,
    SecurityError,
)

# Filter environment variables (allowlist-based)
safe_env = filter_env_vars(
    os.environ,
    include_prefixes=["COPEX_", "MY_APP_"],
)

# Sanitize commands (prevent injection)
try:
    safe_cmd = sanitize_command(["ls", "-la", user_input])
except SecurityError as e:
    print(f"Dangerous input: {e}")

# Validate paths (prevent traversal)
try:
    path = validate_path(
        user_path,
        base_dir="/allowed/directory",
        must_exist=True,
    )
except SecurityError as e:
    print(f"Path violation: {e}")
```

## Contributing

Contributions welcome! Please open an issue or PR at [github.com/Arthur742Ramos/copex](https://github.com/Arthur742Ramos/copex).

## License

MIT

---

## Advanced Features

### Session Persistence

Save and restore conversation history:

```python
from copex import Copex, SessionStore, PersistentSession

store = SessionStore()  # Saves to ~/.copex/sessions/

# Create a persistent session
session = PersistentSession("my-project", store)

async with Copex() as copex:
    response = await copex.chat("Hello!")
    session.add_user_message("Hello!")
    session.add_assistant_message(response)
    # Auto-saved to disk

# Later, restore it
session = PersistentSession("my-project", store)
print(session.messages)  # Previous messages loaded
```

### Checkpointing (Crash Recovery)

Resume Ralph loops after interruption:

```python
from copex import Copex, CheckpointStore, CheckpointedRalph

store = CheckpointStore()  # Saves to ~/.copex/checkpoints/

async with Copex() as copex:
    ralph = CheckpointedRalph(copex, store, loop_id="my-api-project")
    
    # Automatically resumes from last checkpoint if interrupted
    result = await ralph.loop(
        prompt="Build a REST API with tests",
        completion_promise="ALL TESTS PASSING",
        max_iterations=30,
        resume=True,  # Resume from checkpoint
    )
```

### Metrics & Cost Tracking

Track token usage and estimate costs:

```python
from copex import Copex, MetricsCollector

collector = MetricsCollector()

async with Copex() as copex:
    # Track a request
    req = collector.start_request(
        model="gpt-5.2-codex",
        prompt="Write a function..."
    )
    
    response = await copex.chat("Write a function...")
    
    collector.complete_request(
        req.request_id,
        success=True,
        response=response,
    )

# Get summary
print(collector.print_summary())
# Session: 20260117_170000
# Requests: 5 (5 ok, 0 failed)
# Success Rate: 100.0%
# Total Tokens: 12,450
# Estimated Cost: $0.0234

# Export metrics
collector.export_json("metrics.json")
collector.export_csv("metrics.csv")
```

### Parallel Tools

Execute multiple tools concurrently:

```python
from copex import Copex, ParallelToolExecutor

executor = ParallelToolExecutor()

@executor.tool("get_weather", "Get weather for a city")
async def get_weather(city: str) -> str:
    return f"Weather in {city}: Sunny, 72°F"

@executor.tool("get_time", "Get time in timezone")
async def get_time(timezone: str) -> str:
    return f"Time in {timezone}: 2:30 PM"

# Tools execute in parallel when AI calls multiple at once
async with Copex() as copex:
    response = await copex.send(
        "What's the weather in Seattle and the time in PST?",
        tools=executor.get_tool_definitions(),
    )
```

### MCP Server Integration

Connect to external MCP servers:

```python
from copex import Copex, MCPManager, MCPServerConfig

manager = MCPManager()

# Add MCP servers
manager.add_server(MCPServerConfig(
    name="github",
    command="npx",
    args=["-y", "@github/mcp-server"],
    env={"GITHUB_TOKEN": "..."},
))

manager.add_server(MCPServerConfig(
    name="filesystem",
    command="npx", 
    args=["-y", "@anthropic/mcp-server-filesystem", "/path/to/dir"],
))

await manager.connect_all()

# Get all tools from all servers
all_tools = manager.get_all_tools()

# Call a tool
result = await manager.call_tool("github:search_repos", {"query": "copex"})

await manager.disconnect_all()
```

**MCP Config File** (`~/.copex/mcp.json`):

```json
{
  "servers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@github/mcp-server"],
      "env": {"GITHUB_TOKEN": "your-token"}
    },
    "browser": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-puppeteer"]
    }
  }
}
```

```python
from copex import load_mcp_config, MCPManager

configs = load_mcp_config()  # Loads from ~/.copex/mcp.json
manager = MCPManager()
for config in configs:
    manager.add_server(config)
await manager.connect_all()
```
