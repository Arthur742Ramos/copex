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

### Other commands

```bash
# List available models
copex models

# Create default config file
copex init
```

## Configuration

Create a config file at `~/.config/copex/config.toml`:

```toml
model = "gpt-5.2-codex"
reasoning_effort = "xhigh"
streaming = true
timeout = 300.0
auto_continue = true
continue_prompt = "Keep going"

# Skills to enable
skills = ["code-review", "api-design", "test-writer"]

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
