# Copex - Copilot Extended

## Project Overview

Copex is a resilient Python wrapper for the GitHub Copilot SDK. It provides:
- Automatic retry with exponential backoff
- Ralph Wiggum loops for iterative AI development
- Session persistence and checkpointing
- Metrics and cost tracking
- Parallel tool execution
- MCP server integration
- Beautiful CLI with rich terminal output

## Repository Structure

```
src/copex/
├── client.py       # Core Copex client with retry logic and streaming
├── cli.py          # Typer CLI commands (chat, ralph, interactive, etc.)
├── config.py       # CopexConfig and configuration loading
├── models.py       # Model, ReasoningEffort, EventType enums
├── ralph.py        # RalphWiggum loop implementation
├── checkpoint.py   # CheckpointStore and CheckpointedRalph
├── persistence.py  # SessionStore and PersistentSession
├── metrics.py      # MetricsCollector for usage tracking
├── tools.py        # ParallelToolExecutor
├── mcp.py          # MCPManager and MCPServerConfig
├── ui.py           # Rich UI components (CopexUI, Theme, Icons)
└── __init__.py     # Public API exports
```

## Code Style

- Python 3.10+ with type hints everywhere
- Use `from __future__ import annotations` for forward references
- Prefer `dataclass` for data structures
- Use `async/await` for all I/O operations
- Follow PEP 8 naming conventions
- Minimal comments - code should be self-documenting
- Use Rich library for terminal output

## Key Patterns

### Streaming with Callbacks
```python
def on_chunk(chunk: StreamChunk) -> None:
    if chunk.type == "message":
        ui.add_message(chunk.delta)
    elif chunk.type == "reasoning":
        ui.add_reasoning(chunk.delta)

response = await client.send(prompt, on_chunk=on_chunk)
```

### Retry Logic
The client retries on errors using exponential backoff with jitter. Check `_should_retry()` and `_calculate_delay()` in `client.py`.

### Event Handling
Events from the SDK are handled in `on_event()` callback within `_send_once()`. Key events:
- `ASSISTANT_MESSAGE_DELTA` / `ASSISTANT_MESSAGE` - content
- `ASSISTANT_REASONING_DELTA` / `ASSISTANT_REASONING` - thinking
- `TOOL_EXECUTION_START` / `TOOL_EXECUTION_COMPLETE` - tools
- `ASSISTANT_TURN_END` / `SESSION_IDLE` - completion

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=copex --cov-report=term-missing
```

Tests use `FakeSession` to mock the Copilot SDK. See `tests/conftest.py` and `tests/test_client_streaming.py`.

## Building and Publishing

```bash
# Build package
python -m build

# Install locally for development
pip install -e .

# Publish happens automatically via GitHub Actions on release
```

## Version Management

- Version is defined in TWO places (keep in sync):
  - `pyproject.toml` → `version = "X.Y.Z"`
  - `src/copex/cli.py` → `__version__ = "X.Y.Z"`

## Dependencies

Core dependencies (from pyproject.toml):
- `copilot` - GitHub Copilot SDK
- `typer` - CLI framework
- `rich` - Terminal formatting
- `prompt-toolkit` - Interactive input
- `tomli` / `tomli-w` - TOML config

Dev dependencies:
- `pytest`, `pytest-asyncio`, `pytest-cov`
- `ruff` - Linting and formatting
