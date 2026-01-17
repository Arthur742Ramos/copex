---
name: feature
description: Add new features to Copex. Use this when asked to implement new functionality, add CLI commands, or extend the client.
---

# Feature Development Guide for Copex

## Repository Structure

```
src/copex/
├── client.py       # Core client with retry logic and streaming
├── cli.py          # Typer CLI commands
├── config.py       # CopexConfig and configuration
├── models.py       # Model, ReasoningEffort, EventType enums
├── ralph.py        # RalphWiggum loop implementation
├── checkpoint.py   # CheckpointStore and CheckpointedRalph
├── persistence.py  # SessionStore and PersistentSession
├── metrics.py      # MetricsCollector for usage tracking
├── tools.py        # ParallelToolExecutor
├── mcp.py          # MCPManager and MCPServerConfig
├── ui.py           # Rich UI components
└── __init__.py     # Public API exports
```

## Adding a CLI Command

1. Add command in `cli.py`:
```python
@app.command()
def mycommand(
    arg: Annotated[str, typer.Argument(help="Description")],
    option: Annotated[str, typer.Option("--opt", "-o", help="Description")] = "default",
) -> None:
    """Command description shown in help."""
    # Implementation
    asyncio.run(_mycommand_async(arg, option))
```

2. Add async implementation:
```python
async def _mycommand_async(arg: str, option: str) -> None:
    client = Copex(config)
    await client.start()
    try:
        # Implementation
    finally:
        await client.stop()
```

## Adding a New Event Type

1. Add to `models.py`:
```python
class EventType(str, Enum):
    # ... existing
    NEW_EVENT = "new.event"
```

2. Handle in `client.py` `on_event()`:
```python
elif event_type == EventType.NEW_EVENT.value:
    # Handle the event
    if on_chunk:
        on_chunk(StreamChunk(type="new_type", delta=...))
```

## Adding Config Options

1. Add to `CopexConfig` in `config.py`:
```python
@dataclass
class CopexConfig:
    # ... existing
    new_option: str = "default"
```

2. Update `to_client_options()` or `to_session_options()` if needed.

3. Update default config in `cli.py` `init` command.

## Patterns to Follow

### Async/Await
```python
async def my_function() -> str:
    result = await some_async_operation()
    return result
```

### Type Hints
```python
def process(items: list[str], callback: Callable[[str], None] | None = None) -> dict[str, Any]:
    ...
```

### Dataclasses
```python
@dataclass
class MyData:
    field: str
    optional: int | None = None
    items: list[str] = field(default_factory=list)
```

### Rich Output
```python
from rich.console import Console
from rich.panel import Panel

console = Console()
console.print(Panel("Content", title="Title", border_style="blue"))
```

## Testing New Features

1. Add test in `tests/test_*.py`
2. Use `FakeSession` to mock SDK
3. Run: `python -m pytest tests/ -v`

## Export Public API

Add to `src/copex/__init__.py`:
```python
from copex.newmodule import NewClass

__all__ = [
    # ... existing
    "NewClass",
]
```
