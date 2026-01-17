---
name: testing
description: Write and run tests for Copex. Use this when asked to write tests, fix tests, improve coverage, or debug test failures.
---

# Testing Guide for Copex

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=copex --cov-report=term-missing

# Run single test file
python -m pytest tests/test_client_streaming.py -v

# Run single test
python -m pytest tests/test_client_streaming.py::test_name -v
```

## Test Location

Tests are in `tests/` directory:
- `conftest.py` - Shared fixtures
- `test_client_streaming.py` - Client streaming and event handling

## Writing Tests

### Mock the SDK with FakeSession

```python
from types import SimpleNamespace
from copex.client import Copex, StreamChunk
from copex.config import CopexConfig
from copex.models import EventType

class FakeSession:
    def __init__(self, events, messages=None):
        self._events = events
        self._messages = messages or []
        self._handler = None

    def on(self, handler):
        self._handler = handler
        return lambda: None  # unsubscribe

    async def send(self, _options):
        for event in self._events:
            if self._handler:
                self._handler(event)

    async def get_messages(self):
        return self._messages

def build_event(event_type: str, **data):
    return SimpleNamespace(type=event_type, data=SimpleNamespace(**data))
```

### Test Pattern

```python
def test_example():
    # Arrange - create events that simulate SDK behavior
    events = [
        build_event(EventType.ASSISTANT_MESSAGE_DELTA.value, delta_content="Hello"),
        build_event(EventType.ASSISTANT_MESSAGE.value, content="Hello"),
        build_event(EventType.SESSION_IDLE.value),
    ]
    session = FakeSession(events)
    
    # Setup client with fake session
    client = Copex(CopexConfig())
    client._started = True
    client._client = DummyClient()
    client._session = session

    # Act
    chunks: list[StreamChunk] = []
    response = asyncio.run(client.send("prompt", on_chunk=chunks.append))

    # Assert
    assert response.content == "Hello"
```

### Key Event Types

| Event | Purpose |
|-------|---------|
| `ASSISTANT_MESSAGE_DELTA` | Streaming content |
| `ASSISTANT_MESSAGE` | Final content |
| `ASSISTANT_REASONING_DELTA` | Streaming reasoning |
| `ASSISTANT_REASONING` | Final reasoning |
| `TOOL_EXECUTION_START` | Tool call started |
| `TOOL_EXECUTION_COMPLETE` | Tool finished |
| `SESSION_IDLE` | Turn complete |
| `SESSION_ERROR` | Error occurred |

### Testing Errors

```python
import pytest

def test_error_raises():
    events = [build_event(EventType.SESSION_ERROR.value, message="boom")]
    session = FakeSession(events)
    client = Copex(CopexConfig())
    client._started = True
    client._session = session

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(client.send("prompt"))
```

### Testing Streaming vs Non-Streaming

Important: History fallback behavior differs:
- With `on_chunk` (streaming): Does NOT use history fallback
- Without `on_chunk`: Uses history fallback if no content received

```python
def test_streaming_no_history_fallback():
    # Streaming should NOT use stale history
    events = [build_event(EventType.ASSISTANT_TURN_END.value)]
    stale = SimpleNamespace(type=EventType.ASSISTANT_MESSAGE.value,
                           data=SimpleNamespace(content="Stale"))
    session = FakeSession(events, messages=[stale])
    
    chunks = []
    response = asyncio.run(client.send("x", on_chunk=chunks.append))
    assert response.content != "Stale"  # Should NOT use history
```
