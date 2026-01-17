---
name: debugging
description: Debug issues in Copex, especially streaming, content display, and event handling bugs. Use this when investigating bugs, unexpected behavior, or errors.
---

# Debugging Guide for Copex

## Common Issues

### Stale/Wrong Content Displayed

**Symptoms:** Response shows previous turn's content, or old message appears instead of new one.

**Root Cause:** History fallback triggered during streaming mode.

**Investigation:**
1. Check if streaming (`on_chunk` provided)
2. Verify `received_content` flag is set when content arrives
3. Check if history fallback is being used incorrectly

**Fix Pattern:**
```python
# In client.py - only use history when NOT streaming
if not received_content and on_chunk is None:
    # History fallback OK here
    messages = await session.get_messages()
```

**In CLI - prefer streamed content:**
```python
final_message = ui.state.message if ui.state.message else response.content
```

### Empty or Missing Content

**Symptoms:** Response is empty, reasoning missing, or incomplete.

**Debug Steps:**
1. Add logging to `on_event()`:
   ```python
   print(f"Event: {event_type}, data: {event.data}")
   ```
2. Check `content_parts` and `reasoning_parts` lists
3. Verify final events are received

**Event Flow:**
```
ASSISTANT_REASONING_DELTA → reasoning_parts.append()
ASSISTANT_REASONING → final_reasoning = content
ASSISTANT_MESSAGE_DELTA → content_parts.append()
ASSISTANT_MESSAGE → final_content = content
SESSION_IDLE → done.set()
```

### Retry Not Working

**Check:**
1. `_should_retry()` logic in `client.py`
2. Error matches `retry_on_errors` patterns
3. `retry_on_any_error` config setting

**Debug:**
```python
print(f"Error: {error_str}")
print(f"Should retry: {self._should_retry(e)}")
```

### Timeout Despite Activity

**Check:** `last_activity` timestamp updates in `on_event()`:
```python
def on_event(event):
    nonlocal last_activity
    last_activity = asyncio.get_event_loop().time()  # Must update!
```

## Key Files to Check

| Issue | File | Function |
|-------|------|----------|
| Content/streaming | `client.py` | `_send_once()`, `on_event()` |
| UI display | `cli.py` | `_stream_response*()` |
| UI components | `ui.py` | `CopexUI`, `build_*()` |
| Retry logic | `client.py` | `send()`, `_should_retry()` |
| Config | `config.py` | `CopexConfig` |

## Debugging Tools

### Inspect Raw Events
```python
response = await client.send(prompt)
for event in response.raw_events:
    print(f"{event['type']}: {event['data']}")
```

### Write Reproducing Test
```python
def test_reproduces_bug():
    events = [
        # Exact sequence that causes the bug
    ]
    session = FakeSession(events)
    # Verify bug exists, then fix
```

## Debug Checklist

1. [ ] Reproduce consistently
2. [ ] Identify component (client/cli/ui)
3. [ ] Add logging to trace events
4. [ ] Check raw_events for unexpected data
5. [ ] Write failing test
6. [ ] Fix and verify
7. [ ] Remove debug logging
