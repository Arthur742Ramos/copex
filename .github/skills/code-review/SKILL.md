---
name: code-review
description: Review code changes in Copex for bugs, style issues, and best practices. Use this when reviewing PRs or code changes.
---

# Code Review Guide for Copex

## Review Checklist

### Python Style
- [ ] Type hints on all functions
- [ ] `from __future__ import annotations` for forward refs
- [ ] PEP 8 naming (snake_case functions, CamelCase classes)
- [ ] Minimal comments (code should be self-documenting)
- [ ] `async/await` for all I/O

### Client Code (`client.py`)
- [ ] Events handled correctly in `on_event()`
- [ ] `received_content` flag set when content arrives
- [ ] Streaming mode doesn't use history fallback
- [ ] Proper error handling with `error_holder`
- [ ] Retry logic respects config

### CLI Code (`cli.py`)
- [ ] Uses Rich for terminal output
- [ ] Streaming prefers UI state over response object
- [ ] Proper async patterns

### Tests
- [ ] Uses `FakeSession` to mock SDK
- [ ] Tests both success and error paths
- [ ] Tests streaming with `on_chunk`

## Critical Patterns

### Stale Content Bug (MUST CHECK)

When streaming, NEVER use history fallback:
```python
# BAD
if not received_content:
    messages = await session.get_messages()  # Stale!

# GOOD
if not received_content and on_chunk is None:
    messages = await session.get_messages()
```

### UI State Priority

When streaming, prefer accumulated UI state:
```python
# BAD
ui.set_final_content(response.content, response.reasoning)

# GOOD
final_message = ui.state.message if ui.state.message else response.content
ui.set_final_content(final_message, final_reasoning)
```

### Version Sync

Both files must have same version:
- `pyproject.toml` → `version = "X.Y.Z"`
- `src/copex/cli.py` → `__version__ = "X.Y.Z"`

## Review Format

```markdown
## Summary
Brief description of changes

## Status
- [ ] Approved
- [ ] Changes Requested

## Critical Issues
- Must-fix problems

## Suggestions  
- Nice-to-have improvements

## Positive
- Good patterns used
```
