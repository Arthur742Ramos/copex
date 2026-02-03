# Copex UI Design Document

## Executive Summary

This document outlines the UI design philosophy and implementation for copex, drawing inspiration from the best CLI AI tools: OpenCode, Claude Code, Codex CLI, and Aider.

## Research Findings

### OpenCode (Go + Bubble Tea)
- **Structure**: Component-based architecture with separate modules for chat, dialogs, logs
- **Key patterns**:
  - Status bar at bottom with model, session info
  - Dialogs for model selection, file picking, help
  - Keyboard shortcuts (Ctrl+K commands, Ctrl+L logs, Ctrl+O models)
  - Theme system with switchable themes

### Aider (Python + Rich)
- **Streaming**: Uses `MarkdownStream` class with Rich's `Live` display
- **Key patterns**:
  - Sliding window for visible content during streaming
  - Progressive markdown rendering
  - Custom code blocks without padding
  - Left-justified headings

### Common Patterns Across Tools
1. **Clear activity states**: Thinking, streaming, tool calls, complete
2. **Progress indicators**: Spinners, progress bars, elapsed time
3. **Collapsible sections**: Tool calls can expand/collapse
4. **Status bars**: Model, tokens, cost, duration at a glance
5. **Keyboard shortcuts**: Discoverable via help or palette
6. **Clean visual hierarchy**: Consistent use of color and spacing

---

## Design Philosophy

### Core Principles

1. **Reduce Anxiety**: Users should always know what's happening
2. **Progressive Disclosure**: Show summary by default, details on demand
3. **Visual Hierarchy**: Important info is prominent, details are subtle
4. **Consistency**: Same patterns throughout the application
5. **Performance**: Fast rendering, no jank during streaming

### Color Psychology

| Color | Purpose | Emotion |
|-------|---------|---------|
| Cyan | Primary actions, main content | Trust, clarity |
| Magenta/Purple | Reasoning, thinking | Intelligence, creativity |
| Yellow/Orange | Tool calls, warnings | Attention, activity |
| Green | Success, user input | Confirmation, progress |
| Red | Errors only | Alert (use sparingly) |
| Gray/Dim | Secondary info, hints | Calm, non-intrusive |

---

## States to Visualize

### 1. Waiting for API Response
```
 ⠋ Thinking...                    12.3s elapsed
```
- Animated spinner (braille style for smooth animation)
- Status text changes: "Connecting...", "Thinking...", "Reasoning..."
- Elapsed time counter (updates every 100ms)

### 2. Streaming Text
```
╭─ 🤖 Response ───────────────────────────────────╮
│ The quick brown fox jumps over the lazy dog.▌  │
╰─────────────────────────────────────────────────╯
```
- Blinking cursor at end of text
- Border changes color (active vs idle)
- Text appears incrementally

### 3. Reasoning (Extended Thinking)
```
╭─ 🧠 Reasoning ──────────────────────────────────╮
│ Let me think about this problem step by step... │
│ First, I need to understand the requirements... │
╰─────────────────────────────────────────────────╯
```
- Distinct color (magenta/purple)
- Can be collapsed/hidden with Ctrl+G
- Shows live during extended thinking

### 4. Tool Calls
```
 ⠹ 📖 read_file • path=src/main.py  3.2s  Running  ▸
```
Collapsed view shows:
- Status spinner/icon
- Tool icon (contextual)
- Tool name
- Key arguments (truncated)
- Duration
- Status label
- Expand chevron

Expanded view:
```
╭─ ▾ 📖 read_file ────────────────────────────────╮
│ Arguments                                        │
│   path        src/main.py                        │
│   encoding    utf-8                              │
│                                                  │
│ Output                                           │
│   def main():                                    │
│       print("Hello, world!")                     │
│   ...                                            │
╰─────────────────────────────────────────────────╯
```

### 5. Errors
```
╭─ ✗ Error ───────────────────────────────────────╮
│ Connection timeout after 30 seconds              │
│                                                  │
│ Retry 2/5 in 4.2s...                            │
╰─────────────────────────────────────────────────╯
```
- Red border
- Clear error message
- Retry information if applicable

### 6. Success/Complete
```
╭─ ✓ Summary ─────────────────────────────────────╮
│ ⏱ 45.2s elapsed            • 3 tool calls       │
│ 🔧 3 ok                     • no retries         │
╰─────────────────────────────────────────────────╯
```

---

## Progress Indicators

### Ralph Iterations
```
╭─ 🔄 Ralph Wiggum Loop ──────────────────────────╮
│ Iteration 3/20 ━━━━━━━━━━━━░░░░░░░░░░░░  15%    │
│ Status: Implementing feature X                   │
│ Promise: "ALL TESTS PASSING"                     │
╰─────────────────────────────────────────────────╯
```

### Plan Steps
```
╭─ 📋 Plan Execution ─────────────────────────────╮
│ Step 2/5: Implement tests                        │
│ ━━━━━━━━━━━━━━━━░░░░░░░░░░░░░  40% (~12m left)  │
│                                                  │
│ ✅ Step 1: Setup project structure (2m 14s)      │
│ ⏳ Step 2: Implement tests (running...)          │
│ ⬜ Step 3: Add documentation                     │
│ ⬜ Step 4: Write CLI                             │
│ ⬜ Step 5: Final review                          │
╰─────────────────────────────────────────────────╯
```

### Token/Cost Ticker
```
 🤖 claude-opus-4.5 │ 🧠 high │ ⏱ 12.3s │ 2,450 tokens │ $0.0234 │ ⠋ Responding...
```

---

## Layout Options

### Option 1: Simple (Rich Panels + Spinners) ✅ CURRENT
- Uses Rich's `Panel`, `Live`, `Spinner` components
- Best for: One-shot commands, non-interactive use
- Implementation: Current `CopexUI` class

### Option 2: Medium (Status Bar + Panels)
- Adds persistent status bar at bottom
- Keyboard shortcuts visible
- Implementation: Current TUI with `prompt_toolkit`

### Option 3: Advanced (Full TUI with Textual) 
- Split views, resizable panes
- Mouse support
- Best for: Power users, long sessions
- Implementation: Future consideration with Textual framework

---

## Component Specifications

### 1. Status Panel (Live Display)
```python
╭─ 🤖 Copex • claude-opus-4.5 ────────────────────╮
│                                                  │
│  ⠋ Responding...        ⏱ 12.3s │ updated 0.1s  │
│  🤖 2,450 chars         🧠 890 chars             │
│  🔧 2 running • 1 ok    $0.0234                  │
│                                                  │
╰─────────────────────────────────────────────────╯
```

### 2. Spinner Styles
| Style | Characters | Use Case |
|-------|------------|----------|
| braille | ⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏ | Default, smooth |
| dots | ⠁⠂⠄⡀⢀⠠⠐⠈ | Loading |
| pulse | ○◔◑◕●◕◑◔ | Modern, clean |
| arc | ◜◠◝◞◡◟ | Professional |
| bar | ▏▎▍▌▋▊▉█ | Progress-like |

### 3. Progress Bar
```
━━━━━━━━━━━━░░░░░░░░░░░░░░░  40%
```
- Filled: `━` (bright)
- Empty: `░` (dim)
- Optional percentage text

### 4. Tool Icons
| Tool Pattern | Icon | Style |
|-------------|------|-------|
| read/view | 📖 | Info |
| write/edit | 📝 | Warning |
| create | 📄 | Success |
| search/grep/glob | 🔍 | Info |
| shell/bash | 💻 | Warning |
| web/fetch | 🌐 | Info |
| default | ⚡ | Warning |

---

## Theme System

### Default Theme
```python
PRIMARY = "cyan"
SECONDARY = "blue"
ACCENT = "magenta"
SUCCESS = "green"
WARNING = "yellow"
ERROR = "red"
```

### Midnight Theme (High Contrast)
```python
PRIMARY = "bright_cyan"
ACCENT = "bright_magenta"
BORDER = "grey39"
```

### Mono Theme (Accessibility)
```python
PRIMARY = "white"
ACCENT = "white"
# Relies on bold/dim for hierarchy
```

### Sunset Theme (Warm)
```python
PRIMARY = "bright_yellow"
ACCENT = "bright_magenta"
```

---

## Implementation Checklist

### Already Implemented ✅
- [x] `CopexUI` class with live display
- [x] Spinner animations (multiple styles)
- [x] Tool call collapsed/expanded views
- [x] Reasoning panels
- [x] Theme system with presets (default, midnight, mono, sunset)
- [x] Status panel with metrics
- [x] Summary panel after completion
- [x] Full TUI with command palette
- [x] **NEW: `RalphUI` class** - Beautiful Ralph loop visualization
- [x] **NEW: `PlanUI` class** - Step-by-step plan execution display
- [x] **NEW: `build_progress_bar()`** - Styled progress bars
- [x] **NEW: `format_duration()`** - Human-readable time formatting
- [x] **NEW: Ralph iteration progress bar with ETA**
- [x] **NEW: Plan step overview with status icons**
- [x] **NEW: Plan step completion with ETA display**

### Future Considerations 🔮
- [ ] Textual-based TUI for advanced features
- [ ] Split view (input + output)
- [ ] Conversation history browser
- [ ] Export to markdown/HTML
- [ ] Custom theme editor
- [ ] Live token/cost ticker during streaming

---

## Usage Examples

### CLI Chat (One-shot)
```bash
$ copex chat "Explain Python decorators" --model claude-opus-4.5 --reasoning high
```
Shows: Status panel → Reasoning panel → Response panel → Summary

### Interactive Mode
```bash
$ copex
```
Shows: Welcome banner → Prompt → Live display → Summary → Prompt...

### Ralph Loop
```bash
$ copex ralph "Build REST API with tests" --max-iterations 20 --promise "ALL TESTS PASSING"
```
Shows: Loop header → Iteration progress → Per-iteration summary → Final summary

### Plan Execution
```bash
$ copex plan "Build feature X" --execute
```
Shows: Plan overview → Step progress → Per-step results → Plan summary

---

## Accessibility Notes

1. **Color-blind friendly**: Use shapes/icons in addition to color
2. **Screen reader compatible**: Alt text for spinners
3. **High contrast option**: Mono theme available
4. **Keyboard navigation**: Full functionality without mouse
5. **Reduced motion option**: Static indicators available

---

## Performance Considerations

1. **Render throttling**: Max 20 FPS for live display
2. **Content truncation**: Long content truncated in live view
3. **Lazy rendering**: Only visible content rendered
4. **String buffering**: Batch small updates together
5. **Memory management**: Clear old messages periodically

---

## Conclusion

The copex UI should make users feel **calm and in control**. Every state should be clearly communicated, progress should be visible, and errors should be actionable. The visual design draws from the best CLI AI tools while maintaining its own identity through clean panels, smooth animations, and thoughtful use of color.
