# Copex - Copilot Extended

[![Crates.io](https://img.shields.io/crates/v/copex.svg)](https://crates.io/crates/copex)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A fast, native CLI for GitHub Copilot with extended features: plan-and-execute workflows, Ralph Wiggum loops, and tool execution.

## Features

- 🚀 **Native Performance** - Written in Rust for speed and reliability
- 💬 **Chat Mode** - Interactive conversations with AI models
- 📋 **Plan Mode** - Multi-step task planning with automatic execution
- 🔁 **Ralph Mode** - Iterative AI loops that keep trying until success
- 🛠️ **Tool Execution** - Create files, view content, run bash commands
- 🎨 **Rich UI** - Syntax highlighting, streaming output, themes
- 💾 **Session Management** - Save and restore conversation history
- 🔌 **MCP Support** - Connect to external tool servers

## Installation

From crates.io:
```bash
cargo install copex
```

From source:
```bash
git clone https://github.com/Arthur742Ramos/copex
cd copex
cargo build --release
./target/release/copex --version
```

## Prerequisites

- [GitHub Copilot CLI](https://docs.github.com/en/copilot) installed and authenticated
- Active Copilot subscription

## Quick Start

### Chat - Simple Conversations
```bash
# Quick question
copex chat "Explain async/await in Rust"

# With specific model
copex chat "Write a binary search" --model gpt-5.2-codex

# With reasoning
copex chat "Complex problem" --model claude-opus-4.5 --reasoning high
```

### Plan - Multi-Step Workflows
```bash
# Generate and execute a plan
copex plan "Build a REST API with tests" --execute

# Review plan before executing
copex plan "Refactor auth module" --review

# Save plan for later
copex plan "Large feature" --output plan.json

# Resume from step 3
copex plan --load plan.json --execute --from-step 3
```

### Ralph - Iterative Until Success
```bash
# Keep trying until all tests pass
copex ralph "Fix the failing tests" --max-iterations 30

# With completion promise
copex ralph "Build feature X" --promise "ALL TESTS PASSING"
```

### Interactive Mode
```bash
copex interactive
```

### Session Management
```bash
# List sessions
copex session list

# Save current session
copex session save my-project

# Resume session
copex session load my-project
```

## Configuration

Create `~/.config/copex/config.toml`:
```toml
default_model = "gpt-5.2-codex"
default_reasoning = "high"
ui_theme = "monokai"

[retry]
max_retries = 5
base_delay_ms = 1000
```

## Models

```bash
# List available models
copex models

# Use specific model
copex chat "..." --model claude-opus-4.5
copex chat "..." --model gpt-5.2-codex
copex chat "..." --model gemini-3-pro-preview
```

## Reasoning Levels

- `none` - No extended thinking
- `low` - Light reasoning
- `medium` - Balanced (default)
- `high` - Deep reasoning
- `xhigh` - Maximum reasoning (supported by some models)

## Migration from Python

This is a complete rewrite of [copex](https://pypi.org/project/copex/) from Python to Rust. All features are preserved with improved performance.

The Python version is archived in `archive/python-v1.1.0/`.

## License

MIT
