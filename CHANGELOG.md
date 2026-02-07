# Changelog

## 2.0.1 (2026-02-07)

### Bug Fixes

- **fix: patch SDK to remove `--no-auto-update` flag that caused silent model fallback**

  The Copilot SDK's `CopilotClient._start_cli_server` passes `--no-auto-update`
  when spawning the CLI server in headless mode. This prevents the CLI binary from
  fetching the up-to-date model catalogue from the Copilot backend, causing newer
  models (`claude-opus-4.6`, `claude-opus-4.6-fast`) to silently fall back to
  `claude-sonnet-4.5`. Copex now monkey-patches the startup method to strip
  `--no-auto-update`, so the CLI always has the current model list.

### New Models

- Added `claude-opus-4.6-fast` to the `Model` enum and fallback chains.
