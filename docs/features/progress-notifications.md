/speckit.specify Progress notifications and UI feedback system

## Context

This feature implements `lua/lofi/ui.lua` which handles displaying generation progress, playback events, and error notifications to the user. Supports multiple notification backends.

## Constitution Alignment

- Principle I (Zero-Latency): notifications MUST NOT block editor; use async display methods
- Principle IV (Minimal Footprint): gracefully degrade when optional UI plugins absent
- Principle V (Composability): expose events for user scripting

## Progress Reporting

Reference design.md "Progress Reporting" and "Progress Estimation":

### Generation progress
Daemon sends `generation_progress` notifications:
```json
{
  "track_id": "a1b2c3d4",
  "percent": 45,
  "tokens_generated": 450,
  "tokens_estimated": 1000,
  "eta_sec": 35
}
```

**Accuracy guarantees:**
- `percent` capped at 99 until generation actually completes
- If actual tokens exceed estimate, percent stays at 99
- `eta_sec` recalculated each update based on recent speed
- `tokens_estimated` may be revised during generation (it's an estimate)

### Display modes
Reference design.md "ui.progress Options":

| config.ui.progress | Behavior |
|-------------------|----------|
| `"cmdline"` | Show in cmdline: `lofi: generating 45%...` (default) |
| `"fidget"` | Use fidget.nvim LSP-style progress (if installed) |
| `"mini"` | Use mini.notify (if installed) |
| `"notify"` | Use vim.notify with replace (requires nvim-notify) |
| `"none"` | Silent - no progress, only completion notification |

### Throttling
- cmdline: max 2 updates/second to avoid flicker
- Use `vim.api.nvim_echo` with `{redraw = false}` for cmdline
- Other backends: respect their native update mechanisms

## Requirements

### Notification types
1. **Generation progress**: percentage, ETA (during generation)
2. **Generation complete**: track ready, duration, prompt summary
3. **Playback started**: now playing {track_id}
4. **Playback paused/resumed**: state change
5. **Playback ended**: finished, skipped, or stopped
6. **Errors**: daemon errors, generation failures, audio device issues

### Backend detection
On require("lofi.ui"):
1. Check for fidget.nvim: `pcall(require, "fidget")`
2. Check for mini.notify: `pcall(require, "mini.notify")`
3. Check for nvim-notify: `vim.notify ~= vim.api.nvim_echo`
4. Fall back to cmdline for progress, vim.notify for events

### Error display
- config.ui.notify = true enables vim.notify for events
- config.ui.notify_level = vim.log.levels.INFO (default)
- Errors always show at WARN or ERROR level regardless of config

### vim.notify integration
For "notify" progress mode:
- Use notification replacement (nvim-notify `replace` option)
- Single persistent notification updated in-place
- Clear notification on completion

## File structure
```
lua/lofi/
├── ui.lua            -- notifications, progress display
```

## Dependencies
- daemon-lifecycle.md (subscribe to daemon events)
- plugin-setup.md (config.ui settings)

## Optional integrations
- fidget.nvim: LSP-style progress spinner in corner
- mini.notify: Minimal notification system
- nvim-notify: Feature-rich notification manager

All optional; cmdline always available as fallback.

## Lua API
```lua
local ui = require("lofi.ui")

-- Internal API (called by other lofi modules)
ui.show_progress(track_id, percent, eta_sec)
ui.show_complete(track)
ui.show_error(message, code)
ui.show_playback_event(event, track_id)

-- Public: statusline component (see statusline.md)
ui.statusline()
```

## Events

Events fired via lofi.on() for user scripting:
```lua
lofi.on("generation_progress", function(data) end)  -- {track_id, percent, eta_sec}
lofi.on("generation_complete", function(data) end)  -- {track_id, path, ...}
lofi.on("generation_error", function(data) end)     -- {track_id, error, code}
lofi.on("playback_start", function(data) end)
lofi.on("playback_pause", function(data) end)
lofi.on("playback_end", function(data) end)
lofi.on("daemon_error", function(data) end)
```

## Success criteria
- cmdline progress updates without flicker
- fidget.nvim integration shows spinner when generating
- Progress clears immediately on completion
- Errors display at appropriate severity level
- Notifications respect notify_level config
- Event callbacks fire for all state changes
- No UI updates when config.ui.notify = false and progress = "none"
