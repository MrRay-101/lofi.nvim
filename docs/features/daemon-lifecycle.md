/speckit.specify Daemon lifecycle management for lofi-daemon process

## Context

This feature implements `lua/lofi/daemon.lua` which manages the lofi-daemon Rust binary. The daemon handles model inference and audio playback, communicating via JSON-RPC over stdin/stdout.

## Constitution Alignment

- Principle I (Zero-Latency): spawn daemon async, never block editor during startup
- Principle III (Async-First): all daemon communication via non-blocking stdio
- Principle IV (Minimal Footprint): single binary backend, no runtime dependencies

## Architecture

Reference design.md "Backend: lofi-daemon" section:
```
Neovim ←→ stdin/stdout (JSON-RPC) ←→ lofi-daemon (Rust binary)
```

## Requirements

### Daemon spawning
- Spawn on first command that requires daemon (generate, play, status)
- Use vim.loop.spawn() for non-blocking process creation
- Pass CLI flags: --idle-timeout N, --pid-file PATH
- Locate binary: config.daemon.bin or auto-detect in plugin dir (bin/lofi-daemon)

### Stdin/stdout IPC
- Write JSON-RPC requests to daemon stdin
- Read JSON-RPC responses and notifications from stdout
- Parse newline-delimited JSON messages
- Correlate responses to requests via JSON-RPC `id` field
- Buffer partial reads; handle message boundaries correctly

### Orphan prevention
Per design.md "Orphan Prevention":
- Daemon monitors stdin; exits immediately on EOF (Neovim crash/exit)
- Daemon exits after config.daemon.idle_timeout_sec of no messages (default: 300)
- Daemon writes PID to $XDG_RUNTIME_DIR/lofi-daemon-{nvim_pid}.pid

### Graceful shutdown
- On VimLeave autocmd: send shutdown RPC, close stdin, wait for exit
- Timeout after 2s; force kill if daemon hangs
- Clean up PID file

### Crash recovery
Per design.md "Crash Recovery Behavior":
- If daemon exits unexpectedly and config.daemon.restart_on_crash = true:
  - Fire `daemon_exit` event with exit code
  - Attempt restart (max 3 retries with exponential backoff)
  - Fire `daemon_restart` event on success
  - Re-send config (volume, prefetch settings) after reconnect
- State lost on crash: generation queue, current generation, playback position
- State preserved: disk cache (unaffected), Lua-side config

### Manual control commands
- :Lofi daemon start - manually spawn daemon
- :Lofi daemon stop - graceful shutdown
- :Lofi daemon restart - stop then start

## File structure
```
lua/lofi/
├── daemon.lua    -- spawn, IPC, lifecycle management
```

## JSON-RPC protocol
All daemon communication uses JSON-RPC 2.0:
```json
// Request (Lua → Daemon)
{"jsonrpc": "2.0", "method": "...", "params": {...}, "id": 1}

// Response (Daemon → Lua)
{"jsonrpc": "2.0", "result": {...}, "id": 1}

// Notification (Daemon → Lua, no id)
{"jsonrpc": "2.0", "method": "...", "params": {...}}
```

## Dependencies
- plugin-setup.md (config access)

## Error codes
- DAEMON_NOT_FOUND: binary not at expected path
- DAEMON_CRASH: process exited unexpectedly (include exit code)
- DAEMON_TIMEOUT: no response within timeout period
- DAEMON_SPAWN_FAILED: vim.loop.spawn() failed (include errno)

## Lua API
```lua
local daemon = require("lofi.daemon")
daemon.start()           -- spawn if not running
daemon.stop()            -- graceful shutdown
daemon.restart()         -- stop then start
daemon.is_running()      -- returns boolean
daemon.request(method, params, callback)  -- send RPC, invoke callback(err, result)
daemon.on(event, callback)  -- subscribe to notifications
```

## Success criteria
- Daemon spawns in <50ms after first command
- stdin EOF causes daemon exit within 1s
- Crash restart succeeds within 5s (with fresh state)
- No orphan daemons after Neovim force-quit (kill -9)
- PID file correctly tracks running daemon
