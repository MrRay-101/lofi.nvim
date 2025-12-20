/speckit.specify Plugin initialization and configuration system for lofi.nvim

## Context

This feature implements `lua/lofi/init.lua` and `lua/lofi/config.lua` as specified in design.md.

## Constitution Alignment

- Principle I (Zero-Latency): setup() MUST complete in <10ms; defer daemon spawn to first command
- Principle IV (Minimal Footprint): pure Lua, no external dependencies
- Principle V (Composability): expose full Lua API, type annotations via LuaCATS

## Requirements

### setup() function
- Accept optional config table, merge with defaults
- Validate config structure and value types; fail loudly with clear messages
- Register user commands with completion (see User Commands in design.md)
- Register keymaps from config.keymaps (default mappings in design.md)
- Store merged config in module state for access by other modules

### Configuration schema
Reference the full config structure from design.md section "Configuration":
- daemon: bin, auto_start, auto_stop, restart_on_crash, idle_timeout_sec, log_level, log_file
- model: backend, weights_path, quantization, device, threads
- defaults: prompt, duration_sec, seed
- prefetch: enabled, strategy, presets
- playback: volume, loop, crossfade_sec, auto_play
- audio: device, format
- cache: dir, max_mb, max_tracks
- ui: notify, notify_level, progress, statusline, icons
- keymaps: toggle, generate, stop, skip, volume_up, volume_down, prompt, status

### Hot reload behavior
Per design.md "Config Hot Reload" section:
- Lua-side settings (keymaps, ui) apply immediately on re-calling setup()
- Daemon-side settings require :Lofi daemon restart
- Document which settings are hot-reloadable vs require restart

### Lazy loading
- Plugin should be loadable via `cmd = "Lofi"` or `keys = {...}` in lazy.nvim
- setup() should NOT spawn daemon or load model weights
- First :Lofi command triggers daemon spawn (see daemon-lifecycle.md)

## File structure
```
lua/lofi/
├── init.lua      -- setup(), public API exports
├── config.lua    -- defaults, validation, merge logic
```

## Dependencies
- None (entry point)

## Error codes
- INVALID_CONFIG: config validation failed (include field path and expected type)

## Success criteria
- setup({}) completes in <10ms (measure with vim.loop.hrtime)
- Invalid config throws descriptive error: "lofi: config.model.quantization must be one of Q4_K_M, Q5_K_M, Q8_0, F16"
- :Lofi command registered with subcommand completion
- Default keymaps work without explicit keymap config
