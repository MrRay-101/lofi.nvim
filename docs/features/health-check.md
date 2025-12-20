/speckit.specify Health check for :checkhealth lofi

## Context

This feature implements `lua/lofi/health.lua` providing a comprehensive health check via `:checkhealth lofi`. Validates all runtime dependencies and provides actionable guidance.

## Constitution Alignment

- Principle II (Local & Private): verify local components only
- Principle IV (Minimal Footprint): check optional dependencies gracefully

## Health Check Output

Reference design.md "Health Check" section:

```
lofi: require("lofi.health").check()

lofi.nvim ~
- OK Neovim >= 0.9.0
- OK lofi-daemon binary found: ~/.local/share/nvim/lofi/bin/lofi-daemon (v0.1.0)
- OK Model weights found: musicgen-small-Q4_K_M.gguf (198MB)
- OK Audio device available: Built-in Output (48000 Hz)
- OK CPU supports AVX2: yes
- INFO Generation estimate for 30s audio: ~60-90s (CPU)
- WARNING CUDA available but not enabled (set model.device = "cuda")

Audio encoding ~
- WARNING lame not found: MP3 caching disabled
  HINT: Install lame for ~85% smaller cache, or set audio.format = "wav"

Optional integrations ~
- OK telescope.nvim: picker available via :Telescope lofi
- OK fidget.nvim: progress reporting available
- WARNING sox not found: Telescope waveform preview disabled
```

## Requirements

### Required checks

#### Neovim version
- Minimum: 0.9.0 (for vim.loop stability)
- Check: `vim.version().major >= 0 and vim.version().minor >= 9`

#### Daemon binary
- Check: file exists at config.daemon.bin or default location
- Check: binary is executable
- Check: `lofi-daemon --version` returns valid version
- Report: path, version, file size

#### Model weights (if backend = "ggml")
- Check: file exists at config.model.weights_path or default location
- Check: file size matches expected for quantization level
- Report: path, file size, quantization level

#### Audio device
- Check: daemon can query audio devices (requires daemon running)
- If daemon not running: report "will verify on first use"
- Report: default device name, sample rate

### System capability checks

#### CPU features
- Check: AVX2 support (important for GGML performance)
- Method: parse /proc/cpuinfo (Linux) or sysctl (macOS)
- Report: AVX2 yes/no, thread count

#### GPU availability
- Check: CUDA available (nvidia-smi or similar)
- Check: Metal available (macOS only)
- Report: GPU type if detected, suggestion if not using

#### Generation time estimate
Based on hardware detected, estimate generation time:
- Modern CPU (AVX2): ~60-90s for 30s audio
- Apple Silicon: ~30-45s
- CUDA GPU: ~10-20s
- Older CPU: ~3-5min (suggest procedural fallback)

### Optional dependency checks

#### lame (MP3 encoding)
- Check: `vim.fn.executable("lame")`
- If missing and config.audio.format = "mp3": WARNING
- If missing and format = "wav": OK (not needed)
- Hint: install command for common package managers

#### sox (waveform preview)
- Check: `vim.fn.executable("sox")`
- If missing: INFO (optional feature disabled)
- Hint: needed only for Telescope waveform preview

#### telescope.nvim
- Check: `pcall(require, "telescope")`
- If present: OK, extension available
- If missing: INFO (optional)

#### fidget.nvim
- Check: `pcall(require, "fidget")`
- If present: OK, progress reporting available
- If missing: INFO (cmdline progress used instead)

#### mini.notify
- Check: `pcall(require, "mini.notify")`
- If present: OK, notification backend available

## File structure
```
lua/lofi/
├── health.lua        -- :checkhealth lofi implementation
```

## Dependencies
- plugin-setup.md (config access for paths)

## Lua API
```lua
-- Called by :checkhealth lofi
require("lofi.health").check()
```

## Implementation notes

Use Neovim health module:
```lua
local health = vim.health or require("health")
health.ok("message")
health.warn("message", "hint")
health.error("message", "hint")
health.info("message")
health.start("section name")
```

## Success criteria
- All required components have clear OK/ERROR status
- Missing optional components show INFO, not ERROR
- Hints provide actionable guidance (install commands, config changes)
- Generation time estimate helps users set expectations
- Works without daemon running (deferred checks where needed)
- No false positives (don't report errors for optional features)
