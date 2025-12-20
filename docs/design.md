# lofi.nvim — Design Document (v3)

> Local AI-powered lofi beat generation for Neovim

## Overview

`lofi.nvim` is a Neovim plugin that generates ambient lofi beats locally using lightweight AI models. Designed for developers who want non-distracting background music while coding, without leaving the editor or relying on external services.

---

## Goals

- **Zero-latency startup** — lazy load everything; never block init.lua
- **Fully local** — no API keys, no network, no telemetry
- **Async-first** — all inference runs in background jobs
- **Composable** — expose Lua API for scripting and integration
- **Minimal footprint** — single binary backend, pure Lua frontend

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Neovim                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                    lofi.nvim (Lua)                    │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │  │
│  │  │ Config  │  │Commands │  │   API   │  │  State  │   │  │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘   │  │
│  │       └───────────┴───────────┴───────────┘          │  │
│  │                        │                              │  │
│  │              ┌─────────▼─────────┐                    │  │
│  │              │   Job Controller  │                    │  │
│  │              │  (vim.loop/libuv) │                    │  │
│  │              └─────────┬─────────┘                    │  │
│  └────────────────────────│──────────────────────────────┘  │
└───────────────────────────│─────────────────────────────────┘
                            │ stdin/stdout (JSON-RPC)
                    ┌───────▼───────┐
                    │  lofi-daemon  │
                    │   (Rust bin)  │
                    │               │
                    │ ┌───────────┐ │
                    │ │MusicGen.cpp│ │
                    │ │  (GGML)   │ │
                    │ └───────────┘ │
                    │ ┌───────────┐ │
                    │ │  rodio/   │ │
                    │ │  cpal     │ │
                    │ └───────────┘ │
                    └───────────────┘
```

### Components

| Component | Language | Responsibility |
|-----------|----------|----------------|
| `lua/lofi/*.lua` | Lua | Config, commands, keymaps, state, UI |
| `lofi-daemon` | Rust | Model inference, audio playback, IPC |
| Model weights | GGUF | MusicGen-small Q4_K quantized (~200MB) |

---

## Risk Assessment

### Critical Path: GGML MusicGen Implementation

**Current State:**
- `PABannier/encodec.cpp` exists but is experimental (last commit: check before starting)
- No production-ready GGML port of MusicGen exists
- No established `convert.py` for MusicGen → GGUF (unlike llama.cpp ecosystem)

**Required Work:**

| Task | Effort | Risk |
|------|--------|------|
| Fork/extend encodec.cpp for MusicGen | 2-4 weeks | High — architecture differences |
| Build GGUF conversion pipeline | 1-2 weeks | Medium — need to map tensor names |
| Quantization testing for audio quality | 1 week | Medium — Q4 may degrade audio |
| Cross-platform testing | 1 week | Low — GGML is mature |

**Mitigation Strategies:**

1. **Phased approach**: Ship v0.1 with procedural generation only, add AI generation in v0.2
2. **Alternative backend**: If GGML proves too difficult, consider:
   - `candle` (Rust ML framework) — would need to port MusicGen
   - Embedded Python via `PyO3` — defeats "no runtime" but unblocks shipping
   - ONNX after all — accept the complexity if someone else does the export
3. **Scope reduction**: Target a simpler model first (e.g., MusicLM-small if available, or a custom lightweight architecture)

**Go/No-Go Checkpoint:**
Before building the Neovim plugin, spend 2 weeks on a standalone Rust CLI that:
1. Loads MusicGen weights (any format)
2. Generates 10s of audio from a text prompt
3. Runs on CPU in <2 minutes

If this isn't achievable, revisit architecture.

---

## Backend: `lofi-daemon`

A single static binary handling inference and audio. Communicates via JSON-RPC over stdin/stdout.

### Lifecycle & Orphan Handling

```
┌─────────────┐     spawn      ┌─────────────┐
│   Neovim    │ ─────────────▶ │   Daemon    │
│             │ ◀───────────── │             │
│             │   stdin/stdout │             │
└─────────────┘                └─────────────┘
      │                              │
      │ VimLeave / crash             │ stdin EOF / timeout
      ▼                              ▼
   (exit)                    (graceful shutdown)
```

**Orphan Prevention:**

The daemon monitors stdin for activity:
- If stdin closes (Neovim exits normally or crashes), daemon exits immediately
- If no JSON-RPC messages received for `idle_timeout` (default: 5 minutes), daemon exits
- Daemon writes PID to `$XDG_RUNTIME_DIR/lofi-daemon.pid` for manual cleanup

```jsonc
// Daemon CLI flags
lofi-daemon --idle-timeout 300  // exit after 5min of inactivity (0 = never)
lofi-daemon --pid-file /path/to/file.pid
```

**Crash Recovery Behavior:**

When `restart_on_crash = true` and daemon restarts:

| State | Behavior |
|-------|----------|
| Generation queue | **Lost** — not persisted. User notified via `daemon_error` event. |
| Current generation | **Lost** — partial results discarded. |
| Playback position | **Lost** — playback stops. User must call `play()` again. |
| Cached tracks | **Preserved** — cache is on disk, unaffected. |
| Volume/settings | **Restored** — Lua plugin re-sends config on reconnect. |

Rationale: Persisting queue adds complexity (need IPC for queue state, recovery logic). Simpler to lose in-flight work on crash—it's a rare event.

### Audio Pipeline

```
┌──────────────┐    32kHz     ┌──────────────┐    48kHz     ┌──────────────┐
│   MusicGen   │ ──────────▶  │  Resampler   │ ──────────▶  │ Audio Device │
│   (mono)     │              │  (rubato)    │              │              │
└──────────────┘              └──────────────┘              └──────────────┘
```

**Sample Rate Handling:**
- MusicGen outputs 32kHz mono
- Daemon resamples to device's preferred rate (typically 44.1kHz or 48kHz)
- Uses `rubato` crate for high-quality resampling
- Resampling happens during playback, not at generation time (cached files stay 32kHz)

**Stereo Handling:**
- MusicGen outputs mono
- Daemon duplicates to stereo for playback (simple L=R)
- Future: optional pseudo-stereo enhancement (Haas effect / subtle EQ differences)

### Model Strategy

#### Primary: MusicGen via GGML

**Weights format:** GGUF (GGML Universal Format)

Quantization options:

| Variant | Size | Quality | Speed | Recommendation |
|---------|------|---------|-------|----------------|
| Q4_K_M | ~200MB | Good | Fast | Default — best balance |
| Q5_K_M | ~250MB | Better | Medium | Quality-focused users |
| Q8_0 | ~400MB | Best | Slow | Audiophiles only |
| F16 | ~600MB | Reference | Slowest | Development/testing |

**GGUF Conversion Pipeline:**

Since no standard converter exists, we must build one:

```python
# scripts/convert_musicgen_to_gguf.py (to be implemented)

# 1. Load PyTorch checkpoint from HuggingFace
# 2. Extract and rename tensors to GGUF convention
# 3. Quantize weights (using GGML quantization functions)
# 4. Write GGUF file with proper metadata

# Key challenges:
# - MusicGen has transformer + EnCodec decoder (two models)
# - EnCodec uses residual vector quantization (RVQ)
# - Need to handle both in single GGUF or separate files
```

**Fallback: Procedural Generation**

For systems that can't run inference (or while AI backend is in development):

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ Drum loops  │ + │ Chord prog  │ + │ Vinyl noise │ = Lofi beat
│ (samples)   │   │ (generated) │   │ (samples)   │
└─────────────┘   └─────────────┘   └─────────────┘
```

- Ships with ~5MB of royalty-free samples
- Randomizes loop selection, tempo (70-90 BPM), key
- Applies lofi effects: bitcrusher, low-pass filter, tape wobble
- Not AI, but instant and guaranteed to work

### Realistic Performance Expectations

| Hardware | Model | 30s Audio | Notes |
|----------|-------|-----------|-------|
| Modern CPU (AVX2) | Q4_K_M | 60-90s | Acceptable for background generation |
| Modern CPU (AVX2) | Q8_0 | 90-120s | Better quality, slower |
| Apple Silicon (M1+) | Q4_K_M | 30-45s | Metal acceleration |
| CUDA GPU (RTX 3060+) | Q4_K_M | 10-20s | Optimal experience |
| Older CPU (no AVX2) | Q4_K_M | 3-5min | Consider procedural fallback |

### Binary Size Breakdown

```
Component                    Size (stripped)
─────────────────────────────────────────────
Rust runtime + logic         ~2MB
GGML (static)                ~3MB
rubato (resampler)           ~200KB
rodio/cpal (audio)           ~1MB
JSON-RPC / serde             ~500KB
─────────────────────────────────────────────
Total daemon binary          ~7-10MB
Model weights (Q4_K_M)       ~200MB (separate download)
Procedural samples           ~5MB (bundled or separate)
```

### JSON-RPC Interface

```jsonc
// ═══════════════════════════════════════════════════════════
// GENERATION
// ═══════════════════════════════════════════════════════════

// Request: Generate a new track
{
  "jsonrpc": "2.0",
  "method": "generate",
  "params": {
    "prompt": "lofi hip hop, jazzy piano, rain sounds",
    "duration_sec": 30,
    "seed": 42,                    // optional; null = random
    "priority": "normal"          // "normal" | "high" (high = skip queue)
  },
  "id": 1
}

// Response (immediate acknowledgment)
{
  "jsonrpc": "2.0",
  "result": {
    "track_id": "a1b2c3d4",
    "status": "queued",
    "position": 0                  // queue position (0 = generating now)
  },
  "id": 1
}

// Notification: Progress update (daemon → nvim)
// NOTE: percent is approximate and may not be perfectly linear.
// It's capped at 99 until generation actually completes.
{
  "jsonrpc": "2.0",
  "method": "generation_progress",
  "params": {
    "track_id": "a1b2c3d4",
    "percent": 45,                 // 0-99, capped until complete
    "tokens_generated": 450,
    "tokens_estimated": 1000,      // "estimated" not "total" — may be revised
    "eta_sec": 35                  // best-effort estimate
  }
}

// Notification: Generation complete (daemon → nvim)
{
  "jsonrpc": "2.0",
  "method": "generation_complete",
  "params": {
    "track_id": "a1b2c3d4",
    "path": "/home/user/.cache/nvim/lofi/a1b2c3d4.wav",
    "duration_sec": 30.2,
    "sample_rate": 32000,
    "format": "wav",
    "prompt": "lofi hip hop, jazzy piano, rain sounds",
    "seed": 42,                    // actual seed used (useful if input was null)
    "tokens_actual": 1024,         // actual token count (for debugging estimates)
    "generation_time_sec": 72.5
  }
}

// ═══════════════════════════════════════════════════════════
// PREFETCH
// ═══════════════════════════════════════════════════════════

// Request: Configure prefetch behavior
{
  "method": "prefetch_config",
  "params": {
    "enabled": true,
    "strategy": "same_prompt",     // "same_prompt" | "preset_cycle" | "random_preset"
    "presets": [                   // used by preset_cycle and random_preset
      "lofi hip hop, rainy day, mellow",
      "lofi beats, coffee shop, jazz piano",
      "chill lofi, late night coding, ambient"
    ]
  },
  "id": 2
}

// Prefetch triggers automatically when:
// 1. Current track reaches 50% playback AND
// 2. No track is queued AND
// 3. prefetch.enabled = true
//
// Prefetch prompt selection:
// - same_prompt: reuse current track's prompt with new seed
// - preset_cycle: rotate through presets in order
// - random_preset: randomly select from presets

// ═══════════════════════════════════════════════════════════
// QUEUE MANAGEMENT
// ═══════════════════════════════════════════════════════════

// Request: Get generation queue status
{ "method": "queue_status", "id": 3 }

// Response
{
  "result": {
    "current": {
      "track_id": "a1b2c3d4",
      "percent": 45
    },
    "pending": [
      { "track_id": "e5f6g7h8", "prompt": "chill beats...", "position": 1 },
      { "track_id": "i9j0k1l2", "prompt": "rainy day...", "position": 2 }
    ],
    "prefetch_pending": true       // indicates prefetch is queued
  }
}

// Request: Cancel generation
{ "method": "queue_cancel", "params": { "track_id": "e5f6g7h8" }, "id": 4 }

// Request: Clear entire queue (keeps current generation)
{ "method": "queue_clear", "id": 5 }

// ═══════════════════════════════════════════════════════════
// PLAYBACK
// ═══════════════════════════════════════════════════════════

// Request: Play track (adds to playlist, starts if not playing)
{ "method": "play", "params": { "track_id": "a1b2c3d4" }, "id": 6 }

// Request: Playlist operations
{ "method": "playlist_add", "params": { "track_id": "a1b2c3d4" }, "id": 7 }
{ "method": "playlist_remove", "params": { "track_id": "a1b2c3d4" }, "id": 8 }
{ "method": "playlist_clear", "id": 9 }
{ "method": "playlist_get", "id": 10 }  // returns ordered list

// Request: Playback control
{ "method": "pause", "id": 11 }
{ "method": "resume", "id": 12 }
{ "method": "stop", "id": 13 }          // stops and clears playlist position
{ "method": "skip", "id": 14 }          // next track in playlist

// Request: Volume (0.0 - 1.0)
{ "method": "volume_set", "params": { "level": 0.5 }, "id": 15 }
{ "method": "volume_get", "id": 16 }

// Request: Crossfade setting (daemon handles internally)
{ "method": "crossfade_set", "params": { "duration_sec": 2.0 }, "id": 17 }

// Request: Audio device
{ "method": "audio_devices", "id": 18 }  // list available
{ "method": "audio_device_set", "params": { "device": "Built-in Output" }, "id": 19 }

// Response: audio_devices
{
  "result": {
    "current": "Built-in Output",
    "available": [
      { "name": "Built-in Output", "is_default": true, "sample_rate": 48000 },
      { "name": "External Headphones", "is_default": false, "sample_rate": 44100 },
      { "name": "DisplayPort Audio", "is_default": false, "sample_rate": 48000 }
    ]
  }
}

// ═══════════════════════════════════════════════════════════
// STATUS
// ═══════════════════════════════════════════════════════════

{ "method": "status", "id": 20 }

// Response
{
  "result": {
    "playback": {
      "state": "playing",         // "playing" | "paused" | "stopped"
      "track_id": "a1b2c3d4",
      "position_sec": 12.5,
      "duration_sec": 30.0,
      "volume": 0.3
    },
    "generation": {
      "active": true,
      "track_id": "e5f6g7h8",
      "percent": 67
    },
    "queue_length": 2,
    "playlist_length": 5,
    "cache_size_mb": 234,
    "audio_device": "Built-in Output",
    "model": {
      "backend": "ggml",          // "ggml" | "procedural"
      "quantization": "Q4_K_M",
      "device": "cpu"             // "cpu" | "cuda" | "metal"
    }
  }
}

// Notification: Playback events (daemon → nvim)
{ "method": "playback_started", "params": { "track_id": "a1b2c3d4" } }
{ "method": "playback_paused", "params": { "track_id": "a1b2c3d4" } }
{ "method": "playback_ended", "params": { "track_id": "a1b2c3d4", "reason": "finished" } }
// reason: "finished" | "skipped" | "stopped" | "error"

// ═══════════════════════════════════════════════════════════
// CACHE MANAGEMENT
// ═══════════════════════════════════════════════════════════

{ "method": "cache_list", "id": 21 }   // list all cached tracks
{ "method": "cache_delete", "params": { "track_id": "a1b2c3d4" }, "id": 22 }
{ "method": "cache_clear", "id": 23 }  // delete all cached tracks
{ "method": "cache_stats", "id": 24 }  // size, count, oldest, newest
```

### Track ID Generation

Track IDs are generated deterministically to enable cache deduplication:

```
track_id = sha256(prompt + seed + duration_sec + model_version)[:8]
```

- Same prompt + seed + duration = same track_id = cache hit
- If seed is null/random, the daemon generates a seed first, then computes track_id
- 8 hex chars = 4 billion unique IDs, sufficient for local cache
- `model_version` included so quantization changes don't serve stale cache

### Progress Estimation

**Challenge:** MusicGen generates tokens autoregressively. The total token count depends on duration, but the mapping isn't perfectly linear due to EnCodec's variable-rate encoding.

**Approach:**
```
tokens_estimated = duration_sec * tokens_per_second_estimate
// tokens_per_second_estimate ≈ 50 (tuned empirically)

percent = min(99, floor(tokens_generated / tokens_estimated * 100))
```

**Guarantees:**
- `percent` never exceeds 99 until `generation_complete` fires
- If actual tokens exceed estimate, percent stays at 99
- `eta_sec` is recalculated each update based on recent generation speed
- Final `generation_complete` includes `tokens_actual` for calibrating estimates

### Crossfade Behavior

Crossfade is handled entirely by the daemon:

```
Track A: ████████████████████████░░░░  (fade out last N seconds)
Track B:                     ░░░░████████████████████████  (fade in first N seconds)
         ←── crossfade_sec ──→
```

**Edge Cases:**
- If `track.duration < crossfade_sec * 2`: crossfade reduced to `track.duration / 4`
- If `track.duration < 2 seconds`: no crossfade (too short)
- Crossfade only applies when transitioning between playlist tracks
- Manual `skip` command respects crossfade; `stop` does not (immediate silence)

### Concurrent Generation

**Design decision: Serial generation with queue**

Rationale:
- Parallel generation would compete for CPU/GPU resources, slowing both
- Memory usage would spike (model loaded once, but intermediate tensors multiply)
- Queue provides predictable behavior and simpler state management

If a user calls `generate` while generation is active:
1. New request is added to queue
2. Immediate response includes `position` in queue
3. When current generation completes, next in queue starts automatically

### Audio Format

| Property | Value |
|----------|-------|
| Native Sample Rate | 32000 Hz (MusicGen output) |
| Playback Sample Rate | Device-preferred (44100 or 48000 Hz, resampled) |
| Channels | Mono (duplicated to stereo for playback) |
| Bit Depth | 16-bit PCM |
| Cache Format | WAV (default) or MP3 (optional, see below) |
| WAV size | ~1.9MB per 30s |
| MP3 size | ~300KB per 30s (128kbps) |

**MP3 Encoding Behavior:**

```lua
-- Config
audio = {
  format = "mp3",  -- or "wav"
}
```

| Scenario | Behavior |
|----------|----------|
| `format = "wav"` | Always use WAV. No external dependencies. |
| `format = "mp3"` + lame installed | Encode to MP3, save ~85% cache space. |
| `format = "mp3"` + lame missing | **Warn at startup** via `:checkhealth` and `vim.notify`. Fall back to WAV silently for each track. Do not error. |

Daemon checks for `lame` binary on startup and caches result.

---

## Frontend: `lua/lofi/`

```
lua/
└── lofi/
    ├── init.lua          -- setup(), public API
    ├── config.lua        -- defaults, validation, merging
    ├── daemon.lua        -- spawn, IPC, lifecycle
    ├── commands.lua      -- user commands
    ├── state.lua         -- playback state, track cache
    ├── ui.lua            -- statusline component, notifications
    ├── health.lua        -- :checkhealth lofi
    └── telescope.lua     -- telescope extension (optional)
```

### Progress Reporting: `ui.progress` Options

| Value | Behavior |
|-------|----------|
| `"cmdline"` | Show progress in cmdline: `lofi: generating 45%...` (default) |
| `"fidget"` | Use fidget.nvim LSP-style progress (if installed) |
| `"mini"` | Use mini.notify (if installed) |
| `"notify"` | Use vim.notify with replace (requires nvim-notify or similar) |
| `"none"` | Silent — no progress shown, only completion notification |

**Implementation note:** `"cmdline"` uses `vim.api.nvim_echo` with `{redraw = false}` to avoid flicker. Progress updates are throttled to max 2/second.

### Telescope Extension

**Location:** `lua/lofi/telescope.lua` (lazy-loaded when Telescope calls extension)

```lua
-- Registration (in plugin/lofi.lua)
if pcall(require, "telescope") then
  require("telescope").register_extension({
    exports = {
      tracks = require("lofi.telescope").tracks,
      prompts = require("lofi.telescope").prompts,
    },
  })
end

-- Usage
:Telescope lofi tracks   -- browse cached tracks, preview waveform, play/delete
:Telescope lofi prompts  -- browse prompt history, regenerate with same/new seed
```

**Picker features:**
- `tracks`: Lists cached tracks with metadata (duration, prompt, created_at)
- Actions: `<CR>` play, `<C-d>` delete, `<C-a>` add to playlist
- Preview: ASCII waveform visualization (requires `sox` for waveform extraction, gracefully degrades)

### `init.lua` — Public API

```lua
local lofi = require("lofi")

-- ═══════════════════════════════════════════════════════════
-- SETUP
-- ═══════════════════════════════════════════════════════════

lofi.setup({
  -- see config section
})

-- ═══════════════════════════════════════════════════════════
-- GENERATION API
-- ═══════════════════════════════════════════════════════════

-- Generate a new track (async)
-- Returns immediately with track_id
-- @param opts? { prompt?: string, duration_sec?: number, seed?: number, priority?: "normal"|"high" }
-- @param callback? fun(err: table|nil, track: table|nil)
-- @return string track_id
lofi.generate(opts, callback)

-- Get generation queue status
-- @return { current: table|nil, pending: table[] }
lofi.queue()

-- Cancel a queued generation (cannot cancel active generation)
-- @param track_id string
lofi.cancel(track_id)

-- ═══════════════════════════════════════════════════════════
-- PLAYBACK API
-- ═══════════════════════════════════════════════════════════

-- Play a track (or resume if paused, or play last generated)
-- @param track_id? string  If nil, resumes or plays most recent
lofi.play(track_id)

-- Add track to end of playlist
-- @param track_id string
lofi.playlist_add(track_id)

-- Get current playlist
-- @return string[] track_ids
lofi.playlist()

-- Clear playlist
lofi.playlist_clear()

-- Playback control
lofi.pause()
lofi.resume()
lofi.toggle()             -- pause/resume
lofi.stop()               -- stop and reset playlist position
lofi.skip()               -- next track in playlist

-- Volume (0.0 - 1.0)
-- @param level? number  If nil, returns current volume
-- @return number|nil
lofi.volume(level)

-- ═══════════════════════════════════════════════════════════
-- STATUS API
-- ═══════════════════════════════════════════════════════════

-- Get full status
-- @return table  { playback: table, generation: table|nil, queue_length: number, ... }
lofi.status()

-- Check if currently playing
-- @return boolean
lofi.is_playing()

-- Check if currently generating
-- @return boolean
lofi.is_generating()

-- ═══════════════════════════════════════════════════════════
-- EVENTS API
-- ═══════════════════════════════════════════════════════════

-- Subscribe to events
-- @param event string
-- @param callback function
-- @return function unsubscribe
lofi.on(event, callback)

-- Events:
-- "generation_start"    -> { track_id }
-- "generation_progress" -> { track_id, percent, eta_sec }
-- "generation_complete" -> { track_id, path, duration_sec, ... }
-- "generation_error"    -> { track_id, error }
-- "playback_start"      -> { track_id }
-- "playback_pause"      -> { track_id }
-- "playback_resume"     -> { track_id }
-- "playback_end"        -> { track_id, reason }
-- "daemon_error"        -> { message, code }
-- "daemon_exit"         -> { code }
-- "daemon_restart"      -> {}  -- fired when auto-restart succeeds

-- ═══════════════════════════════════════════════════════════
-- CACHE API
-- ═══════════════════════════════════════════════════════════

-- List cached tracks
-- @return table[] { track_id, prompt, duration_sec, path, created_at }
lofi.cache()

-- Delete a cached track
-- @param track_id string
lofi.cache_delete(track_id)

-- Clear all cached tracks
lofi.cache_clear()
```

### Configuration

```lua
require("lofi").setup({
  -- ═══════════════════════════════════════════════════════════
  -- DAEMON
  -- ═══════════════════════════════════════════════════════════
  daemon = {
    bin = nil,                        -- nil = auto-detect in plugin dir
    auto_start = true,                -- start daemon on first command
    auto_stop = true,                 -- stop daemon on VimLeave
    restart_on_crash = true,          -- auto-restart if daemon exits unexpectedly
    idle_timeout_sec = 300,           -- daemon exits after N sec of inactivity (0 = never)
    log_level = "warn",               -- "debug" | "info" | "warn" | "error"
    log_file = nil,                   -- nil = no file logging; path to enable
  },

  -- ═══════════════════════════════════════════════════════════
  -- MODEL
  -- ═══════════════════════════════════════════════════════════
  model = {
    backend = "ggml",                 -- "ggml" | "procedural"
    weights_path = nil,               -- nil = auto (stdpath("data")/lofi/models)
    quantization = "Q4_K_M",          -- "Q4_K_M" | "Q5_K_M" | "Q8_0" | "F16"
    device = "auto",                  -- "auto" | "cpu" | "cuda" | "metal"
    threads = nil,                    -- nil = auto-detect; number of CPU threads
  },

  -- ═══════════════════════════════════════════════════════════
  -- GENERATION DEFAULTS
  -- ═══════════════════════════════════════════════════════════
  defaults = {
    prompt = "lofi hip hop, chill, mellow piano, vinyl crackle",
    duration_sec = 30,
    seed = nil,                       -- nil = random each time
  },

  -- ═══════════════════════════════════════════════════════════
  -- PREFETCH
  -- ═══════════════════════════════════════════════════════════
  prefetch = {
    enabled = true,                   -- auto-generate next track while playing
    strategy = "same_prompt",         -- "same_prompt" | "preset_cycle" | "random_preset"
    presets = {                       -- used by preset_cycle and random_preset
      "lofi hip hop, rainy day, mellow piano",
      "chill beats, coffee shop, jazz chords",
      "lofi, late night coding, ambient synth",
      "lofi hip hop, vinyl crackle, nostalgic",
    },
  },

  -- ═══════════════════════════════════════════════════════════
  -- PLAYBACK
  -- ═══════════════════════════════════════════════════════════
  playback = {
    volume = 0.3,                     -- default volume (0.0 - 1.0)
    loop = true,                      -- loop playlist when finished
    crossfade_sec = 2.0,              -- crossfade between tracks (0 to disable)
    auto_play = true,                 -- auto-play when generation completes
  },

  -- ═══════════════════════════════════════════════════════════
  -- AUDIO
  -- ═══════════════════════════════════════════════════════════
  audio = {
    device = nil,                     -- nil = system default; string = device name
    format = "wav",                   -- cache format: "wav" | "mp3" (mp3 requires lame)
  },

  -- ═══════════════════════════════════════════════════════════
  -- CACHE
  -- ═══════════════════════════════════════════════════════════
  cache = {
    dir = vim.fn.stdpath("cache") .. "/lofi",
    max_mb = 500,                     -- auto-cleanup when exceeded (LRU)
    max_tracks = nil,                 -- nil = no limit; number = max tracks
  },

  -- ═══════════════════════════════════════════════════════════
  -- UI
  -- ═══════════════════════════════════════════════════════════
  ui = {
    notify = true,                    -- use vim.notify for status messages
    notify_level = vim.log.levels.INFO,
    progress = "cmdline",             -- "cmdline" | "fidget" | "mini" | "notify" | "none"
    statusline = true,                -- expose statusline component
    icons = {
      playing = "▶",
      paused = "⏸",
      generating = "◌",
      music = "♪",
    },
  },

  -- ═══════════════════════════════════════════════════════════
  -- KEYMAPS (false to disable all, or set individual to false)
  -- ═══════════════════════════════════════════════════════════
  keymaps = {
    toggle = "<leader>ll",            -- play/pause toggle
    generate = "<leader>lg",          -- generate new track
    stop = "<leader>ls",              -- stop playback
    skip = "<leader>ln",              -- skip to next track
    volume_up = "<leader>l=",         -- volume +10%
    volume_down = "<leader>l-",       -- volume -10%
    prompt = "<leader>lp",            -- open prompt input
    status = "<leader>li",            -- show status float
  },
})
```

---

## User Commands

```vim
:Lofi                        " Toggle playback (generate if nothing cached)
:Lofi play [track_id]        " Play specific track or resume
:Lofi pause
:Lofi resume
:Lofi stop
:Lofi skip                   " Next track in playlist

:Lofi generate [prompt]      " Generate new track with optional prompt
:Lofi prompt                 " Open input for custom prompt
:Lofi queue                  " Show generation queue
:Lofi cancel [track_id]      " Cancel queued generation

:Lofi volume [0-100]         " Get or set volume
:Lofi device [name]          " Get or set audio device

:Lofi status                 " Show current state in float
:Lofi list                   " List cached tracks (Telescope picker if available)
:Lofi cache clear            " Clear cache

:Lofi log                    " Open daemon log in split
:Lofi daemon start           " Manually start daemon
:Lofi daemon stop            " Manually stop daemon
:Lofi daemon restart         " Restart daemon
```

---

## Health Check

`:checkhealth lofi` output:

```
lofi: require("lofi.health").check()

lofi.nvim ~
- OK Neovim >= 0.9.0
- OK lofi-daemon binary found: ~/.local/share/nvim/lofi/bin/lofi-daemon (v0.1.0)
- OK Model weights found: musicgen-small-Q4_K_M.gguf (198MB)
- OK Audio device available: Built-in Output (48000 Hz)
- OK CPU supports AVX2: yes
- INFO Generation estimate for 30s audio: ~60-90s (CPU)
- WARNING CUDA available but not enabled (set model.device = "cuda" for faster generation)

Audio encoding ~
- WARNING lame not found: MP3 caching disabled
  HINT: Install lame for ~85% smaller cache, or set audio.format = "wav" to silence this

Optional integrations ~
- OK telescope.nvim: picker available via :Telescope lofi
- OK fidget.nvim: progress reporting available
- WARNING sox not found: Telescope waveform preview disabled
```

---

## Installation

### lazy.nvim

```lua
{
  "username/lofi.nvim",
  build = "./install.sh",       -- downloads daemon + model weights
  cmd = "Lofi",                 -- lazy load on command
  keys = {
    { "<leader>ll", "<cmd>Lofi<cr>", desc = "Toggle lofi" },
    { "<leader>lg", "<cmd>Lofi generate<cr>", desc = "Generate lofi" },
  },
  opts = {
    -- your config
  },
}
```

### Build Script (`install.sh`)

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO="username/lofi.nvim"
DAEMON_REPO="username/lofi-daemon"
VERSION="${LOFI_VERSION:-latest}"

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="$SCRIPT_DIR/bin"
DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/nvim/lofi"
MODELS_DIR="$DATA_DIR/models"

# Platform detection
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64)  ARCH="x86_64" ;;
  aarch64|arm64) ARCH="aarch64" ;;
  *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac

echo "Installing lofi.nvim for $OS-$ARCH..."

# Create directories
mkdir -p "$BIN_DIR" "$MODELS_DIR"

# Resolve version
if [ "$VERSION" = "latest" ]; then
  VERSION=$(curl -sL "https://api.github.com/repos/$DAEMON_REPO/releases/latest" | grep '"tag_name"' | cut -d'"' -f4)
fi
echo "Version: $VERSION"

# Download daemon binary
DAEMON_URL="https://github.com/$DAEMON_REPO/releases/download/$VERSION/lofi-daemon-$OS-$ARCH.tar.gz"
echo "Downloading daemon from $DAEMON_URL..."
curl -sL "$DAEMON_URL" | tar -xz -C "$BIN_DIR"
chmod +x "$BIN_DIR/lofi-daemon"

# Verify binary
"$BIN_DIR/lofi-daemon" --version || { echo "Binary verification failed"; exit 1; }

# Download model weights (if not present or different quantization requested)
QUANT="${LOFI_QUANTIZATION:-Q4_K_M}"
MODEL_FILE="musicgen-small-$QUANT.gguf"
MODEL_PATH="$MODELS_DIR/$MODEL_FILE"

if [ ! -f "$MODEL_PATH" ]; then
  MODEL_URL="https://huggingface.co/username/musicgen-gguf/resolve/main/$MODEL_FILE"
  echo "Downloading model weights (~200MB)..."
  curl -L --progress-bar "$MODEL_URL" -o "$MODEL_PATH"
else
  echo "Model weights already present: $MODEL_PATH"
fi

# Verify model
echo "Verifying model..."
"$BIN_DIR/lofi-daemon" --verify-model "$MODEL_PATH" || { echo "Model verification failed"; exit 1; }

# Check optional dependencies
echo ""
echo "Checking optional dependencies..."
command -v lame >/dev/null && echo "  ✓ lame (MP3 encoding)" || echo "  ✗ lame not found (MP3 caching disabled)"
command -v sox >/dev/null && echo "  ✓ sox (waveform preview)" || echo "  ✗ sox not found (Telescope preview disabled)"

echo ""
echo "✓ lofi.nvim installed successfully"
echo "  Daemon: $BIN_DIR/lofi-daemon"
echo "  Model:  $MODEL_PATH"
echo ""
echo "Run :checkhealth lofi in Neovim to verify setup."
```

---

## Error Handling

```lua
-- All errors surfaced via events and vim.notify
lofi.on("daemon_error", function(err)
  -- err = { message = "...", code = "..." }
end)

lofi.on("generation_error", function(err)
  -- err = { track_id = "...", message = "...", code = "..." }
end)

-- Error codes:
-- DAEMON_NOT_FOUND      - binary not found at expected path
-- DAEMON_CRASH          - daemon process exited unexpectedly
-- DAEMON_TIMEOUT        - daemon not responding
-- MODEL_NOT_FOUND       - weights file not found
-- MODEL_LOAD_FAILED     - failed to load model (corrupt? wrong format?)
-- MODEL_INFERENCE_FAILED- inference error (OOM? numerical instability?)
-- AUDIO_DEVICE_ERROR    - failed to open audio device
-- AUDIO_PLAYBACK_ERROR  - error during playback
-- CACHE_WRITE_ERROR     - failed to write to cache
-- INVALID_TRACK_ID      - track_id not found in cache
-- LAME_NOT_FOUND        - MP3 requested but lame binary missing (warning, not fatal)
```

---

## Development Phases

### Phase 0: Feasibility (2 weeks)
- [ ] Build standalone Rust CLI that runs MusicGen inference
- [ ] Determine: GGML port, candle, or embedded Python?
- [ ] Produce 10s of audio from text prompt on CPU
- **Go/No-Go decision point**

### Phase 1: Procedural MVP (2 weeks)
- [ ] Daemon with procedural generation only
- [ ] Full JSON-RPC interface
- [ ] Neovim plugin with all commands
- [ ] Release v0.1.0-alpha

### Phase 2: AI Generation (4-6 weeks)
- [ ] Integrate chosen inference backend
- [ ] GGUF conversion pipeline (if GGML)
- [ ] Quantization testing
- [ ] Release v0.2.0-beta

### Phase 3: Polish (2 weeks)
- [ ] Telescope extension
- [ ] Performance optimization
- [ ] Documentation
- [ ] Release v1.0.0

---

## File Structure (Final)

```
lofi.nvim/
├── lua/
│   └── lofi/
│       ├── init.lua
│       ├── config.lua
│       ├── daemon.lua
│       ├── commands.lua
│       ├── state.lua
│       ├── ui.lua
│       ├── health.lua
│       └── telescope.lua     -- lazy-loaded telescope extension
├── plugin/
│   └── lofi.lua              -- autocmds, command registration
├── bin/
│   └── .gitkeep              -- daemon binary installed here
├── doc/
│   └── lofi.txt              -- vimdoc
├── scripts/
│   └── convert_musicgen_to_gguf.py  -- model conversion (if GGML path)
├── install.sh
├── README.md
├── LICENSE
└── .github/
    └── workflows/
        ├── release.yml       -- build daemon for all platforms
        └── convert-model.yml -- GGUF conversion pipeline
```

---

## Additional Clarifications (v3.1)

### Config Hot Reload

**Behavior:** Calling `lofi.setup()` again (e.g., re-sourcing init.lua) applies changes as follows:

| Setting Category | Hot Reload? | Notes |
|------------------|-------------|-------|
| `keymaps`, `ui` | ✓ Yes | Lua-side, immediate effect |
| `defaults`, `prefetch`, `cache` | ✓ Yes | Sent to daemon on next command |
| `daemon.*`, `model.*`, `audio.device` | ✗ No | Requires `:Lofi daemon restart` |

Daemon-side settings are sent at spawn time. To change them without restart, we'd need a `reconfigure` RPC method—possible future enhancement.

### Multiple Neovim Instances

Each Neovim instance spawns its own daemon. This is intentional.

| Resource | Shared? | Behavior |
|----------|---------|----------|
| Disk cache | ✓ Yes | Both instances read/write same cache dir |
| Audio device | ✓ Yes | Both daemons can play simultaneously (audio overlaps) |
| Generation queue | ✗ No | Each daemon has independent queue |
| PID file | ✗ No | Each writes `lofi-daemon-{nvim_pid}.pid` |

If overlapping audio is undesirable, user should manually `:Lofi stop` in one instance. Future enhancement: optional single-daemon mode via Unix socket.

### `model_version` in Track ID

The `model_version` component is a string combining:

```
model_version = "{model_name}-{quantization}-{schema_version}"
# Example: "musicgen-small-Q4_K_M-v1"
```

- `model_name`: e.g., "musicgen-small"
- `quantization`: e.g., "Q4_K_M"
- `schema_version`: bumped when inference code changes in ways that affect output (e.g., "v1", "v2")

This ensures cache invalidation when:
- User switches quantization levels
- Daemon update changes generation behavior

The `schema_version` is hardcoded in the daemon binary, not derived from the GGUF file.

### Cancel Active Generation

Attempting to cancel an in-progress generation returns an error:

```jsonc
// Request
{ "method": "queue_cancel", "params": { "track_id": "a1b2c3d4" }, "id": 1 }

// Response (if track is currently generating)
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32001,
    "message": "Cannot cancel active generation",
    "data": { "track_id": "a1b2c3d4", "reason": "GENERATION_IN_PROGRESS" }
  },
  "id": 1
}

// Response (if track is queued — success)
{
  "jsonrpc": "2.0",
  "result": { "cancelled": true, "track_id": "a1b2c3d4" },
  "id": 1
}
```

Rationale: Interrupting autoregressive generation mid-stream risks corrupted audio and complicates state management. User can `:Lofi daemon restart` if they really need to abort.

### Prefetch Trigger Timing

Currently hardcoded at 50% playback position. Rationale:

- 30s track at 50% = 15s remaining
- Expected generation time on modern CPU = 60-90s for 30s audio
- This means prefetch usually won't complete before current track ends

This is acceptable because:
1. Loop mode replays current track while waiting
2. User likely has multiple tracks cached after first session

**Future enhancement:** Add `prefetch.trigger_percent` config option. On slower systems, user could set to 25% or even trigger on track start.

---

## Changelog from v2

| Issue | Resolution |
|-------|------------|
| musicgen.cpp maturity | Added Risk Assessment section with go/no-go checkpoint |
| GGUF conversion tooling | Acknowledged as custom work; added to Phase 0 |
| Prefetch prompt strategy | Added `prefetch.strategy` config with 3 modes |
| MP3 fallback behavior | Specified: warn at startup, silently fall back per-track |
| Daemon crash recovery | Specified: queue lost, cache preserved, settings restored |
| Token estimation accuracy | Renamed to `tokens_estimated`, capped percent at 99 |
| Telescope extension | Added `lua/lofi/telescope.lua` back to file structure |
| Sample rate resampling | Specified: daemon resamples 32kHz → device rate via rubato |
| Orphan cleanup | Added `idle_timeout` and stdin EOF detection |
| ui.progress = "native" | Renamed to "cmdline", documented all options |