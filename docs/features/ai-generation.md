/speckit.specify AI music generation via MusicGen GGML backend

## Context

This feature implements AI-powered music generation using MusicGen-small with GGML inference. This is a high-risk feature requiring a Phase 0 feasibility checkpoint before full implementation.

## Constitution Alignment

- Principle II (Local & Private): all inference runs locally, no network, no API keys
- Principle III (Async-First): generation runs in background, never blocks editor
- Principle IV (Minimal Footprint): GGML backend, single binary, model weights separate

## Risk Assessment

Reference design.md "Risk Assessment" section:

**Critical Path Challenges:**
- No production-ready GGML port of MusicGen exists
- No standard convert.py for MusicGen → GGUF (unlike llama.cpp)
- PABannier/encodec.cpp exists but is experimental

**Phase 0 Go/No-Go Checkpoint:**
Before building Neovim integration, prove standalone Rust CLI can:
1. Load MusicGen weights (any format)
2. Generate 10s of audio from text prompt
3. Run on CPU in <2 minutes

If not achievable, fallback to procedural-generation.md only.

## Model Strategy

Reference design.md "Model Strategy":

### GGUF Quantization Options
| Variant | Size | Quality | Speed | Use case |
|---------|------|---------|-------|----------|
| Q4_K_M | ~200MB | Good | Fast | Default |
| Q5_K_M | ~250MB | Better | Medium | Quality-focused |
| Q8_0 | ~400MB | Best | Slow | Audiophiles |
| F16 | ~600MB | Reference | Slowest | Development |

### Performance Expectations
| Hardware | Model | 30s Audio | Notes |
|----------|-------|-----------|-------|
| Modern CPU (AVX2) | Q4_K_M | 60-90s | Acceptable |
| Apple Silicon (M1+) | Q4_K_M | 30-45s | Metal acceleration |
| CUDA GPU (RTX 3060+) | Q4_K_M | 10-20s | Optimal |
| Older CPU (no AVX2) | Q4_K_M | 3-5min | Use procedural fallback |

## Requirements

### Generation request
- Accept prompt (text), duration_sec (default 30), seed (optional), priority
- Return track_id immediately (generation is async)
- Queue multiple requests (serial generation, not parallel)
- Support priority: "high" skips queue

### Background inference
- Daemon loads model on first generation request
- Model stays loaded until daemon exits
- Generation cannot be cancelled mid-flight (too complex)

### Progress reporting
- Daemon sends `generation_progress` notifications
- Fields: track_id, percent (0-99), tokens_generated, tokens_estimated, eta_sec
- Percent capped at 99 until `generation_complete` fires
- See progress-notifications.md for UI handling

### Seed reproducibility
- Same prompt + seed + duration + model_version = identical output
- If seed is null, daemon generates random seed, includes in response
- Track ID is hash of these inputs (enables cache deduplication)

### Device selection
- config.model.device: "auto" | "cpu" | "cuda" | "metal"
- "auto": prefer GPU if available, fall back to CPU
- config.model.threads: CPU thread count (nil = auto-detect)

## JSON-RPC methods

From design.md "JSON-RPC Interface" - GENERATION section:
```json
// Request
{"method": "generate", "params": {
  "prompt": "lofi hip hop, jazzy piano",
  "duration_sec": 30,
  "seed": 42,
  "priority": "normal"
}, "id": 1}

// Response (immediate)
{"result": {"track_id": "a1b2c3d4", "status": "queued", "position": 0}}

// Notification: progress
{"method": "generation_progress", "params": {
  "track_id": "a1b2c3d4",
  "percent": 45,
  "tokens_generated": 450,
  "tokens_estimated": 1000,
  "eta_sec": 35
}}

// Notification: complete
{"method": "generation_complete", "params": {
  "track_id": "a1b2c3d4",
  "path": "/home/user/.cache/nvim/lofi/a1b2c3d4.wav",
  "duration_sec": 30.2,
  "sample_rate": 32000,
  "prompt": "lofi hip hop, jazzy piano",
  "seed": 42,
  "generation_time_sec": 72.5
}}
```

## Dependencies
- daemon-lifecycle.md (daemon must be running)
- cache-management.md (track storage)
- progress-notifications.md (UI updates)

## Error codes
- MODEL_NOT_FOUND: weights file not at expected path
- MODEL_LOAD_FAILED: corrupt file, wrong format, OOM
- MODEL_INFERENCE_FAILED: numerical instability, OOM during generation

## Lua API
```lua
local lofi = require("lofi")
lofi.generate({prompt = "...", duration_sec = 30, seed = nil}, callback)
-- callback(err, track) where track = {track_id, path, duration_sec, ...}
lofi.is_generating()  -- returns boolean
```

## Events
```lua
lofi.on("generation_start", function(data) end)     -- {track_id}
lofi.on("generation_progress", function(data) end)  -- {track_id, percent, eta_sec}
lofi.on("generation_complete", function(data) end)  -- {track_id, path, ...}
lofi.on("generation_error", function(data) end)     -- {track_id, error, code}
```

## Success criteria
- Phase 0: 10s audio generated on CPU in <2 minutes
- Model loads in <10s on modern hardware
- Progress updates accurate within 10% of actual completion
- Same seed produces identical output
- GPU acceleration provides >2x speedup over CPU
