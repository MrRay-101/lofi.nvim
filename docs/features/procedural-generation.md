/speckit.specify Procedural lofi beat generation as AI fallback

## Context

This feature implements instant procedural music generation for systems that cannot run AI inference, or as a fallback while AI backend is in development. Ships with Phase 1 (Procedural MVP).

## Constitution Alignment

- Principle I (Zero-Latency): instant generation, no inference delay
- Principle II (Local & Private): bundled samples, no network
- Principle IV (Minimal Footprint): ~5MB of royalty-free samples

## Architecture

Reference design.md "Fallback: Procedural Generation":
```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ Drum loops  │ + │ Chord prog  │ + │ Vinyl noise │ = Lofi beat
│ (samples)   │   │ (generated) │   │ (samples)   │
└─────────────┘   └─────────────┘   └─────────────┘
```

## Requirements

### Sample library
- Ship ~5MB of royalty-free samples
- Categories: drum loops, chord stabs, ambient textures, vinyl noise
- Multiple variations per category for variety
- License: CC0 or equivalent (document in LICENSE)

### Randomized composition
- Tempo: 70-90 BPM (configurable range)
- Key: random selection from common lofi keys (Am, Em, Dm, Gm)
- Loop selection: random from available samples
- Layering: drums + chords + ambient (optional vinyl crackle)

### Lofi effects processing
- Bitcrusher: reduce bit depth for lo-fi texture
- Low-pass filter: roll off highs (cutoff ~8kHz)
- Tape wobble: subtle pitch modulation
- Vinyl crackle: layered ambient noise (adjustable level)

### Seed reproducibility
- Same seed produces identical composition
- Seed controls: tempo, key, sample selection, effect parameters
- Enables cache deduplication (same as AI generation)

### Seamless looping
- Output must loop seamlessly for extended playback
- Crossfade applied at loop points
- Duration flexible (default 30s, configurable)

## JSON-RPC methods

Uses same interface as AI generation with backend indicator:
```json
// When config.model.backend = "procedural"
{"method": "generate", "params": {
  "prompt": "ignored for procedural",
  "duration_sec": 30,
  "seed": 42
}, "id": 1}

// Response (immediate - no queuing needed)
{"result": {"track_id": "...", "status": "complete", "position": 0}}

// generation_complete fires almost immediately (<1s)
```

### Status response includes backend
```json
{"result": {"model": {"backend": "procedural", "quantization": null, "device": "cpu"}}}
```

## Dependencies
- daemon-lifecycle.md (daemon must be running)
- cache-management.md (track storage)

## File structure
```
lofi-daemon/
├── samples/
│   ├── drums/       -- drum loop WAVs
│   ├── chords/      -- chord progression WAVs
│   ├── ambient/     -- texture WAVs
│   └── vinyl/       -- crackle/noise WAVs
```

## Lua API

Same as ai-generation.md - backend is transparent to Lua layer:
```lua
local lofi = require("lofi")
lofi.generate({duration_sec = 30, seed = nil}, callback)
```

## Success criteria
- Generation completes in <1s (instant gratification)
- Output sounds recognizably "lofi" (subjective but testable via user feedback)
- Loops seamlessly with no audible seam
- Same seed produces identical output
- All samples properly licensed (CC0 or equivalent)
- Total sample bundle <5MB
