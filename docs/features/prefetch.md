/speckit.specify Automatic prefetch system to mask generation latency

## Context

This feature implements automatic generation of the next track while the current track is playing, to mask the 60-90s AI generation time and provide seamless continuous playback.

## Constitution Alignment

- Principle I (Zero-Latency): mask generation time by prefetching ahead
- Principle III (Async-First): prefetch runs in background

## Prefetch Behavior

Reference design.md "Prefetch" and "Prefetch Trigger Timing":

### Trigger conditions
Prefetch fires when ALL conditions are met:
1. Current track playback reaches 50% position
2. No track is currently queued for generation
3. config.prefetch.enabled = true

### Timing rationale
- 30s track at 50% = 15s remaining
- AI generation takes 60-90s on modern CPU
- Prefetch usually won't complete before current track ends
- Acceptable because:
  - Loop mode replays current track while waiting
  - After first session, user has multiple cached tracks

### Future enhancement
Consider config.prefetch.trigger_percent for slower systems (e.g., 25% or on track start).

## Requirements

### Prefetch configuration
```lua
prefetch = {
  enabled = true,                   -- master toggle
  strategy = "same_prompt",         -- prompt selection strategy
  presets = {                       -- used by preset strategies
    "lofi hip hop, rainy day, mellow piano",
    "chill beats, coffee shop, jazz chords",
    "lofi, late night coding, ambient synth",
  },
}
```

### Prompt strategies

Reference design.md "PREFETCH" JSON-RPC section:

| Strategy | Behavior |
|----------|----------|
| `same_prompt` | Reuse current track's prompt with new random seed |
| `preset_cycle` | Rotate through presets array in order |
| `random_preset` | Randomly select from presets array |

### Queue integration
- Prefetch adds to generation queue with normal priority
- User-initiated generate() with priority="high" skips prefetch
- queue_status response includes `prefetch_pending` flag
- Prefetch can be cancelled like any queued generation

### Playback integration
- When prefetched track completes, auto-add to playlist if config.playback.auto_play
- Crossfade into prefetched track when current track ends

## JSON-RPC methods

From design.md:
```json
// Configure prefetch
{"method": "prefetch_config", "params": {
  "enabled": true,
  "strategy": "same_prompt",
  "presets": ["...", "...", "..."]
}, "id": 1}

// Queue status shows prefetch
{"method": "queue_status", "id": 2}
// Response:
{"result": {
  "current": {...},
  "pending": [...],
  "prefetch_pending": true
}}
```

## Dependencies
- daemon-lifecycle.md (daemon handles prefetch logic)
- ai-generation.md / procedural-generation.md (generation backend)
- playlist-queue.md (queue management)
- audio-playback.md (playback position monitoring)

## Implementation notes

Prefetch logic lives in daemon, not Lua:
1. Daemon monitors playback position
2. At 50% position, daemon checks prefetch conditions
3. If conditions met, daemon internally queues generation
4. Lua layer receives same generation_progress/complete notifications

This keeps Lua layer thin and avoids polling overhead.

## Lua API
```lua
-- Prefetch is automatic; no explicit API needed
-- Configure via setup():
require("lofi").setup({
  prefetch = {
    enabled = true,
    strategy = "preset_cycle",
    presets = {...}
  }
})
```

## Success criteria
- Prefetch triggers at correct playback position (50%)
- Prefetch does not trigger if generation already queued
- Prefetch can be disabled without side effects
- Strategy correctly selects next prompt
- preset_cycle maintains order across tracks
- random_preset provides variety (no immediate repeats)
- Prefetched tracks seamlessly added to playlist
