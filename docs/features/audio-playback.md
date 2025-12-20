/speckit.specify Audio playback controls for lofi-daemon

## Context

This feature implements playback control in `lua/lofi/init.lua` (public API) and the corresponding JSON-RPC methods handled by the Rust daemon. Audio playback runs entirely in the daemon using rodio/cpal.

## Constitution Alignment

- Principle I (Zero-Latency): playback commands return immediately; audio handled by daemon
- Principle III (Async-First): all playback managed in separate daemon process
- Principle V (Composability): full Lua API for scripting playback workflows

## Audio Pipeline

Reference design.md "Audio Pipeline":
```
MusicGen (32kHz mono) → Resampler (rubato) → Audio Device (44.1/48kHz stereo)
```

- MusicGen outputs 32kHz mono
- Daemon resamples to device's preferred rate (typically 44.1kHz or 48kHz)
- Mono duplicated to stereo for playback (L=R)
- Resampling happens during playback, not at generation time

## Requirements

### Playback controls
- play(track_id?) - play specific track, or resume, or play most recent
- pause() - pause current playback
- resume() - resume paused playback
- toggle() - pause if playing, resume if paused
- stop() - stop playback, clear playlist position
- skip() - next track in playlist (with crossfade)

### Volume control
- volume(level?) - get current (nil) or set (0.0-1.0)
- Default: config.playback.volume (0.3)
- Keyboard shortcuts: volume_up (+10%), volume_down (-10%)

### Audio device selection
- List available devices via `audio_devices` RPC
- Set device via `audio_device_set` RPC
- Default: system default device
- Config: config.audio.device (nil = default, string = device name)

### Sample rate handling
Per design.md:
- Daemon queries device's preferred sample rate on startup
- Uses rubato crate for high-quality resampling from 32kHz
- Cached files remain 32kHz (resampling is per-playback)

### Stereo handling
- MusicGen outputs mono
- Daemon duplicates to stereo (L=R) for playback
- Future: optional pseudo-stereo enhancement (not MVP)

## JSON-RPC methods

From design.md "JSON-RPC Interface" - PLAYBACK section:
```json
{"method": "play", "params": {"track_id": "a1b2c3d4"}, "id": 1}
{"method": "pause", "id": 2}
{"method": "resume", "id": 3}
{"method": "stop", "id": 4}
{"method": "skip", "id": 5}
{"method": "volume_set", "params": {"level": 0.5}, "id": 6}
{"method": "volume_get", "id": 7}
{"method": "audio_devices", "id": 8}
{"method": "audio_device_set", "params": {"device": "Built-in Output"}, "id": 9}
```

### Notifications (daemon → Lua)
```json
{"method": "playback_started", "params": {"track_id": "..."}}
{"method": "playback_paused", "params": {"track_id": "..."}}
{"method": "playback_ended", "params": {"track_id": "...", "reason": "finished|skipped|stopped|error"}}
```

## Dependencies
- daemon-lifecycle.md (daemon.request() for RPC)
- plugin-setup.md (config access)

## Error codes
- AUDIO_DEVICE_ERROR: failed to open audio device
- AUDIO_PLAYBACK_ERROR: error during playback (include details)
- INVALID_TRACK_ID: track_id not found in cache

## Lua API
```lua
local lofi = require("lofi")
lofi.play(track_id?)     -- play/resume
lofi.pause()
lofi.resume()
lofi.toggle()
lofi.stop()
lofi.skip()
lofi.volume(level?)      -- get or set (0.0-1.0)
lofi.is_playing()        -- returns boolean
```

## Events
```lua
lofi.on("playback_start", function(data) end)   -- {track_id}
lofi.on("playback_pause", function(data) end)   -- {track_id}
lofi.on("playback_resume", function(data) end)  -- {track_id}
lofi.on("playback_end", function(data) end)     -- {track_id, reason}
```

## Success criteria
- play() starts playback within 100ms of track being cached
- Volume changes apply within 50ms
- Crossfade transitions smooth (no audible clicks)
- Device switch works without stopping playback
- Mono-to-stereo sounds centered (equal L/R)
