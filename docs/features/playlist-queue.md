/speckit.specify Playlist and generation queue management

## Context

This feature implements two related but distinct queues:
1. **Playlist**: ordered list of tracks for playback (in daemon)
2. **Generation queue**: pending generation requests (in daemon)

Both are managed via JSON-RPC and exposed through the Lua API.

## Constitution Alignment

- Principle III (Async-First): queues managed in daemon, Lua layer is thin client
- Principle V (Composability): full API for playlist scripting

## Requirements

### Playlist management

Playlist is an ordered list of track_ids for sequential playback:

- playlist_add(track_id) - add track to end of playlist
- playlist_remove(track_id) - remove track from playlist
- playlist_clear() - clear entire playlist
- playlist_get() - return ordered list of track_ids
- Current position tracked; skip() advances to next

### Playback behavior

- Loop mode (config.playback.loop): restart playlist when finished
- Auto-play (config.playback.auto_play): play immediately when generation completes
- Skip: advance to next track with crossfade

### Crossfade

Reference design.md "Crossfade Behavior":
```
Track A: ████████████████████████░░░░  (fade out last N seconds)
Track B:                     ░░░░████████████████████████  (fade in first N seconds)
         ←── crossfade_sec ──→
```

- config.playback.crossfade_sec (default: 2.0, 0 to disable)
- Edge cases:
  - If track.duration < crossfade_sec * 2: reduce to track.duration / 4
  - If track.duration < 2 seconds: no crossfade
- Crossfade applies to playlist transitions and skip()
- stop() is immediate (no crossfade)

### Generation queue

Reference design.md "Concurrent Generation":
- Serial generation (not parallel) to avoid resource contention
- New requests added to queue
- Response includes position in queue

Queue operations:
- queue_status() - get current generation and pending list
- queue_cancel(track_id) - cancel pending (NOT active) generation
- queue_clear() - clear pending queue (keeps active generation)

### Cancel behavior

Per design.md "Cancel Active Generation":
- Cancelling in-progress generation returns error
- Only queued (pending) generations can be cancelled
- Error: code -32001, reason: "GENERATION_IN_PROGRESS"

## JSON-RPC methods

From design.md "JSON-RPC Interface":
```json
// Playlist
{"method": "playlist_add", "params": {"track_id": "a1b2c3d4"}, "id": 1}
{"method": "playlist_remove", "params": {"track_id": "a1b2c3d4"}, "id": 2}
{"method": "playlist_clear", "id": 3}
{"method": "playlist_get", "id": 4}

// Queue
{"method": "queue_status", "id": 5}
// Response:
{"result": {
  "current": {"track_id": "a1b2c3d4", "percent": 45},
  "pending": [
    {"track_id": "e5f6g7h8", "prompt": "...", "position": 1}
  ],
  "prefetch_pending": true
}}

{"method": "queue_cancel", "params": {"track_id": "e5f6g7h8"}, "id": 6}
{"method": "queue_clear", "id": 7}

// Crossfade config
{"method": "crossfade_set", "params": {"duration_sec": 2.0}, "id": 8}
```

## Dependencies
- daemon-lifecycle.md (daemon communication)
- audio-playback.md (play, skip use playlist)
- cache-management.md (track_id must exist in cache for playlist)

## Error codes
- INVALID_TRACK_ID: track_id not in cache (for playlist operations)
- GENERATION_IN_PROGRESS: cannot cancel active generation

## Lua API
```lua
local lofi = require("lofi")

-- Playlist
lofi.playlist_add(track_id)
lofi.playlist_remove(track_id)
lofi.playlist_clear()
lofi.playlist()           -- returns string[] of track_ids

-- Queue
lofi.queue()              -- returns {current, pending}
lofi.cancel(track_id)     -- cancel queued generation
```

## User commands
```vim
:Lofi queue               " show generation queue status
:Lofi cancel [track_id]   " cancel queued generation
```

## Success criteria
- Playlist maintains order across add/remove operations
- Crossfade transition is smooth (no audible gap or overlap artifact)
- Loop mode seamlessly restarts playlist
- Queue position accurately reflects actual order
- Cancel of pending generation removes from queue immediately
- Cancel of active generation returns clear error message
