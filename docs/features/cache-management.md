/speckit.specify Track cache management with LRU eviction

## Context

This feature implements persistent storage of generated tracks on disk. Both AI and procedural generation output WAV/MP3 files to the cache directory. Cache is managed by the daemon but queryable from Lua.

## Constitution Alignment

- Principle II (Local & Private): cache on local filesystem only
- Principle III (Async-First): cache I/O handled by daemon

## Cache Architecture

Reference design.md "Audio Format" and "Cache Management":

### Storage location
- Default: vim.fn.stdpath("cache") .. "/lofi" (e.g., ~/.cache/nvim/lofi)
- Configurable: config.cache.dir

### File format
| Property | Value |
|----------|-------|
| Native Sample Rate | 32000 Hz |
| Channels | Mono |
| Bit Depth | 16-bit PCM |
| Format | WAV (default) or MP3 |
| WAV size | ~1.9MB per 30s |
| MP3 size | ~300KB per 30s (128kbps) |

### MP3 encoding
- Requires `lame` binary installed
- config.audio.format = "mp3" enables MP3 encoding
- If lame missing: warn at startup (:checkhealth), fall back to WAV silently
- Check for lame on daemon startup, cache result

## Requirements

### Track ID generation

Reference design.md "Track ID Generation":
```
track_id = sha256(prompt + seed + duration_sec + model_version)[:8]
```

- Deterministic: same inputs = same track_id = cache hit
- 8 hex chars (4 billion unique IDs)
- model_version = "{model_name}-{quantization}-{schema_version}"
- Schema_version bumped when inference code changes output

### Metadata storage

Each track has associated metadata:
- track_id (8 char hex)
- prompt (text used for generation)
- seed (integer)
- duration_sec (actual duration)
- created_at (ISO timestamp)
- format ("wav" or "mp3")
- generation_time_sec (how long generation took)
- model_version (for cache invalidation)

Storage options:
1. Sidecar JSON files: {track_id}.json alongside {track_id}.wav
2. SQLite database in cache dir
3. Single manifest.json with all metadata

Recommendation: sidecar JSON for simplicity and parallel access.

### LRU eviction

- config.cache.max_mb (default: 500)
- config.cache.max_tracks (default: nil = no limit)
- When limit exceeded, delete oldest accessed tracks first
- Update access time on playback (not just creation)
- Eviction runs after each generation completes

### Cache operations

- cache_list() - return all tracks with metadata
- cache_delete(track_id) - delete specific track
- cache_clear() - delete all tracks
- cache_stats() - size, count, oldest, newest

### Concurrent access

Per design.md "Multiple Neovim Instances":
- Multiple Neovim instances share same cache directory
- Both can read/write simultaneously
- Use file locking or atomic operations for writes
- No coordination needed for reads (files are immutable once written)

## JSON-RPC methods

From design.md:
```json
{"method": "cache_list", "id": 1}
// Response:
{"result": [
  {"track_id": "a1b2c3d4", "prompt": "...", "duration_sec": 30, "path": "...", "created_at": "..."}
]}

{"method": "cache_delete", "params": {"track_id": "a1b2c3d4"}, "id": 2}
{"method": "cache_clear", "id": 3}
{"method": "cache_stats", "id": 4}
// Response:
{"result": {"size_mb": 234, "count": 78, "oldest": "2025-01-01T...", "newest": "2025-01-15T..."}}
```

## Dependencies
- daemon-lifecycle.md (daemon handles cache I/O)
- plugin-setup.md (config.cache settings)

## Error codes
- CACHE_WRITE_ERROR: failed to write track to disk
- INVALID_TRACK_ID: track_id not found in cache
- LAME_NOT_FOUND: MP3 requested but lame missing (warning, not error)

## Lua API
```lua
local lofi = require("lofi")
lofi.cache()              -- returns track[] with metadata
lofi.cache_delete(track_id)
lofi.cache_clear()
```

## User commands
```vim
:Lofi list                " list cached tracks (Telescope picker if available)
:Lofi cache clear         " clear all cached tracks
```

## Success criteria
- Cache deduplication works (regenerating same prompt+seed hits cache)
- LRU eviction correctly removes oldest-accessed tracks
- Multiple Neovim instances can read cache concurrently
- MP3 encoding reduces cache size by ~85% when lame available
- Cache survives daemon restart (persistence verified)
- Metadata accurately reflects track properties
