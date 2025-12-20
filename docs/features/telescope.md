/speckit.specify Telescope extension for track browsing and management

## Context

This feature implements `lua/lofi/telescope.lua` as an optional Telescope extension for browsing cached tracks, viewing prompt history, and managing the track library with rich previews.

## Constitution Alignment

- Principle IV (Minimal Footprint): lazy-loaded, gracefully absent if Telescope not installed
- Principle V (Composability): standard Telescope extension pattern, actions exposed

## Architecture

Reference design.md "Telescope Extension":

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
```

## Requirements

### Extension registration
- Only register if Telescope is installed
- Lazy-load telescope.lua when extension accessed
- Located at lua/lofi/telescope.lua

### Pickers

#### :Telescope lofi tracks
Browse all cached tracks with metadata:

**Columns:**
- Track ID (8 char)
- Duration (MM:SS)
- Prompt (truncated)
- Created date

**Preview:**
- Track metadata (full prompt, seed, generation time)
- Optional: ASCII waveform visualization (requires sox)

**Actions:**
| Key | Action |
|-----|--------|
| `<CR>` | Play track |
| `<C-a>` | Add to playlist |
| `<C-d>` | Delete track from cache |
| `<C-y>` | Copy track_id to clipboard |

#### :Telescope lofi prompts
Browse prompt history for regeneration:

**Columns:**
- Prompt text
- Use count
- Last used date

**Actions:**
| Key | Action |
|-----|--------|
| `<CR>` | Generate new track with this prompt (new seed) |
| `<C-s>` | Generate with same seed (reproduce exact track) |
| `<C-e>` | Edit prompt before generating |

### Waveform preview
- Requires `sox` binary for waveform extraction
- If sox missing: show text-only preview, no error
- Cache waveform data to avoid regenerating on each preview
- ASCII art representation (40 chars wide, fits preview window)

### Data source
- tracks: call lofi.cache() to get track list
- prompts: extract unique prompts from cache, count occurrences

## File structure
```
lua/lofi/
├── telescope.lua     -- lazy-loaded Telescope extension
```

## Dependencies
- cache-management.md (lofi.cache() for track data)
- audio-playback.md (lofi.play() for play action)
- ai-generation.md (lofi.generate() for regeneration)

## Optional dependencies
- telescope.nvim (required for this feature)
- sox (optional for waveform preview)

## Lua API
```lua
-- Direct usage (if not using :Telescope command)
require("lofi.telescope").tracks()
require("lofi.telescope").prompts()
```

## User commands
```vim
:Telescope lofi tracks   " browse cached tracks
:Telescope lofi prompts  " browse prompt history
:Lofi list               " alias for :Telescope lofi tracks (if available)
```

## Success criteria
- Extension registers without error when Telescope present
- Extension silently unavailable when Telescope absent
- Track picker shows all cached tracks with accurate metadata
- Play action starts playback within 100ms
- Delete action removes track and refreshes picker
- Waveform preview renders when sox available
- Graceful fallback to text preview when sox missing
- Prompt history correctly counts and sorts by usage
