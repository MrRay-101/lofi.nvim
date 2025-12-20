/speckit.specify Statusline component for lualine/heirline integration

## Context

This feature implements a statusline component in `lua/lofi/ui.lua` that displays current playback state, progress, and generation status. Designed to integrate with popular statusline plugins.

## Constitution Alignment

- Principle I (Zero-Latency): statusline function MUST return immediately (no blocking calls)
- Principle IV (Minimal Footprint): pure Lua, works with any statusline plugin
- Principle V (Composability): function returns string, user controls placement

## Requirements

### Statusline function
```lua
require("lofi.ui").statusline()
-- Returns: string or nil
```

- Returns nil when nothing to display (stopped, no generation)
- Returns formatted string with current state
- MUST NOT make RPC calls; reads from cached Lua state

### Display states

| State | Icon | Example output |
|-------|------|----------------|
| Playing | ▶ | `▶ 0:15/0:30` |
| Paused | ⏸ | `⏸ 0:15/0:30` |
| Generating | ◌ | `◌ 45%` |
| Playing + Generating | ▶ + ◌ | `▶ 0:15 ◌ 45%` |
| Stopped | (nil) | (returns nil, not shown) |

### Position format
- Playback position: `M:SS/M:SS` (current/total)
- Generation: `NN%` (percentage)
- Combined: show both when playing and generating simultaneously

### Configurable icons
Reference design.md config.ui.icons:
```lua
icons = {
  playing = "▶",
  paused = "⏸",
  generating = "◌",
  music = "♪",
}
```

### Conditional display
- config.ui.statusline = true (default) enables component
- If false, statusline() always returns nil
- User can wrap in their own condition logic

## Integration examples

### lualine
```lua
require("lualine").setup({
  sections = {
    lualine_x = {
      { require("lofi.ui").statusline, cond = function() return require("lofi").is_playing() or require("lofi").is_generating() end },
    },
  },
})
```

### heirline
```lua
local Lofi = {
  provider = function()
    return require("lofi.ui").statusline() or ""
  end,
  condition = function()
    local lofi = require("lofi")
    return lofi.is_playing() or lofi.is_generating()
  end,
}
```

### DIY statusline
```lua
vim.o.statusline = "%{%v:lua.require('lofi.ui').statusline() or ''%} ..."
```

## State caching

To ensure zero-latency returns:
1. Subscribe to daemon notifications in ui.lua
2. Update module-level state variables on each notification
3. statusline() reads from cached state, never makes RPC

```lua
-- Internal state (updated by notification handlers)
local state = {
  playback = nil,  -- {state, track_id, position_sec, duration_sec}
  generation = nil, -- {track_id, percent}
}
```

## File structure
```
lua/lofi/
├── ui.lua            -- statusline() function, state caching
```

## Dependencies
- daemon-lifecycle.md (subscribe to notifications)
- plugin-setup.md (config.ui.statusline, config.ui.icons)

## Lua API
```lua
require("lofi.ui").statusline()  -- returns string or nil
require("lofi").is_playing()     -- for condition checks
require("lofi").is_generating()  -- for condition checks
```

## Success criteria
- statusline() returns in <1ms (no RPC, pure cache read)
- Updates reflect state changes within 100ms of notification
- Returns nil when stopped (no visual clutter)
- Icons configurable and render correctly in terminal
- Works with lualine, heirline, and DIY statuslines
- No errors when lofi not active
