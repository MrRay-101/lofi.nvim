# lofi.nvim

AI music generation for Neovim using MusicGen ONNX. Generate custom lofi beats, ambient music, and more from text prompts - all running locally on your machine.

## Features

- **Text-to-music generation** - Describe the music you want and get a WAV file
- **Progress tracking** - Floating window shows generation progress
- **Auto-play** - Generated tracks play automatically on completion
- **Job queue** - Queue up to 10 generation requests with priority support
- **Track cache** - Identical prompts return cached results instantly
- **Hardware acceleration** - Supports CPU, CUDA (Linux), and Metal (macOS)
- **Fully local** - No API keys, no network required after model download

## Requirements

- Neovim 0.9+
- Rust 1.75+ (for building the daemon)
- ~500MB disk space for models
- ~4GB RAM during generation
- macOS, Linux, or Windows

## Installation

### lazy.nvim

```lua
{
  "username/lofi.nvim",
  build = "cd daemon && cargo build --release",
  config = function()
    require("lofi").setup()
  end,
}
```

### Manual

```bash
git clone https://github.com/username/lofi.nvim ~/.local/share/nvim/lofi.nvim
cd ~/.local/share/nvim/lofi.nvim/daemon
cargo build --release
```

Add to your config:

```lua
vim.opt.runtimepath:append("~/.local/share/nvim/lofi.nvim")
require("lofi").setup()
```

## Usage

### Commands

```vim
" Generate music from a prompt (default 10 seconds)
:Lofi lofi hip hop jazzy piano

" Generate with specific duration (in seconds)
:Lofi ambient synthwave 30

" Play the last generated track
:LofiPlay

" Stop playback
:LofiStop
```

### Lua API

```lua
local lofi = require("lofi")

-- Generate with callback
lofi.generate({
  prompt = "lofi hip hop, jazzy piano, relaxing vibes",
  duration_sec = 30,  -- 5-120 seconds
  seed = 12345,       -- optional, for reproducibility
  priority = "high",  -- "normal" or "high"
}, function(err, result)
  if err then
    print("Error: " .. err.message)
  else
    print("Generated: " .. result.path)
  end
end)

-- Check status
lofi.is_generating()  -- true if generation in progress
lofi.current_track()  -- track_id of current generation

-- Event listeners
lofi.on("generation_progress", function(data)
  print(data.percent .. "% complete")
end)

lofi.on("generation_complete", function(data)
  print("Done: " .. data.path)
end)

-- Stop daemon
lofi.stop()
```

## Configuration

```lua
require("lofi").setup({
  -- Path to daemon binary (auto-detected if nil)
  daemon_path = nil,

  -- Path to model files (defaults to ~/.cache/lofi/models)
  model_path = nil,

  -- Device selection: "auto", "cpu", "cuda", "metal"
  device = "auto",

  -- CPU thread count (nil = auto)
  threads = nil,
})
```

### Environment Variables

```bash
LOFI_MODEL_PATH=/path/to/models   # Custom model directory
LOFI_CACHE_PATH=/path/to/cache    # Custom cache directory
LOFI_DEVICE=cpu                    # Force CPU mode
LOFI_THREADS=4                     # Limit CPU threads
```

## Events

Subscribe to generation events:

| Event | Data |
|-------|------|
| `generation_start` | `track_id`, `prompt`, `duration_sec`, `seed` |
| `generation_progress` | `track_id`, `percent`, `tokens_generated`, `eta_sec` |
| `generation_complete` | `track_id`, `path`, `duration_sec`, `generation_time_sec` |
| `generation_error` | `track_id`, `code`, `message` |

## CLI Mode

The daemon also works as a standalone CLI for testing:

```bash
cd daemon
cargo run --release -- --prompt "lofi beats" --duration 10 --output test.wav
```

## Models

On first run, models are automatically downloaded from HuggingFace (~250MB). Files are cached in `~/.cache/lofi/models/`.

Models used:
- MusicGen-small (fp16) from [gabotechs/music_gen](https://huggingface.co/gabotechs/music_gen)

## Performance

Typical generation times on CPU:

| Duration | Time |
|----------|------|
| 10 sec   | ~25s |
| 30 sec   | ~75s |
| 60 sec   | ~150s |

GPU acceleration can provide 2-4x speedup.

## Troubleshooting

**Models not found**: Run `:Lofi test` once with internet access to download models.

**Out of memory**: Try shorter durations or set `LOFI_DEVICE=cpu`.

**No audio in one ear**: Fixed in latest version - audio is now stereo.

**Generation stuck**: Restart Neovim. If persistent, delete cache and re-download models.

## License

MIT
