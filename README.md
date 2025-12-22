# lofi.nvim

AI music generation for Neovim using MusicGen and ACE-Step ONNX models. Generate custom lofi beats, ambient music, and more from text prompts - all running locally on your machine.

## Features

- **Text-to-music generation** - Describe the music you want and get a WAV file
- **Two backends** - MusicGen (fast, up to 120s) and ACE-Step (high quality, up to 240s)
- **Long-form generation** - ACE-Step supports up to 4 minutes of audio
- **Progress tracking** - Floating window shows generation progress with ETA
- **Auto-play** - Generated tracks play automatically on completion
- **Job queue** - Queue up to 10 generation requests with priority support
- **Track cache** - Identical prompts return cached results instantly
- **Hardware acceleration** - Supports CPU, CUDA (Linux), and Metal (macOS)
- **Fully local** - No API keys, no network required after model download

## Requirements

- Neovim 0.9+
- Rust 1.75+ (for building the daemon)
- ~500MB disk space for MusicGen models
- ~8GB disk space for ACE-Step models (optional)
- ~4GB RAM during generation (MusicGen) or ~8GB (ACE-Step)
- macOS, Linux, or Windows

## Installation

### lazy.nvim

```lua
{
  "willibrandon/lofi.nvim",
  build = "cd daemon && cargo build --release",
  config = function()
    require("lofi").setup()
  end,
}
```

### Manual

```bash
git clone https://github.com/willibrandon/lofi.nvim ~/.local/share/nvim/lofi.nvim
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
" Generate music from a prompt (default 10 seconds, MusicGen)
:Lofi lofi hip hop jazzy piano

" Generate with specific duration (in seconds)
:Lofi ambient synthwave 30

" Generate with ACE-Step backend (higher quality, longer duration)
:LofiAce chill electronic beats 120

" Show available backends and their status
:LofiBackends

" Cancel in-progress generation
:LofiCancel

" Play the last generated track
:LofiPlay

" Stop playback
:LofiStop
```

### Lua API

```lua
local lofi = require("lofi")

-- Generate with MusicGen (default)
lofi.generate({
  prompt = "lofi hip hop, jazzy piano, relaxing vibes",
  duration_sec = 30,  -- 5-120 seconds for MusicGen
  seed = 12345,       -- optional, for reproducibility
  priority = "high",  -- "normal" or "high"
}, function(err, result)
  if err then
    print("Error: " .. err.message)
  else
    print("Generated: " .. result.path)
  end
end)

-- Generate with ACE-Step (long-form, higher quality)
lofi.generate({
  prompt = "ambient electronic, slow tempo, dreamy pads",
  duration_sec = 120,         -- 5-240 seconds for ACE-Step
  backend = "ace_step",
  inference_steps = 60,       -- 1-200, higher = better quality
  scheduler = "euler",        -- "euler", "heun", or "pingpong"
  guidance_scale = 7.0,       -- 1.0-30.0, higher = more prompt adherence
  seed = 42,
}, function(err, result)
  if err then
    print("Error: " .. err.message)
  else
    print("Generated: " .. result.path)
  end
end)

-- Check available backends
lofi.get_backends(function(err, result)
  for _, backend in ipairs(result.backends) do
    print(backend.name .. ": " .. backend.status)
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
  print("Done: " .. data.path .. " (" .. data.backend .. ")")
end)

-- Cancel generation
lofi.cancel()

-- Stop daemon
lofi.stop()
```

## Configuration

```lua
require("lofi").setup({
  -- Path to daemon binary (auto-detected if nil)
  daemon_path = nil,

  -- Path to model files (defaults to ~/.cache/lofi.nvim/musicgen)
  model_path = nil,

  -- Device selection: "auto", "cpu", "cuda", "metal"
  device = "auto",

  -- CPU thread count (nil = auto)
  threads = nil,

  -- Default backend: "musicgen" or "ace_step"
  backend = "musicgen",
})
```

### Environment Variables

```bash
LOFI_MODEL_PATH=/path/to/models         # MusicGen model directory
LOFI_ACE_STEP_MODEL_PATH=/path/to/ace   # ACE-Step model directory
LOFI_CACHE_PATH=/path/to/cache          # Generated track cache
LOFI_DEVICE=cpu                          # Force CPU mode
LOFI_THREADS=4                           # Limit CPU threads
LOFI_BACKEND=ace_step                    # Default backend

# ACE-Step specific
LOFI_ACE_STEP_STEPS=60                   # Default inference steps
LOFI_ACE_STEP_SCHEDULER=euler            # Default scheduler
LOFI_ACE_STEP_GUIDANCE=7.0               # Default guidance scale
```

## Events

Subscribe to generation events:

| Event | Data |
|-------|------|
| `generation_start` | `track_id`, `prompt`, `duration_sec`, `seed`, `backend` |
| `generation_progress` | `track_id`, `percent`, `eta_sec`, `current_step`, `total_steps` |
| `generation_complete` | `track_id`, `path`, `duration_sec`, `generation_time_sec`, `backend` |
| `generation_error` | `track_id`, `code`, `message` |
| `download_progress` | `file_name`, `bytes_downloaded`, `bytes_total`, `files_completed` |

## CLI Mode

The daemon also works as a standalone CLI for testing:

```bash
cd daemon

# MusicGen (default)
cargo run --release -- --prompt "lofi beats" --duration 10 --output test.wav

# ACE-Step
cargo run --release -- --backend ace-step --prompt "chill ambient" --duration 60 --output test.wav

# ACE-Step with custom parameters
cargo run --release -- --backend ace-step \
  --prompt "electronic lofi" \
  --duration 120 \
  --steps 80 \
  --scheduler heun \
  --guidance 10.0 \
  --seed 42 \
  --output test.wav
```

## Backends

### MusicGen (Default)

- **Duration**: 5-120 seconds
- **Sample rate**: 32kHz
- **Model size**: ~500MB
- **Speed**: Fast (~2.5s per second of audio on CPU)

### ACE-Step

- **Duration**: 5-240 seconds
- **Sample rate**: 48kHz
- **Model size**: ~8GB
- **Speed**: Slower (~15s per second of audio on CPU)
- **Quality**: Higher quality, better for long-form generation

ACE-Step models are downloaded automatically on first use, or manually via `:LofiBackends`.

## Models

On first run, MusicGen models are automatically downloaded from HuggingFace (~500MB). ACE-Step models (~8GB) are downloaded when first used.

Model sources:
- MusicGen-small (fp16) from [gabotechs/music_gen](https://huggingface.co/gabotechs/music_gen)
- ACE-Step from [willibrandon/lofi-models](https://huggingface.co/willibrandon/lofi-models)

Models are cached in:
- macOS: `~/Library/Caches/lofi.nvim/`
- Linux: `~/.cache/lofi.nvim/`
- Windows: `%LOCALAPPDATA%\lofi.nvim\cache\`

## Performance

### MusicGen (CPU)

| Duration | Time |
|----------|------|
| 10 sec   | ~25s |
| 30 sec   | ~75s |
| 60 sec   | ~150s |

### ACE-Step (CPU)

| Duration | Time |
|----------|------|
| 10 sec   | ~150s |
| 30 sec   | ~200s |
| 60 sec   | ~300s |
| 120 sec  | ~500s |

GPU acceleration provides 2-4x speedup for MusicGen and 5-10x for ACE-Step.

## Troubleshooting

**Models not found**: Run `:Lofi test` once with internet access to download models.

**ACE-Step not available**: Run `:LofiBackends` to check status. Models download automatically on first use.

**Out of memory**: Try shorter durations, reduce `inference_steps`, or set `LOFI_DEVICE=cpu`.

**No audio in one ear**: Fixed in latest version - audio is now stereo.

**Generation stuck**: Use `:LofiCancel` to stop, or restart Neovim.

**Numerical instability**: Try a different seed or reduce `guidance_scale`.

## License

MIT
