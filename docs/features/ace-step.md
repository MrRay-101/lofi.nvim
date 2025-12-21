# ACE-Step: Long-form music generation via flow matching

## Context

This feature implements AI-powered music generation using ACE-Step, a 3.5B parameter diffusion model capable of generating up to 240 seconds (4 minutes) of audio natively. ACE-Step explicitly supports instrumental music generation, making it suitable for lofi production.

**Reference Implementation**: [ACE-Step](https://github.com/ace-step/ACE-Step) - ACE Studio & StepFun. Local clone at `/Users/brandon/src/ACE-Step`.

**Why ACE-Step over alternatives:**
- MusicGen: 30-second limit, continuation produces gaps
- DiffRhythm 2: No instrumental support (explicitly TODO), complex KV cache architecture incompatible with ONNX

## Distribution

ACE-Step is an **optional generation backend** alongside MusicGen.

| Backend | Max Duration | Model Size | Use Case |
|---------|--------------|------------|----------|
| MusicGen | 30s | ~2GB | Quick generations, lower VRAM |
| ACE-Step | 240s | 7.7GB | Long-form tracks without gaps |

### Installation

```
:Lofi setup ace-step        -- Download ACE-Step weights
:Lofi config backend ace-step  -- Switch to ACE-Step backend
:Lofi config backend musicgen  -- Switch back to MusicGen
```

### Model Location

```
~/.cache/lofi.nvim/
├── musicgen/               -- MusicGen weights (~2GB)
├── ace-step/               -- ACE-Step weights (7.7GB, optional)
│   ├── ace_step_transformer/
│   ├── umt5-base/
│   ├── music_dcae_f8c8/
│   └── music_vocoder/
└── tracks/                 -- Generated/cached audio
```

The daemon detects available backends at startup and uses the configured preference.

## Constitution Alignment

- Principle II (Local & Private): all inference runs locally, no network after model download
- Principle III (Async-First): generation runs in background, never blocks editor
- Principle IV (Minimal Footprint): single model checkpoint, weights downloaded once

## Architecture Overview

ACE-Step uses a diffusion-based architecture with flow matching:

| Component | Purpose | Size |
|-----------|---------|------|
| ACEStepTransformer2DModel | Diffusion denoiser | ~3.5B params |
| UMT5 | Text encoder for style prompts | ~300M params |
| MusicDCAE | Latent space encoder/decoder (8x compression) | ~100M params |
| Vocoder | Mel spectrogram to audio | ~50M params |

### Model Configuration

From `acestep/models/config.json`:
```json
{
  "num_layers": 24,
  "inner_dim": 2560,
  "attention_head_dim": 128,
  "num_attention_heads": 20,
  "mlp_ratio": 2.5,
  "max_position": 32768,
  "rope_theta": 1000000.0,
  "text_embedding_dim": 768,
  "lyric_hidden_size": 1024
}
```

### Key Architectural Differences from DiffRhythm 2

| Aspect | ACE-Step | DiffRhythm 2 |
|--------|----------|--------------|
| Framework | diffusers library | Custom |
| Attention | Standard LinearTransformerBlock | LlamaNAR with flex_attention |
| Position encoding | Qwen2RotaryEmbedding (precomputed) | LlamaRotaryEmbedding (dynamic) |
| KV caching | None (fixed diffusion steps) | Block-based with pad_sequence |
| Inference loop | Standard scheduler.step() | Custom ODE with block cache |

## Model Files

From `ACE-Step/ACE-Step-v1-3.5B` on HuggingFace:

| Directory | Purpose |
|-----------|---------|
| `ace_step_transformer/` | Main diffusion model |
| `umt5-base/` | Text encoder |
| `music_dcae_f8c8/` | Latent space autoencoder |
| `music_vocoder/` | Audio synthesis |

Total size: **7.7GB** (verified)

| Component | Size |
|-----------|------|
| ace_step_transformer | 6.2GB |
| umt5-base | 1.1GB |
| music_dcae_f8c8 | 299MB |
| music_vocoder | 197MB |

## Generation Parameters

```python
# From pipeline_ace_step.py
audio_duration: float = 60.0      # 5-240 seconds
prompt: str = "lofi hip hop"      # Style description
lyrics: str = ""                  # Empty for instrumental
infer_step: int = 27              # Diffusion steps (27 fast, 60 quality)
guidance_scale: float = 15.0      # CFG strength
scheduler_type: str = "euler"     # euler, heun, or pingpong
cfg_type: str = "apg"             # apg, cfg, or cfg_star
omega_scale: float = 10.0         # Noise scale
manual_seeds: list = None         # Reproducibility
```

### Scheduler Options

| Scheduler | Speed | Quality | Notes |
|-----------|-------|---------|-------|
| euler | Fastest | Good | Default, 27 steps recommended |
| heun | Medium | Better | 2x NFE per step |
| pingpong | Slowest | Best | SDE-based, better lyric/style alignment |

**Note on pingpong**: This scheduler uses Stochastic Differential Equations (SDE) instead of ODE, which adds controlled noise during sampling. This improves music consistency and quality at the cost of speed. Recommended for final renders, not preview generation.

## Hardware Performance

From official benchmarks:

| Device | RTF (27 steps) | Time for 1 min audio |
|--------|----------------|----------------------|
| RTX 4090 | 34.48x | 1.74s |
| A100 | 27.27x | 2.20s |
| RTX 3090 | 12.76x | 4.70s |
| M2 Max | 2.27x | 26.43s |

RTF = Real-Time Factor. Higher is faster.

### VRAM Requirements

| Mode | VRAM | Decode Speed | Notes |
|------|------|--------------|-------|
| Full precision | **16 GB** | Fast (~0.5s/30s) | All models on GPU |
| CPU offload | **~1 GB active** | Slow (~17s/30s) | Vocoder on CPU bottleneck |
| Quantized (int4) | ~4GB | TBD | Requires torch.compile |

**Verified on M4 Pro 24GB (MPS)**: cpu_offload reduces active VRAM from 16GB to 1GB but vocoder decode becomes 34x slower.

For ONNX deployment: Keep vocoder (~200MB) permanently on GPU. No cpu_offload needed.

### ONNX Memory Management

Unlike PyTorch's `cpu_offload` (which moves tensors per forward pass), the ONNX deployment manages memory by controlling session lifecycle:

| Model | Size | Strategy |
|-------|------|----------|
| UMT5 | 1.1GB | Load → run → drop |
| ACEStepTransformer | 6.2GB | Load → run N steps → drop |
| MusicDCAE Decoder | 299MB | Keep resident |
| Vocoder | 197MB | Keep resident |

Peak VRAM: ~7.5GB (transformer active) or ~500MB (idle between generations).

This approach avoids the vocoder CPU bottleneck observed with PyTorch's cpu_offload.

## Integration Strategy

ONNX export with `ort` crate, consistent with existing MusicGen implementation.

### Models to Export

| Model | Input | Output | ONNX Feasibility |
|-------|-------|--------|------------------|
| UMT5 | text tokens | embeddings [B, seq, 768] | High - standard transformer |
| ACEStepTransformer | latent, timestep, conditions | denoised latent | High - no dynamic cache |
| MusicDCAE Encoder | audio waveform | latent [B, 8, 16, T] | High - conv-based |
| MusicDCAE Decoder | latent | mel spectrogram | High - conv-based |
| Vocoder | mel spectrogram | audio waveform | High - BigVGAN-style |

### ONNX Export Approach

ACE-Step's architecture is ONNX-friendly:

1. **No dynamic KV caching**: Uses fixed diffusion steps, no growing cache tensors
2. **Standard attention**: LinearTransformerBlock without flex_attention
3. **Precomputed RoPE**: Qwen2RotaryEmbedding caches cos/sin tensors
4. **diffusers patterns**: ConfigMixin/ModelMixin provide standard serialization

```rust
// Rust-side diffusion loop
let scheduler = FlowMatchEulerScheduler::new(1000, 3.0);
let timesteps = scheduler.timesteps(infer_steps);

for t in timesteps {
    let noise_pred = transformer_session.run(&[
        latents,
        t,
        text_embeddings,
        attention_mask,
    ])?;

    latents = scheduler.step(noise_pred, t, latents);
}
```

### Lyric Encoder Bypass

For instrumental generation, lyrics are empty. The pipeline handles this:
```python
lyric_token_idx = torch.tensor([0]).repeat(batch_size, 1)
lyric_mask = torch.tensor([0]).repeat(batch_size, 1)
```

No G2P or espeak-ng dependency needed for instrumental tracks.

### ONNX Validation

Before committing to ONNX export, run a smoke test:

```python
import torch
from acestep.models.ace_step_transformer import ACEStepTransformer2DModel

model = ACEStepTransformer2DModel.from_pretrained("path/to/ace_step_transformer")
model.eval()

# Create dummy inputs matching forward signature
dummy_inputs = {
    "hidden_states": torch.randn(1, 8, 16, 256),
    "attention_mask": torch.ones(1, 256),
    "encoder_text_hidden_states": torch.randn(1, 64, 768),
    "text_attention_mask": torch.ones(1, 64),
    "speaker_embeds": torch.zeros(1, 512),
    "lyric_token_idx": torch.zeros(1, 1, dtype=torch.long),
    "lyric_mask": torch.zeros(1, 1, dtype=torch.long),
    "timestep": torch.tensor([500.0]),
}

torch.onnx.export(
    model,
    (dummy_inputs,),
    "ace_step_transformer.onnx",
    opset_version=17,
    input_names=list(dummy_inputs.keys()),
    dynamic_axes={"hidden_states": {3: "seq_len"}, "attention_mask": {1: "seq_len"}}
)
```

If export fails, the error will identify problematic ops immediately.

## JSON-RPC Methods

```json
// Request
{"method": "generate", "params": {
  "prompt": "lofi hip hop, jazzy piano, chill beats, vinyl crackle",
  "duration_sec": 120,
  "infer_steps": 27,
  "guidance_scale": 15.0,
  "scheduler": "euler",
  "seed": 42
}, "id": 1}

// Response (immediate)
{"result": {"track_id": "a1b2c3d4", "status": "queued", "position": 0}}

// Notification: progress
{"method": "generation_progress", "params": {
  "track_id": "a1b2c3d4",
  "percent": 45,
  "step": 12,
  "total_steps": 27,
  "eta_sec": 3
}}

// Notification: complete
{"method": "generation_complete", "params": {
  "track_id": "a1b2c3d4",
  "path": "/home/user/.cache/nvim/lofi/a1b2c3d4.wav",
  "duration_sec": 120.0,
  "sample_rate": 48000,  // resampled from internal 44100 Hz
  "prompt": "lofi hip hop, jazzy piano, chill beats, vinyl crackle",
  "seed": 42,
  "generation_time_sec": 3.5,
  "model_version": "ace-step-v1-3.5b"
}}
```

## Dependencies

- daemon-lifecycle.md (daemon must be running)
- cache-management.md (track storage)
- progress-notifications.md (UI updates)

## Error Codes

- MODEL_NOT_FOUND: Model files not at expected path
- MODEL_LOAD_FAILED: corrupt file, wrong format, OOM, CUDA unavailable
- MODEL_INFERENCE_FAILED: OOM during generation, numerical instability
- QUEUE_FULL: max pending requests exceeded
- INVALID_DURATION: duration outside 5-240 second range

## Lua API

```lua
local lofi = require("lofi")
lofi.generate({
  prompt = "lofi hip hop, chill beats",
  duration_sec = 120,
  seed = nil
}, callback)
-- callback(err, track) where track = {track_id, path, duration_sec, ...}
```

## Events

```lua
lofi.on("generation_start", function(data) end)     -- {track_id}
lofi.on("generation_progress", function(data) end)  -- {track_id, percent, eta_sec}
lofi.on("generation_complete", function(data) end)  -- {track_id, path, ...}
lofi.on("generation_error", function(data) end)     -- {track_id, error, code}
```

## Success Criteria

- 120s audio generated on RTX 3090 in <15 seconds (27 steps)
- Model loads in <30s on modern hardware
- Progress updates accurate within 10% of actual completion
- Same seed produces identical output (on same machine/device)
- No audible artifacts in generated audio
- Instrumental generation works without lyrics input

## Platform Notes

### macOS

Use `--bf16 false` to avoid precision errors on Apple Silicon:
```lua
-- In config
{
  model = {
    bf16 = false  -- Required for macOS
  }
}
```

## Advanced Features (Future)

ACE-Step supports additional capabilities that could be exposed later:

- **Retake**: Regenerate with variations using different seeds
- **Repainting**: Selectively regenerate specific time segments
- **Extend**: Add audio to beginning or end of existing track
- **Audio2Audio**: Use reference audio for style transfer
- **LoRA**: Fine-tune for specific styles (e.g., RapMachine LoRA)
