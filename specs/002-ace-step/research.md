# Research: ACE-Step Long-Form Music Generation

**Phase 0 Output** | **Branch**: `002-ace-step` | **Date**: 2025-12-21

## Executive Summary

ACE-Step is a 3.5B parameter diffusion model capable of generating up to 240 seconds of instrumental audio. Integration into lofi.nvim requires ONNX export of 4 components (text encoder, transformer, DCAE decoder, vocoder), with control flow and scheduling implemented in Rust. The architecture is more complex than MusicGen but feasible for ONNX deployment.

---

## Research Topics Addressed

### 1. ACE-Step Model Architecture

**Decision**: ACE-Step uses a 4-component pipeline architecture

**Rationale**: The pipeline separates concerns clearly:
1. **UMT5 Text Encoder** - Converts prompts to embeddings (768-dim)
2. **ACEStepTransformer** - 24-layer diffusion transformer (2560 hidden, 20 heads)
3. **Music DCAE Decoder** - Latent to mel-spectrogram (8x compression)
4. **ADaMoSHiFiGAN Vocoder** - Mel-spectrogram to 48kHz waveform

**Alternatives Considered**:
- Single monolithic ONNX model: Rejected due to complexity and memory requirements
- Python subprocess: Rejected per constitution (offline, minimal footprint)

### 2. ONNX Export Strategy

**Decision**: Export 4 separate ONNX models with Rust orchestration

**Rationale**: The diffusion loop involves control flow (guidance, scheduling) that's better implemented in Rust than traced into ONNX. This matches the MusicGen pattern already in use.

**Components and Export Feasibility**:

| Component | Exportability | Size (est.) | Notes |
|-----------|---------------|-------------|-------|
| UMT5 Text Encoder | High | ~500MB | Standard transformer, HF export support |
| ACEStepTransformer | Medium | ~6.5GB | Custom attention, rotary embeddings |
| DCAE Decoder | High | ~300MB | Diffusers AutoencoderDC |
| Vocoder | High | ~400MB | Convolutional HiFiGAN variant |

**Total estimated ONNX model size**: ~7.7GB (matches spec assumptions)

**Export Strategy**:
1. Use `torch.onnx.export` with opset 17+ for transformer ops
2. Export text encoder via Optimum/Transformers library
3. Handle rotary embeddings as custom ops or pre-compute tables
4. Test each component independently before integration

### 3. Inference Pipeline Implementation

**Decision**: Implement diffusion loop in Rust with ONNX model calls

**Pipeline Steps (instrumental mode)**:
```
1. Tokenize prompt → UMT5 tokenizer (Rust: tokenizers crate)
2. Encode prompt → UMT5 ONNX model → text_embeddings (B, T, 768)
3. Initialize latent noise → (B, 8, 16, frame_length)
4. For step in 1..infer_steps:
   a. Compute timestep from scheduler
   b. Run transformer with guidance → noise prediction
   c. Update latents via scheduler step
5. Decode latents → DCAE ONNX → mel-spectrogram
6. Synthesize audio → Vocoder ONNX → waveform
7. Resample 44.1kHz → 48kHz if needed
8. Save to cache as WAV
```

**Rationale**: This pattern matches MusicGen's autoregressive loop but swaps token generation for diffusion steps.

### 4. Scheduler Implementation

**Decision**: Implement 3 schedulers in Rust: Euler, Heun, PingPong

**Rationale**: Schedulers are mathematical operations (ODE/SDE solvers) that don't benefit from ONNX acceleration. Implementing in Rust provides:
- No ONNX graph complexity
- Easy customization
- Better debugging

**Scheduler Details**:

| Scheduler | Type | Deterministic | Quality | Speed |
|-----------|------|---------------|---------|-------|
| Euler | ODE | Yes | Good | Fast |
| Heun | ODE | Yes | Better | 2x Euler |
| PingPong | SDE | No | Best | Variable |

**Default**: Euler with 60 steps (matches ACE-Step defaults)

### 5. Guidance Implementation (CFG)

**Decision**: Implement Classifier-Free Guidance in Rust

**Rationale**: CFG requires running the transformer twice per step (conditional + unconditional) and combining outputs. This is control flow, not model computation.

**Implementation**:
```rust
fn apply_cfg(cond_output: Tensor, uncond_output: Tensor, scale: f32) -> Tensor {
    uncond_output + scale * (cond_output - uncond_output)
}
```

**Guidance Scale Default**: 15.0 (from ACE-Step paper)

**Note**: APG (Adaptive Perturbed Guidance) is optional advanced mode. Start with basic CFG.

### 6. Duration and Frame Calculations

**Decision**: Use ACE-Step's native frame calculation formula

**Latent Frame Calculation**:
```
frame_length = int(duration_sec * 44100 / 512 / 8)
            = int(duration_sec * 10.77)
```

| Duration | Frames | Latent Shape |
|----------|--------|--------------|
| 30 sec | 323 | (1, 8, 16, 323) |
| 60 sec | 646 | (1, 8, 16, 646) |
| 120 sec | 1292 | (1, 8, 16, 1292) |
| 240 sec | 2585 | (1, 8, 16, 2585) |

**Memory Scaling**: Latent memory is O(duration), transformer attention is O(duration^2)

### 7. Sample Rate Handling

**Decision**: Generate at 44.1kHz internally, resample to 48kHz for output

**Rationale**: ACE-Step's vocoder outputs 44.1kHz. Resampling to 48kHz matches lofi.nvim's standard output format and provides better compatibility with audio systems.

**Implementation**: Use rubato or samplerate crate for high-quality resampling

### 8. macOS Precision Handling

**Decision**: Force fp32 on Apple Silicon; use bf16 elsewhere when available

**Rationale**: Apple Silicon MPS has bf16 precision issues that cause numerical instability in diffusion models. ONNX Runtime's CoreML EP handles precision automatically, but explicit fp32 is safer.

**Implementation**:
```rust
let precision = if cfg!(target_os = "macos") {
    Precision::Fp32
} else {
    Precision::Bf16
};
```

### 9. Progress Calculation

**Decision**: Progress based on diffusion steps, not tokens

**MusicGen (current)**: Progress = tokens_generated / total_tokens
**ACE-Step**: Progress = steps_completed / inference_steps

**ETA Calculation**:
```rust
fn calculate_eta(elapsed_secs: f64, current_step: u32, total_steps: u32) -> f64 {
    if current_step == 0 { return 0.0; }
    let time_per_step = elapsed_secs / current_step as f64;
    time_per_step * (total_steps - current_step) as f64
}
```

### 10. Backend Abstraction Design

**Decision**: Use enum dispatch rather than trait objects

**Rationale**: Only 2 backends (MusicGen, ACE-Step). Enum dispatch is simpler, avoids dyn overhead, and matches constitution principle V (simplicity).

**Implementation Pattern**:
```rust
pub enum Backend {
    MusicGen,
    AceStep,
}

pub enum LoadedModels {
    MusicGen(MusicGenModels),
    AceStep(AceStepModels),
}

impl LoadedModels {
    pub fn generate(&self, params: &GenerateParams) -> Result<Track, Error> {
        match self {
            Self::MusicGen(m) => m.generate(params),
            Self::AceStep(m) => m.generate(params),
        }
    }
}
```

### 11. Model Download and Storage

**Decision**: Store ACE-Step models in separate subdirectory from MusicGen

**Structure**:
```
~/.cache/lofi.nvim/
├── musicgen/           # Existing MusicGen models (~250MB)
│   ├── tokenizer.json
│   ├── text_encoder.onnx
│   ├── decoder_model.onnx
│   ├── decoder_with_past_model.onnx
│   └── encodec_decode.onnx
├── ace-step/           # New ACE-Step models (~11.5GB)
│   ├── tokenizer.json           # 16.8 MB - UMT5 tokenizer
│   ├── text_encoder.onnx        # 1.13 GB - UMT5 text encoder
│   ├── transformer_encoder.onnx # 424 MB - ACE-Step encoder
│   ├── transformer_decoder.onnx # 35.7 MB - ACE-Step decoder (graph only)
│   ├── transformer_decoder_weights.bin  # 9.97 GB - ACE-Step decoder weights
│   ├── dcae_decoder.onnx        # 317 MB - MusicDCAE latent decoder
│   └── vocoder.onnx             # 412 MB - ADaMoSHiFiGAN vocoder
└── tracks/             # Generated audio cache
```

**Download Source**: HuggingFace `willibrandon/lofi-models` (https://huggingface.co/willibrandon/lofi-models/tree/main/ace-step/)

### 12. Cancellation Implementation

**Decision**: Use atomic flag checked between diffusion steps

**Rationale**: Diffusion models process step-by-step, providing natural cancellation points. Check between steps avoids corrupted state.

**Implementation**:
```rust
for step in 0..inference_steps {
    if cancellation_token.load(Ordering::Relaxed) {
        return Err(Error::Cancelled);
    }
    // Run diffusion step...
}
```

### 13. Cache Key Generation

**Decision**: Include backend identifier in track ID hash

**Current MusicGen**: `SHA256(prompt + seed + duration + model_version)`
**With ACE-Step**: `SHA256(prompt + seed + duration + model_version + backend)`

This ensures MusicGen and ACE-Step tracks don't collide even with identical parameters.

### 14. Lyric Encoding (Instrumental Only)

**Decision**: Skip lyric encoding for instrumental-only initial release

**Rationale**: Spec explicitly scopes this feature to instrumental generation (Out of Scope: "Lyric-based generation"). The lyric encoder adds complexity:
- Language detection (requires additional library)
- BPE tokenization with 6693-token vocab
- Conformer neural network

For instrumental mode, pass empty/zero lyric embeddings to the transformer.

---

## Unresolved Items

None - all NEEDS CLARIFICATION items from Technical Context have been addressed.

---

## Dependencies Identified

### Rust Crates (No New Additions Expected)
- `ort` 2.0.0-rc.9 - Already in use for MusicGen ONNX
- `ndarray` - Already in use for tensor ops
- `tokenizers` - Already in use for text tokenization

### Potential New Crates (Evaluate During Implementation)
- `rubato` - High-quality resampling (44.1kHz → 48kHz)
  - Alternative: `samplerate` crate

### Model Files (HuggingFace)
- `willibrandon/lofi-models` - ONNX exports for ACE-Step (https://huggingface.co/willibrandon/lofi-models/tree/main/ace-step/)
- Original PyTorch source: `ACE-Step/ACE-Step-v1-3.5B`

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Transformer ONNX export fails | Medium | High | Incremental export testing; fallback to smaller model or defer feature |
| Memory exceeds constraints on 8GB VRAM | Medium | Medium | Implement sliding window for long generations |
| Performance below targets on Apple Silicon | Low | Medium | Optimize with CoreML EP; accept slower performance with clear messaging |
| Rotary embedding export issues | Medium | Medium | Pre-compute embedding tables or implement custom op |

---

## Next Steps (Phase 1 Prerequisites)

1. Create ONNX export scripts for ACE-Step components
2. Test export of each component independently
3. Measure exported model sizes and runtime performance
4. Define RPC contract extensions for ACE-Step parameters
5. Design data model for backend abstraction
