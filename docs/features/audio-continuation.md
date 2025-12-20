/speckit.specify Long-form audio generation via sliding window continuation

## Context

MusicGen's position embeddings limit generation to ~40 seconds (2048 tokens at 50Hz). This feature enables generation of longer tracks (up to 10 minutes) using a sliding window approach that conditions each chunk on the previous audio.

**Reference**: Meta's AudioCraft uses this technique - generate 30s windows, slide by 10s, keep 20s of overlap as audio conditioning.

**Local Reference**: `../audiocraft` (clone https://github.com/facebookresearch/audiocraft)

## Constitution Alignment

- Principle II (Local & Private): continuation runs locally, no external services
- Principle III (Async-First): chunked generation streams progress, never blocks
- Principle V (Simplicity): uses existing MusicGen models, adds one encoder model

## Risk Assessment

**Required Model Addition:**
- `encodec_encode.onnx` - converts audio back to codec tokens for conditioning
- Must be exported from PyTorch AudioCraft or sourced from community
- ~60MB estimated size (similar to decode model)

**Approach (proven by AudioCraft):**
1. Generate first 30s chunk from text prompt only
2. Encode last 20s of audio → codec tokens via encodec_encode
3. Concatenate conditioning tokens with text embeddings
4. Generate next 30s chunk (only last 10s is new audio)
5. Repeat until target duration reached
6. Stitch non-overlapping 10s segments

**Phase 0 Go/No-Go Checkpoint:**
Before full implementation:
1. Export or obtain encodec_encode.onnx
2. Verify round-trip: audio → encode → decode → audio matches
3. Generate 60s track (2 chunks) with audible continuity

## Requirements

### Duration extension
- Accept duration_sec up to 600 (10 minutes)
- Automatically chunk into 30s windows with 20s overlap
- Each chunk generates 10s of new audio
- Total chunks = ceil((duration - 30) / 10) + 1

### Audio conditioning
- Encode last 20s of previous chunk to codec tokens
- Pass tokens as decoder conditioning alongside text embeddings
- Maintain consistent prompt across all chunks
- Use same seed base with chunk index for reproducibility

### Seamless stitching
- Extract only new 10s from each chunk (positions 20-30s)
- First chunk uses full 30s
- No crossfade needed - model generates continuous audio
- Output single WAV file

### Progress reporting
- Report overall progress across all chunks
- Percent = (completed_chunks * 10 + current_chunk_progress) / total_new_seconds
- ETA accounts for all remaining chunks

## Model files

### New model required
| File | Size | Source |
|------|------|--------|
| `encodec_encode.onnx` | ~60MB | Export from AudioCraft PyTorch |

### Export procedure
```python
from audiocraft.models import MusicGen
import torch

model = MusicGen.get_pretrained('facebook/musicgen-small')
encoder = model.compression_model.encoder

# Export to ONNX
torch.onnx.export(
    encoder,
    dummy_audio_input,
    "encodec_encode.onnx",
    input_names=["audio"],
    output_names=["codes"],
    dynamic_axes={"audio": {2: "samples"}, "codes": {2: "frames"}}
)
```

## JSON-RPC changes

### Extended generate params
```json
{"method": "generate", "params": {
  "prompt": "lofi hip hop, jazzy piano",
  "duration_sec": 120,
  "seed": 42
}, "id": 1}
```

### Chunk progress notification
```json
{"method": "generation_progress", "params": {
  "track_id": "a1b2c3d4",
  "percent": 45,
  "chunk_current": 3,
  "chunk_total": 10,
  "eta_sec": 180
}}
```

## Dependencies
- ai-generation.md (base generation infrastructure)
- cache-management.md (track storage)
- progress-notifications.md (UI updates)

## Error codes
- ENCODER_NOT_FOUND: encodec_encode.onnx missing
- CHUNK_FAILED: individual chunk generation failed
- CONTINUATION_MISMATCH: audio encoding produced unexpected token shape

## Lua API
```lua
-- Same API, just supports longer durations
lofi.generate({prompt = "...", duration_sec = 120}, callback)
```

## Success criteria
- 120s track generates successfully with audible continuity
- No clicks, pops, or tonal shifts at chunk boundaries
- Generation time scales linearly: ~25s per 10s of audio
- Same seed produces identical output across chunks
- Memory usage stays constant regardless of total duration
