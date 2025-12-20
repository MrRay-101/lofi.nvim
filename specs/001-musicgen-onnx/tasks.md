# Tasks: AI Music Generation via MusicGen ONNX Backend

**Input**: Design documents from `/specs/001-musicgen-onnx/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Not explicitly requested in specification. Verification tasks included per Principle VI.

**Organization**: Tasks organized by implementation phase with Phase 0 as critical go/no-go gate before user story implementation.

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4)
- **[P0]**: Phase 0 feasibility checkpoint task

## Path Conventions

Based on plan.md structure:
- **Daemon**: `daemon/src/` - Rust backend
- **Lua Plugin**: `lua/lofi/` - Neovim integration
- **Tests**: `daemon/tests/` - Rust tests

---

## Phase 1: Setup (Project Initialization)

**Purpose**: Create project structure and configure dependencies per plan.md

- [x] T001 Create daemon directory structure with `mkdir -p daemon/src/{rpc,generation,models,audio,cache}`
- [x] T002 Initialize Rust project with `cargo init --name lofi-daemon` in daemon/
- [x] T003 Configure Cargo.toml with dependencies from research.md: ort 2.0.0-rc.9, ndarray 0.16.1, tokenizers 0.19.1, half 2.4.1, hound 3.5.1, reqwest, tokio, serde, serde_json, sha2, clap, anyhow, directories in daemon/Cargo.toml
- [x] T004 [P] Create Lua plugin directory structure with `mkdir -p lua/lofi`
- [x] T005 [P] Add .gitignore entries for target/, *.onnx, and cache directories

### Phase 1 Verification (MANDATORY - Principle VI)

- [x] V001 [VERIFY] Run `cargo check` in daemon/ - must succeed with zero errors
- [x] V002 [VERIFY] Run `grep -rn "TODO\|FIXME" daemon/src/` - must return empty
- [x] V003 [VERIFY] Confirm daemon/Cargo.toml exists and contains all required dependencies

---

## Phase 2: Foundational (Core Types & Utilities)

**Purpose**: Implement shared types and utilities that ALL phases depend on

**CRITICAL**: No user story work can begin until this phase AND Phase 0 are complete

- [x] T006 Create error types enum with MODEL_NOT_FOUND, MODEL_LOAD_FAILED, MODEL_DOWNLOAD_FAILED, MODEL_INFERENCE_FAILED, QUEUE_FULL, INVALID_DURATION, INVALID_PROMPT in daemon/src/error.rs
- [x] T007 [P] Create Track struct with track_id, path, prompt, duration_sec, sample_rate, seed, model_version, generation_time_sec, created_at per data-model.md in daemon/src/types/track.rs
- [x] T008 [P] Create GenerationJob struct with job_id, track_id, prompt, duration_sec, seed, priority, status, progress fields per data-model.md in daemon/src/types/job.rs
- [x] T009 [P] Create ModelConfig struct with vocab_size, num_hidden_layers, num_attention_heads, d_model, d_kv, audio_channels, sample_rate, codebooks, pad_token_id in daemon/src/types/config.rs
- [x] T010 Create types module re-exporting Track, GenerationJob, ModelConfig, error types in daemon/src/types/mod.rs
- [x] T011 Create config module with Device enum (Auto, Cpu, Cuda, Metal), DaemonConfig struct with model_path, cache_path, device, threads in daemon/src/config.rs
- [x] T012 [P] Create track_id computation function using sha256(prompt:seed:duration:model_version).hex()[0:16] in daemon/src/types/track.rs
- [x] T013 Create lib.rs exposing types, config, error modules in daemon/src/lib.rs

### Phase 2 Verification (MANDATORY - Principle VI)

- [x] V004 [VERIFY] Run `cargo build` in daemon/ - must succeed with zero errors
- [x] V005 [VERIFY] Run `grep -rn "TODO\|FIXME" daemon/src/` - must return empty
- [x] V006 [VERIFY] Confirm all types/mod.rs exports are imported in lib.rs

---

## Phase 3: Phase 0 CLI - Feasibility Checkpoint (GO/NO-GO GATE)

**Purpose**: Prove MusicGen ONNX generation works in standalone Rust CLI before daemon integration

**GO/NO-GO CRITERIA**: 10 seconds of audio generated on CPU in under 2 minutes

**CRITICAL**: If Phase 0 FAILS, stop and investigate. Do NOT proceed to user stories.

### Core Model Pipeline

- [ ] T014 [P0] Create delay pattern masking for 4-codebook architecture with DelayPatternMaskIds struct, push(), last_delayed_masked(), last_de_delayed() per research.md D4 in daemon/src/models/delay_pattern.rs
- [ ] T015 [P0] Create text encoder wrapper with MusicGenTextEncoder struct loading tokenizer.json and text_encoder.onnx, encode(text) -> (embeddings, attention_mask) in daemon/src/models/text_encoder.rs
- [ ] T016 [P0] Create logits processing with apply_free_guidance(scale=3.0), sample_top_k() functions per research.md D5 in daemon/src/models/logits.rs
- [ ] T017 [P0] Create decoder wrapper with MusicGenDecoder struct supporting KV cache, first iteration with decoder_model.onnx, subsequent with decoder_with_past_model.onnx per research.md D3 in daemon/src/models/decoder.rs
- [ ] T018 [P0] Create audio codec wrapper with MusicGenAudioCodec struct loading encodec_decode.onnx, decode(tokens) -> Vec<f32> audio samples in daemon/src/models/audio_codec.rs
- [ ] T019 [P0] Create model loader with load_sessions(model_dir) -> Result<(TextEncoder, Decoder, AudioCodec)> handling ONNX session creation in daemon/src/models/loader.rs
- [ ] T020 [P0] Create models module re-exporting all model components in daemon/src/models/mod.rs

### Audio Output

- [ ] T021 [P0] [P] Create WAV writer with write_wav(samples: &[f32], path: &Path, sample_rate: 32000) using hound crate per research.md D6 in daemon/src/audio/wav.rs
- [ ] T022 [P0] Create audio module re-exporting wav in daemon/src/audio/mod.rs

### Phase 0 CLI Entry Point

- [ ] T023 [P0] Create CLI argument parser with prompt, duration (5-120, default 10), output path, model_dir using clap in daemon/src/cli.rs
- [ ] T024 [P0] Create generation pipeline function generate(prompt, duration, seed, model_dir) -> Result<Vec<f32>> orchestrating text_encoder -> decoder loop -> audio_codec in daemon/src/generation/pipeline.rs
- [ ] T025 [P0] Create main.rs with CLI mode detection, model loading, generation execution, WAV output, timing measurement in daemon/src/main.rs
- [ ] T026 [P0] Create generation module re-exporting pipeline in daemon/src/generation/mod.rs
- [ ] T027 [P0] Update lib.rs to export models, audio, generation modules in daemon/src/lib.rs

### Phase 0 Verification (MANDATORY - Principle VI)

- [ ] V007 [VERIFY] Run `cargo build --release` in daemon/ - must succeed with zero warnings
- [ ] V008 [VERIFY] Run `grep -rn "TODO\|FIXME" daemon/src/` - must return empty
- [ ] V009 [VERIFY] Confirm all daemon/src/*.rs files are imported in lib.rs or main.rs
- [ ] V010 [VERIFY] Run `cargo run --release -- --help` - must show CLI usage
- [ ] V011 [VERIFY] Run Phase 0 validation: `cargo run --release -- --prompt "lofi hip hop" --duration 10 --output test.wav` - must complete in <120s
- [ ] V012 [VERIFY] Confirm test.wav exists and is playable audio
- [ ] V013 [VERIFY] **GO/NO-GO DECISION**: If V011 and V012 pass, proceed to Phase 4. If FAIL, stop and investigate.

---

## Phase 4: User Story 1 - Generate Music from Text Prompt (Priority: P1) MVP

**Goal**: Users can invoke generation command and receive a complete audio file

**Independent Test**: Run `generate` JSON-RPC method and verify WAV file created

**Depends on**: Phase 0 PASS (V013)

### Model Download (FR-024 through FR-027)

- [ ] T028 [US1] Create model downloader with download_model(url, dest, progress_callback) using reqwest streaming in daemon/src/models/downloader.rs
- [ ] T029 [US1] Create model existence check with check_models(model_dir) -> Result<(), MissingModels> in daemon/src/models/loader.rs
- [ ] T030 [US1] Add download_all_models(model_dir, consent, progress_callback) orchestrating 6 model files from HuggingFace gabotechs/music_gen in daemon/src/models/downloader.rs

### Cache Management

- [ ] T031 [US1] [P] Create cache module with Cache struct, get(track_id), put(track), contains(track_id), evict_lru() in daemon/src/cache/tracks.rs
- [ ] T032 [US1] Create cache module re-exporting tracks in daemon/src/cache/mod.rs

### JSON-RPC Types

- [ ] T033 [US1] [P] Create GenerateRequest struct with prompt, duration_sec, seed, priority per contracts/generate.json in daemon/src/rpc/types.rs
- [ ] T034 [US1] [P] Create GenerateResponse struct with track_id, status, position, seed per contracts/generate.json in daemon/src/rpc/types.rs
- [ ] T035 [US1] [P] Create GenerationCompleteNotification struct with track_id, path, duration_sec, sample_rate, prompt, seed, generation_time_sec, model_version per contracts/notifications.json in daemon/src/rpc/types.rs
- [ ] T036 [US1] [P] Create JsonRpcError struct with code, message, data per contracts/errors.json in daemon/src/rpc/types.rs

### Daemon Mode

- [ ] T037 [US1] Create JSON-RPC server reading from stdin, writing to stdout, dispatching methods in daemon/src/rpc/server.rs
- [ ] T038 [US1] Create generate method handler validating request, computing track_id, triggering generation in daemon/src/rpc/methods.rs
- [ ] T039 [US1] Create rpc module re-exporting server, methods, types in daemon/src/rpc/mod.rs
- [ ] T040 [US1] Update main.rs to detect daemon mode (no CLI args) vs CLI mode, start JSON-RPC server in daemon mode in daemon/src/main.rs
- [ ] T041 [US1] Update lib.rs to export rpc, cache modules in daemon/src/lib.rs

### User Story 1 Verification (MANDATORY - Principle VI)

- [ ] V014 [VERIFY] Run `cargo build --release` - must succeed with zero errors
- [ ] V015 [VERIFY] Run `grep -rn "TODO\|FIXME" daemon/src/` - must return empty
- [ ] V016 [VERIFY] Confirm all US1 files are imported and used
- [ ] V017 [VERIFY] Test daemon mode: echo generate request to stdin, verify response with track_id

---

## Phase 5: User Story 2 - Monitor Generation Progress (Priority: P2)

**Goal**: Users receive progress updates during generation showing percent, ETA

**Independent Test**: Initiate generation, observe progress notifications every 5%

**Depends on**: User Story 1 complete

### Progress Calculation

- [ ] T042 [US2] Create progress calculator with ProgressTracker struct, update(tokens_generated), get_percent(), get_eta() computing tokens_estimated = duration * 50 in daemon/src/generation/progress.rs
- [ ] T043 [US2] Create GenerationProgressNotification struct with track_id, percent, tokens_generated, tokens_estimated, eta_sec per contracts/notifications.json in daemon/src/rpc/types.rs

### Progress Streaming

- [ ] T044 [US2] Update generation pipeline to accept progress_callback, call every 5% increment in daemon/src/generation/pipeline.rs
- [ ] T045 [US2] Create send_notification(notification) function writing JSON-RPC notification to stdout in daemon/src/rpc/server.rs
- [ ] T046 [US2] Integrate progress notifications into generate method handler, sending generation_progress every 5% in daemon/src/rpc/methods.rs
- [ ] T047 [US2] Add generation_complete and generation_error notifications on completion/failure in daemon/src/rpc/methods.rs
- [ ] T048 [US2] Update generation module re-exporting progress in daemon/src/generation/mod.rs

### User Story 2 Verification (MANDATORY - Principle VI)

- [ ] V018 [VERIFY] Run `cargo build --release` - must succeed with zero errors
- [ ] V019 [VERIFY] Run `grep -rn "TODO\|FIXME" daemon/src/` - must return empty
- [ ] V020 [VERIFY] Test progress: initiate generation, verify ~20 progress notifications received
- [ ] V021 [VERIFY] Verify progress percent capped at 99 until generation_complete

---

## Phase 6: User Story 3 - Queue Multiple Generation Requests (Priority: P3)

**Goal**: Users can queue up to 10 requests, high-priority skips to front

**Independent Test**: Submit multiple requests, verify queue positions, QUEUE_FULL at 11

**Depends on**: User Story 1 complete (can parallel with US2)

### Job Queue

- [ ] T049 [US3] Create GenerationQueue struct with jobs: VecDeque<GenerationJob>, add(job, priority), pop_next(), len(), is_full() in daemon/src/generation/queue.rs
- [ ] T050 [US3] Implement priority insertion: high-priority jobs inserted at front of queue in daemon/src/generation/queue.rs
- [ ] T051 [US3] Add queue position tracking with get_position(job_id) in daemon/src/generation/queue.rs

### Queue Integration

- [ ] T052 [US3] Update generate method to add job to queue, return position in response in daemon/src/rpc/methods.rs
- [ ] T053 [US3] Create queue processor running in background thread, processing jobs serially in daemon/src/generation/queue.rs
- [ ] T054 [US3] Add QUEUE_FULL error when queue.len() >= 10 in daemon/src/rpc/methods.rs
- [ ] T055 [US3] Update generation module re-exporting queue in daemon/src/generation/mod.rs

### User Story 3 Verification (MANDATORY - Principle VI)

- [ ] V022 [VERIFY] Run `cargo build --release` - must succeed with zero errors
- [ ] V023 [VERIFY] Run `grep -rn "TODO\|FIXME" daemon/src/` - must return empty
- [ ] V024 [VERIFY] Test queue: submit 3 requests, verify positions 0, 1, 2 returned
- [ ] V025 [VERIFY] Test queue full: submit 11 requests, verify 11th returns QUEUE_FULL error

---

## Phase 7: User Story 4 - Configure Hardware Acceleration (Priority: P4)

**Goal**: Users can select device (auto/cpu/cuda/metal) for optimal performance

**Independent Test**: Set device config, verify GPU used when available

**Depends on**: User Story 1 complete (can parallel with US2, US3)

### Device Detection

- [ ] T056 [US4] Create device detection with detect_available_providers() -> Vec<ExecutionProvider> checking CUDA, CoreML availability in daemon/src/models/device.rs
- [ ] T057 [US4] Create execution provider selection with get_providers(device: Device) -> Vec<ExecutionProvider> per research.md D9 in daemon/src/models/device.rs

### Configuration Integration

- [ ] T058 [US4] Update model loader to accept device config, configure ONNX sessions with selected execution providers in daemon/src/models/loader.rs
- [ ] T059 [US4] Add threads configuration support for CPU execution provider in daemon/src/models/loader.rs
- [ ] T060 [US4] Update config module to load device/threads from environment or config file in daemon/src/config.rs
- [ ] T061 [US4] Update models module re-exporting device in daemon/src/models/mod.rs

### User Story 4 Verification (MANDATORY - Principle VI)

- [ ] V026 [VERIFY] Run `cargo build --release` - must succeed with zero errors
- [ ] V027 [VERIFY] Run `grep -rn "TODO\|FIXME" daemon/src/` - must return empty
- [ ] V028 [VERIFY] Test device auto: verify detection runs without error
- [ ] V029 [VERIFY] Test device cpu: verify explicit CPU selection works

---

## Phase 8: Lua Plugin Integration

**Goal**: Neovim users can generate music via Lua API

**Independent Test**: Call lofi.generate() from Neovim, verify track created

**Depends on**: User Stories 1-2 complete (progress notifications)

### Lua Plugin Core

- [ ] T062 Create daemon spawn/management with start_daemon(), stop_daemon(), is_running() in lua/lofi/daemon.lua
- [ ] T063 [P] Create JSON-RPC client with send_request(method, params), handle_notification(callback) in lua/lofi/rpc.lua
- [ ] T064 [P] Create event system with on(event, callback), emit(event, data) for generation_start/progress/complete/error in lua/lofi/events.lua
- [ ] T065 Create public API with generate(opts, callback), is_generating() per spec Lua API section in lua/lofi/init.lua

### Integration

- [ ] T066 Wire generate() to spawn daemon if needed, send generate request, handle callbacks in lua/lofi/init.lua
- [ ] T067 Wire progress/complete/error notifications to event emitters in lua/lofi/init.lua
- [ ] T068 Add setup(opts) function for configuration in lua/lofi/init.lua

### Phase 8 Verification (MANDATORY - Principle VI)

- [ ] V030 [VERIFY] Run `luacheck lua/` - must pass (if luacheck configured)
- [ ] V031 [VERIFY] Run `grep -rn "TODO\|FIXME" lua/` - must return empty
- [ ] V032 [VERIFY] Confirm all lua/lofi/*.lua files are required by init.lua
- [ ] V033 [VERIFY] Test in Neovim: `:lua require('lofi').generate({prompt='test'}, print)` executes without error

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Final integration, error handling, documentation

- [ ] T069 [P] Add comprehensive error messages with recovery hints for all error codes in daemon/src/error.rs
- [ ] T070 [P] Add input validation for prompt length (1-1000 chars), duration (5-120) in daemon/src/rpc/methods.rs
- [ ] T071 [P] Add graceful shutdown handling for daemon on stdin EOF in daemon/src/main.rs
- [ ] T072 [P] Add model version string generation (e.g., "musicgen-small-fp16-v1") in daemon/src/models/loader.rs
- [ ] T073 Run quickstart.md validation steps to confirm Phase 0 CLI still works

### Final Verification (MANDATORY - Principle VI)

- [ ] VFINAL-1 [VERIFY] Run `cargo build --release` - must succeed with zero errors/warnings
- [ ] VFINAL-2 [VERIFY] Run `grep -rn "TODO\|FIXME" daemon/src/ lua/` - must return empty
- [ ] VFINAL-3 [VERIFY] Confirm ALL written files are imported and used
- [ ] VFINAL-4 [VERIFY] Confirm ALL written functions are called
- [ ] VFINAL-5 [VERIFY] Run full smoke test: daemon generation + Lua API call

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (Setup)
    │
    ▼
Phase 2 (Foundational)
    │
    ▼
Phase 3 (Phase 0 CLI) ◄── GO/NO-GO GATE
    │
    ├── PASS ──▶ Continue to Phase 4+
    │
    └── FAIL ──▶ STOP. Investigate. Do not proceed.
    │
    ▼
Phase 4 (US1: Generation) ◄── MVP
    │
    ├────────────────┬────────────────┐
    ▼                ▼                ▼
Phase 5 (US2)    Phase 6 (US3)    Phase 7 (US4)
(Progress)       (Queue)          (Device)
    │                │                │
    └────────────────┴────────────────┘
                     │
                     ▼
              Phase 8 (Lua Plugin)
                     │
                     ▼
              Phase 9 (Polish)
```

### User Story Dependencies

- **User Story 1 (P1)**: Depends on Phase 3 (Phase 0) PASS - Core generation
- **User Story 2 (P2)**: Depends on US1 - Progress builds on generation
- **User Story 3 (P3)**: Depends on US1 - Queue manages generation jobs (can parallel with US2)
- **User Story 4 (P4)**: Depends on US1 - Device selection for model loading (can parallel with US2, US3)

### Within Each Phase

- Types/structs before functions using them
- Core logic before integration
- Module exports after implementations

### Parallel Opportunities

**Phase 2 (all can parallel)**:
- T007, T008, T009 - Independent struct definitions

**Phase 3 (model components can parallel)**:
- T014, T015, T016, T017, T018 - Independent model wrappers
- T021 - WAV writer independent of models

**Phase 4 (some can parallel)**:
- T031, T033-T036 - Cache and RPC types independent

**Phase 5, 6, 7 (can run in parallel with each other after US1)**

**Phase 8 (Lua files can parallel)**:
- T063, T064 - Independent Lua modules

---

## Parallel Example: Phase 3 Model Components

```bash
# Launch all model wrapper tasks together:
Task: "T014 Create delay pattern masking in daemon/src/models/delay_pattern.rs"
Task: "T015 Create text encoder wrapper in daemon/src/models/text_encoder.rs"
Task: "T016 Create logits processing in daemon/src/models/logits.rs"
Task: "T017 Create decoder wrapper in daemon/src/models/decoder.rs"
Task: "T018 Create audio codec wrapper in daemon/src/models/audio_codec.rs"
Task: "T021 Create WAV writer in daemon/src/audio/wav.rs"
```

---

## Implementation Strategy

### Phase 0 Gate (CRITICAL)

Phase 0 is a **GO/NO-GO** decision point:
1. Complete Phase 1-3 fully
2. Run V011 (10s generation in <120s)
3. If PASS: Proceed to Phase 4+
4. If FAIL: Stop. Do not implement user stories. Investigate root cause.

### Complete Implementation (Principle VI)

1. Complete ALL phases in sequence without stopping
2. Implement ALL user stories fully - no partial implementations
3. Write production-ready code for every task
4. Do not stop for validation checkpoints - complete everything first
5. Zero TODOs, zero stubs, zero placeholders in any code

### MVP Scope

- **Minimum**: Phase 1-4 (Setup through US1)
- **Recommended**: Phase 1-6 (through US3 for queue)
- **Full Feature**: Phase 1-9 (all phases)

---

## Notes

- [P] tasks = different files, no dependencies
- [P0] tasks = Phase 0 feasibility checkpoint
- [US#] label maps task to specific user story
- Phase 0 is GO/NO-GO gate - do not proceed if it fails
- All model patterns from MusicGPT reference implementation
- Verification tasks (V###) are mandatory per Constitution Principle VI
