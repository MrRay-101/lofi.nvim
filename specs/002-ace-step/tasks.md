# Tasks: ACE-Step Long-Form Music Generation

**Input**: Design documents from `/specs/002-ace-step/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/jsonrpc.md, quickstart.md

**Tests**: No tests explicitly requested in specification. Implementation tasks only.

**Organization**: Tasks grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4, US5)
- Include exact file paths in descriptions

## Path Conventions

Based on plan.md project structure:
- Rust daemon: `daemon/src/`
- Lua plugin: `lua/lofi/`
- New ACE-Step module: `daemon/src/models/ace_step/`
- Restructured MusicGen: `daemon/src/models/musicgen/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization, ONNX export preparation, and module restructuring

- [X] T001 Create ONNX export environment in /Users/brandon/src/ACE-Step/.venv-export
- [X] T002 [P] Export UMT5 text encoder to text_encoder.onnx using export script
- [X] T003 [P] Export ACEStepTransformer to transformer.onnx using export script
- [X] T004 [P] Export MusicDCAE decoder to dcae_decoder.onnx using export script
- [X] T005 [P] Export ADaMoSHiFiGAN vocoder to vocoder.onnx using export script
- [X] T006 Verify all ONNX exports load correctly with onnxruntime
- [X] T007 Create daemon/src/models/ace_step/ directory structure
- [X] T008 [P] Restructure existing MusicGen code into daemon/src/models/musicgen/mod.rs
- [X] T009 [P] Create daemon/src/models/musicgen/models.rs from existing loader.rs
- [X] T010 [P] Create daemon/src/models/musicgen/text_encoder.rs from existing code
- [X] T011 [P] Create daemon/src/models/musicgen/decoder.rs from existing code
- [X] T012 [P] Create daemon/src/models/musicgen/audio_codec.rs from existing code
- [X] T013 Update daemon/src/models/mod.rs to re-export restructured modules

### Phase 1 Verification (MANDATORY - Principle VI)

- [X] V001 [VERIFY] Run `cargo build` in daemon/ - must succeed with zero errors
- [X] V002 [VERIFY] Run `grep -rn "TODO\|FIXME" daemon/src/` - must return empty
- [X] V003 [VERIFY] List all new/moved files and confirm each is imported/used
- [X] V004 [VERIFY] Existing MusicGen functionality still works after restructure

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Backend abstraction layer that MUST be complete before ANY user story can be implemented

**CRITICAL**: No user story work can begin until this phase is complete

- [ ] T014 Create Backend enum (MusicGen, AceStep) in daemon/src/models/backend.rs
- [ ] T015 Create LoadedModels enum with MusicGen and AceStep variants in daemon/src/models/backend.rs
- [ ] T016 Add BackendType enum to daemon/src/config.rs with default_backend field
- [ ] T017 Add ace_step_model_dir path to DaemonConfig in daemon/src/config.rs
- [ ] T018 Add ACE-Step-specific config defaults (inference_steps, scheduler, guidance_scale) to daemon/src/config.rs
- [ ] T019 [P] Add ACE-Step error codes to daemon/src/error.rs (BACKEND_NOT_INSTALLED, INVALID_DURATION, etc.)
- [ ] T020 Update daemon/src/models/loader.rs to support loading either backend
- [ ] T021 Update daemon/src/cache/tracks.rs to include backend in track ID hash
- [ ] T022 Update daemon/src/audio/wav.rs to support 48kHz sample rate output

### Phase 2 Verification (MANDATORY - Principle VI)

- [ ] V005 [VERIFY] Run `cargo build` in daemon/ - must succeed with zero errors
- [ ] V006 [VERIFY] Run `grep -rn "TODO\|FIXME" daemon/src/` - must return empty
- [ ] V007 [VERIFY] Run `cargo test` - all existing tests pass
- [ ] V008 [VERIFY] Backend enum compiles and MusicGen still loads correctly

---

## Phase 3: User Story 1 - Generate Long-Form Instrumental Track (Priority: P1) MVP

**Goal**: Generate seamless audio from 5-240 seconds using ACE-Step diffusion model

**Independent Test**: Generate a 120-second track with prompt "lofi hip hop, chill beats" and verify valid audio output

### Implementation for User Story 1

- [ ] T023 [P] [US1] Create AceStepModels struct in daemon/src/models/ace_step/models.rs
- [ ] T024 [P] [US1] Implement AceStepModels::load() for loading all 4 ONNX components in daemon/src/models/ace_step/models.rs
- [ ] T025 [P] [US1] Create UMT5 text encoder wrapper in daemon/src/models/ace_step/text_encoder.rs
- [ ] T026 [P] [US1] Create diffusion transformer wrapper in daemon/src/models/ace_step/transformer.rs
- [ ] T027 [P] [US1] Create DCAE latent decoder wrapper in daemon/src/models/ace_step/decoder.rs
- [ ] T028 [P] [US1] Create vocoder wrapper in daemon/src/models/ace_step/vocoder.rs
- [ ] T029 [US1] Create EulerScheduler in daemon/src/models/ace_step/scheduler.rs
- [ ] T030 [US1] Implement scheduler timestep and sigma calculation in daemon/src/models/ace_step/scheduler.rs
- [ ] T031 [US1] Implement scheduler step() method for latent update in daemon/src/models/ace_step/scheduler.rs
- [ ] T032 [US1] Implement apply_cfg() for classifier-free guidance in daemon/src/models/ace_step/guidance.rs
- [ ] T033 [US1] Implement latent initialization with random noise in daemon/src/models/ace_step/latent.rs
- [ ] T034 [US1] Implement frame_length calculation from duration in daemon/src/models/ace_step/latent.rs
- [ ] T035 [US1] Create generate() function implementing diffusion loop in daemon/src/models/ace_step/generate.rs
- [ ] T036 [US1] Implement prompt encoding step in generate() in daemon/src/models/ace_step/generate.rs
- [ ] T037 [US1] Implement diffusion step loop with scheduler in daemon/src/models/ace_step/generate.rs
- [ ] T038 [US1] Implement latent decoding to mel-spectrogram in daemon/src/models/ace_step/generate.rs
- [ ] T039 [US1] Implement vocoding from mel to waveform in daemon/src/models/ace_step/generate.rs
- [ ] T040 [US1] Add 44.1kHz to 48kHz resampling using rubato crate in daemon/src/audio/resample.rs
- [ ] T041 [US1] Wire generate_ace_step() into daemon/src/generation/pipeline.rs
- [ ] T042 [US1] Update daemon/src/models/ace_step/mod.rs to export all submodules
- [ ] T043 [US1] Force fp32 precision on macOS Apple Silicon in daemon/src/models/ace_step/models.rs

### User Story 1 Verification (MANDATORY - Principle VI)

- [ ] V009 [VERIFY] Run `cargo build` in daemon/ - must succeed with zero errors
- [ ] V010 [VERIFY] Run `grep -rn "TODO\|FIXME" daemon/src/models/ace_step/` - must return empty
- [ ] V011 [VERIFY] All US1 files imported in mod.rs and used in generate.rs
- [ ] V012 [VERIFY] Run daemon with --backend ace_step --prompt "lofi beats" --duration 30

---

## Phase 4: User Story 2 - Switch Between Generation Backends (Priority: P2)

**Goal**: Allow users to select MusicGen or ACE-Step backend via configuration

**Independent Test**: Switch backend config and verify correct model used for generation

### Implementation for User Story 2

- [ ] T044 [US2] Implement LoadedModels::generate() dispatch in daemon/src/models/backend.rs
- [ ] T045 [US2] Add backend parameter to GenerateParams in daemon/src/rpc/types.rs
- [ ] T046 [US2] Update handle_generate() to select backend in daemon/src/rpc/methods.rs
- [ ] T047 [US2] Add get_backends RPC method in daemon/src/rpc/methods.rs
- [ ] T048 [US2] Implement BackendInfo struct with status and capabilities in daemon/src/rpc/types.rs
- [ ] T049 [US2] Update daemon startup to detect available backends in daemon/src/main.rs
- [ ] T050 [US2] Add backend field to Lua config in lua/lofi/init.lua
- [ ] T051 [US2] Update Lua generate() to pass backend parameter in lua/lofi/init.lua

### User Story 2 Verification (MANDATORY - Principle VI)

- [ ] V013 [VERIFY] Run `cargo build` in daemon/ - must succeed with zero errors
- [ ] V014 [VERIFY] Run `grep -rn "TODO\|FIXME" daemon/src/rpc/` - must return empty
- [ ] V015 [VERIFY] Lua config accepts backend = "ace_step" or "musicgen"
- [ ] V016 [VERIFY] get_backends RPC returns correct status for both backends

---

## Phase 5: User Story 3 - Download and Setup ACE-Step Model (Priority: P2)

**Goal**: Allow users to download ACE-Step model weights via RPC command

**Independent Test**: Run download_backend command and verify all components downloaded to cache

### Implementation for User Story 3

- [ ] T052 [US3] Add ACE-Step model URLs (https://huggingface.co/willibrandon/lofi-models/resolve/main/ace-step/) to daemon/src/models/downloader.rs
- [ ] T053 [US3] Implement download_ace_step_models() in daemon/src/models/downloader.rs
- [ ] T054 [US3] Add download_backend RPC method in daemon/src/rpc/methods.rs
- [ ] T055 [US3] Implement download progress tracking in daemon/src/models/downloader.rs
- [ ] T056 [US3] Add download_progress notification in daemon/src/rpc/methods.rs
- [ ] T057 [US3] Handle partial download resume in daemon/src/models/downloader.rs
- [ ] T058 [US3] Update BackendStatus enum with Downloading state in daemon/src/models/backend.rs

### User Story 3 Verification (MANDATORY - Principle VI)

- [ ] V017 [VERIFY] Run `cargo build` in daemon/ - must succeed with zero errors
- [ ] V018 [VERIFY] Run `grep -rn "TODO\|FIXME" daemon/src/models/downloader.rs` - must return empty
- [ ] V019 [VERIFY] download_backend RPC starts download and emits progress
- [ ] V020 [VERIFY] Models downloaded to ~/.cache/lofi.nvim/ace-step/

---

## Phase 6: User Story 4 - Monitor Generation Progress (Priority: P3)

**Goal**: Provide real-time progress updates during long-form generation

**Independent Test**: Start generation and verify progress notifications with percentage, step, and ETA

### Implementation for User Story 4

- [ ] T059 [US4] Update daemon/src/generation/progress.rs to support variable step counts
- [ ] T060 [US4] Implement step-based ETA calculation for diffusion in daemon/src/generation/progress.rs
- [ ] T061 [US4] Add current_step and total_steps to progress notification in daemon/src/rpc/types.rs
- [ ] T062 [US4] Emit progress at each diffusion step in daemon/src/models/ace_step/generate.rs
- [ ] T063 [US4] Update generation_complete notification with backend info in daemon/src/rpc/methods.rs

### User Story 4 Verification (MANDATORY - Principle VI)

- [ ] V021 [VERIFY] Run `cargo build` in daemon/ - must succeed with zero errors
- [ ] V022 [VERIFY] Progress notifications include step number and ETA
- [ ] V023 [VERIFY] Progress percentage accurate within 10% of actual completion

---

## Phase 7: User Story 5 - Configure Generation Parameters (Priority: P3)

**Goal**: Allow users to customize inference_steps, scheduler, and guidance_scale

**Independent Test**: Generate with non-default parameters and verify configuration applied

### Implementation for User Story 5

- [ ] T064 [P] [US5] Add inference_steps parameter to GenerateParams in daemon/src/rpc/types.rs
- [ ] T065 [P] [US5] Add scheduler parameter to GenerateParams in daemon/src/rpc/types.rs
- [ ] T066 [P] [US5] Add guidance_scale parameter to GenerateParams in daemon/src/rpc/types.rs
- [ ] T067 [US5] Implement HeunScheduler in daemon/src/models/ace_step/scheduler.rs
- [ ] T068 [US5] Implement PingPongScheduler in daemon/src/models/ace_step/scheduler.rs
- [ ] T069 [US5] Create SchedulerType enum and factory function in daemon/src/models/ace_step/scheduler.rs
- [ ] T070 [US5] Validate parameter ranges in handle_generate() in daemon/src/rpc/methods.rs
- [ ] T071 [US5] Update Lua generate() to accept inference_steps, scheduler, guidance_scale in lua/lofi/init.lua

### User Story 5 Verification (MANDATORY - Principle VI)

- [ ] V024 [VERIFY] Run `cargo build` in daemon/ - must succeed with zero errors
- [ ] V025 [VERIFY] Invalid parameters rejected with clear error messages
- [ ] V026 [VERIFY] Heun and PingPong schedulers produce valid output

---

## Phase 8: Cancellation Support (Cross-Cutting)

**Purpose**: Enable cancellation of in-progress generation (FR-015)

- [ ] T072 Add cancellation_token (AtomicBool) to generation job in daemon/src/generation/queue.rs
- [ ] T073 Check cancellation_token between diffusion steps in daemon/src/models/ace_step/generate.rs
- [ ] T074 Add cancel RPC method in daemon/src/rpc/methods.rs
- [ ] T075 Add generation_cancelled notification in daemon/src/rpc/methods.rs
- [ ] T076 Update Lua to expose cancel function in lua/lofi/init.lua

### Phase 8 Verification (MANDATORY - Principle VI)

- [ ] V027 [VERIFY] Run `cargo build` in daemon/ - must succeed with zero errors
- [ ] V028 [VERIFY] Cancel RPC stops in-progress generation within one step
- [ ] V029 [VERIFY] Cancelled notification includes step where cancelled

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Final cleanup and integration verification

- [ ] T077 [P] Validate all error codes in daemon/src/error.rs match contracts/jsonrpc.md
- [ ] T078 [P] Ensure all RPC methods match contracts/jsonrpc.md specification
- [ ] T079 Run quickstart.md verification steps
- [ ] T080 Verify seed reproducibility (same seed = same output)
- [ ] T081 [P] Add rubato dependency to daemon/Cargo.toml if not present
- [ ] T082 Update daemon version to 0.2.0 in daemon/Cargo.toml

### Final Verification (MANDATORY - Principle VI)

- [ ] VFINAL-1 [VERIFY] Run `cargo build --release` - must succeed with zero errors/warnings
- [ ] VFINAL-2 [VERIFY] Run `grep -rn "TODO\|FIXME" daemon/src/ lua/lofi/` - must return empty
- [ ] VFINAL-3 [VERIFY] Confirm ALL written files are imported and used
- [ ] VFINAL-4 [VERIFY] Confirm ALL written functions are called
- [ ] VFINAL-5 [VERIFY] Full integration test: generate 120s track via Lua API

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies - includes ONNX export (external to Rust) and module restructure
- **Phase 2 (Foundational)**: Depends on Phase 1 - BLOCKS all user stories
- **Phase 3 (US1)**: Depends on Phase 2 - Core generation capability
- **Phase 4 (US2)**: Depends on Phase 2 - Can parallel with US1 after foundation
- **Phase 5 (US3)**: Depends on Phase 2 - Can parallel with US1/US2
- **Phase 6 (US4)**: Depends on US1 (needs generation loop to hook into)
- **Phase 7 (US5)**: Depends on US1 (needs scheduler implementation to extend)
- **Phase 8 (Cancellation)**: Depends on US1 (needs generation loop)
- **Phase 9 (Polish)**: Depends on all user stories complete

### User Story Dependencies

```
Phase 2 (Foundational)
     ├─────────┬─────────┐
     ▼         ▼         ▼
   US1 (P1)  US2 (P2)  US3 (P2)  [can run in parallel]
     │
     ├─────────┬─────────┐
     ▼         ▼         ▼
   US4 (P3)  US5 (P3)  Cancel   [depend on US1]
     │         │         │
     └─────────┴─────────┘
              ▼
         Phase 9 (Polish)
```

### Within Each User Story

- Model loading before generation logic
- Core implementation before RPC integration
- Rust daemon before Lua plugin updates

### Parallel Opportunities

**Phase 1 (Setup)**:
- T002, T003, T004, T005: ONNX exports (different models)
- T008-T012: MusicGen restructure (different files)

**Phase 3 (US1)**:
- T023-T028: Model wrappers (different files)

**Phase 4-5 (US2/US3)**:
- These entire phases can run in parallel after Phase 2

**Phase 7 (US5)**:
- T064, T065, T066: Parameter additions (same file but independent fields)
- T067, T068: Scheduler implementations (same file but independent structs)

---

## Parallel Example: User Story 1

```bash
# Launch all model wrappers together:
Task: "Create UMT5 text encoder wrapper in daemon/src/models/ace_step/text_encoder.rs"
Task: "Create diffusion transformer wrapper in daemon/src/models/ace_step/transformer.rs"
Task: "Create DCAE latent decoder wrapper in daemon/src/models/ace_step/decoder.rs"
Task: "Create vocoder wrapper in daemon/src/models/ace_step/vocoder.rs"

# Then sequentially:
Task: "Create EulerScheduler in daemon/src/models/ace_step/scheduler.rs"
Task: "Create generate() function implementing diffusion loop"
```

---

## Implementation Strategy

### Complete Implementation (Principle VI)

1. Complete ALL phases in sequence without stopping
2. Implement ALL user stories fully - no partial implementations
3. Write production-ready code for every task
4. Do not stop for validation checkpoints - complete everything first
5. Zero TODOs, zero stubs, zero placeholders in any code

### MVP Scope

**Minimum Viable Product**: Complete through User Story 1 (Phase 3)
- ONNX models exported and verified
- Backend abstraction in place
- ACE-Step generation working end-to-end
- Basic 60-step Euler scheduler

This enables generating 5-240 second instrumental audio with default parameters.

### Incremental Delivery

1. **MVP**: US1 complete - basic generation works
2. **Backend Selection**: Add US2 - users can switch backends
3. **Self-Service Setup**: Add US3 - users can download models
4. **User Experience**: Add US4/US5 - progress and parameters
5. **Full Feature**: Add cancellation and polish

---

## Notes

- ONNX export (Phase 1, T001-T006) requires Python environment and is external to Rust build
- Model files (~11.5GB) hosted at https://huggingface.co/willibrandon/lofi-models/tree/main/ace-step/
  - `text_encoder.onnx` (1.13 GB) - UMT5 text encoder
  - `transformer_encoder.onnx` (424 MB) - ACE-Step encoder
  - `transformer_decoder.onnx` (35.7 MB) + `transformer_decoder_weights.bin` (9.97 GB) - ACE-Step decoder
  - `dcae_decoder.onnx` (317 MB) - MusicDCAE latent decoder
  - `vocoder.onnx` (412 MB) - ADaMoSHiFiGAN vocoder
  - `tokenizer.json` (16.8 MB) - UMT5 tokenizer
- macOS requires fp32 precision - detected automatically
- Existing MusicGen tests should continue passing after restructure
- [P] tasks = different files, no dependencies within phase
- [Story] label maps task to specific user story for traceability
