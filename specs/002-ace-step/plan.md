# Implementation Plan: ACE-Step Long-Form Music Generation

**Branch**: `002-ace-step` | **Date**: 2025-12-21 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-ace-step/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Add ACE-Step as a second generation backend to the lofi.nvim daemon, enabling long-form instrumental music generation up to 240 seconds. This requires creating a backend abstraction layer, implementing ONNX model loading for ACE-Step's diffusion architecture, extending the JSON-RPC protocol for backend selection and ACE-Step-specific parameters, and updating cache/progress systems for different model characteristics.

## Technical Context

**Language/Version**: Rust 1.75+ (edition 2021), Lua 5.1 (LuaJIT/Neovim)
**Primary Dependencies**: ort 2.0.0-rc.9 (ONNX Runtime), tokio, serde/serde_json, ndarray, tokenizers
**Storage**: File-based cache at `~/.cache/lofi.nvim/` (tracks + model weights)
**Testing**: cargo test, inline #[cfg(test)] modules
**Target Platform**: macOS (Apple Silicon primary), Linux (CUDA), Windows (CUDA)
**Project Type**: Single project (Rust daemon + Lua plugin)
**Performance Goals**: 120s audio in <15s on RTX 3090 equivalent, model load <30s
**Constraints**: Offline after model download, <7.7GB additional disk for ACE-Step weights, fp32 on macOS
**Scale/Scope**: Single-user Neovim plugin, 2 backends (MusicGen + ACE-Step)
**ACE-Step ONNX Models**: https://huggingface.co/willibrandon/lofi-models/tree/main/ace-step/

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Zero-Latency Experience | PASS | Long-form generation (60-240s tracks) runs async in daemon; Neovim never blocks |
| II. Local & Private | PASS | All inference local; model weights downloaded once from HuggingFace, no telemetry |
| III. Async-First Architecture | PASS | Daemon handles generation; JSON-RPC over stdio; progress via non-blocking notifications |
| IV. Minimal Footprint | PASS | ACE-Step models stored separately (~7.7GB); daemon binary stays <15MB; no new Lua deps |
| V. Simplicity & Composability | PASS | Backend abstraction uses enum dispatch (not trait objects); Lua API unchanged |
| VI. Complete Implementation | PASS | All code production-ready; no stubs; verification tasks required per phase |
| VII. No Fallback Implementations | PASS | If ONNX export fails, defer feature; no procedural fallbacks |

**Gate Result**: PASS - Proceed to Phase 0 research

## Project Structure

### Documentation (this feature)

```text
specs/002-ace-step/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
daemon/
├── src/
│   ├── main.rs                    # Entry point (unchanged)
│   ├── config.rs                  # Add Backend enum + config
│   ├── error.rs                   # Add ACE-Step error codes
│   ├── models/
│   │   ├── mod.rs                 # Module re-exports
│   │   ├── backend.rs             # NEW: Backend enum + dispatch
│   │   ├── loader.rs              # Update: Load either backend
│   │   ├── downloader.rs          # Update: ACE-Step model URLs
│   │   ├── ace_step/              # NEW: ACE-Step implementation
│   │   │   ├── mod.rs
│   │   │   ├── models.rs          # AceStepModels struct
│   │   │   ├── text_encoder.rs    # UMT5 encoder
│   │   │   ├── transformer.rs     # Diffusion transformer
│   │   │   ├── decoder.rs         # MusicDCAE latent decoder
│   │   │   └── vocoder.rs         # Audio vocoder
│   │   └── musicgen/              # Existing MusicGen (restructured)
│   │       ├── mod.rs
│   │       ├── models.rs
│   │       ├── text_encoder.rs
│   │       ├── decoder.rs
│   │       └── audio_codec.rs
│   ├── generation/
│   │   ├── mod.rs
│   │   ├── pipeline.rs            # Update: Backend-aware generation
│   │   ├── queue.rs               # Update: Include backend in jobs
│   │   └── progress.rs            # Update: Variable token rates
│   ├── rpc/
│   │   ├── mod.rs
│   │   ├── server.rs              # Update: Handle cancellation
│   │   ├── methods.rs             # Update: Route to backends
│   │   └── types.rs               # Update: ACE-Step params
│   ├── cache/
│   │   └── tracks.rs              # Update: Backend in track ID
│   └── audio/
│       └── wav.rs                 # Update: 48kHz sample rate support
└── Cargo.toml                     # No new dependencies expected

lua/lofi/
├── init.lua                       # Update: Backend config + generate params
├── daemon.lua                     # No changes expected
├── rpc.lua                        # No changes expected
└── events.lua                     # No changes expected

tests/
├── contract/                      # JSON-RPC schema validation
└── integration/                   # End-to-end daemon tests
```

**Structure Decision**: Existing single-project structure maintained. ACE-Step models organized as submodule parallel to restructured MusicGen module for clear backend separation.

## Complexity Tracking

> **No violations - table empty per constitution compliance**

---

## Post-Design Constitution Re-Check

*Re-evaluated after Phase 1 design completion.*

| Principle | Status | Post-Design Notes |
|-----------|--------|-------------------|
| I. Zero-Latency Experience | PASS | Design confirms: all generation async in daemon, JSON-RPC returns immediately, progress via notifications |
| II. Local & Private | PASS | Design confirms: no network after download, no telemetry, local cache only |
| III. Async-First Architecture | PASS | Design confirms: diffusion loop in daemon, step-by-step cancellation, non-blocking progress |
| IV. Minimal Footprint | PASS | Design confirms: no new Rust dependencies, models in separate cache dir, Lua layer unchanged |
| V. Simplicity & Composability | PASS | Design confirms: enum dispatch (2 variants), flat module structure, no trait complexity |
| VI. Complete Implementation | PASS | Design includes verification tasks, no stubs in quickstart, all code paths defined |
| VII. No Fallback Implementations | PASS | Research explicitly states: if ONNX fails, defer feature entirely |

**Post-Design Gate Result**: PASS - Ready for `/speckit.tasks`

---

## Generated Artifacts

| Artifact | Path | Status |
|----------|------|--------|
| Research | [research.md](./research.md) | Complete |
| Data Model | [data-model.md](./data-model.md) | Complete |
| JSON-RPC Contract | [contracts/jsonrpc.md](./contracts/jsonrpc.md) | Complete |
| Quickstart | [quickstart.md](./quickstart.md) | Complete |
