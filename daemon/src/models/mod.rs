//! Model components for music generation backends.
//!
//! This module contains:
//! - [`musicgen`]: MusicGen ONNX model wrappers for 30-second generation
//! - [`ace_step`]: ACE-Step ONNX model wrappers for long-form generation (to be implemented)
//! - [`backend`]: Backend abstraction for switching between models
//! - [`loader`]: Unified model loading for all backends
//! - [`device`]: Device detection and execution provider selection
//! - [`downloader`]: Model download and management

pub mod ace_step;
pub mod backend;
pub mod device;
pub mod downloader;
pub mod loader;
pub mod musicgen;

// Re-export commonly used types from submodules
pub use backend::{AceStepModels, Backend, LoadedModels};
pub use device::{detect_available_providers, get_device_name, get_providers, AvailableProvider};
pub use downloader::ensure_models;
pub use loader::{check_backend_available, detect_available_backends, load_backend};
pub use musicgen::{
    check_models, detect_model_version, generate_model_version, load_sessions,
    load_sessions_with_device, DelayPatternMaskIds, Logits, MusicGenAudioCodec, MusicGenDecoder,
    MusicGenModels, MusicGenTextEncoder, DEFAULT_GUIDANCE_SCALE, DEFAULT_TOP_K, MODEL_URLS,
    REQUIRED_MODEL_FILES,
};
