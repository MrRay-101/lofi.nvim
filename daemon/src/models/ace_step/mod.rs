//! ACE-Step model components for long-form music generation.
//!
//! This module provides ONNX model wrappers and diffusion pipeline
//! for ACE-Step, a diffusion-based music generation model capable
//! of generating up to 240 seconds of instrumental audio.
//!
//! ## Components
//!
//! - [`models`]: Model loader for all ACE-Step ONNX components
//! - [`text_encoder`]: UMT5 text encoder for prompt conditioning
//! - [`transformer`]: Diffusion transformer for noise prediction
//! - [`decoder`]: DCAE latent decoder for mel-spectrogram generation
//! - [`vocoder`]: ADaMoSHiFiGAN vocoder for audio synthesis
//! - [`scheduler`]: Diffusion schedulers (Euler, Heun, PingPong)
//! - [`guidance`]: Classifier-free guidance implementation
//! - [`latent`]: Latent space initialization and utilities
//! - [`generate`]: Complete generation pipeline

pub mod decoder;
pub mod generate;
pub mod guidance;
pub mod latent;
pub mod models;
pub mod scheduler;
pub mod text_encoder;
pub mod transformer;
pub mod vocoder;

// Re-export commonly used types
pub use generate::{generate, generate_with_progress, GenerationParams};
pub use guidance::{apply_cfg, DEFAULT_GUIDANCE_SCALE, MAX_GUIDANCE_SCALE, MIN_GUIDANCE_SCALE};
pub use latent::{calculate_frame_length, estimate_duration, initialize_latent};
pub use models::{check_models, load_session, AceStepModels, MODEL_URLS, REQUIRED_FILES};
pub use scheduler::{create_scheduler, EulerScheduler, SchedulerType};
