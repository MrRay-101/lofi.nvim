//! Audio generation module.
//!
//! Provides the generation pipeline for MusicGen.

pub mod pipeline;
pub mod progress;

// Re-export commonly used items
pub use pipeline::{
    estimate_generation_time, estimate_samples, generate, generate_with_models,
    generate_with_progress,
};
pub use progress::ProgressTracker;
