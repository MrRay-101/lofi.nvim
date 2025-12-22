//! Audio generation module.
//!
//! Provides the generation pipeline for MusicGen and ACE-Step backends.

pub mod pipeline;
pub mod progress;
pub mod queue;

// Re-export commonly used items
pub use pipeline::{
    estimate_generation_time, estimate_samples, generate, generate_ace_step, generate_with_models,
    generate_with_progress,
};
pub use progress::ProgressTracker;
pub use queue::{GenerationQueue, JobResult, QueueFullError, QueueProcessor, MAX_QUEUE_SIZE};
