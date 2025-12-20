//! Core types for the lofi-daemon.
//!
//! This module re-exports all domain entities used throughout the daemon:
//! - [`Track`] - A generated audio file stored in cache
//! - [`GenerationJob`] - A music generation request with lifecycle tracking
//! - [`ModelConfig`] - MusicGen model configuration parameters
//!
//! The types module also re-exports error types from the error module
//! for convenience.

mod config;
mod job;
mod track;

pub use config::ModelConfig;
pub use job::{GenerationJob, JobStatus, Priority};
pub use track::{compute_track_id, Track};

// Re-export error types for convenience
pub use crate::error::{DaemonError, ErrorCode, Result};
