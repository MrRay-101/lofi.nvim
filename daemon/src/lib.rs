//! lofi-daemon: AI music generation daemon using MusicGen ONNX.
//!
//! This library provides the core functionality for the lofi-daemon,
//! an AI-powered music generation backend that uses MusicGen-small
//! with ONNX Runtime for inference.
//!
//! # Modules
//!
//! - [`config`] - Daemon configuration (device, paths, threads)
//! - [`error`] - Error types and result aliases
//! - [`types`] - Core domain types (Track, GenerationJob, ModelConfig)
//!
//! # Example
//!
//! ```rust,ignore
//! use lofi_daemon::config::DaemonConfig;
//! use lofi_daemon::types::{Track, GenerationJob, Priority};
//!
//! // Create a configuration with default paths
//! let config = DaemonConfig::default();
//!
//! // Check if models are available
//! if !config.models_exist() {
//!     println!("Missing models: {:?}", config.missing_models());
//! }
//! ```

pub mod config;
pub mod error;
pub mod types;

// Re-export commonly used types at crate root for convenience
pub use config::{DaemonConfig, Device};
pub use error::{DaemonError, ErrorCode, Result};
pub use types::{compute_track_id, GenerationJob, JobStatus, ModelConfig, Priority, Track};
