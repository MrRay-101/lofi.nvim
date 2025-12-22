//! Error types for the lofi-daemon.
//!
//! Defines all error codes and types used throughout the daemon for
//! consistent error handling and reporting.

use std::fmt;

/// Error codes returned by the daemon in error responses.
///
/// These codes are used in JSON-RPC error responses and allow clients
/// to programmatically handle specific error conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    /// ONNX model files not found at expected path.
    /// Trigger: Model files missing from cache directory.
    ModelNotFound,

    /// Failed to load ONNX model into memory.
    /// Trigger: Corrupt file, wrong format, or OOM during load.
    ModelLoadFailed,

    /// Failed to download model from remote source.
    /// Trigger: Network error, disk full during download.
    ModelDownloadFailed,

    /// Model inference failed during generation.
    /// Trigger: Numerical instability, OOM during generation.
    ModelInferenceFailed,

    /// Generation queue is at maximum capacity.
    /// Trigger: 10 pending requests already queued.
    QueueFull,

    /// Requested duration is outside valid range.
    /// Trigger: Duration outside 5-120 second range (MusicGen) or 5-240 (ACE-Step).
    InvalidDuration,

    /// Prompt text is invalid.
    /// Trigger: Empty prompt or exceeds 1000 characters.
    InvalidPrompt,

    /// Requested backend is not installed.
    /// Trigger: ACE-Step model files not downloaded.
    BackendNotInstalled,

    /// Invalid inference steps value.
    /// Trigger: Steps outside valid range (1-200).
    InvalidInferenceSteps,

    /// Invalid guidance scale value.
    /// Trigger: Scale outside valid range (1.0-20.0).
    InvalidGuidanceScale,

    /// Invalid scheduler type.
    /// Trigger: Unknown scheduler name specified.
    InvalidScheduler,

    /// Generation was cancelled.
    /// Trigger: User requested cancellation via cancel RPC.
    GenerationCancelled,
}

impl ErrorCode {
    /// Returns the string representation of the error code.
    pub fn as_str(&self) -> &'static str {
        match self {
            ErrorCode::ModelNotFound => "MODEL_NOT_FOUND",
            ErrorCode::ModelLoadFailed => "MODEL_LOAD_FAILED",
            ErrorCode::ModelDownloadFailed => "MODEL_DOWNLOAD_FAILED",
            ErrorCode::ModelInferenceFailed => "MODEL_INFERENCE_FAILED",
            ErrorCode::QueueFull => "QUEUE_FULL",
            ErrorCode::InvalidDuration => "INVALID_DURATION",
            ErrorCode::InvalidPrompt => "INVALID_PROMPT",
            ErrorCode::BackendNotInstalled => "BACKEND_NOT_INSTALLED",
            ErrorCode::InvalidInferenceSteps => "INVALID_INFERENCE_STEPS",
            ErrorCode::InvalidGuidanceScale => "INVALID_GUIDANCE_SCALE",
            ErrorCode::InvalidScheduler => "INVALID_SCHEDULER",
            ErrorCode::GenerationCancelled => "GENERATION_CANCELLED",
        }
    }

    /// Returns a human-readable description of the error.
    pub fn description(&self) -> &'static str {
        match self {
            ErrorCode::ModelNotFound => "ONNX model files not found at expected path",
            ErrorCode::ModelLoadFailed => "Failed to load ONNX model into memory",
            ErrorCode::ModelDownloadFailed => "Failed to download model from remote source",
            ErrorCode::ModelInferenceFailed => "Model inference failed during generation",
            ErrorCode::QueueFull => "Generation queue is at maximum capacity (10 jobs)",
            ErrorCode::InvalidDuration => "Duration must be between 5 and 240 seconds",
            ErrorCode::InvalidPrompt => "Prompt must be non-empty and at most 1000 characters",
            ErrorCode::BackendNotInstalled => "Requested backend is not installed",
            ErrorCode::InvalidInferenceSteps => "Inference steps must be between 1 and 200",
            ErrorCode::InvalidGuidanceScale => "Guidance scale must be between 1.0 and 20.0",
            ErrorCode::InvalidScheduler => "Unknown scheduler type specified",
            ErrorCode::GenerationCancelled => "Generation was cancelled by user request",
        }
    }

    /// Returns a recovery hint suggesting how to resolve this error.
    pub fn recovery_hint(&self) -> &'static str {
        match self {
            ErrorCode::ModelNotFound => {
                "Run the daemon once with network access to download models automatically, \
                 or manually download from https://huggingface.co/gabotechs/music_gen"
            }
            ErrorCode::ModelLoadFailed => {
                "Check available memory (4GB+ recommended), verify model files are not corrupted, \
                 or delete cache and re-download models"
            }
            ErrorCode::ModelDownloadFailed => {
                "Check internet connection, verify disk space (500MB+ required for MusicGen, 8GB+ for ACE-Step), \
                 or try again later if HuggingFace is unavailable"
            }
            ErrorCode::ModelInferenceFailed => {
                "Try reducing duration, restart the daemon, or check system memory. \
                 If issue persists, try CPU-only mode with LOFI_DEVICE=cpu"
            }
            ErrorCode::QueueFull => {
                "Wait for pending generations to complete before submitting new requests. \
                 Maximum queue size is 10 concurrent requests"
            }
            ErrorCode::InvalidDuration => {
                "Specify a duration between 5-120 seconds for MusicGen or 5-240 seconds for ACE-Step"
            }
            ErrorCode::InvalidPrompt => {
                "Provide a descriptive prompt between 1 and 1000 characters \
                 (e.g., 'lofi hip hop, jazzy piano, relaxing vibes')"
            }
            ErrorCode::BackendNotInstalled => {
                "Download the backend models first using the download_backend RPC method, \
                 or set LOFI_BACKEND to an installed backend"
            }
            ErrorCode::InvalidInferenceSteps => {
                "Specify inference_steps between 1 and 200. Default is 60 for Euler scheduler"
            }
            ErrorCode::InvalidGuidanceScale => {
                "Specify guidance_scale between 1.0 and 20.0. Default is 7.0"
            }
            ErrorCode::InvalidScheduler => {
                "Use one of: 'euler', 'heun', or 'pingpong'"
            }
            ErrorCode::GenerationCancelled => {
                "Generation was stopped as requested. Start a new generation to continue"
            }
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Main error type for daemon operations.
#[derive(Debug)]
pub struct DaemonError {
    /// The error code identifying the type of error.
    pub code: ErrorCode,
    /// Human-readable error message with context.
    pub message: String,
    /// Optional underlying cause of the error.
    pub source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl DaemonError {
    /// Creates a new DaemonError with the given code and message.
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            source: None,
        }
    }

    /// Creates a new DaemonError with an underlying cause.
    pub fn with_source(
        code: ErrorCode,
        message: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self {
            code,
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    /// Creates a MODEL_NOT_FOUND error.
    pub fn model_not_found(path: impl Into<String>) -> Self {
        Self::new(
            ErrorCode::ModelNotFound,
            format!("Model files not found at: {}", path.into()),
        )
    }

    /// Creates a MODEL_LOAD_FAILED error.
    pub fn model_load_failed(reason: impl Into<String>) -> Self {
        Self::new(
            ErrorCode::ModelLoadFailed,
            format!("Failed to load model: {}", reason.into()),
        )
    }

    /// Creates a MODEL_DOWNLOAD_FAILED error.
    pub fn model_download_failed(reason: impl Into<String>) -> Self {
        Self::new(
            ErrorCode::ModelDownloadFailed,
            format!("Failed to download model: {}", reason.into()),
        )
    }

    /// Creates a MODEL_INFERENCE_FAILED error.
    pub fn model_inference_failed(reason: impl Into<String>) -> Self {
        Self::new(
            ErrorCode::ModelInferenceFailed,
            format!("Inference failed: {}", reason.into()),
        )
    }

    /// Creates a QUEUE_FULL error.
    pub fn queue_full() -> Self {
        Self::new(
            ErrorCode::QueueFull,
            "Generation queue is full (maximum 10 pending jobs)",
        )
    }

    /// Creates an INVALID_DURATION error.
    pub fn invalid_duration(duration: u32) -> Self {
        Self::new(
            ErrorCode::InvalidDuration,
            format!(
                "Invalid duration: {} seconds (must be between 5 and 120)",
                duration
            ),
        )
    }

    /// Creates an INVALID_PROMPT error for empty prompts.
    pub fn empty_prompt() -> Self {
        Self::new(ErrorCode::InvalidPrompt, "Prompt cannot be empty")
    }

    /// Creates an INVALID_PROMPT error for prompts that are too long.
    pub fn prompt_too_long(len: usize) -> Self {
        Self::new(
            ErrorCode::InvalidPrompt,
            format!("Prompt too long: {} characters (maximum 1000)", len),
        )
    }

    /// Creates a BACKEND_NOT_INSTALLED error.
    pub fn backend_not_installed(backend: &str) -> Self {
        Self::new(
            ErrorCode::BackendNotInstalled,
            format!("Backend '{}' is not installed", backend),
        )
    }

    /// Creates an INVALID_INFERENCE_STEPS error.
    pub fn invalid_inference_steps(steps: u32) -> Self {
        Self::new(
            ErrorCode::InvalidInferenceSteps,
            format!(
                "Invalid inference steps: {} (must be between 1 and 200)",
                steps
            ),
        )
    }

    /// Creates an INVALID_GUIDANCE_SCALE error.
    pub fn invalid_guidance_scale(scale: f32) -> Self {
        Self::new(
            ErrorCode::InvalidGuidanceScale,
            format!(
                "Invalid guidance scale: {} (must be between 1.0 and 20.0)",
                scale
            ),
        )
    }

    /// Creates an INVALID_SCHEDULER error.
    pub fn invalid_scheduler(scheduler: &str) -> Self {
        Self::new(
            ErrorCode::InvalidScheduler,
            format!(
                "Invalid scheduler: '{}' (must be 'euler', 'heun', or 'pingpong')",
                scheduler
            ),
        )
    }

    /// Creates a GENERATION_CANCELLED error.
    pub fn generation_cancelled() -> Self {
        Self::new(
            ErrorCode::GenerationCancelled,
            "Generation was cancelled by user request",
        )
    }
}

impl fmt::Display for DaemonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {}. Recovery: {}",
            self.code,
            self.message,
            self.code.recovery_hint()
        )
    }
}

impl std::error::Error for DaemonError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_ref()
            .map(|e| e.as_ref() as &(dyn std::error::Error + 'static))
    }
}

/// Result type alias using DaemonError.
pub type Result<T> = std::result::Result<T, DaemonError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_code_as_str() {
        assert_eq!(ErrorCode::ModelNotFound.as_str(), "MODEL_NOT_FOUND");
        assert_eq!(ErrorCode::ModelLoadFailed.as_str(), "MODEL_LOAD_FAILED");
        assert_eq!(
            ErrorCode::ModelDownloadFailed.as_str(),
            "MODEL_DOWNLOAD_FAILED"
        );
        assert_eq!(
            ErrorCode::ModelInferenceFailed.as_str(),
            "MODEL_INFERENCE_FAILED"
        );
        assert_eq!(ErrorCode::QueueFull.as_str(), "QUEUE_FULL");
        assert_eq!(ErrorCode::InvalidDuration.as_str(), "INVALID_DURATION");
        assert_eq!(ErrorCode::InvalidPrompt.as_str(), "INVALID_PROMPT");
        assert_eq!(
            ErrorCode::BackendNotInstalled.as_str(),
            "BACKEND_NOT_INSTALLED"
        );
        assert_eq!(
            ErrorCode::InvalidInferenceSteps.as_str(),
            "INVALID_INFERENCE_STEPS"
        );
        assert_eq!(
            ErrorCode::InvalidGuidanceScale.as_str(),
            "INVALID_GUIDANCE_SCALE"
        );
        assert_eq!(ErrorCode::InvalidScheduler.as_str(), "INVALID_SCHEDULER");
        assert_eq!(
            ErrorCode::GenerationCancelled.as_str(),
            "GENERATION_CANCELLED"
        );
    }

    #[test]
    fn error_code_recovery_hints_not_empty() {
        // Ensure all error codes have non-empty recovery hints
        assert!(!ErrorCode::ModelNotFound.recovery_hint().is_empty());
        assert!(!ErrorCode::ModelLoadFailed.recovery_hint().is_empty());
        assert!(!ErrorCode::ModelDownloadFailed.recovery_hint().is_empty());
        assert!(!ErrorCode::ModelInferenceFailed.recovery_hint().is_empty());
        assert!(!ErrorCode::QueueFull.recovery_hint().is_empty());
        assert!(!ErrorCode::InvalidDuration.recovery_hint().is_empty());
        assert!(!ErrorCode::InvalidPrompt.recovery_hint().is_empty());
        assert!(!ErrorCode::BackendNotInstalled.recovery_hint().is_empty());
        assert!(!ErrorCode::InvalidInferenceSteps.recovery_hint().is_empty());
        assert!(!ErrorCode::InvalidGuidanceScale.recovery_hint().is_empty());
        assert!(!ErrorCode::InvalidScheduler.recovery_hint().is_empty());
        assert!(!ErrorCode::GenerationCancelled.recovery_hint().is_empty());
    }

    #[test]
    fn daemon_error_display() {
        let err = DaemonError::invalid_duration(200);
        assert!(err.to_string().contains("INVALID_DURATION"));
        assert!(err.to_string().contains("200"));
        assert!(err.to_string().contains("Recovery:"));
    }

    #[test]
    fn ace_step_error_constructors() {
        let err = DaemonError::backend_not_installed("ace_step");
        assert_eq!(err.code, ErrorCode::BackendNotInstalled);
        assert!(err.message.contains("ace_step"));

        let err = DaemonError::invalid_inference_steps(300);
        assert_eq!(err.code, ErrorCode::InvalidInferenceSteps);
        assert!(err.message.contains("300"));

        let err = DaemonError::invalid_guidance_scale(25.0);
        assert_eq!(err.code, ErrorCode::InvalidGuidanceScale);
        assert!(err.message.contains("25"));

        let err = DaemonError::invalid_scheduler("unknown");
        assert_eq!(err.code, ErrorCode::InvalidScheduler);
        assert!(err.message.contains("unknown"));

        let err = DaemonError::generation_cancelled();
        assert_eq!(err.code, ErrorCode::GenerationCancelled);
    }
}
