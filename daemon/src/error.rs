//! Error types for the lofi-daemon.
//!
//! Provides a comprehensive error enum for all daemon operations including
//! model loading, inference, validation, and queue management.

use std::fmt;

/// Error codes matching the JSON-RPC error contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    /// ONNX model files not found at expected path.
    ModelNotFound,
    /// Failed to load ONNX model (corrupt, wrong format, or OOM).
    ModelLoadFailed,
    /// Failed to download model from remote source.
    ModelDownloadFailed,
    /// Model inference failed (numerical instability, OOM).
    ModelInferenceFailed,
    /// Generation queue is full (max 10 pending jobs).
    QueueFull,
    /// Duration outside valid range (5-120 seconds).
    InvalidDuration,
    /// Prompt is empty or exceeds maximum length (1000 chars).
    InvalidPrompt,
}

impl ErrorCode {
    /// Returns the string code for JSON-RPC error responses.
    pub fn as_str(&self) -> &'static str {
        match self {
            ErrorCode::ModelNotFound => "MODEL_NOT_FOUND",
            ErrorCode::ModelLoadFailed => "MODEL_LOAD_FAILED",
            ErrorCode::ModelDownloadFailed => "MODEL_DOWNLOAD_FAILED",
            ErrorCode::ModelInferenceFailed => "MODEL_INFERENCE_FAILED",
            ErrorCode::QueueFull => "QUEUE_FULL",
            ErrorCode::InvalidDuration => "INVALID_DURATION",
            ErrorCode::InvalidPrompt => "INVALID_PROMPT",
        }
    }

    /// Returns the numeric error code for JSON-RPC responses.
    /// Uses negative codes per JSON-RPC 2.0 spec for application errors.
    pub fn as_code(&self) -> i32 {
        match self {
            ErrorCode::ModelNotFound => -32001,
            ErrorCode::ModelLoadFailed => -32002,
            ErrorCode::ModelDownloadFailed => -32003,
            ErrorCode::ModelInferenceFailed => -32004,
            ErrorCode::QueueFull => -32005,
            ErrorCode::InvalidDuration => -32006,
            ErrorCode::InvalidPrompt => -32007,
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
    /// The error code category.
    pub code: ErrorCode,
    /// Human-readable error message.
    pub message: String,
    /// Optional additional context (file path, model name, etc.).
    pub context: Option<String>,
}

impl DaemonError {
    /// Creates a new DaemonError with the given code and message.
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            context: None,
        }
    }

    /// Creates a new DaemonError with additional context.
    pub fn with_context(code: ErrorCode, message: impl Into<String>, context: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            context: Some(context.into()),
        }
    }

    /// Model not found at the specified path.
    pub fn model_not_found(path: impl Into<String>) -> Self {
        let path = path.into();
        Self::with_context(
            ErrorCode::ModelNotFound,
            format!("ONNX model files not found at expected path: {}", path),
            path,
        )
    }

    /// Model failed to load.
    pub fn model_load_failed(reason: impl Into<String>) -> Self {
        Self::new(ErrorCode::ModelLoadFailed, reason)
    }

    /// Model download failed.
    pub fn model_download_failed(reason: impl Into<String>) -> Self {
        Self::new(ErrorCode::ModelDownloadFailed, reason)
    }

    /// Model inference failed.
    pub fn model_inference_failed(reason: impl Into<String>) -> Self {
        Self::new(ErrorCode::ModelInferenceFailed, reason)
    }

    /// Queue is full (max 10 jobs).
    pub fn queue_full() -> Self {
        Self::new(
            ErrorCode::QueueFull,
            "Generation queue is full. Maximum 10 pending jobs allowed.",
        )
    }

    /// Invalid duration (must be 5-120 seconds).
    pub fn invalid_duration(duration: u32) -> Self {
        Self::with_context(
            ErrorCode::InvalidDuration,
            format!("Duration must be between 5 and 120 seconds, got {}", duration),
            duration.to_string(),
        )
    }

    /// Invalid prompt (empty or too long).
    pub fn invalid_prompt(reason: impl Into<String>) -> Self {
        Self::new(ErrorCode::InvalidPrompt, reason)
    }
}

impl fmt::Display for DaemonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)?;
        if let Some(ctx) = &self.context {
            write!(f, " (context: {})", ctx)?;
        }
        Ok(())
    }
}

impl std::error::Error for DaemonError {}

/// Result type alias using DaemonError.
pub type Result<T> = std::result::Result<T, DaemonError>;
