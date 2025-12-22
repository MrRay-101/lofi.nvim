//! JSON-RPC types for the daemon protocol.
//!
//! Implements the contracts defined in contracts/generate.json, notifications.json, and errors.json.

use serde::{Deserialize, Serialize};

use crate::models::Backend;

/// JSON-RPC version constant.
pub const JSONRPC_VERSION: &str = "2.0";

/// A JSON-RPC request ID.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum RequestId {
    Integer(i64),
    String(String),
}

impl From<i64> for RequestId {
    fn from(id: i64) -> Self {
        RequestId::Integer(id)
    }
}

impl From<String> for RequestId {
    fn from(id: String) -> Self {
        RequestId::String(id)
    }
}

/// A JSON-RPC request wrapper.
#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub method: String,
    pub id: RequestId,
    #[serde(default)]
    pub params: serde_json::Value,
}

/// A JSON-RPC response wrapper.
#[derive(Debug, Serialize)]
pub struct JsonRpcResponse<T: Serialize> {
    pub jsonrpc: &'static str,
    pub id: RequestId,
    pub result: T,
}

impl<T: Serialize> JsonRpcResponse<T> {
    pub fn new(id: RequestId, result: T) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION,
            id,
            result,
        }
    }
}

/// A JSON-RPC error response.
#[derive(Debug, Serialize)]
pub struct JsonRpcErrorResponse {
    pub jsonrpc: &'static str,
    pub id: Option<RequestId>,
    pub error: JsonRpcError,
}

impl JsonRpcErrorResponse {
    pub fn new(id: Option<RequestId>, error: JsonRpcError) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION,
            id,
            error,
        }
    }
}

/// A JSON-RPC error object.
#[derive(Debug, Serialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<JsonRpcErrorData>,
}

/// Extended error data for application-specific errors.
#[derive(Debug, Serialize)]
pub struct JsonRpcErrorData {
    pub error_code: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

impl JsonRpcError {
    /// Creates a parse error (-32700).
    pub fn parse_error(message: impl Into<String>) -> Self {
        Self {
            code: -32700,
            message: message.into(),
            data: None,
        }
    }

    /// Creates an invalid request error (-32600).
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            code: -32600,
            message: message.into(),
            data: None,
        }
    }

    /// Creates a method not found error (-32601).
    pub fn method_not_found(method: &str) -> Self {
        Self {
            code: -32601,
            message: format!("Method not found: {}", method),
            data: None,
        }
    }

    /// Creates an invalid params error (-32602).
    pub fn invalid_params(message: impl Into<String>) -> Self {
        Self {
            code: -32602,
            message: message.into(),
            data: None,
        }
    }

    /// Creates an internal error (-32603).
    pub fn internal_error(message: impl Into<String>) -> Self {
        Self {
            code: -32603,
            message: message.into(),
            data: None,
        }
    }

    /// Creates a model not found error (-32000).
    pub fn model_not_found(details: impl Into<String>) -> Self {
        Self {
            code: -32000,
            message: "Model not found".to_string(),
            data: Some(JsonRpcErrorData {
                error_code: "MODEL_NOT_FOUND".to_string(),
                details: Some(details.into()),
            }),
        }
    }

    /// Creates a model load failed error (-32001).
    pub fn model_load_failed(details: impl Into<String>) -> Self {
        Self {
            code: -32001,
            message: "Model load failed".to_string(),
            data: Some(JsonRpcErrorData {
                error_code: "MODEL_LOAD_FAILED".to_string(),
                details: Some(details.into()),
            }),
        }
    }

    /// Creates a model download failed error (-32002).
    pub fn model_download_failed(details: impl Into<String>) -> Self {
        Self {
            code: -32002,
            message: "Model download failed".to_string(),
            data: Some(JsonRpcErrorData {
                error_code: "MODEL_DOWNLOAD_FAILED".to_string(),
                details: Some(details.into()),
            }),
        }
    }

    /// Creates a model inference failed error (-32003).
    pub fn model_inference_failed(details: impl Into<String>) -> Self {
        Self {
            code: -32003,
            message: "Model inference failed".to_string(),
            data: Some(JsonRpcErrorData {
                error_code: "MODEL_INFERENCE_FAILED".to_string(),
                details: Some(details.into()),
            }),
        }
    }

    /// Creates a queue full error (-32004).
    pub fn queue_full(current_size: usize) -> Self {
        Self {
            code: -32004,
            message: "Queue full".to_string(),
            data: Some(JsonRpcErrorData {
                error_code: "QUEUE_FULL".to_string(),
                details: Some(format!("Maximum 10 pending requests. Current queue: {}", current_size)),
            }),
        }
    }

    /// Creates an invalid duration error (-32005).
    pub fn invalid_duration(duration: i64) -> Self {
        Self {
            code: -32005,
            message: "Invalid duration".to_string(),
            data: Some(JsonRpcErrorData {
                error_code: "INVALID_DURATION".to_string(),
                details: Some(format!(
                    "Duration {} is outside valid range of 5-120 seconds",
                    duration
                )),
            }),
        }
    }

    /// Creates an invalid prompt error (-32006).
    pub fn invalid_prompt(reason: impl Into<String>) -> Self {
        Self {
            code: -32006,
            message: "Invalid prompt".to_string(),
            data: Some(JsonRpcErrorData {
                error_code: "INVALID_PROMPT".to_string(),
                details: Some(reason.into()),
            }),
        }
    }

    /// Creates an invalid backend error (-32007).
    pub fn invalid_backend(backend: impl Into<String>) -> Self {
        Self {
            code: -32007,
            message: "Invalid backend".to_string(),
            data: Some(JsonRpcErrorData {
                error_code: "INVALID_BACKEND".to_string(),
                details: Some(format!(
                    "Unknown backend: '{}'. Valid options: 'musicgen', 'ace_step'",
                    backend.into()
                )),
            }),
        }
    }

    /// Creates a backend not installed error (-32008).
    pub fn backend_not_installed(backend: &Backend) -> Self {
        Self {
            code: -32008,
            message: "Backend not installed".to_string(),
            data: Some(JsonRpcErrorData {
                error_code: "BACKEND_NOT_INSTALLED".to_string(),
                details: Some(format!(
                    "Backend '{}' is not installed. Use download_backend to download it.",
                    backend.as_str()
                )),
            }),
        }
    }

    /// Creates an invalid duration error for a specific backend (-32005).
    pub fn invalid_duration_for_backend(duration: i64, backend: Backend) -> Self {
        Self {
            code: -32005,
            message: "Invalid duration".to_string(),
            data: Some(JsonRpcErrorData {
                error_code: "INVALID_DURATION".to_string(),
                details: Some(format!(
                    "Duration {} is outside valid range of {}-{} seconds for {} backend",
                    duration,
                    backend.min_duration_sec(),
                    backend.max_duration_sec(),
                    backend.as_str()
                )),
            }),
        }
    }

    /// Creates an invalid inference steps error (-32009).
    pub fn invalid_inference_steps(steps: u32) -> Self {
        Self {
            code: -32009,
            message: "Invalid inference steps".to_string(),
            data: Some(JsonRpcErrorData {
                error_code: "INVALID_INFERENCE_STEPS".to_string(),
                details: Some(format!(
                    "Inference steps {} is outside valid range of 1-200",
                    steps
                )),
            }),
        }
    }

    /// Creates an invalid guidance scale error (-32010).
    pub fn invalid_guidance_scale(scale: f32) -> Self {
        Self {
            code: -32010,
            message: "Invalid guidance scale".to_string(),
            data: Some(JsonRpcErrorData {
                error_code: "INVALID_GUIDANCE_SCALE".to_string(),
                details: Some(format!(
                    "Guidance scale {} is outside valid range of 1.0-30.0",
                    scale
                )),
            }),
        }
    }

    /// Creates an invalid scheduler error (-32011).
    pub fn invalid_scheduler(scheduler: impl Into<String>) -> Self {
        Self {
            code: -32011,
            message: "Invalid scheduler".to_string(),
            data: Some(JsonRpcErrorData {
                error_code: "INVALID_SCHEDULER".to_string(),
                details: Some(format!(
                    "Unknown scheduler: '{}'. Valid options: 'euler', 'heun', 'pingpong'",
                    scheduler.into()
                )),
            }),
        }
    }
}

// ============================================================================
// Generate Request/Response (contracts/generate.json)
// ============================================================================

/// Priority level for generation requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Priority {
    #[default]
    Normal,
    High,
}

/// Parameters for a generate request.
#[derive(Debug, Deserialize)]
pub struct GenerateParams {
    /// Text description of desired music.
    pub prompt: String,

    /// Duration of audio to generate in seconds (5-120 for MusicGen, 5-240 for ACE-Step).
    #[serde(default = "default_duration")]
    pub duration_sec: u32,

    /// Random seed for reproducibility; null for random.
    pub seed: Option<u64>,

    /// Queue priority.
    #[serde(default)]
    pub priority: Priority,

    /// Backend to use for generation. Defaults to config default_backend.
    pub backend: Option<String>,

    /// ACE-Step only: Number of diffusion inference steps (1-200, default 60).
    pub inference_steps: Option<u32>,

    /// ACE-Step only: Scheduler type ("euler", "heun", "pingpong", default "euler").
    pub scheduler: Option<String>,

    /// ACE-Step only: Classifier-free guidance scale (1.0-30.0, default 15.0).
    pub guidance_scale: Option<f32>,
}

fn default_duration() -> u32 {
    30
}

impl GenerateParams {
    /// Parses the backend parameter, returning the default if not specified.
    pub fn resolve_backend(&self, default: Backend) -> Result<Backend, JsonRpcError> {
        match &self.backend {
            Some(backend_str) => Backend::parse(backend_str)
                .ok_or_else(|| JsonRpcError::invalid_backend(backend_str)),
            None => Ok(default),
        }
    }

    /// Validates the request parameters for a specific backend.
    pub fn validate(&self, backend: Backend) -> Result<(), JsonRpcError> {
        // Check prompt
        if self.prompt.is_empty() {
            return Err(JsonRpcError::invalid_prompt("Prompt cannot be empty"));
        }
        if self.prompt.len() > 1000 {
            return Err(JsonRpcError::invalid_prompt(format!(
                "Prompt too long: {} characters (max 1000)",
                self.prompt.len()
            )));
        }

        // Check duration based on backend
        let min_duration = backend.min_duration_sec();
        let max_duration = backend.max_duration_sec();
        if self.duration_sec < min_duration || self.duration_sec > max_duration {
            return Err(JsonRpcError::invalid_duration_for_backend(
                self.duration_sec as i64,
                backend,
            ));
        }

        // Validate ACE-Step specific parameters
        if backend == Backend::AceStep {
            if let Some(steps) = self.inference_steps {
                if steps < 1 || steps > 200 {
                    return Err(JsonRpcError::invalid_inference_steps(steps));
                }
            }
            if let Some(scale) = self.guidance_scale {
                if !(1.0..=30.0).contains(&scale) {
                    return Err(JsonRpcError::invalid_guidance_scale(scale));
                }
            }
            if let Some(ref scheduler) = self.scheduler {
                let valid_schedulers = ["euler", "heun", "pingpong"];
                if !valid_schedulers.contains(&scheduler.to_lowercase().as_str()) {
                    return Err(JsonRpcError::invalid_scheduler(scheduler));
                }
            }
        }

        Ok(())
    }
}

/// Response for a generate request.
#[derive(Debug, Serialize)]
pub struct GenerateResult {
    /// Unique identifier for this generation.
    pub track_id: String,

    /// Initial status after request.
    pub status: GenerationStatus,

    /// Queue position (0 = next to generate).
    pub position: usize,

    /// Seed that will be used.
    pub seed: u64,

    /// Backend being used for generation.
    pub backend: String,
}

/// Status of a generation job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GenerationStatus {
    Queued,
    Generating,
    Complete,
    Error,
}

// ============================================================================
// Notifications (contracts/notifications.json)
// ============================================================================

/// A JSON-RPC notification (no id field).
#[derive(Debug, Serialize)]
pub struct JsonRpcNotification<T: Serialize> {
    pub jsonrpc: &'static str,
    pub method: &'static str,
    pub params: T,
}

impl<T: Serialize> JsonRpcNotification<T> {
    pub fn new(method: &'static str, params: T) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION,
            method,
            params,
        }
    }
}

/// Progress notification sent every 5% during generation.
#[derive(Debug, Serialize)]
pub struct GenerationProgressParams {
    /// Track being generated.
    pub track_id: String,

    /// Progress percentage (capped at 99 until complete).
    pub percent: u8,

    /// Number of token frames generated so far (for MusicGen).
    /// For ACE-Step, this equals current_step.
    pub tokens_generated: usize,

    /// Estimated total tokens (for MusicGen).
    /// For ACE-Step, this equals total_steps.
    pub tokens_estimated: usize,

    /// Estimated seconds remaining.
    pub eta_sec: f32,

    /// Current diffusion step (ACE-Step only).
    /// None for MusicGen token-based generation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_step: Option<usize>,

    /// Total diffusion steps (ACE-Step only).
    /// None for MusicGen token-based generation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_steps: Option<usize>,
}

/// Notification sent when generation finishes successfully.
#[derive(Debug, Serialize)]
pub struct GenerationCompleteParams {
    /// Completed track identifier.
    pub track_id: String,

    /// Absolute path to generated WAV file.
    pub path: String,

    /// Actual duration of generated audio.
    pub duration_sec: f32,

    /// Audio sample rate in Hz.
    pub sample_rate: u32,

    /// Original prompt used.
    pub prompt: String,

    /// Seed used for generation.
    pub seed: u64,

    /// Wall-clock time for generation.
    pub generation_time_sec: f32,

    /// Model identifier.
    pub model_version: String,

    /// Backend used for generation.
    pub backend: String,
}

/// Notification sent when generation fails.
#[derive(Debug, Serialize)]
pub struct GenerationErrorParams {
    /// Track that failed.
    pub track_id: String,

    /// Error code.
    pub code: String,

    /// Human-readable error message.
    pub message: String,
}

/// Download progress notification.
#[derive(Debug, Serialize)]
pub struct DownloadProgressParams {
    /// Current file being downloaded.
    pub file_name: String,

    /// Bytes received for current file.
    pub bytes_downloaded: u64,

    /// Total size of current file.
    pub bytes_total: u64,

    /// Number of files fully downloaded.
    pub files_completed: usize,

    /// Total number of files to download.
    pub files_total: usize,
}

// ============================================================================
// get_backends Request/Response
// ============================================================================

/// Status of a backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendStatus {
    /// Backend is not installed (model weights not downloaded).
    NotInstalled,
    /// Backend is currently downloading.
    Downloading,
    /// Backend is loading into memory.
    Loading,
    /// Backend is ready for generation.
    Ready,
    /// Backend encountered an error.
    Error,
}

impl Default for BackendStatus {
    fn default() -> Self {
        BackendStatus::NotInstalled
    }
}

/// Information about a specific backend.
#[derive(Debug, Clone, Serialize)]
pub struct BackendInfo {
    /// Backend type identifier (e.g., "musicgen", "ace_step").
    #[serde(rename = "type")]
    pub backend_type: String,

    /// Human-readable name.
    pub name: String,

    /// Current status.
    pub status: BackendStatus,

    /// Minimum duration in seconds.
    pub min_duration_sec: u32,

    /// Maximum duration in seconds.
    pub max_duration_sec: u32,

    /// Output sample rate in Hz.
    pub sample_rate: u32,

    /// Model version string (None if not installed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,
}

impl BackendInfo {
    /// Creates a BackendInfo for a given backend.
    pub fn new(backend: Backend, status: BackendStatus, model_version: Option<String>) -> Self {
        let name = match backend {
            Backend::MusicGen => "MusicGen-Small".to_string(),
            Backend::AceStep => "ACE-Step-3.5B".to_string(),
        };

        Self {
            backend_type: backend.as_str().to_string(),
            name,
            status,
            min_duration_sec: backend.min_duration_sec(),
            max_duration_sec: backend.max_duration_sec(),
            sample_rate: backend.sample_rate(),
            model_version,
        }
    }
}

/// Response for get_backends request.
#[derive(Debug, Serialize)]
pub struct GetBackendsResult {
    /// List of available backends with their status.
    pub backends: Vec<BackendInfo>,

    /// Default backend type.
    pub default_backend: String,
}

// ============================================================================
// download_backend Request/Response
// ============================================================================

/// Parameters for a download_backend request.
#[derive(Debug, Deserialize)]
pub struct DownloadBackendParams {
    /// Backend to download models for ("musicgen" or "ace_step").
    pub backend: String,
}

impl DownloadBackendParams {
    /// Parses and validates the backend parameter.
    pub fn validate(&self) -> Result<Backend, JsonRpcError> {
        Backend::parse(&self.backend)
            .ok_or_else(|| JsonRpcError::invalid_backend(&self.backend))
    }
}

/// Response for a download_backend request.
#[derive(Debug, Serialize)]
pub struct DownloadBackendResult {
    /// Backend that was downloaded.
    pub backend: String,

    /// Status of the download.
    pub status: String,

    /// Number of files downloaded.
    pub files_downloaded: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_params(prompt: &str, duration_sec: u32) -> GenerateParams {
        GenerateParams {
            prompt: prompt.to_string(),
            duration_sec,
            seed: None,
            priority: Priority::Normal,
            backend: None,
            inference_steps: None,
            scheduler: None,
            guidance_scale: None,
        }
    }

    #[test]
    fn request_id_from_int() {
        let id: RequestId = 42.into();
        assert_eq!(id, RequestId::Integer(42));
    }

    #[test]
    fn request_id_from_string() {
        let id: RequestId = "abc".to_string().into();
        assert_eq!(id, RequestId::String("abc".to_string()));
    }

    #[test]
    fn priority_default() {
        assert_eq!(Priority::default(), Priority::Normal);
    }

    #[test]
    fn generate_params_validate_empty_prompt() {
        let params = make_params("", 30);
        let err = params.validate(Backend::MusicGen).unwrap_err();
        assert_eq!(err.code, -32006);
    }

    #[test]
    fn generate_params_validate_long_prompt() {
        let params = make_params(&"x".repeat(1001), 30);
        let err = params.validate(Backend::MusicGen).unwrap_err();
        assert_eq!(err.code, -32006);
    }

    #[test]
    fn generate_params_validate_short_duration() {
        let params = make_params("test", 4);
        let err = params.validate(Backend::MusicGen).unwrap_err();
        assert_eq!(err.code, -32005);
    }

    #[test]
    fn generate_params_validate_long_duration_musicgen() {
        let params = make_params("test", 121);
        let err = params.validate(Backend::MusicGen).unwrap_err();
        assert_eq!(err.code, -32005);
    }

    #[test]
    fn generate_params_validate_long_duration_ace_step_ok() {
        let params = make_params("test", 121);
        // ACE-Step supports up to 240s, so 121 is valid
        assert!(params.validate(Backend::AceStep).is_ok());
    }

    #[test]
    fn generate_params_validate_too_long_duration_ace_step() {
        let params = make_params("test", 241);
        let err = params.validate(Backend::AceStep).unwrap_err();
        assert_eq!(err.code, -32005);
    }

    #[test]
    fn generate_params_validate_ok() {
        let params = GenerateParams {
            prompt: "test".to_string(),
            duration_sec: 30,
            seed: Some(42),
            priority: Priority::High,
            backend: None,
            inference_steps: None,
            scheduler: None,
            guidance_scale: None,
        };
        assert!(params.validate(Backend::MusicGen).is_ok());
    }

    #[test]
    fn generate_params_validate_ace_step_params() {
        let mut params = make_params("test", 60);
        params.inference_steps = Some(30);
        params.scheduler = Some("euler".to_string());
        params.guidance_scale = Some(7.0);
        assert!(params.validate(Backend::AceStep).is_ok());
    }

    #[test]
    fn generate_params_invalid_inference_steps() {
        let mut params = make_params("test", 60);
        params.inference_steps = Some(300);
        let err = params.validate(Backend::AceStep).unwrap_err();
        assert_eq!(err.code, -32009);
    }

    #[test]
    fn generate_params_invalid_guidance_scale() {
        let mut params = make_params("test", 60);
        params.guidance_scale = Some(50.0);
        let err = params.validate(Backend::AceStep).unwrap_err();
        assert_eq!(err.code, -32010);
    }

    #[test]
    fn generate_params_invalid_scheduler() {
        let mut params = make_params("test", 60);
        params.scheduler = Some("unknown".to_string());
        let err = params.validate(Backend::AceStep).unwrap_err();
        assert_eq!(err.code, -32011);
    }

    #[test]
    fn resolve_backend_default() {
        let params = make_params("test", 30);
        assert_eq!(
            params.resolve_backend(Backend::MusicGen).unwrap(),
            Backend::MusicGen
        );
        assert_eq!(
            params.resolve_backend(Backend::AceStep).unwrap(),
            Backend::AceStep
        );
    }

    #[test]
    fn resolve_backend_explicit() {
        let mut params = make_params("test", 30);
        params.backend = Some("ace_step".to_string());
        assert_eq!(
            params.resolve_backend(Backend::MusicGen).unwrap(),
            Backend::AceStep
        );
    }

    #[test]
    fn resolve_backend_invalid() {
        let mut params = make_params("test", 30);
        params.backend = Some("invalid".to_string());
        let err = params.resolve_backend(Backend::MusicGen).unwrap_err();
        assert_eq!(err.code, -32007);
    }

    #[test]
    fn json_rpc_error_codes() {
        assert_eq!(JsonRpcError::parse_error("").code, -32700);
        assert_eq!(JsonRpcError::invalid_request("").code, -32600);
        assert_eq!(JsonRpcError::method_not_found("").code, -32601);
        assert_eq!(JsonRpcError::invalid_params("").code, -32602);
        assert_eq!(JsonRpcError::internal_error("").code, -32603);
        assert_eq!(JsonRpcError::model_not_found("").code, -32000);
        assert_eq!(JsonRpcError::model_load_failed("").code, -32001);
        assert_eq!(JsonRpcError::model_download_failed("").code, -32002);
        assert_eq!(JsonRpcError::model_inference_failed("").code, -32003);
        assert_eq!(JsonRpcError::queue_full(10).code, -32004);
        assert_eq!(JsonRpcError::invalid_duration(0).code, -32005);
        assert_eq!(JsonRpcError::invalid_prompt("").code, -32006);
        assert_eq!(JsonRpcError::invalid_backend("").code, -32007);
        assert_eq!(JsonRpcError::backend_not_installed(&Backend::AceStep).code, -32008);
        assert_eq!(JsonRpcError::invalid_inference_steps(0).code, -32009);
        assert_eq!(JsonRpcError::invalid_guidance_scale(0.0).code, -32010);
        assert_eq!(JsonRpcError::invalid_scheduler("").code, -32011);
    }

    #[test]
    fn backend_info_creation() {
        let info = BackendInfo::new(Backend::MusicGen, BackendStatus::Ready, Some("v1".to_string()));
        assert_eq!(info.backend_type, "musicgen");
        assert_eq!(info.name, "MusicGen-Small");
        assert_eq!(info.status, BackendStatus::Ready);
        assert_eq!(info.min_duration_sec, 5);
        assert_eq!(info.max_duration_sec, 120);
        assert_eq!(info.sample_rate, 32000);
        assert_eq!(info.model_version, Some("v1".to_string()));

        let info = BackendInfo::new(Backend::AceStep, BackendStatus::NotInstalled, None);
        assert_eq!(info.backend_type, "ace_step");
        assert_eq!(info.name, "ACE-Step-3.5B");
        assert_eq!(info.status, BackendStatus::NotInstalled);
        assert_eq!(info.min_duration_sec, 5);
        assert_eq!(info.max_duration_sec, 240);
        assert_eq!(info.sample_rate, 48000);
        assert!(info.model_version.is_none());
    }
}
