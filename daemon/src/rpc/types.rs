//! JSON-RPC types for the daemon protocol.
//!
//! Implements the contracts defined in contracts/generate.json, notifications.json, and errors.json.

use serde::{Deserialize, Serialize};

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

    /// Duration of audio to generate in seconds (5-120).
    #[serde(default = "default_duration")]
    pub duration_sec: u32,

    /// Random seed for reproducibility; null for random.
    pub seed: Option<u64>,

    /// Queue priority.
    #[serde(default)]
    pub priority: Priority,
}

fn default_duration() -> u32 {
    30
}

impl GenerateParams {
    /// Validates the request parameters.
    pub fn validate(&self) -> Result<(), JsonRpcError> {
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

        // Check duration
        if !(5..=120).contains(&self.duration_sec) {
            return Err(JsonRpcError::invalid_duration(self.duration_sec as i64));
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

    /// Number of token frames generated so far.
    pub tokens_generated: usize,

    /// Estimated total tokens.
    pub tokens_estimated: usize,

    /// Estimated seconds remaining.
    pub eta_sec: f32,
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

#[cfg(test)]
mod tests {
    use super::*;

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
        let params = GenerateParams {
            prompt: "".to_string(),
            duration_sec: 30,
            seed: None,
            priority: Priority::Normal,
        };
        let err = params.validate().unwrap_err();
        assert_eq!(err.code, -32006);
    }

    #[test]
    fn generate_params_validate_long_prompt() {
        let params = GenerateParams {
            prompt: "x".repeat(1001),
            duration_sec: 30,
            seed: None,
            priority: Priority::Normal,
        };
        let err = params.validate().unwrap_err();
        assert_eq!(err.code, -32006);
    }

    #[test]
    fn generate_params_validate_short_duration() {
        let params = GenerateParams {
            prompt: "test".to_string(),
            duration_sec: 4,
            seed: None,
            priority: Priority::Normal,
        };
        let err = params.validate().unwrap_err();
        assert_eq!(err.code, -32005);
    }

    #[test]
    fn generate_params_validate_long_duration() {
        let params = GenerateParams {
            prompt: "test".to_string(),
            duration_sec: 121,
            seed: None,
            priority: Priority::Normal,
        };
        let err = params.validate().unwrap_err();
        assert_eq!(err.code, -32005);
    }

    #[test]
    fn generate_params_validate_ok() {
        let params = GenerateParams {
            prompt: "test".to_string(),
            duration_sec: 30,
            seed: Some(42),
            priority: Priority::High,
        };
        assert!(params.validate().is_ok());
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
    }
}
