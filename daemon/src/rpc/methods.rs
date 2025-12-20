//! JSON-RPC method handlers.
//!
//! Implements the handlers for all supported JSON-RPC methods.

use std::time::Instant;

use std::cell::RefCell;

use crate::audio::{write_wav, SAMPLE_RATE};
use crate::cli::TOKENS_PER_SECOND;
use crate::generation::{generate_with_models, ProgressTracker};
use crate::models::ensure_models;
use crate::types::{compute_track_id, Track};

use super::server::{send_notification, ServerState};
use super::types::{
    GenerateParams, GenerateResult, GenerationCompleteParams, GenerationErrorParams,
    GenerationProgressParams, GenerationStatus, JsonRpcError,
};

/// Handles a JSON-RPC method call.
pub fn handle_request(
    method: &str,
    params: serde_json::Value,
    state: &mut ServerState,
) -> Result<serde_json::Value, JsonRpcError> {
    match method {
        "generate" => handle_generate(params, state),
        "ping" => handle_ping(),
        "shutdown" => handle_shutdown(state),
        _ => Err(JsonRpcError::method_not_found(method)),
    }
}

/// Handles the ping method for health checks.
fn handle_ping() -> Result<serde_json::Value, JsonRpcError> {
    Ok(serde_json::json!({ "status": "ok" }))
}

/// Handles the shutdown method.
fn handle_shutdown(state: &mut ServerState) -> Result<serde_json::Value, JsonRpcError> {
    state.shutdown();
    Ok(serde_json::json!({ "status": "shutting_down" }))
}

/// Handles the generate method.
fn handle_generate(
    params: serde_json::Value,
    state: &mut ServerState,
) -> Result<serde_json::Value, JsonRpcError> {
    // Parse and validate parameters
    let params: GenerateParams = serde_json::from_value(params)
        .map_err(|e| JsonRpcError::invalid_params(format!("Invalid params: {}", e)))?;

    params.validate()?;

    // Generate seed if not provided
    let seed = params.seed.unwrap_or_else(rand::random);

    // Ensure models are downloaded
    let model_dir = state.config.effective_model_path();
    if let Err(e) = ensure_models(&model_dir) {
        return Err(JsonRpcError::model_download_failed(e.to_string()));
    }

    // Load models if not already loaded
    if state.models.is_none() {
        match crate::models::load_sessions(&model_dir) {
            Ok(models) => state.set_models(models),
            Err(e) => return Err(JsonRpcError::model_load_failed(e.to_string())),
        }
    }

    let models = state.models.as_mut().unwrap();
    let model_version = models.version().to_string();

    // Compute track ID
    let track_id = compute_track_id(
        &params.prompt,
        seed,
        params.duration_sec as f32,
        &model_version,
    );

    // Check cache for existing track
    if let Some(track) = state.cache.get(&track_id) {
        // Return cached track immediately
        send_notification(
            "generation_complete",
            GenerationCompleteParams {
                track_id: track.track_id.clone(),
                path: track.path.to_string_lossy().to_string(),
                duration_sec: track.duration_sec,
                sample_rate: track.sample_rate,
                prompt: track.prompt.clone(),
                seed: track.seed,
                generation_time_sec: 0.0, // Cached, no generation time
                model_version: track.model_version.clone(),
            },
        );

        return Ok(serde_json::to_value(GenerateResult {
            track_id: track.track_id.clone(),
            status: GenerationStatus::Complete,
            position: 0,
            seed,
        })
        .unwrap());
    }

    // For now, process synchronously (queue will be added in Phase 6)
    // Return response indicating generation is starting
    let result = GenerateResult {
        track_id: track_id.clone(),
        status: GenerationStatus::Generating,
        position: 0,
        seed,
    };

    // Perform generation
    let start_time = Instant::now();
    let max_tokens = params.duration_sec as usize * TOKENS_PER_SECOND;

    // Create progress tracker for 5% increment notifications
    let progress_tracker = RefCell::new(ProgressTracker::new(params.duration_sec));
    let track_id_for_progress = track_id.clone();

    match generate_with_models(models, &params.prompt, max_tokens, |current, total| {
        let mut tracker = progress_tracker.borrow_mut();
        tracker.update(current);

        // Check if we should send a notification (every 5% increment)
        if let Some(percent) = tracker.should_notify() {
            let (_, tokens_generated, tokens_estimated, eta_sec) = tracker.get_progress();
            send_notification(
                "generation_progress",
                GenerationProgressParams {
                    track_id: track_id_for_progress.clone(),
                    percent,
                    tokens_generated,
                    tokens_estimated,
                    eta_sec,
                },
            );
        }

        // Also report at 100% (though capped at 99 by ProgressTracker until complete)
        if current == total {
            let (percent, tokens_generated, tokens_estimated, eta_sec) = tracker.get_progress();
            send_notification(
                "generation_progress",
                GenerationProgressParams {
                    track_id: track_id_for_progress.clone(),
                    percent,
                    tokens_generated,
                    tokens_estimated,
                    eta_sec,
                },
            );
        }
    }) {
        Ok(samples) => {
            let generation_time = start_time.elapsed().as_secs_f32();
            let actual_duration = samples.len() as f32 / SAMPLE_RATE as f32;

            // Write to cache directory
            let cache_dir = state.config.effective_cache_path();
            std::fs::create_dir_all(&cache_dir).ok();
            let output_path = cache_dir.join(format!("{}.wav", track_id));

            if let Err(e) = write_wav(&samples, &output_path, SAMPLE_RATE) {
                send_notification(
                    "generation_error",
                    GenerationErrorParams {
                        track_id: track_id.clone(),
                        code: "MODEL_INFERENCE_FAILED".to_string(),
                        message: format!("Failed to write audio file: {}", e),
                    },
                );
                return Err(JsonRpcError::model_inference_failed(format!(
                    "Failed to write audio file: {}",
                    e
                )));
            }

            // Create track and cache it
            let track = Track::new(
                output_path.clone(),
                params.prompt.clone(),
                actual_duration,
                seed,
                model_version.clone(),
                generation_time,
            );
            state.cache.put(track);

            // Send completion notification
            send_notification(
                "generation_complete",
                GenerationCompleteParams {
                    track_id: track_id.clone(),
                    path: output_path.to_string_lossy().to_string(),
                    duration_sec: actual_duration,
                    sample_rate: SAMPLE_RATE,
                    prompt: params.prompt,
                    seed,
                    generation_time_sec: generation_time,
                    model_version,
                },
            );
        }
        Err(e) => {
            send_notification(
                "generation_error",
                GenerationErrorParams {
                    track_id: track_id.clone(),
                    code: "MODEL_INFERENCE_FAILED".to_string(),
                    message: e.to_string(),
                },
            );
            return Err(JsonRpcError::model_inference_failed(e.to_string()));
        }
    }

    Ok(serde_json::to_value(result).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_ping() {
        let result = super::handle_ping();
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["status"], "ok");
    }

    #[test]
    fn handle_unknown_method() {
        use crate::config::{DaemonConfig, Device};
        let mut state = ServerState::new(DaemonConfig {
            model_path: None,
            cache_path: None,
            device: Device::Cpu,
            threads: Some(4),
        });
        let result = handle_request("nonexistent", serde_json::Value::Null, &mut state);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code, -32601);
    }

    #[test]
    fn handle_generate_invalid_params() {
        use crate::config::{DaemonConfig, Device};
        let mut state = ServerState::new(DaemonConfig {
            model_path: None,
            cache_path: None,
            device: Device::Cpu,
            threads: Some(4),
        });
        let params = serde_json::json!({});
        let result = handle_request("generate", params, &mut state);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code, -32602); // Invalid params
    }

    #[test]
    fn handle_generate_empty_prompt() {
        use crate::config::{DaemonConfig, Device};
        let mut state = ServerState::new(DaemonConfig {
            model_path: None,
            cache_path: None,
            device: Device::Cpu,
            threads: Some(4),
        });
        let params = serde_json::json!({ "prompt": "" });
        let result = handle_request("generate", params, &mut state);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code, -32006); // Invalid prompt
    }

    #[test]
    fn handle_shutdown() {
        use crate::config::{DaemonConfig, Device};
        let mut state = ServerState::new(DaemonConfig {
            model_path: None,
            cache_path: None,
            device: Device::Cpu,
            threads: Some(4),
        });
        let result = super::handle_shutdown(&mut state);
        assert!(result.is_ok());
        assert!(state.is_shutdown());
    }
}
