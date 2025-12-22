//! JSON-RPC method handlers.
//!
//! Implements the handlers for all supported JSON-RPC methods.

use std::cell::RefCell;
use std::time::Instant;

use crate::audio::{write_wav, SAMPLE_RATE};
use crate::cli::TOKENS_PER_SECOND;
use crate::generation::{generate_with_models, ProgressTracker};
use crate::models::{ensure_models, Backend};
use crate::types::{compute_track_id, GenerationJob, JobPriority, Track};

use super::server::{send_notification, ServerState};
use super::types::{
    GenerateParams, GenerateResult, GenerationCompleteParams, GenerationErrorParams,
    GenerationProgressParams, GenerationStatus, JsonRpcError, Priority,
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

    // Check if queue is full before proceeding
    if state.queue.is_full() {
        return Err(JsonRpcError::queue_full(state.queue.len()));
    }

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

    // For now, always use MusicGen backend (Phase 4 will add backend selection)
    let backend = Backend::MusicGen;

    // Compute track ID (includes backend for uniqueness)
    let track_id = compute_track_id(
        backend,
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

    // Convert RPC priority to job priority
    let job_priority = match params.priority {
        Priority::High => JobPriority::High,
        Priority::Normal => JobPriority::Normal,
    };

    // Create a generation job
    let job = GenerationJob::new(
        params.prompt.clone(),
        params.duration_sec,
        Some(seed),
        job_priority,
        &model_version,
    );

    // Add job to queue and get position
    let position = state
        .queue
        .add(job)
        .map_err(|e| JsonRpcError::queue_full(e.current_size))?;

    // Check if this job should start immediately (position 0 and nothing generating)
    let should_generate_now = position == 0;

    if should_generate_now {
        // Pop the job from queue since we're processing it now
        let mut job = state.queue.pop_next().unwrap();
        job.set_generating();

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
                    backend,
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

                // Process next job in queue if any
                process_next_job(state);
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

                // Process next job in queue even after failure
                process_next_job(state);

                return Err(JsonRpcError::model_inference_failed(e.to_string()));
            }
        }

        Ok(serde_json::to_value(result).unwrap())
    } else {
        // Job is queued, return immediately with queue position
        Ok(serde_json::to_value(GenerateResult {
            track_id,
            status: GenerationStatus::Queued,
            position,
            seed,
        })
        .unwrap())
    }
}

/// Process the next job in the queue if any.
fn process_next_job(state: &mut ServerState) {
    if let Some(mut job) = state.queue.pop_next() {
        job.set_generating();

        let track_id = job.track_id.clone();
        let prompt = job.prompt.clone();
        let duration_sec = job.duration_sec;
        let seed = job.seed.unwrap_or_else(rand::random);

        let models = state.models.as_mut().unwrap();
        let model_version = models.version().to_string();

        let start_time = Instant::now();
        let max_tokens = duration_sec as usize * TOKENS_PER_SECOND;

        // Create progress tracker for 5% increment notifications
        let progress_tracker = RefCell::new(ProgressTracker::new(duration_sec));
        let track_id_for_progress = track_id.clone();

        match generate_with_models(models, &prompt, max_tokens, |current, total| {
            let mut tracker = progress_tracker.borrow_mut();
            tracker.update(current);

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
                } else {
                    let track = Track::new(
                        output_path.clone(),
                        prompt.clone(),
                        actual_duration,
                        seed,
                        model_version.clone(),
                        Backend::MusicGen,
                        generation_time,
                    );
                    state.cache.put(track);

                    send_notification(
                        "generation_complete",
                        GenerationCompleteParams {
                            track_id: track_id.clone(),
                            path: output_path.to_string_lossy().to_string(),
                            duration_sec: actual_duration,
                            sample_rate: SAMPLE_RATE,
                            prompt,
                            seed,
                            generation_time_sec: generation_time,
                            model_version,
                        },
                    );
                }

                // Continue processing queue
                process_next_job(state);
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

                // Continue processing queue even after failure
                process_next_job(state);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> crate::config::DaemonConfig {
        crate::config::DaemonConfig::default()
    }

    #[test]
    fn handle_ping() {
        let result = super::handle_ping();
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["status"], "ok");
    }

    #[test]
    fn handle_unknown_method() {
        let mut state = ServerState::new(test_config());
        let result = handle_request("nonexistent", serde_json::Value::Null, &mut state);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code, -32601);
    }

    #[test]
    fn handle_generate_invalid_params() {
        let mut state = ServerState::new(test_config());
        let params = serde_json::json!({});
        let result = handle_request("generate", params, &mut state);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code, -32602); // Invalid params
    }

    #[test]
    fn handle_generate_empty_prompt() {
        let mut state = ServerState::new(test_config());
        let params = serde_json::json!({ "prompt": "" });
        let result = handle_request("generate", params, &mut state);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code, -32006); // Invalid prompt
    }

    #[test]
    fn handle_shutdown() {
        let mut state = ServerState::new(test_config());
        let result = super::handle_shutdown(&mut state);
        assert!(result.is_ok());
        assert!(state.is_shutdown());
    }
}
