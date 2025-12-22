//! JSON-RPC method handlers.
//!
//! Implements the handlers for all supported JSON-RPC methods.

use std::cell::RefCell;
use std::time::Instant;

use crate::audio::write_wav;
use crate::models::{
    check_backend_available, download_backend_with_progress, ensure_ace_step_models, ensure_models,
    load_backend, Backend, GenerateDispatchParams,
};
use crate::types::{compute_track_id, GenerationJob, JobPriority, Track};

use super::server::{send_notification, ServerState};
use super::types::{
    BackendInfo, BackendStatus, DownloadBackendParams, DownloadBackendResult, DownloadProgressParams,
    GenerateParams, GenerateResult, GenerationCompleteParams, GenerationErrorParams,
    GenerationProgressParams, GenerationStatus, GetBackendsResult, JsonRpcError, Priority,
};

/// Handles a JSON-RPC method call.
pub fn handle_request(
    method: &str,
    params: serde_json::Value,
    state: &mut ServerState,
) -> Result<serde_json::Value, JsonRpcError> {
    match method {
        "generate" => handle_generate(params, state),
        "get_backends" => handle_get_backends(state),
        "download_backend" => handle_download_backend(params, state),
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
    // Parse parameters
    let params: GenerateParams = serde_json::from_value(params)
        .map_err(|e| JsonRpcError::invalid_params(format!("Invalid params: {}", e)))?;

    // Resolve which backend to use
    let backend = params.resolve_backend(state.config.default_backend)?;

    // Validate parameters for the selected backend
    params.validate(backend)?;

    // Check if queue is full before proceeding
    if state.queue.is_full() {
        return Err(JsonRpcError::queue_full(state.queue.len()));
    }

    // Generate seed if not provided
    let seed = params.seed.unwrap_or_else(rand::random);

    // Ensure models are downloaded for the selected backend
    match backend {
        Backend::MusicGen => {
            let model_dir = state.config.effective_model_path();
            if let Err(e) = ensure_models(&model_dir) {
                return Err(JsonRpcError::model_download_failed(e.to_string()));
            }
        }
        Backend::AceStep => {
            let model_dir = state.config.effective_ace_step_model_path();
            if let Err(e) = ensure_ace_step_models(&model_dir) {
                return Err(JsonRpcError::model_download_failed(e.to_string()));
            }
        }
    }

    // Check if the loaded models match the requested backend
    let current_backend = state.models.backend();
    if current_backend != Some(backend) {
        // Need to load the correct backend
        let model_dir = match backend {
            Backend::MusicGen => state.config.effective_model_path(),
            Backend::AceStep => state.config.effective_ace_step_model_path(),
        };
        match load_backend(backend, &model_dir, &state.config) {
            Ok(models) => state.set_models(models),
            Err(e) => return Err(JsonRpcError::model_load_failed(e.to_string())),
        }
    }

    let model_version = state.models.version().unwrap_or("unknown").to_string();

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
                backend: track.backend.as_str().to_string(),
            },
        );

        return Ok(serde_json::to_value(GenerateResult {
            track_id: track.track_id.clone(),
            status: GenerationStatus::Complete,
            position: 0,
            seed,
            backend: backend.as_str().to_string(),
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
            backend: backend.as_str().to_string(),
        };

        // Build dispatch params
        let dispatch_params = GenerateDispatchParams::new(
            params.prompt.clone(),
            params.duration_sec,
            seed,
            backend,
        )
        .with_ace_step_params(
            params.inference_steps,
            params.scheduler.clone(),
            params.guidance_scale,
        );

        // Perform generation
        let start_time = Instant::now();
        let sample_rate = backend.sample_rate();

        // Track progress - use RefCell for interior mutability in closure
        let last_percent = RefCell::new(0u8);
        let track_id_for_progress = track_id.clone();

        // Track if this is step-based (ACE-Step) or token-based (MusicGen)
        let is_step_based = backend == Backend::AceStep;

        match state.models.generate(&dispatch_params, |current, total| {
            if total == 0 {
                return;
            }

            // Calculate percent directly from callback values
            let percent = std::cmp::min((current * 100 / total) as u8, 99);
            let mut last = last_percent.borrow_mut();

            // Report every 5% increment
            let next_threshold = (*last / 5 + 1) * 5;
            if percent >= next_threshold || current == total {
                *last = (percent / 5) * 5;

                let elapsed = start_time.elapsed().as_secs_f32();
                let eta_sec = if current > 0 && elapsed > 0.0 {
                    let remaining = total.saturating_sub(current);
                    (remaining as f32 / current as f32) * elapsed
                } else {
                    0.0
                };

                // Include step info for ACE-Step, None for MusicGen
                let (current_step, total_steps) = if is_step_based {
                    (Some(current), Some(total))
                } else {
                    (None, None)
                };

                send_notification(
                    "generation_progress",
                    GenerationProgressParams {
                        track_id: track_id_for_progress.clone(),
                        percent: if current == total { 100 } else { percent },
                        tokens_generated: current,
                        tokens_estimated: total,
                        eta_sec,
                        current_step,
                        total_steps,
                    },
                );
            }
        }) {
            Ok(samples) => {
                let generation_time = start_time.elapsed().as_secs_f32();
                let actual_duration = samples.len() as f32 / sample_rate as f32;

                // Write to cache directory
                let cache_dir = state.config.effective_cache_path();
                std::fs::create_dir_all(&cache_dir).ok();
                let output_path = cache_dir.join(format!("{}.wav", track_id));

                if let Err(e) = write_wav(&samples, &output_path, sample_rate) {
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
                        sample_rate,
                        prompt: params.prompt,
                        seed,
                        generation_time_sec: generation_time,
                        model_version,
                        backend: backend.as_str().to_string(),
                    },
                );

                // Process next job in queue if any
                process_next_job(state, backend);
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
                process_next_job(state, backend);

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
            backend: backend.as_str().to_string(),
        })
        .unwrap())
    }
}

/// Process the next job in the queue if any.
fn process_next_job(state: &mut ServerState, backend: Backend) {
    if let Some(mut job) = state.queue.pop_next() {
        job.set_generating();

        let track_id = job.track_id.clone();
        let prompt = job.prompt.clone();
        let duration_sec = job.duration_sec;
        let seed = job.seed.unwrap_or_else(rand::random);

        let model_version = state.models.version().unwrap_or("unknown").to_string();
        let sample_rate = backend.sample_rate();

        // Build dispatch params for queued job (uses defaults for ACE-Step params)
        let dispatch_params = GenerateDispatchParams::new(prompt.clone(), duration_sec, seed, backend);

        let start_time = Instant::now();

        // Track progress
        let last_percent = RefCell::new(0u8);
        let track_id_for_progress = track_id.clone();
        let is_step_based = backend == Backend::AceStep;

        match state.models.generate(&dispatch_params, |current, total| {
            if total == 0 {
                return;
            }

            let percent = std::cmp::min((current * 100 / total) as u8, 99);
            let mut last = last_percent.borrow_mut();

            let next_threshold = (*last / 5 + 1) * 5;
            if percent >= next_threshold || current == total {
                *last = (percent / 5) * 5;

                let elapsed = start_time.elapsed().as_secs_f32();
                let eta_sec = if current > 0 && elapsed > 0.0 {
                    let remaining = total.saturating_sub(current);
                    (remaining as f32 / current as f32) * elapsed
                } else {
                    0.0
                };

                // Include step info for ACE-Step, None for MusicGen
                let (current_step, total_steps) = if is_step_based {
                    (Some(current), Some(total))
                } else {
                    (None, None)
                };

                send_notification(
                    "generation_progress",
                    GenerationProgressParams {
                        track_id: track_id_for_progress.clone(),
                        percent: if current == total { 100 } else { percent },
                        tokens_generated: current,
                        tokens_estimated: total,
                        eta_sec,
                        current_step,
                        total_steps,
                    },
                );
            }
        }) {
            Ok(samples) => {
                let generation_time = start_time.elapsed().as_secs_f32();
                let actual_duration = samples.len() as f32 / sample_rate as f32;

                let cache_dir = state.config.effective_cache_path();
                std::fs::create_dir_all(&cache_dir).ok();
                let output_path = cache_dir.join(format!("{}.wav", track_id));

                if let Err(e) = write_wav(&samples, &output_path, sample_rate) {
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
                        backend,
                        generation_time,
                    );
                    state.cache.put(track);

                    send_notification(
                        "generation_complete",
                        GenerationCompleteParams {
                            track_id: track_id.clone(),
                            path: output_path.to_string_lossy().to_string(),
                            duration_sec: actual_duration,
                            sample_rate,
                            prompt,
                            seed,
                            generation_time_sec: generation_time,
                            model_version,
                            backend: backend.as_str().to_string(),
                        },
                    );
                }

                // Continue processing queue
                process_next_job(state, backend);
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
                process_next_job(state, backend);
            }
        }
    }
}

/// Handles the get_backends method.
fn handle_get_backends(state: &ServerState) -> Result<serde_json::Value, JsonRpcError> {
    // Check installation status for each backend
    // "Ready" means models are downloaded and can be loaded on-demand
    let musicgen_status = if check_backend_available(Backend::MusicGen, &state.config.effective_model_path()) {
        // Models exist on disk - report as Ready (loadable on-demand)
        BackendStatus::Ready
    } else {
        BackendStatus::NotInstalled
    };

    let ace_step_status = if check_backend_available(Backend::AceStep, &state.config.effective_ace_step_model_path()) {
        // Models exist on disk - report as Ready (loadable on-demand)
        BackendStatus::Ready
    } else {
        BackendStatus::NotInstalled
    };

    // Get model versions if loaded
    let musicgen_version = if state.models.backend() == Some(Backend::MusicGen) {
        state.models.version().map(|s| s.to_string())
    } else {
        None
    };

    let ace_step_version = if state.models.backend() == Some(Backend::AceStep) {
        state.models.version().map(|s| s.to_string())
    } else {
        None
    };

    let result = GetBackendsResult {
        backends: vec![
            BackendInfo::new(Backend::MusicGen, musicgen_status, musicgen_version),
            BackendInfo::new(Backend::AceStep, ace_step_status, ace_step_version),
        ],
        default_backend: state.config.default_backend.as_str().to_string(),
    };

    Ok(serde_json::to_value(result).unwrap())
}

/// Handles the download_backend method.
///
/// Downloads model files for the specified backend, emitting progress notifications
/// as files are downloaded. Supports resuming partial downloads.
fn handle_download_backend(
    params: serde_json::Value,
    state: &mut ServerState,
) -> Result<serde_json::Value, JsonRpcError> {
    // Parse and validate parameters
    let params: DownloadBackendParams = serde_json::from_value(params)
        .map_err(|e| JsonRpcError::invalid_params(format!("Invalid params: {}", e)))?;

    let backend = params.validate()?;

    // Check if already downloading
    if state.backend_status.get(backend) == BackendStatus::Downloading {
        return Ok(serde_json::to_value(DownloadBackendResult {
            backend: backend.as_str().to_string(),
            status: "already_downloading".to_string(),
            files_downloaded: 0,
        })
        .unwrap());
    }

    // Check if already installed
    let model_dir = match backend {
        Backend::MusicGen => state.config.effective_model_path(),
        Backend::AceStep => state.config.effective_ace_step_model_path(),
    };

    if check_backend_available(backend, &model_dir) {
        return Ok(serde_json::to_value(DownloadBackendResult {
            backend: backend.as_str().to_string(),
            status: "already_installed".to_string(),
            files_downloaded: 0,
        })
        .unwrap());
    }

    // Update status to downloading
    state.backend_status.set(backend, BackendStatus::Downloading);

    // Create progress callback that sends notifications
    let on_progress = Box::new(move |file_name: &str, bytes_downloaded: u64, bytes_total: u64, files_completed: usize, files_total: usize| {
        send_notification(
            "download_progress",
            DownloadProgressParams {
                file_name: file_name.to_string(),
                bytes_downloaded,
                bytes_total,
                files_completed,
                files_total,
            },
        );
    });

    // Perform download
    match download_backend_with_progress(backend, &model_dir, Some(on_progress)) {
        Ok(()) => {
            state.backend_status.set(backend, BackendStatus::Ready);
            Ok(serde_json::to_value(DownloadBackendResult {
                backend: backend.as_str().to_string(),
                status: "complete".to_string(),
                files_downloaded: match backend {
                    Backend::MusicGen => 6, // Number of MusicGen files
                    Backend::AceStep => 7,   // Number of ACE-Step files
                },
            })
            .unwrap())
        }
        Err(e) => {
            state.backend_status.set(backend, BackendStatus::Error);
            Err(JsonRpcError::model_download_failed(e.to_string()))
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
