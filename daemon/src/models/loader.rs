//! Unified model loader for all backends.
//!
//! Provides a single entry point for loading either MusicGen or ACE-Step models,
//! returning a LoadedModels enum that can be used for generation.

use std::path::Path;

use crate::config::DaemonConfig;
use crate::error::Result;
use crate::models::backend::{Backend, LoadedModels};
use crate::models::musicgen;

/// Loads models for the specified backend.
///
/// # Arguments
///
/// * `backend` - Which backend to load (MusicGen or AceStep)
/// * `config` - Daemon configuration with paths and device settings
///
/// # Returns
///
/// Returns `LoadedModels` containing the loaded model sessions.
/// Returns an error if the model files are not found or fail to load.
pub fn load_backend(backend: Backend, config: &DaemonConfig) -> Result<LoadedModels> {
    match backend {
        Backend::MusicGen => load_musicgen(config),
        Backend::AceStep => load_ace_step(config),
    }
}

/// Loads MusicGen models from the configured path.
fn load_musicgen(config: &DaemonConfig) -> Result<LoadedModels> {
    let model_path = config.effective_model_path();
    let models =
        musicgen::load_sessions_with_device(&model_path, config.device, config.threads)?;
    Ok(LoadedModels::MusicGen(models))
}

/// Loads ACE-Step models from the configured path.
///
/// Note: This is a placeholder implementation that will be fully
/// implemented in Phase 3 (User Story 1) when ACE-Step model wrappers are added.
fn load_ace_step(config: &DaemonConfig) -> Result<LoadedModels> {
    let model_path = config.effective_ace_step_model_path();

    // Check if model directory exists
    if !model_path.exists() {
        return Err(crate::error::DaemonError::backend_not_installed("ace_step"));
    }

    // Check for required model files
    check_ace_step_models(&model_path)?;

    // Placeholder: actual model loading will be implemented in Phase 3
    // For now, return a placeholder that indicates the backend exists but
    // actual generation will fail until Phase 3 is complete
    Err(crate::error::DaemonError::backend_not_installed(
        "ace_step (model loading not yet implemented)",
    ))
}

/// Required model files for ACE-Step.
const ACE_STEP_REQUIRED_FILES: &[&str] = &[
    "text_encoder.onnx",
    "transformer_encoder.onnx",
    "transformer_decoder.onnx",
    "dcae_decoder.onnx",
    "vocoder.onnx",
    "tokenizer.json",
];

/// Checks if all required ACE-Step model files exist.
fn check_ace_step_models(model_dir: &Path) -> Result<()> {
    let mut missing = Vec::new();

    for file in ACE_STEP_REQUIRED_FILES {
        let path = model_dir.join(file);
        if !path.exists() {
            missing.push(*file);
        }
    }

    if missing.is_empty() {
        Ok(())
    } else {
        Err(crate::error::DaemonError::model_not_found(format!(
            "Missing ACE-Step model files in {}: {}",
            model_dir.display(),
            missing.join(", ")
        )))
    }
}

/// Checks if a backend's models are available without loading them.
///
/// This is useful for quickly checking backend availability without
/// the overhead of loading large models into memory.
pub fn check_backend_available(backend: Backend, config: &DaemonConfig) -> bool {
    match backend {
        Backend::MusicGen => {
            let path = config.effective_model_path();
            musicgen::check_models(&path).is_ok()
        }
        Backend::AceStep => {
            let path = config.effective_ace_step_model_path();
            check_ace_step_models(&path).is_ok()
        }
    }
}

/// Returns the model version string for a backend if available.
pub fn get_backend_version(backend: Backend, config: &DaemonConfig) -> Option<String> {
    match backend {
        Backend::MusicGen => {
            let path = config.effective_model_path();
            Some(musicgen::detect_model_version(&path))
        }
        Backend::AceStep => {
            let path = config.effective_ace_step_model_path();
            if path.exists() {
                Some("ace-step-v1".to_string())
            } else {
                None
            }
        }
    }
}

/// Detects which backends are available.
///
/// Returns a list of backends that have all required model files present.
pub fn detect_available_backends(config: &DaemonConfig) -> Vec<Backend> {
    let mut available = Vec::new();

    if check_backend_available(Backend::MusicGen, config) {
        available.push(Backend::MusicGen);
    }

    if check_backend_available(Backend::AceStep, config) {
        available.push(Backend::AceStep);
    }

    available
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ace_step_required_files() {
        // Verify all required files are listed
        assert!(ACE_STEP_REQUIRED_FILES.contains(&"text_encoder.onnx"));
        assert!(ACE_STEP_REQUIRED_FILES.contains(&"vocoder.onnx"));
        assert!(ACE_STEP_REQUIRED_FILES.contains(&"tokenizer.json"));
    }

    #[test]
    fn check_nonexistent_dir_fails() {
        let path = std::path::Path::new("/nonexistent/path");
        let result = check_ace_step_models(path);
        assert!(result.is_err());
    }
}
