//! ACE-Step model loader.
//!
//! Loads all ONNX model components required for ACE-Step diffusion-based
//! music generation: UMT5 text encoder, diffusion transformer (encoder/decoder),
//! DCAE latent decoder, and ADaMoSHiFiGAN vocoder.

use std::path::Path;

use ort::execution_providers::ExecutionProviderDispatch;
use ort::session::Session;

use crate::config::DaemonConfig;
use crate::error::{DaemonError, Result};
use crate::models::device::{get_device_name, get_providers};

use super::decoder::DcaeDecoder;
use super::text_encoder::Umt5TextEncoder;
use super::transformer::DiffusionTransformer;
use super::vocoder::Vocoder;

/// Complete set of loaded ACE-Step models.
pub struct AceStepModels {
    /// UMT5 text encoder for converting prompts to embeddings.
    pub text_encoder: Umt5TextEncoder,
    /// Diffusion transformer for latent generation.
    pub transformer: DiffusionTransformer,
    /// DCAE decoder for latent to mel-spectrogram conversion.
    pub decoder: DcaeDecoder,
    /// Vocoder for mel-spectrogram to waveform conversion.
    pub vocoder: Vocoder,
    /// Model version string.
    version: String,
    /// Device name used for inference.
    device_name: String,
}

impl std::fmt::Debug for AceStepModels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AceStepModels")
            .field("version", &self.version)
            .field("device_name", &self.device_name)
            .finish_non_exhaustive()
    }
}

impl AceStepModels {
    /// Returns the model version string.
    pub fn version(&self) -> &str {
        &self.version
    }

    /// Returns the device name used for inference.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Loads all ACE-Step models from the specified directory.
    ///
    /// # Arguments
    ///
    /// * `model_dir` - Directory containing the ONNX model files
    /// * `config` - Daemon configuration for device and threading settings
    ///
    /// # Required Files
    ///
    /// The directory should contain:
    /// - `text_encoder.onnx` - UMT5 text encoder (~1.13 GB)
    /// - `transformer_encoder.onnx` - Diffusion transformer encoder (~424 MB)
    /// - `transformer_decoder.onnx` - Diffusion transformer decoder (~35.7 MB + external weights)
    /// - `dcae_decoder.onnx` - MusicDCAE latent decoder (~317 MB)
    /// - `vocoder.onnx` - ADaMoSHiFiGAN vocoder (~412 MB)
    /// - `tokenizer.json` - UMT5 tokenizer (~16.8 MB)
    pub fn load(model_dir: &Path, config: &DaemonConfig) -> Result<Self> {
        // Get execution providers based on device config
        let providers = get_providers(config.device, config.threads);
        let device_name = get_device_name(config.device).to_string();

        // On macOS, we force fp32 for numerical stability
        let force_fp32 = cfg!(target_os = "macos");

        Self::load_with_providers(model_dir, &providers, &device_name, force_fp32)
    }

    /// Loads all ACE-Step models with specific execution providers.
    ///
    /// # Arguments
    ///
    /// * `model_dir` - Directory containing the ONNX model files
    /// * `providers` - Execution providers for ONNX Runtime
    /// * `device_name` - Name of the device for logging
    /// * `force_fp32` - Force fp32 precision (required on macOS)
    pub fn load_with_providers(
        model_dir: &Path,
        providers: &[ExecutionProviderDispatch],
        device_name: &str,
        force_fp32: bool,
    ) -> Result<Self> {
        eprintln!("Loading ACE-Step models from {}...", model_dir.display());
        eprintln!("Using device: {} (fp32 forced: {})", device_name, force_fp32);

        // Load text encoder
        eprintln!("Loading UMT5 text encoder...");
        let text_encoder = Umt5TextEncoder::load(model_dir, providers)?;

        // Load diffusion transformer (encoder + decoder)
        eprintln!("Loading diffusion transformer...");
        let transformer = DiffusionTransformer::load(model_dir, providers)?;

        // Load DCAE decoder
        eprintln!("Loading DCAE decoder...");
        let decoder = DcaeDecoder::load(model_dir, providers)?;

        // Load vocoder
        eprintln!("Loading vocoder...");
        let vocoder = Vocoder::load(model_dir, providers)?;

        eprintln!("All ACE-Step models loaded successfully.");

        Ok(Self {
            text_encoder,
            transformer,
            decoder,
            vocoder,
            version: "ace-step-v1".to_string(),
            device_name: device_name.to_string(),
        })
    }
}

/// Required model files for ACE-Step.
pub const REQUIRED_FILES: &[&str] = &[
    "text_encoder.onnx",
    "transformer_encoder.onnx",
    "transformer_decoder.onnx",
    "transformer_decoder_weights.bin", // External weights for decoder (~10GB)
    "dcae_decoder.onnx",
    "vocoder.onnx",
    "tokenizer.json",
];

/// Download URLs for ACE-Step model files.
/// Hosted at https://huggingface.co/willibrandon/lofi-models/tree/main/ace-step/
pub const MODEL_URLS: &[(&str, &str)] = &[
    (
        "tokenizer.json",
        "https://huggingface.co/willibrandon/lofi-models/resolve/main/ace-step/tokenizer.json",
    ),
    (
        "text_encoder.onnx",
        "https://huggingface.co/willibrandon/lofi-models/resolve/main/ace-step/text_encoder.onnx",
    ),
    (
        "transformer_encoder.onnx",
        "https://huggingface.co/willibrandon/lofi-models/resolve/main/ace-step/transformer_encoder.onnx",
    ),
    (
        "transformer_decoder.onnx",
        "https://huggingface.co/willibrandon/lofi-models/resolve/main/ace-step/transformer_decoder.onnx",
    ),
    (
        "transformer_decoder_weights.bin",
        "https://huggingface.co/willibrandon/lofi-models/resolve/main/ace-step/transformer_decoder_weights.bin",
    ),
    (
        "dcae_decoder.onnx",
        "https://huggingface.co/willibrandon/lofi-models/resolve/main/ace-step/dcae_decoder.onnx",
    ),
    (
        "vocoder.onnx",
        "https://huggingface.co/willibrandon/lofi-models/resolve/main/ace-step/vocoder.onnx",
    ),
];

/// Checks if all required ACE-Step model files exist.
pub fn check_models(model_dir: &Path) -> Result<()> {
    let mut missing = Vec::new();

    for file in REQUIRED_FILES {
        let path = model_dir.join(file);
        if !path.exists() {
            missing.push(*file);
        }
    }

    if missing.is_empty() {
        Ok(())
    } else {
        Err(DaemonError::model_not_found(format!(
            "Missing ACE-Step model files in {}: {}",
            model_dir.display(),
            missing.join(", ")
        )))
    }
}

/// Loads an ONNX session from a file with the given providers.
pub fn load_session(
    model_path: &Path,
    providers: &[ExecutionProviderDispatch],
) -> Result<Session> {
    if !model_path.exists() {
        return Err(DaemonError::model_not_found(format!(
            "Model file not found: {}",
            model_path.display()
        )));
    }

    let builder = Session::builder().map_err(|e| {
        DaemonError::model_load_failed(format!("Failed to create session builder: {}", e))
    })?;

    // Register execution providers if any
    let builder = if !providers.is_empty() {
        builder
            .with_execution_providers(providers.to_vec())
            .map_err(|e| {
                DaemonError::model_load_failed(format!("Failed to set execution providers: {}", e))
            })?
    } else {
        builder
    };

    builder.commit_from_file(model_path).map_err(|e| {
        DaemonError::model_load_failed(format!(
            "Failed to load model {}: {}",
            model_path.display(),
            e
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_files_list() {
        assert_eq!(REQUIRED_FILES.len(), 7);
        assert!(REQUIRED_FILES.contains(&"text_encoder.onnx"));
        assert!(REQUIRED_FILES.contains(&"transformer_decoder_weights.bin"));
        assert!(REQUIRED_FILES.contains(&"vocoder.onnx"));
        assert!(REQUIRED_FILES.contains(&"tokenizer.json"));
    }

    #[test]
    fn check_nonexistent_dir_fails() {
        let path = Path::new("/nonexistent/path");
        let result = check_models(path);
        assert!(result.is_err());
    }
}
