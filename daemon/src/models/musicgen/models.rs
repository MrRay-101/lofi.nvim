//! Model loader for MusicGen ONNX models.
//!
//! Handles loading all required model components and configuration.

use std::path::Path;

use crate::config::Device;
use crate::error::{DaemonError, Result};
use crate::types::ModelConfig;

use super::audio_codec::MusicGenAudioCodec;
use super::decoder::MusicGenDecoder;
use super::text_encoder::MusicGenTextEncoder;
use crate::models::device::{get_device_name, get_providers};

/// Complete set of loaded MusicGen models.
pub struct MusicGenModels {
    /// Text encoder for converting prompts to embeddings.
    pub text_encoder: MusicGenTextEncoder,
    /// Decoder for autoregressive token generation.
    pub decoder: MusicGenDecoder,
    /// Audio codec for converting tokens to audio samples.
    pub audio_codec: MusicGenAudioCodec,
    /// Model configuration.
    pub config: ModelConfig,
    /// Model version string.
    pub version: String,
    /// Active device name.
    pub device_name: String,
}

impl std::fmt::Debug for MusicGenModels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MusicGenModels")
            .field("config", &self.config)
            .field("version", &self.version)
            .field("device_name", &self.device_name)
            .finish_non_exhaustive()
    }
}

impl MusicGenModels {
    /// Returns the model version string.
    pub fn version(&self) -> &str {
        &self.version
    }

    /// Returns the active device name.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }
}

/// Required model files for MusicGen.
pub const REQUIRED_MODEL_FILES: &[&str] = &[
    "tokenizer.json",
    "text_encoder.onnx",
    "decoder_model.onnx",
    "decoder_with_past_model.onnx",
    "encodec_decode.onnx",
];

/// Checks if all required model files exist in the directory.
///
/// Returns Ok(()) if all files exist, or an error listing missing files.
pub fn check_models(model_dir: &Path) -> Result<()> {
    let mut missing = Vec::new();

    for file in REQUIRED_MODEL_FILES {
        let path = model_dir.join(file);
        if !path.exists() {
            missing.push(*file);
        }
    }

    if missing.is_empty() {
        Ok(())
    } else {
        Err(DaemonError::model_not_found(format!(
            "Missing model files in {}: {}",
            model_dir.display(),
            missing.join(", ")
        )))
    }
}

/// Loads all MusicGen model sessions from a directory.
///
/// The directory should contain:
/// - `tokenizer.json` - HuggingFace tokenizer
/// - `text_encoder.onnx` - T5 text encoder
/// - `decoder_model.onnx` - First pass decoder
/// - `decoder_with_past_model.onnx` - Decoder with KV cache
/// - `encodec_decode.onnx` - EnCodec audio decoder
///
/// Optionally:
/// - `config.json` - Model configuration (uses defaults if not present)
pub fn load_sessions(model_dir: &Path) -> Result<MusicGenModels> {
    load_sessions_with_device(model_dir, Device::Auto, None)
}

/// Loads all MusicGen model sessions from a directory with specific device configuration.
///
/// # Arguments
///
/// * `model_dir` - Directory containing model files
/// * `device` - Device to use for inference (Auto, Cpu, Cuda, Metal)
/// * `threads` - Optional number of threads for CPU execution
///
/// The directory should contain:
/// - `tokenizer.json` - HuggingFace tokenizer
/// - `text_encoder.onnx` - T5 text encoder
/// - `decoder_model.onnx` - First pass decoder
/// - `decoder_with_past_model.onnx` - Decoder with KV cache
/// - `encodec_decode.onnx` - EnCodec audio decoder
///
/// Optionally:
/// - `config.json` - Model configuration (uses defaults if not present)
pub fn load_sessions_with_device(
    model_dir: &Path,
    device: Device,
    threads: Option<u32>,
) -> Result<MusicGenModels> {
    // Check all required files exist first
    check_models(model_dir)?;

    // Get execution providers for the device
    let providers = get_providers(device, threads);
    let device_name = get_device_name(device).to_string();

    eprintln!("Using device: {}", device_name);

    eprintln!("Loading text encoder...");
    let text_encoder = MusicGenTextEncoder::load_with_providers(model_dir, &providers)?;

    // Load or create config
    let config = load_or_default_config(model_dir)?;

    eprintln!("Loading decoder models...");
    let decoder = MusicGenDecoder::load_with_providers(model_dir, config.clone(), &providers)?;

    eprintln!("Loading audio codec...");
    let audio_codec = MusicGenAudioCodec::load_with_providers(model_dir, &providers)?;

    // Determine version from directory name or default
    let version = detect_model_version(model_dir);

    eprintln!("All models loaded successfully.");

    Ok(MusicGenModels {
        text_encoder,
        decoder,
        audio_codec,
        config,
        version,
        device_name,
    })
}

/// Loads model configuration from config.json or uses defaults.
fn load_or_default_config(model_dir: &Path) -> Result<ModelConfig> {
    let config_path = model_dir.join("config.json");

    if config_path.exists() {
        let content = std::fs::read_to_string(&config_path).map_err(|e| {
            DaemonError::model_load_failed(format!("Failed to read config.json: {}", e))
        })?;

        // Parse the config - MusicGen config has nested structure
        let json: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
            DaemonError::model_load_failed(format!("Failed to parse config.json: {}", e))
        })?;

        // Extract decoder config values
        let decoder = json.get("decoder").ok_or_else(|| {
            DaemonError::model_load_failed("config.json missing 'decoder' section".to_string())
        })?;

        let num_hidden_layers = decoder
            .get("num_hidden_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(24) as u32;

        let num_attention_heads = decoder
            .get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(16) as u32;

        let vocab_size = decoder
            .get("vocab_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(2048) as u32;

        let pad_token_id = decoder
            .get("pad_token_id")
            .and_then(|v| v.as_i64())
            .unwrap_or(2048);

        let text_encoder = json.get("text_encoder");
        let d_kv = text_encoder
            .and_then(|te| te.get("d_kv"))
            .and_then(|v| v.as_u64())
            .unwrap_or(64) as u32;

        let d_model = text_encoder
            .and_then(|te| te.get("d_model"))
            .and_then(|v| v.as_u64())
            .unwrap_or(1024) as u32;

        Ok(ModelConfig {
            vocab_size,
            num_hidden_layers,
            num_attention_heads,
            d_model,
            d_kv,
            audio_channels: 1,
            sample_rate: 32000,
            codebooks: 4,
            pad_token_id,
        })
    } else {
        // Use default musicgen-small config
        Ok(ModelConfig::musicgen_small())
    }
}

/// Generates a model version string from size and precision.
///
/// # Arguments
///
/// * `size` - Model size: "small" or "medium"
/// * `precision` - Model precision: "fp16" or "fp32"
/// * `version` - Version number (e.g., 1)
///
/// # Example
///
/// ```
/// use lofi_daemon::models::musicgen::generate_model_version;
/// assert_eq!(generate_model_version("small", "fp16", 1), "musicgen-small-fp16-v1");
/// ```
pub fn generate_model_version(size: &str, precision: &str, version: u32) -> String {
    format!("musicgen-{}-{}-v{}", size, precision, version)
}

/// Detects model version from directory structure.
pub fn detect_model_version(model_dir: &Path) -> String {
    let dir_name = model_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");

    // Check for common patterns
    if dir_name.contains("fp16") {
        if dir_name.contains("medium") {
            return generate_model_version("medium", "fp16", 1);
        }
        return generate_model_version("small", "fp16", 1);
    }

    if dir_name.contains("fp32") {
        if dir_name.contains("medium") {
            return generate_model_version("medium", "fp32", 1);
        }
        return generate_model_version("small", "fp32", 1);
    }

    if dir_name.contains("medium") {
        return generate_model_version("medium", "fp16", 1);
    }

    // Default
    generate_model_version("small", "fp16", 1)
}

/// HuggingFace model URLs for musicgen-small-fp16.
pub const MODEL_URLS: &[(&str, &str)] = &[
    (
        "config.json",
        "https://huggingface.co/gabotechs/music_gen/resolve/main/small/config.json",
    ),
    (
        "tokenizer.json",
        "https://huggingface.co/gabotechs/music_gen/resolve/main/small/tokenizer.json",
    ),
    (
        "text_encoder.onnx",
        "https://huggingface.co/gabotechs/music_gen/resolve/main/small_fp16/text_encoder.onnx",
    ),
    (
        "decoder_model.onnx",
        "https://huggingface.co/gabotechs/music_gen/resolve/main/small_fp16/decoder_model.onnx",
    ),
    (
        "decoder_with_past_model.onnx",
        "https://huggingface.co/gabotechs/music_gen/resolve/main/small_fp16/decoder_with_past_model.onnx",
    ),
    (
        "encodec_decode.onnx",
        "https://huggingface.co/gabotechs/music_gen/resolve/main/small_fp16/encodec_decode.onnx",
    ),
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn generate_version_string() {
        assert_eq!(
            generate_model_version("small", "fp16", 1),
            "musicgen-small-fp16-v1"
        );
        assert_eq!(
            generate_model_version("medium", "fp32", 2),
            "musicgen-medium-fp32-v2"
        );
    }

    #[test]
    fn detect_version_fp16() {
        let path = PathBuf::from("/path/to/small_fp16");
        assert_eq!(detect_model_version(&path), "musicgen-small-fp16-v1");
    }

    #[test]
    fn detect_version_medium() {
        let path = PathBuf::from("/path/to/medium_fp32");
        assert_eq!(detect_model_version(&path), "musicgen-medium-fp32-v1");
    }

    #[test]
    fn required_files_list() {
        assert_eq!(REQUIRED_MODEL_FILES.len(), 5);
        assert!(REQUIRED_MODEL_FILES.contains(&"tokenizer.json"));
        assert!(REQUIRED_MODEL_FILES.contains(&"encodec_decode.onnx"));
    }
}
