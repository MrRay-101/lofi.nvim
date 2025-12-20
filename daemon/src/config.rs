//! Daemon configuration module.
//!
//! Provides configuration types for device selection, threading,
//! and model/cache paths.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Hardware device for model inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Device {
    /// Automatically select best available device.
    #[default]
    Auto,
    /// Force CPU execution.
    Cpu,
    /// Use NVIDIA CUDA GPU.
    Cuda,
    /// Use Apple Metal GPU (macOS only).
    Metal,
}

impl Device {
    /// Returns the ONNX Runtime execution provider name.
    pub fn execution_provider(&self) -> &'static str {
        match self {
            Device::Auto => "auto",
            Device::Cpu => "CPUExecutionProvider",
            Device::Cuda => "CUDAExecutionProvider",
            Device::Metal => "CoreMLExecutionProvider",
        }
    }
}

/// Configuration for the lofi daemon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonConfig {
    /// Path to the directory containing ONNX model files.
    pub model_path: PathBuf,

    /// Path to the cache directory for generated tracks.
    pub cache_path: PathBuf,

    /// Device to use for inference.
    pub device: Device,

    /// Number of threads for CPU execution (0 = auto).
    pub threads: u32,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        // Use platform-appropriate cache directory
        let base_cache = directories::BaseDirs::new()
            .map(|d| d.cache_dir().to_path_buf())
            .unwrap_or_else(|| PathBuf::from(".cache"));

        let lofi_cache = base_cache.join("lofi");

        Self {
            model_path: lofi_cache.join("models").join("musicgen-small-fp16"),
            cache_path: lofi_cache.join("tracks"),
            device: Device::Auto,
            threads: 0, // Auto-detect
        }
    }
}

impl DaemonConfig {
    /// Creates a new DaemonConfig with the specified model path.
    pub fn with_model_path(model_path: PathBuf) -> Self {
        Self {
            model_path,
            ..Default::default()
        }
    }

    /// Creates a new DaemonConfig with custom paths.
    pub fn new(model_path: PathBuf, cache_path: PathBuf, device: Device, threads: u32) -> Self {
        Self {
            model_path,
            cache_path,
            device,
            threads,
        }
    }

    /// Returns the path to the tokenizer.json file.
    pub fn tokenizer_path(&self) -> PathBuf {
        self.model_path.join("tokenizer.json")
    }

    /// Returns the path to the config.json file.
    pub fn config_path(&self) -> PathBuf {
        self.model_path.join("config.json")
    }

    /// Returns the path to the text encoder ONNX model.
    pub fn text_encoder_path(&self) -> PathBuf {
        self.model_path.join("text_encoder.onnx")
    }

    /// Returns the path to the decoder ONNX model (first iteration).
    pub fn decoder_path(&self) -> PathBuf {
        self.model_path.join("decoder_model.onnx")
    }

    /// Returns the path to the decoder with past ONNX model (subsequent iterations).
    pub fn decoder_with_past_path(&self) -> PathBuf {
        self.model_path.join("decoder_with_past_model.onnx")
    }

    /// Returns the path to the audio codec ONNX model.
    pub fn audio_codec_path(&self) -> PathBuf {
        self.model_path.join("encodec_decode.onnx")
    }

    /// Checks if all required model files exist.
    pub fn models_exist(&self) -> bool {
        self.tokenizer_path().exists()
            && self.config_path().exists()
            && self.text_encoder_path().exists()
            && self.decoder_path().exists()
            && self.decoder_with_past_path().exists()
            && self.audio_codec_path().exists()
    }

    /// Returns a list of missing model files.
    pub fn missing_models(&self) -> Vec<PathBuf> {
        let paths = [
            self.tokenizer_path(),
            self.config_path(),
            self.text_encoder_path(),
            self.decoder_path(),
            self.decoder_with_past_path(),
            self.audio_codec_path(),
        ];

        paths
            .into_iter()
            .filter(|p| !p.exists())
            .collect()
    }
}
