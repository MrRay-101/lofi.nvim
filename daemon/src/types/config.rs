//! Model configuration types.
//!
//! Defines the ModelConfig struct containing parameters loaded from
//! the MusicGen config.json file.

use serde::{Deserialize, Serialize};

/// Configuration parameters for the MusicGen model.
///
/// These values are loaded from the config.json file that accompanies
/// the ONNX model files. They define the model architecture and
/// generation parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Token vocabulary size.
    pub vocab_size: u32,

    /// Number of decoder transformer layers.
    pub num_hidden_layers: u32,

    /// Number of attention heads per layer.
    pub num_attention_heads: u32,

    /// Hidden dimension (model width).
    pub d_model: u32,

    /// Key/value dimension per attention head.
    pub d_kv: u32,

    /// Number of audio channels (always 1 for mono).
    pub audio_channels: u32,

    /// Audio sample rate in Hz (always 32000 for MusicGen).
    pub sample_rate: u32,

    /// Number of codebooks for audio tokenization (always 4 for MusicGen).
    pub codebooks: u32,

    /// Padding token ID for the tokenizer.
    pub pad_token_id: u32,
}

impl Default for ModelConfig {
    /// Default configuration for MusicGen-small.
    fn default() -> Self {
        Self {
            vocab_size: 2048,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            d_model: 1024,
            d_kv: 64,
            audio_channels: 1,
            sample_rate: 32000,
            codebooks: 4,
            pad_token_id: 2048,
        }
    }
}

impl ModelConfig {
    /// Loads model configuration from a JSON file.
    ///
    /// # Arguments
    /// * `json_str` - JSON string containing model configuration
    ///
    /// # Returns
    /// Parsed ModelConfig or defaults if parsing fails
    pub fn from_json(json_str: &str) -> Self {
        // Parse the HuggingFace config format which has nested structure
        #[derive(Deserialize)]
        struct HfConfig {
            vocab_size: Option<u32>,
            num_hidden_layers: Option<u32>,
            num_attention_heads: Option<u32>,
            hidden_size: Option<u32>,
            d_kv: Option<u32>,
            audio_channels: Option<u32>,
            sampling_rate: Option<u32>,
            num_codebooks: Option<u32>,
            pad_token_id: Option<u32>,
        }

        let hf: HfConfig = serde_json::from_str(json_str).unwrap_or(HfConfig {
            vocab_size: None,
            num_hidden_layers: None,
            num_attention_heads: None,
            hidden_size: None,
            d_kv: None,
            audio_channels: None,
            sampling_rate: None,
            num_codebooks: None,
            pad_token_id: None,
        });

        let default = Self::default();
        Self {
            vocab_size: hf.vocab_size.unwrap_or(default.vocab_size),
            num_hidden_layers: hf.num_hidden_layers.unwrap_or(default.num_hidden_layers),
            num_attention_heads: hf.num_attention_heads.unwrap_or(default.num_attention_heads),
            d_model: hf.hidden_size.unwrap_or(default.d_model),
            d_kv: hf.d_kv.unwrap_or(default.d_kv),
            audio_channels: hf.audio_channels.unwrap_or(default.audio_channels),
            sample_rate: hf.sampling_rate.unwrap_or(default.sample_rate),
            codebooks: hf.num_codebooks.unwrap_or(default.codebooks),
            pad_token_id: hf.pad_token_id.unwrap_or(default.pad_token_id),
        }
    }

    /// Calculates the expected number of tokens for a given duration.
    ///
    /// MusicGen generates approximately 50 token frames per second of audio.
    pub fn tokens_for_duration(&self, duration_sec: f32) -> u32 {
        (duration_sec * 50.0).ceil() as u32
    }
}
