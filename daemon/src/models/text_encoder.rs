//! Text encoder wrapper for MusicGen.
//!
//! Handles tokenization and T5 text encoding for text prompts.

use std::path::Path;

use ort::execution_providers::ExecutionProviderDispatch;
use ort::session::Session;
use ort::value::{DynValue, Tensor};
use tokenizers::Tokenizer;

use crate::error::{DaemonError, Result};

/// MusicGen text encoder combining tokenizer and T5 encoder.
pub struct MusicGenTextEncoder {
    tokenizer: Tokenizer,
    text_encoder: Session,
}

impl MusicGenTextEncoder {
    /// Creates a new text encoder from model directory.
    ///
    /// Loads `tokenizer.json` and `text_encoder.onnx` from the given directory.
    pub fn load(model_dir: &Path) -> Result<Self> {
        Self::load_with_providers(model_dir, &[])
    }

    /// Creates a new text encoder from model directory with specific execution providers.
    ///
    /// Loads `tokenizer.json` and `text_encoder.onnx` from the given directory,
    /// using the provided execution providers for the ONNX session.
    pub fn load_with_providers(
        model_dir: &Path,
        providers: &[ExecutionProviderDispatch],
    ) -> Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let encoder_path = model_dir.join("text_encoder.onnx");

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            DaemonError::model_load_failed(format!("Failed to load tokenizer: {}", e))
        })?;

        tokenizer
            .with_padding(None)
            .with_truncation(None)
            .map_err(|e| {
                DaemonError::model_load_failed(format!("Failed to configure tokenizer: {}", e))
            })?;

        let mut builder = Session::builder()
            .map_err(|e| DaemonError::model_load_failed(format!("Failed to create session: {}", e)))?;

        if !providers.is_empty() {
            builder = builder.with_execution_providers(providers).map_err(|e| {
                DaemonError::model_load_failed(format!("Failed to set execution providers: {}", e))
            })?;
        }

        let text_encoder = builder.commit_from_file(&encoder_path).map_err(|e| {
            DaemonError::model_load_failed(format!("Failed to load text_encoder.onnx: {}", e))
        })?;

        Ok(Self {
            tokenizer,
            text_encoder,
        })
    }

    /// Encodes text into embeddings and attention mask.
    ///
    /// Returns a tuple of (last_hidden_state, attention_mask) as DynValue tensors.
    pub fn encode(&mut self, text: &str) -> Result<(DynValue, DynValue)> {
        let tokens = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| {
                DaemonError::model_inference_failed(format!("Tokenization failed: {}", e))
            })?
            .get_ids()
            .iter()
            .map(|e| *e as i64)
            .collect::<Vec<_>>();

        let tokens_len = tokens.len();

        // Create input tensors
        let input_ids = Tensor::from_array(([1, tokens_len], tokens)).map_err(|e| {
            DaemonError::model_inference_failed(format!("Failed to create input tensor: {}", e))
        })?;

        let attention_mask_data: Vec<i64> = vec![1; tokens_len];
        let attention_mask = Tensor::from_array(([1, tokens_len], attention_mask_data)).map_err(|e| {
            DaemonError::model_inference_failed(format!("Failed to create attention mask: {}", e))
        })?;

        // Run the text encoder
        let mut output = self
            .text_encoder
            .run(ort::inputs![input_ids, attention_mask])
            .map_err(|e| {
                DaemonError::model_inference_failed(format!("Text encoder inference failed: {}", e))
            })?;

        let last_hidden_state = output
            .remove("last_hidden_state")
            .ok_or_else(|| {
                DaemonError::model_inference_failed(
                    "last_hidden_state not found in output",
                )
            })?;

        // Create attention mask for decoder
        let decoder_attention_mask_data: Vec<i64> = vec![1; tokens_len];
        let decoder_attention_mask = Tensor::from_array(([1, tokens_len], decoder_attention_mask_data))
            .map_err(|e| {
                DaemonError::model_inference_failed(format!("Failed to create decoder attention mask: {}", e))
            })?;

        Ok((last_hidden_state, decoder_attention_mask.into_dyn()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn get_model_dir() -> Option<PathBuf> {
        let proj_dirs = directories::ProjectDirs::from("", "", "lofi-daemon")?;
        let path = proj_dirs.data_dir().join("models");
        if path.exists() {
            Some(path)
        } else {
            None
        }
    }

    #[test]
    fn text_encoder_loads_successfully() {
        let Some(model_dir) = get_model_dir() else {
            eprintln!("Skipping test: models not found");
            return;
        };

        let result = MusicGenTextEncoder::load(&model_dir);
        assert!(result.is_ok(), "Failed to load text encoder: {:?}", result.err());
    }

    #[test]
    fn text_encoder_encodes_prompt() {
        let Some(model_dir) = get_model_dir() else {
            eprintln!("Skipping test: models not found");
            return;
        };

        let mut encoder = MusicGenTextEncoder::load(&model_dir).unwrap();
        let result = encoder.encode("lofi hip hop beats");
        assert!(result.is_ok(), "Failed to encode text: {:?}", result.err());

        let (hidden_state, attention_mask) = result.unwrap();
        // Verify we got tensors back
        assert!(hidden_state.try_extract_tensor::<f32>().is_ok() ||
                hidden_state.try_extract_tensor::<half::f16>().is_ok());
        assert!(attention_mask.try_extract_tensor::<i64>().is_ok());
    }
}

