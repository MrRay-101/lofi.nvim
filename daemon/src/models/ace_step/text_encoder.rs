//! UMT5 text encoder for ACE-Step.
//!
//! Wraps the UMT5 ONNX model for encoding text prompts into embeddings
//! that condition the diffusion transformer.

use std::path::Path;

use ndarray::{Array2, Array3, Axis};
use ort::execution_providers::ExecutionProviderDispatch;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::error::{DaemonError, Result};

use super::models::load_session;

/// Maximum sequence length for text encoding.
pub const MAX_SEQ_LENGTH: usize = 512;

/// UMT5 text encoder for ACE-Step prompt conditioning.
///
/// The UMT5 (Universal Multilingual T5) encoder converts text prompts into
/// dense embeddings that guide the diffusion process. Output dimension is 768.
pub struct Umt5TextEncoder {
    /// The ONNX session for the text encoder.
    session: Session,
    /// The tokenizer for text preprocessing.
    tokenizer: Tokenizer,
}

impl std::fmt::Debug for Umt5TextEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Umt5TextEncoder")
            .finish_non_exhaustive()
    }
}

impl Umt5TextEncoder {
    /// Loads the UMT5 text encoder from the model directory.
    ///
    /// # Arguments
    ///
    /// * `model_dir` - Directory containing `text_encoder.onnx` and `tokenizer.json`
    /// * `providers` - Execution providers for ONNX Runtime
    pub fn load(model_dir: &Path, providers: &[ExecutionProviderDispatch]) -> Result<Self> {
        let encoder_path = model_dir.join("text_encoder.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        // Load the ONNX session
        let session = load_session(&encoder_path, providers)?;

        // Load the tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            DaemonError::model_load_failed(format!("Failed to load tokenizer: {}", e))
        })?;

        Ok(Self { session, tokenizer })
    }

    /// Encodes a text prompt into hidden states.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The text prompt to encode
    ///
    /// # Returns
    ///
    /// A tuple of:
    /// - `encoder_hidden_states`: Shape (1, seq_len, 768) - text embeddings
    /// - `encoder_attention_mask`: Shape (1, seq_len) - attention mask
    pub fn encode(&mut self, prompt: &str) -> Result<(Array3<f32>, Array2<i64>)> {
        // Tokenize the prompt
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| DaemonError::model_inference_failed(format!("Tokenization failed: {}", e)))?;

        let token_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&m| m as i64).collect();

        // Truncate if needed
        let seq_len = token_ids.len().min(MAX_SEQ_LENGTH);
        let token_ids = token_ids[..seq_len].to_vec();
        let attention_mask = attention_mask[..seq_len].to_vec();

        // Create ONNX tensors using the shape-data tuple pattern
        let input_ids_tensor = Tensor::from_array(([1, seq_len], token_ids.clone()))
            .map_err(|e| DaemonError::model_inference_failed(format!("Failed to create input_ids tensor: {}", e)))?;
        let attention_mask_tensor = Tensor::from_array(([1, seq_len], attention_mask.clone()))
            .map_err(|e| DaemonError::model_inference_failed(format!("Failed to create attention_mask tensor: {}", e)))?;

        // Run the encoder
        let mut outputs = self
            .session
            .run(ort::inputs![input_ids_tensor, attention_mask_tensor])
            .map_err(|e| DaemonError::model_inference_failed(format!("Encoder inference failed: {}", e)))?;

        // Extract encoder hidden states - shape (1, seq_len, 768)
        let output_key = outputs.keys().next().map(|s| s.to_string()).ok_or_else(|| {
            DaemonError::model_inference_failed("Missing encoder output tensor".to_string())
        })?;
        let hidden_states = outputs.remove(&output_key).ok_or_else(|| {
            DaemonError::model_inference_failed("Failed to remove encoder output".to_string())
        })?;

        let (shape, data) = hidden_states
            .try_extract_tensor::<f32>()
            .map_err(|e| DaemonError::model_inference_failed(format!("Failed to extract hidden states: {}", e)))?;

        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let hidden_states_array = Array3::from_shape_vec(
            (dims[0], dims[1], dims[2]),
            data.to_vec(),
        )
        .map_err(|e| DaemonError::model_inference_failed(format!("Failed to reshape hidden states: {}", e)))?;

        // Create attention mask array for return
        let attention_mask_array = Array2::from_shape_vec((1, seq_len), attention_mask)
            .map_err(|e| DaemonError::model_inference_failed(format!("Failed to create attention mask array: {}", e)))?;

        Ok((hidden_states_array, attention_mask_array))
    }

    /// Encodes a text prompt with pooled output for conditioning.
    ///
    /// Returns the mean-pooled embedding across the sequence dimension.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The text prompt to encode
    ///
    /// # Returns
    ///
    /// A 1D array of shape (768,) representing the pooled text embedding.
    pub fn encode_pooled(&mut self, prompt: &str) -> Result<ndarray::Array1<f32>> {
        let (hidden_states, attention_mask) = self.encode(prompt)?;

        // Mean pooling with attention mask
        let mask = attention_mask.mapv(|m| m as f32);
        let mask_for_sum = mask.clone();
        let mask_expanded = mask.insert_axis(Axis(2));

        // Sum of hidden states weighted by mask
        let masked = &hidden_states * &mask_expanded;
        let sum = masked.sum_axis(Axis(1));

        // Sum of mask
        let mask_sum = mask_for_sum.sum_axis(Axis(1)).mapv(|s| s.max(1e-9));

        // Mean
        let pooled = &sum.index_axis(Axis(0), 0) / &mask_sum.index_axis(Axis(0), 0);

        Ok(pooled.to_owned())
    }

    /// Creates an empty (unconditioned) encoding for classifier-free guidance.
    ///
    /// Returns embeddings for the empty prompt, used for the unconditional branch
    /// in classifier-free guidance.
    pub fn encode_unconditioned(&mut self, _seq_len: usize) -> Result<(Array3<f32>, Array2<i64>)> {
        // Use empty or padding tokens
        self.encode("")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_seq_length_reasonable() {
        assert!(MAX_SEQ_LENGTH >= 64);
        assert!(MAX_SEQ_LENGTH <= 1024);
    }
}
