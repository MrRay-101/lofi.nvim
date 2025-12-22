//! Diffusion transformer for ACE-Step.
//!
//! Wraps the ACE-Step transformer ONNX models (encoder and decoder)
//! for the diffusion denoising process.
//!
//! ## Model Signatures
//!
//! **transformer_encoder.onnx:**
//! - Inputs:
//!   - `encoder_text_hidden_states`: (batch, text_seq_len, 768)
//!   - `text_attention_mask`: (batch, text_seq_len)
//!   - `speaker_embeds`: (batch, 512)
//!   - `lyric_token_idx`: (batch, lyric_seq_len)
//!   - `lyric_mask`: (batch, lyric_seq_len)
//! - Outputs:
//!   - `encoder_hidden_states`: (batch, total_seq_len, 2560)
//!   - `encoder_hidden_mask`: (batch, total_seq_len)
//!
//! **transformer_decoder.onnx:**
//! - Inputs:
//!   - `hidden_states`: (batch, 8, 16, frame_length)
//!   - `attention_mask`: (batch, frame_length)
//!   - `encoder_hidden_states`: (batch, encoder_seq_len, 2560)
//!   - `encoder_hidden_mask`: (batch, encoder_seq_len)
//!   - `timestep`: (1,)
//! - Outputs:
//!   - `sample`: (batch, 8, 16, frame_length)

use std::path::Path;

use ndarray::{Array2, Array3, Array4};
use ort::execution_providers::ExecutionProviderDispatch;
use ort::session::Session;
use ort::value::Tensor;

use crate::error::{DaemonError, Result};

use super::models::load_session;

/// Number of channels in the latent space.
pub const LATENT_CHANNELS: usize = 8;

/// Height of the latent representation.
pub const LATENT_HEIGHT: usize = 16;

/// Dimension of speaker embeddings.
pub const SPEAKER_EMBED_DIM: usize = 512;

/// Dimension of transformer encoder output.
pub const ENCODER_HIDDEN_DIM: usize = 2560;

/// Diffusion transformer for ACE-Step noise prediction.
pub struct DiffusionTransformer {
    encoder: Session,
    decoder: Session,
}

impl std::fmt::Debug for DiffusionTransformer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiffusionTransformer")
            .finish_non_exhaustive()
    }
}

impl DiffusionTransformer {
    /// Loads the diffusion transformer from the model directory.
    pub fn load(model_dir: &Path, providers: &[ExecutionProviderDispatch]) -> Result<Self> {
        let encoder_path = model_dir.join("transformer_encoder.onnx");
        let decoder_path = model_dir.join("transformer_decoder.onnx");

        let encoder = load_session(&encoder_path, providers)?;
        let decoder = load_session(&decoder_path, providers)?;

        Ok(Self { encoder, decoder })
    }

    /// Encodes text embeddings into transformer context.
    ///
    /// For instrumental generation, speaker_embeds and lyrics are zeros.
    ///
    /// # Arguments
    ///
    /// * `text_hidden_states` - Text encoder output, shape (batch, seq_len, 768)
    /// * `text_attention_mask` - Attention mask, shape (batch, seq_len)
    ///
    /// # Returns
    ///
    /// Tuple of (encoder_hidden_states, encoder_hidden_mask) with hidden dim 2560.
    pub fn encode_context(
        &mut self,
        text_hidden_states: &Array3<f32>,
        text_attention_mask: &Array2<i64>,
    ) -> Result<(Array3<f32>, Array2<f32>)> {
        let batch_size = text_hidden_states.shape()[0];
        let text_seq_len = text_hidden_states.shape()[1];

        // Create tensors for text
        let text_hs_data: Vec<f32> = text_hidden_states.iter().copied().collect();
        let text_hs_tensor = Tensor::from_array((
            [batch_size, text_seq_len, 768],
            text_hs_data,
        ))
        .map_err(|e| DaemonError::model_inference_failed(format!("Failed to create text_hidden_states tensor: {}", e)))?;

        let text_mask_data: Vec<i64> = text_attention_mask.iter().copied().collect();
        let text_mask_tensor = Tensor::from_array((
            [batch_size, text_seq_len],
            text_mask_data,
        ))
        .map_err(|e| DaemonError::model_inference_failed(format!("Failed to create text_attention_mask tensor: {}", e)))?;

        // Create zero speaker embeddings for instrumental generation
        let speaker_data: Vec<f32> = vec![0.0; batch_size * SPEAKER_EMBED_DIM];
        let speaker_tensor = Tensor::from_array((
            [batch_size, SPEAKER_EMBED_DIM],
            speaker_data,
        ))
        .map_err(|e| DaemonError::model_inference_failed(format!("Failed to create speaker_embeds tensor: {}", e)))?;

        // Create empty lyrics (single padding token) for instrumental generation
        let lyric_seq_len = 1;
        let lyric_data: Vec<i64> = vec![0; batch_size * lyric_seq_len];
        let lyric_tensor = Tensor::from_array((
            [batch_size, lyric_seq_len],
            lyric_data,
        ))
        .map_err(|e| DaemonError::model_inference_failed(format!("Failed to create lyric_token_idx tensor: {}", e)))?;

        let lyric_mask_data: Vec<i64> = vec![0; batch_size * lyric_seq_len];
        let lyric_mask_tensor = Tensor::from_array((
            [batch_size, lyric_seq_len],
            lyric_mask_data,
        ))
        .map_err(|e| DaemonError::model_inference_failed(format!("Failed to create lyric_mask tensor: {}", e)))?;

        // Run encoder with named inputs
        let mut outputs = self
            .encoder
            .run(ort::inputs![
                "encoder_text_hidden_states" => text_hs_tensor,
                "text_attention_mask" => text_mask_tensor,
                "speaker_embeds" => speaker_tensor,
                "lyric_token_idx" => lyric_tensor,
                "lyric_mask" => lyric_mask_tensor,
            ])
            .map_err(|e| DaemonError::model_inference_failed(format!("Transformer encoder failed: {}", e)))?;

        // Extract encoder_hidden_states
        let hidden_states = outputs.remove("encoder_hidden_states").ok_or_else(|| {
            DaemonError::model_inference_failed("Missing encoder_hidden_states output".to_string())
        })?;
        let (hs_shape, hs_data) = hidden_states
            .try_extract_tensor::<f32>()
            .map_err(|e| DaemonError::model_inference_failed(format!("Failed to extract encoder_hidden_states: {}", e)))?;
        let hs_dims: Vec<usize> = hs_shape.iter().map(|&d| d as usize).collect();
        let encoder_hidden_states = Array3::from_shape_vec(
            (hs_dims[0], hs_dims[1], hs_dims[2]),
            hs_data.to_vec(),
        )
        .map_err(|e| DaemonError::model_inference_failed(format!("Failed to reshape encoder_hidden_states: {}", e)))?;

        // Extract encoder_hidden_mask (f32, will convert to i64 for decoder)
        let hidden_mask = outputs.remove("encoder_hidden_mask").ok_or_else(|| {
            DaemonError::model_inference_failed("Missing encoder_hidden_mask output".to_string())
        })?;
        let (mask_shape, mask_data) = hidden_mask
            .try_extract_tensor::<f32>()
            .map_err(|e| DaemonError::model_inference_failed(format!("Failed to extract encoder_hidden_mask: {}", e)))?;
        let mask_dims: Vec<usize> = mask_shape.iter().map(|&d| d as usize).collect();
        let encoder_hidden_mask = Array2::from_shape_vec(
            (mask_dims[0], mask_dims[1]),
            mask_data.to_vec(),
        )
        .map_err(|e| DaemonError::model_inference_failed(format!("Failed to reshape encoder_hidden_mask: {}", e)))?;

        Ok((encoder_hidden_states, encoder_hidden_mask))
    }

    /// Predicts noise for the given latent at a specific timestep.
    ///
    /// # Arguments
    ///
    /// * `latent` - Noisy latent, shape (batch, 8, 16, frame_length)
    /// * `timestep` - Current diffusion timestep (scalar)
    /// * `encoder_hidden_states` - From encode_context, shape (batch, seq_len, 2560)
    /// * `encoder_hidden_mask` - From encode_context, shape (batch, seq_len)
    ///
    /// # Returns
    ///
    /// Noise prediction with same shape as input latent.
    pub fn predict_noise(
        &mut self,
        latent: &Array4<f32>,
        timestep: f32,
        encoder_hidden_states: &Array3<f32>,
        encoder_hidden_mask: &Array2<f32>,
    ) -> Result<Array4<f32>> {
        let batch_size = latent.shape()[0];
        let frame_length = latent.shape()[3];
        let encoder_seq_len = encoder_hidden_states.shape()[1];

        // Create latent tensor (hidden_states in model terms)
        let latent_data: Vec<f32> = latent.iter().copied().collect();
        let latent_tensor = Tensor::from_array((
            [batch_size, LATENT_CHANNELS, LATENT_HEIGHT, frame_length],
            latent_data,
        ))
        .map_err(|e| DaemonError::model_inference_failed(format!("Failed to create hidden_states tensor: {}", e)))?;

        // Create attention mask for latent frames (all ones = attend to all)
        let attn_mask_data: Vec<f32> = vec![1.0; batch_size * frame_length];
        let attn_mask_tensor = Tensor::from_array((
            [batch_size, frame_length],
            attn_mask_data,
        ))
        .map_err(|e| DaemonError::model_inference_failed(format!("Failed to create attention_mask tensor: {}", e)))?;

        // Create encoder hidden states tensor
        let enc_hs_data: Vec<f32> = encoder_hidden_states.iter().copied().collect();
        let enc_hs_tensor = Tensor::from_array((
            [batch_size, encoder_seq_len, ENCODER_HIDDEN_DIM],
            enc_hs_data,
        ))
        .map_err(|e| DaemonError::model_inference_failed(format!("Failed to create encoder_hidden_states tensor: {}", e)))?;

        // Create encoder hidden mask tensor
        let enc_mask_data: Vec<f32> = encoder_hidden_mask.iter().copied().collect();
        let enc_mask_tensor = Tensor::from_array((
            [batch_size, encoder_seq_len],
            enc_mask_data,
        ))
        .map_err(|e| DaemonError::model_inference_failed(format!("Failed to create encoder_hidden_mask tensor: {}", e)))?;

        // Create timestep tensor
        let timestep_tensor = Tensor::from_array(([1], vec![timestep]))
            .map_err(|e| DaemonError::model_inference_failed(format!("Failed to create timestep tensor: {}", e)))?;

        // Run decoder with named inputs
        let mut outputs = self
            .decoder
            .run(ort::inputs![
                "hidden_states" => latent_tensor,
                "attention_mask" => attn_mask_tensor,
                "encoder_hidden_states" => enc_hs_tensor,
                "encoder_hidden_mask" => enc_mask_tensor,
                "timestep" => timestep_tensor,
            ])
            .map_err(|e| DaemonError::model_inference_failed(format!("Transformer decoder failed: {}", e)))?;

        // Extract sample output
        let sample = outputs.remove("sample").ok_or_else(|| {
            DaemonError::model_inference_failed("Missing sample output".to_string())
        })?;
        let (sample_shape, sample_data) = sample
            .try_extract_tensor::<f32>()
            .map_err(|e| DaemonError::model_inference_failed(format!("Failed to extract sample: {}", e)))?;
        let sample_dims: Vec<usize> = sample_shape.iter().map(|&d| d as usize).collect();
        let noise_pred = Array4::from_shape_vec(
            (sample_dims[0], sample_dims[1], sample_dims[2], sample_dims[3]),
            sample_data.to_vec(),
        )
        .map_err(|e| DaemonError::model_inference_failed(format!("Failed to reshape sample: {}", e)))?;

        Ok(noise_pred)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latent_dimensions() {
        assert_eq!(LATENT_CHANNELS, 8);
        assert_eq!(LATENT_HEIGHT, 16);
    }

    #[test]
    fn encoder_hidden_dim() {
        assert_eq!(ENCODER_HIDDEN_DIM, 2560);
    }

    #[test]
    fn speaker_embed_dim() {
        assert_eq!(SPEAKER_EMBED_DIM, 512);
    }
}
