//! MusicGen decoder wrapper with KV cache support.
//!
//! Implements autoregressive token generation using split decoder architecture
//! with KV cache optimization for efficient inference.

use std::borrow::Cow;
use std::collections::VecDeque;
use std::path::Path;

use half::f16;
use ort::session::{Session, SessionInputValue};
use ort::value::{DynValue, Tensor};

use crate::error::{DaemonError, Result};
use crate::types::ModelConfig;

use super::delay_pattern::DelayPatternMaskIds;
use super::logits::{Logits, DEFAULT_GUIDANCE_SCALE, DEFAULT_TOP_K};

/// MusicGen decoder using split architecture with KV cache.
pub struct MusicGenDecoder {
    decoder_model: Session,
    decoder_with_past: Session,
    config: ModelConfig,
    use_fp16: bool,
}

impl MusicGenDecoder {
    /// Loads the decoder models from a directory.
    ///
    /// Expects `decoder_model.onnx` and `decoder_with_past_model.onnx` in the directory.
    pub fn load(model_dir: &Path, config: ModelConfig) -> Result<Self> {
        let decoder_path = model_dir.join("decoder_model.onnx");
        let decoder_with_past_path = model_dir.join("decoder_with_past_model.onnx");

        let decoder_model = Session::builder()
            .map_err(|e| DaemonError::model_load_failed(format!("Failed to create session: {}", e)))?
            .commit_from_file(&decoder_path)
            .map_err(|e| {
                DaemonError::model_load_failed(format!("Failed to load decoder_model.onnx: {}", e))
            })?;

        let decoder_with_past = Session::builder()
            .map_err(|e| DaemonError::model_load_failed(format!("Failed to create session: {}", e)))?
            .commit_from_file(&decoder_with_past_path)
            .map_err(|e| {
                DaemonError::model_load_failed(format!(
                    "Failed to load decoder_with_past_model.onnx: {}",
                    e
                ))
            })?;

        // Detect if using fp16 by checking model path
        let use_fp16 = model_dir
            .to_str()
            .map(|s| s.contains("fp16"))
            .unwrap_or(false);

        Ok(Self {
            decoder_model,
            decoder_with_past,
            config,
            use_fp16,
        })
    }

    /// Generates tokens autoregressively from the encoder hidden states.
    ///
    /// Returns a VecDeque of `[i64; 4]` token arrays.
    /// Note: max_len is the desired number of output tokens. We generate extra
    /// tokens to compensate for the delay pattern masking (which loses N-1 tokens
    /// at the start, where N=4 codebooks).
    pub fn generate_tokens(
        &mut self,
        encoder_hidden_states: DynValue,
        encoder_attention_mask: DynValue,
        max_len: usize,
    ) -> Result<VecDeque<[i64; 4]>> {
        self.generate_tokens_with_progress(encoder_hidden_states, encoder_attention_mask, max_len, |_, _| {})
    }

    /// Generates tokens autoregressively with a progress callback.
    ///
    /// # Arguments
    ///
    /// * `encoder_hidden_states` - Encoded text embeddings
    /// * `encoder_attention_mask` - Attention mask for encoder
    /// * `max_len` - Number of output tokens desired
    /// * `on_progress` - Callback receiving (tokens_generated, total_tokens)
    pub fn generate_tokens_with_progress<F>(
        &mut self,
        encoder_hidden_states: DynValue,
        encoder_attention_mask: DynValue,
        max_len: usize,
        on_progress: F,
    ) -> Result<VecDeque<[i64; 4]>>
    where
        F: Fn(usize, usize),
    {
        // Compensate for delay pattern: we need N-1 extra tokens (where N=4 codebooks)
        // to get the desired number of output tokens
        let generation_len = max_len + 3;
        // Get model parameters
        let num_hidden_layers = self.config.num_hidden_layers as usize;
        let pad_token_id = self.config.pad_token_id;

        // Duplicate encoder states for classifier-free guidance (conditional + unconditional)
        let encoder_hidden_states = duplicate_with_zeros(&encoder_hidden_states, self.use_fp16)?;
        let encoder_attention_mask = duplicate_with_zeros_i64(&encoder_attention_mask)?;

        // Build initial inputs map
        let mut inputs: Vec<(String, DynValue)> = Vec::new();

        // Add encoder inputs
        inputs.push(("encoder_attention_mask".to_string(), encoder_attention_mask));
        inputs.push(("encoder_hidden_states".to_string(), encoder_hidden_states));

        // Add initial input_ids (batch of 8 with pad tokens)
        let initial_input_ids = Tensor::from_array(([8usize, 1], vec![pad_token_id; 8]))
            .map_err(|e| DaemonError::model_inference_failed(format!("Failed to create input_ids: {}", e)))?;
        inputs.push(("input_ids".to_string(), initial_input_ids.into_dyn()));

        // Run first pass with full decoder
        let session_inputs: Vec<(Cow<str>, SessionInputValue)> = inputs
            .iter()
            .map(|(k, v)| (Cow::from(k.as_str()), SessionInputValue::from(v.view())))
            .collect();

        let mut outputs = self.decoder_model.run(session_inputs).map_err(|e| {
            DaemonError::model_inference_failed(format!("Initial decoder inference failed: {}", e))
        })?;

        let mut delay_pattern_mask_ids = DelayPatternMaskIds::<4>::new();

        // Process first iteration logits
        let logits_value = outputs.remove("logits").ok_or_else(|| {
            DaemonError::model_inference_failed("logits not found in output")
        })?;
        let logits = Logits::from_3d_dyn_value(&logits_value)?;
        delay_pattern_mask_ids.push(
            logits
                .apply_free_guidance(DEFAULT_GUIDANCE_SCALE)
                .sample_top_k(DEFAULT_TOP_K)
                .iter()
                .map(|e| e.0),
        );

        // Extract KV cache from first pass
        let mut kv_cache: Vec<(String, DynValue)> = Vec::new();
        for j in 0..num_hidden_layers {
            let dk = outputs.remove(format!("present.{j}.decoder.key")).ok_or_else(|| {
                DaemonError::model_inference_failed(format!("present.{j}.decoder.key not found"))
            })?;
            let dv = outputs.remove(format!("present.{j}.decoder.value")).ok_or_else(|| {
                DaemonError::model_inference_failed(format!("present.{j}.decoder.value not found"))
            })?;
            let ek = outputs.remove(format!("present.{j}.encoder.key")).ok_or_else(|| {
                DaemonError::model_inference_failed(format!("present.{j}.encoder.key not found"))
            })?;
            let ev = outputs.remove(format!("present.{j}.encoder.value")).ok_or_else(|| {
                DaemonError::model_inference_failed(format!("present.{j}.encoder.value not found"))
            })?;

            kv_cache.push((format!("past_key_values.{j}.decoder.key"), dk));
            kv_cache.push((format!("past_key_values.{j}.decoder.value"), dv));
            kv_cache.push((format!("past_key_values.{j}.encoder.key"), ek));
            kv_cache.push((format!("past_key_values.{j}.encoder.value"), ev));
        }

        // Store encoder attention mask for subsequent passes
        let encoder_attention_mask = inputs
            .into_iter()
            .find(|(k, _)| k == "encoder_attention_mask")
            .map(|(_, v)| v)
            .ok_or_else(|| {
                DaemonError::model_inference_failed("encoder_attention_mask not found")
            })?;

        // Collect results
        let mut results = VecDeque::new();

        // Run autoregressive generation
        for i in 0..generation_len {
            // Call progress callback with current token count
            on_progress(i, generation_len);
            let [a, b, c, d] = delay_pattern_mask_ids.last_delayed_masked(pad_token_id);

            // Create new input_ids
            let input_ids = Tensor::from_array(([8usize, 1], vec![a, b, c, d, a, b, c, d]))
                .map_err(|e| DaemonError::model_inference_failed(format!("Failed to create input_ids: {}", e)))?;

            // Build inputs for decoder_with_past
            let mut session_inputs: Vec<(Cow<str>, SessionInputValue)> = vec![
                (Cow::from("input_ids"), SessionInputValue::from(input_ids.view())),
                (Cow::from("encoder_attention_mask"), SessionInputValue::from(encoder_attention_mask.view())),
            ];

            for (k, v) in &kv_cache {
                session_inputs.push((Cow::from(k.as_str()), SessionInputValue::from(v.view())));
            }

            let mut outputs = self.decoder_with_past.run(session_inputs).map_err(|e| {
                DaemonError::model_inference_failed(format!(
                    "Decoder with past inference failed: {}",
                    e
                ))
            })?;

            let logits_value = outputs.remove("logits").ok_or_else(|| {
                DaemonError::model_inference_failed("logits not found")
            })?;
            let logits = Logits::from_3d_dyn_value(&logits_value)?;
            delay_pattern_mask_ids.push(
                logits
                    .apply_free_guidance(DEFAULT_GUIDANCE_SCALE)
                    .sample_top_k(DEFAULT_TOP_K)
                    .iter()
                    .map(|e| e.0),
            );

            if let Some(last_de_delayed) = delay_pattern_mask_ids.last_de_delayed() {
                results.push_back(last_de_delayed);
            }

            // Update KV cache (only decoder keys/values change)
            let num_layers = kv_cache.len() / 4;
            for j in 0..num_layers {
                let dk = outputs.remove(format!("present.{j}.decoder.key")).ok_or_else(|| {
                    DaemonError::model_inference_failed(format!("present.{j}.decoder.key not found"))
                })?;
                let dv = outputs.remove(format!("present.{j}.decoder.value")).ok_or_else(|| {
                    DaemonError::model_inference_failed(format!("present.{j}.decoder.value not found"))
                })?;

                kv_cache[j * 4] = (format!("past_key_values.{j}.decoder.key"), dk);
                kv_cache[j * 4 + 1] = (format!("past_key_values.{j}.decoder.value"), dv);
            }
        }

        // Final progress callback
        on_progress(generation_len, generation_len);

        Ok(results)
    }
}

/// Duplicates a tensor along the first dimension, filling new entries with zeros.
/// Used for classifier-free guidance where we need both conditional and unconditional embeddings.
/// Automatically detects f16 vs f32 tensor type.
fn duplicate_with_zeros(tensor: &DynValue, _use_fp16: bool) -> Result<DynValue> {
    // Try f16 first (common for fp16 models), then f32
    if let Ok(result) = duplicate_with_zeros_typed::<f16>(tensor) {
        return Ok(result);
    }
    duplicate_with_zeros_typed::<f32>(tensor)
}

fn duplicate_with_zeros_typed<T>(tensor: &DynValue) -> Result<DynValue>
where
    T: ort::tensor::PrimitiveTensorElementType + Clone + Default + std::fmt::Debug + 'static,
{
    let (shape, data_slice) = tensor.try_extract_tensor::<T>().map_err(|e| {
        DaemonError::model_inference_failed(format!("Failed to extract tensor: {}", e))
    })?;

    let shape_vec: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
    let data: Vec<T> = data_slice.to_vec();

    let mut new_shape = shape_vec;
    new_shape[0] *= 2;

    let zeros = vec![T::default(); data.len()];
    let combined: Vec<T> = data.into_iter().chain(zeros).collect();

    let result = Tensor::from_array((new_shape, combined)).map_err(|e| {
        DaemonError::model_inference_failed(format!("Failed to create duplicated tensor: {}", e))
    })?;

    Ok(result.into_dyn())
}

fn duplicate_with_zeros_i64(tensor: &DynValue) -> Result<DynValue> {
    let (shape, data_slice) = tensor.try_extract_tensor::<i64>().map_err(|e| {
        DaemonError::model_inference_failed(format!("Failed to extract i64 tensor: {}", e))
    })?;

    let shape_vec: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
    let data: Vec<i64> = data_slice.to_vec();

    let mut new_shape = shape_vec;
    new_shape[0] *= 2;

    let zeros = vec![0i64; data.len()];
    let combined: Vec<i64> = data.into_iter().chain(zeros).collect();

    let result = Tensor::from_array((new_shape, combined)).map_err(|e| {
        DaemonError::model_inference_failed(format!("Failed to create duplicated i64 tensor: {}", e))
    })?;

    Ok(result.into_dyn())
}

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder_test() {
        // Model loading tests require actual model files
        assert!(true);
    }
}
