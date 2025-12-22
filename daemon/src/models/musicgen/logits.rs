//! Logits processing for MusicGen decoder output.
//!
//! Handles classifier-free guidance and top-k sampling for token generation.

use std::fmt::{Debug, Formatter};
use std::ops::{Deref, DerefMut};

use half::f16;
use ndarray::{s, Array, Array2, Axis, Ix3, IxDyn};
use ort::tensor::ArrayExtensions;
use ort::value::DynValue;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::thread_rng;

use crate::error::{DaemonError, Result};

/// Wrapper around 2D logits array with processing methods.
pub struct Logits(Array2<f32>);

impl Deref for Logits {
    type Target = Array2<f32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Logits {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Debug for Logits {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Logits({:?})", self.0.dim())
    }
}

impl Logits {
    /// Creates Logits from a 3D DynValue, supporting both f32 and f16.
    ///
    /// The input shape is expected to be [batch_size, decoder_sequence_length, vocab_size].
    /// Since decoder_sequence_length is always 1, we remove that axis.
    pub fn from_3d_dyn_value(value: &DynValue) -> Result<Self> {
        let (shape, data): (Vec<usize>, Vec<f32>) =
            if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
                let shape_vec: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
                (shape_vec, data.to_vec())
            } else if let Ok((shape, data)) = value.try_extract_tensor::<f16>() {
                let shape_vec: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
                let data_f32: Vec<f32> = data.iter().map(|e| f32::from(*e)).collect();
                (shape_vec, data_f32)
            } else {
                return Err(DaemonError::model_inference_failed(
                    "Logits must be f32 or f16",
                ));
            };

        // Create ndarray from raw data
        let arr = Array::from_shape_vec(IxDyn(&shape), data)
            .map_err(|e| DaemonError::model_inference_failed(format!("Failed to create array: {}", e)))?;

        let arr = arr
            .into_dimensionality::<Ix3>()
            .map_err(|e| DaemonError::model_inference_failed(format!("Expected 3D logits: {}", e)))?;

        // logits come in the following shape float32[batch_size,decoder_sequence_length,2048]
        // based on transformers.js we can assume that decoder_sequence_length is going
        // to be 1, so we can just remove it.
        let arr = arr.remove_axis(Axis(1));
        Ok(Self(arr))
    }

    /// Applies classifier-free guidance to the logits.
    ///
    /// The batch is expected to have conditional logits in the first half
    /// and unconditional logits in the second half. The formula applied is:
    /// `guided = uncond + (cond - uncond) * scale`
    ///
    /// # Panics
    ///
    /// Panics if the first dimension is not even.
    pub fn apply_free_guidance(self, guidance_scale: usize) -> Self {
        if !self.0.dim().0.is_multiple_of(2) {
            panic!("In order to apply free guidance to the logits, the first size of the first dimension must be even")
        }

        let unguided_bsz = self.0.dim().0 / 2;
        let cond_logits = self.0.slice(s![0..unguided_bsz, ..]);
        let uncond_logits = self.0.slice(s![unguided_bsz.., ..]);

        // Based on transformers.js, src/generation/logits_process.js#L603:
        // scores = uncond_logits + (cond_logits - uncond_logits) * guidance_scale
        Self((cond_logits.into_owned() - uncond_logits) * guidance_scale as f32 + uncond_logits)
    }

    /// Samples from the logits using top-k sampling.
    ///
    /// Returns a vector of (token_id, log_probability) pairs, one per batch entry.
    ///
    /// # Arguments
    ///
    /// * `k` - Take into account only top k logits in each batch
    pub fn sample_top_k(&self, k: usize) -> Vec<(i64, f32)> {
        let mut result = vec![];
        let softmax_logits = self.0.softmax(Axis(1));

        for batch in softmax_logits.axis_iter(Axis(0)) {
            let k = k.min(batch.len());

            // Vec<(token_id, softmax_prob)>
            let mut softmax_logits_batch = batch
                .iter()
                .enumerate()
                .map(|(i, e)| (i as i64, *e))
                .collect::<Vec<_>>();

            // Sort based on softmax_prob in order to bring the most probable tokens to the front.
            softmax_logits_batch.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .expect("Could not compare two numbers in order to sort them")
            });

            // Trim based on provided k.
            softmax_logits_batch = softmax_logits_batch[0..k].to_vec();

            // Create a distribution based on the softmax probabilities.
            let distribution = WeightedIndex::new(softmax_logits_batch.iter().map(|e| e.1))
                .expect("Could not create WeightedIndex distribution");

            // Sample a random index based on the softmax probabilities.
            let (idx, softmax_prob) = softmax_logits_batch[distribution.sample(&mut thread_rng())];

            // Use natural log for log probability
            result.push((idx, softmax_prob.ln()));
        }
        result
    }
}

/// Default guidance scale for MusicGen.
pub const DEFAULT_GUIDANCE_SCALE: usize = 3;

/// Default top-k value for sampling.
pub const DEFAULT_TOP_K: usize = 250;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn free_guidance() {
        let arr = Array::from_shape_vec((2, 3), vec![10., -1., 3., -1., 1., 11.]).unwrap();
        let logits = Logits(arr);
        let logits = logits.apply_free_guidance(3);
        assert_eq!(logits.shape(), &[1, 3]);
    }

    #[test]
    fn sample_top_k_returns_valid_indices() {
        let arr = Array::from_shape_vec((2, 3), vec![0.1, 0.2, 0.7, 0.3, 0.4, 0.3]).unwrap();
        let logits = Logits(arr);
        let samples = logits.sample_top_k(2);
        assert_eq!(samples.len(), 2);
        for (idx, _log_prob) in &samples {
            assert!(*idx >= 0 && *idx < 3);
        }
    }
}
