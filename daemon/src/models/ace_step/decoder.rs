//! DCAE latent decoder for ACE-Step.
//!
//! Wraps the MusicDCAE ONNX model for decoding latent representations
//! into mel-spectrograms.
//!
//! Note: The ONNX model has a fixed input size of 128 frames.
//! For longer audio, we decode in chunks and concatenate.

use std::path::Path;

use ndarray::{s, Array3, Array4, Axis};
use ort::execution_providers::ExecutionProviderDispatch;
use ort::session::Session;
use ort::value::Tensor;

use crate::error::{DaemonError, Result};

use super::models::load_session;

/// Number of mel frequency bins in the spectrogram output.
pub const MEL_BINS: usize = 128;

/// Hop length for mel spectrogram (samples between frames).
pub const HOP_LENGTH: usize = 512;

/// Maximum frames per decode chunk (ONNX model limit).
pub const MAX_DECODE_FRAMES: usize = 128;

/// DCAE (Deep Convolutional AutoEncoder) decoder for ACE-Step.
///
/// Converts latent representations from the diffusion process into
/// mel-spectrograms that can be vocoded into audio.
pub struct DcaeDecoder {
    /// The ONNX session for the DCAE decoder.
    session: Session,
}

impl std::fmt::Debug for DcaeDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DcaeDecoder")
            .finish_non_exhaustive()
    }
}

impl DcaeDecoder {
    /// Loads the DCAE decoder from the model directory.
    ///
    /// # Arguments
    ///
    /// * `model_dir` - Directory containing `dcae_decoder.onnx`
    /// * `providers` - Execution providers for ONNX Runtime
    pub fn load(model_dir: &Path, providers: &[ExecutionProviderDispatch]) -> Result<Self> {
        let decoder_path = model_dir.join("dcae_decoder.onnx");
        let session = load_session(&decoder_path, providers)?;
        Ok(Self { session })
    }

    /// Decodes latent representation to mel-spectrogram.
    ///
    /// For latents longer than 128 frames, decodes in chunks and concatenates.
    ///
    /// # Arguments
    ///
    /// * `latent` - Latent representation from diffusion, shape (1, channels, height, frame_length)
    ///
    /// # Returns
    ///
    /// Mel-spectrogram with shape (1, mel_bins, time_frames).
    pub fn decode(&mut self, latent: &Array4<f32>) -> Result<Array3<f32>> {
        let frame_length = latent.shape()[3];

        if frame_length <= MAX_DECODE_FRAMES {
            // Single chunk - decode directly
            self.decode_chunk(latent)
        } else {
            // Multiple chunks needed
            let num_chunks = (frame_length + MAX_DECODE_FRAMES - 1) / MAX_DECODE_FRAMES;
            eprintln!("Decoding in {} chunks of {} frames...", num_chunks, MAX_DECODE_FRAMES);

            let mut mel_chunks: Vec<Array3<f32>> = Vec::new();

            for i in 0..num_chunks {
                let start = i * MAX_DECODE_FRAMES;
                let end = ((i + 1) * MAX_DECODE_FRAMES).min(frame_length);
                let chunk_len = end - start;

                // Extract chunk - need to pad to 128 if smaller
                let chunk = if chunk_len < MAX_DECODE_FRAMES {
                    // Pad the last chunk with zeros
                    let mut padded = Array4::<f32>::zeros((1, 8, 16, MAX_DECODE_FRAMES));
                    padded.slice_mut(s![.., .., .., ..chunk_len])
                        .assign(&latent.slice(s![.., .., .., start..end]));
                    padded
                } else {
                    latent.slice(s![.., .., .., start..end]).to_owned()
                };

                let mel_chunk = self.decode_chunk(&chunk)?;

                // If padded, trim the mel output proportionally
                if chunk_len < MAX_DECODE_FRAMES {
                    let mel_frames = mel_chunk.shape()[2];
                    let expected_frames = (mel_frames * chunk_len) / MAX_DECODE_FRAMES;
                    let trimmed = mel_chunk.slice(s![.., .., ..expected_frames]).to_owned();
                    mel_chunks.push(trimmed);
                } else {
                    mel_chunks.push(mel_chunk);
                }
            }

            // Concatenate along time axis
            let views: Vec<_> = mel_chunks.iter().map(|c| c.view()).collect();
            let concatenated = ndarray::concatenate(Axis(2), &views)
                .map_err(|e| DaemonError::model_inference_failed(format!("Failed to concatenate mel chunks: {}", e)))?;

            Ok(concatenated)
        }
    }

    /// Decodes a single chunk (must be exactly 128 frames or less with padding).
    fn decode_chunk(&mut self, latent: &Array4<f32>) -> Result<Array3<f32>> {
        let shape = latent.shape();
        let data: Vec<f32> = latent.iter().copied().collect();
        let latent_tensor = Tensor::from_array(([shape[0], shape[1], shape[2], shape[3]], data))
            .map_err(|e| DaemonError::model_inference_failed(format!("Failed to create latent tensor: {}", e)))?;

        let mut outputs = self
            .session
            .run(ort::inputs!["latents" => latent_tensor])
            .map_err(|e| DaemonError::model_inference_failed(format!("DCAE decoder failed: {}", e)))?;

        // Get mel_spectrogram output
        let mel = outputs.remove("mel_spectrogram").ok_or_else(|| {
            DaemonError::model_inference_failed("Missing mel_spectrogram output".to_string())
        })?;

        let (mel_shape, mel_data) = mel
            .try_extract_tensor::<f32>()
            .map_err(|e| DaemonError::model_inference_failed(format!("Failed to extract mel spectrogram: {}", e)))?;

        let dims: Vec<usize> = mel_shape.iter().map(|&d| d as usize).collect();

        // Handle 4D output (1, 2, mel_bins, time) or 3D output (1, mel_bins, time)
        // Take first channel if 4D with 2 channels
        let output = if dims.len() == 4 {
            // Shape is (1, 2, mel_bins, time) - take first channel
            let channel_size = dims[2] * dims[3];
            let first_channel: Vec<f32> = mel_data.iter()
                .take(channel_size)
                .copied()
                .collect();
            Array3::from_shape_vec(
                (dims[0], dims[2], dims[3]),
                first_channel,
            )
            .map_err(|e| DaemonError::model_inference_failed(format!("Failed to reshape mel: {}", e)))?
        } else if dims.len() == 3 {
            Array3::from_shape_vec(
                (dims[0], dims[1], dims[2]),
                mel_data.to_vec(),
            )
            .map_err(|e| DaemonError::model_inference_failed(format!("Failed to reshape mel: {}", e)))?
        } else {
            return Err(DaemonError::model_inference_failed(format!(
                "Unexpected DCAE output shape: {:?}",
                dims
            )));
        };

        Ok(output)
    }

    /// Estimates the output time frames from latent frame length.
    ///
    /// The DCAE has an 8x compression ratio.
    pub fn estimate_output_frames(latent_frame_length: usize) -> usize {
        latent_frame_length * 8
    }

    /// Estimates audio samples from mel spectrogram time frames.
    ///
    /// Each mel frame corresponds to HOP_LENGTH samples.
    pub fn estimate_samples(mel_time_frames: usize) -> usize {
        mel_time_frames * HOP_LENGTH
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_dimensions() {
        assert_eq!(MEL_BINS, 128);
        assert_eq!(HOP_LENGTH, 512);
    }

    #[test]
    fn estimate_output_frames_8x() {
        assert_eq!(DcaeDecoder::estimate_output_frames(100), 800);
        assert_eq!(DcaeDecoder::estimate_output_frames(323), 2584);
    }

    #[test]
    fn estimate_samples_hop_length() {
        // 800 frames * 512 hop = 409600 samples
        assert_eq!(DcaeDecoder::estimate_samples(800), 409600);
    }
}
