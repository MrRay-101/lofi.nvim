//! ADaMoSHiFiGAN vocoder for ACE-Step.
//!
//! Wraps the ADaMoSHiFiGAN ONNX model for converting mel-spectrograms
//! into audio waveforms.

use std::path::Path;

use ndarray::{Array1, Array3};
use ort::execution_providers::ExecutionProviderDispatch;
use ort::session::Session;
use ort::value::Tensor;

use crate::error::{DaemonError, Result};

use super::models::load_session;

/// Output sample rate of the vocoder (44.1 kHz).
pub const VOCODER_SAMPLE_RATE: u32 = 44100;

/// Target sample rate for lofi.nvim output (48 kHz).
pub const TARGET_SAMPLE_RATE: u32 = 48000;

/// ADaMoSHiFiGAN vocoder for ACE-Step.
///
/// Converts mel-spectrograms from the DCAE decoder into audio waveforms
/// at 44.1 kHz sample rate.
pub struct Vocoder {
    /// The ONNX session for the vocoder.
    session: Session,
}

impl std::fmt::Debug for Vocoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vocoder")
            .finish_non_exhaustive()
    }
}

impl Vocoder {
    /// Loads the vocoder from the model directory.
    ///
    /// # Arguments
    ///
    /// * `model_dir` - Directory containing `vocoder.onnx`
    /// * `providers` - Execution providers for ONNX Runtime
    pub fn load(model_dir: &Path, providers: &[ExecutionProviderDispatch]) -> Result<Self> {
        let vocoder_path = model_dir.join("vocoder.onnx");
        let session = load_session(&vocoder_path, providers)?;
        Ok(Self { session })
    }

    /// Converts a mel-spectrogram to audio waveform.
    ///
    /// # Arguments
    ///
    /// * `mel` - Mel-spectrogram with shape (1, mel_bins, time_frames)
    ///
    /// # Returns
    ///
    /// Audio waveform as a 1D array of f32 samples at 44.1 kHz.
    pub fn synthesize(&mut self, mel: &Array3<f32>) -> Result<Array1<f32>> {
        // Create input tensor from flat data
        let shape = mel.shape();
        let data: Vec<f32> = mel.iter().copied().collect();
        let mel_tensor = Tensor::from_array(([shape[0], shape[1], shape[2]], data))
            .map_err(|e| DaemonError::model_inference_failed(format!("Failed to create mel tensor: {}", e)))?;

        let mut outputs = self
            .session
            .run(ort::inputs![mel_tensor])
            .map_err(|e| DaemonError::model_inference_failed(format!("Vocoder inference failed: {}", e)))?;

        // Get first output
        let output_key = outputs.keys().next().map(|s| s.to_string()).ok_or_else(|| {
            DaemonError::model_inference_failed("Missing vocoder output".to_string())
        })?;
        let audio = outputs.remove(&output_key).ok_or_else(|| {
            DaemonError::model_inference_failed("Failed to remove vocoder output".to_string())
        })?;

        let (_, audio_data) = audio
            .try_extract_tensor::<f32>()
            .map_err(|e| DaemonError::model_inference_failed(format!("Failed to extract audio: {}", e)))?;

        // Flatten to 1D - output may be (1, samples) or (1, 1, samples) or (samples,)
        let samples: Vec<f32> = audio_data.to_vec();

        Ok(Array1::from_vec(samples))
    }

    /// Returns the native output sample rate.
    pub fn sample_rate(&self) -> u32 {
        VOCODER_SAMPLE_RATE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_rates() {
        assert_eq!(VOCODER_SAMPLE_RATE, 44100);
        assert_eq!(TARGET_SAMPLE_RATE, 48000);
    }
}
