//! Audio resampling utilities.
//!
//! Provides high-quality resampling between different sample rates,
//! primarily for converting ACE-Step's 44.1kHz output to 48kHz.

use rubato::{FftFixedIn, Resampler};

use crate::error::{DaemonError, Result};

/// Resamples audio from one sample rate to another.
///
/// Uses FFT-based resampling for high quality. This is primarily used
/// to convert ACE-Step's 44.1kHz vocoder output to 48kHz for consistency
/// with the lofi.nvim output format.
///
/// # Arguments
///
/// * `samples` - Input audio samples
/// * `from_rate` - Source sample rate in Hz
/// * `to_rate` - Target sample rate in Hz
///
/// # Returns
///
/// Resampled audio at the target sample rate.
///
/// # Example
///
/// ```ignore
/// use lofi_daemon::audio::resample::resample;
///
/// // Convert from 44.1kHz to 48kHz
/// let resampled = resample(&samples, 44100, 48000)?;
/// ```
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }

    // Calculate parameters
    let chunk_size = 1024;
    let sub_chunks = 2;

    // Create resampler
    let mut resampler = FftFixedIn::<f32>::new(
        from_rate as usize,
        to_rate as usize,
        chunk_size,
        sub_chunks,
        1, // mono
    )
    .map_err(|e| DaemonError::model_inference_failed(format!("Failed to create resampler: {}", e)))?;

    // Calculate expected output size
    let output_size = (samples.len() as f64 * to_rate as f64 / from_rate as f64).ceil() as usize;
    let mut output = Vec::with_capacity(output_size);

    // Process in chunks
    let input_frames = resampler.input_frames_next();
    let mut position = 0;

    while position < samples.len() {
        let end = (position + input_frames).min(samples.len());
        let mut chunk = samples[position..end].to_vec();

        // Pad the last chunk if needed
        if chunk.len() < input_frames {
            chunk.resize(input_frames, 0.0);
        }

        let input = vec![chunk];
        let resampled = resampler
            .process(&input, None)
            .map_err(|e| DaemonError::model_inference_failed(format!("Resampling failed: {}", e)))?;

        output.extend_from_slice(&resampled[0]);
        position += input_frames;
    }

    // Trim to expected length (remove padding artifacts)
    let expected_len = (samples.len() as f64 * to_rate as f64 / from_rate as f64).round() as usize;
    output.truncate(expected_len);

    Ok(output)
}

/// Resamples audio from 44.1kHz to 48kHz.
///
/// This is a convenience function for the common case of converting
/// ACE-Step vocoder output to lofi.nvim's standard 48kHz format.
///
/// # Arguments
///
/// * `samples` - Input audio at 44.1kHz
///
/// # Returns
///
/// Audio resampled to 48kHz.
pub fn resample_44100_to_48000(samples: &[f32]) -> Result<Vec<f32>> {
    resample(samples, 44100, 48000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_rate_returns_copy() {
        let samples = vec![0.0, 0.5, 1.0, 0.5, 0.0];
        let result = resample(&samples, 44100, 44100).unwrap();
        assert_eq!(result, samples);
    }

    #[test]
    fn upsample_increases_length() {
        // 1 second at 44.1kHz = 44100 samples
        // Should become ~48000 samples at 48kHz
        let samples: Vec<f32> = (0..44100)
            .map(|i| (i as f32 / 44100.0 * 2.0 * std::f32::consts::PI).sin())
            .collect();

        let result = resample(&samples, 44100, 48000).unwrap();

        // Should be close to 48000 samples
        let expected = 48000;
        let tolerance = 100;
        assert!(
            (result.len() as i64 - expected as i64).abs() < tolerance,
            "Expected ~{} samples, got {}",
            expected,
            result.len()
        );
    }

    #[test]
    fn downsample_decreases_length() {
        // 1 second at 48kHz = 48000 samples
        // Should become ~44100 samples at 44.1kHz
        let samples: Vec<f32> = (0..48000)
            .map(|i| (i as f32 / 48000.0 * 2.0 * std::f32::consts::PI).sin())
            .collect();

        let result = resample(&samples, 48000, 44100).unwrap();

        // Should be close to 44100 samples
        let expected = 44100;
        let tolerance = 100;
        assert!(
            (result.len() as i64 - expected as i64).abs() < tolerance,
            "Expected ~{} samples, got {}",
            expected,
            result.len()
        );
    }

    #[test]
    fn resample_44100_to_48000_convenience() {
        let samples: Vec<f32> = (0..4410) // 0.1 seconds
            .map(|i| (i as f32 / 4410.0 * std::f32::consts::PI).sin())
            .collect();

        let result = resample_44100_to_48000(&samples).unwrap();

        // Should be approximately 4800 samples (0.1s at 48kHz)
        assert!(
            result.len() > 4700 && result.len() < 4900,
            "Expected ~4800 samples, got {}",
            result.len()
        );
    }

    #[test]
    fn empty_input() {
        let samples: Vec<f32> = vec![];
        let result = resample(&samples, 44100, 48000).unwrap();
        assert!(result.is_empty());
    }
}
