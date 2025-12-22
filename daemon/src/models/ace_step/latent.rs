//! Latent space utilities for ACE-Step.
//!
//! Provides functions for initializing and manipulating latent representations
//! used in the diffusion process.

use ndarray::Array4;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use super::transformer::{LATENT_CHANNELS, LATENT_HEIGHT};

/// Sample rate used for frame length calculation (vocoder native rate).
const SAMPLE_RATE: f32 = 44100.0;

/// Hop length for the DCAE (samples per latent frame, after 8x compression).
const HOP_LENGTH: f32 = 512.0 * 8.0; // 4096

/// Initializes a latent tensor with random Gaussian noise.
///
/// For Flow Matching, the initial latent is pure standard normal noise
/// (NOT scaled by sigma - that's for Karras/EDM diffusion).
///
/// # Arguments
///
/// * `batch_size` - Number of samples to generate (typically 1)
/// * `frame_length` - Number of frames in the time dimension
/// * `_initial_sigma` - Unused (kept for API compatibility)
/// * `seed` - Random seed for reproducibility
///
/// # Returns
///
/// A latent tensor of shape (batch_size, LATENT_CHANNELS, LATENT_HEIGHT, frame_length)
/// initialized with standard Gaussian noise (mean=0, std=1).
pub fn initialize_latent(
    batch_size: usize,
    frame_length: usize,
    _initial_sigma: f32,
    seed: u64,
) -> Array4<f32> {
    let shape = (batch_size, LATENT_CHANNELS, LATENT_HEIGHT, frame_length);
    let total_elements = batch_size * LATENT_CHANNELS * LATENT_HEIGHT * frame_length;

    // Use ChaCha8 for reproducible random generation
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Generate standard normal samples using Box-Muller transform
    let mut samples = Vec::with_capacity(total_elements);
    while samples.len() < total_elements {
        // Box-Muller transform for Gaussian samples
        let u1: f32 = rng.gen_range(1e-10..1.0); // Avoid log(0)
        let u2: f32 = rng.gen_range(0.0..1.0);

        let mag = (-2.0 * u1.ln()).sqrt();
        let z0 = mag * (2.0 * std::f32::consts::PI * u2).cos();
        let z1 = mag * (2.0 * std::f32::consts::PI * u2).sin();

        // Flow Matching uses unscaled standard normal noise
        samples.push(z0);
        if samples.len() < total_elements {
            samples.push(z1);
        }
    }

    Array4::from_shape_vec(shape, samples)
        .expect("Shape calculation should be correct")
}

/// Calculates the latent frame length from audio duration.
///
/// The frame length determines the temporal resolution of the latent.
/// Formula: frame_length = duration_sec * sample_rate / hop_length
///
/// # Arguments
///
/// * `duration_sec` - Target audio duration in seconds
///
/// # Returns
///
/// The number of latent frames required for the given duration.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(calculate_frame_length(30.0), 323);
/// assert_eq!(calculate_frame_length(60.0), 646);
/// assert_eq!(calculate_frame_length(120.0), 1292);
/// ```
pub fn calculate_frame_length(duration_sec: f32) -> usize {
    // frame_length = duration_sec * sample_rate / hop_length
    // = duration_sec * 44100 / 4096
    // ≈ duration_sec * 10.77
    ((duration_sec * SAMPLE_RATE / HOP_LENGTH).ceil() as usize).max(1)
}

/// Estimates the output audio duration from frame length.
///
/// This is the inverse of `calculate_frame_length`.
///
/// # Arguments
///
/// * `frame_length` - Number of latent frames
///
/// # Returns
///
/// Estimated audio duration in seconds.
pub fn estimate_duration(frame_length: usize) -> f32 {
    frame_length as f32 * HOP_LENGTH / SAMPLE_RATE
}

/// Estimates the number of audio samples from frame length.
///
/// # Arguments
///
/// * `frame_length` - Number of latent frames
///
/// # Returns
///
/// Estimated number of audio samples at 44.1 kHz.
pub fn estimate_samples(frame_length: usize) -> usize {
    frame_length * HOP_LENGTH as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_length_30_seconds() {
        let frames = calculate_frame_length(30.0);
        // 30 * 44100 / 4096 ≈ 323
        assert!(frames >= 320 && frames <= 330, "Got {} frames for 30s", frames);
    }

    #[test]
    fn frame_length_60_seconds() {
        let frames = calculate_frame_length(60.0);
        // 60 * 44100 / 4096 ≈ 646
        assert!(frames >= 640 && frames <= 660, "Got {} frames for 60s", frames);
    }

    #[test]
    fn frame_length_120_seconds() {
        let frames = calculate_frame_length(120.0);
        // 120 * 44100 / 4096 ≈ 1292
        assert!(frames >= 1280 && frames <= 1310, "Got {} frames for 120s", frames);
    }

    #[test]
    fn frame_length_240_seconds() {
        let frames = calculate_frame_length(240.0);
        // 240 * 44100 / 4096 ≈ 2585
        assert!(frames >= 2570 && frames <= 2600, "Got {} frames for 240s", frames);
    }

    #[test]
    fn frame_length_minimum() {
        // Even for very short durations, should have at least 1 frame
        assert_eq!(calculate_frame_length(0.001), 1);
        assert_eq!(calculate_frame_length(0.0), 1);
    }

    #[test]
    fn initialize_latent_shape() {
        let latent = initialize_latent(1, 100, 80.0, 42);
        assert_eq!(latent.shape(), &[1, LATENT_CHANNELS, LATENT_HEIGHT, 100]);
    }

    #[test]
    fn initialize_latent_reproducible() {
        let latent1 = initialize_latent(1, 50, 1.0, 12345);
        let latent2 = initialize_latent(1, 50, 1.0, 12345);

        // Same seed should produce identical results
        assert_eq!(latent1, latent2);
    }

    #[test]
    fn initialize_latent_different_seeds() {
        let latent1 = initialize_latent(1, 50, 1.0, 12345);
        let latent2 = initialize_latent(1, 50, 1.0, 54321);

        // Different seeds should produce different results
        assert_ne!(latent1, latent2);
    }

    #[test]
    fn initialize_latent_standard_normal() {
        let latent = initialize_latent(1, 100, 1.0, 42);

        // Standard normal: most values should be within [-4, 4]
        for &val in latent.iter() {
            assert!(
                val.abs() < 6.0,
                "Value {} unexpectedly large for standard normal",
                val,
            );
        }

        // Check approximate mean is near 0
        let mean = latent.mean().unwrap_or(0.0);
        assert!(
            mean.abs() < 0.5,
            "Mean {} too far from 0 for standard normal",
            mean
        );
    }

    #[test]
    fn estimate_duration_inverse() {
        for duration in [5.0, 30.0, 60.0, 120.0, 240.0] {
            let frames = calculate_frame_length(duration);
            let estimated = estimate_duration(frames);
            // Should be close to original (within 1%)
            let error = (estimated - duration).abs() / duration;
            assert!(
                error < 0.02,
                "Duration {} -> {} frames -> {:.2}s (error: {:.2}%)",
                duration,
                frames,
                estimated,
                error * 100.0
            );
        }
    }
}
