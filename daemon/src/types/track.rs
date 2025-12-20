//! Track entity representing a generated audio file.
//!
//! Tracks are stored in the cache with a unique ID computed from
//! the generation parameters for deduplication.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::time::SystemTime;

/// A successfully generated audio file stored in the cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Track {
    /// SHA256-derived unique identifier (16 hex chars).
    /// Computed from: prompt + seed + duration + model_version
    pub track_id: String,

    /// Absolute filesystem path to the WAV file.
    pub path: PathBuf,

    /// Original text prompt used for generation (1-1000 chars).
    pub prompt: String,

    /// Actual duration of generated audio in seconds.
    pub duration_sec: f32,

    /// Audio sample rate in Hz (always 32000 for MusicGen).
    pub sample_rate: u32,

    /// Random seed used for generation.
    pub seed: u64,

    /// Model identifier for reproducibility (e.g., "musicgen-small-fp16-v1").
    pub model_version: String,

    /// Time taken to generate in seconds.
    pub generation_time_sec: f32,

    /// When the track was created (ISO 8601 timestamp).
    pub created_at: SystemTime,
}

impl Track {
    /// Creates a new Track with the given parameters.
    ///
    /// The track_id is automatically computed from the generation parameters.
    pub fn new(
        path: PathBuf,
        prompt: String,
        duration_sec: f32,
        seed: u64,
        model_version: String,
        generation_time_sec: f32,
    ) -> Self {
        let track_id = compute_track_id(&prompt, seed, duration_sec, &model_version);
        Self {
            track_id,
            path,
            prompt,
            duration_sec,
            sample_rate: 32000,
            seed,
            model_version,
            generation_time_sec,
            created_at: SystemTime::now(),
        }
    }
}

/// Computes a unique track ID from generation parameters.
///
/// The ID is the first 16 hex characters of SHA256(prompt:seed:duration:model_version).
/// This enables cache deduplication - identical parameters always produce the same ID.
///
/// # Arguments
/// * `prompt` - The text prompt for generation
/// * `seed` - The random seed
/// * `duration` - The duration in seconds
/// * `model_version` - The model version string
///
/// # Returns
/// A 16-character hex string suitable for use as a unique identifier.
pub fn compute_track_id(prompt: &str, seed: u64, duration: f32, model_version: &str) -> String {
    let input = format!("{}:{}:{}:{}", prompt, seed, duration, model_version);
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    let result = hasher.finalize();
    // Take first 8 bytes (16 hex chars)
    hex::encode(&result[..8])
}

/// Hex encoding for track IDs.
mod hex {
    const HEX_CHARS: &[u8; 16] = b"0123456789abcdef";

    pub fn encode(bytes: &[u8]) -> String {
        let mut result = String::with_capacity(bytes.len() * 2);
        for &byte in bytes {
            result.push(HEX_CHARS[(byte >> 4) as usize] as char);
            result.push(HEX_CHARS[(byte & 0x0f) as usize] as char);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_track_id_deterministic() {
        let id1 = compute_track_id("lofi beats", 12345, 10.0, "musicgen-small-fp16-v1");
        let id2 = compute_track_id("lofi beats", 12345, 10.0, "musicgen-small-fp16-v1");
        assert_eq!(id1, id2);
        assert_eq!(id1.len(), 16);
    }

    #[test]
    fn test_compute_track_id_different_params() {
        let id1 = compute_track_id("lofi beats", 12345, 10.0, "musicgen-small-fp16-v1");
        let id2 = compute_track_id("jazz piano", 12345, 10.0, "musicgen-small-fp16-v1");
        assert_ne!(id1, id2);
    }
}
