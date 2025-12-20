//! Generation pipeline for MusicGen.
//!
//! Orchestrates the text encoder, decoder, and audio codec to generate
//! audio from text prompts.

use std::path::Path;

use crate::cli::TOKENS_PER_SECOND;
use crate::error::Result;
use crate::models::{load_sessions, MusicGenModels};

/// Generates audio from a text prompt.
///
/// # Arguments
///
/// * `prompt` - Text description of the music to generate
/// * `duration_sec` - Duration of audio to generate in seconds
/// * `seed` - Random seed for reproducible generation (not yet implemented)
/// * `model_dir` - Path to directory containing ONNX model files
///
/// # Returns
///
/// A vector of f32 audio samples at 32kHz sample rate.
///
/// # Example
///
/// ```ignore
/// use lofi_daemon::generation::generate;
///
/// let samples = generate(
///     "lofi hip hop beats to relax to",
///     10,
///     Some(42),
///     Path::new("/path/to/models"),
/// )?;
/// ```
pub fn generate(
    prompt: &str,
    duration_sec: u32,
    _seed: Option<u64>,
    model_dir: &Path,
) -> Result<Vec<f32>> {
    generate_with_progress(prompt, duration_sec, _seed, model_dir, |_, _| {})
}

/// Generates audio with progress callback.
///
/// # Arguments
///
/// * `prompt` - Text description of the music to generate
/// * `duration_sec` - Duration of audio to generate in seconds
/// * `seed` - Random seed for reproducible generation
/// * `model_dir` - Path to directory containing ONNX model files
/// * `on_progress` - Callback function receiving (tokens_generated, tokens_total)
///
/// # Returns
///
/// A vector of f32 audio samples at 32kHz sample rate.
pub fn generate_with_progress<F>(
    prompt: &str,
    duration_sec: u32,
    _seed: Option<u64>,
    model_dir: &Path,
    on_progress: F,
) -> Result<Vec<f32>>
where
    F: Fn(usize, usize),
{
    // Load models
    let mut models = load_sessions(model_dir)?;

    // Calculate target tokens
    let max_tokens = duration_sec as usize * TOKENS_PER_SECOND;

    // Generate audio using the models
    generate_with_models(&mut models, prompt, max_tokens, on_progress)
}

/// Generates audio using pre-loaded models.
///
/// This is useful for batch generation where models should be loaded once.
/// The callback receives (tokens_generated, tokens_total) on every token.
pub fn generate_with_models<F>(
    models: &mut MusicGenModels,
    prompt: &str,
    max_tokens: usize,
    on_progress: F,
) -> Result<Vec<f32>>
where
    F: Fn(usize, usize),
{
    eprintln!("Encoding prompt: \"{}\"", prompt);

    // Step 1: Encode the text prompt
    let (encoder_hidden_states, encoder_attention_mask) = models.text_encoder.encode(prompt)?;

    eprintln!("Generating {} tokens...", max_tokens);

    // Step 2: Generate tokens autoregressively with progress
    // The on_progress callback is called for every token, allowing the caller
    // to filter by 5% increments using ProgressTracker
    let tokens = models.decoder.generate_tokens_with_progress(
        encoder_hidden_states,
        encoder_attention_mask,
        max_tokens,
        &on_progress,
    )?;

    let token_count = tokens.len();

    eprintln!("Generated {} tokens, decoding audio...", token_count);

    // Step 3: Decode tokens to audio
    let audio_samples = models.audio_codec.decode(tokens)?;

    eprintln!(
        "Generated {} audio samples ({:.2}s at 32kHz)",
        audio_samples.len(),
        audio_samples.len() as f32 / 32000.0
    );

    Ok(audio_samples.into())
}

/// Estimates the number of audio samples for a given token count.
///
/// MusicGen generates approximately 640 samples per token at 32kHz.
pub fn estimate_samples(token_count: usize) -> usize {
    // Each token represents approximately 640 samples at 32kHz
    // (32000 samples/sec) / (50 tokens/sec) = 640 samples/token
    token_count * 640
}

/// Estimates generation time based on token count.
///
/// Returns an estimate in seconds. Actual time depends on hardware.
pub fn estimate_generation_time(token_count: usize) -> f32 {
    // Rough estimate: ~0.1 seconds per token on CPU
    // This is conservative; GPU can be much faster
    token_count as f32 * 0.1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_samples_calculation() {
        // 10 seconds = 500 tokens = 320,000 samples
        assert_eq!(estimate_samples(500), 320_000);
    }

    #[test]
    fn estimate_generation_time_calculation() {
        // 500 tokens at 0.1s each = 50s
        assert_eq!(estimate_generation_time(500), 50.0);
    }

    #[test]
    fn tokens_per_second_matches_cli() {
        assert_eq!(TOKENS_PER_SECOND, 50);
    }
}
