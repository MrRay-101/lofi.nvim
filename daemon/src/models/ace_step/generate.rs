//! ACE-Step generation pipeline.
//!
//! Implements the complete diffusion-based audio generation loop using
//! all ACE-Step model components.

use crate::error::Result;

use super::guidance::{apply_cfg, DEFAULT_GUIDANCE_SCALE};
use super::latent::{calculate_frame_length, initialize_latent};
use super::models::AceStepModels;
use super::scheduler::{create_scheduler, SchedulerType};

/// Generation parameters for ACE-Step.
#[derive(Debug, Clone)]
pub struct GenerationParams {
    /// Text description of the music to generate.
    pub prompt: String,
    /// Target duration in seconds (5-240).
    pub duration_sec: f32,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Number of diffusion steps (1-200, default 60).
    pub inference_steps: u32,
    /// Scheduler type (Euler, Heun, PingPong).
    pub scheduler: SchedulerType,
    /// Classifier-free guidance scale (1.0-20.0, default 7.0).
    pub guidance_scale: f32,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            duration_sec: 30.0,
            seed: 42,
            inference_steps: 60,
            scheduler: SchedulerType::Euler,
            guidance_scale: DEFAULT_GUIDANCE_SCALE,
        }
    }
}

/// Generates audio using the ACE-Step diffusion pipeline.
pub fn generate(models: &mut AceStepModels, params: GenerationParams) -> Result<Vec<f32>> {
    generate_with_progress(models, params, |_, _| {})
}

/// Generates audio with progress callback.
///
/// # Arguments
///
/// * `models` - Loaded ACE-Step models
/// * `params` - Generation parameters
/// * `on_progress` - Callback receiving (current_step, total_steps)
///
/// # Returns
///
/// Audio samples at 44.1 kHz sample rate.
pub fn generate_with_progress<F>(
    models: &mut AceStepModels,
    params: GenerationParams,
    on_progress: F,
) -> Result<Vec<f32>>
where
    F: Fn(usize, usize),
{
    eprintln!(
        "Generating {:.1}s audio with {} steps, guidance={:.1}",
        params.duration_sec, params.inference_steps, params.guidance_scale
    );

    // Step 1: Encode the text prompt
    eprintln!("Encoding prompt: \"{}\"", params.prompt);
    let (text_hidden_states, text_attention_mask) = models.text_encoder.encode(&params.prompt)?;

    // Step 2: Encode empty prompt for classifier-free guidance
    let (uncond_text_hidden_states, uncond_text_attention_mask) = models.text_encoder.encode("")?;

    // Step 3: Get transformer context for conditional and unconditional
    eprintln!("Encoding transformer context...");
    let (cond_context, cond_mask) = models.transformer.encode_context(
        &text_hidden_states,
        &text_attention_mask,
    )?;
    let (uncond_context, uncond_mask) = models.transformer.encode_context(
        &uncond_text_hidden_states,
        &uncond_text_attention_mask,
    )?;

    eprintln!(
        "Context shape: {:?} (dim=2560)",
        cond_context.shape()
    );

    // Step 4: Calculate latent dimensions
    let frame_length = calculate_frame_length(params.duration_sec);
    eprintln!(
        "Latent shape: (1, 8, 16, {}) for {:.1}s",
        frame_length, params.duration_sec
    );

    // Step 5: Create scheduler
    let mut scheduler = create_scheduler(params.scheduler, params.inference_steps);

    // Step 6: Initialize latent with random noise
    let initial_sigma = scheduler.sigma();
    let mut latent = initialize_latent(1, frame_length, initial_sigma, params.seed);

    eprintln!("Running {} diffusion steps...", params.inference_steps);

    // Step 7: Diffusion loop
    let total_steps = params.inference_steps as usize;
    for step in 0..total_steps {
        on_progress(step, total_steps);

        let timestep = scheduler.timestep();

        // Get conditional noise prediction
        let cond_noise = models.transformer.predict_noise(
            &latent,
            timestep,
            &cond_context,
            &cond_mask,
        )?;

        // Get unconditional noise prediction
        let uncond_noise = models.transformer.predict_noise(
            &latent,
            timestep,
            &uncond_context,
            &uncond_mask,
        )?;

        // Apply classifier-free guidance
        let guided_noise = apply_cfg(&cond_noise, &uncond_noise, params.guidance_scale);

        // Update latent with scheduler step
        latent = scheduler.step(&latent, &guided_noise);

        if step % 10 == 0 || step == total_steps - 1 {
            eprintln!("Step {}/{}", step + 1, total_steps);
        }
    }

    // Final progress callback
    on_progress(total_steps, total_steps);

    eprintln!("Decoding latent to mel-spectrogram...");

    // Step 8: Decode latent to mel-spectrogram
    let mel = models.decoder.decode(&latent)?;

    eprintln!(
        "Mel shape: {:?}, synthesizing audio...",
        mel.shape()
    );

    // Step 9: Synthesize audio from mel-spectrogram
    let audio = models.vocoder.synthesize(&mel)?;

    eprintln!(
        "Generated {} samples ({:.2}s at 44.1kHz)",
        audio.len(),
        audio.len() as f32 / 44100.0
    );

    Ok(audio.to_vec())
}

/// Estimates the generation time based on parameters.
pub fn estimate_generation_time(_duration_sec: f32, inference_steps: u32) -> f32 {
    let step_time = 0.2;
    let overhead = 2.0;
    (inference_steps as f32 * step_time) + overhead
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_params() {
        let params = GenerationParams::default();
        assert_eq!(params.inference_steps, 60);
        assert_eq!(params.guidance_scale, DEFAULT_GUIDANCE_SCALE);
        assert_eq!(params.scheduler, SchedulerType::Euler);
    }

    #[test]
    fn estimate_generation_reasonable() {
        let estimate = estimate_generation_time(30.0, 60);
        assert!(estimate > 10.0 && estimate < 20.0);
    }
}
