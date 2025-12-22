//! Classifier-free guidance for ACE-Step.
//!
//! Implements CFG (Classifier-Free Guidance) which improves prompt adherence
//! by combining conditional and unconditional predictions.

use ndarray::{Array4, Zip};

/// Default guidance scale for ACE-Step.
/// Higher values = stronger prompt adherence.
pub const DEFAULT_GUIDANCE_SCALE: f32 = 7.0;

/// Minimum guidance scale (essentially no guidance).
pub const MIN_GUIDANCE_SCALE: f32 = 1.0;

/// Maximum guidance scale (very strong guidance).
pub const MAX_GUIDANCE_SCALE: f32 = 20.0;

/// Applies classifier-free guidance to noise predictions.
///
/// CFG formula: output = uncond + scale * (cond - uncond)
///
/// # Arguments
///
/// * `cond_noise` - Noise prediction with text conditioning
/// * `uncond_noise` - Noise prediction without conditioning (empty prompt)
/// * `guidance_scale` - Strength of guidance (typically 7.0-15.0)
///
/// # Returns
///
/// Guided noise prediction combining both conditional and unconditional paths.
///
/// # Example
///
/// ```ignore
/// use lofi_daemon::models::ace_step::guidance::apply_cfg;
///
/// let guided = apply_cfg(&cond_noise, &uncond_noise, 7.0);
/// ```
pub fn apply_cfg(
    cond_noise: &Array4<f32>,
    uncond_noise: &Array4<f32>,
    guidance_scale: f32,
) -> Array4<f32> {
    // CFG: output = uncond + scale * (cond - uncond)
    // Which simplifies to: output = (1 - scale) * uncond + scale * cond
    // But the first form is more numerically stable

    let mut result = Array4::zeros(cond_noise.raw_dim());

    Zip::from(&mut result)
        .and(cond_noise)
        .and(uncond_noise)
        .for_each(|r, &c, &u| {
            *r = u + guidance_scale * (c - u);
        });

    result
}

/// Validates a guidance scale value.
///
/// Returns an error message if the scale is outside the valid range.
pub fn validate_guidance_scale(scale: f32) -> Option<String> {
    if scale < MIN_GUIDANCE_SCALE {
        Some(format!(
            "Guidance scale {} is below minimum {}",
            scale, MIN_GUIDANCE_SCALE
        ))
    } else if scale > MAX_GUIDANCE_SCALE {
        Some(format!(
            "Guidance scale {} exceeds maximum {}",
            scale, MAX_GUIDANCE_SCALE
        ))
    } else if scale.is_nan() || scale.is_infinite() {
        Some("Guidance scale must be a finite number".to_string())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;

    #[test]
    fn cfg_with_scale_1() {
        // With scale=1, should return conditional output
        let cond = Array4::from_elem((1, 2, 2, 2), 1.0f32);
        let uncond = Array4::from_elem((1, 2, 2, 2), 0.0f32);

        let result = apply_cfg(&cond, &uncond, 1.0);

        // uncond + 1.0 * (cond - uncond) = uncond + cond - uncond = cond
        assert!((result[[0, 0, 0, 0]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cfg_with_scale_0() {
        // With scale=0, should return unconditional output
        let cond = Array4::from_elem((1, 2, 2, 2), 1.0f32);
        let uncond = Array4::from_elem((1, 2, 2, 2), 0.5f32);

        let result = apply_cfg(&cond, &uncond, 0.0);

        // uncond + 0.0 * (cond - uncond) = uncond
        assert!((result[[0, 0, 0, 0]] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn cfg_with_default_scale() {
        let cond = Array4::from_elem((1, 2, 2, 2), 1.0f32);
        let uncond = Array4::from_elem((1, 2, 2, 2), 0.0f32);

        let result = apply_cfg(&cond, &uncond, DEFAULT_GUIDANCE_SCALE);

        // uncond + 7.0 * (cond - uncond) = 0 + 7.0 * 1 = 7.0
        assert!((result[[0, 0, 0, 0]] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn validate_valid_scales() {
        assert!(validate_guidance_scale(1.0).is_none());
        assert!(validate_guidance_scale(7.0).is_none());
        assert!(validate_guidance_scale(20.0).is_none());
    }

    #[test]
    fn validate_invalid_scales() {
        assert!(validate_guidance_scale(0.5).is_some());
        assert!(validate_guidance_scale(25.0).is_some());
        assert!(validate_guidance_scale(f32::NAN).is_some());
        assert!(validate_guidance_scale(f32::INFINITY).is_some());
    }
}
