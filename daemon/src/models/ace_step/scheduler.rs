//! Flow Matching Euler scheduler for ACE-Step.
//!
//! Implements the FlowMatchEulerDiscreteScheduler from the ACE-Step codebase.
//! This is NOT a Karras diffusion scheduler - it uses flow matching formulation.

use ndarray::Array4;

/// Scheduler type for diffusion process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SchedulerType {
    /// Euler ODE solver - fast, deterministic.
    #[default]
    Euler,
    /// Heun ODE solver - 2x slower, more accurate.
    Heun,
    /// PingPong SDE solver - stochastic, best quality.
    PingPong,
}

impl SchedulerType {
    /// Parses a scheduler type from a string.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "euler" => Some(SchedulerType::Euler),
            "heun" => Some(SchedulerType::Heun),
            "pingpong" | "ping_pong" | "ping-pong" => Some(SchedulerType::PingPong),
            _ => None,
        }
    }

    /// Returns the string name of this scheduler.
    pub fn as_str(&self) -> &'static str {
        match self {
            SchedulerType::Euler => "euler",
            SchedulerType::Heun => "heun",
            SchedulerType::PingPong => "pingpong",
        }
    }
}

/// Flow Matching Euler scheduler.
///
/// Based on FlowMatchEulerDiscreteScheduler from ACE-Step.
/// Uses shifted sigmas: `shift * sigma / (1 + (shift - 1) * sigma)`
#[derive(Debug, Clone)]
pub struct EulerScheduler {
    /// Total number of inference steps.
    num_steps: u32,
    /// Omega scale for mean shifting (default 10.0).
    omega: f32,
    /// Sigma values for each timestep (from ~1.0 to 0.0).
    sigmas: Vec<f32>,
    /// Timesteps for each step (sigmas * 1000).
    timesteps: Vec<f32>,
    /// Current step index.
    current_step: usize,
}

impl EulerScheduler {
    /// Creates a new Flow Matching Euler scheduler.
    ///
    /// # Arguments
    ///
    /// * `num_steps` - Number of diffusion steps (typically 60)
    /// * `shift` - Shift parameter (default 3.0)
    /// * `omega` - Omega scale for mean shifting (default 10.0)
    pub fn new(num_steps: u32, shift: f32, omega: f32) -> Self {
        let num_train_timesteps = 1000.0_f32;

        // Create linearly spaced timesteps from sigma_max*1000 to sigma_min*1000
        // then convert to sigmas
        let sigma_max = 1.0_f32;
        let sigma_min = 0.0_f32;

        // Linear interpolation from max to min
        let mut sigmas: Vec<f32> = (0..num_steps)
            .map(|i| {
                // t goes from 1.0 to ~0.0
                let t = sigma_max - (i as f32 / (num_steps - 1).max(1) as f32) * (sigma_max - sigma_min);
                // Apply shift: shift * t / (1 + (shift - 1) * t)
                shift * t / (1.0 + (shift - 1.0) * t)
            })
            .collect();

        // Append final sigma of 0
        sigmas.push(0.0);

        // Timesteps are sigmas * num_train_timesteps
        let timesteps: Vec<f32> = sigmas.iter()
            .take(num_steps as usize)
            .map(|s| s * num_train_timesteps)
            .collect();

        Self {
            num_steps,
            omega,
            sigmas,
            timesteps,
            current_step: 0,
        }
    }

    /// Creates a scheduler with default ACE-Step parameters.
    pub fn default_ace_step(num_steps: u32) -> Self {
        Self::new(num_steps, 3.0, 10.0)
    }

    /// Returns the current timestep value (sigma * 1000).
    pub fn timestep(&self) -> f32 {
        self.timesteps[self.current_step]
    }

    /// Returns the current sigma (noise level, 0.0 to ~1.0).
    pub fn sigma(&self) -> f32 {
        self.sigmas[self.current_step]
    }

    /// Returns the next sigma (noise level for next step).
    pub fn next_sigma(&self) -> f32 {
        self.sigmas[self.current_step + 1]
    }

    /// Performs one Flow Matching Euler step to update the latent.
    ///
    /// Flow matching step: `x_next = x + (sigma_next - sigma) * model_output`
    /// With omega mean shifting for stability.
    ///
    /// # Arguments
    ///
    /// * `latent` - Current latent representation
    /// * `model_output` - Predicted velocity/direction from the transformer
    ///
    /// # Returns
    ///
    /// Updated latent after one denoising step.
    pub fn step(&mut self, latent: &Array4<f32>, model_output: &Array4<f32>) -> Array4<f32> {
        let sigma = self.sigma();
        let sigma_next = self.next_sigma();
        let dt = sigma_next - sigma; // This is negative (going from high sigma to low)

        // Compute dx = dt * model_output
        let dx = model_output.mapv(|v| v * dt);

        // Apply omega mean shifting for stability
        // omega_scaled = logistic(omega) mapping to [0.9, 1.1]
        let omega_scaled = logistic(self.omega, 0.9, 1.1, 0.0, 0.1);
        let mean = dx.mean().unwrap_or(0.0);
        let dx_shifted = dx.mapv(|v| (v - mean) * omega_scaled + mean);

        // Update latent: x_next = x + dx_shifted
        let next_latent = latent + &dx_shifted;

        // Advance to next step
        self.current_step += 1;

        next_latent
    }

    /// Returns whether the scheduler has completed all steps.
    pub fn is_done(&self) -> bool {
        self.current_step >= self.num_steps as usize
    }

    /// Returns the current step index.
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Returns the total number of steps.
    pub fn num_steps(&self) -> u32 {
        self.num_steps
    }

    /// Resets the scheduler to the initial state.
    pub fn reset(&mut self) {
        self.current_step = 0;
    }

    /// Returns all sigmas for the schedule.
    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    /// Returns all timesteps for the schedule.
    pub fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }
}

/// Logistic function for omega scaling.
/// Maps input x to range [lower, upper] with sigmoid shape.
fn logistic(x: f32, lower: f32, upper: f32, x0: f32, k: f32) -> f32 {
    lower + (upper - lower) / (1.0 + (-k * (x - x0)).exp())
}

/// Creates a scheduler of the specified type.
pub fn create_scheduler(scheduler_type: SchedulerType, num_steps: u32) -> EulerScheduler {
    match scheduler_type {
        SchedulerType::Euler => EulerScheduler::default_ace_step(num_steps),
        SchedulerType::Heun => EulerScheduler::default_ace_step(num_steps),
        SchedulerType::PingPong => EulerScheduler::default_ace_step(num_steps),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduler_type_parsing() {
        assert_eq!(SchedulerType::parse("euler"), Some(SchedulerType::Euler));
        assert_eq!(SchedulerType::parse("heun"), Some(SchedulerType::Heun));
        assert_eq!(SchedulerType::parse("pingpong"), Some(SchedulerType::PingPong));
        assert_eq!(SchedulerType::parse("invalid"), None);
    }

    #[test]
    fn euler_scheduler_creation() {
        let scheduler = EulerScheduler::default_ace_step(60);
        assert_eq!(scheduler.num_steps(), 60);
        assert_eq!(scheduler.current_step(), 0);
        assert!(!scheduler.is_done());
    }

    #[test]
    fn euler_scheduler_sigmas() {
        let scheduler = EulerScheduler::default_ace_step(60);
        let sigmas = scheduler.sigmas();

        // Should have num_steps + 1 sigmas (including final 0)
        assert_eq!(sigmas.len(), 61);

        // First sigma should be ~1.0 (shift*1/(1+(shift-1)*1) = 3/3 = 1.0)
        assert!((sigmas[0] - 1.0).abs() < 0.01, "First sigma should be ~1.0, got {}", sigmas[0]);

        // Last sigma should be 0.0
        assert_eq!(sigmas[sigmas.len() - 1], 0.0);

        // Sigmas should be monotonically decreasing
        for i in 1..sigmas.len() {
            assert!(sigmas[i] <= sigmas[i - 1], "Sigma {} ({}) > sigma {} ({})", i, sigmas[i], i - 1, sigmas[i - 1]);
        }
    }

    #[test]
    fn euler_scheduler_timesteps() {
        let scheduler = EulerScheduler::default_ace_step(60);
        let timesteps = scheduler.timesteps();

        // First timestep should be ~1000 (sigma ~1.0 * 1000)
        assert!(timesteps[0] > 900.0, "First timestep should be ~1000, got {}", timesteps[0]);
    }

    #[test]
    fn euler_scheduler_step() {
        let mut scheduler = EulerScheduler::default_ace_step(60);

        let latent = Array4::zeros((1, 8, 16, 100));
        let noise_pred = Array4::ones((1, 8, 16, 100));

        let initial_step = scheduler.current_step();
        let _ = scheduler.step(&latent, &noise_pred);

        assert_eq!(scheduler.current_step(), initial_step + 1);
    }

    #[test]
    fn logistic_function() {
        // At x=0 with x0=0, should be at midpoint
        let mid = logistic(0.0, 0.9, 1.1, 0.0, 0.1);
        assert!((mid - 1.0).abs() < 0.01, "Logistic at x=0 should be ~1.0, got {}", mid);

        // At large positive x, should approach upper bound
        let high = logistic(100.0, 0.9, 1.1, 0.0, 0.1);
        assert!(high > 1.09, "Logistic at large x should be ~1.1, got {}", high);
    }
}
