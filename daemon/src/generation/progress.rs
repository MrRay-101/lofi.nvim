//! Progress tracking for music generation.
//!
//! Provides utilities for calculating generation progress, percentages,
//! and estimated time remaining. Supports both token-based progress
//! (MusicGen) and step-based progress (ACE-Step diffusion).

use std::time::Instant;

/// Token generation rate (tokens per second of audio).
const TOKENS_PER_SECOND: usize = 50;

/// Progress tracking mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressMode {
    /// Token-based progress for autoregressive models like MusicGen.
    /// Total is estimated from duration * tokens_per_second.
    Tokens,
    /// Step-based progress for diffusion models like ACE-Step.
    /// Total is the number of inference steps.
    Steps,
}

/// Tracks progress during generation.
///
/// Computes percentage and ETA based on tokens/steps generated vs estimated.
/// Supports both token-based (MusicGen) and step-based (ACE-Step) tracking.
#[derive(Debug)]
pub struct ProgressTracker {
    /// Target duration in seconds.
    duration_sec: u32,
    /// Estimated total units (tokens or steps).
    units_estimated: usize,
    /// Current units completed (tokens or steps).
    units_completed: usize,
    /// Time when generation started.
    start_time: Instant,
    /// Last reported percentage (for 5% increment tracking).
    last_reported_percent: u8,
    /// Progress tracking mode.
    mode: ProgressMode,
}

impl ProgressTracker {
    /// Creates a new token-based progress tracker for the given duration.
    ///
    /// # Arguments
    ///
    /// * `duration_sec` - Target duration in seconds
    ///
    /// # Example
    ///
    /// ```
    /// use lofi_daemon::generation::ProgressTracker;
    ///
    /// let tracker = ProgressTracker::new(30); // 30 second track
    /// assert_eq!(tracker.get_percent(), 0);
    /// assert_eq!(tracker.units_estimated(), 1500); // 30 * 50
    /// ```
    pub fn new(duration_sec: u32) -> Self {
        Self {
            duration_sec,
            units_estimated: duration_sec as usize * TOKENS_PER_SECOND,
            units_completed: 0,
            start_time: Instant::now(),
            last_reported_percent: 0,
            mode: ProgressMode::Tokens,
        }
    }

    /// Creates a new step-based progress tracker for diffusion models.
    ///
    /// # Arguments
    ///
    /// * `duration_sec` - Target duration in seconds
    /// * `total_steps` - Total number of diffusion inference steps
    ///
    /// # Example
    ///
    /// ```
    /// use lofi_daemon::generation::ProgressTracker;
    ///
    /// let tracker = ProgressTracker::for_steps(30, 60); // 30s, 60 steps
    /// assert_eq!(tracker.get_percent(), 0);
    /// assert_eq!(tracker.units_estimated(), 60);
    /// ```
    pub fn for_steps(duration_sec: u32, total_steps: usize) -> Self {
        Self {
            duration_sec,
            units_estimated: total_steps,
            units_completed: 0,
            start_time: Instant::now(),
            last_reported_percent: 0,
            mode: ProgressMode::Steps,
        }
    }

    /// Updates the progress with the current number of units completed.
    ///
    /// # Arguments
    ///
    /// * `units_completed` - Current number of tokens/steps completed
    pub fn update(&mut self, units_completed: usize) {
        self.units_completed = units_completed;
    }

    /// Returns the current progress percentage (0-99).
    ///
    /// Progress is capped at 99 until generation is complete.
    /// The completion notification signals 100%.
    pub fn get_percent(&self) -> u8 {
        if self.units_estimated == 0 {
            return 0;
        }
        let percent = (self.units_completed * 100) / self.units_estimated;
        // Cap at 99 until complete
        std::cmp::min(percent, 99) as u8
    }

    /// Returns the estimated time remaining in seconds.
    ///
    /// Based on current generation rate extrapolated to remaining units.
    pub fn get_eta(&self) -> f32 {
        let elapsed = self.start_time.elapsed().as_secs_f32();

        if self.units_completed == 0 || elapsed == 0.0 {
            // Can't estimate without data, use rough estimate
            return estimate_generation_time(self.units_estimated, self.mode);
        }

        let units_per_sec = self.units_completed as f32 / elapsed;
        let units_remaining = self.units_estimated.saturating_sub(self.units_completed);

        if units_per_sec > 0.0 {
            units_remaining as f32 / units_per_sec
        } else {
            estimate_generation_time(units_remaining, self.mode)
        }
    }

    /// Returns the number of units (tokens/steps) completed so far.
    pub fn units_completed(&self) -> usize {
        self.units_completed
    }

    /// Returns the estimated total units (tokens/steps).
    pub fn units_estimated(&self) -> usize {
        self.units_estimated
    }

    /// Returns the number of tokens generated so far.
    /// For step-based tracking, returns the step count.
    #[deprecated(since = "0.2.0", note = "use units_completed() instead")]
    pub fn tokens_generated(&self) -> usize {
        self.units_completed
    }

    /// Returns the estimated total tokens.
    /// For step-based tracking, returns the step count.
    #[deprecated(since = "0.2.0", note = "use units_estimated() instead")]
    pub fn tokens_estimated(&self) -> usize {
        self.units_estimated
    }

    /// Returns the target duration in seconds.
    pub fn duration_sec(&self) -> u32 {
        self.duration_sec
    }

    /// Returns the progress tracking mode.
    pub fn mode(&self) -> ProgressMode {
        self.mode
    }

    /// Returns the current step for step-based progress.
    /// Returns None for token-based progress.
    pub fn current_step(&self) -> Option<usize> {
        if self.mode == ProgressMode::Steps {
            Some(self.units_completed)
        } else {
            None
        }
    }

    /// Returns the total steps for step-based progress.
    /// Returns None for token-based progress.
    pub fn total_steps(&self) -> Option<usize> {
        if self.mode == ProgressMode::Steps {
            Some(self.units_estimated)
        } else {
            None
        }
    }

    /// Checks if a progress notification should be sent (every 5% increment).
    ///
    /// Returns `Some(percent)` if a notification should be sent, `None` otherwise.
    /// Updates internal state to track the last reported percentage.
    pub fn should_notify(&mut self) -> Option<u8> {
        let current_percent = self.get_percent();
        // Report every 5% increment
        let next_threshold = (self.last_reported_percent / 5 + 1) * 5;

        if current_percent >= next_threshold {
            self.last_reported_percent = (current_percent / 5) * 5;
            Some(current_percent)
        } else {
            None
        }
    }

    /// Forces an update and returns current progress info.
    ///
    /// Returns (percent, units_completed, units_estimated, eta_sec).
    pub fn get_progress(&self) -> (u8, usize, usize, f32) {
        (
            self.get_percent(),
            self.units_completed,
            self.units_estimated,
            self.get_eta(),
        )
    }

    /// Returns extended progress info including step data.
    ///
    /// Returns (percent, units_completed, units_estimated, eta_sec, current_step, total_steps).
    /// current_step and total_steps are None for token-based progress.
    pub fn get_extended_progress(&self) -> (u8, usize, usize, f32, Option<usize>, Option<usize>) {
        (
            self.get_percent(),
            self.units_completed,
            self.units_estimated,
            self.get_eta(),
            self.current_step(),
            self.total_steps(),
        )
    }
}

/// Estimates generation time based on unit count and mode.
///
/// Returns an estimate in seconds. Actual time depends on hardware.
fn estimate_generation_time(unit_count: usize, mode: ProgressMode) -> f32 {
    match mode {
        ProgressMode::Tokens => {
            // Rough estimate: ~0.05 seconds per token on CPU
            // This is conservative; GPU can be much faster
            unit_count as f32 * 0.05
        }
        ProgressMode::Steps => {
            // Rough estimate: ~0.2 seconds per diffusion step on CPU
            // GPU with fp16 can be 5-10x faster
            unit_count as f32 * 0.2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_tracker_new() {
        let tracker = ProgressTracker::new(10);
        assert_eq!(tracker.units_estimated(), 500); // 10 * 50
        assert_eq!(tracker.units_completed(), 0);
        assert_eq!(tracker.get_percent(), 0);
        assert_eq!(tracker.duration_sec(), 10);
        assert_eq!(tracker.mode(), ProgressMode::Tokens);
    }

    #[test]
    fn progress_tracker_for_steps() {
        let tracker = ProgressTracker::for_steps(30, 60);
        assert_eq!(tracker.units_estimated(), 60);
        assert_eq!(tracker.units_completed(), 0);
        assert_eq!(tracker.get_percent(), 0);
        assert_eq!(tracker.duration_sec(), 30);
        assert_eq!(tracker.mode(), ProgressMode::Steps);
        assert_eq!(tracker.current_step(), Some(0));
        assert_eq!(tracker.total_steps(), Some(60));
    }

    #[test]
    fn progress_tracker_update() {
        let mut tracker = ProgressTracker::new(10);
        tracker.update(250);
        assert_eq!(tracker.units_completed(), 250);
        assert_eq!(tracker.get_percent(), 50);
    }

    #[test]
    fn progress_tracker_steps_update() {
        let mut tracker = ProgressTracker::for_steps(30, 60);
        tracker.update(30);
        assert_eq!(tracker.units_completed(), 30);
        assert_eq!(tracker.get_percent(), 50);
        assert_eq!(tracker.current_step(), Some(30));
    }

    #[test]
    fn progress_tracker_percent_capped_at_99() {
        let mut tracker = ProgressTracker::new(10);
        tracker.update(500); // 100%
        assert_eq!(tracker.get_percent(), 99); // Capped at 99

        tracker.update(600); // Over 100%
        assert_eq!(tracker.get_percent(), 99); // Still capped
    }

    #[test]
    fn progress_tracker_eta() {
        let tracker = ProgressTracker::new(10);
        // With no tokens generated, ETA should be positive
        let eta = tracker.get_eta();
        assert!(eta > 0.0);
    }

    #[test]
    fn progress_tracker_steps_eta() {
        let tracker = ProgressTracker::for_steps(30, 60);
        // With no steps completed, ETA should be positive
        let eta = tracker.get_eta();
        assert!(eta > 0.0);
    }

    #[test]
    fn progress_tracker_should_notify_5_percent() {
        let mut tracker = ProgressTracker::new(100); // 5000 tokens

        // 0% - no notification
        assert!(tracker.should_notify().is_none());

        // Update to just under 5%
        tracker.update(240); // 4.8%
        assert!(tracker.should_notify().is_none());

        // Update to 5%
        tracker.update(250); // 5%
        assert_eq!(tracker.should_notify(), Some(5));

        // Same 5% shouldn't notify again
        assert!(tracker.should_notify().is_none());

        // Update to 10%
        tracker.update(500); // 10%
        assert_eq!(tracker.should_notify(), Some(10));
    }

    #[test]
    fn progress_tracker_get_progress() {
        let mut tracker = ProgressTracker::new(10);
        tracker.update(250);

        let (percent, completed, estimated, eta) = tracker.get_progress();
        assert_eq!(percent, 50);
        assert_eq!(completed, 250);
        assert_eq!(estimated, 500);
        assert!(eta >= 0.0);
    }

    #[test]
    fn progress_tracker_get_extended_progress() {
        let mut tracker = ProgressTracker::for_steps(30, 60);
        tracker.update(30);

        let (percent, completed, estimated, eta, current_step, total_steps) =
            tracker.get_extended_progress();
        assert_eq!(percent, 50);
        assert_eq!(completed, 30);
        assert_eq!(estimated, 60);
        assert!(eta >= 0.0);
        assert_eq!(current_step, Some(30));
        assert_eq!(total_steps, Some(60));
    }

    #[test]
    fn progress_tracker_token_no_steps() {
        let tracker = ProgressTracker::new(10);
        assert_eq!(tracker.current_step(), None);
        assert_eq!(tracker.total_steps(), None);
    }

    #[test]
    fn estimate_generation_time_tokens() {
        // 500 tokens at 0.05s each = 25s
        assert_eq!(estimate_generation_time(500, ProgressMode::Tokens), 25.0);
    }

    #[test]
    fn estimate_generation_time_steps() {
        // 60 steps at 0.2s each = 12s
        assert_eq!(estimate_generation_time(60, ProgressMode::Steps), 12.0);
    }
}
