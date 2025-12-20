//! Progress tracking for music generation.
//!
//! Provides utilities for calculating generation progress, percentages,
//! and estimated time remaining.

use std::time::Instant;

/// Token generation rate (tokens per second of audio).
const TOKENS_PER_SECOND: usize = 50;

/// Tracks progress during generation.
///
/// Computes percentage and ETA based on tokens generated vs estimated.
#[derive(Debug)]
pub struct ProgressTracker {
    /// Target duration in seconds.
    duration_sec: u32,
    /// Estimated total tokens.
    tokens_estimated: usize,
    /// Current tokens generated.
    tokens_generated: usize,
    /// Time when generation started.
    start_time: Instant,
    /// Last reported percentage (for 5% increment tracking).
    last_reported_percent: u8,
}

impl ProgressTracker {
    /// Creates a new progress tracker for the given duration.
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
    /// assert_eq!(tracker.tokens_estimated(), 1500); // 30 * 50
    /// ```
    pub fn new(duration_sec: u32) -> Self {
        Self {
            duration_sec,
            tokens_estimated: duration_sec as usize * TOKENS_PER_SECOND,
            tokens_generated: 0,
            start_time: Instant::now(),
            last_reported_percent: 0,
        }
    }

    /// Updates the progress with the current number of tokens generated.
    ///
    /// # Arguments
    ///
    /// * `tokens_generated` - Current number of tokens generated
    pub fn update(&mut self, tokens_generated: usize) {
        self.tokens_generated = tokens_generated;
    }

    /// Returns the current progress percentage (0-99).
    ///
    /// Progress is capped at 99 until generation is complete.
    /// The completion notification signals 100%.
    pub fn get_percent(&self) -> u8 {
        if self.tokens_estimated == 0 {
            return 0;
        }
        let percent = (self.tokens_generated * 100) / self.tokens_estimated;
        // Cap at 99 until complete
        std::cmp::min(percent, 99) as u8
    }

    /// Returns the estimated time remaining in seconds.
    ///
    /// Based on current generation rate extrapolated to remaining tokens.
    pub fn get_eta(&self) -> f32 {
        let elapsed = self.start_time.elapsed().as_secs_f32();

        if self.tokens_generated == 0 || elapsed == 0.0 {
            // Can't estimate without data, use rough estimate
            return estimate_generation_time(self.tokens_estimated);
        }

        let tokens_per_sec = self.tokens_generated as f32 / elapsed;
        let tokens_remaining = self.tokens_estimated.saturating_sub(self.tokens_generated);

        if tokens_per_sec > 0.0 {
            tokens_remaining as f32 / tokens_per_sec
        } else {
            estimate_generation_time(tokens_remaining)
        }
    }

    /// Returns the number of tokens generated so far.
    pub fn tokens_generated(&self) -> usize {
        self.tokens_generated
    }

    /// Returns the estimated total tokens.
    pub fn tokens_estimated(&self) -> usize {
        self.tokens_estimated
    }

    /// Returns the target duration in seconds.
    pub fn duration_sec(&self) -> u32 {
        self.duration_sec
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
    /// Returns (percent, tokens_generated, tokens_estimated, eta_sec).
    pub fn get_progress(&self) -> (u8, usize, usize, f32) {
        (
            self.get_percent(),
            self.tokens_generated,
            self.tokens_estimated,
            self.get_eta(),
        )
    }
}

/// Estimates generation time based on token count.
///
/// Returns an estimate in seconds. Actual time depends on hardware.
fn estimate_generation_time(token_count: usize) -> f32 {
    // Rough estimate: ~0.05 seconds per token on CPU
    // This is conservative; GPU can be much faster
    token_count as f32 * 0.05
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_tracker_new() {
        let tracker = ProgressTracker::new(10);
        assert_eq!(tracker.tokens_estimated(), 500); // 10 * 50
        assert_eq!(tracker.tokens_generated(), 0);
        assert_eq!(tracker.get_percent(), 0);
        assert_eq!(tracker.duration_sec(), 10);
    }

    #[test]
    fn progress_tracker_update() {
        let mut tracker = ProgressTracker::new(10);
        tracker.update(250);
        assert_eq!(tracker.tokens_generated(), 250);
        assert_eq!(tracker.get_percent(), 50);
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

        let (percent, generated, estimated, eta) = tracker.get_progress();
        assert_eq!(percent, 50);
        assert_eq!(generated, 250);
        assert_eq!(estimated, 500);
        assert!(eta >= 0.0);
    }

    #[test]
    fn estimate_generation_time_calculation() {
        // 500 tokens at 0.05s each = 25s
        assert_eq!(estimate_generation_time(500), 25.0);
    }
}
