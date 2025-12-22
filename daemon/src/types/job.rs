//! GenerationJob type for tracking music generation requests.
//!
//! A GenerationJob tracks a request for music generation from submission
//! through completion, including progress updates and error information.

use serde::{Deserialize, Serialize};
use std::time::SystemTime;

use crate::models::Backend;

use super::track::compute_track_id;

/// Priority level for generation jobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum JobPriority {
    /// Normal priority - processed in FIFO order.
    #[default]
    Normal,
    /// High priority - processed before normal priority jobs.
    High,
}

/// Status of a generation job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    /// Job received, validating parameters.
    #[default]
    Pending,
    /// Validated, waiting in queue for generation.
    Queued,
    /// Actively generating audio.
    Generating,
    /// Generation completed successfully.
    Complete,
    /// Generation failed mid-process.
    Failed,
    /// Invalid request rejected (bad duration, queue full, etc.).
    Rejected,
}

impl JobStatus {
    /// Returns true if the job is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(self, JobStatus::Complete | JobStatus::Failed | JobStatus::Rejected)
    }

    /// Returns true if the job is actively being processed.
    pub fn is_active(&self) -> bool {
        matches!(self, JobStatus::Generating)
    }
}

/// A request for music generation, tracked from submission through completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationJob {
    /// Unique job identifier (UUID v4 format).
    pub job_id: String,

    /// Computed track_id for deduplication.
    /// This is derived from prompt, seed, duration, and model version.
    pub track_id: String,

    /// Text description of desired music (1-1000 characters).
    pub prompt: String,

    /// Requested audio duration in seconds (5-120, default 30).
    pub duration_sec: u32,

    /// Random seed for generation. If None, system generates random seed.
    pub seed: Option<u64>,

    /// Queue priority for this job.
    pub priority: JobPriority,

    /// Current job state.
    pub status: JobStatus,

    /// Position in queue (0-9), None if not queued.
    pub queue_position: Option<u8>,

    /// Generation progress as percentage (0-99, 100 only on complete).
    pub progress_percent: u8,

    /// Number of token frames generated so far.
    pub tokens_generated: u32,

    /// Estimated total tokens (duration_sec * 50).
    pub tokens_estimated: u32,

    /// Estimated seconds remaining for generation.
    pub eta_sec: f32,

    /// Error code if job failed or was rejected.
    pub error_code: Option<String>,

    /// Human-readable error message.
    pub error_message: Option<String>,

    /// When the job was submitted.
    #[serde(with = "system_time_serde")]
    pub created_at: SystemTime,

    /// When generation started (None if not started).
    #[serde(with = "option_system_time_serde")]
    pub started_at: Option<SystemTime>,

    /// When generation finished (None if not complete).
    #[serde(with = "option_system_time_serde")]
    pub completed_at: Option<SystemTime>,
}

impl GenerationJob {
    /// Creates a new pending GenerationJob.
    ///
    /// The job_id is generated as a UUID v4, and track_id is computed
    /// from the generation parameters.
    pub fn new(
        prompt: String,
        duration_sec: u32,
        seed: Option<u64>,
        priority: JobPriority,
        model_version: &str,
    ) -> Self {
        // Default to MusicGen backend (Phase 4 will add backend selection)
        Self::with_backend(prompt, duration_sec, seed, priority, model_version, Backend::MusicGen)
    }

    /// Creates a new pending GenerationJob with a specific backend.
    pub fn with_backend(
        prompt: String,
        duration_sec: u32,
        seed: Option<u64>,
        priority: JobPriority,
        model_version: &str,
        backend: Backend,
    ) -> Self {
        let job_id = generate_uuid_v4();
        let actual_seed = seed.unwrap_or_else(generate_random_seed);
        let track_id = compute_track_id(backend, &prompt, actual_seed, duration_sec as f32, model_version);
        let tokens_estimated = duration_sec * 50;

        Self {
            job_id,
            track_id,
            prompt,
            duration_sec,
            seed: Some(actual_seed),
            priority,
            status: JobStatus::Pending,
            queue_position: None,
            progress_percent: 0,
            tokens_generated: 0,
            tokens_estimated,
            eta_sec: 0.0,
            error_code: None,
            error_message: None,
            created_at: SystemTime::now(),
            started_at: None,
            completed_at: None,
        }
    }

    /// Validates job parameters.
    ///
    /// Returns an error message if validation fails, None otherwise.
    pub fn validate(&self) -> Option<String> {
        // Prompt must be 1-1000 characters
        if self.prompt.is_empty() {
            return Some("Prompt cannot be empty".to_string());
        }
        if self.prompt.len() > 1000 {
            return Some(format!(
                "Prompt too long: {} characters (max 1000)",
                self.prompt.len()
            ));
        }

        // Duration must be 5-120 seconds
        if !(5..=120).contains(&self.duration_sec) {
            return Some(format!(
                "Duration must be between 5 and 120 seconds, got {}",
                self.duration_sec
            ));
        }

        None
    }

    /// Updates progress based on tokens generated.
    pub fn update_progress(&mut self, tokens_generated: u32, generation_rate_per_sec: f32) {
        self.tokens_generated = tokens_generated;

        // Calculate progress percentage (cap at 99 until complete)
        let progress = if self.tokens_estimated > 0 {
            ((tokens_generated as f32 / self.tokens_estimated as f32) * 100.0) as u8
        } else {
            0
        };
        self.progress_percent = progress.min(99);

        // Calculate ETA
        let remaining_tokens = self.tokens_estimated.saturating_sub(tokens_generated);
        self.eta_sec = if generation_rate_per_sec > 0.0 {
            remaining_tokens as f32 / generation_rate_per_sec
        } else {
            0.0
        };
    }

    /// Marks the job as queued with the given position.
    pub fn set_queued(&mut self, position: u8) {
        self.status = JobStatus::Queued;
        self.queue_position = Some(position);
    }

    /// Marks the job as generating.
    pub fn set_generating(&mut self) {
        self.status = JobStatus::Generating;
        self.queue_position = None;
        self.started_at = Some(SystemTime::now());
    }

    /// Marks the job as complete.
    pub fn set_complete(&mut self) {
        self.status = JobStatus::Complete;
        self.progress_percent = 100;
        self.eta_sec = 0.0;
        self.completed_at = Some(SystemTime::now());
    }

    /// Marks the job as failed with an error.
    pub fn set_failed(&mut self, error_code: &str, error_message: &str) {
        self.status = JobStatus::Failed;
        self.error_code = Some(error_code.to_string());
        self.error_message = Some(error_message.to_string());
        self.completed_at = Some(SystemTime::now());
    }

    /// Marks the job as rejected with an error.
    pub fn set_rejected(&mut self, error_code: &str, error_message: &str) {
        self.status = JobStatus::Rejected;
        self.error_code = Some(error_code.to_string());
        self.error_message = Some(error_message.to_string());
        self.completed_at = Some(SystemTime::now());
    }
}

/// Generates a simple UUID v4 (random) without external dependencies.
fn generate_uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    // Use system time and a counter for randomness
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    let nanos = now.as_nanos();
    let secs = now.as_secs();

    // Create pseudo-random bytes from time components
    let bytes: [u8; 16] = [
        nanos as u8,
        (nanos >> 8) as u8,
        (nanos >> 16) as u8,
        (nanos >> 24) as u8,
        secs as u8,
        (secs >> 8) as u8,
        0x40 | ((nanos >> 32) as u8 & 0x0f), // Version 4
        (nanos >> 40) as u8,
        0x80 | ((secs >> 16) as u8 & 0x3f), // Variant 1
        (secs >> 24) as u8,
        (secs >> 32) as u8,
        (secs >> 40) as u8,
        (nanos >> 48) as u8,
        (nanos >> 56) as u8,
        (secs >> 48) as u8,
        (secs >> 56) as u8,
    ];

    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5],
        bytes[6], bytes[7],
        bytes[8], bytes[9],
        bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
    )
}

/// Generates a random seed for generation.
fn generate_random_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    // Mix time components for pseudo-randomness
    let nanos = now.as_nanos() as u64;
    let secs = now.as_secs();

    nanos.wrapping_mul(6364136223846793005).wrapping_add(secs)
}

/// Custom serde implementation for SystemTime.
mod system_time_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    pub fn serialize<S>(time: &SystemTime, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let duration = time.duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO);
        duration.as_secs().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<SystemTime, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(UNIX_EPOCH + Duration::from_secs(secs))
    }
}

/// Custom serde implementation for Option<SystemTime>.
mod option_system_time_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    pub fn serialize<S>(time: &Option<SystemTime>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match time {
            Some(t) => {
                let duration = t.duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO);
                Some(duration.as_secs()).serialize(serializer)
            }
            None => None::<u64>.serialize(serializer),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<SystemTime>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let opt = Option::<u64>::deserialize(deserializer)?;
        Ok(opt.map(|secs| UNIX_EPOCH + Duration::from_secs(secs)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn job_status_terminal() {
        assert!(JobStatus::Complete.is_terminal());
        assert!(JobStatus::Failed.is_terminal());
        assert!(JobStatus::Rejected.is_terminal());
        assert!(!JobStatus::Pending.is_terminal());
        assert!(!JobStatus::Queued.is_terminal());
        assert!(!JobStatus::Generating.is_terminal());
    }

    #[test]
    fn job_validation() {
        let job = GenerationJob::new(
            "lofi beats".to_string(),
            30,
            Some(42),
            JobPriority::Normal,
            "v1",
        );
        assert!(job.validate().is_none());

        let empty_prompt = GenerationJob::new(
            "".to_string(),
            30,
            Some(42),
            JobPriority::Normal,
            "v1",
        );
        assert!(empty_prompt.validate().is_some());
    }

    #[test]
    fn progress_update() {
        let mut job = GenerationJob::new(
            "test".to_string(),
            30, // 1500 tokens estimated
            Some(42),
            JobPriority::Normal,
            "v1",
        );

        job.update_progress(750, 50.0);
        assert_eq!(job.progress_percent, 50);
        assert!(job.eta_sec > 0.0);
    }
}
