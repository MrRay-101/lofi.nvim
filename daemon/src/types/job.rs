//! GenerationJob entity representing a music generation request.
//!
//! Jobs track the lifecycle of a generation request from submission
//! through completion or failure.

use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// Priority level for generation jobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Priority {
    /// Normal priority - processed in order.
    #[default]
    Normal,
    /// High priority - inserted at front of queue.
    High,
}

/// Status states for a generation job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    /// Job received, validating.
    #[default]
    Pending,
    /// Validated, waiting in queue.
    Queued,
    /// Actively generating audio.
    Generating,
    /// Generation completed successfully.
    Complete,
    /// Generation failed mid-process.
    Failed,
    /// Invalid request (bad duration, queue full).
    Rejected,
}

impl JobStatus {
    /// Returns true if the job is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(self, JobStatus::Complete | JobStatus::Failed | JobStatus::Rejected)
    }

    /// Returns true if the job is actively processing.
    pub fn is_active(&self) -> bool {
        matches!(self, JobStatus::Generating)
    }
}

/// A request for music generation, tracked from submission through completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationJob {
    /// Unique job identifier (UUID v4 format).
    pub job_id: String,

    /// Computed track_id for deduplication (derived from generation params).
    pub track_id: String,

    /// Text description of desired music (1-1000 chars).
    pub prompt: String,

    /// Requested audio duration in seconds (5-120).
    pub duration_sec: u32,

    /// Random seed for generation (auto-generated if not provided).
    pub seed: u64,

    /// Queue priority.
    pub priority: Priority,

    /// Current job status.
    pub status: JobStatus,

    /// Position in queue (0-9), None if not queued.
    pub queue_position: Option<u32>,

    /// Generation progress percentage (0-99, capped until complete).
    pub progress_percent: u32,

    /// Number of token frames generated so far.
    pub tokens_generated: u32,

    /// Estimated total tokens (duration_sec * 50).
    pub tokens_estimated: u32,

    /// Estimated seconds remaining.
    pub eta_sec: f32,

    /// Error code if failed.
    pub error_code: Option<String>,

    /// Human-readable error message if failed.
    pub error_message: Option<String>,

    /// When job was submitted.
    pub created_at: SystemTime,

    /// When generation started.
    pub started_at: Option<SystemTime>,

    /// When generation finished.
    pub completed_at: Option<SystemTime>,
}

impl GenerationJob {
    /// Creates a new pending generation job.
    ///
    /// The tokens_estimated is calculated as duration_sec * 50 based on
    /// MusicGen's token generation rate.
    pub fn new(
        job_id: String,
        track_id: String,
        prompt: String,
        duration_sec: u32,
        seed: u64,
        priority: Priority,
    ) -> Self {
        let tokens_estimated = duration_sec * 50;
        Self {
            job_id,
            track_id,
            prompt,
            duration_sec,
            seed,
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

    /// Marks the job as queued at the given position.
    pub fn set_queued(&mut self, position: u32) {
        self.status = JobStatus::Queued;
        self.queue_position = Some(position);
    }

    /// Marks the job as actively generating.
    pub fn set_generating(&mut self) {
        self.status = JobStatus::Generating;
        self.queue_position = None;
        self.started_at = Some(SystemTime::now());
    }

    /// Updates generation progress.
    ///
    /// Progress is capped at 99% until explicitly marked complete.
    pub fn update_progress(&mut self, tokens_generated: u32, eta_sec: f32) {
        self.tokens_generated = tokens_generated;
        self.eta_sec = eta_sec;
        if self.tokens_estimated > 0 {
            let percent = (tokens_generated * 100 / self.tokens_estimated).min(99);
            self.progress_percent = percent;
        }
    }

    /// Marks the job as completed successfully.
    pub fn set_complete(&mut self) {
        self.status = JobStatus::Complete;
        self.progress_percent = 100;
        self.eta_sec = 0.0;
        self.completed_at = Some(SystemTime::now());
    }

    /// Marks the job as failed with an error.
    pub fn set_failed(&mut self, code: impl Into<String>, message: impl Into<String>) {
        self.status = JobStatus::Failed;
        self.error_code = Some(code.into());
        self.error_message = Some(message.into());
        self.completed_at = Some(SystemTime::now());
    }

    /// Marks the job as rejected (validation failure).
    pub fn set_rejected(&mut self, code: impl Into<String>, message: impl Into<String>) {
        self.status = JobStatus::Rejected;
        self.error_code = Some(code.into());
        self.error_message = Some(message.into());
        self.completed_at = Some(SystemTime::now());
    }
}
