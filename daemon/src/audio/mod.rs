//! Audio output module.
//!
//! Provides WAV file writing and resampling for generated audio.

pub mod resample;
pub mod wav;

// Re-export commonly used items
pub use resample::{resample, resample_44100_to_48000};
pub use wav::{
    samples_to_duration, write_wav, write_wav_to_buffer, CHANNELS, SAMPLE_RATE,
    SAMPLE_RATE_ACE_STEP, SAMPLE_RATE_MUSICGEN,
};
