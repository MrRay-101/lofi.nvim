//! WAV file writer for audio output.
//!
//! Writes audio samples to WAV format using the hound crate.

use std::path::Path;

use hound::{SampleFormat, WavSpec, WavWriter};

use crate::error::{DaemonError, Result};

/// Audio sample rate for MusicGen output (32kHz).
pub const SAMPLE_RATE_MUSICGEN: u32 = 32000;

/// Audio sample rate for ACE-Step output (48kHz).
pub const SAMPLE_RATE_ACE_STEP: u32 = 48000;

/// Default sample rate (MusicGen).
pub const SAMPLE_RATE: u32 = SAMPLE_RATE_MUSICGEN;

/// Number of audio channels (stereo).
pub const CHANNELS: u16 = 2;

/// Writes audio samples to a WAV file.
///
/// # Arguments
///
/// * `samples` - Audio samples as f32 values
/// * `path` - Output file path
/// * `sample_rate` - Sample rate in Hz (typically 32000 for MusicGen)
///
/// # Example
///
/// ```ignore
/// use lofi_daemon::audio::write_wav;
///
/// let samples = vec![0.0, 0.5, -0.5, 0.0];
/// write_wav(&samples, "/tmp/test.wav", 32000)?;
/// ```
pub fn write_wav(samples: &[f32], path: &Path, sample_rate: u32) -> Result<()> {
    let spec = WavSpec {
        channels: CHANNELS,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut writer = WavWriter::create(path, spec).map_err(|e| {
        DaemonError::model_inference_failed(format!("Failed to create WAV file: {}", e))
    })?;

    for sample in samples {
        // Write same sample to both left and right channels
        writer.write_sample(*sample).map_err(|e| {
            DaemonError::model_inference_failed(format!("Failed to write sample: {}", e))
        })?;
        writer.write_sample(*sample).map_err(|e| {
            DaemonError::model_inference_failed(format!("Failed to write sample: {}", e))
        })?;
    }

    writer.finalize().map_err(|e| {
        DaemonError::model_inference_failed(format!("Failed to finalize WAV file: {}", e))
    })?;

    Ok(())
}

/// Writes audio samples to an in-memory WAV buffer.
///
/// Returns the WAV file contents as a byte vector.
pub fn write_wav_to_buffer(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let spec = WavSpec {
        channels: CHANNELS,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut buffer = Vec::new();
    let cursor = std::io::Cursor::new(&mut buffer);
    let buf_writer = std::io::BufWriter::new(cursor);

    {
        let mut writer = WavWriter::new(buf_writer, spec).map_err(|e| {
            DaemonError::model_inference_failed(format!("Failed to create WAV writer: {}", e))
        })?;

        for sample in samples {
            // Write same sample to both left and right channels
            writer.write_sample(*sample).map_err(|e| {
                DaemonError::model_inference_failed(format!("Failed to write sample: {}", e))
            })?;
            writer.write_sample(*sample).map_err(|e| {
                DaemonError::model_inference_failed(format!("Failed to write sample: {}", e))
            })?;
        }
    }

    Ok(buffer)
}

/// Calculates the duration of audio in seconds from sample count.
pub fn samples_to_duration(sample_count: usize, sample_rate: u32) -> f32 {
    sample_count as f32 / sample_rate as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn write_wav_creates_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wav");

        let samples = vec![0.0f32, 0.5, -0.5, 0.0];
        write_wav(&samples, &path, SAMPLE_RATE).unwrap();

        assert!(path.exists());

        // Verify file is valid WAV
        let reader = hound::WavReader::open(&path).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.channels, CHANNELS);
        assert_eq!(spec.sample_rate, SAMPLE_RATE);
        assert_eq!(spec.sample_format, SampleFormat::Float);
    }

    #[test]
    fn write_wav_to_buffer_returns_valid_wav() {
        let samples = vec![0.0f32, 0.5, -0.5, 0.0];
        let buffer = write_wav_to_buffer(&samples, SAMPLE_RATE).unwrap();

        assert!(!buffer.is_empty());
        // WAV files start with "RIFF"
        assert_eq!(&buffer[0..4], b"RIFF");
    }

    #[test]
    fn samples_to_duration_calculation() {
        assert_eq!(samples_to_duration(32000, 32000), 1.0);
        assert_eq!(samples_to_duration(64000, 32000), 2.0);
        assert_eq!(samples_to_duration(16000, 32000), 0.5);
    }
}
