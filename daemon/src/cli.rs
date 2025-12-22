//! CLI argument parser for Phase 0 standalone mode.
//!
//! Provides command-line interface for testing music generation
//! without the full daemon infrastructure.

use std::path::PathBuf;

use clap::{Parser, ValueEnum};

/// Available generation backends.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum)]
pub enum BackendArg {
    /// MusicGen: 5-30 second autoregressive generation at 32kHz
    #[default]
    Musicgen,
    /// ACE-Step: 5-240 second diffusion generation at 48kHz
    AceStep,
}

/// Number of token frames generated per second of audio.
/// MusicGen generates approximately 50 tokens per second.
pub const TOKENS_PER_SECOND: usize = 50;

/// lofi-daemon: AI music generation with MusicGen and ACE-Step backends
#[derive(Parser, Debug)]
#[command(name = "lofi-daemon")]
#[command(about = "AI music generation daemon with MusicGen and ACE-Step backends")]
#[command(version)]
pub struct Cli {
    /// Text prompt describing the music to generate
    #[arg(short, long)]
    pub prompt: Option<String>,

    /// Duration of audio to generate in seconds (5-240 for ACE-Step, 5-30 for MusicGen)
    #[arg(short, long, default_value = "10", value_parser = clap::value_parser!(u32).range(5..=240))]
    pub duration: u32,

    /// Output WAV file path
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Path to directory containing ONNX model files
    #[arg(short, long)]
    pub model_dir: Option<PathBuf>,

    /// Random seed for reproducible generation
    #[arg(short, long)]
    pub seed: Option<u64>,

    /// Generation backend to use
    #[arg(short, long, value_enum, default_value_t = BackendArg::Musicgen)]
    pub backend: BackendArg,

    /// Number of diffusion steps (ACE-Step only, default 60)
    #[arg(long, default_value = "60")]
    pub steps: u32,

    /// Guidance scale for classifier-free guidance (ACE-Step only, default 7.0)
    #[arg(long, default_value = "7.0")]
    pub guidance: f32,

    /// Run in daemon mode (JSON-RPC over stdio)
    #[arg(long)]
    pub daemon: bool,
}

impl Cli {
    /// Parses command-line arguments.
    pub fn parse_args() -> Self {
        Cli::parse()
    }

    /// Returns true if running in CLI mode (not daemon mode).
    pub fn is_cli_mode(&self) -> bool {
        !self.daemon && self.prompt.is_some()
    }

    /// Returns true if running in daemon mode.
    pub fn is_daemon_mode(&self) -> bool {
        self.daemon
    }

    /// Calculates the number of tokens to generate based on duration.
    pub fn tokens_to_generate(&self) -> usize {
        self.duration as usize * TOKENS_PER_SECOND
    }

    /// Returns the effective output path.
    ///
    /// Defaults to "output.wav" in the current directory if not specified.
    pub fn output_path(&self) -> PathBuf {
        self.output.clone().unwrap_or_else(|| PathBuf::from("output.wav"))
    }

    /// Returns the effective model directory for MusicGen.
    ///
    /// Defaults to platform-specific cache location if not specified.
    pub fn model_directory(&self) -> PathBuf {
        if let Some(ref path) = self.model_dir {
            path.clone()
        } else {
            default_model_path()
        }
    }

    /// Returns the model directory for ACE-Step models.
    pub fn ace_step_model_directory(&self) -> PathBuf {
        if let Some(ref path) = self.model_dir {
            path.clone()
        } else {
            default_ace_step_model_path()
        }
    }

    /// Returns true if using ACE-Step backend.
    pub fn is_ace_step(&self) -> bool {
        self.backend == BackendArg::AceStep
    }
}

/// Returns the platform-specific default model storage path for MusicGen.
fn default_model_path() -> PathBuf {
    if let Some(proj_dirs) = directories::ProjectDirs::from("", "", "lofi.nvim") {
        proj_dirs.cache_dir().join("musicgen")
    } else {
        PathBuf::from("./models/musicgen")
    }
}

/// Returns the platform-specific default model storage path for ACE-Step.
fn default_ace_step_model_path() -> PathBuf {
    if let Some(proj_dirs) = directories::ProjectDirs::from("", "", "lofi.nvim") {
        proj_dirs.cache_dir().join("ace-step")
    } else {
        PathBuf::from("./models/ace-step")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_per_second_constant() {
        assert_eq!(TOKENS_PER_SECOND, 50);
    }

    #[test]
    fn default_model_path_is_valid() {
        let path = default_model_path();
        assert!(!path.as_os_str().is_empty());
    }

    #[test]
    fn tokens_calculation() {
        let cli = Cli {
            prompt: Some("test".to_string()),
            duration: 10,
            output: None,
            model_dir: None,
            seed: None,
            backend: BackendArg::Musicgen,
            steps: 60,
            guidance: 7.0,
            daemon: false,
        };
        assert_eq!(cli.tokens_to_generate(), 500);
    }

    #[test]
    fn cli_mode_detection() {
        let cli_mode = Cli {
            prompt: Some("test".to_string()),
            duration: 10,
            output: None,
            model_dir: None,
            seed: None,
            backend: BackendArg::Musicgen,
            steps: 60,
            guidance: 7.0,
            daemon: false,
        };
        assert!(cli_mode.is_cli_mode());
        assert!(!cli_mode.is_daemon_mode());

        let daemon_mode = Cli {
            prompt: None,
            duration: 10,
            output: None,
            model_dir: None,
            seed: None,
            backend: BackendArg::Musicgen,
            steps: 60,
            guidance: 7.0,
            daemon: true,
        };
        assert!(!daemon_mode.is_cli_mode());
        assert!(daemon_mode.is_daemon_mode());
    }

    #[test]
    fn output_path_default() {
        let cli = Cli {
            prompt: Some("test".to_string()),
            duration: 10,
            output: None,
            model_dir: None,
            seed: None,
            backend: BackendArg::Musicgen,
            steps: 60,
            guidance: 7.0,
            daemon: false,
        };
        assert_eq!(cli.output_path(), PathBuf::from("output.wav"));
    }

    #[test]
    fn ace_step_backend_detection() {
        let ace_step = Cli {
            prompt: Some("test".to_string()),
            duration: 60,
            output: None,
            model_dir: None,
            seed: Some(42),
            backend: BackendArg::AceStep,
            steps: 60,
            guidance: 7.0,
            daemon: false,
        };
        assert!(ace_step.is_ace_step());

        let musicgen = Cli {
            prompt: Some("test".to_string()),
            duration: 10,
            output: None,
            model_dir: None,
            seed: None,
            backend: BackendArg::Musicgen,
            steps: 60,
            guidance: 7.0,
            daemon: false,
        };
        assert!(!musicgen.is_ace_step());
    }

    #[test]
    fn ace_step_model_path_is_valid() {
        let path = default_ace_step_model_path();
        assert!(!path.as_os_str().is_empty());
        assert!(path.to_string_lossy().contains("ace-step"));
    }
}
