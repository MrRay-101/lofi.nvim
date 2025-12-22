//! Daemon configuration module.
//!
//! Contains the runtime configuration for the lofi-daemon, including
//! execution device selection, backend selection, and path configuration.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::models::Backend;

/// Execution device for ONNX inference.
///
/// Determines which hardware backend to use for model inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Device {
    /// Automatically detect and use the best available device.
    /// Priority: Metal (macOS) > CUDA (Linux/Windows) > CPU
    #[default]
    Auto,

    /// Force CPU execution.
    /// Slowest but universally available.
    Cpu,

    /// Use CUDA for NVIDIA GPU acceleration.
    /// Requires CUDA toolkit and compatible GPU.
    Cuda,

    /// Use Metal/CoreML for Apple Silicon acceleration.
    /// Only available on macOS with Apple Silicon.
    Metal,
}

impl Device {
    /// Returns the string representation of the device.
    pub fn as_str(&self) -> &'static str {
        match self {
            Device::Auto => "auto",
            Device::Cpu => "cpu",
            Device::Cuda => "cuda",
            Device::Metal => "metal",
        }
    }

    /// Parses a device from a string.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "auto" => Some(Device::Auto),
            "cpu" => Some(Device::Cpu),
            "cuda" => Some(Device::Cuda),
            "metal" | "coreml" => Some(Device::Metal),
            _ => None,
        }
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Runtime configuration for the daemon.
///
/// This configuration is typically loaded from command-line arguments
/// or environment variables at startup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonConfig {
    /// Path to the directory containing MusicGen ONNX model files.
    /// If None, uses the platform-specific default cache location.
    pub model_path: Option<PathBuf>,

    /// Path to the directory containing ACE-Step ONNX model files.
    /// If None, uses the platform-specific default cache location.
    pub ace_step_model_path: Option<PathBuf>,

    /// Path to the directory for storing generated audio files.
    /// If None, uses the platform-specific default cache location.
    pub cache_path: Option<PathBuf>,

    /// Execution device for inference.
    pub device: Device,

    /// Default music generation backend.
    pub default_backend: Backend,

    /// Number of threads for intra-op parallelism in ONNX Runtime.
    /// If None, uses ONNX Runtime's default (typically number of CPU cores).
    pub threads: Option<u32>,

    /// ACE-Step specific configuration.
    pub ace_step: AceStepConfig,
}

/// ACE-Step specific configuration options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AceStepConfig {
    /// Number of diffusion inference steps.
    /// Higher values = better quality but slower generation.
    /// Default: 60 (Euler scheduler)
    pub inference_steps: u32,

    /// Scheduler type for diffusion process.
    /// Options: "euler", "heun", "pingpong"
    pub scheduler: String,

    /// Classifier-free guidance scale.
    /// Higher values = more adherence to prompt.
    /// Default: 7.0
    pub guidance_scale: f32,
}

impl Default for AceStepConfig {
    fn default() -> Self {
        Self {
            inference_steps: 60,
            scheduler: "euler".to_string(),
            guidance_scale: 7.0,
        }
    }
}

impl DaemonConfig {
    /// Creates a new DaemonConfig with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a DaemonConfig from environment variables.
    ///
    /// Reads the following environment variables:
    /// - `LOFI_MODEL_PATH` - Path to MusicGen model directory
    /// - `LOFI_ACE_STEP_MODEL_PATH` - Path to ACE-Step model directory
    /// - `LOFI_CACHE_PATH` - Path to cache directory
    /// - `LOFI_DEVICE` - Device selection (auto, cpu, cuda, metal)
    /// - `LOFI_BACKEND` - Default backend (musicgen, ace_step)
    /// - `LOFI_THREADS` - Number of threads for CPU execution
    /// - `LOFI_ACE_STEP_STEPS` - ACE-Step inference steps
    /// - `LOFI_ACE_STEP_SCHEDULER` - ACE-Step scheduler (euler, heun, pingpong)
    /// - `LOFI_ACE_STEP_GUIDANCE` - ACE-Step guidance scale
    ///
    /// Falls back to defaults for unset variables.
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(path) = std::env::var("LOFI_MODEL_PATH") {
            config.model_path = Some(PathBuf::from(path));
        }

        if let Ok(path) = std::env::var("LOFI_ACE_STEP_MODEL_PATH") {
            config.ace_step_model_path = Some(PathBuf::from(path));
        }

        if let Ok(path) = std::env::var("LOFI_CACHE_PATH") {
            config.cache_path = Some(PathBuf::from(path));
        }

        if let Ok(device_str) = std::env::var("LOFI_DEVICE") {
            if let Some(device) = Device::parse(&device_str) {
                config.device = device;
            }
        }

        if let Ok(backend_str) = std::env::var("LOFI_BACKEND") {
            if let Some(backend) = Backend::parse(&backend_str) {
                config.default_backend = backend;
            }
        }

        if let Ok(threads_str) = std::env::var("LOFI_THREADS") {
            if let Ok(threads) = threads_str.parse::<u32>() {
                if threads > 0 {
                    config.threads = Some(threads);
                }
            }
        }

        // ACE-Step specific env vars
        if let Ok(steps_str) = std::env::var("LOFI_ACE_STEP_STEPS") {
            if let Ok(steps) = steps_str.parse::<u32>() {
                if steps > 0 && steps <= 200 {
                    config.ace_step.inference_steps = steps;
                }
            }
        }

        if let Ok(scheduler) = std::env::var("LOFI_ACE_STEP_SCHEDULER") {
            let scheduler = scheduler.to_lowercase();
            if ["euler", "heun", "pingpong"].contains(&scheduler.as_str()) {
                config.ace_step.scheduler = scheduler;
            }
        }

        if let Ok(guidance_str) = std::env::var("LOFI_ACE_STEP_GUIDANCE") {
            if let Ok(guidance) = guidance_str.parse::<f32>() {
                if (1.0..=20.0).contains(&guidance) {
                    config.ace_step.guidance_scale = guidance;
                }
            }
        }

        config
    }

    /// Returns the effective MusicGen model path, using platform defaults if not specified.
    pub fn effective_model_path(&self) -> PathBuf {
        if let Some(ref path) = self.model_path {
            path.clone()
        } else {
            default_model_path()
        }
    }

    /// Returns the effective ACE-Step model path, using platform defaults if not specified.
    pub fn effective_ace_step_model_path(&self) -> PathBuf {
        if let Some(ref path) = self.ace_step_model_path {
            path.clone()
        } else {
            default_ace_step_model_path()
        }
    }

    /// Returns the effective cache path, using platform defaults if not specified.
    pub fn effective_cache_path(&self) -> PathBuf {
        if let Some(ref path) = self.cache_path {
            path.clone()
        } else {
            default_cache_path()
        }
    }

    /// Validates the configuration.
    ///
    /// Returns an error message if validation fails, None otherwise.
    pub fn validate(&self) -> Option<String> {
        // Validate thread count if specified
        if let Some(threads) = self.threads {
            if threads == 0 {
                return Some("threads must be > 0".to_string());
            }
            if threads > 256 {
                return Some(format!("threads too high: {} (max 256)", threads));
            }
        }

        None
    }
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            model_path: None,
            ace_step_model_path: None,
            cache_path: None,
            device: Device::Auto,
            default_backend: Backend::default(),
            threads: None,
            ace_step: AceStepConfig::default(),
        }
    }
}

/// Returns the platform-specific default model storage path.
///
/// Uses the `directories` crate to find appropriate locations:
/// - macOS: ~/Library/Application Support/lofi-daemon/models
/// - Linux: ~/.local/share/lofi-daemon/models
/// - Windows: C:\Users\<user>\AppData\Local\lofi-daemon\models
fn default_model_path() -> PathBuf {
    if let Some(proj_dirs) = directories::ProjectDirs::from("", "", "lofi-daemon") {
        proj_dirs.data_dir().join("models")
    } else {
        // Fallback to current directory
        PathBuf::from("./models")
    }
}

/// Returns the platform-specific default cache storage path.
///
/// Uses the `directories` crate to find appropriate locations:
/// - macOS: ~/Library/Caches/lofi-daemon/tracks
/// - Linux: ~/.cache/lofi-daemon/tracks
/// - Windows: C:\Users\<user>\AppData\Local\lofi-daemon\cache\tracks
fn default_cache_path() -> PathBuf {
    if let Some(proj_dirs) = directories::ProjectDirs::from("", "", "lofi-daemon") {
        proj_dirs.cache_dir().join("tracks")
    } else {
        // Fallback to current directory
        PathBuf::from("./cache")
    }
}

/// Returns the platform-specific default ACE-Step model storage path.
///
/// Uses the `directories` crate to find appropriate locations:
/// - macOS: ~/Library/Application Support/lofi-daemon/ace-step
/// - Linux: ~/.local/share/lofi-daemon/ace-step
/// - Windows: C:\Users\<user>\AppData\Local\lofi-daemon\ace-step
fn default_ace_step_model_path() -> PathBuf {
    if let Some(proj_dirs) = directories::ProjectDirs::from("", "", "lofi-daemon") {
        proj_dirs.data_dir().join("ace-step")
    } else {
        // Fallback to current directory
        PathBuf::from("./ace-step")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_parsing() {
        assert_eq!(Device::parse("auto"), Some(Device::Auto));
        assert_eq!(Device::parse("CPU"), Some(Device::Cpu));
        assert_eq!(Device::parse("cuda"), Some(Device::Cuda));
        assert_eq!(Device::parse("metal"), Some(Device::Metal));
        assert_eq!(Device::parse("coreml"), Some(Device::Metal));
        assert_eq!(Device::parse("invalid"), None);
    }

    #[test]
    fn device_display() {
        assert_eq!(Device::Auto.to_string(), "auto");
        assert_eq!(Device::Cpu.to_string(), "cpu");
    }

    #[test]
    fn config_validation() {
        let mut config = DaemonConfig::new();
        assert!(config.validate().is_none());

        config.threads = Some(0);
        assert!(config.validate().is_some());

        config.threads = Some(4);
        assert!(config.validate().is_none());
    }

    #[test]
    fn effective_paths() {
        let config = DaemonConfig::new();
        let model_path = config.effective_model_path();
        let cache_path = config.effective_cache_path();
        let ace_step_path = config.effective_ace_step_model_path();

        // Paths should be non-empty
        assert!(!model_path.as_os_str().is_empty());
        assert!(!cache_path.as_os_str().is_empty());
        assert!(!ace_step_path.as_os_str().is_empty());
    }

    #[test]
    fn from_env_defaults() {
        // When no env vars are set, should use defaults
        // Note: This test doesn't set any env vars so we get defaults
        let config = DaemonConfig::from_env();
        assert_eq!(config.device, Device::Auto);
        assert_eq!(config.default_backend, Backend::MusicGen);
        assert!(config.threads.is_none());
    }

    #[test]
    fn ace_step_config_defaults() {
        let config = AceStepConfig::default();
        assert_eq!(config.inference_steps, 60);
        assert_eq!(config.scheduler, "euler");
        assert_eq!(config.guidance_scale, 7.0);
    }

    #[test]
    fn daemon_config_has_ace_step_config() {
        let config = DaemonConfig::new();
        assert_eq!(config.ace_step.inference_steps, 60);
        assert_eq!(config.ace_step.scheduler, "euler");
        assert_eq!(config.ace_step.guidance_scale, 7.0);
    }
}
