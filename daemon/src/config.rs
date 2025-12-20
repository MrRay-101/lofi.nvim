//! Daemon configuration module.
//!
//! Contains the runtime configuration for the lofi-daemon, including
//! execution device selection and path configuration.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

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
    /// Path to the directory containing ONNX model files.
    /// If None, uses the platform-specific default cache location.
    pub model_path: Option<PathBuf>,

    /// Path to the directory for storing generated audio files.
    /// If None, uses the platform-specific default cache location.
    pub cache_path: Option<PathBuf>,

    /// Execution device for inference.
    pub device: Device,

    /// Number of threads for intra-op parallelism in ONNX Runtime.
    /// If None, uses ONNX Runtime's default (typically number of CPU cores).
    pub threads: Option<u32>,
}

impl DaemonConfig {
    /// Creates a new DaemonConfig with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the effective model path, using platform defaults if not specified.
    pub fn effective_model_path(&self) -> PathBuf {
        if let Some(ref path) = self.model_path {
            path.clone()
        } else {
            default_model_path()
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
            cache_path: None,
            device: Device::Auto,
            threads: None,
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

        // Paths should be non-empty
        assert!(!model_path.as_os_str().is_empty());
        assert!(!cache_path.as_os_str().is_empty());
    }
}
