//! Device detection and execution provider selection for ONNX Runtime.
//!
//! Provides automatic detection of available hardware accelerators (CUDA, CoreML)
//! and returns appropriate execution providers based on device configuration.

use ort::execution_providers::{
    CPUExecutionProvider, CoreMLExecutionProvider, CUDAExecutionProvider, ExecutionProvider,
    ExecutionProviderDispatch,
};
use ort::session::Session;

use crate::config::Device;

/// Represents an available execution provider with its name.
#[derive(Debug, Clone)]
pub struct AvailableProvider {
    /// Human-readable name of the provider.
    pub name: &'static str,
    /// The execution provider dispatch.
    pub provider: ExecutionProviderDispatch,
}

/// Detects available execution providers on the current system.
///
/// Attempts to register each provider with a dummy session builder to check
/// if the hardware/driver is available. Returns a list of working providers
/// in priority order:
/// 1. CUDA (NVIDIA GPUs)
/// 2. CoreML (Apple Silicon)
/// 3. CPU (always available)
///
/// # Example
///
/// ```no_run
/// use lofi_daemon::models::device::detect_available_providers;
///
/// let providers = detect_available_providers();
/// for p in &providers {
///     println!("Available: {}", p.name);
/// }
/// ```
pub fn detect_available_providers() -> Vec<AvailableProvider> {
    let mut available = Vec::new();

    // Try CUDA
    if let Ok(mut builder) = Session::builder() {
        let cuda = CUDAExecutionProvider::default();
        if cuda.register(&mut builder).is_ok() {
            available.push(AvailableProvider {
                name: "CUDA",
                provider: cuda.build(),
            });
        }
    }

    // Try CoreML (macOS/iOS)
    if let Ok(mut builder) = Session::builder() {
        let coreml = CoreMLExecutionProvider::default();
        if coreml.register(&mut builder).is_ok() {
            available.push(AvailableProvider {
                name: "CoreML",
                provider: coreml.build(),
            });
        }
    }

    // CPU is always available
    available.push(AvailableProvider {
        name: "CPU",
        provider: CPUExecutionProvider::default().build(),
    });

    available
}

/// Gets the execution providers for a given device configuration.
///
/// # Arguments
///
/// * `device` - The device selection from configuration
/// * `threads` - Optional number of threads for CPU execution
///
/// # Returns
///
/// A vector of execution provider dispatches to pass to the session builder.
/// For Auto mode, returns the best available provider.
///
/// # Example
///
/// ```no_run
/// use lofi_daemon::config::Device;
/// use lofi_daemon::models::device::get_providers;
///
/// let providers = get_providers(Device::Auto, None);
/// println!("Using {} provider(s)", providers.len());
/// ```
pub fn get_providers(device: Device, threads: Option<u32>) -> Vec<ExecutionProviderDispatch> {
    match device {
        Device::Auto => {
            // Detect and return best available
            let available = detect_available_providers();
            if let Some(first) = available.into_iter().next() {
                vec![first.provider]
            } else {
                // Fallback to default CPU
                vec![build_cpu_provider(threads)]
            }
        }
        Device::Cpu => {
            vec![build_cpu_provider(threads)]
        }
        Device::Cuda => {
            vec![CUDAExecutionProvider::default().build()]
        }
        Device::Metal => {
            vec![CoreMLExecutionProvider::default().build()]
        }
    }
}

/// Builds a CPU execution provider.
///
/// Note: Thread configuration is handled at the session level via
/// `SessionBuilder::with_intra_threads()`, not at the provider level.
fn build_cpu_provider(_threads: Option<u32>) -> ExecutionProviderDispatch {
    CPUExecutionProvider::default().build()
}

/// Gets a human-readable name for the device configuration.
///
/// For Auto mode, returns the name of the detected provider.
/// For explicit modes, returns the device name.
pub fn get_device_name(device: Device) -> &'static str {
    match device {
        Device::Auto => {
            let available = detect_available_providers();
            available.first().map(|p| p.name).unwrap_or("CPU")
        }
        Device::Cpu => "CPU",
        Device::Cuda => "CUDA",
        Device::Metal => "CoreML",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_providers_includes_cpu() {
        let providers = detect_available_providers();
        assert!(!providers.is_empty(), "Should have at least CPU provider");

        let has_cpu = providers.iter().any(|p| p.name == "CPU");
        assert!(has_cpu, "CPU provider should always be available");
    }

    #[test]
    fn get_providers_auto_returns_something() {
        let providers = get_providers(Device::Auto, None);
        assert!(!providers.is_empty(), "Auto should return at least one provider");
    }

    #[test]
    fn get_providers_cpu_explicit() {
        let providers = get_providers(Device::Cpu, None);
        assert_eq!(providers.len(), 1, "CPU should return exactly one provider");
    }

    #[test]
    fn get_providers_cpu_with_threads() {
        let providers = get_providers(Device::Cpu, Some(4));
        assert_eq!(providers.len(), 1, "Should return one provider with threads config");
    }

    #[test]
    fn get_device_name_explicit() {
        assert_eq!(get_device_name(Device::Cpu), "CPU");
        assert_eq!(get_device_name(Device::Cuda), "CUDA");
        assert_eq!(get_device_name(Device::Metal), "CoreML");
    }

    #[test]
    fn get_device_name_auto() {
        // Auto should return a valid provider name
        let name = get_device_name(Device::Auto);
        assert!(
            name == "CPU" || name == "CUDA" || name == "CoreML",
            "Auto should return a known provider name, got: {}",
            name
        );
    }
}
