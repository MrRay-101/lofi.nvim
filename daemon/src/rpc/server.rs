//! JSON-RPC server over stdin/stdout.
//!
//! Implements the JSON-RPC 2.0 protocol for daemon communication.

use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::cache::TrackCache;
use crate::config::DaemonConfig;
use crate::error::Result;
use crate::generation::GenerationQueue;
use crate::models::MusicGenModels;

use super::methods::handle_request;
use super::types::{JsonRpcError, JsonRpcErrorResponse, JsonRpcNotification, JsonRpcRequest};

/// State shared across all request handlers.
pub struct ServerState {
    /// Loaded models for generation.
    pub models: Option<MusicGenModels>,
    /// Track cache.
    pub cache: TrackCache,
    /// Daemon configuration.
    pub config: DaemonConfig,
    /// Generation queue for pending jobs.
    pub queue: GenerationQueue,
    /// Flag to signal server shutdown.
    shutdown: Arc<AtomicBool>,
}

impl ServerState {
    /// Creates new server state.
    pub fn new(config: DaemonConfig) -> Self {
        Self {
            models: None,
            cache: TrackCache::new(),
            config,
            queue: GenerationQueue::new(),
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Sets the loaded models.
    pub fn set_models(&mut self, models: MusicGenModels) {
        self.models = Some(models);
    }

    /// Signals the server to shut down.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }

    /// Returns true if shutdown has been requested.
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::SeqCst)
    }
}

/// Runs the JSON-RPC server, reading from stdin and writing to stdout.
pub fn run_server(mut state: ServerState) -> Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let reader = stdin.lock();

    eprintln!("JSON-RPC server started, waiting for requests...");

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                eprintln!("Stdin closed (EOF), shutting down gracefully...");
                break;
            }
            Err(e) => {
                eprintln!("Error reading stdin: {}", e);
                break;
            }
        };

        // Skip empty lines
        if line.trim().is_empty() {
            continue;
        }

        // Parse JSON-RPC request
        let response = process_request(&line, &mut state);

        // Write response
        if let Some(response) = response {
            writeln!(stdout, "{}", response).ok();
            stdout.flush().ok();
        }

        // Check for shutdown
        if state.is_shutdown() {
            eprintln!("Server shutdown requested");
            break;
        }
    }

    eprintln!("JSON-RPC server stopped");
    Ok(())
}

/// Processes a single JSON-RPC request line.
fn process_request(line: &str, state: &mut ServerState) -> Option<String> {
    // Parse JSON
    let request: JsonRpcRequest = match serde_json::from_str(line) {
        Ok(r) => r,
        Err(e) => {
            let error = JsonRpcErrorResponse::new(
                None,
                JsonRpcError::parse_error(format!("Invalid JSON: {}", e)),
            );
            return Some(serde_json::to_string(&error).unwrap_or_default());
        }
    };

    // Validate JSON-RPC version
    if request.jsonrpc != "2.0" {
        let error = JsonRpcErrorResponse::new(
            Some(request.id),
            JsonRpcError::invalid_request("Invalid JSON-RPC version (expected 2.0)"),
        );
        return Some(serde_json::to_string(&error).unwrap_or_default());
    }

    // Handle the request
    let result = handle_request(&request.method, request.params.clone(), state);

    match result {
        Ok(response) => Some(
            serde_json::to_string(&serde_json::json!({
                "jsonrpc": "2.0",
                "id": request.id,
                "result": response
            }))
            .unwrap_or_default(),
        ),
        Err(error) => Some(
            serde_json::to_string(&JsonRpcErrorResponse::new(Some(request.id), error))
                .unwrap_or_default(),
        ),
    }
}

/// Sends a JSON-RPC notification to stdout.
pub fn send_notification<T: serde::Serialize>(method: &'static str, params: T) {
    let notification = JsonRpcNotification::new(method, params);
    if let Ok(json) = serde_json::to_string(&notification) {
        let mut stdout = io::stdout();
        writeln!(stdout, "{}", json).ok();
        stdout.flush().ok();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> DaemonConfig {
        DaemonConfig::default()
    }

    #[test]
    fn server_state_new() {
        let state = ServerState::new(test_config());
        assert!(state.models.is_none());
        assert!(!state.is_shutdown());
    }

    #[test]
    fn server_state_shutdown() {
        let state = ServerState::new(test_config());
        state.shutdown();
        assert!(state.is_shutdown());
    }

    #[test]
    fn process_invalid_json() {
        let mut state = ServerState::new(test_config());
        let response = process_request("not json", &mut state);
        assert!(response.is_some());
        let response = response.unwrap();
        assert!(response.contains("-32700")); // Parse error
    }

    #[test]
    fn process_invalid_version() {
        let mut state = ServerState::new(test_config());
        let request = r#"{"jsonrpc":"1.0","method":"test","id":1}"#;
        let response = process_request(request, &mut state);
        assert!(response.is_some());
        let response = response.unwrap();
        assert!(response.contains("-32600")); // Invalid request
    }

    #[test]
    fn process_unknown_method() {
        let mut state = ServerState::new(test_config());
        let request = r#"{"jsonrpc":"2.0","method":"unknown","id":1}"#;
        let response = process_request(request, &mut state);
        assert!(response.is_some());
        let response = response.unwrap();
        assert!(response.contains("-32601")); // Method not found
    }
}
