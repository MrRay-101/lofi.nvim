//! lofi-daemon: AI music generation daemon using MusicGen and ACE-Step backends.
//!
//! This binary can run in two modes:
//! - CLI mode: Standalone music generation for testing
//! - Daemon mode: JSON-RPC server for Neovim integration

use std::time::Instant;

use lofi_daemon::audio::write_wav;
use lofi_daemon::cli::{BackendArg, Cli};
use lofi_daemon::config::DaemonConfig;
use lofi_daemon::error::Result;
use lofi_daemon::generation::{generate_ace_step, generate_with_progress};
use lofi_daemon::models::ace_step::AceStepModels;
use lofi_daemon::models::{ensure_ace_step_models, ensure_models};
use lofi_daemon::rpc::{run_server, ServerState};

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse_args();

    if cli.is_daemon_mode() {
        run_daemon_mode()
    } else if cli.is_cli_mode() {
        run_cli_mode(&cli)
    } else {
        print_usage();
        Ok(())
    }
}

/// Runs the CLI mode for music generation.
fn run_cli_mode(cli: &Cli) -> Result<()> {
    let prompt = cli.prompt.as_ref().expect("Prompt required in CLI mode");
    let output_path = cli.output_path();

    match cli.backend {
        BackendArg::Musicgen => run_musicgen_cli(cli, prompt, &output_path),
        BackendArg::AceStep => run_ace_step_cli(cli, prompt, &output_path),
    }
}

/// Runs MusicGen generation in CLI mode.
fn run_musicgen_cli(cli: &Cli, prompt: &str, output_path: &std::path::Path) -> Result<()> {
    let model_dir = cli.model_directory();

    eprintln!("=== lofi-daemon MusicGen CLI ===");
    eprintln!("Backend: MusicGen (32kHz, 5-30s)");
    eprintln!("Prompt: \"{}\"", prompt);
    eprintln!("Duration: {}s", cli.duration);
    eprintln!("Output: {}", output_path.display());
    eprintln!("Model directory: {}", model_dir.display());
    if let Some(seed) = cli.seed {
        eprintln!("Seed: {}", seed);
    }
    eprintln!();

    // Validate duration for MusicGen
    if cli.duration > 30 {
        eprintln!("Warning: MusicGen supports up to 30s. Consider using --backend ace_step for longer audio.");
    }

    // Ensure models are downloaded
    eprintln!("Checking model files...");
    ensure_models(&model_dir)?;
    eprintln!();

    // Start timing
    let start_time = Instant::now();

    // Generate audio with progress callback
    let samples = generate_with_progress(
        prompt,
        cli.duration,
        cli.seed,
        &model_dir,
        |current, total| {
            let _ = (current, total);
        },
    )?;

    // Calculate generation time
    let generation_time = start_time.elapsed();
    let generation_time_sec = generation_time.as_secs_f32();

    eprintln!();
    eprintln!("Generation complete!");
    eprintln!("  Time: {:.2}s", generation_time_sec);
    eprintln!("  Samples: {}", samples.len());
    eprintln!(
        "  Audio duration: {:.2}s",
        samples.len() as f32 / 32000.0
    );
    eprintln!();

    // Write to WAV file (32kHz for MusicGen)
    eprintln!("Writing WAV file...");
    write_wav(&samples, output_path, 32000)?;
    eprintln!("Saved to: {}", output_path.display());

    Ok(())
}

/// Runs ACE-Step generation in CLI mode.
fn run_ace_step_cli(cli: &Cli, prompt: &str, output_path: &std::path::Path) -> Result<()> {
    let model_dir = cli.ace_step_model_directory();
    let seed = cli.seed.unwrap_or(42);

    eprintln!("=== lofi-daemon ACE-Step CLI ===");
    eprintln!("Backend: ACE-Step (48kHz, 5-240s)");
    eprintln!("Prompt: \"{}\"", prompt);
    eprintln!("Duration: {}s", cli.duration);
    eprintln!("Steps: {}", cli.steps);
    eprintln!("Guidance: {:.1}", cli.guidance);
    eprintln!("Seed: {}", seed);
    eprintln!("Output: {}", output_path.display());
    eprintln!("Model directory: {}", model_dir.display());
    eprintln!();

    // Ensure models are downloaded
    eprintln!("Checking ACE-Step model files...");
    ensure_ace_step_models(&model_dir)?;
    eprintln!();

    // Load models
    let config = DaemonConfig::default();
    let mut models = AceStepModels::load(&model_dir, &config)?;

    // Start timing
    let start_time = Instant::now();

    // Generate audio
    let samples = generate_ace_step(
        &mut models,
        prompt,
        cli.duration as f32,
        seed,
        cli.steps,
        "euler",
        cli.guidance,
        |step, total| {
            if step % 5 == 0 || step == total {
                eprintln!("Progress: {}/{} steps", step, total);
            }
        },
    )?;

    // Calculate generation time
    let generation_time = start_time.elapsed();
    let generation_time_sec = generation_time.as_secs_f32();

    eprintln!();
    eprintln!("Generation complete!");
    eprintln!("  Time: {:.2}s", generation_time_sec);
    eprintln!("  Samples: {}", samples.len());
    eprintln!(
        "  Audio duration: {:.2}s",
        samples.len() as f32 / 48000.0
    );
    eprintln!();

    // Write to WAV file (48kHz for ACE-Step)
    eprintln!("Writing WAV file...");
    write_wav(&samples, output_path, 48000)?;
    eprintln!("Saved to: {}", output_path.display());

    Ok(())
}

/// Runs the daemon mode (JSON-RPC server).
fn run_daemon_mode() -> Result<()> {
    eprintln!("=== lofi-daemon JSON-RPC Server ===");
    eprintln!("Reading from stdin, writing to stdout.");
    eprintln!("Send JSON-RPC requests to control the daemon.");
    eprintln!();

    let config = DaemonConfig::default();
    let state = ServerState::new(config);

    run_server(state)
}

/// Prints usage information.
fn print_usage() {
    eprintln!("lofi-daemon: AI music generation using MusicGen and ACE-Step");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  MusicGen (default, 5-30s at 32kHz):");
    eprintln!("    lofi-daemon --prompt \"lofi hip hop beats\" --duration 10 --output test.wav");
    eprintln!();
    eprintln!("  ACE-Step (5-240s at 48kHz):");
    eprintln!("    lofi-daemon --backend ace-step --prompt \"lofi beats\" --duration 60 --output long.wav");
    eprintln!();
    eprintln!("  Daemon mode (JSON-RPC server):");
    eprintln!("    lofi-daemon --daemon");
    eprintln!();
    eprintln!("Run 'lofi-daemon --help' for full options.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn print_usage_doesnt_panic() {
        print_usage();
    }
}
