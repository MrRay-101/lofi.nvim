//! Model downloader for MusicGen ONNX models.
//!
//! Downloads model files from HuggingFace if not present locally.

use std::fs;
use std::io::{Read, Write};
use std::path::Path;

use crate::error::{DaemonError, Result};

use super::musicgen::{MODEL_URLS, REQUIRED_MODEL_FILES};

/// Downloads all required model files if not present.
///
/// Returns Ok(()) if all files exist or were successfully downloaded.
pub fn ensure_models(model_dir: &Path) -> Result<()> {
    // Create model directory if it doesn't exist
    if !model_dir.exists() {
        fs::create_dir_all(model_dir).map_err(|e| {
            DaemonError::model_download_failed(format!(
                "Failed to create model directory {}: {}",
                model_dir.display(),
                e
            ))
        })?;
    }

    // Check which files are missing
    let mut missing: Vec<&str> = Vec::new();
    for file in REQUIRED_MODEL_FILES {
        let path = model_dir.join(file);
        if !path.exists() {
            missing.push(file);
        }
    }

    if missing.is_empty() {
        eprintln!("All model files present.");
        return Ok(());
    }

    eprintln!("Downloading {} missing model files...", missing.len());
    eprintln!("(This may take several minutes on first run)");
    eprintln!();

    // Download missing files
    for file in &missing {
        // Find the URL for this file
        let url = MODEL_URLS
            .iter()
            .find(|(name, _)| name == file)
            .map(|(_, url)| *url);

        if let Some(url) = url {
            download_file_streaming(url, &model_dir.join(file))?;
        } else {
            return Err(DaemonError::model_download_failed(format!(
                "No download URL for {}",
                file
            )));
        }
    }

    // Also download config.json if missing (optional but useful)
    let config_path = model_dir.join("config.json");
    if !config_path.exists() {
        if let Some((_, url)) = MODEL_URLS.iter().find(|(name, _)| *name == "config.json") {
            let _ = download_file_streaming(url, &config_path); // Ignore error, config is optional
        }
    }

    eprintln!();
    eprintln!("All models downloaded successfully.");
    Ok(())
}

/// Downloads a file using streaming to handle large files.
fn download_file_streaming(url: &str, dest: &Path) -> Result<()> {
    let filename = dest.file_name().unwrap_or_default().to_string_lossy();
    eprint!("  Downloading {}... ", filename);

    // Create a client with longer timeout for large files
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(3600)) // 1 hour timeout
        .build()
        .map_err(|e| {
            DaemonError::model_download_failed(format!("Failed to create HTTP client: {}", e))
        })?;

    let mut response = client.get(url).send().map_err(|e| {
        DaemonError::model_download_failed(format!("Failed to download {}: {}", url, e))
    })?;

    if !response.status().is_success() {
        return Err(DaemonError::model_download_failed(format!(
            "HTTP {} for {}",
            response.status(),
            url
        )));
    }

    // Get content length for progress
    let total_size = response.content_length().unwrap_or(0);

    // Create output file
    let mut file = fs::File::create(dest).map_err(|e| {
        DaemonError::model_download_failed(format!(
            "Failed to create file {}: {}",
            dest.display(),
            e
        ))
    })?;

    // Stream the download in chunks
    let mut downloaded: u64 = 0;
    let mut buffer = [0u8; 65536]; // 64KB buffer
    let mut last_progress = 0;

    loop {
        let bytes_read = response.read(&mut buffer).map_err(|e| {
            DaemonError::model_download_failed(format!("Failed to read response: {}", e))
        })?;

        if bytes_read == 0 {
            break;
        }

        file.write_all(&buffer[..bytes_read]).map_err(|e| {
            DaemonError::model_download_failed(format!("Failed to write file: {}", e))
        })?;

        downloaded += bytes_read as u64;

        // Print progress every 10%
        if total_size > 0 {
            let progress = (downloaded * 100 / total_size) as usize;
            if progress >= last_progress + 10 {
                eprint!("{}%... ", progress);
                last_progress = progress;
            }
        }
    }

    let size_mb = downloaded as f64 / (1024.0 * 1024.0);
    eprintln!("done ({:.1} MB)", size_mb);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn get_model_dir() -> Option<PathBuf> {
        let proj_dirs = directories::ProjectDirs::from("", "", "lofi-daemon")?;
        let path = proj_dirs.data_dir().join("models");
        if path.exists() {
            Some(path)
        } else {
            None
        }
    }

    #[test]
    fn ensure_models_succeeds_when_present() {
        let Some(model_dir) = get_model_dir() else {
            eprintln!("Skipping test: models not found");
            return;
        };

        // Should succeed without downloading since models already exist
        let result = ensure_models(&model_dir);
        assert!(result.is_ok(), "ensure_models failed: {:?}", result.err());
    }

    #[test]
    fn model_urls_are_configured() {
        // Verify all required model files have URLs
        for file in REQUIRED_MODEL_FILES {
            let has_url = MODEL_URLS.iter().any(|(name, _)| name == file);
            assert!(has_url, "Missing URL for required file: {}", file);
        }
    }
}

