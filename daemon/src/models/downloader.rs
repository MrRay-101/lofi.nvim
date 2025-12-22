//! Model downloader for ONNX models.
//!
//! Downloads model files from HuggingFace if not present locally.
//! Supports both MusicGen and ACE-Step backends with progress tracking
//! and partial download resume.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

use crate::error::{DaemonError, Result};
use crate::models::Backend;

use super::ace_step::{MODEL_URLS as ACE_STEP_URLS, REQUIRED_FILES as ACE_STEP_FILES};
use super::musicgen::{MODEL_URLS, REQUIRED_MODEL_FILES};

/// Progress callback for download operations.
///
/// Parameters:
/// - `file_name`: Name of the file being downloaded
/// - `bytes_downloaded`: Bytes downloaded for current file
/// - `bytes_total`: Total bytes for current file
/// - `files_completed`: Number of files fully downloaded
/// - `files_total`: Total number of files to download
pub type DownloadProgressCallback = Box<dyn Fn(&str, u64, u64, usize, usize) + Send>;

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

/// Downloads all required ACE-Step model files if not present.
///
/// Returns Ok(()) if all files exist or were successfully downloaded.
/// Note: ACE-Step models are larger (~11.5GB total).
pub fn ensure_ace_step_models(model_dir: &Path) -> Result<()> {
    download_ace_step_models_with_progress(model_dir, None)
}

/// Downloads all required ACE-Step model files with progress tracking.
///
/// # Arguments
///
/// * `model_dir` - Directory to download models to
/// * `on_progress` - Optional callback for progress updates
///
/// Returns Ok(()) if all files exist or were successfully downloaded.
/// Note: ACE-Step models are larger (~11.5GB total).
pub fn download_ace_step_models_with_progress(
    model_dir: &Path,
    on_progress: Option<DownloadProgressCallback>,
) -> Result<()> {
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

    // Check which files are missing or incomplete
    let mut to_download: Vec<(&str, bool)> = Vec::new(); // (file, is_resume)
    for file in ACE_STEP_FILES {
        let path = model_dir.join(file);
        let partial_path = model_dir.join(format!("{}.partial", file));

        if path.exists() {
            // File exists, skip
            continue;
        } else if partial_path.exists() {
            // Partial file exists, resume
            to_download.push((file, true));
        } else {
            // File doesn't exist, full download
            to_download.push((file, false));
        }
    }

    if to_download.is_empty() {
        eprintln!("All ACE-Step model files present.");
        return Ok(());
    }

    let files_total = ACE_STEP_FILES.len();
    let mut files_completed = files_total - to_download.len();

    eprintln!("Downloading {} missing ACE-Step model files...", to_download.len());
    eprintln!("(This may take a while - total ~11.5GB)");
    eprintln!();

    // Download missing files
    for (file, is_resume) in &to_download {
        // Find the URL for this file
        let url = ACE_STEP_URLS
            .iter()
            .find(|(name, _)| name == file)
            .map(|(_, url)| *url);

        if let Some(url) = url {
            let dest = model_dir.join(file);
            if *is_resume {
                download_file_with_resume(url, &dest, files_completed, files_total, &on_progress)?;
            } else {
                download_file_with_progress(url, &dest, files_completed, files_total, &on_progress)?;
            }
            files_completed += 1;
        } else {
            return Err(DaemonError::model_download_failed(format!(
                "No download URL for ACE-Step file {}",
                file
            )));
        }
    }

    eprintln!();
    eprintln!("All ACE-Step models downloaded successfully.");
    Ok(())
}

/// Downloads backend models with progress tracking.
///
/// # Arguments
///
/// * `backend` - Which backend to download models for
/// * `model_dir` - Directory to download models to
/// * `on_progress` - Callback for progress updates
pub fn download_backend_with_progress(
    backend: Backend,
    model_dir: &Path,
    on_progress: Option<DownloadProgressCallback>,
) -> Result<()> {
    match backend {
        Backend::MusicGen => download_musicgen_models_with_progress(model_dir, on_progress),
        Backend::AceStep => download_ace_step_models_with_progress(model_dir, on_progress),
    }
}

/// Downloads all required MusicGen model files with progress tracking.
fn download_musicgen_models_with_progress(
    model_dir: &Path,
    on_progress: Option<DownloadProgressCallback>,
) -> Result<()> {
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

    // Check which files are missing or incomplete
    let mut to_download: Vec<(&str, bool)> = Vec::new();
    for file in REQUIRED_MODEL_FILES {
        let path = model_dir.join(file);
        let partial_path = model_dir.join(format!("{}.partial", file));

        if path.exists() {
            continue;
        } else if partial_path.exists() {
            to_download.push((file, true));
        } else {
            to_download.push((file, false));
        }
    }

    if to_download.is_empty() {
        eprintln!("All MusicGen model files present.");
        return Ok(());
    }

    let files_total = REQUIRED_MODEL_FILES.len();
    let mut files_completed = files_total - to_download.len();

    eprintln!("Downloading {} missing MusicGen model files...", to_download.len());
    eprintln!();

    for (file, is_resume) in &to_download {
        let url = MODEL_URLS
            .iter()
            .find(|(name, _)| name == file)
            .map(|(_, url)| *url);

        if let Some(url) = url {
            let dest = model_dir.join(file);
            if *is_resume {
                download_file_with_resume(url, &dest, files_completed, files_total, &on_progress)?;
            } else {
                download_file_with_progress(url, &dest, files_completed, files_total, &on_progress)?;
            }
            files_completed += 1;
        } else {
            return Err(DaemonError::model_download_failed(format!(
                "No download URL for {}",
                file
            )));
        }
    }

    // Download config.json if missing
    let config_path = model_dir.join("config.json");
    if !config_path.exists() {
        if let Some((_, url)) = MODEL_URLS.iter().find(|(name, _)| *name == "config.json") {
            let _ = download_file_with_progress(url, &config_path, files_completed, files_total, &on_progress);
        }
    }

    eprintln!();
    eprintln!("All MusicGen models downloaded successfully.");
    Ok(())
}

/// Downloads a file using streaming to handle large files.
fn download_file_streaming(url: &str, dest: &Path) -> Result<()> {
    download_file_with_progress(url, dest, 0, 1, &None)
}

/// Downloads a file with progress callback support.
///
/// # Arguments
///
/// * `url` - URL to download from
/// * `dest` - Destination path (without .partial suffix)
/// * `files_completed` - Number of files already completed
/// * `files_total` - Total number of files to download
/// * `on_progress` - Optional progress callback
fn download_file_with_progress(
    url: &str,
    dest: &Path,
    files_completed: usize,
    files_total: usize,
    on_progress: &Option<DownloadProgressCallback>,
) -> Result<()> {
    let filename = dest.file_name().unwrap_or_default().to_string_lossy();
    let partial_path = dest.with_extension(
        dest.extension()
            .map(|e| format!("{}.partial", e.to_string_lossy()))
            .unwrap_or_else(|| "partial".to_string()),
    );

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

    // Create partial file for download
    let mut file = File::create(&partial_path).map_err(|e| {
        DaemonError::model_download_failed(format!(
            "Failed to create file {}: {}",
            partial_path.display(),
            e
        ))
    })?;

    // Stream the download in chunks
    let mut downloaded: u64 = 0;
    let mut buffer = [0u8; 65536]; // 64KB buffer
    let mut last_progress = 0;
    let mut last_callback_percent = 0;

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

            // Call progress callback every 1%
            if let Some(ref callback) = on_progress {
                let callback_percent = (downloaded * 100 / total_size) as usize;
                if callback_percent > last_callback_percent {
                    callback(&filename, downloaded, total_size, files_completed, files_total);
                    last_callback_percent = callback_percent;
                }
            }
        }
    }

    // Sync to disk before rename
    file.sync_all().map_err(|e| {
        DaemonError::model_download_failed(format!("Failed to sync file: {}", e))
    })?;
    drop(file);

    // Rename partial file to final destination
    fs::rename(&partial_path, dest).map_err(|e| {
        DaemonError::model_download_failed(format!(
            "Failed to rename {} to {}: {}",
            partial_path.display(),
            dest.display(),
            e
        ))
    })?;

    let size_mb = downloaded as f64 / (1024.0 * 1024.0);
    eprintln!("done ({:.1} MB)", size_mb);

    // Final progress callback
    if let Some(ref callback) = on_progress {
        callback(&filename, downloaded, downloaded, files_completed + 1, files_total);
    }

    Ok(())
}

/// Downloads a file with resume support for partial downloads.
///
/// # Arguments
///
/// * `url` - URL to download from
/// * `dest` - Final destination path (without .partial suffix)
/// * `files_completed` - Number of files already completed
/// * `files_total` - Total number of files to download
/// * `on_progress` - Optional progress callback
fn download_file_with_resume(
    url: &str,
    dest: &Path,
    files_completed: usize,
    files_total: usize,
    on_progress: &Option<DownloadProgressCallback>,
) -> Result<()> {
    let filename = dest.file_name().unwrap_or_default().to_string_lossy();
    let partial_path = dest.with_extension(
        dest.extension()
            .map(|e| format!("{}.partial", e.to_string_lossy()))
            .unwrap_or_else(|| "partial".to_string()),
    );

    // Check existing partial file size
    let existing_size = if partial_path.exists() {
        fs::metadata(&partial_path)
            .map(|m| m.len())
            .unwrap_or(0)
    } else {
        0
    };

    if existing_size == 0 {
        // No partial file, do full download
        return download_file_with_progress(url, dest, files_completed, files_total, on_progress);
    }

    eprint!("  Resuming {} from {} bytes... ", filename, existing_size);

    // Create a client with longer timeout for large files
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(3600))
        .build()
        .map_err(|e| {
            DaemonError::model_download_failed(format!("Failed to create HTTP client: {}", e))
        })?;

    // Try to resume with Range header
    let mut response = client
        .get(url)
        .header("Range", format!("bytes={}-", existing_size))
        .send()
        .map_err(|e| {
            DaemonError::model_download_failed(format!("Failed to download {}: {}", url, e))
        })?;

    let status = response.status();
    if status == reqwest::StatusCode::PARTIAL_CONTENT {
        // Server supports resume, continue from where we left off
        let content_length = response.content_length().unwrap_or(0);
        let total_size = existing_size + content_length;

        // Open file for appending
        let mut file = OpenOptions::new()
            .write(true)
            .append(true)
            .open(&partial_path)
            .map_err(|e| {
                DaemonError::model_download_failed(format!(
                    "Failed to open file for resume {}: {}",
                    partial_path.display(),
                    e
                ))
            })?;

        let mut downloaded = existing_size;
        let mut buffer = [0u8; 65536];
        let mut last_progress = (existing_size * 100 / total_size.max(1)) as usize;
        let mut last_callback_percent = last_progress;

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

            if total_size > 0 {
                let progress = (downloaded * 100 / total_size) as usize;
                if progress >= last_progress + 10 {
                    eprint!("{}%... ", progress);
                    last_progress = progress;
                }

                if let Some(ref callback) = on_progress {
                    let callback_percent = (downloaded * 100 / total_size) as usize;
                    if callback_percent > last_callback_percent {
                        callback(&filename, downloaded, total_size, files_completed, files_total);
                        last_callback_percent = callback_percent;
                    }
                }
            }
        }

        // Sync to disk before rename
        file.sync_all().map_err(|e| {
            DaemonError::model_download_failed(format!("Failed to sync file: {}", e))
        })?;
        drop(file);

        // Rename partial file to final destination
        fs::rename(&partial_path, dest).map_err(|e| {
            DaemonError::model_download_failed(format!(
                "Failed to rename {} to {}: {}",
                partial_path.display(),
                dest.display(),
                e
            ))
        })?;

        let size_mb = downloaded as f64 / (1024.0 * 1024.0);
        eprintln!("done ({:.1} MB total)", size_mb);

        if let Some(ref callback) = on_progress {
            callback(&filename, downloaded, downloaded, files_completed + 1, files_total);
        }

        Ok(())
    } else if status.is_success() {
        // Server doesn't support resume (returned 200 OK instead of 206 Partial Content)
        // Delete partial and do full download
        eprintln!("server doesn't support resume, restarting...");
        let _ = fs::remove_file(&partial_path);
        download_file_with_progress(url, dest, files_completed, files_total, on_progress)
    } else {
        Err(DaemonError::model_download_failed(format!(
            "HTTP {} for {}",
            status,
            url
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn get_model_dir() -> Option<PathBuf> {
        let proj_dirs = directories::ProjectDirs::from("", "", "lofi.nvim")?;
        let path = proj_dirs.cache_dir().join("musicgen");
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

