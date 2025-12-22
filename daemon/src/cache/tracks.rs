//! Track cache with LRU eviction.
//!
//! Provides in-memory caching of generated tracks with hash-based deduplication.

use std::collections::HashMap;
use std::time::Instant;

use crate::types::Track;

/// Maximum number of tracks to keep in cache.
const DEFAULT_MAX_ENTRIES: usize = 100;

/// Track cache with LRU eviction policy.
pub struct TrackCache {
    /// Tracks indexed by track_id.
    tracks: HashMap<String, CacheEntry>,
    /// Maximum number of entries to keep.
    max_entries: usize,
}

/// A cached track with access timestamp.
struct CacheEntry {
    track: Track,
    last_accessed: Instant,
}

impl TrackCache {
    /// Creates a new cache with default capacity.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_MAX_ENTRIES)
    }

    /// Creates a new cache with specified capacity.
    pub fn with_capacity(max_entries: usize) -> Self {
        Self {
            tracks: HashMap::new(),
            max_entries,
        }
    }

    /// Returns a track by ID, updating its access time.
    pub fn get(&mut self, track_id: &str) -> Option<&Track> {
        if let Some(entry) = self.tracks.get_mut(track_id) {
            entry.last_accessed = Instant::now();
            Some(&entry.track)
        } else {
            None
        }
    }

    /// Inserts a track into the cache.
    ///
    /// If the cache is full, the least recently used entry is evicted first.
    pub fn put(&mut self, track: Track) {
        // Evict if at capacity and this is a new entry
        if self.tracks.len() >= self.max_entries && !self.tracks.contains_key(&track.track_id) {
            self.evict_lru();
        }

        let track_id = track.track_id.clone();
        self.tracks.insert(
            track_id,
            CacheEntry {
                track,
                last_accessed: Instant::now(),
            },
        );
    }

    /// Checks if a track ID exists in the cache.
    pub fn contains(&self, track_id: &str) -> bool {
        self.tracks.contains_key(track_id)
    }

    /// Returns the number of tracks in the cache.
    pub fn len(&self) -> usize {
        self.tracks.len()
    }

    /// Returns true if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }

    /// Evicts the least recently used entry.
    ///
    /// Returns the evicted track if any.
    pub fn evict_lru(&mut self) -> Option<Track> {
        if self.tracks.is_empty() {
            return None;
        }

        // Find the entry with the oldest access time
        let oldest_key = self
            .tracks
            .iter()
            .min_by_key(|(_, entry)| entry.last_accessed)
            .map(|(k, _)| k.clone())?;

        self.tracks.remove(&oldest_key).map(|entry| entry.track)
    }

    /// Removes a specific track from the cache.
    pub fn remove(&mut self, track_id: &str) -> Option<Track> {
        self.tracks.remove(track_id).map(|entry| entry.track)
    }

    /// Clears all entries from the cache.
    pub fn clear(&mut self) {
        self.tracks.clear();
    }
}

impl Default for TrackCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Backend;
    use std::thread;
    use std::time::Duration;

    fn make_track(id: &str) -> Track {
        use std::path::PathBuf;
        use std::time::SystemTime;
        Track {
            track_id: id.to_string(),
            path: PathBuf::from(format!("/path/to/{}.wav", id)),
            prompt: "test prompt".to_string(),
            duration_sec: 10.0,
            sample_rate: 32000,
            seed: 12345,
            model_version: "musicgen-small-fp16-v1".to_string(),
            backend: Backend::MusicGen,
            generation_time_sec: 25.0,
            created_at: SystemTime::now(),
        }
    }

    #[test]
    fn new_cache_is_empty() {
        let cache = TrackCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn put_and_get() {
        let mut cache = TrackCache::new();
        let track = make_track("abc123");

        cache.put(track.clone());

        assert!(cache.contains("abc123"));
        assert_eq!(cache.len(), 1);

        let retrieved = cache.get("abc123").unwrap();
        assert_eq!(retrieved.track_id, "abc123");
    }

    #[test]
    fn get_nonexistent_returns_none() {
        let mut cache = TrackCache::new();
        assert!(cache.get("nonexistent").is_none());
    }

    #[test]
    fn evict_lru_removes_oldest() {
        let mut cache = TrackCache::with_capacity(2);

        cache.put(make_track("first"));
        thread::sleep(Duration::from_millis(10));
        cache.put(make_track("second"));

        // Access first to make it more recent
        cache.get("first");
        thread::sleep(Duration::from_millis(10));

        // Adding third should evict second (least recently accessed)
        cache.put(make_track("third"));

        assert!(cache.contains("first"));
        assert!(!cache.contains("second"));
        assert!(cache.contains("third"));
    }

    #[test]
    fn remove_track() {
        let mut cache = TrackCache::new();
        cache.put(make_track("abc123"));

        let removed = cache.remove("abc123");
        assert!(removed.is_some());
        assert!(!cache.contains("abc123"));
    }

    #[test]
    fn clear_removes_all() {
        let mut cache = TrackCache::new();
        cache.put(make_track("a"));
        cache.put(make_track("b"));
        cache.put(make_track("c"));

        cache.clear();

        assert!(cache.is_empty());
    }
}
