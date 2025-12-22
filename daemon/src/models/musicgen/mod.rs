//! MusicGen model components.
//!
//! This module contains all the ONNX model wrappers for MusicGen:
//! - [`TextEncoder`](text_encoder::MusicGenTextEncoder): Text prompt encoding
//! - [`Decoder`](decoder::MusicGenDecoder): Autoregressive token generation
//! - [`AudioCodec`](audio_codec::MusicGenAudioCodec): Token to audio decoding
//! - [`DelayPatternMaskIds`](delay_pattern::DelayPatternMaskIds): 4-codebook delay pattern
//! - [`Logits`](logits::Logits): Logits processing and sampling

pub mod audio_codec;
pub mod decoder;
pub mod delay_pattern;
pub mod logits;
pub mod models;
pub mod text_encoder;

// Re-export commonly used types
pub use audio_codec::MusicGenAudioCodec;
pub use decoder::MusicGenDecoder;
pub use delay_pattern::DelayPatternMaskIds;
pub use logits::{Logits, DEFAULT_GUIDANCE_SCALE, DEFAULT_TOP_K};
pub use models::{
    check_models, detect_model_version, generate_model_version, load_sessions,
    load_sessions_with_device, MusicGenModels, MODEL_URLS, REQUIRED_MODEL_FILES,
};
pub use text_encoder::MusicGenTextEncoder;
