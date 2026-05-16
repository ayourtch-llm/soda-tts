//! Long-form synthesis: chunk text → `Supertonic::synthesize_one` per chunk →
//! concat with a configurable silence gap. Pattern mirrors
//! `kokoro-tts::synthesis` so the two can be swapped behind a small trait.

use anyhow::{bail, Result};
use std::time::Duration;

use crate::model::{Supertonic, SynthesisConfig, VoiceStyle};
use crate::text::{chunk_max_for, chunk_text};

/// Progress callback: (chunk_text, 1-based chunk index, total chunks, elapsed).
pub type ProgressFn = Box<dyn Fn(&str, usize, usize, Duration) + Send>;

/// Streaming audio callback: invoked once per chunk with (samples, idx, total).
pub type OnChunkFn = Box<dyn Fn(&[f32], usize, usize) + Send>;

#[derive(Debug, Clone)]
pub struct SynthesisOptions {
    /// Silence inserted between chunks (in seconds). 0.3 matches the
    /// reference; bump to 0.5 for audiobook-style pacing.
    pub silence_seconds: f32,
    /// Override `chunk_max_for(lang)` if set.
    pub max_chunk_chars: Option<usize>,
}

impl Default for SynthesisOptions {
    fn default() -> Self {
        Self { silence_seconds: 0.3, max_chunk_chars: None }
    }
}

/// One-shot synthesis: returns the full concatenated waveform.
pub fn synthesize_text(
    model: &mut Supertonic,
    text: &str,
    lang: &str,
    voice: &VoiceStyle,
    cfg: SynthesisConfig,
    opts: &SynthesisOptions,
) -> Result<Vec<f32>> {
    synthesize_text_streaming(model, text, lang, voice, cfg, opts, None, None)
}

/// Synthesis with optional per-chunk callbacks. Use `on_chunk` to start
/// playback before the full text finishes rendering.
pub fn synthesize_text_streaming(
    model: &mut Supertonic,
    text: &str,
    lang: &str,
    voice: &VoiceStyle,
    cfg: SynthesisConfig,
    opts: &SynthesisOptions,
    progress: Option<&ProgressFn>,
    on_chunk: Option<&OnChunkFn>,
) -> Result<Vec<f32>> {
    let max_chars = opts.max_chunk_chars.unwrap_or_else(|| chunk_max_for(lang));
    let chunks = chunk_text(text, max_chars);
    if chunks.is_empty() {
        bail!("no chunks to synthesize");
    }
    let total = chunks.len();
    let sr = model.sample_rate() as f32;
    let silence_len = (opts.silence_seconds * sr) as usize;

    let mut out: Vec<f32> = Vec::new();
    let t0 = std::time::Instant::now();
    for (idx, chunk) in chunks.iter().enumerate() {
        if chunk.trim().is_empty() {
            let silence = vec![0.0f32; silence_len];
            if let Some(cb) = on_chunk {
                cb(&silence, idx + 1, total);
            }
            out.extend_from_slice(&silence);
            if let Some(cb) = progress {
                cb(chunk, idx + 1, total, t0.elapsed());
            }
            continue;
        }
        let (raw_wav, dur_s) = model.synthesize_one(chunk, lang, voice, cfg)?;
        let audible = (dur_s * sr) as usize;
        let slice = &raw_wav[..audible.min(raw_wav.len())];
        if let Some(cb) = on_chunk {
            cb(slice, idx + 1, total);
        }
        out.extend_from_slice(slice);
        if idx + 1 < total {
            let silence = vec![0.0f32; silence_len];
            if let Some(cb) = on_chunk {
                cb(&silence, idx + 1, total);
            }
            out.extend_from_slice(&silence);
        }
        if let Some(cb) = progress {
            cb(chunk, idx + 1, total, t0.elapsed());
        }
    }
    Ok(out)
}
