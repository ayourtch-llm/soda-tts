//! Audio I/O. 16-bit PCM WAV writer at the model's native 44.1 kHz.

use anyhow::{Context, Result};
use hound::{SampleFormat, WavSpec, WavWriter};
use std::path::Path;

/// Write a mono f32 sample buffer as 16-bit PCM WAV at `sample_rate` Hz.
/// Samples are clipped to [-1, 1] before quantization.
pub fn write_wav(samples: &[f32], path: &Path, sample_rate: i32) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
    }
    let spec = WavSpec {
        channels: 1,
        sample_rate: sample_rate as u32,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut w = WavWriter::create(path, spec)
        .with_context(|| format!("creating wav {}", path.display()))?;
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let pcm = (clamped * i16::MAX as f32) as i16;
        w.write_sample(pcm).context("write_sample")?;
    }
    w.finalize().context("finalize")?;
    Ok(())
}

/// Linear resampling. Adequate for ASR ingestion (speech is band-limited
/// well below the Nyquist of either rate); not for archival audio.
pub fn resample_linear(input: &[f32], src_hz: i32, dst_hz: i32) -> Vec<f32> {
    if input.is_empty() || src_hz == dst_hz {
        return input.to_vec();
    }
    let out_len = input.len() * dst_hz as usize / src_hz as usize;
    if out_len == 0 || input.len() < 2 {
        return Vec::new();
    }
    let ratio = input.len() as f64 / out_len as f64;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f64 * ratio;
        let t0 = pos.floor() as usize;
        let t1 = (t0 + 1).min(input.len() - 1);
        let frac = (pos - t0 as f64) as f32;
        out.push(input[t0] * (1.0 - frac) + input[t1] * frac);
    }
    out
}
