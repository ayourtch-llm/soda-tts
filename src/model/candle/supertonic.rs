//! End-to-end candle Supertonic 3: text -> 44.1 kHz mono WAV samples.
//!
//! Mirrors the external API of `crate::model::Supertonic` (the ort-backed
//! version), so callers can swap engines by changing the type. Each of
//! the 4 sub-models is loaded once at construction.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::path::Path;

use crate::model::candle::{
    CandleDurationPredictor, CandleTextEncoder, CandleVectorEstimator, CandleVocoder,
};
use crate::model::config::Config;
use crate::model::voice::VoiceStyle;
use crate::model::SynthesisConfig;
use crate::text::{preprocess, UnicodeIndexer};

pub struct CandleSupertonic {
    pub config: Config,
    pub indexer: UnicodeIndexer,
    device: Device,
    duration: CandleDurationPredictor,
    text_encoder: CandleTextEncoder,
    vector_estimator: CandleVectorEstimator,
    vocoder: CandleVocoder,
}

impl CandleSupertonic {
    pub fn sample_rate(&self) -> i32 {
        self.config.ae.sample_rate
    }

    /// Load all four candle models + tokenizer + config. `model_dir`
    /// expects two sub-directories:
    ///   model_dir/onnx/{tts.json, unicode_indexer.json}
    ///   model_dir/safetensors/{duration_predictor, text_encoder,
    ///                          vector_estimator, vocoder}.safetensors
    pub fn load(model_dir: &Path, device: &Device) -> Result<Self> {
        let onnx_dir = model_dir.join("onnx");
        let st_dir = model_dir.join("safetensors");
        let config = Config::load(&onnx_dir.join("tts.json"))?;
        let indexer = UnicodeIndexer::load(&onnx_dir.join("unicode_indexer.json"))?;
        let duration = CandleDurationPredictor::load(
            &st_dir.join("duration_predictor.safetensors"), device,
        ).context("loading candle duration_predictor")?;
        let text_encoder = CandleTextEncoder::load(
            &st_dir.join("text_encoder.safetensors"), device,
        ).context("loading candle text_encoder")?;
        let vector_estimator = CandleVectorEstimator::load(
            &st_dir.join("vector_estimator.safetensors"), device,
        ).context("loading candle vector_estimator")?;
        let vocoder = CandleVocoder::load(
            &st_dir.join("vocoder.safetensors"), device,
        ).context("loading candle vocoder")?;
        Ok(Self {
            config, indexer, device: device.clone(),
            duration, text_encoder, vector_estimator, vocoder,
        })
    }

    /// Render a single utterance to f32 samples + duration in seconds.
    /// Mirrors `model::Supertonic::synthesize_one`.
    pub fn synthesize_one(
        &self,
        text: &str,
        lang: &str,
        voice: &VoiceStyle,
        cfg: SynthesisConfig,
    ) -> Result<(Vec<f32>, f32)> {
        // ----- 1. Tokenize. ---------------------------------------------
        let processed = preprocess(text, lang)?;
        let (text_ids_rows, _lengths, text_mask_arr) = self.indexer.encode_batch(&[processed]);
        let bsz = 1;
        let t_text = text_ids_rows[0].len();
        let mut flat = Vec::with_capacity(bsz * t_text);
        for row in &text_ids_rows { flat.extend_from_slice(row); }
        let text_ids = Tensor::from_vec(flat, (bsz, t_text), &self.device)
            .context("text_ids tensor")?;
        // text_mask_arr is an ndarray::Array3<f32> shape [B, 1, T]. Flatten + reshape.
        let mask_vec: Vec<f32> = text_mask_arr.iter().copied().collect();
        let text_mask = Tensor::from_vec(mask_vec, (bsz, 1usize, t_text), &self.device)
            .context("text_mask tensor")?;

        // Voice style tensors: ndarray::Array3<f32> -> candle Tensor.
        let style_ttl = ndarray3_to_candle(&voice.ttl, &self.device)?;
        let style_dp = ndarray3_to_candle(&voice.dp, &self.device)?;

        // ----- 2. Duration prediction. ----------------------------------
        let dur_t = self.duration.forward(&text_ids, &style_dp, &text_mask)
            .context("candle duration forward")?;
        let dur_raw: f32 = dur_t.to_vec1::<f32>()?[0];
        let duration_seconds = dur_raw / cfg.speed;

        // ----- 3. Text encoder. -----------------------------------------
        let text_emb = self.text_encoder.forward(&text_ids, &style_ttl, &text_mask)
            .context("candle text_encoder forward")?;

        // ----- 4. Sample initial Gaussian latent. -----------------------
        let sample_rate = self.config.ae.sample_rate;
        let base_chunk = self.config.ae.base_chunk_size;
        let chunk_compress = self.config.ttl.chunk_compress_factor;
        let latent_dim = self.config.ttl.latent_dim;
        // Optional: if SODA_USE_GOLDEN_NOISE=1, load noise from the
        // already-extracted tmp/golden.safetensors for reproducibility.
        // This is for debugging only -- production should sample via RNG.
        let (noisy_latent, latent_mask) = if std::env::var("SODA_USE_GOLDEN_NOISE").is_ok() {
            load_golden_noise(&self.device)?
        } else {
            sample_noisy_latent(
                duration_seconds, sample_rate, base_chunk, chunk_compress, latent_dim,
                cfg.seed, &self.device,
            )?
        };

        // ----- 5. Flow-matching ODE loop. -------------------------------
        let total_step = Tensor::new(&[cfg.total_steps as f32], &self.device)?;
        let mut xt = noisy_latent;
        for step in 0..cfg.total_steps {
            let current_step = Tensor::new(&[step as f32], &self.device)?;
            xt = self.vector_estimator.forward(
                &xt, &text_emb, &style_ttl,
                &latent_mask, &text_mask,
                &current_step, &total_step,
            ).with_context(|| format!("vector_estimator step {step}"))?;
        }

        // ----- 6. Vocode. -----------------------------------------------
        let wav = self.vocoder.forward(&xt).context("candle vocoder forward")?;
        // wav shape: [B, T_samples]. Take batch 0 as a Vec<f32>.
        let samples: Vec<f32> = wav.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        // Audible prefix per the duration the DP predicted.
        let audible = (duration_seconds * sample_rate as f32) as usize;
        let cut = audible.min(samples.len());
        Ok((samples[..cut].to_vec(), duration_seconds))
    }
}

/// Sample the same Gaussian latent that the ort backend would (deterministic
/// per `seed`, or non-deterministic if `None`). Returns (noisy, mask).
/// Shape: noisy `[1, latent_dim*chunk_compress, latent_len]`, mask `[1, 1, latent_len]`.
fn sample_noisy_latent(
    duration_s: f32, sample_rate: i32, base_chunk_size: i32,
    chunk_compress: i32, latent_dim: i32, seed: Option<u64>, device: &Device,
) -> Result<(Tensor, Tensor)> {
    let wav_len = (duration_s * sample_rate as f32) as usize;
    let chunk_size = (base_chunk_size * chunk_compress) as usize;
    let latent_len = wav_len.div_ceil(chunk_size).max(1);
    let latent_channels = (latent_dim * chunk_compress) as usize;
    let n = latent_channels * latent_len;
    let normal = Normal::new(0.0f32, 1.0).unwrap();
    let noise: Vec<f32> = match seed {
        Some(s) => {
            let mut rng = rand::rngs::StdRng::seed_from_u64(s);
            (0..n).map(|_| normal.sample(&mut rng)).collect()
        }
        None => {
            let mut rng = rand::thread_rng();
            (0..n).map(|_| normal.sample(&mut rng)).collect()
        }
    };
    let noisy = Tensor::from_vec(noise, (1, latent_channels, latent_len), device)
        .context("noisy_latent tensor")?;
    let mask = Tensor::ones((1usize, 1usize, latent_len), DType::F32, device)
        .context("latent_mask tensor")?;
    Ok((noisy, mask))
}

fn ndarray3_to_candle(arr: &ndarray::Array3<f32>, device: &Device) -> Result<Tensor> {
    let (b, t, d) = arr.dim();
    // Flatten in row-major order (the default for ndarray, matches what
    // candle expects for from_vec).
    let v: Vec<f32> = arr.iter().copied().collect();
    Tensor::from_vec(v, (b, t, d), device).context("Array3 -> Tensor")
}

fn load_golden_noise(device: &Device) -> Result<(Tensor, Tensor)> {
    use safetensors::SafeTensors;
    let bytes = std::fs::read("tmp/golden.safetensors").context("reading golden")?;
    let st = SafeTensors::deserialize(&bytes).context("parse golden")?;
    let view = st.tensor("input/noisy_latent").map_err(|e| anyhow::anyhow!("{e}"))?;
    let dims = view.shape().to_vec();
    let v: Vec<f32> = view.data().chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let noisy = Tensor::from_vec(v, dims.clone(), device)?;
    let view = st.tensor("input/latent_mask").map_err(|e| anyhow::anyhow!("{e}"))?;
    let mdims = view.shape().to_vec();
    let mv: Vec<f32> = view.data().chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let mask = Tensor::from_vec(mv, mdims, device)?;
    Ok((noisy, mask))
}

#[allow(dead_code)]
fn _unused() -> Result<()> {
    bail!("unused")
}
