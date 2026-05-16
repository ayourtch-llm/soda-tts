//! Supertonic 3 model: 4 ONNX sessions + voice style + end-to-end forward.

pub mod config;
pub mod voice;

use anyhow::{anyhow, Context, Result};
use ndarray::{Array, Array3};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::fmt::Display;
use std::path::Path;

pub use config::Config;
pub use voice::VoiceStyle;

use crate::text::{preprocess, UnicodeIndexer};

/// `ort` errors carry a generic phantom over the operation type, which
/// breaks `anyhow::Context`'s `Send + Sync + StdError` bound. Round-trip
/// through the `Display` impl to get a normal `anyhow::Error`.
fn ort_err<E: Display>(stage: &'static str) -> impl FnOnce(E) -> anyhow::Error {
    move |e| anyhow!("{stage}: {e}")
}

/// Knobs that control synthesis quality and speed.
#[derive(Debug, Clone, Copy)]
pub struct SynthesisConfig {
    /// Flow-matching ODE steps. Reference default is 8. Lower = faster &
    /// rougher, higher = slower & smoother. 4–16 is the useful range.
    pub total_steps: usize,
    /// Speech rate multiplier. >1 speeds up, <1 slows down. 1.05 is the
    /// reference default (slightly faster than natural).
    pub speed: f32,
    /// Optional RNG seed for the Gaussian latent. `None` = nondeterministic.
    pub seed: Option<u64>,
}

impl Default for SynthesisConfig {
    fn default() -> Self {
        Self { total_steps: 8, speed: 1.05, seed: None }
    }
}

/// All four ONNX sessions plus the tokenizer and config.
pub struct Supertonic {
    pub config: Config,
    pub indexer: UnicodeIndexer,
    text_encoder: Session,
    duration_predictor: Session,
    vector_estimator: Session,
    vocoder: Session,
}

impl Supertonic {
    pub fn sample_rate(&self) -> i32 {
        self.config.ae.sample_rate
    }

    /// Load all four ONNX files + tts.json + unicode_indexer.json from
    /// `onnx_dir`. The layout matches `Supertone/supertonic-3` on HF:
    ///
    /// ```text
    /// onnx_dir/
    ///   tts.json
    ///   unicode_indexer.json
    ///   text_encoder.onnx
    ///   duration_predictor.onnx
    ///   vector_estimator.onnx
    ///   vocoder.onnx
    /// ```
    pub fn load(onnx_dir: &Path) -> Result<Self> {
        let config = Config::load(&onnx_dir.join("tts.json"))?;
        let indexer = UnicodeIndexer::load(&onnx_dir.join("unicode_indexer.json"))?;

        let load = |name: &str| -> Result<Session> {
            let path = onnx_dir.join(name);
            Session::builder()
                .map_err(ort_err("session builder"))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(ort_err("set optimization level"))?
                .commit_from_file(&path)
                .map_err(ort_err("commit_from_file"))
                .with_context(|| format!("loading {}", path.display()))
        };

        Ok(Self {
            config,
            indexer,
            text_encoder: load("text_encoder.onnx")?,
            duration_predictor: load("duration_predictor.onnx")?,
            vector_estimator: load("vector_estimator.onnx")?,
            vocoder: load("vocoder.onnx")?,
        })
    }

    /// Render a single utterance (no chunking). For text longer than
    /// ~300 chars (or ~120 for ko/ja), use `synthesis::synthesize_text`
    /// which calls this in a loop.
    ///
    /// Returns `(samples, duration_seconds)`. `samples.len()` is the
    /// vocoder's raw output length; the *audible* prefix is
    /// `(duration * sample_rate) as usize` — anything past that is
    /// padding from the model's chunked latent.
    pub fn synthesize_one(
        &mut self,
        text: &str,
        lang: &str,
        voice: &VoiceStyle,
        cfg: SynthesisConfig,
    ) -> Result<(Vec<f32>, f32)> {
        let processed = preprocess(text, lang)?;
        let (text_ids, _lengths, text_mask) = self.indexer.encode_batch(&[processed]);

        let bsz = 1usize;
        let seq_len = text_ids[0].len();
        let mut flat = Vec::with_capacity(bsz * seq_len);
        for row in &text_ids {
            flat.extend_from_slice(row);
        }
        let text_ids_arr = Array::from_shape_vec((bsz, seq_len), flat)?;

        // --- Duration predictor: returns one f32 per batch row (seconds). --
        let v_text_ids = Value::from_array(text_ids_arr.clone()).map_err(ort_err("text_ids"))?;
        let v_style_dp = Value::from_array(voice.dp.clone()).map_err(ort_err("style_dp"))?;
        let v_text_mask = Value::from_array(text_mask.clone()).map_err(ort_err("text_mask"))?;
        let dp_out = self
            .duration_predictor
            .run(ort::inputs! {
                "text_ids"  => v_text_ids,
                "style_dp"  => v_style_dp,
                "text_mask" => v_text_mask,
            })
            .map_err(ort_err("duration_predictor.run"))?;
        let (_shape, dur_slice) = dp_out["duration"]
            .try_extract_tensor::<f32>()
            .map_err(ort_err("extract duration"))?;
        let mut duration: Vec<f32> = dur_slice.to_vec();
        for d in &mut duration {
            *d /= cfg.speed;
        }
        let dur_s = duration[0];

        // --- Text encoder: returns [B, T, D] text embeddings. -----------
        let v_text_ids = Value::from_array(text_ids_arr).map_err(ort_err("text_ids"))?;
        let v_style_ttl = Value::from_array(voice.ttl.clone()).map_err(ort_err("style_ttl"))?;
        let v_text_mask = Value::from_array(text_mask.clone()).map_err(ort_err("text_mask"))?;
        let te_out = self
            .text_encoder
            .run(ort::inputs! {
                "text_ids"  => v_text_ids,
                "style_ttl" => v_style_ttl,
                "text_mask" => v_text_mask,
            })
            .map_err(ort_err("text_encoder.run"))?;
        let (te_shape, te_slice) = te_out["text_emb"]
            .try_extract_tensor::<f32>()
            .map_err(ort_err("extract text_emb"))?;
        let te_shape3 = (
            te_shape[0] as usize,
            te_shape[1] as usize,
            te_shape[2] as usize,
        );
        let text_emb = Array3::<f32>::from_shape_vec(te_shape3, te_slice.to_vec())?;

        // --- Sample initial Gaussian latent and its mask. ---------------
        let (mut xt, latent_mask) = sample_noisy_latent(
            &duration,
            self.config.ae.sample_rate,
            self.config.ae.base_chunk_size,
            self.config.ttl.chunk_compress_factor,
            self.config.ttl.latent_dim,
            cfg.seed,
        );

        // --- Flow-matching ODE loop (Euler-like; the model does its own
        // step internally given current/total). -------------------------
        let total_step_arr = Array::from_elem((bsz,), cfg.total_steps as f32);
        for step in 0..cfg.total_steps {
            let cur_arr = Array::from_elem((bsz,), step as f32);
            let v_noisy = Value::from_array(xt.clone()).map_err(ort_err("noisy_latent"))?;
            let v_te = Value::from_array(text_emb.clone()).map_err(ort_err("text_emb"))?;
            let v_st = Value::from_array(voice.ttl.clone()).map_err(ort_err("style_ttl"))?;
            let v_lm = Value::from_array(latent_mask.clone()).map_err(ort_err("latent_mask"))?;
            let v_tm = Value::from_array(text_mask.clone()).map_err(ort_err("text_mask"))?;
            let v_cur = Value::from_array(cur_arr).map_err(ort_err("current_step"))?;
            let v_tot = Value::from_array(total_step_arr.clone()).map_err(ort_err("total_step"))?;
            let ve_out = self
                .vector_estimator
                .run(ort::inputs! {
                    "noisy_latent" => v_noisy,
                    "text_emb"     => v_te,
                    "style_ttl"    => v_st,
                    "latent_mask"  => v_lm,
                    "text_mask"    => v_tm,
                    "current_step" => v_cur,
                    "total_step"   => v_tot,
                })
                .map_err(ort_err("vector_estimator.run"))
                .with_context(|| format!("flow step {step}"))?;
            let (sh, slc) = ve_out["denoised_latent"]
                .try_extract_tensor::<f32>()
                .map_err(ort_err("extract denoised_latent"))?;
            xt = Array3::<f32>::from_shape_vec(
                (sh[0] as usize, sh[1] as usize, sh[2] as usize),
                slc.to_vec(),
            )?;
        }

        // --- Vocoder → waveform [B, 1, N_samples] (or [B, N_samples]). --
        let v_latent = Value::from_array(xt).map_err(ort_err("latent"))?;
        let voc_out = self
            .vocoder
            .run(ort::inputs! { "latent" => v_latent })
            .map_err(ort_err("vocoder.run"))?;
        let (_wsh, wav_slice) = voc_out["wav_tts"]
            .try_extract_tensor::<f32>()
            .map_err(ort_err("extract wav_tts"))?;
        Ok((wav_slice.to_vec(), dur_s))
    }
}

/// Sample N(0, 1) of shape (B, latent_dim * chunk_compress, latent_len),
/// then mask the time dimension by per-row latent_lengths. Returns the
/// noisy latent and the 3-D mask `(B, 1, latent_len)`.
fn sample_noisy_latent(
    duration_s: &[f32],
    sample_rate: i32,
    base_chunk_size: i32,
    chunk_compress: i32,
    latent_dim: i32,
    seed: Option<u64>,
) -> (Array3<f32>, Array3<f32>) {
    let bsz = duration_s.len();
    let max_dur = duration_s.iter().copied().fold(0.0_f32, f32::max);

    let wav_len_max = (max_dur * sample_rate as f32) as usize;
    let chunk_size = (base_chunk_size * chunk_compress) as usize;
    let latent_len = wav_len_max.div_ceil(chunk_size).max(1);
    let latent_channels = (latent_dim * chunk_compress) as usize;

    let normal = Normal::new(0.0f32, 1.0f32).unwrap();
    let mut noise: Vec<f32> = match seed {
        Some(s) => {
            let mut rng = rand::rngs::StdRng::seed_from_u64(s);
            (0..bsz * latent_channels * latent_len)
                .map(|_| normal.sample(&mut rng))
                .collect()
        }
        None => {
            let mut rng = rand::thread_rng();
            (0..bsz * latent_channels * latent_len)
                .map(|_| normal.sample(&mut rng))
                .collect()
        }
    };

    let mut mask = Array3::<f32>::zeros((bsz, 1, latent_len));
    for (b, &d) in duration_s.iter().enumerate() {
        let wav_len = (d * sample_rate as f32) as usize;
        let row_latent_len = wav_len.div_ceil(chunk_size).min(latent_len);
        for t in 0..row_latent_len {
            mask[[b, 0, t]] = 1.0;
        }
        // Zero out the padding region in the noise tensor itself, matching
        // the reference implementation (which masks before the first ODE step).
        for c in 0..latent_channels {
            for t in row_latent_len..latent_len {
                let idx = b * latent_channels * latent_len + c * latent_len + t;
                noise[idx] = 0.0;
            }
        }
    }

    let xt = Array3::<f32>::from_shape_vec((bsz, latent_channels, latent_len), noise).unwrap();
    (xt, mask)
}
