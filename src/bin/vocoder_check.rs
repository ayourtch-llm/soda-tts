//! Per-layer numerical diff: candle vocoder vs onnxruntime golden.
//!
//! Reads tmp/golden.safetensors (from tools/dump_golden.py) and
//! models/safetensors/vocoder.safetensors, runs the candle vocoder
//! with the same input, and prints max-abs delta per layer hook + the
//! end-to-end wav delta.
//!
//! Build + run:
//!   cargo build --release --bin vocoder_check
//!   ./target/release/vocoder_check

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use safetensors::{tensor::TensorView, SafeTensors};
use soda_tts::model::candle::{CandleVocoder, vocoder::Hooks};
use std::fs;
use std::path::PathBuf;

/// (golden_key, candle_hook_name). Hooks are listed in forward order so
/// the first divergence shows up earliest in the printout.
fn hook_map() -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = Vec::new();
    v.push(("vocoder/Add_output_0".into(), "pre_embed/after_unfold".into()));
    v.push(("vocoder/embed/net/Conv_output_0".into(), "embed/net/Conv_output_0".into()));
    for i in 0..10 {
        for sub in ["dwconv/net/Conv_output_0", "norm/norm/LayerNormalization_output_0",
                    "pwconv1/Conv_output_0", "pwconv2/Conv_output_0", "Add_output_0"] {
            v.push((
                format!("vocoder/convnext.{i}/{sub}"),
                format!("convnext.{i}/{sub}"),
            ));
        }
    }
    v.push(("vocoder/final_norm/BatchNormalization_output_0".into(),
            "final_norm/BatchNormalization_output_0".into()));
    v.push(("vocoder/head/layer1/net/Conv_output_0".into(),
            "head/layer1/net/Conv_output_0".into()));
    v.push(("vocoder/head/act/PRelu_output_0".into(),
            "head/act/PRelu_output_0".into()));
    v.push(("vocoder/head/layer2/Conv_output_0".into(),
            "head/layer2/Conv_output_0".into()));
    v
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let model_st = PathBuf::from("models/safetensors/vocoder.safetensors");
    let golden = PathBuf::from("tmp/golden.safetensors");

    eprintln!("loading candle vocoder from {} ...", model_st.display());
    let voc = CandleVocoder::load(&model_st, &device).context("load CandleVocoder")?;

    eprintln!("mmap golden activations from {} ...", golden.display());
    let bytes = fs::read(&golden).with_context(|| format!("reading {}", golden.display()))?;
    let st = SafeTensors::deserialize(&bytes).context("parse safetensors")?;

    let latent_in = load_f32_3d(&st, "vocoder/latent", &device)
        .or_else(|_| {
            // The golden file's vocoder input was named via the model's input
            // declaration. ort exposes it under `vocoder/latent` or under
            // `intermediate/final_latent` (set by dump_golden.py).
            load_f32_3d(&st, "intermediate/final_latent", &device)
        })
        .context("locating vocoder latent input in golden")?;
    eprintln!("vocoder input latent shape = {:?}", latent_in.dims());

    let mut hooks = Hooks::default();
    let t0 = std::time::Instant::now();
    let wav = voc.forward_with_hooks(&latent_in, &mut hooks)?;
    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!(
        "candle forward in {elapsed:.3}s  ->  wav shape {:?}",
        wav.dims()
    );

    // Compare each hook against its golden counterpart.
    let mut worst = 0.0f32;
    let mut worst_name = String::new();
    for (golden_key, hook_name) in &hook_map() {
        let ours = match hooks.get(hook_name) {
            Some(t) => t,
            None => {
                println!("  {:>60}: hook not recorded", hook_name);
                continue;
            }
        };
        let theirs = match try_load_any_shape(&st, golden_key, &device) {
            Ok(t) => t,
            Err(_) => {
                println!("  {:>60}: golden key absent", golden_key);
                continue;
            }
        };
        let (max_abs, ratio) = diff_stats(ours, &theirs)?;
        if max_abs > worst {
            worst = max_abs;
            worst_name = hook_name.to_string();
        }
        println!(
            "  {:>60}  max_abs={:.3e}  rel={:.3e}  shapes=ours{:?}/golden{:?}",
            hook_name,
            max_abs,
            ratio,
            ours.dims(),
            theirs.dims()
        );
    }

    // Final wav comparison.
    let wav_golden = load_f32_2d(&st, "vocoder/wav_tts", &device)
        .context("vocoder/wav_tts missing from golden")?;
    let (max_abs_wav, _) = diff_stats(&wav, &wav_golden)?;
    println!();
    println!("FINAL wav max_abs_delta = {max_abs_wav:.3e}");
    println!("worst per-layer hook   : {worst_name} ({worst:.3e})");

    // Also write the candle wav so we can run it through ASR for a
    // functional check — numerical delta below ~1e-2 may still be
    // perceptually identical.
    let wav_path = PathBuf::from("tmp/candle_vocoder.wav");
    let samples_f32 = wav
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    soda_tts::audio::write_wav(&samples_f32, &wav_path, 44_100)?;
    eprintln!("wrote candle wav to {}", wav_path.display());

    if max_abs_wav < 5e-4 {
        eprintln!("\nPASS: end-to-end wav within 5e-4 of golden.");
    } else {
        eprintln!(
            "\nFAIL: wav delta {max_abs_wav:.3e} exceeds 5e-4 threshold.\n\
             Try ASR on {} to check whether the difference is perceptible.",
            wav_path.display(),
        );
        std::process::exit(1);
    }
    Ok(())
}

fn diff_stats(ours: &Tensor, theirs: &Tensor) -> Result<(f32, f32)> {
    let a = ours.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = theirs.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    if a.len() != b.len() {
        bail!("shape mismatch: ours has {} elems, golden has {}", a.len(), b.len());
    }
    let mut max_abs = 0.0f32;
    let mut sum_abs_diff = 0.0f64;
    let mut sum_abs = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > max_abs {
            max_abs = d;
        }
        sum_abs_diff += d as f64;
        sum_abs += y.abs() as f64;
    }
    let rel = if sum_abs > 0.0 {
        (sum_abs_diff / sum_abs) as f32
    } else {
        0.0
    };
    Ok((max_abs, rel))
}

fn load_f32_3d(st: &SafeTensors, key: &str, device: &Device) -> Result<Tensor> {
    let view = st
        .tensor(key)
        .map_err(|e| anyhow!("{key}: {e}"))?;
    let dims = view.shape();
    if dims.len() != 3 {
        bail!("{key}: expected 3D, got {dims:?}");
    }
    let arr = view_to_f32_vec(&view)?;
    Tensor::from_vec(arr, (dims[0], dims[1], dims[2]), device).context("Tensor::from_vec")
}

fn load_f32_2d(st: &SafeTensors, key: &str, device: &Device) -> Result<Tensor> {
    let view = st.tensor(key).map_err(|e| anyhow!("{key}: {e}"))?;
    let dims = view.shape();
    if dims.len() != 2 {
        bail!("{key}: expected 2D, got {dims:?}");
    }
    let arr = view_to_f32_vec(&view)?;
    Tensor::from_vec(arr, (dims[0], dims[1]), device).context("Tensor::from_vec")
}

fn try_load_any_shape(st: &SafeTensors, key: &str, device: &Device) -> Result<Tensor> {
    let view = st.tensor(key).map_err(|e| anyhow!("{key}: {e}"))?;
    let dims = view.shape();
    let arr = view_to_f32_vec(&view)?;
    Tensor::from_vec(arr, dims.to_vec(), device).context("Tensor::from_vec")
}

fn view_to_f32_vec(view: &TensorView) -> Result<Vec<f32>> {
    use safetensors::Dtype as D;
    let bytes = view.data();
    let n = view.shape().iter().product::<usize>();
    let v: Vec<f32> = match view.dtype() {
        D::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        other => bail!("unsupported dtype {other:?} (n={n}); golden uses f32 only"),
    };
    if v.len() != n {
        bail!("decoded {} elements, expected {n}", v.len());
    }
    Ok(v)
}
