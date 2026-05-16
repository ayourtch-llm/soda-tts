//! Per-layer numerical diff: candle duration_predictor vs onnxruntime golden.
//!
//! Run with no args after `tools/dump_golden.py` and the safetensors
//! conversion have populated `tmp/golden.safetensors` and
//! `models/safetensors/duration_predictor.safetensors`.

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use safetensors::{tensor::TensorView, SafeTensors};
use soda_tts::model::candle::{
    duration_predictor::Hooks, CandleDurationPredictor,
};
use std::fs;
use std::path::PathBuf;

/// (golden_key, candle_hook_name). Order = forward order so the first
/// divergence shows up earliest.
fn hook_map() -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = Vec::new();
    v.push(("duration_predictor/Add_output_0".into(), "Add_output_0".into()));
    // ConvNeXt sub-layers
    for i in 0..6 {
        for (gold_sub, cand_sub) in [
            ("dwconv/Conv_output_0", "dwconv/Conv_output_0"),
            ("norm/norm/LayerNormalization_output_0", "norm/norm/LayerNormalization_output_0"),
            ("pwconv1/Conv_output_0", "pwconv1/Conv_output_0"),
            ("pwconv2/Conv_output_0", "pwconv2/Conv_output_0"),
            ("Add_output_0", "Add_output_0"),
        ] {
            v.push((
                format!("duration_predictor/convnext/convnext.{i}/{gold_sub}"),
                format!("convnext.{i}/{cand_sub}"),
            ));
        }
    }
    // AttnEncoder per-layer hooks
    for i in 0..2 {
        v.push((
            format!("duration_predictor/attn_encoder/norm_layers_1.{i}/norm/LayerNormalization_output_0"),
            format!("norm_layers_1.{i}/norm/LayerNormalization_output_0"),
        ));
        for sub in ["conv_q/Conv_output_0", "conv_k/Conv_output_0", "conv_v/Conv_output_0",
                    "MatMul_output_0", "MatMul_1_output_0", "Add_2_output_0",
                    "Softmax_output_0", "MatMul_2_output_0", "MatMul_3_output_0",
                    "Add_4_output_0", "conv_o/Conv_output_0"] {
            v.push((
                format!("duration_predictor/attn_encoder/attn_layers.{i}/{sub}"),
                format!("attn_layers.{i}/{sub}"),
            ));
        }
        v.push((
            format!("duration_predictor/attn_encoder/Add_{}_output_0", 2 * i),
            format!("attn_encoder/Add_{}_output_0", 2 * i),
        ));
        v.push((
            format!("duration_predictor/attn_encoder/norm_layers_2.{i}/norm/LayerNormalization_output_0"),
            format!("norm_layers_2.{i}/norm/LayerNormalization_output_0"),
        ));
        for sub in ["conv_1/Conv_output_0", "conv_2/Conv_output_0"] {
            v.push((
                format!("duration_predictor/attn_encoder/ffn_layers.{i}/{sub}"),
                format!("ffn_layers.{i}/{sub}"),
            ));
        }
        v.push((
            format!("duration_predictor/attn_encoder/Add_{}_output_0", 2 * i + 1),
            format!("attn_encoder/Add_{}_output_0", 2 * i + 1),
        ));
    }
    v.push(("duration_predictor/proj_out/net/Conv_output_0".into(),
            "proj_out/net/Conv_output_0".into()));
    v.push(("duration_predictor/layers.0/Gemm_output_0".into(),
            "layers.0/Gemm_output_0".into()));
    v.push(("duration_predictor/activation/PRelu_output_0".into(),
            "activation/PRelu_output_0".into()));
    v.push(("duration_predictor/layers.1/Gemm_output_0".into(),
            "layers.1/Gemm_output_0".into()));
    v
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let model_st = PathBuf::from("models/safetensors/duration_predictor.safetensors");
    let golden = PathBuf::from("tmp/golden.safetensors");

    eprintln!("loading candle dp from {} ...", model_st.display());
    let dp = CandleDurationPredictor::load(&model_st, &device).context("load CandleDP")?;

    eprintln!("mmap golden activations from {} ...", golden.display());
    let bytes = fs::read(&golden)?;
    let st = SafeTensors::deserialize(&bytes)?;

    let text_ids = load_i64(&st, "input/text_ids", &device)?;
    let text_mask = load_f32(&st, "input/text_mask", &device)?;
    let style_dp = load_f32(&st, "input/style_dp", &device)?;
    eprintln!("text_ids {:?}, text_mask {:?}, style_dp {:?}",
        text_ids.dims(), text_mask.dims(), style_dp.dims());

    let mut hooks = Hooks::default();
    let t0 = std::time::Instant::now();
    let duration = dp.forward_with_hooks(&text_ids, &style_dp, &text_mask, &mut hooks)?;
    eprintln!("candle dp forward in {:.3}s -> duration {:?}",
        t0.elapsed().as_secs_f64(), duration.dims());

    let mut worst = 0.0f32;
    let mut worst_name = String::new();
    for (golden_key, hook_name) in &hook_map() {
        let ours = match hooks.get(hook_name) {
            Some(t) => t,
            None => { println!("  {:>70}: hook not recorded", hook_name); continue; }
        };
        let theirs = match try_load_any_shape(&st, golden_key, &device) {
            Ok(t) => t,
            Err(_) => { println!("  {:>70}: golden absent", golden_key); continue; }
        };
        let (max_abs, rel) = diff_stats(ours, &theirs)?;
        if max_abs > worst { worst = max_abs; worst_name = hook_name.into(); }
        println!("  {:>70}  max_abs={:.3e}  rel={:.3e}  ours{:?}/gold{:?}",
            hook_name, max_abs, rel, ours.dims(), theirs.dims());
    }

    let dur_golden = load_f32(&st, "duration_predictor/duration", &device)?;
    let (max_abs_dur, _) = diff_stats(&duration, &dur_golden)?;
    println!("\nFINAL duration max_abs = {max_abs_dur:.3e}");
    println!("worst per-layer hook  : {worst_name} ({worst:.3e})");
    let ours_s: f32 = duration.to_vec1()?[0];
    let theirs_s: f32 = dur_golden.to_vec1()?[0];
    println!("duration: ours={ours_s:.6}s  gold={theirs_s:.6}s");

    if max_abs_dur < 1e-4 {
        eprintln!("\nPASS: duration within 1e-4 of golden.");
    } else {
        eprintln!("\nFAIL: duration delta {max_abs_dur:.3e} > 1e-4");
        std::process::exit(1);
    }
    Ok(())
}

// Same helpers as vocoder_check; duplicated here to keep each bin standalone.

fn diff_stats(ours: &Tensor, theirs: &Tensor) -> Result<(f32, f32)> {
    let a = ours.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = theirs.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    if a.len() != b.len() {
        bail!("len mismatch: {} vs {}", a.len(), b.len());
    }
    let mut max = 0f32;
    let mut s_dif = 0f64;
    let mut s_abs = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > max { max = d; }
        s_dif += d as f64;
        s_abs += y.abs() as f64;
    }
    Ok((max, if s_abs > 0.0 { (s_dif / s_abs) as f32 } else { 0.0 }))
}

fn load_i64(st: &SafeTensors, key: &str, device: &Device) -> Result<Tensor> {
    let v = st.tensor(key).map_err(|e| anyhow!("{key}: {e}"))?;
    let dims = v.shape();
    let bytes = v.data();
    let arr: Vec<i64> = bytes.chunks_exact(8)
        .map(|c| i64::from_le_bytes([c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7]])).collect();
    Tensor::from_vec(arr, dims.to_vec(), device).context("from_vec")
}

fn load_f32(st: &SafeTensors, key: &str, device: &Device) -> Result<Tensor> {
    let v = st.tensor(key).map_err(|e| anyhow!("{key}: {e}"))?;
    let dims = v.shape();
    let arr = view_to_f32(&v)?;
    Tensor::from_vec(arr, dims.to_vec(), device).context("from_vec")
}

fn try_load_any_shape(st: &SafeTensors, key: &str, device: &Device) -> Result<Tensor> {
    let v = st.tensor(key).map_err(|e| anyhow!("{key}: {e}"))?;
    let dims = v.shape();
    let arr = view_to_f32(&v)?;
    Tensor::from_vec(arr, dims.to_vec(), device).context("from_vec")
}

fn view_to_f32(view: &TensorView) -> Result<Vec<f32>> {
    use safetensors::Dtype as D;
    let n = view.shape().iter().product::<usize>();
    let v: Vec<f32> = match view.dtype() {
        D::F32 => view.data().chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        other => bail!("unsupported dtype {other:?}"),
    };
    if v.len() != n { bail!("decoded {} elems, expected {n}", v.len()); }
    Ok(v)
}
