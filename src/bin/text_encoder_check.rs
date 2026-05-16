//! Per-layer numerical diff: candle text_encoder vs onnxruntime golden.

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use safetensors::{tensor::TensorView, SafeTensors};
use soda_tts::model::candle::{
    text_encoder::Hooks, CandleTextEncoder,
};
use std::fs;
use std::path::PathBuf;

fn hook_map() -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = Vec::new();
    for i in 0..6 {
        for sub in ["dwconv/Conv_output_0",
                    "norm/norm/LayerNormalization_output_0",
                    "pwconv1/Conv_output_0", "pwconv2/Conv_output_0",
                    "Add_output_0"] {
            v.push((
                format!("text_encoder/convnext/convnext.{i}/{sub}"),
                format!("convnext.{i}/{sub}"),
            ));
        }
    }
    for i in 0..4 {
        for sub in ["conv_q/Conv_output_0", "conv_k/Conv_output_0",
                    "conv_v/Conv_output_0", "MatMul_output_0",
                    "MatMul_1_output_0", "Add_2_output_0", "Softmax_output_0",
                    "MatMul_2_output_0", "MatMul_3_output_0", "Add_4_output_0",
                    "conv_o/Conv_output_0"] {
            v.push((
                format!("text_encoder/attn_encoder/attn_layers.{i}/{sub}"),
                format!("attn_layers.{i}/{sub}"),
            ));
        }
        v.push((
            format!("text_encoder/attn_encoder/norm_layers_1.{i}/norm/LayerNormalization_output_0"),
            format!("norm_layers_1.{i}/norm/LayerNormalization_output_0"),
        ));
        v.push((
            format!("text_encoder/attn_encoder/norm_layers_2.{i}/norm/LayerNormalization_output_0"),
            format!("norm_layers_2.{i}/norm/LayerNormalization_output_0"),
        ));
        for sub in ["conv_1/Conv_output_0", "conv_2/Conv_output_0"] {
            v.push((
                format!("text_encoder/attn_encoder/ffn_layers.{i}/{sub}"),
                format!("ffn_layers.{i}/{sub}"),
            ));
        }
    }
    v.push(("text_encoder/Add_output_0".into(), "spte/Add_output_0".into()));
    v.push(("text_encoder/Add_1_output_0".into(), "spte/Add_1_output_0".into()));
    v.push(("text_encoder/norm/norm/LayerNormalization_output_0".into(),
            "spte/norm/output".into()));
    v.push(("text_encoder/text_emb".into(), "spte/final_text_emb".into()));
    // spte cross-attention hooks. Golden's layout for the inner MatMul
    // outputs is [H, B, T, ...] (heads-first); we record ours in the
    // [B, T, ...] post-projection form so we cross-check at the
    // projection boundary instead.
    for i in 1..=2usize {
        v.push((
            format!("text_encoder/attention{i}/W_query/linear/Add_output_0"),
            format!("spte/attention{i}/W_query"),
        ));
        v.push((
            format!("text_encoder/attention{i}/W_key/linear/Add_output_0"),
            format!("spte/attention{i}/W_key"),
        ));
        v.push((
            format!("text_encoder/attention{i}/W_value/linear/Add_output_0"),
            format!("spte/attention{i}/W_value"),
        ));
        v.push((
            format!("text_encoder/attention{i}/out_fc/linear/Add_output_0"),
            format!("spte/attention{i}/out_fc"),
        ));
    }
    v
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let model_st = PathBuf::from("models/safetensors/text_encoder.safetensors");
    let golden = PathBuf::from("tmp/golden.safetensors");

    eprintln!("loading candle text_encoder from {} ...", model_st.display());
    let te = CandleTextEncoder::load(&model_st, &device).context("load")?;

    eprintln!("mmap golden activations from {} ...", golden.display());
    let bytes = fs::read(&golden)?;
    let st = SafeTensors::deserialize(&bytes)?;

    let text_ids = load_i64(&st, "input/text_ids", &device)?;
    let text_mask = load_f32(&st, "input/text_mask", &device)?;
    let style_ttl = load_f32(&st, "input/style_ttl", &device)?;
    eprintln!("text_ids {:?}, text_mask {:?}, style_ttl {:?}",
        text_ids.dims(), text_mask.dims(), style_ttl.dims());

    let mut hooks = Hooks::default();
    let t0 = std::time::Instant::now();
    let text_emb = te.forward_with_hooks(&text_ids, &style_ttl, &text_mask, &mut hooks)?;
    eprintln!("candle te forward in {:.3}s -> text_emb {:?}",
        t0.elapsed().as_secs_f64(), text_emb.dims());

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
        let (m, r) = diff_stats(ours, &theirs)?;
        if m > worst { worst = m; worst_name = hook_name.into(); }
        println!("  {:>70}  max_abs={:.3e}  rel={:.3e}  ours{:?}/gold{:?}",
            hook_name, m, r, ours.dims(), theirs.dims());
    }

    let te_golden = load_f32(&st, "text_encoder/text_emb", &device)?;
    let (max_abs_emb, _) = diff_stats(&text_emb, &te_golden)?;
    println!("\nFINAL text_emb max_abs = {max_abs_emb:.3e}");
    println!("worst per-layer hook  : {worst_name} ({worst:.3e})");

    if max_abs_emb < 1e-3 {
        eprintln!("\nPASS: text_emb within 1e-3 of golden.");
    } else {
        eprintln!("\nFAIL: text_emb delta {max_abs_emb:.3e} > 1e-3");
        std::process::exit(1);
    }
    Ok(())
}

fn diff_stats(a: &Tensor, b: &Tensor) -> Result<(f32, f32)> {
    let av = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let bv = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    if av.len() != bv.len() { bail!("len mismatch {} vs {}", av.len(), bv.len()); }
    let mut max = 0f32; let mut s_dif = 0f64; let mut s_abs = 0f64;
    for (x, y) in av.iter().zip(bv.iter()) {
        let d = (x - y).abs();
        if d > max { max = d; }
        s_dif += d as f64; s_abs += y.abs() as f64;
    }
    Ok((max, if s_abs > 0.0 { (s_dif / s_abs) as f32 } else { 0.0 }))
}

fn load_i64(st: &SafeTensors, key: &str, device: &Device) -> Result<Tensor> {
    let v = st.tensor(key).map_err(|e| anyhow!("{key}: {e}"))?;
    let dims = v.shape();
    let arr: Vec<i64> = v.data().chunks_exact(8)
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
