//! Phase 5 check binary: vector_estimator candle vs ort golden.
//!
//! Run after `tools/dump_golden.py` regenerates tmp/golden.safetensors.
//! Loads the candle VE, executes one denoising step (step 0 of 8), and
//! diffs the result against the golden's `vector_estimator/step_00/denoised_latent`.

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use safetensors::{tensor::TensorView, SafeTensors};
use soda_tts::model::candle::CandleVectorEstimator;
use std::fs;
use std::path::PathBuf;

fn main() -> Result<()> {
    let device = Device::Cpu;
    let model_st = PathBuf::from("models/safetensors/vector_estimator.safetensors");
    let golden = PathBuf::from("tmp/golden.safetensors");

    eprintln!("loading candle vector_estimator from {} ...", model_st.display());
    let t_load = std::time::Instant::now();
    let ve = CandleVectorEstimator::load(&model_st, &device).context("load VE")?;
    eprintln!("loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    eprintln!("mmap golden activations from {} ...", golden.display());
    let bytes = fs::read(&golden)?;
    let st = SafeTensors::deserialize(&bytes)?;

    // Inputs for step 0 of the ODE.
    let noisy_latent = load_f32(&st, "input/noisy_latent", &device)?;
    let text_emb = load_f32(&st, "text_encoder/text_emb", &device)
        .or_else(|_| load_f32(&st, "input/text_emb_step0", &device))?;
    let style_ttl = load_f32(&st, "input/style_ttl", &device)?;
    let latent_mask = load_f32(&st, "input/latent_mask", &device)?;
    let text_mask = load_f32(&st, "input/text_mask", &device)?;
    let total_step = Tensor::new(&[8.0f32], &device)?;
    let current_step = Tensor::new(&[0.0f32], &device)?;

    eprintln!(
        "shapes: noisy_latent={:?}, text_emb={:?}, style_ttl={:?}",
        noisy_latent.dims(), text_emb.dims(), style_ttl.dims()
    );

    let t0 = std::time::Instant::now();
    let (v_b2, denoised, hooks) = ve.forward_diagnostic(
        &noisy_latent, &text_emb, &style_ttl,
        &latent_mask, &text_mask,
        &current_step, &total_step,
    )?;
    eprintln!("ve forward in {:.3}s -> v_b2 {:?}, denoised {:?}, {} hooks",
        t0.elapsed().as_secs_f64(), v_b2.dims(), denoised.dims(), hooks.len());

    // Translate hook name -> golden key. Some hooks have a 1-1 ONNX path,
    // others (like text_cond internals) need rewriting.
    fn golden_key(hook: &str) -> String {
        if let Some(rest) = hook.strip_suffix("/text_cond/W_query/Add_output_0") {
            return format!("vector_estimator/step_00/{rest}/attn/W_query/linear/Add_output_0");
        }
        if let Some(rest) = hook.strip_suffix("/text_cond/W_key/Add_output_0") {
            return format!("vector_estimator/step_00/{rest}/attn/W_key/linear/Add_output_0");
        }
        if let Some(rest) = hook.strip_suffix("/text_cond/W_value/Add_output_0") {
            return format!("vector_estimator/step_00/{rest}/attn/W_value/linear/Add_output_0");
        }
        if let Some(rest) = hook.strip_suffix("/text_cond/out_fc/Add_output_0") {
            return format!("vector_estimator/step_00/{rest}/attn/out_fc/linear/Add_output_0");
        }
        if let Some(rest) = hook.strip_suffix("/text_cond/Add_output_0") {
            return format!("vector_estimator/step_00/{rest}/Add_output_0");
        }
        if let Some(rest) = hook.strip_suffix("/text_cond/norm/output") {
            return format!("vector_estimator/step_00/{rest}/norm/norm/LayerNormalization_output_0");
        }
        if let Some(rest) = hook.strip_suffix("/style_cond/W_query/Add_output_0") {
            return format!("vector_estimator/step_00/{rest}/attention/W_query/linear/Add_output_0");
        }
        if let Some(rest) = hook.strip_suffix("/style_cond/W_key/Add_output_0") {
            return format!("vector_estimator/step_00/{rest}/attention/W_key/linear/Add_output_0");
        }
        if let Some(rest) = hook.strip_suffix("/style_cond/W_value/Add_output_0") {
            return format!("vector_estimator/step_00/{rest}/attention/W_value/linear/Add_output_0");
        }
        if let Some(rest) = hook.strip_suffix("/style_cond/out_fc/Add_output_0") {
            return format!("vector_estimator/step_00/{rest}/attention/out_fc/linear/Add_output_0");
        }
        if let Some(rest) = hook.strip_suffix("/style_cond/Add_output_0") {
            return format!("vector_estimator/step_00/{rest}/Add_output_0");
        }
        if let Some(rest) = hook.strip_suffix("/style_cond/norm/output") {
            return format!("vector_estimator/step_00/{rest}/norm/norm/LayerNormalization_output_0");
        }
        format!("vector_estimator/step_00/{hook}")
    }

    println!("\n--- per-hook v_cond divergence ---");
    let mut first_div = None;
    for (name, ours) in &hooks {
        let key = golden_key(name);
        let golden = match load_f32(&st, &key, &device) {
            Ok(g) => g, Err(_) => continue,
        };
        // Compare batch 0 only (= cond).
        let our_c = ours.narrow(0, 0, 1)?;
        let gold_c = if golden.dim(0)? > 1 {
            golden.narrow(0, 0, 1)?
        } else {
            golden.clone()
        };
        let (m, _) = diff_stats(&our_c, &gold_c)?;
        let marker = if m > 1e-3 { " <-- diverged" } else { "" };
        if m > 1e-3 && first_div.is_none() {
            first_div = Some(name.clone());
        }
        println!("  {name:<60}  max_abs={m:.3e}{marker}");
    }
    if let Some(n) = first_div {
        println!("\nFIRST DIVERGENCE: {n}");
    }
    println!();

    // Compare vector field output if we have a golden for it. We need to
    // dump it from the ort side first; for now this key is optional.
    if let Ok(golden_v) = load_f32(&st, "vector_estimator/step_00/proj_out/Mul_output_0", &device) {
        let (max_abs_v, rel_v) = diff_stats(&v_b2, &golden_v)?;
        println!("vector field max_abs={max_abs_v:.3e}  rel={rel_v:.3e}");
        // Per-batch breakdown
        let gv_c = golden_v.narrow(0, 0, 1)?;
        let gv_u = golden_v.narrow(0, 1, 1)?;
        let ov_c = v_b2.narrow(0, 0, 1)?;
        let ov_u = v_b2.narrow(0, 1, 1)?;
        let (m_c, _) = diff_stats(&ov_c, &gv_c)?;
        let (m_u, _) = diff_stats(&ov_u, &gv_u)?;
        println!("  v_cond max_abs   = {m_c:.3e}");
        println!("  v_uncond max_abs = {m_u:.3e}");
    } else {
        eprintln!("(skipping vector-field diff: golden key not present)");
    }
    // Also check proj_in output
    if let Ok(golden_pi) = load_f32(&st, "vector_estimator/step_00/proj_in/Mul_output_0", &device) {
        // We don't have a candle hook for proj_in; would need a separate path.
        eprintln!("(golden proj_in available, shape={:?}; add candle hook to compare)",
            golden_pi.dims());
    }

    let golden_denoised = load_f32(&st, "vector_estimator/step_00/denoised_latent", &device)?;
    let (max_abs, rel) = diff_stats(&denoised, &golden_denoised)?;
    println!("FINAL denoised_latent max_abs={max_abs:.3e}  rel={rel:.3e}");
    println!("ours mean/std: {:.4}/{:.4}",
        mean(&denoised)?, std(&denoised)?);
    println!("gold mean/std: {:.4}/{:.4}",
        mean(&golden_denoised)?, std(&golden_denoised)?);
    // Also print v_b2 stats for diagnostic
    let v_cond = v_b2.narrow(0, 0, 1)?;
    let v_uncond = v_b2.narrow(0, 1, 1)?;
    println!("ours v_cond   mean/std: {:.4}/{:.4}", mean(&v_cond)?, std(&v_cond)?);
    println!("ours v_uncond mean/std: {:.4}/{:.4}", mean(&v_uncond)?, std(&v_uncond)?);

    if max_abs < 1e-3 {
        eprintln!("\nPASS: denoised within 1e-3 of golden.");
    } else {
        eprintln!("\nFAIL: denoised delta {max_abs:.3e} > 1e-3");
        std::process::exit(1);
    }
    Ok(())
}

fn mean(t: &Tensor) -> Result<f32> {
    let v = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(v.iter().sum::<f32>() / v.len() as f32)
}

fn std(t: &Tensor) -> Result<f32> {
    let v = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let m = v.iter().sum::<f32>() / v.len() as f32;
    let var: f32 = v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len() as f32;
    Ok(var.sqrt())
}

fn diff_stats(a: &Tensor, b: &Tensor) -> Result<(f32, f32)> {
    let av = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let bv = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    if av.len() != bv.len() {
        bail!("len mismatch {} vs {}", av.len(), bv.len());
    }
    let mut max = 0f32; let mut s_dif = 0f64; let mut s_abs = 0f64;
    for (x, y) in av.iter().zip(bv.iter()) {
        let d = (x - y).abs();
        if d > max { max = d; }
        s_dif += d as f64; s_abs += y.abs() as f64;
    }
    Ok((max, if s_abs > 0.0 { (s_dif / s_abs) as f32 } else { 0.0 }))
}

fn load_f32(st: &SafeTensors, key: &str, device: &Device) -> Result<Tensor> {
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
