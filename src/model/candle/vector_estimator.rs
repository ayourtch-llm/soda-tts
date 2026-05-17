//! Candle port of the Supertonic 3 vector_estimator (flow-matching ODE).
//!
//! This is the biggest of the four models -- 1004 ops, 315 weight
//! tensors -- and the only one with rotary text cross-attention plus a
//! bundled ODE Euler step in the graph itself.
//!
//! High-level pipeline (from ONNX walk + tts.json):
//!
//! ```text
//! noisy_latent [B, 144, T_lat]   text_emb [B, 256, T_text]
//! style_ttl    [B, 50, 256]      latent_mask [B, 1, T_lat]
//! text_mask    [B, 1, T_text]    current_step [B]   total_step [B]
//!
//!   t = current_step / total_step               # scalar per batch
//!   t_emb = sinusoidal(t) -> Linear(64,256) -> Mish -> Linear(256,64)  [B, 64, 1]
//!
//!   h = proj_in: Conv1d(144 -> 512, k=1) (no bias) -> * latent_mask  [B, 512, T_lat]
//!
//!   for block_idx in 0..4:                     # 4 main_blocks
//!     # convnext_0: 4 dilated layers (dilations 1, 2, 4, 8)
//!     h = convnext_stack(h, dilations=[1,2,4,8])
//!     # time_cond: project t_emb to 512 and add
//!     h = h + transpose(Linear(t_emb, 64->512))
//!     h = h * latent_mask
//!     # convnext_1: 1 layer
//!     h = convnext_stack(h, dilations=[1])
//!     # text_cond: cross-attn with RoPE on Q/K
//!     h = h + text_cross_attn_rope(h, text_emb)  [residual]
//!     # convnext_2: 1 layer
//!     h = convnext_stack(h, dilations=[1])
//!     # style_cond: cross-attn (Q @ tanh(K^T), like spte) + post-norm
//!     h = post_norm(h + style_cross_attn(h, style_ttl))
//!
//!   h = last_convnext_stack(h, dilations=[1,1,1,1])
//!   v = proj_out: Conv1d(512 -> 144, k=1)(h) * latent_mask   # predicted velocity
//!
//!   # Bundled ODE step (rectified-flow Euler):
//!   denoised_latent = noisy_latent * a + v * b
//!   (a, b derived from current_step, total_step -- see ode_step_coeffs)
//! ```

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, VarBuilder};
use std::path::Path;

const HIDDEN: usize = 512; // main backbone hidden dim
const LATENT_DIM: usize = 144; // 24 * 6 (latent_dim * chunk_compress_factor)
const TIME_DIM: usize = 64;
const TIME_HDIM: usize = 256;
const TEXT_DIM: usize = 256;
const STYLE_DIM: usize = 256;
const N_STYLE: usize = 50;
const N_MAIN_BLOCKS: usize = 4;
const LN_EPS: f64 = 1e-6;
// Text cross-attention (RoPE)
const TEXT_N_HEADS: usize = 8;
const TEXT_HEAD_DIM: usize = HIDDEN / TEXT_N_HEADS; // 64
const TEXT_ROPE_BASE: f64 = 10000.0;
const TEXT_ROPE_SCALE: f64 = 10.0;
// Style cross-attention (Q@tanh(K^T))
const STYLE_N_HEADS: usize = 2;
const STYLE_INNER_DIM: usize = 256; // Q, K, V projected to 256 (not 512)
const STYLE_HEAD_DIM: usize = STYLE_INNER_DIM / STYLE_N_HEADS; // 128
// ConvNeXt
const CONVNEXT_KSZ: usize = 5;
const CONVNEXT_INTERMEDIATE: usize = 2048;
// Per-main_block dilations
const CONVNEXT_0_DILATIONS: [usize; 4] = [1, 2, 4, 8];
const CONVNEXT_1_DILATIONS: [usize; 1] = [1];
const CONVNEXT_2_DILATIONS: [usize; 1] = [1];
const LAST_CONVNEXT_DILATIONS: [usize; 4] = [1, 1, 1, 1];

// ============================ ConvNeXt block ===============================
// Identical structure to text_encoder's ConvNextBlock but at 512 channels,
// 2048 intermediate, with arbitrary dilation.

struct ConvNextBlock {
    dwconv: Conv1d,
    norm: LayerNorm,
    pwconv1: Conv1d,
    pwconv2: Conv1d,
    gamma: Tensor,
    dilation: usize,
}

impl ConvNextBlock {
    fn load(vb: VarBuilder, dilation: usize) -> Result<Self> {
        let dwconv = candle_nn::conv1d(HIDDEN, HIDDEN, CONVNEXT_KSZ,
            Conv1dConfig { padding: 0, stride: 1, dilation, groups: HIDDEN,
                ..Default::default() },
            vb.pp("dwconv"),
        ).context("dwconv")?;
        let norm = candle_nn::layer_norm(HIDDEN, LN_EPS, vb.pp("norm.norm"))
            .context("convnext norm")?;
        let pwconv1 = candle_nn::conv1d(HIDDEN, CONVNEXT_INTERMEDIATE, 1,
            Conv1dConfig { padding: 0, stride: 1, dilation: 1, groups: 1,
                ..Default::default() },
            vb.pp("pwconv1"),
        ).context("pwconv1")?;
        let pwconv2 = candle_nn::conv1d(CONVNEXT_INTERMEDIATE, HIDDEN, 1,
            Conv1dConfig { padding: 0, stride: 1, dilation: 1, groups: 1,
                ..Default::default() },
            vb.pp("pwconv2"),
        ).context("pwconv2")?;
        let gamma = vb.get((1, HIDDEN, 1), "gamma").context("gamma")?;
        Ok(Self { dwconv, norm, pwconv1, pwconv2, gamma, dilation })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> candle_core::Result<Tensor> {
        let total = self.dilation * (CONVNEXT_KSZ - 1);
        let l = total / 2;
        let r = total - l;
        let h = pad_edge(x, l, r)?;
        let h = self.dwconv.forward(&h)?.broadcast_mul(mask)?;
        let h_nhwc = h.transpose(1, 2)?.contiguous()?;
        let h_ln = self.norm.forward(&h_nhwc)?;
        let h = h_ln.transpose(1, 2)?.contiguous()?;
        let h = self.pwconv1.forward(&h)?;
        let h = gelu_erf(&h)?;
        let h = self.pwconv2.forward(&h)?;
        let h = self.gamma.broadcast_mul(&h)?;
        let out = (x + h)?;
        out.broadcast_mul(mask)
    }
}

// ============================ Time encoder =================================
// sinusoidal(t) -> Linear(64,256) -> Mish -> Linear(256,64) -> [B, 64, 1]

struct TimeEncoder {
    /// pre-scale factor for `t` -- loaded from
    /// `time_encoder/sinusoidal/Constant_2_output_0` (= 1000.0).
    t_scale: f32,
    /// inverse-frequency table loaded from
    /// `time_encoder/sinusoidal/Constant_3_output_0` shape [1, 32].
    /// Geometric with ratio ~0.7430 (NOT 1/10000^(2i/64) like canonical RoPE).
    inv_freq: Tensor,
    mlp0: candle_nn::Linear, // 64 -> 256
    mlp2: candle_nn::Linear, // 256 -> 64
}

impl TimeEncoder {
    fn load(vb: VarBuilder, root_vb: &VarBuilder) -> Result<Self> {
        // Both constants are stored at the root (no `tts.*` prefix) because
        // ONNX named them as graph-internal `/...Constant_*_output_0` paths.
        let t_scale_t = root_vb.get(
            (), "/vector_estimator/vector_field/time_encoder/sinusoidal/Constant_2_output_0",
        ).context("time_encoder Constant_2 (t_scale)")?;
        let t_scale = t_scale_t.to_dtype(DType::F32)?.to_scalar::<f32>()?;
        let inv_freq = root_vb.get(
            (1, TIME_DIM / 2),
            "/vector_estimator/vector_field/time_encoder/sinusoidal/Constant_3_output_0",
        ).context("time_encoder Constant_3 (inv_freq)")?;

        let mlp0 = candle_nn::linear(TIME_DIM, TIME_HDIM, vb.pp("mlp.0.linear"))
            .context("time mlp.0")?;
        let mlp2 = candle_nn::linear(TIME_HDIM, TIME_DIM, vb.pp("mlp.2.linear"))
            .context("time mlp.2")?;
        Ok(Self { t_scale, inv_freq, mlp0, mlp2 })
    }

    /// `t`: [B] f32 (= current_step / total_step). Returns [B, 64, 1].
    fn forward(&self, t: &Tensor) -> candle_core::Result<Tensor> {
        let b = t.dim(0)?;
        // ONNX: phase = (t * 1000) * inv_freq. Without the 1000x prescale
        // the sinusoidal embedding for t=1/8 happens to be visually
        // similar to the right answer but accumulates error per step.
        let t_col = t.reshape((b, 1))?.affine(self.t_scale as f64, 0.0)?;
        let phase = t_col.broadcast_mul(&self.inv_freq)?;
        let sin = phase.sin()?;
        let cos = phase.cos()?;
        let emb = Tensor::cat(&[sin, cos], D::Minus1)?; // [B, 64]
        let h = self.mlp0.forward(&emb)?;
        let h = mish(&h)?;
        let h = self.mlp2.forward(&h)?;
        h.reshape((b, TIME_DIM, 1))
    }
}

fn mish(x: &Tensor) -> candle_core::Result<Tensor> {
    // mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    let sp = softplus(x)?;
    let t = sp.tanh()?;
    x * t
}

fn softplus(x: &Tensor) -> candle_core::Result<Tensor> {
    // softplus(x) = ln(1 + exp(x)) = log1p(exp(x)). Use affine(0, 1)
    // for the +1 so it broadcasts cleanly over any input shape.
    let e = x.exp()?;
    e.affine(1.0, 1.0)?.log()
}

// ============================ time_cond layer ==============================
// Linear(64 -> 512) applied to time_emb, added (broadcast over T) to features.
// Weight is stored as onnx::MatMul_<id> (in shape [64, 512] = [in, out]).
// Bias is stored as `linear.linear.bias` [512].

struct TimeCondLayer {
    weight: Tensor, // [64, 512] - kept as-is for X @ W matmul
    bias: Tensor,   // [512]
}

impl TimeCondLayer {
    fn load(vb_block: VarBuilder, root_vb: &VarBuilder, weight_name: &str) -> Result<Self> {
        let weight = root_vb.get((TIME_DIM, HIDDEN), weight_name)
            .with_context(|| format!("time_cond weight {weight_name}"))?;
        let bias = vb_block.pp("linear").pp("linear").get(HIDDEN, "bias")
            .context("time_cond bias")?;
        Ok(Self { weight, bias })
    }

    /// `x`: [B, 512, T_lat]   `t_emb`: [B, 64, 1]   `mask`: [B, 1, T_lat]
    fn forward(&self, x: &Tensor, t_emb: &Tensor, mask: &Tensor)
        -> candle_core::Result<Tensor>
    {
        // ONNX: Transpose t_emb [B, 64, 1] -> [B, 1, 64], MatMul -> [B, 1, 512],
        //       Add bias, Transpose back -> [B, 512, 1]. Then Add (broadcast over
        //       T) into x. Then Mul mask.
        let t_t = t_emb.transpose(1, 2)?.contiguous()?;
        let proj = t_t.broadcast_matmul(&self.weight)?
            .broadcast_add(&self.bias)?;
        let proj_t = proj.transpose(1, 2)?.contiguous()?; // [B, 512, 1]
        let out = x.broadcast_add(&proj_t)?;
        out.broadcast_mul(mask)
    }
}

// ============================ Text cross-attention with RoPE ==============
//
// Q, K, V all 512-dim (n_heads=8, head_dim=64). RoPE applied to Q and K via
// precomputed `attn.theta` (per-channel half head_dim) and `attn.increments`
// (per-position indices). The model uses TWO different position vectors:
// `Sin/Cos` for Q (latent positions) and `Sin_1/Cos_1` for K (text positions).
//
// After RoPE, standard scaled-dot-product attention, then out_fc, then mask,
// then residual + post-norm.

struct TextCondLayer {
    w_q: candle_nn::Linear,
    w_k: candle_nn::Linear,
    w_v: candle_nn::Linear,
    out_fc: candle_nn::Linear,
    theta: Tensor,       // [1, 1, 32] -- already includes rotary_scale factor
    norm: LayerNorm,
}

impl TextCondLayer {
    /// `shared_theta` is loaded once at the top (stored only under
    /// main_blocks.3.attn.theta but reused across all 4 text_cond layers).
    /// Positions are computed at forward time via arange.
    fn load(
        vb_block: VarBuilder, root_vb: &VarBuilder, weight_ids: [u32; 4],
        shared_theta: Tensor,
    ) -> Result<Self> {
        let attn = vb_block.pp("attn");
        let make = |w: Tensor, bn: &str, in_dim: usize, out_dim: usize|
            -> Result<candle_nn::Linear>
        {
            // ONNX matmul: input @ W where W is [in, out]. candle Linear stores
            // [out, in] then does input @ W^T.
            let wt = w.t()?.contiguous()?;
            assert_eq!(wt.dims(), &[out_dim, in_dim],
                "{bn}: expected weight [{out_dim}, {in_dim}], got {:?}", wt.dims());
            let b = attn.pp(bn).pp("linear").get(out_dim, "bias")
                .with_context(|| format!("{bn}.bias"))?;
            Ok(candle_nn::Linear::new(wt, Some(b)))
        };
        // Per the empirical mapping:
        //   W_query: [512, 512]  (Q from 512-d latent stream)
        //   W_key:   [256, 512]  (K from 256-d text stream, projected up)
        //   W_value: [256, 512]  (V same)
        //   out_fc:  [512, 512]
        let w_q_raw = root_vb.get((HIDDEN, HIDDEN),
            &format!("onnx::MatMul_{}", weight_ids[0]))?;
        let w_k_raw = root_vb.get((TEXT_DIM, HIDDEN),
            &format!("onnx::MatMul_{}", weight_ids[1]))?;
        let w_v_raw = root_vb.get((TEXT_DIM, HIDDEN),
            &format!("onnx::MatMul_{}", weight_ids[2]))?;
        let out_raw = root_vb.get((HIDDEN, HIDDEN),
            &format!("onnx::MatMul_{}", weight_ids[3]))?;
        let w_q = make(w_q_raw, "W_query", HIDDEN, HIDDEN)?;
        let w_k = make(w_k_raw, "W_key", TEXT_DIM, HIDDEN)?;
        let w_v = make(w_v_raw, "W_value", TEXT_DIM, HIDDEN)?;
        let out_fc = make(out_raw, "out_fc", HIDDEN, HIDDEN)?;
        let norm = candle_nn::layer_norm(HIDDEN, LN_EPS, vb_block.pp("norm.norm"))
            .context("text_cond norm")?;
        Ok(Self {
            w_q, w_k, w_v, out_fc,
            theta: shared_theta,
            norm,
        })
    }

    /// `x`: [B, 512, T_lat]   `text`: [B, 256, T_text]
    /// `latent_mask`: [B, 1, T_lat]   `text_mask`: [B, 1, T_text]
    /// Returns [B, 512, T_lat] (post-norm, residual already added internally).
    fn forward(
        &self, x: &Tensor, text: &Tensor,
        latent_mask: &Tensor, text_mask: &Tensor,
    ) -> candle_core::Result<Tensor> {
        self.forward_hooked(x, text, latent_mask, text_mask, None)
    }

    fn forward_hooked(
        &self, x: &Tensor, text: &Tensor,
        latent_mask: &Tensor, text_mask: &Tensor,
        mut hooks: Option<&mut Vec<(String, Tensor)>>,
    ) -> candle_core::Result<Tensor> {
        let (b, _, t_lat) = x.dims3()?;
        let (_, _, t_text) = text.dims3()?;

        // Project Q from x (latent stream), K and V from text.
        let x_t = x.transpose(1, 2)?.contiguous()?;       // [B, T_lat, 512]
        let text_t = text.transpose(1, 2)?.contiguous()?; // [B, T_text, 256]

        // W_q maps 512 -> 512, but the underlying weight from onnx::MatMul is
        // [512, 512]. W_k and W_v have weights [256, 512] (in=256, out=512)
        // because they project the text stream up to the 512 backbone dim.
        // (We loaded them all as [HIDDEN, HIDDEN] above, which is WRONG for
        // K/V -- need to fix to handle the asymmetric shapes.)
        let q = self.w_q.forward(&x_t)?;        // [B, T_lat, 512]
        let k = self.w_k.forward(&text_t)?;     // [B, T_text, 512]
        let v = self.w_v.forward(&text_t)?;     // [B, T_text, 512]
        // Hooks for diagnostic comparison vs golden.
        if let Some(h) = hooks.as_deref_mut() {
            h.push(("text_cond/W_query/Add_output_0".into(), q.clone()));
            h.push(("text_cond/W_key/Add_output_0".into(), k.clone()));
            h.push(("text_cond/W_value/Add_output_0".into(), v.clone()));
        }

        // Split into heads: [B, T, 512] -> [B, T, 8, 64] -> [B, 8, T, 64]
        let split = |t_: &Tensor, seq: usize| -> candle_core::Result<Tensor> {
            t_.reshape((b, seq, TEXT_N_HEADS, TEXT_HEAD_DIM))?
              .transpose(1, 2)?.contiguous()
        };
        let q = split(&q, t_lat)?;
        let k = split(&k, t_text)?;
        let v = split(&v, t_text)?;

        // Apply RoPE. The angle for position p and channel k is:
        //   angle(p, k) = (p / sequence_length) * theta[k]
        // where sequence_length is the count of valid mask positions
        // (= ReduceSum(mask) along all axes except batch). The position
        // is normalized to [0, 1) before being scaled by theta. This is
        // different from canonical RoPE (which uses `angle = p * inv_freq`)
        // and from a static rotary_scale divide.
        //
        // For Q we use latent positions and latent_mask; for K, text.
        // We assume B=1 so the scalar mask sum is shared across batch.
        let lat_len = sum_mask(latent_mask)?;
        let text_len = sum_mask(text_mask)?;
        let q_rot = rope_apply_norm(&q, &self.theta, t_lat, lat_len)?;
        let k_rot = rope_apply_norm(&k, &self.theta, t_text, text_len)?;

        // Scaled dot-product attention. The text_cond scale is sqrt(text_dim)=16
        // (NOT sqrt(head_dim)=8 nor sqrt(n_units)=22.6). Empirically confirmed
        // via the ONNX /Div_4 constant = 16.0. The uncond batch happens to be
        // insensitive to this scale (constant K -> uniform attention), so the
        // bug only shows up on the cond path.
        let scale = (TEXT_DIM as f64).sqrt().recip();
        let kt = k_rot.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let scores = q_rot.matmul(&kt)?;
        let scores = (scores * scale)?;

        // Mask: where text_mask==0 use -1e4.
        let m_q = latent_mask.reshape((b, t_lat, 1))?;
        let m_k = text_mask.reshape((b, 1, t_text))?;
        let attn_mask = m_q.broadcast_matmul(&m_k)?.reshape((b, 1, t_lat, t_text))?;
        let mb = attn_mask.broadcast_as(scores.shape())?;
        let bias = mb.affine(-1.0, 1.0)?.affine(-1e4, 0.0)?;
        let scores = (scores + bias)?;
        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;

        let out = attn.matmul(&v)?; // [B, 8, T_lat, 64]
        // Merge heads: [B, 8, T_lat, 64] -> [B, T_lat, 512]
        let merged = out.transpose(1, 2)?.contiguous()?
            .reshape((b, t_lat, HIDDEN))?;
        let proj = self.out_fc.forward(&merged)?; // [B, T_lat, 512]
        // Transpose back to [B, 512, T_lat] and mask.
        let proj_back = proj.transpose(1, 2)?.contiguous()?;
        let proj_masked = proj_back.broadcast_mul(latent_mask)?;

        // Residual + post-norm.
        let h = (x + proj_masked)?;
        if let Some(h_hooks) = hooks.as_deref_mut() {
            h_hooks.push(("text_cond/Add_output_0".into(), h.clone()));
            h_hooks.push(("text_cond/out_fc/Add_output_0".into(), proj.clone()));
        }
        let h_t = h.transpose(1, 2)?.contiguous()?;
        let h_norm = self.norm.forward(&h_t)?;
        if let Some(h_hooks) = hooks.as_deref_mut() {
            h_hooks.push(("text_cond/norm/output".into(), h_norm.clone()));
        }
        let out = h_norm.transpose(1, 2)?.contiguous()?;
        out.broadcast_mul(latent_mask)
    }
}

/// Apply length-normalized rotary position embeddings to `x` shape
/// [B, H, T, D]. Angle = (position / seq_len) * theta.
fn rope_apply_norm(x: &Tensor, theta: &Tensor, t: usize, seq_len: f32)
    -> candle_core::Result<Tensor>
{
    let (b, h, _, d) = x.dims4()?;
    let pos = Tensor::arange(0u32, t as u32, x.device())?
        .to_dtype(x.dtype())?
        .reshape((1, t, 1))?;
    let pos_norm = pos.affine(1.0 / seq_len as f64, 0.0)?;
    let angles = pos_norm.broadcast_mul(theta)?; // [1, T, D/2]
    let sin = angles.sin()?;
    let cos = angles.cos()?;
    // Broadcast sin/cos to [B, H, T, D/2].
    let sin_b = sin.unsqueeze(0)?.broadcast_as((b, h, t, d / 2))?.contiguous()?;
    let cos_b = cos.unsqueeze(0)?.broadcast_as((b, h, t, d / 2))?.contiguous()?;
    // Split x into [x_even, x_odd] pairs along the head dim.
    let x_even = x.narrow(D::Minus1, 0, d / 2)?;
    let x_odd = x.narrow(D::Minus1, d / 2, d / 2)?;
    // rotate_half: (x_even, x_odd) -> (-x_odd, x_even)
    let neg_odd = x_odd.neg()?;
    // rotated = x * cos + rotate_half(x) * sin
    let rot_even = (x_even.broadcast_mul(&cos_b)? + neg_odd.broadcast_mul(&sin_b)?)?;
    let rot_odd = (x_odd.broadcast_mul(&cos_b)? + x_even.broadcast_mul(&sin_b)?)?;
    Tensor::cat(&[rot_even, rot_odd], D::Minus1)
}

// ============================ Style cross-attention ========================
//
// Same Q@tanh(K^T) pattern as the text_encoder's spte, but bigger out_fc:
// Q/K/V project to 256 (n_heads=2, head_dim=128); out_fc projects 256 -> 512.

struct StyleCondLayer {
    w_q: candle_nn::Linear,
    w_k: candle_nn::Linear,
    w_v: candle_nn::Linear,
    out_fc: candle_nn::Linear,
    norm: LayerNorm,
}

impl StyleCondLayer {
    fn load(vb_block: VarBuilder, root_vb: &VarBuilder, weight_ids: [u32; 4]) -> Result<Self> {
        let attn = vb_block.pp("attention");
        let make = |w: Tensor, bn: &str, in_dim: usize, out_dim: usize|
            -> Result<candle_nn::Linear>
        {
            let wt = w.t()?.contiguous()?;
            let b = attn.pp(bn).pp("linear").get(out_dim, "bias")
                .with_context(|| format!("{bn}.bias"))?;
            // Sanity check shapes.
            assert_eq!(wt.dims(), &[out_dim, in_dim]);
            Ok(candle_nn::Linear::new(wt, Some(b)))
        };
        // Per-weight shape from inventory:
        //   onnx::MatMul_3407 etc = [256, 256] or [256, 512] or [512, 256]
        // We tag in/out dims explicitly.
        let w_q_raw = root_vb.get((HIDDEN, STYLE_INNER_DIM), &format!("onnx::MatMul_{}", weight_ids[0]))
            .with_context(|| format!("style W_query weight {}", weight_ids[0]))?;
        let w_k_raw = root_vb.get((STYLE_DIM, STYLE_INNER_DIM), &format!("onnx::MatMul_{}", weight_ids[1]))
            .with_context(|| format!("style W_key weight {}", weight_ids[1]))?;
        let w_v_raw = root_vb.get((STYLE_DIM, STYLE_INNER_DIM), &format!("onnx::MatMul_{}", weight_ids[2]))
            .with_context(|| format!("style W_value weight {}", weight_ids[2]))?;
        let out_raw = root_vb.get((STYLE_INNER_DIM, HIDDEN), &format!("onnx::MatMul_{}", weight_ids[3]))
            .with_context(|| format!("style out_fc weight {}", weight_ids[3]))?;
        let w_q = make(w_q_raw, "W_query", HIDDEN, STYLE_INNER_DIM)?;
        let w_k = make(w_k_raw, "W_key", STYLE_DIM, STYLE_INNER_DIM)?;
        let w_v = make(w_v_raw, "W_value", STYLE_DIM, STYLE_INNER_DIM)?;
        let out_fc = make(out_raw, "out_fc", STYLE_INNER_DIM, HIDDEN)?;
        let norm = candle_nn::layer_norm(HIDDEN, LN_EPS, vb_block.pp("norm.norm"))
            .context("style_cond norm")?;
        Ok(Self { w_q, w_k, w_v, out_fc, norm })
    }

    /// `x`: [B, 512, T_lat]    `style_k`: [B, 50, 256]    `style_v`: [B, 50, 256]
    /// `mask`: [B, 1, T_lat]
    ///
    /// K and V take separate inputs because the CFG batch construction
    /// feeds different `style_*_special_token`s to the K and V slots in
    /// the unconditional half of the batch.
    fn forward(&self, x: &Tensor, style_k: &Tensor, style_v: &Tensor, mask: &Tensor)
        -> candle_core::Result<Tensor>
    {
        self.forward_hooked(x, style_k, style_v, mask, None)
    }

    fn forward_hooked(
        &self, x: &Tensor, style_k: &Tensor, style_v: &Tensor, mask: &Tensor,
        mut hooks: Option<&mut Vec<(String, Tensor)>>,
    ) -> candle_core::Result<Tensor> {
        let (b, _, t_lat) = x.dims3()?;
        let q_in = x.transpose(1, 2)?.contiguous()?; // [B, T_lat, 512]
        let q = self.w_q.forward(&q_in)?;            // [B, T_lat, 256]
        let k = self.w_k.forward(style_k)?;          // [B, 50, 256]
        let v = self.w_v.forward(style_v)?;          // [B, 50, 256]
        if let Some(h) = hooks.as_deref_mut() {
            h.push(("style_cond/W_query/Add_output_0".into(), q.clone()));
            h.push(("style_cond/W_key/Add_output_0".into(), k.clone()));
            h.push(("style_cond/W_value/Add_output_0".into(), v.clone()));
        }

        let split = |t_: &Tensor, seq: usize| -> candle_core::Result<Tensor> {
            t_.reshape((b, seq, STYLE_N_HEADS, STYLE_HEAD_DIM))?
              .transpose(1, 2)?.contiguous()
        };
        let q = split(&q, t_lat)?;
        let k = split(&k, N_STYLE)?;
        let v = split(&v, N_STYLE)?;

        // Q @ tanh(K^T) / sqrt(STYLE_INNER_DIM)
        let scale = (STYLE_INNER_DIM as f64).sqrt().recip();
        let kt = k.transpose(D::Minus2, D::Minus1)?.contiguous()?.tanh()?;
        let scores = q.matmul(&kt)?;
        let scores = (scores * scale)?;
        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let out = attn.matmul(&v)?; // [B, H, T_lat, 128]
        let merged = out.transpose(1, 2)?.contiguous()?
            .reshape((b, t_lat, STYLE_INNER_DIM))?;
        let proj = self.out_fc.forward(&merged)?; // [B, T_lat, 512]
        if let Some(h) = hooks.as_deref_mut() {
            h.push(("style_cond/out_fc/Add_output_0".into(), proj.clone()));
        }
        let proj_back = proj.transpose(1, 2)?.contiguous()?;
        let proj_masked = proj_back.broadcast_mul(mask)?;
        let h = (x + proj_masked)?;
        if let Some(hk) = hooks.as_deref_mut() {
            hk.push(("style_cond/Add_output_0".into(), h.clone()));
        }
        // Post-norm
        let h_t = h.transpose(1, 2)?.contiguous()?;
        let h_norm = self.norm.forward(&h_t)?;
        if let Some(hk) = hooks.as_deref_mut() {
            hk.push(("style_cond/norm/output".into(), h_norm.clone()));
        }
        let out = h_norm.transpose(1, 2)?.contiguous()?;
        out.broadcast_mul(mask)
    }
}

// ============================ One MainBlock ===============================

struct MainBlock {
    convnext_0: Vec<ConvNextBlock>,
    time_cond: TimeCondLayer,
    convnext_1: Vec<ConvNextBlock>,
    text_cond: TextCondLayer,
    convnext_2: Vec<ConvNextBlock>,
    style_cond: StyleCondLayer,
}

impl MainBlock {
    /// `vb_root`: pointing at vector_estimator.tts.ttl.vector_field
    /// `block_idx`: 0..4 (the logical main_block index)
    fn load(vb_root: VarBuilder, root_vb: &VarBuilder, block_idx: usize,
            time_cond_w: &str, text_cond_ids: [u32; 4], style_cond_ids: [u32; 4],
            shared_theta: Tensor)
        -> Result<Self>
    {
        let mb = vb_root.pp("main_blocks");
        // The PyTorch n_blocks=4 unrolled to 24 flat sub-modules in ONNX.
        // sub_idx = block_idx * 6 + offset (offset 0..5).
        let base = block_idx * 6;
        let load_convnext_stack = |start: usize, dilations: &[usize]|
            -> Result<Vec<ConvNextBlock>>
        {
            let block_root = mb.pp(&start.to_string());
            let mut out = Vec::with_capacity(dilations.len());
            for (i, &d) in dilations.iter().enumerate() {
                out.push(ConvNextBlock::load(
                    block_root.pp("convnext").pp(&i.to_string()), d
                ).with_context(|| format!("block {start} convnext.{i}"))?);
            }
            Ok(out)
        };
        let convnext_0 = load_convnext_stack(base, &CONVNEXT_0_DILATIONS)
            .with_context(|| format!("block {block_idx} convnext_0"))?;
        let time_cond = TimeCondLayer::load(
            mb.pp(&(base + 1).to_string()), root_vb, time_cond_w,
        )?;
        let convnext_1 = load_convnext_stack(base + 2, &CONVNEXT_1_DILATIONS)
            .with_context(|| format!("block {block_idx} convnext_1"))?;
        let text_cond = TextCondLayer::load(
            mb.pp(&(base + 3).to_string()), root_vb, text_cond_ids,
            shared_theta,
        )?;
        let convnext_2 = load_convnext_stack(base + 4, &CONVNEXT_2_DILATIONS)
            .with_context(|| format!("block {block_idx} convnext_2"))?;
        let style_cond = StyleCondLayer::load(
            mb.pp(&(base + 5).to_string()), root_vb, style_cond_ids,
        )?;
        Ok(Self { convnext_0, time_cond, convnext_1, text_cond, convnext_2, style_cond })
    }

    fn forward(
        &self, x: &Tensor, t_emb: &Tensor,
        text: &Tensor, style_k: &Tensor, style_v: &Tensor,
        latent_mask: &Tensor, text_mask: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let mut h = x.clone();
        for b in &self.convnext_0 { h = b.forward(&h, latent_mask)?; }
        h = self.time_cond.forward(&h, t_emb, latent_mask)?;
        for b in &self.convnext_1 { h = b.forward(&h, latent_mask)?; }
        h = self.text_cond.forward(&h, text, latent_mask, text_mask)?;
        for b in &self.convnext_2 { h = b.forward(&h, latent_mask)?; }
        h = self.style_cond.forward(&h, style_k, style_v, latent_mask)?;
        Ok(h)
    }
}

// ============================ The full vector_estimator ====================

pub struct CandleVectorEstimator {
    proj_in: Conv1d,
    time_encoder: TimeEncoder,
    main_blocks: Vec<MainBlock>,
    last_convnext: Vec<ConvNextBlock>,
    proj_out: Conv1d,
    // Uncond special tokens for classifier-free guidance. The model bundles
    // CFG with scale=4 in its tail; we substitute these for text_emb /
    // style_ttl in the second half of a batch=2 forward pass.
    text_special: Tensor,       // [1, 256, 1]
    style_key_special: Tensor,  // [1, 50, 256]
    style_value_special: Tensor, // [1, 50, 256]
    // Static learned prototype used as the *cond* K input for style_cond
    // (the V side uses real style_ttl). Stored in ONNX as the constant
    // /vector_estimator/Expand_output_0 [1, 50, 256] -- it was originally
    // the output of an Expand that the exporter folded into a constant.
    style_key_cond: Tensor,
}

/// CFG scale baked into the ONNX graph (= 4*v_cond - 3*v_uncond, i.e.
/// v_uncond + 4*(v_cond - v_uncond)).
const CFG_SCALE: f64 = 4.0;

impl CandleVectorEstimator {
    pub fn load(safetensors: &Path, device: &Device) -> Result<Self> {
        let root_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[safetensors], DType::F32, device)
                .context("mmap ve safetensors")?
        };
        // The whole tree lives under `vector_estimator.tts.ttl.vector_field.*`
        let ve = root_vb.pp("vector_estimator");
        let vf = ve.pp("tts").pp("ttl").pp("vector_field");

        // proj_in: weight [512, 144, 1], no bias.
        let proj_in = {
            let w = vf.pp("proj_in").pp("net").get((HIDDEN, LATENT_DIM, 1), "weight")
                .context("proj_in weight")?;
            Conv1d::new(w, None, Conv1dConfig { padding: 0, stride: 1, dilation: 1, groups: 1,
                ..Default::default() })
        };
        let time_encoder = TimeEncoder::load(vf.pp("time_encoder"), &root_vb)?;
        let _ = device; // not needed for this loader anymore

        // The 36 hidden `onnx::MatMul_*` weights follow a clean per-block
        // pattern empirically derived in tools (see git log):
        //   per main_block, base = 3384 + 45*block_idx, offsets within:
        //     time_cond.linear         = base + 0
        //     text_cond.W_query        = base + 6
        //     text_cond.W_key          = base + 7
        //     text_cond.W_value        = base + 8
        //     text_cond.out_fc         = base + 15
        //     style_cond.W_query       = base + 21
        //     style_cond.W_key         = base + 22
        //     style_cond.W_value       = base + 23
        //     style_cond.out_fc        = base + 24
        const BASE: u32 = 3384;
        const PER_BLOCK_STRIDE: u32 = 45;
        let id_for = |block_idx: usize, offset: u32| -> u32 {
            BASE + (block_idx as u32) * PER_BLOCK_STRIDE + offset
        };
        // RoPE theta is stored ONLY under main_blocks.3.attn.theta but
        // shared by all 4 text_cond layers. The accompanying `increments`
        // tensor (i64) is just [0..999], computed via arange at forward time.
        let shared_theta = vf.pp("main_blocks").pp("3").pp("attn")
            .get((1, 1, TEXT_HEAD_DIM / 2), "theta")
            .context("shared theta")?;

        let mut main_blocks = Vec::with_capacity(N_MAIN_BLOCKS);
        for i in 0..N_MAIN_BLOCKS {
            let time_cond_w = format!("onnx::MatMul_{}", id_for(i, 0));
            let text_cond_ids: [u32; 4] = [id_for(i, 6), id_for(i, 7), id_for(i, 8), id_for(i, 15)];
            let style_cond_ids: [u32; 4] = [id_for(i, 21), id_for(i, 22), id_for(i, 23), id_for(i, 24)];
            main_blocks.push(MainBlock::load(
                vf.clone(), &root_vb, i,
                &time_cond_w, text_cond_ids, style_cond_ids,
                shared_theta.clone(),
            ).with_context(|| format!("main_block[{i}]"))?);
        }

        let last_convnext = {
            let lcvb = vf.pp("last_convnext").pp("convnext");
            let mut out = Vec::with_capacity(LAST_CONVNEXT_DILATIONS.len());
            for (i, &d) in LAST_CONVNEXT_DILATIONS.iter().enumerate() {
                out.push(ConvNextBlock::load(lcvb.pp(&i.to_string()), d)
                    .with_context(|| format!("last_convnext[{i}]"))?);
            }
            out
        };
        let proj_out = {
            let w = vf.pp("proj_out").pp("net").get((LATENT_DIM, HIDDEN, 1), "weight")
                .context("proj_out weight")?;
            Conv1d::new(w, None, Conv1dConfig { padding: 0, stride: 1, dilation: 1, groups: 1,
                ..Default::default() })
        };

        // Uncond special tokens, stored under tts.ttl.uncond_masker.*
        let um = ve.pp("tts").pp("ttl").pp("uncond_masker");
        let text_special = um.get((1, TEXT_DIM, 1), "text_special_token")
            .context("text_special_token")?;
        let style_key_special = um.get((1, N_STYLE, STYLE_DIM), "style_key_special_token")
            .context("style_key_special_token")?;
        let style_value_special = um.get((1, N_STYLE, STYLE_DIM), "style_value_special_token")
            .context("style_value_special_token")?;
        // Cond-side style K prototype (a constant-folded Expand output).
        let style_key_cond = root_vb.get((1, N_STYLE, STYLE_DIM), "/vector_estimator/Expand_output_0")
            .context("style_key_cond prototype")?;

        Ok(Self {
            proj_in, time_encoder, main_blocks, last_convnext, proj_out,
            text_special, style_key_special, style_value_special, style_key_cond,
        })
    }

    /// Run the full vector_estimator INCLUDING the bundled CFG + ODE step.
    /// Returns `denoised_latent`: [B, 144, T_lat].
    ///
    /// Internally duplicates inputs to batch=2 (cond + uncond using special
    /// tokens), runs the vector field on both, then applies:
    ///   v_cfg = 4 * v_cond - 3 * v_uncond     (= CFG with scale 4)
    ///   denoised = (noisy_latent + v_cfg / total_step) * latent_mask
    pub fn forward(
        &self,
        noisy_latent: &Tensor, text_emb: &Tensor, style_ttl: &Tensor,
        latent_mask: &Tensor, text_mask: &Tensor,
        current_step: &Tensor, total_step: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let (b, _, t_lat) = noisy_latent.dims3()?;
        let (_, _, t_text) = text_emb.dims3()?;
        debug_assert_eq!(b, 1, "CandleVectorEstimator only supports batch=1 for now");

        // Build batched-2 inputs: [cond, uncond].
        let noisy_b2 = Tensor::cat(&[noisy_latent, noisy_latent], 0)?;
        let latent_mask_b2 = Tensor::cat(&[latent_mask, latent_mask], 0)?;
        let text_mask_b2 = Tensor::cat(&[text_mask, text_mask], 0)?;

        // Uncond text_emb: text_special_token [1, 256, 1] broadcast to
        // [1, 256, T_text], then mask to zero out padding.
        let text_special_bcast = self.text_special
            .broadcast_as((1, TEXT_DIM, t_text))?
            .contiguous()?
            .broadcast_mul(text_mask)?;
        let text_emb_b2 = Tensor::cat(&[text_emb, &text_special_bcast], 0)?;

        // style_k and style_v differ in the uncond half.
        let style_k_b2 = Tensor::cat(&[&self.style_key_cond, &self.style_key_special], 0)?;
        let style_v_b2 = Tensor::cat(&[style_ttl, &self.style_value_special], 0)?;

        // Time embedding (single t value, broadcast to batch=2).
        let t = (current_step / total_step)?;
        let t_b2 = Tensor::cat(&[&t, &t], 0)?;
        let t_emb = self.time_encoder.forward(&t_b2)?; // [2, 64, 1]

        // proj_in.
        let h = self.proj_in.forward(&noisy_b2)?
            .broadcast_mul(&latent_mask_b2)?;
        let mut h = h;
        for block in &self.main_blocks {
            h = block.forward(
                &h, &t_emb,
                &text_emb_b2, &style_k_b2, &style_v_b2,
                &latent_mask_b2, &text_mask_b2,
            )?;
        }
        for b in &self.last_convnext { h = b.forward(&h, &latent_mask_b2)?; }
        let v_b2 = self.proj_out.forward(&h)?.broadcast_mul(&latent_mask_b2)?;

        // CFG combine + Euler step.
        let v_cond = v_b2.narrow(0, 0, 1)?;
        let v_uncond = v_b2.narrow(0, 1, 1)?;
        // v_cfg = CFG_SCALE * v_cond - (CFG_SCALE - 1) * v_uncond
        let v_cfg = ((v_cond * CFG_SCALE)? - (v_uncond * (CFG_SCALE - 1.0))?)?;
        let total = total_step.to_dtype(DType::F32)?.to_vec1::<f32>()?[0];
        let dt = 1.0_f64 / total as f64;
        let scaled = v_cfg.affine(dt, 0.0)?;
        let combined = (noisy_latent + scaled)?;
        combined.broadcast_mul(latent_mask)
    }

    /// Diagnostic forward: returns (v_b2, denoised, hooks).
    /// `hooks` is a Vec<(name, tensor)> sampled at every main_block exit
    /// plus a few other diagnostic points. Used by ve_check to bisect
    /// where the candle output starts diverging from ort.
    pub fn forward_diagnostic(
        &self,
        noisy_latent: &Tensor, text_emb: &Tensor, style_ttl: &Tensor,
        latent_mask: &Tensor, text_mask: &Tensor,
        current_step: &Tensor, total_step: &Tensor,
    ) -> candle_core::Result<(Tensor, Tensor, Vec<(String, Tensor)>)> {
        let mut hooks: Vec<(String, Tensor)> = Vec::new();
        let (b, _, _t_lat) = noisy_latent.dims3()?;
        let (_, _, t_text) = text_emb.dims3()?;
        debug_assert_eq!(b, 1);
        let noisy_b2 = Tensor::cat(&[noisy_latent, noisy_latent], 0)?;
        let latent_mask_b2 = Tensor::cat(&[latent_mask, latent_mask], 0)?;
        let text_mask_b2 = Tensor::cat(&[text_mask, text_mask], 0)?;
        let text_special_bcast = self.text_special
            .broadcast_as((1, TEXT_DIM, t_text))?
            .contiguous()?.broadcast_mul(text_mask)?;
        let text_emb_b2 = Tensor::cat(&[text_emb, &text_special_bcast], 0)?;
        let style_k_b2 = Tensor::cat(&[&self.style_key_cond, &self.style_key_special], 0)?;
        let style_v_b2 = Tensor::cat(&[style_ttl, &self.style_value_special], 0)?;
        let t = (current_step / total_step)?;
        let t_b2 = Tensor::cat(&[&t, &t], 0)?;
        let t_emb = self.time_encoder.forward(&t_b2)?;
        hooks.push(("time_encoder/output".into(), t_emb.clone()));
        let h = self.proj_in.forward(&noisy_b2)?.broadcast_mul(&latent_mask_b2)?;
        hooks.push(("proj_in/Mul_output_0".into(), h.clone()));
        let mut h = h;
        for (block_idx, block) in self.main_blocks.iter().enumerate() {
            let sub_base = block_idx * 6;
            // convnext_0 stack (4 layers)
            for (i, b) in block.convnext_0.iter().enumerate() {
                h = b.forward(&h, &latent_mask_b2)?;
                hooks.push((
                    format!("main_blocks.{}/convnext.{i}/Add_output_0", sub_base),
                    h.clone(),
                ));
            }
            h = block.time_cond.forward(&h, &t_emb, &latent_mask_b2)?;
            hooks.push((format!("main_blocks.{}/Mul_output_0", sub_base + 1), h.clone()));
            for (i, b) in block.convnext_1.iter().enumerate() {
                h = b.forward(&h, &latent_mask_b2)?;
                hooks.push((
                    format!("main_blocks.{}/convnext.{i}/Add_output_0", sub_base + 2),
                    h.clone(),
                ));
            }
            // Hook into text_cond's internals -- this is currently the
            // first divergence point. We tag the hooks with the sub-block
            // index so multiple blocks can be inspected together.
            let mut text_inner: Vec<(String, Tensor)> = Vec::new();
            h = block.text_cond.forward_hooked(
                &h, &text_emb_b2, &latent_mask_b2, &text_mask_b2,
                Some(&mut text_inner),
            )?;
            for (n, t) in text_inner {
                hooks.push((format!("main_blocks.{}/{}", sub_base + 3, n), t));
            }
            hooks.push((format!("main_blocks.{}/Mul_1_output_0", sub_base + 3), h.clone()));
            for (i, b) in block.convnext_2.iter().enumerate() {
                h = b.forward(&h, &latent_mask_b2)?;
                hooks.push((
                    format!("main_blocks.{}/convnext.{i}/Add_output_0", sub_base + 4),
                    h.clone(),
                ));
            }
            let mut style_inner: Vec<(String, Tensor)> = Vec::new();
            h = block.style_cond.forward_hooked(
                &h, &style_k_b2, &style_v_b2, &latent_mask_b2,
                Some(&mut style_inner),
            )?;
            for (n, t) in style_inner {
                hooks.push((format!("main_blocks.{}/{}", sub_base + 5, n), t));
            }
            hooks.push((format!("main_blocks.{}/Mul_1_output_0", sub_base + 5), h.clone()));
        }
        for (i, b) in self.last_convnext.iter().enumerate() {
            h = b.forward(&h, &latent_mask_b2)?;
            hooks.push((format!("last_convnext/convnext.{i}/Add_output_0"), h.clone()));
        }
        let v_b2 = self.proj_out.forward(&h)?.broadcast_mul(&latent_mask_b2)?;
        hooks.push(("proj_out/Mul_output_0".into(), v_b2.clone()));
        let v_cond = v_b2.narrow(0, 0, 1)?;
        let v_uncond = v_b2.narrow(0, 1, 1)?;
        let v_cfg = ((v_cond * CFG_SCALE)? - (v_uncond * (CFG_SCALE - 1.0))?)?;
        let total = total_step.to_dtype(DType::F32)?.to_vec1::<f32>()?[0];
        let scaled = v_cfg.affine(1.0 / total as f64, 0.0)?;
        let combined = (noisy_latent + scaled)?;
        let denoised = combined.broadcast_mul(latent_mask)?;
        Ok((v_b2, denoised, hooks))
    }

    /// Diagnostic variant: also returns the raw batched-2 vector field
    /// output (= what the ONNX `proj_out/Mul_output_0` node produces).
    /// Useful when bisecting where divergence begins vs ort.
    pub fn forward_with_vfield(
        &self,
        noisy_latent: &Tensor, text_emb: &Tensor, style_ttl: &Tensor,
        latent_mask: &Tensor, text_mask: &Tensor,
        current_step: &Tensor, total_step: &Tensor,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let (b, _, _t_lat) = noisy_latent.dims3()?;
        let (_, _, t_text) = text_emb.dims3()?;
        debug_assert_eq!(b, 1);
        let noisy_b2 = Tensor::cat(&[noisy_latent, noisy_latent], 0)?;
        let latent_mask_b2 = Tensor::cat(&[latent_mask, latent_mask], 0)?;
        let text_mask_b2 = Tensor::cat(&[text_mask, text_mask], 0)?;
        let text_special_bcast = self.text_special
            .broadcast_as((1, TEXT_DIM, t_text))?
            .contiguous()?.broadcast_mul(text_mask)?;
        let text_emb_b2 = Tensor::cat(&[text_emb, &text_special_bcast], 0)?;
        let style_k_b2 = Tensor::cat(&[&self.style_key_cond, &self.style_key_special], 0)?;
        let style_v_b2 = Tensor::cat(&[style_ttl, &self.style_value_special], 0)?;
        let t = (current_step / total_step)?;
        let t_b2 = Tensor::cat(&[&t, &t], 0)?;
        let t_emb = self.time_encoder.forward(&t_b2)?;
        let h = self.proj_in.forward(&noisy_b2)?.broadcast_mul(&latent_mask_b2)?;
        let mut h = h;
        for block in &self.main_blocks {
            h = block.forward(
                &h, &t_emb, &text_emb_b2, &style_k_b2, &style_v_b2,
                &latent_mask_b2, &text_mask_b2,
            )?;
        }
        for b in &self.last_convnext { h = b.forward(&h, &latent_mask_b2)?; }
        let v_b2 = self.proj_out.forward(&h)?.broadcast_mul(&latent_mask_b2)?;
        // ODE step
        let v_cond = v_b2.narrow(0, 0, 1)?;
        let v_uncond = v_b2.narrow(0, 1, 1)?;
        let v_cfg = ((v_cond * CFG_SCALE)? - (v_uncond * (CFG_SCALE - 1.0))?)?;
        let total = total_step.to_dtype(DType::F32)?.to_vec1::<f32>()?[0];
        let scaled = v_cfg.affine(1.0 / total as f64, 0.0)?;
        let combined = (noisy_latent + scaled)?;
        let denoised = combined.broadcast_mul(latent_mask)?;
        Ok((v_b2, denoised))
    }
}

// ============================ Helpers (copied for now) =====================

/// Effective sequence length from a [B, 1, T] mask. We use sample 0 only
/// (B=1 in our supported case).
fn sum_mask(mask: &Tensor) -> candle_core::Result<f32> {
    let v = mask.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = mask.dim(0)?;
    let per_batch = v.len() / b;
    Ok(v[..per_batch].iter().sum())
}

fn pad_edge(x: &Tensor, left: usize, right: usize) -> candle_core::Result<Tensor> {
    if left == 0 && right == 0 { return Ok(x.clone()); }
    let t = x.dim(D::Minus1)?;
    let mut parts: Vec<Tensor> = Vec::with_capacity(3);
    if left > 0 {
        let first = x.narrow(D::Minus1, 0, 1)?;
        let mut shape = first.dims().to_vec();
        *shape.last_mut().unwrap() = left;
        parts.push(first.broadcast_as(shape)?.contiguous()?);
    }
    parts.push(x.clone());
    if right > 0 {
        let last = x.narrow(D::Minus1, t - 1, 1)?;
        let mut shape = last.dims().to_vec();
        *shape.last_mut().unwrap() = right;
        parts.push(last.broadcast_as(shape)?.contiguous()?);
    }
    Tensor::cat(&parts, D::Minus1)
}

fn gelu_erf(x: &Tensor) -> candle_core::Result<Tensor> {
    let inv_sqrt2 = 2.0f64.sqrt().recip();
    let half = Tensor::new(0.5f32, x.device())?.to_dtype(x.dtype())?;
    let one = Tensor::new(1.0f32, x.device())?.to_dtype(x.dtype())?;
    let scale = Tensor::new(inv_sqrt2 as f32, x.device())?.to_dtype(x.dtype())?;
    let erf_arg = x.broadcast_mul(&scale)?.erf()?;
    let inner = erf_arg.broadcast_add(&one)?;
    let scaled = x.broadcast_mul(&half)?;
    scaled.broadcast_mul(&inner)
}

#[allow(dead_code)]
fn _unused(x: &Tensor) -> candle_core::Result<Tensor> { x.i(0) }
