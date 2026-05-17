//! Candle port of the Supertonic 3 text_encoder.
//!
//! ```text
//! text_ids [B, T] (i64)
//!   ► char_embedder Gather -> [B, T, 256] -> transpose -> [B, 256, T]
//!   ► * text_mask  (mask padding)
//!   ► ConvNeXt × 6 (k=5, dilations [1,1,2,2,4,4], idim=256, intermediate=1024)
//!   ► AttnEncoder × 4 (post-norm, n_heads=4, head_dim=64, window=4)
//!   ► outer residual: text_after_attn + convnext_out
//!   ► * text_mask  (proj_out is JUST a mask -- no Conv weight)
//!   ► SpeechPromptedTextEncoder:
//!       Q from text, K from style_key prototype (tiled), V from style_ttl
//!       attention1 + residual + attention2 + residual + LayerNorm
//!   ► * text_mask  -> text_emb [B, 256, T]
//! ```
//!
//! Architecturally a near-twin of DP's sentence_encoder, but wider,
//! with dilated ConvNeXts, no sentence_token prepend, and a 2-step
//! cross-attention block at the tail that conditions on the voice
//! style. The first three stages reuse the same patterns Phase 3
//! locked down (post-norm, symmetric edge pad, mask muls, the
//! shift-with-RIGHT-pad in `relative_position_to_absolute_position`).

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, VarBuilder};
use std::path::Path;

const CHAR_VOCAB: usize = 8322;
const HIDDEN: usize = 256;
const INTERMEDIATE: usize = 1024;
const N_HEADS_ATTN: usize = 4; // attn_encoder
const HEAD_DIM_ATTN: usize = HIDDEN / N_HEADS_ATTN; // 64
const WINDOW: usize = 4; // emb_rel_k/v has 2*W+1 = 9 positions
const N_CONVNEXT: usize = 6;
const CONVNEXT_KSZ: usize = 5;
const CONVNEXT_DILATIONS: [usize; N_CONVNEXT] = [1, 1, 2, 2, 4, 4];
const N_ATTN_LAYERS: usize = 4;
const LN_EPS: f64 = 1e-6;
/// spte cross-attention head count.
const SPTE_N_HEADS: usize = 2;
const SPTE_HEAD_DIM: usize = HIDDEN / SPTE_N_HEADS; // 128
const SPTE_STYLE_TOKENS: usize = 50; // n_style

// ============================ ConvNeXt block ===============================

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
        let dwconv = candle_nn::conv1d(
            HIDDEN, HIDDEN, CONVNEXT_KSZ,
            Conv1dConfig { padding: 0, stride: 1, dilation, groups: HIDDEN,
                ..Default::default() },
            vb.pp("dwconv"),
        ).context("dwconv")?;
        let norm = candle_nn::layer_norm(HIDDEN, LN_EPS, vb.pp("norm.norm"))
            .context("convnext norm")?;
        let pwconv1 = candle_nn::conv1d(
            HIDDEN, INTERMEDIATE, 1,
            Conv1dConfig { padding: 0, stride: 1, dilation: 1, groups: 1,
                ..Default::default() },
            vb.pp("pwconv1"),
        ).context("pwconv1")?;
        let pwconv2 = candle_nn::conv1d(
            INTERMEDIATE, HIDDEN, 1,
            Conv1dConfig { padding: 0, stride: 1, dilation: 1, groups: 1,
                ..Default::default() },
            vb.pp("pwconv2"),
        ).context("pwconv2")?;
        let gamma = vb.get((1, HIDDEN, 1), "gamma").context("gamma")?;
        Ok(Self { dwconv, norm, pwconv1, pwconv2, gamma, dilation })
    }

    fn forward(
        &self, x: &Tensor, mask: &Tensor, idx: usize, hooks: &mut Hooks,
    ) -> candle_core::Result<Tensor> {
        // Symmetric edge pad: total = dilation * (k - 1).
        let total = self.dilation * (CONVNEXT_KSZ - 1);
        let l = total / 2;
        let r = total - l;
        let h = pad_edge(x, l, r)?;
        let h = self.dwconv.forward(&h)?;
        hooks.record(&format!("convnext.{idx}/dwconv/Conv_output_0"), &h)?;
        let h = h.broadcast_mul(mask)?;
        let h_nhwc = h.transpose(1, 2)?.contiguous()?;
        let h_ln = self.norm.forward(&h_nhwc)?;
        hooks.record(
            &format!("convnext.{idx}/norm/norm/LayerNormalization_output_0"),
            &h_ln,
        )?;
        let h = h_ln.transpose(1, 2)?.contiguous()?;
        let h = self.pwconv1.forward(&h)?;
        hooks.record(&format!("convnext.{idx}/pwconv1/Conv_output_0"), &h)?;
        let h = gelu_erf(&h)?;
        let h = self.pwconv2.forward(&h)?;
        hooks.record(&format!("convnext.{idx}/pwconv2/Conv_output_0"), &h)?;
        let h = self.gamma.broadcast_mul(&h)?;
        let out = (x + h)?;
        hooks.record(&format!("convnext.{idx}/Add_output_0"), &out)?;
        out.broadcast_mul(mask)
    }
}

// ============================ Relative attention (VITS-style) ==============

struct AttnLayer {
    conv_q: Conv1d,
    conv_k: Conv1d,
    conv_v: Conv1d,
    conv_o: Conv1d,
    emb_rel_k: Tensor,
    emb_rel_v: Tensor,
}

impl AttnLayer {
    fn load(vb: VarBuilder) -> Result<Self> {
        let make = |name: &str| -> Result<Conv1d> {
            candle_nn::conv1d(HIDDEN, HIDDEN, 1,
                Conv1dConfig { padding: 0, stride: 1, dilation: 1, groups: 1,
                    ..Default::default() },
                vb.pp(name),
            ).with_context(|| format!("attn.{name}"))
        };
        let emb_rel_k = vb.get((1, 2 * WINDOW + 1, HEAD_DIM_ATTN), "emb_rel_k")
            .context("emb_rel_k")?;
        let emb_rel_v = vb.get((1, 2 * WINDOW + 1, HEAD_DIM_ATTN), "emb_rel_v")
            .context("emb_rel_v")?;
        Ok(Self {
            conv_q: make("conv_q")?, conv_k: make("conv_k")?,
            conv_v: make("conv_v")?, conv_o: make("conv_o")?,
            emb_rel_k, emb_rel_v,
        })
    }

    fn forward(
        &self, x: &Tensor, attn_mask: &Tensor, idx: usize, hooks: &mut Hooks,
    ) -> candle_core::Result<Tensor> {
        let q = self.conv_q.forward(x)?;
        hooks.record(&format!("attn_layers.{idx}/conv_q/Conv_output_0"), &q)?;
        let k = self.conv_k.forward(x)?;
        hooks.record(&format!("attn_layers.{idx}/conv_k/Conv_output_0"), &k)?;
        let v = self.conv_v.forward(x)?;
        hooks.record(&format!("attn_layers.{idx}/conv_v/Conv_output_0"), &v)?;
        let (b, _c, t) = q.dims3()?;
        let split = |t_: &Tensor| -> candle_core::Result<Tensor> {
            t_.reshape((b, N_HEADS_ATTN, HEAD_DIM_ATTN, t))?
                .transpose(2, 3)?.contiguous()
        };
        let q = split(&q)?;
        let k = split(&k)?;
        let v = split(&v)?;

        let inv_sqrt_d = (HEAD_DIM_ATTN as f64).sqrt().recip();
        let q_scaled = (q.clone() * inv_sqrt_d)?.contiguous()?;
        let kt = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let scores = q_scaled.matmul(&kt)?;
        hooks.record(&format!("attn_layers.{idx}/MatMul_output_0"), &scores)?;

        let ext_k = get_relative_embeddings(&self.emb_rel_k, t)?.contiguous()?;
        let ext_kt = ext_k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let rel_logits = q_scaled.broadcast_matmul(&ext_kt)?;
        hooks.record(&format!("attn_layers.{idx}/MatMul_1_output_0"), &rel_logits)?;
        let rel_scores = relative_position_to_absolute_position(&rel_logits)?;
        let scores = (scores + rel_scores)?;
        hooks.record(&format!("attn_layers.{idx}/Add_2_output_0"), &scores)?;

        let mask_b = attn_mask.broadcast_as(scores.shape())?;
        let bias = mask_b.affine(-1.0, 1.0)?.affine(-1e4, 0.0)?;
        let scores = (scores + bias)?;
        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
        hooks.record(&format!("attn_layers.{idx}/Softmax_output_0"), &attn)?;

        let out_av = attn.matmul(&v)?;
        hooks.record(&format!("attn_layers.{idx}/MatMul_2_output_0"), &out_av)?;
        let rel_w = absolute_position_to_relative_position(&attn)?.contiguous()?;
        let ext_v = get_relative_embeddings(&self.emb_rel_v, t)?.contiguous()?;
        let out_rv = rel_w.broadcast_matmul(&ext_v)?;
        hooks.record(&format!("attn_layers.{idx}/MatMul_3_output_0"), &out_rv)?;
        let out = (out_av + out_rv)?;
        hooks.record(&format!("attn_layers.{idx}/Add_4_output_0"), &out)?;

        let merged = out.transpose(2, 3)?.contiguous()?.reshape((b, HIDDEN, t))?;
        let o = self.conv_o.forward(&merged)?;
        hooks.record(&format!("attn_layers.{idx}/conv_o/Conv_output_0"), &o)?;
        Ok(o)
    }
}

fn get_relative_embeddings(rel_emb: &Tensor, t: usize) -> candle_core::Result<Tensor> {
    let want = 2 * t - 1;
    let pad_len = (t as isize - (WINDOW as isize + 1)).max(0) as usize;
    let slice_start = ((WINDOW as isize + 1) - t as isize).max(0) as usize;
    let padded = if pad_len > 0 {
        let device = rel_emb.device();
        let dtype = rel_emb.dtype();
        let (_b, _, d) = rel_emb.dims3()?;
        let zeros = Tensor::zeros((1, pad_len, d), dtype, device)?;
        Tensor::cat(&[&zeros, rel_emb, &zeros], 1)?
    } else {
        rel_emb.clone()
    };
    padded.narrow(1, slice_start, want)
}

fn relative_position_to_absolute_position(x: &Tensor) -> candle_core::Result<Tensor> {
    // Same right-pad shift as DP -- see Phase 3's notes on why this differs
    // from canonical VITS.
    let (b, h, t, two_t_minus_1) = x.dims4()?;
    debug_assert_eq!(two_t_minus_1, 2 * t - 1);
    let x = x.pad_with_zeros(D::Minus1, 0, 1)?;
    let x = x.reshape((b, h, t * 2 * t))?;
    let x = x.pad_with_zeros(D::Minus1, 0, t - 1)?;
    let x = x.reshape((b, h, t + 1, 2 * t - 1))?;
    x.narrow(2, 0, t)?.narrow(3, t - 1, t)
}

fn absolute_position_to_relative_position(x: &Tensor) -> candle_core::Result<Tensor> {
    let (b, h, t, t2) = x.dims4()?;
    debug_assert_eq!(t, t2);
    let x = x.pad_with_zeros(D::Minus1, 0, t - 1)?;
    let x = x.reshape((b, h, t * (2 * t - 1)))?;
    let x = x.pad_with_zeros(D::Minus1, t, 0)?;
    let x = x.reshape((b, h, t, 2 * t))?;
    x.narrow(3, 1, 2 * t - 1)
}

// ============================ FFN (ReLU, with mask muls) ===================

struct FfnLayer {
    conv_1: Conv1d,
    conv_2: Conv1d,
}

impl FfnLayer {
    fn load(vb: VarBuilder) -> Result<Self> {
        let conv_1 = candle_nn::conv1d(HIDDEN, INTERMEDIATE, 1,
            Conv1dConfig { padding: 0, stride: 1, dilation: 1, groups: 1,
                ..Default::default() },
            vb.pp("conv_1"),
        ).context("ffn.conv_1")?;
        let conv_2 = candle_nn::conv1d(INTERMEDIATE, HIDDEN, 1,
            Conv1dConfig { padding: 0, stride: 1, dilation: 1, groups: 1,
                ..Default::default() },
            vb.pp("conv_2"),
        ).context("ffn.conv_2")?;
        Ok(Self { conv_1, conv_2 })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor, idx: usize, hooks: &mut Hooks)
        -> candle_core::Result<Tensor>
    {
        let h = x.broadcast_mul(mask)?;
        let h = self.conv_1.forward(&h)?;
        hooks.record(&format!("ffn_layers.{idx}/conv_1/Conv_output_0"), &h)?;
        let h = h.relu()?;
        let h = h.broadcast_mul(mask)?;
        let h = self.conv_2.forward(&h)?;
        hooks.record(&format!("ffn_layers.{idx}/conv_2/Conv_output_0"), &h)?;
        h.broadcast_mul(mask)
    }
}

// ============================ AttnEncoder (4 post-norm layers) =============

struct AttnEncoder {
    attn_layers: Vec<AttnLayer>,
    ffn_layers: Vec<FfnLayer>,
    norm_layers_1: Vec<LayerNorm>,
    norm_layers_2: Vec<LayerNorm>,
}

impl AttnEncoder {
    fn load(vb: VarBuilder) -> Result<Self> {
        let mut attn_layers = Vec::with_capacity(N_ATTN_LAYERS);
        let mut ffn_layers = Vec::with_capacity(N_ATTN_LAYERS);
        let mut norm_layers_1 = Vec::with_capacity(N_ATTN_LAYERS);
        let mut norm_layers_2 = Vec::with_capacity(N_ATTN_LAYERS);
        for i in 0..N_ATTN_LAYERS {
            attn_layers.push(AttnLayer::load(vb.pp("attn_layers").pp(&i.to_string()))
                .with_context(|| format!("attn_layers[{i}]"))?);
            ffn_layers.push(FfnLayer::load(vb.pp("ffn_layers").pp(&i.to_string()))
                .with_context(|| format!("ffn_layers[{i}]"))?);
            norm_layers_1.push(candle_nn::layer_norm(HIDDEN, LN_EPS,
                vb.pp("norm_layers_1").pp(&i.to_string()).pp("norm"),
            ).with_context(|| format!("norm_layers_1[{i}]"))?);
            norm_layers_2.push(candle_nn::layer_norm(HIDDEN, LN_EPS,
                vb.pp("norm_layers_2").pp(&i.to_string()).pp("norm"),
            ).with_context(|| format!("norm_layers_2[{i}]"))?);
        }
        Ok(Self { attn_layers, ffn_layers, norm_layers_1, norm_layers_2 })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor, hooks: &mut Hooks)
        -> candle_core::Result<Tensor>
    {
        let (b, _, t) = mask.dims3()?;
        let m_q = mask.reshape((b, t, 1))?;
        let m_k = mask.reshape((b, 1, t))?;
        let attn_mask = m_q.broadcast_matmul(&m_k)?.reshape((b, 1, t, t))?;
        let mut h = x.clone();
        for i in 0..N_ATTN_LAYERS {
            let attn_out = self.attn_layers[i].forward(&h, &attn_mask, i, hooks)?;
            h = (h + attn_out)?;
            let h_nhwc = h.transpose(1, 2)?.contiguous()?;
            let h_norm = self.norm_layers_1[i].forward(&h_nhwc)?;
            hooks.record(
                &format!("norm_layers_1.{i}/norm/LayerNormalization_output_0"),
                &h_norm,
            )?;
            h = h_norm.transpose(1, 2)?.contiguous()?;

            let ffn_out = self.ffn_layers[i].forward(&h, mask, i, hooks)?;
            h = (h + ffn_out)?;
            let h_nhwc = h.transpose(1, 2)?.contiguous()?;
            let h_norm = self.norm_layers_2[i].forward(&h_nhwc)?;
            hooks.record(
                &format!("norm_layers_2.{i}/norm/LayerNormalization_output_0"),
                &h_norm,
            )?;
            h = h_norm.transpose(1, 2)?.contiguous()?;
        }
        h.broadcast_mul(mask)
    }
}

// ============================ spte cross-attention =========================
//
// Standard MHA cross-attention -- no rel positions. Q from text, K from a
// tiled prototype tensor (style_key), V from style_ttl.

struct SpteCrossAttn {
    w_q: candle_nn::Linear,
    w_k: candle_nn::Linear,
    w_v: candle_nn::Linear,
    out_fc: candle_nn::Linear,
}

impl SpteCrossAttn {
    /// `vb` should point at the module root (e.g. spte.attention1).
    /// Weights are stored under the renamed `onnx::MatMul_*` names; the
    /// caller passes a slice of those weight tensors in matmul order:
    /// [w_q, w_k, w_v, out_fc].
    fn load(vb: VarBuilder, weights: [Tensor; 4]) -> Result<Self> {
        let bias = |n: &str| -> Result<Tensor> {
            vb.pp(n).pp("linear").get(HIDDEN, "bias").with_context(|| format!("{n}.bias"))
        };
        let make = |w: Tensor, b: Tensor| -> candle_nn::Linear {
            // Linear in candle stores weight as [out, in]; the ONNX `MatMul`
            // does X @ W (so W is [in, out]). Transpose to candle layout.
            let wt = w.t().expect("transpose w").contiguous().expect("contig");
            candle_nn::Linear::new(wt, Some(b))
        };
        let w_q = make(weights[0].clone(), bias("W_query")?);
        let w_k = make(weights[1].clone(), bias("W_key")?);
        let w_v = make(weights[2].clone(), bias("W_value")?);
        let out_fc = make(weights[3].clone(), bias("out_fc")?);
        Ok(Self { w_q, w_k, w_v, out_fc })
    }

    /// `q_in`:  [B, T, 256]  (text features, transposed)
    /// `k_in`:  [B, 50, 256] (tiled style_key prototypes)
    /// `v_in`:  [B, 50, 256] (style_ttl input)
    /// `mask`:  [B, 1, T]    (text_mask, for the final output mul)
    /// Returns [B, T, 256] (still in [B, T, C] layout for the residual).
    ///
    /// The Supertonic variant has a non-standard scaled-dot-product:
    ///   scores = Q @ tanh(K^T) / sqrt(d_head)
    /// The tanh on K is a Supertonic-specific choice -- empirically
    /// found by tracing the ONNX `Transpose -> Tanh -> MatMul` chain
    /// on the K side. Without it the per-head scale is wrong by a
    /// factor that depends on K magnitudes.
    fn forward(
        &self, q_in: &Tensor, k_in: &Tensor, v_in: &Tensor, mask: &Tensor,
        idx: usize, hooks: &mut Hooks,
    ) -> candle_core::Result<Tensor> {
        let q = self.w_q.forward(q_in)?;
        let k = self.w_k.forward(k_in)?;
        let v = self.w_v.forward(v_in)?;
        hooks.record(&format!("spte/attention{idx}/W_query"), &q)?;
        hooks.record(&format!("spte/attention{idx}/W_key"), &k)?;
        hooks.record(&format!("spte/attention{idx}/W_value"), &v)?;

        let (b, tq, _) = q.dims3()?;
        let (_, tk, _) = k.dims3()?;
        let split = |t: &Tensor, seq: usize| -> candle_core::Result<Tensor> {
            t.reshape((b, seq, SPTE_N_HEADS, SPTE_HEAD_DIM))?
                .transpose(1, 2)?.contiguous()
        };
        let q = split(&q, tq)?;
        let k = split(&k, tk)?;
        let v = split(&v, tk)?;

        // Supertonic divides by sqrt(HIDDEN) (= 16 for HIDDEN=256), NOT by
        // sqrt(head_dim). Empirically confirmed: the ONNX Div has constant
        // 16.0, regardless of how heads are split.
        let scale = (HIDDEN as f64).sqrt().recip();
        let kt = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let kt_tanh = kt.tanh()?; // Supertonic-specific: tanh-squash K before QK^T
        let scores = q.matmul(&kt_tanh)?;
        let scores = (scores * scale)?;
        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let out = attn.matmul(&v)?;

        let merged = out.transpose(1, 2)?.contiguous()?.reshape((b, tq, HIDDEN))?;
        let proj = self.out_fc.forward(&merged)?;
        // Final mask multiplication: text_mask is [B, 1, T] -> need [B, T, 1]
        // to broadcast over the channel dim of `proj`.
        let mask_b = mask.transpose(1, 2)?.contiguous()?;
        let proj = proj.broadcast_mul(&mask_b)?;
        hooks.record(&format!("spte/attention{idx}/out_fc"), &proj)?;
        Ok(proj)
    }
}

struct SpeechPromptedTextEncoder {
    attention1: SpteCrossAttn,
    attention2: SpteCrossAttn,
    norm: LayerNorm,
    style_key_proto: Tensor, // [1, 50, 256]
}

impl SpeechPromptedTextEncoder {
    fn load(
        ttl_vb: VarBuilder, root_vb: VarBuilder,
        style_key_proto: Tensor,
    ) -> Result<Self> {
        let spte = ttl_vb.pp("speech_prompted_text_encoder");
        // The 8 attention weights live at top level as onnx::MatMul_3680..3687
        // (verified earlier). We unpack them in slot order:
        // 3680 = attention1.W_query, 3681 = attention1.W_key, 3682 = attention1.W_value,
        // 3683 = attention1.out_fc, 3684..3687 = attention2.* (same order).
        let mut weights = Vec::with_capacity(8);
        for id in 3680..=3687u32 {
            let w = root_vb.get((HIDDEN, HIDDEN), &format!("onnx::MatMul_{id}"))
                .with_context(|| format!("onnx::MatMul_{id}"))?;
            weights.push(w);
        }
        let attention1 = SpteCrossAttn::load(
            spte.pp("attention1"),
            [weights[0].clone(), weights[1].clone(), weights[2].clone(), weights[3].clone()],
        )?;
        let attention2 = SpteCrossAttn::load(
            spte.pp("attention2"),
            [weights[4].clone(), weights[5].clone(), weights[6].clone(), weights[7].clone()],
        )?;
        let norm = candle_nn::layer_norm(HIDDEN, LN_EPS, spte.pp("norm.norm"))
            .context("spte.norm.norm")?;
        Ok(Self { attention1, attention2, norm, style_key_proto })
    }

    /// `text_in`: [B, 256, T] (already mask-multiplied).
    /// `style_ttl`: [B, 50, 256].
    /// Returns [B, 256, T] (and the caller multiplies by mask).
    fn forward(
        &self, text_in: &Tensor, style_ttl: &Tensor, mask: &Tensor, hooks: &mut Hooks,
    ) -> candle_core::Result<Tensor> {
        let (b, _, _t) = text_in.dims3()?;
        // Transpose text to [B, T, 256] for MatMul.
        let q_in = text_in.transpose(1, 2)?.contiguous()?;
        // Tile style_key to [B, 50, 256].
        let k_in = self.style_key_proto.broadcast_as((b, SPTE_STYLE_TOKENS, HIDDEN))?
            .contiguous()?;
        // V comes straight from style_ttl input.
        let v_in = style_ttl.clone();

        // Both residuals are added to the ORIGINAL text input, not to the
        // running stream. Confirmed by tracing the ONNX Adds:
        //   Add   = attention1(q_in) + q_in
        //   Add_1 = attention2(h1)   + q_in     <- note: NOT + h1
        // Attention2 takes h1 (the post-Add output) as Q input, but the
        // residual baseline rewinds to q_in.
        let a1 = self.attention1.forward(&q_in, &k_in, &v_in, mask, 1, hooks)?;
        let h1 = (&q_in + &a1)?;
        hooks.record("spte/Add_output_0", &h1)?;

        let a2 = self.attention2.forward(&h1, &k_in, &v_in, mask, 2, hooks)?;
        let h2 = (&q_in + &a2)?;
        hooks.record("spte/Add_1_output_0", &h2)?;

        // Final LayerNorm on the last dim of [B, T, 256] (i.e., over channels).
        let h_norm = self.norm.forward(&h2)?;
        hooks.record("spte/norm/output", &h_norm)?;
        // Transpose back to [B, 256, T] and mask.
        let out = h_norm.transpose(1, 2)?.contiguous()?;
        let out = out.broadcast_mul(mask)?;
        hooks.record("spte/final_text_emb", &out)?;
        Ok(out)
    }
}

// ============================ Glue: TextEncoder ============================

pub struct CandleTextEncoder {
    char_embed: Tensor,
    convnext: Vec<ConvNextBlock>,
    attn_encoder: AttnEncoder,
    spte: SpeechPromptedTextEncoder,
}

impl CandleTextEncoder {
    pub fn load(safetensors: &Path, device: &Device) -> Result<Self> {
        let root_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[safetensors], DType::F32, device)
                .context("mmap text_encoder safetensors")?
        };
        let ttl = root_vb.pp("tts").pp("ttl");
        let te = ttl.pp("text_encoder");
        let char_embed = te.pp("text_embedder").pp("char_embedder")
            .get((CHAR_VOCAB, HIDDEN), "weight").context("char_embedder.weight")?;
        let mut convnext = Vec::with_capacity(N_CONVNEXT);
        for (i, &d) in CONVNEXT_DILATIONS.iter().enumerate() {
            convnext.push(ConvNextBlock::load(
                te.pp("convnext").pp("convnext").pp(&i.to_string()), d,
            ).with_context(|| format!("convnext[{i}] (dilation={d})"))?);
        }
        let attn_encoder = AttnEncoder::load(te.pp("attn_encoder"))?;
        let style_key_proto = ttl.pp("style_encoder").pp("style_token_layer")
            .get((1, SPTE_STYLE_TOKENS, HIDDEN), "style_key")
            .context("style_key prototype")?;
        let spte = SpeechPromptedTextEncoder::load(ttl, root_vb, style_key_proto)?;
        Ok(Self { char_embed, convnext, attn_encoder, spte })
    }

    /// `text_ids`: [B, T] i64.  `style_ttl`: [B, 50, 256].  `text_mask`: [B, 1, T].
    /// Returns text_emb: [B, 256, T].
    pub fn forward(
        &self, text_ids: &Tensor, style_ttl: &Tensor, text_mask: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let mut hooks = Hooks::default();
        self.forward_with_hooks(text_ids, style_ttl, text_mask, &mut hooks)
    }

    pub fn forward_with_hooks(
        &self, text_ids: &Tensor, style_ttl: &Tensor, text_mask: &Tensor,
        hooks: &mut Hooks,
    ) -> candle_core::Result<Tensor> {
        let (b, t) = text_ids.dims2()?;
        let ids_flat = text_ids.flatten_all()?;
        let emb_flat = self.char_embed.embedding(&ids_flat)?;
        let emb = emb_flat.reshape((b, t, HIDDEN))?
            .transpose(1, 2)?.contiguous()?;
        let emb_masked = emb.broadcast_mul(text_mask)?;

        // ConvNeXt stack: each block consumes and returns a masked tensor.
        let mut h = emb_masked;
        for (i, block) in self.convnext.iter().enumerate() {
            h = block.forward(&h, text_mask, i, hooks)?;
        }
        let convnext_out = h.clone();

        // Attn encoder + outer residual + proj_out mask mul.
        let attn_out = self.attn_encoder.forward(&convnext_out, text_mask, hooks)?;
        let proj_out_in = (convnext_out + attn_out)?;
        hooks.record("Add_output_0", &proj_out_in)?;
        let proj_out_out = proj_out_in.broadcast_mul(text_mask)?;
        hooks.record("proj_out/Mul_output_0", &proj_out_out)?;

        // spte cross-attn block.
        let text_emb = self.spte.forward(&proj_out_out, style_ttl, text_mask, hooks)?;
        hooks.record("text_emb", &text_emb)?;
        Ok(text_emb)
    }
}

// ============================ Helpers (shared with DP) ====================

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

#[derive(Default)]
pub struct Hooks {
    pub records: Vec<(String, Tensor)>,
}

impl Hooks {
    fn record(&mut self, name: &str, t: &Tensor) -> candle_core::Result<()> {
        self.records.push((name.to_string(), t.clone()));
        Ok(())
    }
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.records.iter().find(|(n, _)| n == name).map(|(_, t)| t)
    }
}

#[allow(dead_code)]
fn _unused(x: &Tensor) -> candle_core::Result<Tensor> { x.i(0) }
