//! Candle port of the Supertonic 3 duration predictor.
//!
//! Architecture (derived from ONNX walk + tts.json):
//!
//! ```text
//! text_ids [B, T] (i64)
//!   ► char_embedder: nn.Embedding(8322, 64) -> [B, T, 64] -> transpose -> [B, 64, T]
//!   ► × text_mask [B, 1, T]    (zero out padding)
//!   ► prepend sentence_token [1, 64, 1] -> [B, 64, T+1]
//!   ► ConvNeXt × 6 (k=5, dilation=1, idim=64, intermediate=256)
//!   ► AttnEncoder × 2:  (LN -> rel-attn -> +residual) -> (LN -> FFN -> +residual)
//!     -- rel-attn is VITS-style with emb_rel_k/v of shape [1, 2W+1=9, 32]
//!     -- masked with attn_mask = (m_q @ m_k^T) so positions outside the
//!        valid sentence_token+text region don't attend across each other.
//!   ► slice first time step -> [B, 64, 1]
//!   ► proj_out Conv1d(64 -> 64, k=1, no bias) -> [B, 64, 1] -> squeeze -> [B, 64]
//!   ► concat with flatten(style_dp) [B, 128] -> [B, 192]
//!   ► predictor: Linear(192 -> 128) -> PReLU(alpha[1]) -> Linear(128 -> 1) -> squeeze
//!   = duration: [B] in seconds
//! ```

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{
    Conv1d, Conv1dConfig, LayerNorm, Linear, ModuleT, VarBuilder,
};
use std::path::Path;

const CHAR_VOCAB: usize = 8322;
const HIDDEN_DIM: usize = 64;
const INTERMEDIATE_DIM: usize = 256;
const N_HEADS: usize = 2;
const HEAD_DIM: usize = HIDDEN_DIM / N_HEADS; // 32
const WINDOW_SIZE: usize = 4; // emb_rel_k/v has 2*W+1 = 9 positions
const N_CONVNEXT: usize = 6;
const CONVNEXT_KSZ: usize = 5;
const N_ATTN_LAYERS: usize = 2;
/// ONNX LayerNormalization default eps; same as in the vocoder.
const LN_EPS: f64 = 1e-6;
/// Style ↦ predictor flat dim. style_dp is [B, 8, 16] → flatten → 128.
const STYLE_FLAT_DIM: usize = 8 * 16;
/// Predictor hidden width.
const PREDICTOR_HDIM: usize = 128;
/// Predictor input = sentence_dim (64) + style_flat (128) = 192.
const PREDICTOR_IN_DIM: usize = HIDDEN_DIM + STYLE_FLAT_DIM;

// =========================== ConvNeXt block (DP variant) ===================
// Same structure as the vocoder's ConvNeXt but with channel dim 64, k=5, no
// dilation. Kept separate from vocoder.rs so the constants stay readable.

struct ConvNextBlock {
    dwconv: Conv1d,
    norm: LayerNorm,
    pwconv1: Conv1d,
    pwconv2: Conv1d,
    gamma: Tensor,
}

impl ConvNextBlock {
    fn load(vb: VarBuilder) -> Result<Self> {
        let dwconv = candle_nn::conv1d(
            HIDDEN_DIM, HIDDEN_DIM, CONVNEXT_KSZ,
            Conv1dConfig { padding: 0, stride: 1, dilation: 1, groups: HIDDEN_DIM,
                ..Default::default() },
            vb.pp("dwconv"),
        ).context("dwconv")?;
        // Note: dwconv weights are .dwconv.weight / .dwconv.bias here -- NO ".net."
        // sub-prefix (unlike the vocoder, which wraps dwconv in a weight-norm
        // adapter). That's already handled by passing vb.pp("dwconv") above.
        let norm = candle_nn::layer_norm(HIDDEN_DIM, LN_EPS, vb.pp("norm.norm"))
            .context("convnext norm")?;
        let pwconv1 = candle_nn::conv1d(
            HIDDEN_DIM, INTERMEDIATE_DIM, 1,
            Conv1dConfig { padding: 0, stride: 1, dilation: 1, groups: 1,
                ..Default::default() },
            vb.pp("pwconv1"),
        ).context("pwconv1")?;
        let pwconv2 = candle_nn::conv1d(
            INTERMEDIATE_DIM, HIDDEN_DIM, 1,
            Conv1dConfig { padding: 0, stride: 1, dilation: 1, groups: 1,
                ..Default::default() },
            vb.pp("pwconv2"),
        ).context("pwconv2")?;
        let gamma = vb.get((1, HIDDEN_DIM, 1), "gamma").context("gamma")?;
        Ok(Self { dwconv, norm, pwconv1, pwconv2, gamma })
    }

    /// DP ConvNeXt forward. Differs from the vocoder's in three ways:
    ///   - symmetric edge pad (2 left + 2 right for k=5), not all-causal
    ///   - mask multiplications after dwconv AND after the residual
    ///   - input is assumed already masked (per the ONNX `Mul`-before-Pad
    ///     pattern; the caller is responsible for masking once at the boundary)
    fn forward(
        &self, x: &Tensor, mask: &Tensor, idx: usize, hooks: &mut Hooks,
    ) -> candle_core::Result<Tensor> {
        // Symmetric edge pad on time axis: total = CONVNEXT_KSZ-1 = 4 -> 2+2.
        let total_pad = CONVNEXT_KSZ - 1;
        let l = total_pad / 2;
        let r = total_pad - l;
        let h = pad_edge(x, l, r)?;
        let h = self.dwconv.forward(&h)?;
        hooks.record(&format!("convnext.{idx}/dwconv/Conv_output_0"), &h)?;
        // Mul_1: mask after dwconv (zeros out values that leaked into pad).
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
        // Mul_2: gamma scale (learnable per-channel weight).
        let h = self.gamma.broadcast_mul(&h)?;
        let out = (x + h)?;
        hooks.record(&format!("convnext.{idx}/Add_output_0"), &out)?;
        // Mul_3: mask after residual. The next block consumes a masked tensor.
        let out = out.broadcast_mul(mask)?;
        Ok(out)
    }
}

// =========================== Attention encoder =============================

struct AttnLayer {
    conv_q: Conv1d,
    conv_k: Conv1d,
    conv_v: Conv1d,
    conv_o: Conv1d,
    emb_rel_k: Tensor, // [1, 2W+1, head_dim]
    emb_rel_v: Tensor,
}

impl AttnLayer {
    fn load(vb: VarBuilder) -> Result<Self> {
        let make_kqv = |name: &str| -> Result<Conv1d> {
            candle_nn::conv1d(
                HIDDEN_DIM, HIDDEN_DIM, 1,
                Conv1dConfig { padding: 0, stride: 1, dilation: 1, groups: 1,
                    ..Default::default() },
                vb.pp(name),
            ).with_context(|| format!("attn.{name}"))
        };
        let conv_q = make_kqv("conv_q")?;
        let conv_k = make_kqv("conv_k")?;
        let conv_v = make_kqv("conv_v")?;
        let conv_o = make_kqv("conv_o")?;
        let emb_rel_k = vb
            .get((1, 2 * WINDOW_SIZE + 1, HEAD_DIM), "emb_rel_k")
            .context("emb_rel_k")?;
        let emb_rel_v = vb
            .get((1, 2 * WINDOW_SIZE + 1, HEAD_DIM), "emb_rel_v")
            .context("emb_rel_v")?;
        Ok(Self { conv_q, conv_k, conv_v, conv_o, emb_rel_k, emb_rel_v })
    }

    /// `x`: [B, 64, T]. `attn_mask`: [B, 1, T, T] (1 where attended, 0 where masked).
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
        // [B, 64, T] -> [B, n_heads, head_dim, T] -> [B, n_heads, T, head_dim]
        let split_heads = |t_: &Tensor| -> candle_core::Result<Tensor> {
            t_.reshape((b, N_HEADS, HEAD_DIM, t))?.transpose(2, 3)?.contiguous()
        };
        let q = split_heads(&q)?;
        let k = split_heads(&k)?;
        let v = split_heads(&v)?;

        let inv_sqrt_d = (HEAD_DIM as f64).sqrt().recip();
        // scores = (Q / sqrt(d)) @ K^T  ->  [B, H, T, T]
        let q_scaled = (q.clone() * inv_sqrt_d)?.contiguous()?;
        let kt = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let scores = q_scaled.matmul(&kt)?;
        hooks.record(&format!("attn_layers.{idx}/MatMul_output_0"), &scores)?;

        // K-side relative positions:
        //   pad emb_rel_k from [1, 2W+1, D] to [1, 2T-1, D] (with zeros where
        //   the window doesn't reach), then matmul Q_scaled @ ext^T to get
        //   [B, H, T, 2T-1] of relative scores. Then a pad-+-reshape "shift"
        //   turns that into [B, H, T, T] aligned with `scores`.
        let ext_k = get_relative_embeddings(&self.emb_rel_k, t)?.contiguous()?;
        let ext_kt = ext_k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let rel_logits = q_scaled.broadcast_matmul(&ext_kt)?;
        hooks.record(&format!("attn_layers.{idx}/MatMul_1_output_0"), &rel_logits)?;
        let scores_local = relative_position_to_absolute_position(&rel_logits)?;
        let scores = (scores + scores_local)?;
        hooks.record(&format!("attn_layers.{idx}/Add_2_output_0"), &scores)?;

        // Apply mask. ONNX uses -1e4 for masked positions so softmax
        // produces ~0. Compute `scores + (-1e4) * (1 - mask)`, which is
        // identity where mask=1 and -1e4 where mask=0. Cheaper than
        // `where_cond` (which would need an integer-typed condition).
        let mask_b = attn_mask.broadcast_as(scores.shape())?;
        let bias = mask_b.affine(-1.0, 1.0)?.affine(-1e4, 0.0)?;
        let scores = (scores + bias)?;
        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
        hooks.record(&format!("attn_layers.{idx}/Softmax_output_0"), &attn)?;

        // out = attn @ V
        let out_av = attn.matmul(&v)?;
        hooks.record(&format!("attn_layers.{idx}/MatMul_2_output_0"), &out_av)?;

        // V-side relative positions: shift attn from absolute to relative,
        // then matmul with extended emb_rel_v.
        let rel_weights = absolute_position_to_relative_position(&attn)?.contiguous()?;
        let ext_v = get_relative_embeddings(&self.emb_rel_v, t)?.contiguous()?;
        let out_rv = rel_weights.broadcast_matmul(&ext_v)?;
        hooks.record(&format!("attn_layers.{idx}/MatMul_3_output_0"), &out_rv)?;
        let out = (out_av + out_rv)?;
        hooks.record(&format!("attn_layers.{idx}/Add_4_output_0"), &out)?;

        // Merge heads: [B, H, T, head_dim] -> [B, H, head_dim, T] -> [B, 64, T]
        let merged = out.transpose(2, 3)?.contiguous()?.reshape((b, HIDDEN_DIM, t))?;
        let o = self.conv_o.forward(&merged)?;
        hooks.record(&format!("attn_layers.{idx}/conv_o/Conv_output_0"), &o)?;
        Ok(o)
    }
}

/// Pad/slice `rel_emb` ([1, 2W+1, D]) to [1, 2T-1, D] zero-padded where the
/// window doesn't reach. Matches VITS `_get_relative_embeddings`.
fn get_relative_embeddings(rel_emb: &Tensor, t: usize) -> candle_core::Result<Tensor> {
    let max_rel = 2 * WINDOW_SIZE + 1;
    let want = 2 * t - 1;
    let pad_len = (t as isize - (WINDOW_SIZE as isize + 1)).max(0) as usize;
    let slice_start = ((WINDOW_SIZE as isize + 1) - t as isize).max(0) as usize;
    let slice_end = slice_start + want;
    let padded = if pad_len > 0 {
        // Pad along dim 1 (the position dim) with zeros on both sides.
        let device = rel_emb.device();
        let dtype = rel_emb.dtype();
        let (_b, _, d) = rel_emb.dims3()?;
        let zeros = Tensor::zeros((1, pad_len, d), dtype, device)?;
        Tensor::cat(&[&zeros, rel_emb, &zeros], 1)?
    } else {
        rel_emb.clone()
    };
    let len_padded = padded.dim(1)?;
    let _ = max_rel; // suppress unused
    let _ = len_padded;
    padded.narrow(1, slice_start, want)
}

/// Supertonic's variant of VITS `_relative_position_to_absolute_position`.
/// `x`: [B, H, T, 2T-1]  ->  [B, H, T, T]
///
/// The differences from canonical VITS: the second pad is on the RIGHT,
/// not the left. Verified empirically by diffing against the ort golden
/// (rel scores at attn_layers.0/Add_2 became bit-exact only with this
/// pad direction).
fn relative_position_to_absolute_position(x: &Tensor) -> candle_core::Result<Tensor> {
    let (b, h, t, two_t_minus_1) = x.dims4()?;
    debug_assert_eq!(two_t_minus_1, 2 * t - 1);
    let x = x.pad_with_zeros(D::Minus1, 0, 1)?;            // right-pad by 1 -> [B,H,T,2T]
    let x = x.reshape((b, h, t * 2 * t))?;                  // flatten -> [B,H,T*2T]
    let x = x.pad_with_zeros(D::Minus1, 0, t - 1)?;         // right-pad by T-1
    let x = x.reshape((b, h, t + 1, 2 * t - 1))?;           // -> [B,H,T+1,2T-1]
    let x = x.narrow(2, 0, t)?.narrow(3, t - 1, t)?;        // -> [B,H,T,T]
    Ok(x)
}

/// VITS `_absolute_position_to_relative_position`.
/// `x`: [B, H, T, T]  ->  [B, H, T, 2T-1]
fn absolute_position_to_relative_position(x: &Tensor) -> candle_core::Result<Tensor> {
    let (b, h, t, t2) = x.dims4()?;
    debug_assert_eq!(t, t2);
    // 1. Right-pad last dim by T-1: [B, H, T, 2T-1]
    let x = x.pad_with_zeros(D::Minus1, 0, t - 1)?;
    // 2. Flatten last two dims: [B, H, T * (2T-1)]
    let x = x.reshape((b, h, t * (2 * t - 1)))?;
    // 3. Left-pad by T: [B, H, T * (2T-1) + T]
    let x = x.pad_with_zeros(D::Minus1, t, 0)?;
    // 4. Reshape to [B, H, T, 2T]
    let x = x.reshape((b, h, t, 2 * t))?;
    // 5. Drop first column: [..., 1:] -> [B, H, T, 2T-1]
    let x = x.narrow(3, 1, 2 * t - 1)?;
    Ok(x)
}

// =========================== FFN inside attn_encoder =======================
// VITS-style FFN: two Conv1d k=1 with GELU in between.

struct FfnLayer {
    conv_1: Conv1d,
    conv_2: Conv1d,
}

impl FfnLayer {
    fn load(vb: VarBuilder) -> Result<Self> {
        let conv_1 = candle_nn::conv1d(
            HIDDEN_DIM, INTERMEDIATE_DIM, 1,
            Conv1dConfig { padding: 0, stride: 1, dilation: 1, groups: 1,
                ..Default::default() },
            vb.pp("conv_1"),
        ).context("ffn.conv_1")?;
        let conv_2 = candle_nn::conv1d(
            INTERMEDIATE_DIM, HIDDEN_DIM, 1,
            Conv1dConfig { padding: 0, stride: 1, dilation: 1, groups: 1,
                ..Default::default() },
            vb.pp("conv_2"),
        ).context("ffn.conv_2")?;
        Ok(Self { conv_1, conv_2 })
    }

    /// VITS-style FFN with ReLU (not GELU), and mask multiplications
    /// before conv_1, between Relu and conv_2, and after conv_2 (matches
    /// the ONNX `Mul -> conv_1 -> Relu -> Mul_1 -> conv_2 -> Mul_2`
    /// pattern in the DP graph).
    fn forward(
        &self, x: &Tensor, mask: &Tensor, idx: usize, hooks: &mut Hooks,
    ) -> candle_core::Result<Tensor> {
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

// =========================== AttnEncoder (stack) ===========================

struct AttnEncoder {
    attn_layers: Vec<AttnLayer>,
    ffn_layers: Vec<FfnLayer>,
    norm_layers_1: Vec<LayerNorm>, // before attn
    norm_layers_2: Vec<LayerNorm>, // before ffn
}

impl AttnEncoder {
    fn load(vb: VarBuilder) -> Result<Self> {
        let mut attn_layers = Vec::with_capacity(N_ATTN_LAYERS);
        let mut ffn_layers = Vec::with_capacity(N_ATTN_LAYERS);
        let mut norm_layers_1 = Vec::with_capacity(N_ATTN_LAYERS);
        let mut norm_layers_2 = Vec::with_capacity(N_ATTN_LAYERS);
        for i in 0..N_ATTN_LAYERS {
            attn_layers.push(
                AttnLayer::load(vb.pp("attn_layers").pp(&i.to_string()))
                    .with_context(|| format!("attn_layers[{i}]"))?,
            );
            ffn_layers.push(
                FfnLayer::load(vb.pp("ffn_layers").pp(&i.to_string()))
                    .with_context(|| format!("ffn_layers[{i}]"))?,
            );
            norm_layers_1.push(
                candle_nn::layer_norm(
                    HIDDEN_DIM, LN_EPS,
                    vb.pp("norm_layers_1").pp(&i.to_string()).pp("norm"),
                ).with_context(|| format!("norm_layers_1[{i}]"))?,
            );
            norm_layers_2.push(
                candle_nn::layer_norm(
                    HIDDEN_DIM, LN_EPS,
                    vb.pp("norm_layers_2").pp(&i.to_string()).pp("norm"),
                ).with_context(|| format!("norm_layers_2[{i}]"))?,
            );
        }
        Ok(Self { attn_layers, ffn_layers, norm_layers_1, norm_layers_2 })
    }

    /// Post-norm transformer:
    ///   for each block:
    ///     y = x + attn(x);     y = norm_1(y)     [transpose-LN-transpose]
    ///     z = y + ffn(y);      z = norm_2(z)
    ///   final = z (* mask)
    ///
    /// `x` is assumed already masked.
    fn forward(
        &self, x: &Tensor, mask: &Tensor, hooks: &mut Hooks,
    ) -> candle_core::Result<Tensor> {
        // Build attn_mask [B, 1, T, T] from mask [B, 1, T].
        let (b, _, t) = mask.dims3()?;
        let m_q = mask.reshape((b, t, 1))?;
        let m_k = mask.reshape((b, 1, t))?;
        let attn_mask = m_q.broadcast_matmul(&m_k)?.reshape((b, 1, t, t))?;

        let mut h = x.clone();
        for i in 0..N_ATTN_LAYERS {
            // Attn sublayer: residual + post-norm.
            let attn_out = self.attn_layers[i].forward(&h, &attn_mask, i, hooks)?;
            h = (h + attn_out)?;
            hooks.record(&format!("attn_encoder/Add_{}_output_0", if i == 0 { 0 } else { 2 }), &h)?;
            let h_nhwc = h.transpose(1, 2)?.contiguous()?;
            let h_norm = self.norm_layers_1[i].forward(&h_nhwc)?;
            hooks.record(
                &format!("norm_layers_1.{i}/norm/LayerNormalization_output_0"),
                &h_norm,
            )?;
            h = h_norm.transpose(1, 2)?.contiguous()?;

            // FFN sublayer: residual + post-norm.
            let ffn_out = self.ffn_layers[i].forward(&h, mask, i, hooks)?;
            h = (h + ffn_out)?;
            hooks.record(&format!("attn_encoder/Add_{}_output_0", 2 * i + 1), &h)?;
            let h_nhwc = h.transpose(1, 2)?.contiguous()?;
            let h_norm = self.norm_layers_2[i].forward(&h_nhwc)?;
            hooks.record(
                &format!("norm_layers_2.{i}/norm/LayerNormalization_output_0"),
                &h_norm,
            )?;
            h = h_norm.transpose(1, 2)?.contiguous()?;
        }
        // Final mask multiplication (corresponds to /attn_encoder/Mul_2).
        h.broadcast_mul(mask)
    }
}

// =========================== Predictor MLP =================================

struct Predictor {
    linear0: Linear,
    linear1: Linear,
    prelu_alpha: Tensor, // shape [1]
}

impl Predictor {
    fn load(vb: VarBuilder) -> Result<Self> {
        let linear0 = candle_nn::linear(PREDICTOR_IN_DIM, PREDICTOR_HDIM, vb.pp("layers.0"))
            .context("predictor.layers.0")?;
        let linear1 = candle_nn::linear(PREDICTOR_HDIM, 1, vb.pp("layers.1"))
            .context("predictor.layers.1")?;
        let prelu_alpha = vb.pp("activation").get(1, "weight").context("activation.weight")?;
        Ok(Self { linear0, linear1, prelu_alpha })
    }

    fn forward(&self, x: &Tensor, hooks: &mut Hooks) -> candle_core::Result<Tensor> {
        let h = self.linear0.forward(x)?;
        hooks.record("layers.0/Gemm_output_0", &h)?;
        // PReLU with shared alpha. alpha is a 1-element tensor.
        let h = prelu_shared(&h, &self.prelu_alpha)?;
        hooks.record("activation/PRelu_output_0", &h)?;
        let h = self.linear1.forward(&h)?;
        hooks.record("layers.1/Gemm_output_0", &h)?;
        Ok(h)
    }
}

// =========================== Glue: DP forward =============================

pub struct CandleDurationPredictor {
    char_embed: Tensor, // [CHAR_VOCAB, HIDDEN_DIM]
    sentence_token: Tensor, // [1, 64, 1]
    convnext: Vec<ConvNextBlock>,
    attn_encoder: AttnEncoder,
    proj_out: Conv1d,
    predictor: Predictor,
}

impl CandleDurationPredictor {
    pub fn load(safetensors: &Path, device: &Device) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[safetensors], DType::F32, device)
                .context("mmap dp safetensors")?
        };
        let dp = vb.pp("tts").pp("dp");
        let se = dp.pp("sentence_encoder");
        let char_embed = se
            .pp("text_embedder")
            .pp("char_embedder")
            .get((CHAR_VOCAB, HIDDEN_DIM), "weight")
            .context("char_embedder.weight")?;
        let sentence_token = se.get((1, HIDDEN_DIM, 1), "sentence_token").context("sentence_token")?;

        let mut convnext = Vec::with_capacity(N_CONVNEXT);
        for i in 0..N_CONVNEXT {
            convnext.push(
                ConvNextBlock::load(se.pp("convnext").pp("convnext").pp(&i.to_string()))
                    .with_context(|| format!("convnext[{i}]"))?,
            );
        }
        let attn_encoder = AttnEncoder::load(se.pp("attn_encoder")).context("attn_encoder")?;
        // proj_out: stored at sentence_encoder.proj_out.net.weight, k=1, no bias.
        let proj_out = {
            let w = se.pp("proj_out").pp("net").get((HIDDEN_DIM, HIDDEN_DIM, 1), "weight")
                .context("proj_out.net.weight")?;
            Conv1d::new(w, None, Conv1dConfig { padding: 0, stride: 1, dilation: 1, groups: 1,
                ..Default::default() })
        };
        let predictor = Predictor::load(dp.pp("predictor"))?;
        Ok(Self { char_embed, sentence_token, convnext, attn_encoder, proj_out, predictor })
    }

    /// `text_ids`: [B, T] i64.  `style_dp`: [B, 8, 16].  `text_mask`: [B, 1, T].
    /// Returns duration in seconds, shape [B] (1 value per sample).
    pub fn forward(
        &self,
        text_ids: &Tensor,
        style_dp: &Tensor,
        text_mask: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let mut hooks = Hooks::default();
        self.forward_with_hooks(text_ids, style_dp, text_mask, &mut hooks)
    }

    pub fn forward_with_hooks(
        &self,
        text_ids: &Tensor,
        style_dp: &Tensor,
        text_mask: &Tensor,
        hooks: &mut Hooks,
    ) -> candle_core::Result<Tensor> {
        let (b, t) = text_ids.dims2()?;
        // 1. Embed: [B, T] (i64) -> [B*T] -> embedding -> [B*T, 64] -> [B, T, 64].
        //    candle's `embedding` requires 1-D indices, so flatten + reshape.
        let ids_flat = text_ids.flatten_all()?;
        let emb_flat = self.char_embed.embedding(&ids_flat)?; // [B*T, 64]
        let emb = emb_flat.reshape((b, t, HIDDEN_DIM))?;
        let emb = emb.transpose(1, 2)?.contiguous()?; // [B, 64, T]
        // Zero out padding positions: text_mask is [B, 1, T] of 1/0.
        let emb_masked = emb.broadcast_mul(text_mask)?;
        // 2. Prepend sentence_token: [B, 64, 1] from broadcast of [1, 64, 1].
        let st = self.sentence_token.broadcast_as((b, HIDDEN_DIM, 1))?.contiguous()?;
        let mixed = Tensor::cat(&[st, emb_masked], D::Minus1)?; // [B, 64, T+1]
        // Note: golden's "Add_output_0" is the OUTER residual (after attn_encoder),
        // not this mixed embedding -- we record it after the residual below.
        // Extended mask for the sentence_token position (always valid).
        let ones = Tensor::ones((b, 1, 1), text_mask.dtype(), text_mask.device())?;
        let mask_ext = Tensor::cat(&[ones, text_mask.clone()], D::Minus1)?; // [B, 1, T+1]

        // 3. ConvNeXt × 6. Each block consumes and returns a masked tensor.
        let mut h = mixed.broadcast_mul(&mask_ext)?;
        for (i, block) in self.convnext.iter().enumerate() {
            h = block.forward(&h, &mask_ext, i, hooks)?;
        }
        let convnext_out = h.clone();

        // 4. AttnEncoder, then outer residual: final = convnext_out + attn(convnext_out).
        let attn_out = self.attn_encoder.forward(&convnext_out, &mask_ext, hooks)?;
        let h = (convnext_out + attn_out)?;
        hooks.record("Add_output_0", &h)?;

        // 5. Slice the first time-step (the sentence_token slot) and proj_out.
        let h_first = h.narrow(D::Minus1, 0, 1)?.contiguous()?; // [B, 64, 1]
        let h_first = self.proj_out.forward(&h_first)?; // [B, 64, 1]
        hooks.record("proj_out/net/Conv_output_0", &h_first)?;
        let sentence_vec = h_first.squeeze(D::Minus1)?; // [B, 64]

        // 6. Flatten style_dp -> concat -> Predictor.
        let style_flat = style_dp.reshape((b, STYLE_FLAT_DIM))?;
        let combined = Tensor::cat(&[sentence_vec, style_flat], D::Minus1)?; // [B, 192]
        let h = self.predictor.forward(&combined, hooks)?; // [B, 1]
        // ONNX has `Exp` after the final Gemm so the predictor learns
        // log-duration. exp() guarantees a positive number of seconds.
        let duration = h.exp()?.squeeze(D::Minus1)?; // [B]
        hooks.record("duration", &duration)?;
        Ok(duration)
    }
}

// =========================== Helpers ======================================

/// Edge ("replicate") pad on the last axis. `left` copies of x[..., 0] on
/// the left, `right` copies of x[..., -1] on the right.
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

fn prelu_shared(x: &Tensor, alpha: &Tensor) -> candle_core::Result<Tensor> {
    // alpha is shape [1] -> broadcast to anything.
    let zero = Tensor::zeros_like(x)?;
    let pos = x.maximum(&zero)?;
    let neg = x.minimum(&zero)?;
    pos + neg.broadcast_mul(alpha)?
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
