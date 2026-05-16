//! Candle port of the Supertonic 3 vocoder.
//!
//! Architecture (derived from ONNX graph walk + tts.json):
//!
//! ```text
//! latent [B, 144, T_lat]
//!   ÷ normalizer.scale            (scalar from tts.ttl.normalizer.scale)
//!   ► unfold [B, 144, T] -> [B, 24, T*6]  (chunk_compress_factor = 6)
//!   * latent_std + latent_mean    (both [1, 24, 1], inverse-AE)
//!   ► pad + Conv1d(24 -> 512, k=7)        (input proj, weights named onnx::Conv_1441/2)
//!   ► 10× ConvNeXt block with dilations [1,2,4,1,2,4,1,1,1,1]
//!   ► BatchNorm1d(512)
//!   ► PReLU(alpha = onnx::PRelu_1506)
//!   ► pad + Conv1d(512 -> 2048, k=3)      (head.layer1)
//!   ► PReLU
//!   ► Conv1d(2048 -> 512, k=1, no bias)   (head.layer2)
//!   ► transpose + reshape -> wav [B, T*6 * 512]   (= e.g. 64512 samples)
//! ```
//!
//! Each ConvNeXt block:
//! ```text
//!   h = dwconv(x)                          (depthwise Conv1d, k=7, dilation=d)
//!   h = LayerNorm(h)                       (over channel dim, weight + bias [512])
//!   h = pwconv1(h)                         (Conv1d 512 -> 2048, k=1)
//!   h = GELU(h)
//!   h = pwconv2(h)                         (Conv1d 2048 -> 512, k=1)
//!   h = h * gamma                          (learnable [1, 512, 1] scale)
//!   return x + h                           (residual)
//! ```

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{
    BatchNorm, BatchNormConfig, Conv1d, Conv1dConfig, LayerNorm, ModuleT, VarBuilder,
};
use std::path::Path;

/// Fixed architecture constants from `tts.json`. We don't read them
/// dynamically because they're load-bearing for the candle code below
/// — changing them silently would shift weight shapes and produce
/// quiet numerical garbage rather than a loud error.
const INPUT_DIM: usize = 24; // ae.encoder.idim
const HIDDEN_DIM: usize = 512; // ae.decoder.hdim
const INTERMEDIATE_DIM: usize = 2048; // ae.decoder.intermediate_dim
const HEAD_INTERMEDIATE_DIM: usize = 2048; // ae.decoder.head.hdim
const KSZ_INIT: usize = 7;
const KSZ_CONVNEXT: usize = 7;
const HEAD_KSZ: usize = 3;
const N_CONVNEXT_LAYERS: usize = 10;
const DILATIONS: [usize; 10] = [1, 2, 4, 1, 2, 4, 1, 1, 1, 1];
const CHUNK_COMPRESS_FACTOR: usize = 6;
/// BatchNorm1d default eps from PyTorch.
const BN_EPS: f64 = 1e-5;
/// ONNX `LayerNormalization` op default `epsilon = 1e-6` (not PyTorch's
/// 1e-5). Using the wrong value compounds into ~1e-2 wav error.
const LN_EPS: f64 = 1e-6;

/// One ConvNeXt block: dwconv → LayerNorm → pwconv1 → GELU → pwconv2 → γ → residual.
struct ConvNextBlock {
    dwconv: Conv1d,
    norm: LayerNorm,
    pwconv1: Conv1d,
    pwconv2: Conv1d,
    gamma: Tensor, // [1, HIDDEN_DIM, 1]
    dilation: usize,
}

impl ConvNextBlock {
    fn load(vb: VarBuilder, dilation: usize) -> Result<Self> {
        // dwconv is depthwise — groups = HIDDEN_DIM. ONNX stored it as
        // [HIDDEN_DIM, 1, KSZ_CONVNEXT].
        let dwconv = candle_nn::conv1d(
            HIDDEN_DIM,
            HIDDEN_DIM,
            KSZ_CONVNEXT,
            Conv1dConfig {
                padding: 0, // explicit Pad below; matches ONNX's external Pad op
                stride: 1,
                dilation,
                groups: HIDDEN_DIM,
                ..Default::default()
            },
            vb.pp("dwconv.net"),
        )
        .context("dwconv")?;
        let norm =
            candle_nn::layer_norm(HIDDEN_DIM, LN_EPS, vb.pp("norm.norm")).context("norm")?;
        let pwconv1 = candle_nn::conv1d(
            HIDDEN_DIM,
            INTERMEDIATE_DIM,
            1,
            Conv1dConfig {
                padding: 0,
                stride: 1,
                dilation: 1,
                groups: 1,
                ..Default::default()
            },
            vb.pp("pwconv1"),
        )
        .context("pwconv1")?;
        let pwconv2 = candle_nn::conv1d(
            INTERMEDIATE_DIM,
            HIDDEN_DIM,
            1,
            Conv1dConfig {
                padding: 0,
                stride: 1,
                dilation: 1,
                groups: 1,
                ..Default::default()
            },
            vb.pp("pwconv2"),
        )
        .context("pwconv2")?;
        let gamma = vb
            .get((1, HIDDEN_DIM, 1), "gamma")
            .context("gamma")?;
        Ok(Self {
            dwconv,
            norm,
            pwconv1,
            pwconv2,
            gamma,
            dilation,
        })
    }

    fn forward_hooked(
        &self,
        x: &Tensor,
        idx: usize,
        hooks: &mut Hooks,
    ) -> candle_core::Result<Tensor> {
        // Causal (all-left) pad: matches the ONNX `Pad mode=edge` with
        // `pads = [0,0,d*(k-1), 0,0,0]` pattern used throughout the
        // vocoder. Splitting the pad symmetrically would silently shift
        // the audio by ~half a kernel.
        let pad = self.dilation * (KSZ_CONVNEXT - 1);
        let h = pad_temporal(x, pad, 0)?;
        let h = self.dwconv.forward(&h)?;
        hooks.record(&format!("convnext.{idx}/dwconv/net/Conv_output_0"), &h)?;
        // LayerNorm across channel dim: [B, C, T] -> [B, T, C], norm, swap back.
        let h_nhwc = h.transpose(1, 2)?.contiguous()?;
        let h_ln = self.norm.forward(&h_nhwc)?;
        hooks.record(
            &format!("convnext.{idx}/norm/norm/LayerNormalization_output_0"),
            &h_ln,
        )?;
        let h = h_ln.transpose(1, 2)?;
        let h = self.pwconv1.forward(&h)?;
        hooks.record(&format!("convnext.{idx}/pwconv1/Conv_output_0"), &h)?;
        let h = gelu_erf(&h)?;
        let h = self.pwconv2.forward(&h)?;
        hooks.record(&format!("convnext.{idx}/pwconv2/Conv_output_0"), &h)?;
        let h = h.broadcast_mul(&self.gamma)?;
        let out = (x + h)?;
        hooks.record(&format!("convnext.{idx}/Add_output_0"), &out)?;
        Ok(out)
    }
}

/// Edge ("replicate") padding along the time (last) axis. The first and
/// last time-step are repeated `left` and `right` times respectively.
/// All Conv/dwconv pads in the Supertonic 3 ONNX graph use mode=edge,
/// not zero — getting this wrong silently corrupts the first output
/// frame of every Conv.
fn pad_temporal(x: &Tensor, left: usize, right: usize) -> candle_core::Result<Tensor> {
    if left == 0 && right == 0 {
        return Ok(x.clone());
    }
    let t = x.dim(D::Minus1)?;
    let mut parts: Vec<Tensor> = Vec::with_capacity(3);
    if left > 0 {
        let first = x.narrow(D::Minus1, 0, 1)?;
        // Repeat the [..., 1] slice `left` times along the time axis.
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

/// LayerNorm over the last dim of `x`, scaled and shifted by `weight` and
/// `bias` (both 1-D of size `x.last_dim`). Matches PyTorch nn.LayerNorm.
fn layer_norm_last_dim(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f64,
) -> candle_core::Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let centered = x.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
    let denom = (var + eps)?.sqrt()?;
    let normed = centered.broadcast_div(&denom)?;
    normed.broadcast_mul(weight)?.broadcast_add(bias)
}

/// GELU using the exact erf formula (matches PyTorch's default
/// `F.gelu(approximate='none')` and the `Div/Erf/Add/Mul` chain ONNX
/// emits).
///
/// gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
fn gelu_erf(x: &Tensor) -> candle_core::Result<Tensor> {
    let inv_sqrt2 = (2.0f64.sqrt()).recip();
    let half = Tensor::new(0.5f32, x.device())?.to_dtype(x.dtype())?;
    let one = Tensor::new(1.0f32, x.device())?.to_dtype(x.dtype())?;
    let scale = Tensor::new(inv_sqrt2 as f32, x.device())?.to_dtype(x.dtype())?;
    let erf_arg = x.broadcast_mul(&scale)?.erf()?;
    let inner = erf_arg.broadcast_add(&one)?;
    let scaled = x.broadcast_mul(&half)?;
    scaled.broadcast_mul(&inner)
}

fn load_batch_norm(vb: VarBuilder, channels: usize, eps: f64) -> Result<BatchNorm> {
    candle_nn::batch_norm(
        channels,
        BatchNormConfig { eps, ..Default::default() },
        vb,
    )
    .context("batch_norm")
}

/// PReLU with a single shared alpha (shape [1, 1] in ONNX, broadcast over
/// everything else).
fn prelu(x: &Tensor, alpha: &Tensor) -> candle_core::Result<Tensor> {
    let zero = Tensor::zeros_like(x)?;
    let pos = x.maximum(&zero)?;
    let neg = x.minimum(&zero)?;
    // alpha is [1, 1] from ONNX; reshape to [1, 1, 1] so it broadcasts over time.
    let alpha = alpha.reshape((1, 1, 1))?;
    pos + neg.broadcast_mul(&alpha)?
}

pub struct CandleVocoder {
    normalizer_scale: f32,
    latent_mean: Tensor, // [1, 24, 1]
    latent_std: Tensor,
    embed: Conv1d,             // input proj
    blocks: Vec<ConvNextBlock>,
    final_norm: BatchNorm,
    head_prelu_alpha: Tensor,
    head_layer1: Conv1d,
    head_layer2: Conv1d, // no bias
    device: Device,
}

impl CandleVocoder {
    pub fn load(safetensors_path: &Path, device: &Device) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[safetensors_path], DType::F32, device)
                .context("mmap vocoder.safetensors")?
        };

        // Constants live under tts.* prefixes — pp() through them.
        let ae = vb.pp("tts").pp("ae");
        let dec = ae.pp("decoder");
        let ttl = vb.pp("tts").pp("ttl");

        let normalizer_scale = {
            let scale_t = ttl
                .pp("normalizer")
                .get((), "scale")
                .context("normalizer.scale")?;
            scale_t.to_dtype(DType::F32)?.to_scalar::<f32>()?
        };
        let latent_mean = ae.get((1, INPUT_DIM, 1), "latent_mean").context("latent_mean")?;
        let latent_std = ae.get((1, INPUT_DIM, 1), "latent_std").context("latent_std")?;

        // The input projection was renamed by the exporter to onnx::Conv_1441/2.
        // We materialize a Conv1d "by hand" because candle_nn::conv1d() expects
        // weight under "weight" / bias under "bias", and these don't follow that.
        let embed = {
            let w = vb.get((HIDDEN_DIM, INPUT_DIM, KSZ_INIT), "onnx::Conv_1441")?;
            let b = vb.get(HIDDEN_DIM, "onnx::Conv_1442")?;
            Conv1d::new(
                w,
                Some(b),
                Conv1dConfig {
                    padding: 0,
                    stride: 1,
                    dilation: 1,
                    groups: 1,
                ..Default::default()
                },
            )
        };

        let mut blocks = Vec::with_capacity(N_CONVNEXT_LAYERS);
        for (i, &d) in DILATIONS.iter().enumerate() {
            let bvb = dec.pp("convnext").pp(&i.to_string());
            blocks.push(ConvNextBlock::load(bvb, d).with_context(|| format!("convnext[{i}]"))?);
        }
        let final_norm = load_batch_norm(
            dec.pp("final_norm").pp("norm"),
            HIDDEN_DIM,
            BN_EPS,
        )?;
        let head_prelu_alpha = vb.get((1, 1), "onnx::PRelu_1506").context("PRelu alpha")?;
        let head_layer1 = candle_nn::conv1d(
            HIDDEN_DIM,
            HEAD_INTERMEDIATE_DIM,
            HEAD_KSZ,
            Conv1dConfig {
                padding: 0,
                stride: 1,
                dilation: 1,
                groups: 1,
                ..Default::default()
            },
            dec.pp("head").pp("layer1").pp("net"),
        )
        .context("head.layer1")?;
        // head.layer2 has no bias in ONNX (only `.weight`). Load by hand.
        let head_layer2 = {
            let w = dec
                .pp("head")
                .pp("layer2")
                .get((HIDDEN_DIM, HEAD_INTERMEDIATE_DIM, 1), "weight")
                .context("head.layer2 weight")?;
            Conv1d::new(
                w,
                None,
                Conv1dConfig {
                    padding: 0,
                    stride: 1,
                    dilation: 1,
                    groups: 1,
                ..Default::default()
                },
            )
        };
        Ok(Self {
            normalizer_scale,
            latent_mean,
            latent_std,
            embed,
            blocks,
            final_norm,
            head_prelu_alpha,
            head_layer1,
            head_layer2,
            device: device.clone(),
        })
    }

    /// Run the vocoder. `latent` shape: `[B, 144, T_latent]`. Returns
    /// `[B, T_latent * 6 * 512]` audio samples at 44.1 kHz.
    pub fn forward(&self, latent: &Tensor) -> candle_core::Result<Tensor> {
        let mut hooks = Hooks::default();
        self.forward_with_hooks(latent, &mut hooks)
    }

    /// Variant that also records named intermediates into `hooks` for the
    /// per-layer diff harness. Names match the ONNX node-output keys in
    /// `tmp/golden.safetensors` so a check binary can directly compare.
    pub fn forward_with_hooks(
        &self,
        latent: &Tensor,
        hooks: &mut Hooks,
    ) -> candle_core::Result<Tensor> {
        // --- 1. de-normalize -------------------------------------------------
        let scaled = (latent / self.normalizer_scale as f64)?;
        // unfold [B, 144, T] -> [B, 24, 6, T] -> permute (0, 1, 3, 2) ->
        // [B, 24, T, 6] -> [B, 24, T*6]. PyTorch path: x.reshape(B, 24, 6, T)
        // produces channels grouped in the original 144 layout; the Transpose
        // we saw in ONNX swaps the inner (6, T) to (T, 6) before flattening.
        let (b, _, t) = scaled.dims3()?;
        let unfolded = scaled
            .reshape((b, INPUT_DIM, CHUNK_COMPRESS_FACTOR, t))?
            .permute((0, 1, 3, 2))?
            .reshape((b, INPUT_DIM, t * CHUNK_COMPRESS_FACTOR))?;
        let z = unfolded
            .broadcast_mul(&self.latent_std)?
            .broadcast_add(&self.latent_mean)?;

        hooks.record("pre_embed/after_unfold", &z)?;

        // --- 2. input projection: causal pad + Conv1d(24 -> 512, k=7) -------
        let z = pad_temporal(&z, KSZ_INIT - 1, 0)?;
        let mut h = self.embed.forward(&z)?;
        hooks.record("embed/net/Conv_output_0", &h)?;

        // --- 3. ConvNeXt stack ----------------------------------------------
        for (i, block) in self.blocks.iter().enumerate() {
            h = block.forward_hooked(&h, i, hooks)?;
        }

        // --- 4. final_norm (BatchNorm1d). Pass train=false explicitly so
        //        BatchNorm uses its stored running_mean / running_var.
        h = self.final_norm.forward_t(&h, false)?;
        hooks.record("final_norm/BatchNormalization_output_0", &h)?;

        // --- 5. head.layer1 (causal pad + Conv k=3, no activation before) --
        h = pad_temporal(&h, HEAD_KSZ - 1, 0)?;
        h = self.head_layer1.forward(&h)?;
        hooks.record("head/layer1/net/Conv_output_0", &h)?;

        // --- 6. PReLU + head.layer2 (Conv k=1, no bias) ---------------------
        h = prelu(&h, &self.head_prelu_alpha)?;
        hooks.record("head/act/PRelu_output_0", &h)?;
        h = self.head_layer2.forward(&h)?;
        hooks.record("head/layer2/Conv_output_0", &h)?;

        // --- 7. transpose + reshape -> wav ----------------------------------
        let (b, c, t) = h.dims3()?;
        let wav = h.transpose(1, 2)?.reshape((b, t * c))?;
        Ok(wav)
    }
}

/// Activation recorder used by per-layer validation. Stays empty in
/// production paths.
#[derive(Default)]
pub struct Hooks {
    pub records: Vec<(String, Tensor)>,
}

impl Hooks {
    fn record(&mut self, name: &str, t: &Tensor) -> candle_core::Result<()> {
        // Clone is cheap (just a refcount bump). We snapshot here so the
        // caller can mutate `t` in-place later without poisoning the dump.
        self.records.push((name.to_string(), t.clone()));
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.records.iter().find(|(n, _)| n == name).map(|(_, t)| t)
    }
}

// Bring the `i` helper from IndexOp into scope so we don't get unused-import warnings.
#[allow(dead_code)]
fn _unused_index_helper(x: &Tensor) -> candle_core::Result<Tensor> {
    x.i(0)
}
