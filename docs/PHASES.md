# Phases — how the candle port was built

Eight phases, each landed as a single `main` commit. Every phase's commit
message has the full bug list and resolution notes; this file is the
short-form index.

## Phase 0 — ONNX → safetensors weight extraction

- `tools/convert_weights.py` walks every initializer in each of the four
  Supertonic ONNX files and dumps it to a parallel `.safetensors`.
- `tools/verify_weights.py` byte-checks that every kept tensor matches the
  source ONNX initializer.
- **Subtle bug fixed:** name-based filters dropped two classes of real
  weights — `nn.Linear` weights renamed `onnx::MatMul_<N>` by the exporter,
  and small Expand outputs constant-folded to `/.../Expand_output_0`.
  Replaced with a dtype filter (`f32`/`f16`), which catches both.
- Output: **708** tensors, byte-identical, 375 MB.

## Phase 1 — Golden activation dump

- `tools/dump_golden.py` runs one deterministic synthesis ("Hello, world.",
  voice M1, seed 42) through all four ONNX models with `onnxruntime`,
  hooking every `LayerNormalization`, `Conv`, `MatMul`, `Softmax`, `Gemm`,
  `Add`, and `PRelu` as a graph output.
- Writes `tmp/golden.safetensors` (~114 MB, **906** tensors) — the oracle
  for every per-layer diff in Phases 2–5.

## Phase 2 — Vocoder in candle

- 10-block ConvNeXt stack + BatchNorm + 2-layer head + reshape.
- Per-layer diff ≤ 5e-4, final wav max-abs `3.3e-6`.
- **Bugs found** (committed reference for Phases 3–5):
  1. ONNX `Pad mode=edge`, **not** zero padding. Wrote an edge-pad helper.
  2. Padding is **causal** (all-left), not symmetric.
  3. No PReLU between BN and `head.layer1` — the activation only appears
     between layer1 and layer2.
  4. ONNX `LayerNormalization` defaults to `epsilon=1e-6`, not PyTorch's
     `1e-5`. Compounded to ~1e-2 wav error before fix.

## Phase 3 — Duration predictor in candle

- text_embedder + sentence_token concat + 6 ConvNeXt blocks + 2-layer VITS-
  style relative attention + small predictor MLP.
- **Bugs found:**
  1. DP FFN uses **ReLU**, not GELU.
  2. DP ConvNeXt uses **symmetric** edge pad (not the vocoder's causal pad).
  3. Per-block mask multiplications throughout (after dwconv, after
     residual, in/middle/out of FFN).
  4. AttnEncoder is **post-norm** with an outer residual:
     `final = convnext_out + (norm(z) * mask)`.
  5. `_relative_position_to_absolute_position` uses **right** pad for the
     second pad step (canonical VITS is left).
  6. Final `Exp` after the predictor MLP — the model emits log-duration.
     Missing the exp made all outputs 4× too small.

## Phase 4 — Text encoder in candle

- Mostly reuses Phase 3 patterns at wider dims (256 vs 64). New piece is
  `SpeechPromptedTextEncoder` (spte): 2-layer cross-attention conditioning
  text on the voice style.
- **Bugs found:**
  1. ConvNeXt has **dilated** dwconvs (dilations `[1, 1, 2, 2, 4, 4]`).
  2. spte attention has a **`tanh` on K** before the QK matmul. Non-
     standard scaled-dot-product.
  3. spte scale divisor is `sqrt(HIDDEN=256)=16`, not `sqrt(head_dim)=11.3`.
  4. spte residuals both **rebase on the original `q_in`**, not on the
     running stream: `h2 = q_in + attention2(h1)`, not `h1 + ...`.

## Phase 5 — Vector estimator in candle

The hardest port — 1004 ops, 315 weight tensors, classifier-free guidance,
RoPE text cross-attention, bundled ODE step. Per-layer diff ≤ 1.7e-6.

- **Bugs found** (nine non-obvious ones, each surfaced by adding more hooks
  and bisecting `v_cond` divergence):
  1. `attn.increments` is i64, dropped by the Phase-0 dtype filter. Just
     `[0..999]` — recompute via `Tensor::arange`.
  2. `attn.theta` is stored only at `main_blocks.3.attn.theta` but shared
     across all 4 text_cond layers.
  3. text_cond weights are asymmetric: Q/out_fc are `[512,512]`, K/V are
     `[256,512]` (text dim → backbone dim).
  4. The 36 hidden `onnx::MatMul_*` weights follow a clean per-block stride
     of 45 starting at 3384. Offsets within: `0=time, 6/7/8/15=text Q/K/V/
     out, 21/22/23/24=style Q/K/V/out`.
  5. Internal **classifier-free guidance**: graph duplicates `noisy_latent`
     to batch=2 and substitutes three different `uncond_masker` tokens
     (text_special, style_key_special, style_value_special) for the uncond
     half.
  6. Bundled ODE step:
     `denoised = (noisy + (4·v_cond − 3·v_uncond) / total_step) · mask`.
     CFG scale = 4 baked in.
  7. text_cond attention scale is `1/sqrt(text_dim=256)`, NOT `1/sqrt(
     HIDDEN)`. Uncond batch is bit-exact because its constant K masks the
     wrong scale.
  8. RoPE is **length-normalized**: `angle = (pos / mask_sum) · theta`. The
     `rotary_scale=10` from `tts.json` is a red herring; the real divisor
     is `ReduceSum(mask)`.
  9. style_cond K comes from a **learned `[1, 50, 256]` prototype** that
     was constant-folded to `/vector_estimator/Expand_output_0`, NOT from
     `style_ttl`. V uses `style_ttl`.

## Phase 6 — End-to-end candle Supertonic + speak_candle binary

- `CandleSupertonic` bundles the four candle sub-models behind the same
  external API as the ort-backed `Supertonic`.
- `speak_candle` mirrors `speak`'s CLI.
- **One more bug at integration time:** `TimeEncoder.inv_freq` was being
  computed from `1/1000^(2k/64)` (a guess from the `Constant_2=1000`
  scalar), but the actual table is stored as a non-canonical geometric
  sequence at `/.../Constant_3_output_0`. Now loads both constants
  directly from safetensors. The bug was invisible to `ve_check` because
  step 0 has t=0 and the inv_freq table doesn't matter when `phase=0`;
  the divergence compounded across the 8 ODE steps.
- ASR round-trip: 10/10 sentences verbatim.

## Phase 7 — Metal feature + benchmark

- Added a `metal` Cargo feature and a `--device cpu|metal|auto` flag on
  `speak_candle`. Required scattered `.contiguous()` calls before matmuls
  (the candle Metal kernel rejects strided inputs that the CPU kernel
  tolerates).
- Benchmark on a 5.65 s utterance: ort CPU 1.44 s (3.9×), candle CPU 3.37 s
  (1.7×), candle Metal 6.12 s (0.9×). Metal is slower because each ODE step
  dispatches ~thousands of small kernels and dispatch overhead dominates.
- Both backends ship; CPU is the recommended candle device.
