#!/usr/bin/env python3
"""Run one deterministic synthesis pass through all four Supertonic 3
ONNX models and save every input + key intermediate + final output to a
single .safetensors. The candle port loads this artifact and diffs its
own outputs against the reference at every layer boundary.

Determinism: the noisy_latent fed into vector_estimator is sampled from
numpy with `--seed` (default 42) so the candle side can reproduce the
identical noise. All other inputs are derived from `--text` + `--voice`
+ `--lang`, which the candle side mirrors via its own preprocessing.

The script bakes a coarse set of intermediate node outputs (per-block
LayerNorm + Conv exits, attention scores, final outputs of each
sub-module) into the dump. Pass `--detail full` to dump every named
non-trivial node output -- ~10x bigger, only useful when chasing a
single-layer bug.

Usage:
    python3 tools/dump_golden.py \\
        [--onnx-dir models/onnx] \\
        [--voice models/voice_styles/M1.json] \\
        [--text "Hello, world."] \\
        [--lang en] [--seed 42] [--steps 8] [--speed 1.0] \\
        [--detail coarse|full|minimal] \\
        [--out tmp/golden.safetensors]
"""
from __future__ import annotations
import argparse
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from safetensors.numpy import save_file


# Ops whose outputs are "interesting" at coarse granularity. These are
# module-exit points: LayerNorm closes a ConvNeXt block, Conv is the
# heart of the dwconv, MatMul+Add is a Linear projection, Softmax is
# the attention score peak.
COARSE_INTERESTING_OPS = {"LayerNormalization", "Conv", "MatMul", "Softmax", "Gemm"}
# "Full" detail adds Mul/Add/Sub/Erf/Tanh outputs too. Excluded by default
# because the count explodes (thousands per vector_estimator) and most
# such outputs aren't useful for diagnosing a layer-level bug.
FULL_EXTRA_OPS = {"Mul", "Add", "Sub", "Erf", "Tanh", "Where", "Softmax", "Transpose"}


def preprocess(text: str, lang: str) -> str:
    """Mirror of src/text.rs `preprocess` for the simple sentences we use
    here. Skips the emoji/quote folds (irrelevant for the test sentence)
    so the only meaningful op is NFKD + language tag wrap."""
    t = unicodedata.normalize("NFKD", text)
    if not t.endswith((".", "!", "?", ";", ":", ",", ")", "]", "}", '"', "'")):
        t = t + "."
    return f"<{lang}>{t}</{lang}>"


def encode_ids(processed: str, indexer_path: Path) -> np.ndarray:
    table = json.loads(indexer_path.read_text())
    ids = []
    for ch in processed:
        cp = ord(ch)
        if cp < len(table):
            ids.append(table[cp])
        else:
            ids.append(-1)
    return np.array([ids], dtype=np.int64)


def load_voice(voice_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(voice_path.read_text())
    def parse(comp):
        dims = comp["dims"]
        flat = np.array(comp["data"], dtype=np.float32).reshape(dims)
        return flat
    return parse(data["style_ttl"]), parse(data["style_dp"])


def add_intermediate_outputs(onnx_path: Path, detail: str) -> Path:
    """Rewrite an ONNX file to expose interesting node outputs as graph
    outputs. Returns a path to the rewritten file. We run shape
    inference first so each new ValueInfoProto carries the dtype/shape
    that onnxruntime requires for graph outputs."""
    model = onnx.load(str(onnx_path))
    if detail == "minimal":
        return onnx_path
    # Shape inference fills `value_info` with type/shape for every
    # internal edge. Without this, intermediates we expose as graph
    # outputs have UNDEFINED dtype and onnxruntime rejects the model.
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"  WARN: shape_inference failed on {onnx_path.name}: {e}", file=sys.stderr)
    value_info_by_name = {vi.name: vi for vi in model.graph.value_info}
    interesting = set(COARSE_INTERESTING_OPS)
    if detail == "full":
        interesting |= FULL_EXTRA_OPS

    existing_outputs = {o.name for o in model.graph.output}
    added = 0
    skipped_no_info = 0
    for node in model.graph.node:
        if node.op_type not in interesting:
            continue
        for out in node.output:
            if not out or out in existing_outputs:
                continue
            # Tag-style outputs (auto-generated) with no module-name
            # prefix tend to be intermediate plumbing; require a slash.
            if "/" not in out:
                continue
            vi = value_info_by_name.get(out)
            if vi is None:
                skipped_no_info += 1
                continue
            model.graph.output.append(vi)
            existing_outputs.add(out)
            added += 1
    if skipped_no_info:
        print(f"  ({onnx_path.name}: skipped {skipped_no_info} intermediates with no inferred type info)")
    tmp_path = onnx_path.with_suffix(f".dump-{detail}.onnx")
    onnx.save(model, str(tmp_path))
    return tmp_path


def short_name(full_name: str) -> str:
    """Take an ONNX node output name like
    '/vector_estimator/vector_field/main_blocks.5/attention/Softmax_output_0'
    and shorten it to 'main_blocks.5/attention/Softmax_output_0' for the
    safetensors key. Strips the model name and the redundant 'tts.X.X.'
    prefix while keeping enough context to identify the layer."""
    s = full_name.lstrip("/")
    # Drop the leading model component (vector_estimator, text_encoder, ...)
    if "/" in s:
        s = s.split("/", 1)[1]
    # Many names have a `vector_field/` or `text_encoder/` second segment
    # we can drop too; keep things human-scannable.
    for prefix in ("vector_field/", "text_encoder/", "sentence_encoder/", "ae/decoder/"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s


def run_model(
    name: str,
    onnx_path: Path,
    inputs: dict[str, np.ndarray],
    detail: str,
) -> dict[str, np.ndarray]:
    """Run a single ONNX model and return {short_key: tensor} for every
    declared graph output plus any intermediates we exposed."""
    augmented_path = add_intermediate_outputs(onnx_path, detail)
    sess = ort.InferenceSession(str(augmented_path), providers=["CPUExecutionProvider"])
    output_names = [o.name for o in sess.get_outputs()]
    outs = sess.run(output_names, inputs)
    result: dict[str, np.ndarray] = {}
    for n, arr in zip(output_names, outs):
        if arr.dtype not in (np.float32, np.float16, np.int64, np.int32):
            continue  # skip exotic dtypes (rare; safetensors handles f32/i64)
        key = short_name(n) if "/" in n else n
        if key in result:
            # Disambiguate by appending the original tag chunk.
            key = f"{key}__{abs(hash(n)) % 10000:04d}"
        # safetensors needs C-contiguous arrays
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        result[f"{name}/{key}"] = arr
    # Cleanup augmented file if we created one
    if augmented_path != onnx_path:
        try: augmented_path.unlink()
        except FileNotFoundError: pass
    return result


def sample_noisy_latent(duration_s: float, sample_rate: int, base_chunk_size: int,
                        chunk_compress: int, latent_dim: int, seed: int):
    rng = np.random.default_rng(seed)
    wav_len = int(duration_s * sample_rate)
    chunk_size = base_chunk_size * chunk_compress
    latent_len = max(1, (wav_len + chunk_size - 1) // chunk_size)
    latent_channels = latent_dim * chunk_compress
    noisy = rng.standard_normal((1, latent_channels, latent_len)).astype(np.float32)
    mask = np.ones((1, 1, latent_len), dtype=np.float32)  # all valid
    return noisy, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", default="models/onnx")
    ap.add_argument("--voice", default="models/voice_styles/M1.json")
    ap.add_argument("--text", default="Hello, world.")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--detail", choices=["minimal", "coarse", "full"], default="coarse")
    ap.add_argument("--out", default="tmp/golden.safetensors")
    args = ap.parse_args()

    onnx_dir = Path(args.onnx_dir)
    voice_path = Path(args.voice)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((onnx_dir / "tts.json").read_text())
    sample_rate = int(cfg["ae"]["sample_rate"])
    base_chunk_size = int(cfg["ae"]["base_chunk_size"])
    chunk_compress = int(cfg["ttl"]["chunk_compress_factor"])
    latent_dim = int(cfg["ttl"]["latent_dim"])

    # --- 1. Inputs derived from text + voice. ---
    processed = preprocess(args.text, args.lang)
    text_ids = encode_ids(processed, onnx_dir / "unicode_indexer.json")
    text_mask = np.ones((1, 1, text_ids.shape[1]), dtype=np.float32)
    style_ttl, style_dp = load_voice(voice_path)
    print(f"text:        {processed!r}  -> shape {text_ids.shape}")
    print(f"style_ttl:   {style_ttl.shape}")
    print(f"style_dp:    {style_dp.shape}")

    bundle: dict[str, np.ndarray] = {
        "input/text_ids": text_ids,
        "input/text_mask": text_mask,
        "input/style_ttl": style_ttl,
        "input/style_dp": style_dp,
        "input/processed_text_utf8": np.frombuffer(processed.encode("utf-8"), dtype=np.uint8),
    }
    meta = {
        "text": args.text,
        "lang": args.lang,
        "voice": str(voice_path),
        "seed": str(args.seed),
        "steps": str(args.steps),
        "speed": str(args.speed),
        "detail": args.detail,
    }

    # --- 2. Duration predictor. ---
    print("\n=== duration_predictor ===")
    dp_out = run_model("duration_predictor", onnx_dir / "duration_predictor.onnx",
                        {"text_ids": text_ids, "style_dp": style_dp, "text_mask": text_mask},
                        args.detail)
    bundle.update(dp_out)
    duration = dp_out["duration_predictor/duration"]
    duration_scaled = (duration / args.speed).astype(np.float32)
    bundle["intermediate/duration_after_speed"] = duration_scaled
    print(f"  duration (raw)    = {duration.tolist()}")
    print(f"  duration / speed  = {duration_scaled.tolist()}")
    print(f"  dump entries: {len(dp_out)}")

    # --- 3. Text encoder. ---
    print("\n=== text_encoder ===")
    te_out = run_model("text_encoder", onnx_dir / "text_encoder.onnx",
                        {"text_ids": text_ids, "style_ttl": style_ttl, "text_mask": text_mask},
                        args.detail)
    bundle.update(te_out)
    text_emb = te_out["text_encoder/text_emb"]
    print(f"  text_emb shape    = {text_emb.shape}")
    print(f"  dump entries: {len(te_out)}")

    # --- 4. Sample seeded noisy latent. ---
    dur_s = float(duration_scaled[0])
    noisy_latent, latent_mask = sample_noisy_latent(
        dur_s, sample_rate, base_chunk_size, chunk_compress, latent_dim, args.seed
    )
    bundle["input/noisy_latent"] = noisy_latent
    bundle["input/latent_mask"] = latent_mask
    print(f"\nnoisy_latent shape = {noisy_latent.shape}  (seed={args.seed}, dur={dur_s:.3f}s)")

    # --- 5. Vector estimator: dump per-step outputs. ---
    print("\n=== vector_estimator ===")
    xt = noisy_latent.copy()
    total_step = np.array([args.steps], dtype=np.float32)
    for step in range(args.steps):
        cur = np.array([step], dtype=np.float32)
        # Only dump intermediates for step 0 and the last step --
        # repeating per step would multiply the dump size by ~steps×
        # without helping a layer-level diff (which targets one step).
        detail_for_step = args.detail if step in (0, args.steps - 1) else "minimal"
        out = run_model(
            f"vector_estimator/step_{step:02d}",
            onnx_dir / "vector_estimator.onnx",
            {
                "noisy_latent": xt, "text_emb": text_emb,
                "style_ttl": style_ttl, "latent_mask": latent_mask,
                "text_mask": text_mask, "current_step": cur, "total_step": total_step,
            },
            detail_for_step,
        )
        bundle.update(out)
        xt = out[f"vector_estimator/step_{step:02d}/denoised_latent"]
        print(f"  step {step}: xt mean={xt.mean():.4f} std={xt.std():.4f} "
              f"(dump entries: {len(out)})")

    bundle["intermediate/final_latent"] = xt

    # --- 6. Vocoder. ---
    print("\n=== vocoder ===")
    voc_out = run_model("vocoder", onnx_dir / "vocoder.onnx",
                         {"latent": xt}, args.detail)
    bundle.update(voc_out)
    wav = voc_out["vocoder/wav_tts"]
    print(f"  wav shape         = {wav.shape}")
    print(f"  wav mean/std/min/max = {wav.mean():.4f} {wav.std():.4f} {wav.min():.4f} {wav.max():.4f}")
    print(f"  dump entries: {len(voc_out)}")

    # --- 7. Save bundle + manifest. ---
    save_file(bundle, str(out_path), metadata={k: str(v) for k, v in meta.items()})
    total_mb = sum(a.nbytes for a in bundle.values()) / 1e6
    print(f"\nwrote {out_path}: {len(bundle)} tensors, {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
