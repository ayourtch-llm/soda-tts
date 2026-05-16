#!/usr/bin/env python3
"""Extract learned parameters from each Supertonic 3 ONNX file and
write them as safetensors. Graph-internal constants (artifacts of the
ONNX exporter, e.g. `/.../Constant_*_output_0`, `onnx::Tile_1065`)
are filtered out — they're not model parameters, just folded constants
that the candle port will recompute from the architecture spec.

Usage:
    python3 tools/convert_weights.py \\
        [--input  models/onnx] \\
        [--output models/safetensors]

For each `{name}.onnx` in --input, writes `{name}.safetensors` to
--output, prints a one-line tensor count + byte total, and emits a
JSON manifest at `{output}/index.json` summarizing what was extracted.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file


# ONNX data_type → numpy dtype. Covers what Supertonic 3 actually uses.
ONNX_DTYPE_TO_NP = {
    1:  np.float32,   # FLOAT
    2:  np.uint8,     # UINT8
    3:  np.int8,      # INT8
    6:  np.int32,     # INT32
    7:  np.int64,     # INT64
    9:  np.bool_,     # BOOL
    10: np.float16,   # FLOAT16
    11: np.float64,   # DOUBLE
}


def is_param(name: str, dtype: int) -> bool:
    """A real learned parameter, not a graph-internal constant.

    Subtle: the ONNX exporter renames some real weights (`nn.Linear`
    that get inlined as MatMul) to `onnx::MatMul_<n>` and constant-folds
    others (small Expand outputs) under `/.../Expand_output_0`. A
    name-based filter alone misses those.

    So: anything stored in float (f32/f16) is treated as a parameter
    regardless of name. Integer tensors (axes, shape specs for Tile /
    ReduceSum / Reshape / Concat) are skipped — they encode graph
    structure that the candle port reconstructs from the architecture
    spec, not learned values.

    A handful of scalar f32 constants (GELU's 0.5 and sqrt(2), inverse-
    freq tables of ~32 elements) sneak through under this rule. That's
    a few hundred bytes of duplicate-but-recomputable data per model —
    much cheaper than the risk of dropping a real weight.
    """
    if dtype not in (1, 10):  # 1=FLOAT, 10=FLOAT16
        return False
    return True


def convert_one(onnx_path: Path, st_path: Path) -> dict:
    model = onnx.load(str(onnx_path))
    all_inits = list(model.graph.initializer)
    kept: dict[str, np.ndarray] = {}
    skipped: list[str] = []
    dropped_unknown_dtype: list[tuple[str, int]] = []
    total_bytes = 0

    for ini in all_inits:
        if not is_param(ini.name, ini.data_type):
            skipped.append(ini.name)
            continue
        if ini.data_type not in ONNX_DTYPE_TO_NP:
            dropped_unknown_dtype.append((ini.name, ini.data_type))
            continue
        arr = numpy_helper.to_array(ini)
        # Force the dtype to be exactly what ONNX advertised, just in case
        # numpy_helper widened it for older int kinds.
        expected = ONNX_DTYPE_TO_NP[ini.data_type]
        if arr.dtype != expected:
            arr = arr.astype(expected, copy=False)
        # safetensors requires contiguous arrays.
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        if ini.name in kept:
            raise RuntimeError(f"duplicate initializer name: {ini.name!r}")
        kept[ini.name] = arr
        total_bytes += arr.nbytes

    st_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(kept, str(st_path))
    return {
        "onnx": str(onnx_path),
        "safetensors": str(st_path),
        "kept_tensors": len(kept),
        "skipped_graph_constants": len(skipped),
        "dropped_unknown_dtype": [
            {"name": n, "onnx_dtype": d} for n, d in dropped_unknown_dtype
        ],
        "total_param_bytes": total_bytes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="models/onnx", help="dir of *.onnx files")
    ap.add_argument("--output", default="models/safetensors", help="output dir")
    args = ap.parse_args()
    in_dir = Path(args.input)
    out_dir = Path(args.output)
    if not in_dir.is_dir():
        sys.exit(f"input dir not found: {in_dir}")

    onnx_files = sorted(p for p in in_dir.iterdir() if p.suffix == ".onnx")
    if not onnx_files:
        sys.exit(f"no .onnx files in {in_dir}")

    manifest = {"models": []}
    for src in onnx_files:
        dst = out_dir / (src.stem + ".safetensors")
        info = convert_one(src, dst)
        manifest["models"].append(info)
        print(
            f"  {src.name:>30s}  -> {dst.name}  "
            f"{info['kept_tensors']:>4d} tensors, "
            f"{info['total_param_bytes']/1e6:>7.1f} MB"
            + (
                f"  (skipped {info['skipped_graph_constants']} graph constants)"
                if info["skipped_graph_constants"]
                else ""
            )
        )
        if info["dropped_unknown_dtype"]:
            print(
                f"    WARN: dropped {len(info['dropped_unknown_dtype'])} tensors "
                f"with unknown dtypes: {info['dropped_unknown_dtype'][:3]}{'...' if len(info['dropped_unknown_dtype'])>3 else ''}"
            )

    manifest_path = out_dir / "index.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
