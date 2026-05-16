#!/usr/bin/env python3
"""Verify that every extracted safetensors tensor is byte-identical to
the matching ONNX initializer.

If this passes, candle can load the safetensors and we know the bits
came across without any cast/repack damage.

Usage:
    python3 tools/verify_weights.py [--input models/onnx] [--st models/safetensors]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import load_file


def verify_one(onnx_path: Path, st_path: Path) -> tuple[int, int, int]:
    """Returns (compared, mismatched, missing_in_st)."""
    model = onnx.load(str(onnx_path))
    # Mirror tools/convert_weights.py:is_param exactly: keep f32/f16.
    onnx_params = {
        ini.name: numpy_helper.to_array(ini)
        for ini in model.graph.initializer
        if ini.data_type in (1, 10)  # FLOAT, FLOAT16
    }
    st = load_file(str(st_path))
    compared = 0
    mismatched = 0
    missing = 0
    for name, onnx_arr in onnx_params.items():
        if name not in st:
            missing += 1
            print(f"  MISSING in safetensors: {name}")
            continue
        st_arr = st[name]
        compared += 1
        if st_arr.dtype != onnx_arr.dtype:
            mismatched += 1
            print(f"  DTYPE MISMATCH {name}: onnx={onnx_arr.dtype}, st={st_arr.dtype}")
            continue
        if st_arr.shape != onnx_arr.shape:
            mismatched += 1
            print(f"  SHAPE MISMATCH {name}: onnx={onnx_arr.shape}, st={st_arr.shape}")
            continue
        if not np.array_equal(st_arr, onnx_arr):
            mismatched += 1
            # Sample the first delta for diagnosis
            diff_mask = (st_arr != onnx_arr).flatten()
            idx = int(np.argmax(diff_mask))
            flat_onnx = onnx_arr.flatten()
            flat_st = st_arr.flatten()
            print(f"  VALUE MISMATCH {name}: first delta at flat idx {idx} "
                  f"onnx={flat_onnx[idx]} st={flat_st[idx]}")
            continue
    extra = sorted(set(st.keys()) - set(onnx_params.keys()))
    if extra:
        print(f"  EXTRA in safetensors (not in onnx params): {extra[:3]}{'...' if len(extra)>3 else ''}")
    return compared, mismatched, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="models/onnx")
    ap.add_argument("--st", default="models/safetensors")
    args = ap.parse_args()

    in_dir = Path(args.input)
    st_dir = Path(args.st)
    if not in_dir.is_dir() or not st_dir.is_dir():
        sys.exit(f"need both {in_dir} and {st_dir}")

    overall_compared = 0
    overall_mismatched = 0
    overall_missing = 0
    for onnx_path in sorted(in_dir.glob("*.onnx")):
        st_path = st_dir / (onnx_path.stem + ".safetensors")
        if not st_path.exists():
            print(f"  SKIP {onnx_path.name}: no matching {st_path.name}")
            continue
        print(f"=== {onnx_path.stem} ===")
        c, m, miss = verify_one(onnx_path, st_path)
        print(f"  compared={c}  mismatched={m}  missing={miss}")
        overall_compared += c
        overall_mismatched += m
        overall_missing += miss

    print()
    print(f"TOTAL: compared={overall_compared} mismatched={overall_mismatched} missing={overall_missing}")
    if overall_mismatched or overall_missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
