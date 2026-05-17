# soda-tts

A native Rust port of [Supertonic 3](https://huggingface.co/Supertone/supertonic-3), a 31-language on-device TTS model. Two backends, same speech:

- **`speak`** — ONNX Runtime via the `ort` crate (faster: ~4× realtime on CPU).
- **`speak_candle`** — pure Rust via [candle](https://github.com/huggingface/candle) (~1.7× realtime on CPU; no native libonnxruntime dependency).

Both produce bit-exact-equivalent output (max-abs delta ≤ 3e-6 vs ort across all four models). Verified end-to-end via round-trip through `nemotron-speech`'s ASR — 10/10 test sentences transcribed verbatim.

## What you get

```text
text  →  44.1 kHz mono WAV  (with one of 10 voices: F1..F5, M1..M5)
```

31 languages supported (English, Korean, Japanese, Arabic, Bulgarian, Czech, Danish, German, Greek, Spanish, Estonian, Finnish, French, Hindi, Croatian, Hungarian, Indonesian, Italian, Lithuanian, Latvian, Dutch, Polish, Portuguese, Romanian, Russian, Slovak, Slovenian, Swedish, Turkish, Ukrainian, Vietnamese), no G2P / lexicon pipeline needed — the model takes raw Unicode codepoints plus a `<lang>...</lang>` wrapper.

## Quick start

Prerequisites: Rust toolchain, `curl`, and [`uv`](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`). `uv` is only used **once**, to run the ONNX→safetensors conversion script in an ephemeral environment — nothing is installed to your system Python. The ort backend doesn't need `uv` at all.

```sh
git clone <this repo> soda-tts
cd soda-tts

# One-shot setup: builds Rust, downloads ~400 MB of model files, converts
# them, verifies bit-for-bit. Takes ~3-5 minutes on a fast connection.
./tools/setup.sh

# Speak (ort backend, fastest):
./target/release/speak --text "Hello, world." --out hello.wav

# Speak (pure Rust candle backend):
./target/release/speak_candle --text "Hello, world." --out hello.wav
```

If `setup.sh` fails (e.g. behind a proxy), see [Manual setup](#manual-setup) below.

## CLI

Both binaries share most flags:

| Flag | Default | Notes |
|---|---|---|
| `--text "..."` | (required) | Input text. Or use `--infile FILE`. |
| `--lang en` | `en` | One of the 31 supported language codes. |
| `--voice PATH` | `models/voice_styles/M1.json` | One of F1..F5, M1..M5. |
| `--out FILE` | `hello.wav` | Output 44.1 kHz mono 16-bit WAV. |
| `--speed F` | `1.05` | Speech rate multiplier. |
| `--steps N` | `8` | Flow-matching ODE steps. 4–16 useful range. |
| `--seed N` | (none) | Reproducible noise sampling. |
| `--silence-ms N` | `300` | Silence between sentences for multi-chunk text. |
| `--verbose` | — | Per-chunk timing. |

`speak_candle` adds:

| Flag | Default | Notes |
|---|---|---|
| `--device cpu\|metal\|auto` | `auto` | Metal needs `--features metal` at build time. |

## Picking a backend

| | ort (`speak`) | candle (`speak_candle`) |
|---|---|---|
| Speed (CPU) | ~4× realtime | ~1.7× realtime |
| Native deps | libonnxruntime (auto-downloaded on first build) | none |
| Binary size | larger (includes ort) | smaller |
| Metal | no | yes (but slower than CPU for this model — see [Metal note](#metal-note)) |
| Numerical accuracy | reference | matches ort to ~3e-6 max-abs |

**Recommendation:** use `speak` (ort) unless you specifically want pure-Rust. Both are functionally identical.

## Manual setup

If the one-shot script fails or you want to see what's happening:

```sh
# 1. Build the Rust binaries.
cargo build --release --bin speak --bin speak_candle --bin download-model

# 2. Download ONNX model files + voice styles (~400 MB).
./target/release/download-model
# This populates models/onnx/ and models/voice_styles/.

# 3. (Candle backend only) Convert ONNX -> safetensors via `uv`. The
#    --with flags create an ephemeral env containing onnx/safetensors/numpy;
#    nothing is installed to your system Python.
uv run --with onnx --with safetensors --with numpy python tools/convert_weights.py
# This populates models/safetensors/.

# 4. (Optional) Verify the conversion is byte-identical.
uv run --with onnx --with safetensors --with numpy python tools/verify_weights.py
# Should print "TOTAL: compared=708 mismatched=0 missing=0".

# 5. Speak!
./target/release/speak        --text "Hello, world." --out hello.wav
./target/release/speak_candle --text "Hello, world." --out hello.wav
```

### If `download-model` fails

`hf-hub` (Rust HF Hub client) has occasional issues with small non-LFS files; `download-model` falls back to `curl` automatically. If even that fails (e.g. proxies), the seven small files can be fetched manually:

```sh
mkdir -p models/onnx models/voice_styles
BASE=https://huggingface.co/Supertone/supertonic-3/resolve/main
for f in onnx/text_encoder.onnx onnx/duration_predictor.onnx \
         onnx/vector_estimator.onnx onnx/vocoder.onnx \
         onnx/tts.json onnx/unicode_indexer.json; do
    curl -sSL -o "models/$f" "$BASE/$f"
done
for v in F1 F2 F3 F4 F5 M1 M2 M3 M4 M5; do
    curl -sSL -o "models/voice_styles/${v}.json" "$BASE/voice_styles/${v}.json"
done
```

## Project structure

```
src/
  text.rs                       # Unicode tokenizer + preprocessing + chunker
  audio.rs                      # 44.1 kHz WAV writer + linear resampler
  synthesis.rs                  # Long-form chunking + streaming callback (ort path)
  model/
    mod.rs                      # ort Supertonic: 4 ONNX sessions + orchestration
    config.rs voice.rs          # tts.json loader, voice-style JSON loader
    candle/
      mod.rs                    # candle backend (pure Rust)
      supertonic.rs             # candle Supertonic (mirrors ort API)
      vocoder.rs                # 10-block ConvNeXt + STFT-ish generator
      duration_predictor.rs     # ConvNeXt + VITS rel-attn + small MLP
      text_encoder.rs           # ConvNeXt + rel-attn + speech-prompted cross-attn
      vector_estimator.rs       # Flow-matching: time enc + 4 main_blocks + ODE step
  bin/
    speak.rs                    # ort CLI
    speak_candle.rs             # candle CLI
    download-model.rs           # pulls ONNX + voice_styles from HuggingFace
    tts_asr_roundtrip.rs        # accuracy harness: TTS -> ASR -> WER
    {vocoder,dp,text_encoder,ve}_check.rs   # per-layer diff vs ort golden

tools/
  convert_weights.py            # ONNX initializers -> safetensors
  verify_weights.py             # byte-check the conversion
  dump_golden.py                # generate reference activations (dev only)
  setup.sh                      # one-shot installer

models/                         # populated by download-model + convert_weights
  onnx/{four .onnx files, tts.json, unicode_indexer.json}
  safetensors/{four .safetensors}   # candle weights
  voice_styles/{F1..F5,M1..M5}.json

tmp/                            # build artifacts + test fixtures
```

## Accuracy & validation

The candle backend is **bit-exact** to ort across every named intermediate:

| Model | Per-layer max-abs delta | Final-output max-abs delta |
|---|---|---|
| vocoder | ≤ 2.4e-3 | 3.3e-6 (wav samples) |
| duration_predictor | ≤ 1.7e-6 | 2.4e-7 (seconds) |
| text_encoder | ≤ 5.3e-5 | 1.3e-6 (text_emb) |
| vector_estimator | ≤ 1.7e-6 | 1.2e-6 (denoised_latent) |

To re-verify on your machine:

```sh
# Regenerate the ort "golden" activation dump (uv-managed Python deps).
uv run --with onnx --with onnxruntime --with safetensors --with numpy \
    python tools/dump_golden.py
# Writes tmp/golden.safetensors (~114 MB).

# Run each per-layer check (each prints max_abs delta per hook + PASS/FAIL).
cargo run --release --bin vocoder_check
cargo run --release --bin dp_check
cargo run --release --bin text_encoder_check
cargo run --release --bin ve_check
```

End-to-end round-trip (TTS → audio → ASR → WER), needs sibling `nemotron-speech` repo:

```sh
# Build the nemotron transcribe binary once.
(cd ../nemotron-speech && cargo build --release --bin transcribe)

cargo run --release --bin tts_asr_roundtrip -- \
    --samples tmp/samples.txt --start 0 --end 10
# 10/10 verbatim with M1 voice on the included sample set.
```

## Performance

5.65 s of audio synthesized from "A gentle breeze moved through the open window
while everyone listened to the story.", 8 ODE steps, M1 voice, Apple Silicon
M-series CPU, averaged over 3 runs:

| Backend | Wall time | Realtime factor |
|---|---|---|
| `speak` (ort, CPU) | 1.44 s | 3.92× |
| `speak_candle --device cpu` | 3.37 s | 1.68× |
| `speak_candle --device metal` | 6.12 s | 0.92× |

### Metal note

For this ~100M-param model, candle's Metal backend is **slower** than the CPU
backend (1.8× slower in the benchmark above). The vector_estimator alone has
1004 ops × 8 ODE steps × batch=2 (internal CFG), so per-kernel GPU dispatch
overhead dominates the savings from parallel compute. The `metal` feature
flag exists for experimentation; **CPU is the recommended candle device**.

If you re-port to a model with significantly more compute per kernel (≥1B
params, longer sequences, custom fused-attention kernels), Metal would
plausibly win.

## How the candle port was built

Eight phases, summarized in [docs/PHASES.md](docs/PHASES.md) (and in the
corresponding commit messages on `main`):

| Phase | Outcome |
|---|---|
| 0. ONNX → safetensors | 708 weight tensors extracted, byte-identical |
| 1. Golden activation dump | 906 reference tensors for per-layer diffing |
| 2. Vocoder in candle | bit-exact, 5e-4 threshold passed |
| 3. Duration predictor in candle | bit-exact |
| 4. Text encoder in candle | bit-exact |
| 5. Vector estimator in candle | bit-exact (hardest — 1004 ops, RoPE, CFG, bundled ODE) |
| 6. Wire-up + speak_candle binary | 10/10 ASR round-trip |
| 7. Metal feature + benchmark | both backends shipped |

The per-layer hook diff was the central enabling tool — every numerical bug
was localized to a single sub-op in seconds rather than blind-staring at the
final output.

## Licensing

- **Code** in this repo: MIT.
- **Model weights** (downloaded by `download-model`): [OpenRAIL-M](https://huggingface.co/Supertone/supertonic-3/blob/main/LICENSE), © 2026 Supertone Inc.

## Acknowledgments

- [Supertonic 3](https://github.com/supertone-inc/supertonic) — upstream model and the reference Rust implementation that this builds on.
- [candle](https://github.com/huggingface/candle) — the Rust ML framework powering the pure-Rust backend.
- [ort](https://github.com/pykeio/ort) — the Rust wrapper around ONNX Runtime.
- [nemotron-speech-rs](../nemotron-speech) — sibling streaming ASR used for round-trip validation.
- [kokoro-tts](../kokoro-tts) — sibling TTS project whose Phase structure and per-layer validation approach this borrows heavily from.
