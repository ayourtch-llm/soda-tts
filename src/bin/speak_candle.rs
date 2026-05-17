//! Pure-Rust `speak`: text -> 44.1 kHz mono WAV using the candle backend.
//! Same CLI as `speak` (the ort-backed binary), just routes through
//! `CandleSupertonic` instead of `Supertonic`.

use anyhow::{bail, Context, Result};
use candle_core::Device;
use soda_tts::audio::write_wav;
use soda_tts::model::candle::CandleSupertonic;
use soda_tts::model::voice::VoiceStyle;
use soda_tts::model::SynthesisConfig;
use soda_tts::text::{chunk_max_for, chunk_text};
use std::path::PathBuf;

struct Args {
    model_dir: PathBuf,
    voice: PathBuf,
    text: Option<String>,
    infile: Option<PathBuf>,
    out: PathBuf,
    lang: String,
    speed: f32,
    steps: usize,
    seed: Option<u64>,
    silence_ms: u32,
    verbose: bool,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut a = Self {
            model_dir: PathBuf::from("models"),
            voice: PathBuf::from("models/voice_styles/M1.json"),
            text: None, infile: None,
            out: PathBuf::from("hello.wav"),
            lang: "en".to_string(),
            speed: 1.05, steps: 8, seed: None,
            silence_ms: 300, verbose: false,
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--model-dir" => a.model_dir = PathBuf::from(it.next().context("--model-dir")?),
                "--voice" => a.voice = PathBuf::from(it.next().context("--voice")?),
                "--text" => a.text = Some(it.next().context("--text")?),
                "--infile" => a.infile = Some(PathBuf::from(it.next().context("--infile")?)),
                "--out" => a.out = PathBuf::from(it.next().context("--out")?),
                "--lang" => a.lang = it.next().context("--lang")?,
                "--speed" => a.speed = it.next().context("--speed")?.parse()?,
                "--steps" => a.steps = it.next().context("--steps")?.parse()?,
                "--seed" => a.seed = Some(it.next().context("--seed")?.parse()?),
                "--silence-ms" => a.silence_ms = it.next().context("--silence-ms")?.parse()?,
                "--verbose" => a.verbose = true,
                "-h" | "--help" => {
                    println!(
                        "usage: speak_candle --text \"...\" [--lang en] [--voice PATH] \\\n\
                         \t[--model-dir DIR] [--out FILE] [--speed F] [--steps N] [--seed N] \\\n\
                         \t[--silence-ms N] [--verbose]\n\
                         model-dir expects subdirs onnx/ (config + tokenizer), safetensors/\n\
                         (model weights), and voice_styles/."
                    );
                    std::process::exit(0);
                }
                other => bail!("unknown argument {other}"),
            }
        }
        Ok(a)
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse()?;
    let text = match (&args.text, &args.infile) {
        (Some(t), None) => t.clone(),
        (None, Some(p)) => std::fs::read_to_string(p)
            .with_context(|| format!("reading {}", p.display()))?,
        (Some(_), Some(_)) => bail!("--text and --infile are mutually exclusive"),
        (None, None) => bail!("supply --text or --infile"),
    };

    let device = Device::Cpu;
    if args.verbose {
        eprintln!("loading candle Supertonic from {}", args.model_dir.display());
    }
    let t_load = std::time::Instant::now();
    let model = CandleSupertonic::load(&args.model_dir, &device)?;
    if args.verbose {
        eprintln!("loaded in {:.2}s", t_load.elapsed().as_secs_f64());
    }
    let voice = VoiceStyle::load(&args.voice)?;
    if args.verbose {
        eprintln!(
            "voice {} loaded: ttl {:?}, dp {:?}",
            args.voice.display(), voice.ttl.shape(), voice.dp.shape()
        );
    }

    let cfg = SynthesisConfig {
        total_steps: args.steps, speed: args.speed, seed: args.seed,
    };
    let sample_rate = model.sample_rate();
    let silence_samples = (args.silence_ms as f32 / 1000.0 * sample_rate as f32) as usize;
    let max_chars = chunk_max_for(&args.lang);
    let chunks = chunk_text(&text, max_chars);
    if chunks.is_empty() {
        bail!("no chunks to synthesize");
    }

    let t_synth = std::time::Instant::now();
    let mut out: Vec<f32> = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        if chunk.trim().is_empty() {
            out.extend(std::iter::repeat(0.0).take(silence_samples));
            continue;
        }
        let (samples, dur_s) = model.synthesize_one(chunk, &args.lang, &voice, cfg)
            .with_context(|| format!("synthesizing chunk {}", i + 1))?;
        if args.verbose {
            eprintln!(
                "chunk {}/{}: {:.2}s, {} samples",
                i + 1, chunks.len(), dur_s, samples.len()
            );
        }
        out.extend_from_slice(&samples);
        if i + 1 < chunks.len() {
            out.extend(std::iter::repeat(0.0).take(silence_samples));
        }
    }
    let elapsed = t_synth.elapsed().as_secs_f64();
    let audio_s = out.len() as f64 / sample_rate as f64;
    eprintln!(
        "synthesized {} samples ({:.2}s audio) in {:.2}s ({:.2}x realtime)",
        out.len(), audio_s, elapsed, audio_s / elapsed.max(1e-6)
    );

    write_wav(&out, &args.out, sample_rate)?;
    eprintln!("wrote {}", args.out.display());
    Ok(())
}
