//! `speak`: text → 44.1 kHz mono WAV using Supertonic 3.

use anyhow::{bail, Context, Result};
use soda_tts::audio::write_wav;
use soda_tts::model::{Supertonic, SynthesisConfig, VoiceStyle};
use soda_tts::synthesis::{synthesize_text, SynthesisOptions};
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
            text: None,
            infile: None,
            out: PathBuf::from("hello.wav"),
            lang: "en".to_string(),
            speed: 1.05,
            steps: 8,
            seed: None,
            silence_ms: 300,
            verbose: false,
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
                        "usage: speak --text \"...\" [--lang en] [--voice PATH] \\\n\
                         \t[--model-dir DIR] [--out FILE] [--speed F] [--steps N] [--seed N]\\\n\
                         \t[--silence-ms N] [--verbose]\n\
                         model-dir expects subdirs onnx/ and voice_styles/."
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

    let onnx_dir = args.model_dir.join("onnx");
    if args.verbose {
        eprintln!("loading model from {}", onnx_dir.display());
    }
    let t_load = std::time::Instant::now();
    let mut model = Supertonic::load(&onnx_dir)?;
    if args.verbose {
        eprintln!("model loaded in {:.2}s", t_load.elapsed().as_secs_f64());
    }
    let voice = VoiceStyle::load(&args.voice)?;
    if args.verbose {
        eprintln!(
            "voice {} loaded: ttl {:?}, dp {:?}",
            args.voice.display(),
            voice.ttl.shape(),
            voice.dp.shape()
        );
    }

    let cfg = SynthesisConfig {
        total_steps: args.steps,
        speed: args.speed,
        seed: args.seed,
    };
    let opts = SynthesisOptions {
        silence_seconds: args.silence_ms as f32 / 1000.0,
        max_chunk_chars: None,
    };

    let t_synth = std::time::Instant::now();
    let samples = synthesize_text(&mut model, &text, &args.lang, &voice, cfg, &opts)?;
    let elapsed = t_synth.elapsed().as_secs_f64();
    let audio_s = samples.len() as f64 / model.sample_rate() as f64;
    eprintln!(
        "synthesized {} samples ({:.2}s audio) in {:.2}s ({:.2}x realtime)",
        samples.len(),
        audio_s,
        elapsed,
        audio_s / elapsed.max(1e-6)
    );

    write_wav(&samples, &args.out, model.sample_rate())?;
    eprintln!("wrote {}", args.out.display());
    Ok(())
}
