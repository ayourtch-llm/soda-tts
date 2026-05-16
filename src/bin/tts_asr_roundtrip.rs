//! TTS → ASR round-trip harness.
//!
//! Synthesizes each sentence with soda-tts (Supertonic 3, in-process)
//! then shells out to `nemotron-speech`'s `transcribe` binary for ASR.
//! Two processes because `sentencepiece-sys` and ONNX Runtime both
//! statically link incompatible libprotobuf builds and clash at
//! startup if loaded in the same process.
//!
//! Prints `idx | WER | original | hypothesis` per line, then an
//! aggregate WER at the bottom — that's the main quality signal.
//!
//! Usage:
//!     ./target/release/tts_asr_roundtrip \
//!         --samples tmp/samples.txt --start 0 --end 50 \
//!         [--model-dir models] [--voice models/voice_styles/M1.json] \
//!         [--lang en] [--speed 1.0] [--steps 8] [--seed 42] \
//!         [--asr-bin ../nemotron-speech/target/release/transcribe] \
//!         [--nemo-st ../nemotron-speech/models/...safetensors] \
//!         [--nemo-tok ../nemotron-speech/models/tokenizer.model]

use anyhow::{bail, Context, Result};
use soda_tts::audio::{resample_linear, write_wav};
use soda_tts::model::{Supertonic, SynthesisConfig, VoiceStyle};
use soda_tts::synthesis::{synthesize_text, SynthesisOptions};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

struct Args {
    samples: PathBuf,
    start: usize,
    end: Option<usize>,
    lines: Option<Vec<(usize, usize)>>,
    model_dir: PathBuf,
    voice: PathBuf,
    lang: String,
    speed: f32,
    steps: usize,
    seed: Option<u64>,
    asr_bin: PathBuf,
    nemo_st: PathBuf,
    nemo_tok: PathBuf,
    tmp_dir: PathBuf,
    keep_wavs: bool,
    summary: bool,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut a = Self {
            samples: PathBuf::from("tmp/samples.txt"),
            start: 0,
            end: None,
            lines: None,
            model_dir: PathBuf::from("models"),
            voice: PathBuf::from("models/voice_styles/M1.json"),
            lang: "en".to_string(),
            speed: 1.0,
            steps: 8,
            seed: None,
            asr_bin: PathBuf::from("../nemotron-speech/target/release/transcribe"),
            nemo_st: PathBuf::from(
                "../nemotron-speech/models/nemotron-speech-streaming-en-0.6b.safetensors",
            ),
            nemo_tok: PathBuf::from("../nemotron-speech/models/tokenizer.model"),
            tmp_dir: PathBuf::from("tmp/roundtrip"),
            keep_wavs: false,
            summary: true,
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--samples" => a.samples = PathBuf::from(it.next().context("--samples")?),
                "--start" => a.start = it.next().context("--start")?.parse()?,
                "--end" => a.end = Some(it.next().context("--end")?.parse()?),
                "--lines" => a.lines = Some(parse_line_spec(&it.next().context("--lines")?)?),
                "--model-dir" => a.model_dir = PathBuf::from(it.next().context("--model-dir")?),
                "--voice" => a.voice = PathBuf::from(it.next().context("--voice")?),
                "--lang" => a.lang = it.next().context("--lang")?,
                "--speed" => a.speed = it.next().context("--speed")?.parse()?,
                "--steps" => a.steps = it.next().context("--steps")?.parse()?,
                "--seed" => a.seed = Some(it.next().context("--seed")?.parse()?),
                "--asr-bin" => a.asr_bin = PathBuf::from(it.next().context("--asr-bin")?),
                "--nemo-st" => a.nemo_st = PathBuf::from(it.next().context("--nemo-st")?),
                "--nemo-tok" => a.nemo_tok = PathBuf::from(it.next().context("--nemo-tok")?),
                "--tmp-dir" => a.tmp_dir = PathBuf::from(it.next().context("--tmp-dir")?),
                "--keep-wavs" => a.keep_wavs = true,
                "--no-summary" => a.summary = false,
                "-h" | "--help" => {
                    println!(
                        "usage: tts_asr_roundtrip --samples FILE [--start N --end M | --lines spec]\n\
                         optional: --model-dir DIR --voice PATH --lang en\n\
                                   --speed F --steps N --seed N\n\
                                   --asr-bin PATH --nemo-st PATH --nemo-tok PATH\n\
                                   --tmp-dir DIR --keep-wavs --no-summary\n\
                         --lines spec: comma-separated ranges, e.g. 5,10-20,42"
                    );
                    std::process::exit(0);
                }
                other => bail!("unknown arg {other}"),
            }
        }
        Ok(a)
    }
}

fn parse_line_spec(spec: &str) -> Result<Vec<(usize, usize)>> {
    let mut out = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() { continue; }
        if let Some((lo, hi)) = part.split_once('-') {
            out.push((lo.parse()?, hi.parse()?));
        } else {
            let n: usize = part.parse()?;
            out.push((n, n));
        }
    }
    Ok(out)
}

fn normalize_for_wer(s: &str) -> Vec<String> {
    s.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .map(|s| s.to_string())
        .collect()
}

fn word_edit_distance(a: &[String], b: &[String]) -> usize {
    let (m, n) = (a.len(), b.len());
    if m == 0 { return n; }
    if n == 0 { return m; }
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 0..=m { dp[i][0] = i; }
    for j in 0..=n { dp[0][j] = j; }
    for i in 1..=m {
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i][j - 1] + 1)
                .min(dp[i - 1][j] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }
    dp[m][n]
}

fn wer(reference: &str, hypothesis: &str) -> (f64, usize, usize) {
    let r = normalize_for_wer(reference);
    let h = normalize_for_wer(hypothesis);
    let denom = r.len().max(h.len());
    if denom == 0 { return (0.0, 0, 0); }
    let d = word_edit_distance(&r, &h);
    (d as f64 / denom as f64, d, denom)
}

fn read_samples(path: &Path) -> Result<Vec<String>> {
    let f = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut out = Vec::new();
    for line in BufReader::new(f).lines() { out.push(line?); }
    Ok(out)
}

fn selected_indices(args: &Args, total: usize) -> Vec<usize> {
    let mut out = Vec::new();
    if let Some(ranges) = &args.lines {
        for (lo, hi) in ranges {
            for i in *lo..=*hi { if i < total { out.push(i); } }
        }
    } else {
        let end = args.end.unwrap_or(total).min(total);
        for i in args.start..end { out.push(i); }
    }
    out.sort();
    out.dedup();
    out
}

/// Run the external `transcribe` binary on a WAV file, returning the
/// hypothesis text (parsed from the binary's stdout — it currently
/// prints the transcript on stdout). If that format changes, adjust
/// `pluck_transcript`.
fn run_asr(args: &Args, wav_path: &Path) -> Result<String> {
    let out = Command::new(&args.asr_bin)
        .arg("--audio").arg(wav_path)
        .arg("--st").arg(&args.nemo_st)
        .arg("--tok").arg(&args.nemo_tok)
        .arg("--cpu")
        .output()
        .with_context(|| format!("spawning {}", args.asr_bin.display()))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        bail!("transcribe failed ({}): {stderr}", out.status);
    }
    let stdout = String::from_utf8(out.stdout).context("non-utf8 stdout from transcribe")?;
    Ok(pluck_transcript(&stdout))
}

/// `nemotron-speech`'s transcribe binary prints the final transcript as
/// the last non-empty line of stdout (in current builds). Be defensive
/// in case it adds a "transcript: " prefix or similar in the future.
fn pluck_transcript(stdout: &str) -> String {
    let last = stdout
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .last()
        .unwrap_or("")
        .to_string();
    if let Some(rest) = last.strip_prefix("transcript:") {
        rest.trim().to_string()
    } else {
        last
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse()?;

    if !args.asr_bin.exists() {
        bail!(
            "ASR binary not found at {}. Build nemotron-speech first:\n  \
             (cd ../nemotron-speech && cargo build --release --bin transcribe)\n\
             or pass --asr-bin PATH.",
            args.asr_bin.display()
        );
    }

    let onnx_dir = args.model_dir.join("onnx");
    eprintln!("loading soda-tts from {} ...", onnx_dir.display());
    let mut tts = Supertonic::load(&onnx_dir)?;
    let voice = VoiceStyle::load(&args.voice)?;
    let synth_cfg = SynthesisConfig {
        total_steps: args.steps,
        speed: args.speed,
        seed: args.seed,
    };
    let synth_opts = SynthesisOptions { silence_seconds: 0.2, max_chunk_chars: None };

    let samples = read_samples(&args.samples)?;
    let indices = selected_indices(&args, samples.len());
    eprintln!(
        "samples: {} total; running {} (first={:?} last={:?})",
        samples.len(),
        indices.len(),
        indices.first(),
        indices.last(),
    );

    fs::create_dir_all(&args.tmp_dir)
        .with_context(|| format!("creating tmp dir {}", args.tmp_dir.display()))?;

    let sr = tts.sample_rate();
    let mut total_edits = 0usize;
    let mut total_words = 0usize;
    let mut total_samples = 0usize;
    let mut errors = 0usize;
    let mut skipped = 0usize;
    // Flush each line so output streams during long runs.
    let mut stdout = std::io::stdout().lock();

    for idx in indices {
        let text = &samples[idx];
        if text.trim().is_empty() {
            skipped += 1;
            continue;
        }
        let synth_t0 = std::time::Instant::now();
        let audio_native = match synthesize_text(&mut tts, text, &args.lang, &voice, synth_cfg, &synth_opts) {
            Ok(s) => s,
            Err(e) => {
                writeln!(stdout, "{idx:>6} | ERR synth: {e:#}")?;
                writeln!(stdout, "       | orig: {text}")?;
                errors += 1;
                continue;
            }
        };
        let synth_s = synth_t0.elapsed().as_secs_f64();
        let audio_16k = resample_linear(&audio_native, sr, 16_000);
        if audio_16k.is_empty() {
            writeln!(stdout, "{idx:>6} | ERR empty audio after resample")?;
            errors += 1;
            continue;
        }
        let wav_path = args.tmp_dir.join(format!("sample_{idx:06}.wav"));
        if let Err(e) = write_wav(&audio_16k, &wav_path, 16_000) {
            writeln!(stdout, "{idx:>6} | ERR write wav: {e:#}")?;
            errors += 1;
            continue;
        }
        let hyp = match run_asr(&args, &wav_path) {
            Ok(h) => h,
            Err(e) => {
                writeln!(stdout, "{idx:>6} | ERR asr: {e:#}")?;
                writeln!(stdout, "       | orig: {text}")?;
                errors += 1;
                if !args.keep_wavs { let _ = fs::remove_file(&wav_path); }
                continue;
            }
        };
        if !args.keep_wavs { let _ = fs::remove_file(&wav_path); }

        let (rate, d, denom) = wer(text, &hyp);
        total_edits += d;
        total_words += denom;
        total_samples += 1;
        writeln!(stdout, "{idx:>6} | WER={rate:.3} ({d}/{denom} words) | synth={synth_s:.2}s")?;
        writeln!(stdout, "       | orig: {text}")?;
        writeln!(stdout, "       | asr : {}", hyp.trim())?;
        stdout.flush()?;
    }

    if args.summary {
        let agg = if total_words == 0 { 0.0 } else { total_edits as f64 / total_words as f64 };
        writeln!(stdout)?;
        writeln!(
            stdout,
            "summary: aggregate WER={agg:.3} ({total_edits} edits / {total_words} ref-words); \
             samples={total_samples} skipped={skipped} errors={errors}",
        )?;
    }
    Ok(())
}
