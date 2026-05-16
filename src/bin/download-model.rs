//! Download Supertonic 3 ONNX assets from HuggingFace.
//!
//! Default destination: `./models`. After this runs, the layout is:
//!   models/
//!     onnx/{text_encoder,duration_predictor,vector_estimator,vocoder}.onnx
//!     onnx/{tts.json, unicode_indexer.json}
//!     voice_styles/{F1..F5,M1..M5}.json
//!
//! Usage:
//!     cargo run --release --bin download-model
//!     cargo run --release --bin download-model -- --path some/dir
//!     cargo run --release --bin download-model -- --voices M1,F2

use anyhow::{Context, Result};
use hf_hub::api::sync::{ApiBuilder, ApiRepo};
use hf_hub::{Repo, RepoType};
use std::path::{Path, PathBuf};

const REPO_ID: &str = "Supertone/supertonic-3";
const ONNX_FILES: &[&str] = &[
    "onnx/text_encoder.onnx",
    "onnx/duration_predictor.onnx",
    "onnx/vector_estimator.onnx",
    "onnx/vocoder.onnx",
    "onnx/tts.json",
    "onnx/unicode_indexer.json",
];
const DEFAULT_VOICES: &[&str] = &["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"];

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let mut output = PathBuf::from("./models");
    let mut voices: Vec<String> = DEFAULT_VOICES.iter().map(|s| s.to_string()).collect();
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--path" => {
                output = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--voices" => {
                voices = args[i + 1].split(',').map(|s| s.trim().to_string()).collect();
                i += 2;
            }
            "-h" | "--help" => {
                println!(
                    "usage: download-model [--path DIR] [--voices F1,M2,...]\n\
                     default path: ./models\n\
                     default voices: {DEFAULT_VOICES:?}"
                );
                return Ok(());
            }
            other => anyhow::bail!("unknown argument {other}"),
        }
    }

    println!("Downloading Supertonic 3 to: {}", output.display());
    let api = ApiBuilder::new().build()?;
    let repo = api.repo(Repo::new(REPO_ID.to_string(), RepoType::Model));

    let onnx_dir = output.join("onnx");
    std::fs::create_dir_all(&onnx_dir)?;
    for f in ONNX_FILES {
        let dest = onnx_dir.join(Path::new(f).file_name().unwrap());
        fetch_to(&repo, f, &dest)?;
        println!("  {f} -> {}", dest.display());
    }

    let voices_dir = output.join("voice_styles");
    std::fs::create_dir_all(&voices_dir)?;
    let mut ok = 0;
    for v in &voices {
        let rel = format!("voice_styles/{v}.json");
        let dest = voices_dir.join(format!("{v}.json"));
        match fetch_to(&repo, &rel, &dest) {
            Ok(_) => {
                ok += 1;
                println!("  {rel} -> {}", dest.display());
            }
            Err(e) => eprintln!("  WARN: {rel} failed: {e:#}"),
        }
    }

    println!(
        "\nDone. {} ONNX files, {ok}/{} voices in {}.",
        ONNX_FILES.len(),
        voices.len(),
        output.display()
    );
    Ok(())
}

fn fetch_to(repo: &ApiRepo, rel: &str, dest: &Path) -> Result<()> {
    if dest.exists() {
        return Ok(());
    }
    // hf-hub 0.3 fails on some small non-LFS files (URL construction bug),
    // so we try its happy path first and fall back to a direct HTTPS GET.
    match repo.get(rel) {
        Ok(cache_path) => {
            std::fs::copy(&cache_path, dest).with_context(|| {
                format!("copying {} -> {}", cache_path.display(), dest.display())
            })?;
        }
        Err(_) => {
            let url = format!("https://huggingface.co/{REPO_ID}/resolve/main/{rel}");
            let resp = reqwest_blocking_get(&url)
                .with_context(|| format!("direct fetch {url}"))?;
            std::fs::write(dest, &resp)
                .with_context(|| format!("writing {}", dest.display()))?;
        }
    }
    Ok(())
}

/// Minimal blocking GET via std::process::Command(`curl`). Avoids
/// pulling reqwest in just for the small-file fallback path.
fn reqwest_blocking_get(url: &str) -> Result<Vec<u8>> {
    use std::process::Command;
    let out = Command::new("curl")
        .arg("-sSLf")
        .arg(url)
        .output()
        .context("spawning curl")?;
    if !out.status.success() {
        anyhow::bail!(
            "curl failed ({}): {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(out.stdout)
}
