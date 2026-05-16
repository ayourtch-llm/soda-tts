//! Subset of `onnx/tts.json` that we actually need at runtime.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub ae: AeConfig,
    pub ttl: TtlConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AeConfig {
    pub sample_rate: i32,
    pub base_chunk_size: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TtlConfig {
    pub latent_dim: i32,
    pub chunk_compress_factor: i32,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let cfg: Config =
            serde_json::from_reader(BufReader::new(f)).context("parsing tts.json")?;
        Ok(cfg)
    }
}
