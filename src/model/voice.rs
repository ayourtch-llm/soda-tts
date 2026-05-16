//! Voice-style loader. A voice file is a JSON blob with two tensors:
//!   - `style_ttl` shape (1, 50, 256)  — used by text encoder + vector estimator
//!   - `style_dp`  shape (1, 8, 16)    — used by duration predictor

use anyhow::{bail, Context, Result};
use ndarray::Array3;
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct RawComponent {
    data: Vec<Vec<Vec<f32>>>,
    dims: Vec<usize>,
    #[serde(rename = "type")]
    dtype: String,
}

#[derive(Debug, Deserialize)]
struct RawVoice {
    style_ttl: RawComponent,
    style_dp: RawComponent,
}

/// A single voice's style tensors, ready to feed to the ONNX models.
#[derive(Debug, Clone)]
pub struct VoiceStyle {
    pub ttl: Array3<f32>, // (1, 50, 256)
    pub dp: Array3<f32>,  // (1, 8, 16)
}

impl VoiceStyle {
    pub fn load(path: &Path) -> Result<Self> {
        let f = File::open(path).with_context(|| format!("opening voice {}", path.display()))?;
        let raw: RawVoice =
            serde_json::from_reader(BufReader::new(f)).context("parsing voice style JSON")?;
        Ok(Self {
            ttl: component_to_array(&raw.style_ttl, "style_ttl")?,
            dp: component_to_array(&raw.style_dp, "style_dp")?,
        })
    }
}

fn component_to_array(c: &RawComponent, name: &str) -> Result<Array3<f32>> {
    if c.dtype != "float32" {
        bail!("voice {name}: expected float32, got {}", c.dtype);
    }
    if c.dims.len() != 3 {
        bail!("voice {name}: expected 3-D dims, got {:?}", c.dims);
    }
    let (b, t, d) = (c.dims[0], c.dims[1], c.dims[2]);
    let mut flat = Vec::with_capacity(b * t * d);
    if c.data.len() != b {
        bail!("voice {name}: data batch {} != dims[0] {}", c.data.len(), b);
    }
    for (bi, batch) in c.data.iter().enumerate() {
        if batch.len() != t {
            bail!("voice {name}: batch {bi} time {} != dims[1] {}", batch.len(), t);
        }
        for (ti, row) in batch.iter().enumerate() {
            if row.len() != d {
                bail!(
                    "voice {name}: batch {bi} time {ti} dim {} != dims[2] {}",
                    row.len(),
                    d
                );
            }
            flat.extend_from_slice(row);
        }
    }
    Array3::<f32>::from_shape_vec((b, t, d), flat).context("assembling voice array")
}
