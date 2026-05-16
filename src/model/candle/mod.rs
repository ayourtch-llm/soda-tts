//! Pure-Rust (candle) port of the four Supertonic 3 models. Mirrors
//! `src/model/mod.rs`'s ort-backed implementation; later they'll sit
//! behind a `Backend` trait so callers can swap engines.

pub mod duration_predictor;
pub mod vocoder;

pub use duration_predictor::CandleDurationPredictor;
pub use vocoder::CandleVocoder;
