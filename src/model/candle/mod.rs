//! Pure-Rust (candle) port of the four Supertonic 3 models. Mirrors
//! `src/model/mod.rs`'s ort-backed implementation; later they'll sit
//! behind a `Backend` trait so callers can swap engines.

pub mod vocoder;

pub use vocoder::CandleVocoder;
