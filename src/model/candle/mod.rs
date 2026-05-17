//! Pure-Rust (candle) port of the four Supertonic 3 models. Mirrors
//! `src/model/mod.rs`'s ort-backed implementation; later they'll sit
//! behind a `Backend` trait so callers can swap engines.

pub mod duration_predictor;
pub mod text_encoder;
pub mod vector_estimator;
pub mod vocoder;

pub use duration_predictor::CandleDurationPredictor;
pub use text_encoder::CandleTextEncoder;
pub use vector_estimator::CandleVectorEstimator;
pub use vocoder::CandleVocoder;
