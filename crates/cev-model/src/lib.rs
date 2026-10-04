//! Neural backend for cev: a prefill-only Qwen3 decoder in candle that reads
//! typed answers from the logits of single-token option codes.

pub mod engine;
#[cfg(not(target_arch = "wasm32"))]
pub mod hub;
pub mod quant;
pub mod qwen3;
pub mod weights;

pub use engine::{Engine, EngineOptions};
