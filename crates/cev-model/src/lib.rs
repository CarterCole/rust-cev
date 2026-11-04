//! Neural backends for cev: prefill-only decoders in candle (Qwen3, Qwen3.5 and
//! LFM2 families) that read typed answers from option-code logits, pointer
//! heads or decision heads, with the vision and audio encoders the
//! multimodal decision models carry.

#[cfg(not(target_arch = "wasm32"))]
pub mod audio;
pub mod decoder;
pub mod engine;
#[cfg(not(target_arch = "wasm32"))]
pub mod hub;
#[cfg(not(target_arch = "wasm32"))]
pub mod hybrid;
pub mod lfm2;
#[cfg(not(target_arch = "wasm32"))]
pub mod load;
#[cfg(not(target_arch = "wasm32"))]
pub mod lora;
#[cfg(not(target_arch = "wasm32"))]
pub mod media;
#[cfg(not(target_arch = "wasm32"))]
pub mod omni;
pub mod quant;
pub mod qwen3;
pub mod qwen35;
pub mod qwen35_vision;
#[cfg(not(target_arch = "wasm32"))]
pub mod siglip2;
pub mod weights;

pub use engine::{Engine, EngineOptions};
#[cfg(not(target_arch = "wasm32"))]
pub use load::load;
