//! Neural backends for cev: prefill-only decoders in candle (Qwen3, Qwen3.5 and
//! LFM2 families) that read typed answers from option-code logits, pointer
//! heads or decision heads, with the vision and audio encoders the
//! multimodal decision models carry.

pub mod audio;
pub mod decoder;
pub mod engine;
#[cfg(not(target_arch = "wasm32"))]
pub mod hub;
pub mod hybrid;
pub mod lfm2;
pub mod load;
pub mod lora;
pub mod media;
pub mod omni;
pub mod quant;
pub mod qwen3;
pub mod qwen35;
pub mod qwen35_vision;
pub mod siglip2;
pub mod weights;

pub use engine::{Engine, EngineOptions};
pub use load::{MemoryModel, from_memory};
#[cfg(not(target_arch = "wasm32"))]
pub use load::load;
