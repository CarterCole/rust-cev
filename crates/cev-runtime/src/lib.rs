//! Decision service with a durable decision/feedback log and online learning.

pub mod adapter;
pub mod service;
pub mod store;

pub use adapter::LearnConfig;
pub use service::{Cev, CevError, CevResult, ImportStats, RuntimeConfig, TaskInfo};
pub use store::{AdapterRow, ExportFilter, ExportRow, Store};
