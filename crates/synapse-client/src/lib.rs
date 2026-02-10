#![allow(clippy::must_use_candidate, clippy::missing_errors_doc)]

//! Typed Rust HTTP client for the Synapse AI router
//!
//! Provides a high-level interface to Synapse's LLM, MCP, STT, and TTS
//! endpoints with streaming support

mod client;
pub mod error;
pub mod types;

pub use client::SynapseClient;
pub use error::{SynapseClientError, Result};
pub use types::*;
