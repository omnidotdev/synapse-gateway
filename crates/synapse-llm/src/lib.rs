//! Core LLM routing crate for Synapse
//!
//! Provides a unified interface over multiple LLM providers (`OpenAI`, Anthropic,
//! Google, AWS Bedrock) with automatic model routing, format conversion, and
//! both `OpenAI`-compatible and Anthropic-compatible API endpoints.

#![allow(clippy::must_use_candidate, clippy::missing_errors_doc)]

pub mod convert;
pub mod discovery;
pub mod error;
pub mod health;
pub mod protocol;
pub mod provider;
pub mod router;
pub mod routing;
pub mod types;

pub use error::LlmError;
pub use provider::{Provider, ProviderCapabilities};
pub use router::{LlmState, llm_router};
pub use routing::{ModelRouter, ResolvedModel};
pub use types::{CompletionRequest, CompletionResponse, StreamEvent};
