#![allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_const_for_fn,
    clippy::module_name_repetitions
)]

pub mod error;
pub mod provider;
pub mod types;

pub use error::{EmbeddingsError, Result};
pub use provider::EmbeddingsProvider;
pub use types::{EmbedInput, EmbeddingData, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage};
