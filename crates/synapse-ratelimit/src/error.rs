use thiserror::Error;

/// Rate limiting errors
#[derive(Debug, Error)]
pub enum RateLimitError {
    /// Configuration error
    #[error("rate limit configuration error: {0}")]
    Config(String),

    /// Redis connection error
    #[error("redis connection error: {0}")]
    Redis(String),

    /// Rate limit exceeded
    #[error("rate limit exceeded")]
    Exceeded {
        /// Seconds until the limit resets
        retry_after: u64,
    },

    /// Internal error
    #[error("rate limit internal error: {0}")]
    Internal(String),
}
