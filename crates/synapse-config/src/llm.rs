use std::collections::HashMap;

use indexmap::IndexMap;
use secrecy::SecretString;
use serde::Deserialize;
use url::Url;

use crate::headers::HeaderRuleConfig;

/// Top-level LLM configuration
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LlmConfig {
    /// LLM provider configurations keyed by name
    #[serde(default)]
    pub providers: IndexMap<String, LlmProviderConfig>,
}

/// Configuration for a single LLM provider
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LlmProviderConfig {
    /// Provider protocol type
    #[serde(rename = "type")]
    pub provider_type: LlmProviderType,
    /// API key for authentication
    #[serde(default)]
    pub api_key: Option<SecretString>,
    /// Base URL override
    #[serde(default)]
    pub base_url: Option<Url>,
    /// Model configuration
    #[serde(default)]
    pub models: ModelConfig,
    /// Header rules for this provider
    #[serde(default)]
    pub headers: Vec<HeaderRuleConfig>,
    /// Forward the client's bearer token to the provider
    #[serde(default)]
    pub forward_authorization: bool,
    /// Rate limit for this provider (requests per window)
    #[serde(default)]
    pub rate_limit: Option<ProviderRateLimit>,
}

/// Supported LLM provider protocols
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmProviderType {
    /// OpenAI-compatible API
    Openai,
    /// Anthropic Messages API
    Anthropic,
    /// Google Generative Language API
    Google,
    /// AWS Bedrock
    Bedrock(BedrockConfig),
}

/// AWS Bedrock-specific configuration
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BedrockConfig {
    /// AWS region
    pub region: String,
    /// Access key ID (optional, uses default credential chain if absent)
    #[serde(default)]
    pub access_key_id: Option<SecretString>,
    /// Secret access key
    #[serde(default)]
    pub secret_access_key: Option<SecretString>,
}

/// Model configuration for a provider
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// Include models matching these patterns (regex)
    #[serde(default)]
    pub include: Vec<String>,
    /// Exclude models matching these patterns (regex)
    #[serde(default)]
    pub exclude: Vec<String>,
    /// Per-model overrides
    #[serde(default)]
    pub overrides: HashMap<String, ModelOverride>,
}

/// Per-model configuration overrides
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelOverride {
    /// Custom display name
    #[serde(default)]
    pub alias: Option<String>,
    /// Rate limit override for this model
    #[serde(default)]
    pub rate_limit: Option<ProviderRateLimit>,
}

/// Rate limit configuration for a provider or model
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderRateLimit {
    /// Maximum requests per window
    pub requests: u32,
    /// Window duration (e.g. "1m", "1h")
    pub window: String,
}
