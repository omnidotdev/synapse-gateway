//! Programmatic configuration builder for integration tests

use std::net::SocketAddr;

use secrecy::SecretString;
use synapse_config::{
    Config, CorsConfig, CsrfConfig, HealthConfig, LlmConfig, LlmProviderConfig, LlmProviderType, McpConfig,
    ModelConfig, RateLimitConfig, ServerConfig, SttConfig, TtsConfig,
};

/// Builder for constructing test configurations
pub struct ConfigBuilder {
    config: Config,
}

impl ConfigBuilder {
    /// Create a new builder with minimal defaults
    pub fn new() -> Self {
        Self {
            config: Config {
                server: ServerConfig {
                    listen_address: Some(SocketAddr::from(([127, 0, 0, 1], 0))),
                    health: HealthConfig {
                        enabled: true,
                        ..HealthConfig::default()
                    },
                    ..ServerConfig::default()
                },
                llm: LlmConfig::default(),
                mcp: McpConfig::default(),
                stt: SttConfig::default(),
                tts: TtsConfig::default(),
                telemetry: None,
                proxy: None,
            },
        }
    }

    /// Add an OpenAI-compatible provider pointed at a mock backend
    pub fn with_openai_provider(mut self, name: &str, base_url: &str) -> Self {
        self.config.llm.providers.insert(
            name.to_owned(),
            LlmProviderConfig {
                provider_type: LlmProviderType::Openai,
                api_key: Some(SecretString::from("test-key")),
                base_url: Some(base_url.parse().expect("valid URL")),
                models: ModelConfig::default(),
                headers: Vec::new(),
                forward_authorization: false,
                rate_limit: None,
            },
        );
        self
    }

    /// Set CORS configuration
    pub fn with_cors(mut self, config: CorsConfig) -> Self {
        self.config.server.cors = Some(config);
        self
    }

    /// Set CSRF configuration
    pub fn with_csrf(mut self, config: CsrfConfig) -> Self {
        self.config.server.csrf = Some(config);
        self
    }

    /// Set rate limit configuration
    pub fn with_rate_limit(mut self, config: RateLimitConfig) -> Self {
        self.config.server.rate_limit = Some(config);
        self
    }

    /// Disable health endpoint
    pub fn without_health(mut self) -> Self {
        self.config.server.health.enabled = false;
        self
    }

    /// Build the final config
    pub fn build(self) -> Config {
        self.config
    }
}
