mod auth;
mod billing_identity;
mod client_id;
mod cors;
mod csrf;
mod entitlement;
mod entitlement_cache;
mod health;
mod invalidate;
mod rate_limit;
mod request_context;
mod webhook;

use std::net::SocketAddr;
use std::sync::Arc;

use axum::Router;
use synapse_config::Config;
use synapse_llm::LlmState;
use synapse_mcp::McpState;
use tower_http::trace::TraceLayer;

/// Assembled server with all routes and middleware
pub struct Server {
    router: Router,
    listen_address: SocketAddr,
}

impl Server {
    /// Build the server from configuration
    ///
    /// # Errors
    ///
    /// Returns an error if subsystem initialization (LLM, MCP, STT, TTS) or
    /// rate-limiter construction fails
    #[allow(clippy::too_many_lines)]
    pub async fn new(config: Config) -> anyhow::Result<Self> {
        let listen_address = config
            .server
            .listen_address
            .unwrap_or_else(|| SocketAddr::from(([0, 0, 0, 0], 3000)));

        // Initialize subsystems that borrow config before LLM consumes it
        let stt_state = stt::build_server(&config)?;
        let tts_state = tts::build_server(&config)?;
        let embeddings_state = synapse_embeddings::build_server(&config)?;
        let imagegen_state = synapse_imagegen::build_server(&config)?;
        let mut llm_state = LlmState::from_config(config.llm).await?;

        // Configure billing for LLM state when enabled
        if let Some(ref billing_config) = config.billing
            && billing_config.enabled
        {
            // Set managed provider keys
            use secrecy::SecretString;
            use std::collections::HashMap;

            let mut managed_keys: HashMap<String, SecretString> = HashMap::new();
            let mut managed_margins: HashMap<String, f64> = HashMap::new();
            for (name, provider_config) in &billing_config.managed_providers {
                managed_keys.insert(name.clone(), provider_config.api_key.clone());
                managed_margins.insert(name.clone(), provider_config.margin);
            }
            llm_state.set_managed_providers(managed_keys, managed_margins);

            // Attach usage recorder
            let recorder_client = synapse_billing::AetherClient::new(
                billing_config.aether_url.clone(),
                billing_config.app_id.clone(),
                billing_config.service_api_key.clone(),
            )?;
            let meter_keys = synapse_billing::MeterKeys {
                input_tokens: billing_config.meters.input_tokens.clone(),
                output_tokens: billing_config.meters.output_tokens.clone(),
                requests: billing_config.meters.requests.clone(),
            };
            let recorder = synapse_billing::UsageRecorder::new(recorder_client, meter_keys);
            llm_state.set_usage_recorder(recorder);
        }

        let mcp_state = Arc::new(McpState::new(&config.mcp).await?);

        // Build base router with feature routes
        let mut app = Router::new();

        // Health check
        if config.server.health.enabled {
            app = app.route(&config.server.health.path, axum::routing::get(health::health_handler));
        }

        // LLM routes
        app = app.merge(synapse_llm::llm_router(llm_state));

        // MCP routes
        app = app.merge(synapse_mcp::mcp_router(mcp_state));

        // STT routes
        app = app.merge(stt::endpoint_router().with_state(stt_state));

        // TTS routes
        app = app.merge(tts::endpoint_router().with_state(tts_state));

        // Embeddings routes
        app = app.merge(synapse_embeddings::endpoint_router().with_state(embeddings_state));

        // Image generation routes
        app = app.merge(synapse_imagegen::endpoint_router().with_state(imagegen_state));

        // Apply middleware layers (innermost first)

        // Request context (innermost — collects auth/identity data, runs just before handlers)
        app = app.layer(axum::middleware::from_fn(request_context::request_context_middleware));

        // Tracing
        app = app.layer(TraceLayer::new_for_http());

        // CORS
        if let Some(ref cors_config) = config.server.cors {
            app = app.layer(cors::cors_layer(cors_config));
        }

        // CSRF protection
        if let Some(ref csrf_config) = config.server.csrf
            && csrf_config.enabled
        {
            let header_name = csrf_config.header_name.clone();
            app = app.layer(axum::middleware::from_fn(move |req, next| {
                let header_name = header_name.clone();
                async move { csrf::csrf_middleware(header_name, req, next).await }
            }));
        }

        // API key authentication
        if let Some(ref auth_config) = config.auth
            && auth_config.enabled
        {
            let resolver = synapse_auth::ApiKeyResolver::new(
                auth_config.api_url.clone(),
                auth_config.gateway_secret.clone(),
                std::time::Duration::from_secs(auth_config.cache_ttl_seconds),
                auth_config.cache_capacity,
                auth_config.tls_skip_verify,
            )?;

            // Cache invalidation endpoint
            let invalidate_state = invalidate::InvalidateState {
                resolver: resolver.clone(),
                gateway_secret: auth_config.gateway_secret.clone(),
            };
            app = app.route(
                "/internal/invalidate-key",
                axum::routing::post(invalidate::invalidate_key_handler)
                    .with_state(invalidate_state),
            );

            let usage_reporter = synapse_auth::UsageReporter::spawn(
                auth_config.api_url.clone(),
                auth_config.gateway_secret.clone(),
                std::time::Duration::from_secs(10),
            );

            let public_paths = auth_config.public_paths.clone();
            let reporter = Some(usage_reporter);
            app = app.layer(axum::middleware::from_fn(move |req, next| {
                let resolver = resolver.clone();
                let public_paths = public_paths.clone();
                let reporter = reporter.clone();
                async move {
                    auth::auth_middleware(resolver, public_paths, reporter, req, next).await
                }
            }));
        }

        // Billing identity (after OAuth, before ClientId)
        if let Some(ref billing_config) = config.billing
            && billing_config.enabled
        {
            let billing = billing_config.clone();
            app = app.layer(axum::middleware::from_fn(move |req, next| {
                let config = billing.clone();
                async move { billing_identity::billing_identity_middleware(config, req, next).await }
            }));
        }

        // Client identification
        if let Some(ref client_id_config) = config.server.client_identification {
            let cid_config = client_id_config.clone();
            app = app.layer(axum::middleware::from_fn(move |req, next| {
                let config = cid_config.clone();
                async move { client_id::client_id_middleware(config, req, next).await }
            }));
        }

        // Rate limiting
        if let Some(ref rl_config) = config.server.rate_limit {
            let limiter = Arc::new(synapse_ratelimit::create_request_limiter(rl_config)?);
            app = app.layer(axum::middleware::from_fn(move |req, next| {
                let limiter = Arc::clone(&limiter);
                async move { rate_limit::rate_limit_middleware_arc(limiter, req, next).await }
            }));
        }

        // Entitlement gate (after billing identity, before request context)
        if let Some(ref billing_config) = config.billing
            && billing_config.enabled
        {
            let aether_client = synapse_billing::AetherClient::new(
                billing_config.aether_url.clone(),
                billing_config.app_id.clone(),
                billing_config.service_api_key.clone(),
            )?;
            let ent_state = entitlement::EntitlementState::new(aether_client, billing_config.clone());

            // Entitlement webhook endpoint (shares cache with middleware)
            if let Some(ref auth_config) = config.auth {
                let webhook_state = webhook::WebhookState {
                    cache: ent_state.cache(),
                    gateway_secret: auth_config.gateway_secret.clone(),
                };
                app = app.route(
                    "/webhooks/entitlements",
                    axum::routing::post(webhook::entitlement_webhook_handler)
                        .with_state(webhook_state),
                );
            }

            app = app.layer(axum::middleware::from_fn(move |req, next| {
                let state = ent_state.clone();
                async move { entitlement::entitlement_middleware(state, req, next).await }
            }));
        }

        Ok(Self {
            router: app,
            listen_address,
        })
    }

    /// Get the configured listen address
    #[must_use]
    pub const fn listen_address(&self) -> SocketAddr {
        self.listen_address
    }

    /// Consume the server and return the inner router
    ///
    /// Useful for testing when the caller manages the listener
    pub fn into_router(self) -> Router {
        self.router
    }

    /// Start serving requests
    ///
    /// Blocks until the cancellation token is triggered.
    ///
    /// # Errors
    ///
    /// Returns an error if binding the TCP listener or serving fails
    pub async fn serve(self, shutdown: tokio_util::sync::CancellationToken) -> anyhow::Result<()> {
        let listener = tokio::net::TcpListener::bind(self.listen_address).await?;
        let local_addr = listener.local_addr()?;
        tracing::info!(%local_addr, "server listening");

        axum::serve(listener, self.router)
            .with_graceful_shutdown(async move {
                shutdown.cancelled().await;
                tracing::info!("graceful shutdown initiated");
            })
            .await?;

        Ok(())
    }
}
