mod client_id;
mod cors;
mod csrf;
mod health;
mod rate_limit;
mod request_context;

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
    pub async fn new(config: Config) -> anyhow::Result<Self> {
        let listen_address = config
            .server
            .listen_address
            .unwrap_or_else(|| SocketAddr::from(([0, 0, 0, 0], 3000)));

        // Initialize subsystems (STT/TTS borrow config, so build before LLM consumes it)
        let stt_state = stt::build_server(&config)?;
        let tts_state = tts::build_server(&config)?;
        let llm_state = LlmState::from_config(config.llm).await?;
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

        // Apply middleware layers (outermost first)

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

        // Request context (innermost — runs before route handlers)
        app = app.layer(axum::middleware::from_fn(request_context::request_context_middleware));

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
