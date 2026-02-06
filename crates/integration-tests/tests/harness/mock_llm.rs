//! Mock LLM backend server for integration tests
//!
//! Implements a minimal OpenAI-compatible API that returns canned responses

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use axum::extract::State;
use axum::response::IntoResponse;
use axum::{Json, Router, routing};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

/// Mock LLM backend that returns predictable responses
pub struct MockLlm {
    addr: SocketAddr,
    shutdown: CancellationToken,
    state: Arc<MockLlmState>,
}

struct MockLlmState {
    request_count: AtomicU32,
    completion_count: AtomicU32,
}

impl MockLlm {
    /// Start the mock server, returning immediately
    pub async fn start() -> anyhow::Result<Self> {
        let state = Arc::new(MockLlmState {
            request_count: AtomicU32::new(0),
            completion_count: AtomicU32::new(0),
        });

        let app = Router::new()
            .route("/v1/chat/completions", routing::post(handle_chat_completions))
            .route("/v1/models", routing::get(handle_models))
            .with_state(Arc::clone(&state));

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        let shutdown = CancellationToken::new();
        let shutdown_clone = shutdown.clone();

        tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    shutdown_clone.cancelled().await;
                })
                .await
                .ok();
        });

        Ok(Self { addr, shutdown, state })
    }

    /// Base URL for configuring the mock as a provider
    ///
    /// Includes `/v1` since the OpenAI provider appends paths like `/chat/completions`
    pub fn base_url(&self) -> String {
        format!("http://{}/v1", self.addr)
    }

    /// Number of completion requests received
    pub fn completion_count(&self) -> u32 {
        self.state.completion_count.load(Ordering::Relaxed)
    }
}

impl Drop for MockLlm {
    fn drop(&mut self) {
        self.shutdown.cancel();
    }
}

// -- Wire types matching OpenAI format --

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    #[allow(dead_code)]
    role: String,
    #[allow(dead_code)]
    content: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct Choice {
    index: u32,
    message: ResponseMessage,
    finish_reason: String,
}

#[derive(Debug, Serialize)]
struct ResponseMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Debug, Serialize)]
struct ModelListResponse {
    object: String,
    data: Vec<ModelObject>,
}

#[derive(Debug, Serialize)]
struct ModelObject {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
}

async fn handle_chat_completions(
    State(state): State<Arc<MockLlmState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    state.request_count.fetch_add(1, Ordering::Relaxed);
    state.completion_count.fetch_add(1, Ordering::Relaxed);

    let response = ChatCompletionResponse {
        id: "chatcmpl-test-123".to_owned(),
        object: "chat.completion".to_owned(),
        created: 1_700_000_000,
        model: req.model,
        choices: vec![Choice {
            index: 0,
            message: ResponseMessage {
                role: "assistant".to_owned(),
                content: "Hello from mock LLM".to_owned(),
            },
            finish_reason: "stop".to_owned(),
        }],
        usage: Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
        },
    };

    Json(response)
}

async fn handle_models(State(state): State<Arc<MockLlmState>>) -> impl IntoResponse {
    state.request_count.fetch_add(1, Ordering::Relaxed);

    let response = ModelListResponse {
        object: "list".to_owned(),
        data: vec![ModelObject {
            id: "mock-model-1".to_owned(),
            object: "model".to_owned(),
            created: 1_700_000_000,
            owned_by: "mock".to_owned(),
        }],
    };

    Json(response)
}
