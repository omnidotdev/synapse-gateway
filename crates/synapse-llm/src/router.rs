//! Axum route handlers for OpenAI-compatible and Anthropic-compatible endpoints

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::{Json, Router, routing};
use futures_util::{Stream, StreamExt};
use synapse_config::{FailoverConfig, LlmConfig, LlmProviderType, RoutingConfig};
use synapse_core::RequestContext;
use synapse_routing::{FeedbackTracker, ModelRegistry, RequestFeedback};

use crate::convert;
use crate::discovery;
use crate::error::LlmError;
use crate::health::ProviderHealthTracker;
use crate::protocol::anthropic::{AnthropicRequest, AnthropicResponse};
use crate::protocol::openai::{OpenAiModel, OpenAiModelList, OpenAiRequest, OpenAiResponse};
use crate::provider::Provider;
use crate::routing::ModelRouter;
use crate::types::{CompletionRequest, CompletionResponse, StreamEvent};

/// Virtual model names that trigger smart routing
const ROUTING_CLASSES: &[&str] = &["auto", "fast", "best", "cheap"];

/// Shared state for LLM route handlers
#[derive(Clone)]
pub struct LlmState {
    inner: Arc<LlmStateInner>,
}

struct LlmStateInner {
    router: ModelRouter,
    providers: HashMap<String, Arc<dyn Provider>>,
    health: ProviderHealthTracker,
    failover: FailoverConfig,
    routing_config: RoutingConfig,
    model_registry: ModelRegistry,
    feedback: FeedbackTracker,
}

impl LlmState {
    /// Build `LlmState` from configuration, constructing all providers
    ///
    /// # Errors
    ///
    /// Returns an error if any provider fails to initialize.
    pub async fn from_config(config: LlmConfig) -> Result<Self, LlmError> {
        let mut providers: HashMap<String, Arc<dyn Provider>> = HashMap::new();

        for (name, provider_config) in &config.providers {
            let provider: Arc<dyn Provider> = match &provider_config.provider_type {
                LlmProviderType::Openai => Arc::new(crate::provider::openai::OpenAiProvider::new(
                    name.clone(),
                    provider_config,
                )?),
                LlmProviderType::Anthropic => Arc::new(crate::provider::anthropic::AnthropicProvider::new(
                    name.clone(),
                    provider_config,
                )?),
                LlmProviderType::Google => Arc::new(crate::provider::google::GoogleProvider::new(
                    name.clone(),
                    provider_config,
                )?),
                LlmProviderType::Bedrock(_) => {
                    Arc::new(crate::provider::bedrock::BedrockProvider::new(name.clone(), provider_config).await?)
                }
            };

            providers.insert(name.clone(), provider);
        }

        let health = ProviderHealthTracker::new(config.failover.circuit_breaker.clone());
        let failover = config.failover.clone();
        let routing_config = config.routing.clone();
        let model_registry = ModelRegistry::from_config(&config.routing.models);
        let router = ModelRouter::new(&config);
        let feedback = FeedbackTracker::new();

        // Start background model discovery
        discovery::start_discovery(config, router.known_models());

        Ok(Self {
            inner: Arc::new(LlmStateInner {
                router,
                providers,
                health,
                failover,
                routing_config,
                model_registry,
                feedback,
            }),
        })
    }

    /// Resolve a model name and get the corresponding provider
    ///
    /// Handles both normal model names and virtual routing classes
    /// ("auto", "fast", "best", "cheap") when smart routing is enabled.
    async fn resolve_provider(
        &self,
        model: &str,
        request: &CompletionRequest,
    ) -> Result<(String, String, Arc<dyn Provider>), LlmError> {
        // Check for virtual routing classes
        if self.inner.routing_config.enabled && ROUTING_CLASSES.contains(&model) {
            return self.resolve_via_routing(model, request);
        }

        let resolved = self.inner.router.resolve(model).await?;
        let provider = self
            .inner
            .providers
            .get(&resolved.provider_name)
            .ok_or_else(|| LlmError::ProviderNotFound {
                provider: resolved.provider_name.clone(),
            })?;
        Ok((resolved.provider_name.clone(), resolved.model_id, Arc::clone(provider)))
    }

    /// Resolve a virtual model name via the smart routing system
    fn resolve_via_routing(
        &self,
        routing_class: &str,
        request: &CompletionRequest,
    ) -> Result<(String, String, Arc<dyn Provider>), LlmError> {
        // Convert internal messages to JSON values for analysis
        let messages: Vec<serde_json::Value> = request
            .messages
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": serde_json::to_value(&m.role).unwrap_or_default(),
                    "content": m.content.as_text()
                })
            })
            .collect();

        let has_tools = request.tools.as_ref().is_some_and(|t| !t.is_empty());

        // Apply routing class overrides
        let config = self.map_routing_class(routing_class);

        let decision = synapse_routing::route_request(
            &messages,
            has_tools,
            &self.inner.model_registry,
            &config,
            Some(&self.inner.feedback),
        )
        .map_err(|e| LlmError::InvalidRequest(format!("routing failed: {e}")))?;

        let provider = self
            .inner
            .providers
            .get(&decision.provider)
            .ok_or_else(|| LlmError::ProviderNotFound {
                provider: decision.provider.clone(),
            })?;

        tracing::info!(
            routing_class,
            provider = %decision.provider,
            model = %decision.model,
            reason = ?decision.reason,
            "smart routing resolved virtual model"
        );

        Ok((decision.provider, decision.model, Arc::clone(provider)))
    }

    /// Map a routing class name to an appropriate routing config
    fn map_routing_class(&self, class: &str) -> RoutingConfig {
        let mut config = self.inner.routing_config.clone();

        match class {
            "fast" | "cheap" => {
                // Force cost strategy for cheap/fast
                config.strategy = synapse_config::RoutingStrategy::Cost;
            }
            "best" => {
                // Force threshold strategy biased toward high quality
                config.strategy = synapse_config::RoutingStrategy::Threshold;
                config.threshold.quality_floor = 0.9;
            }
            // "auto" uses whatever strategy is configured
            _ => {}
        }

        config
    }

    /// Execute a non-streaming completion with failover support
    async fn complete_with_failover(
        &self,
        request: &CompletionRequest,
        context: &RequestContext,
        provider_name: &str,
        model_id: &str,
        provider: &Arc<dyn Provider>,
    ) -> Result<CompletionResponse, LlmError> {
        // Try primary provider
        let mut req = request.clone();
        model_id.clone_into(&mut req.model);

        let start = Instant::now();
        match provider.complete(&req, context).await {
            Ok(response) => {
                self.inner.health.record_success(provider_name);
                self.inner.feedback.record(&RequestFeedback {
                    provider: provider_name.to_owned(),
                    model: model_id.to_owned(),
                    latency: start.elapsed(),
                    success: true,
                    input_tokens: response.usage.as_ref().map(|u| u.prompt_tokens),
                    output_tokens: response.usage.as_ref().map(|u| u.completion_tokens),
                });
                Ok(response)
            }
            Err(e) => {
                self.inner.health.record_failure(provider_name);
                self.inner.feedback.record(&RequestFeedback {
                    provider: provider_name.to_owned(),
                    model: model_id.to_owned(),
                    latency: start.elapsed(),
                    success: false,
                    input_tokens: None,
                    output_tokens: None,
                });

                if !self.inner.failover.enabled || !e.is_retryable() {
                    return Err(e);
                }

                tracing::warn!(
                    provider = provider_name,
                    model = model_id,
                    error = %e,
                    "primary provider failed, attempting failover"
                );

                let alternatives = ModelRouter::find_equivalents(
                    provider_name,
                    model_id,
                    &self.inner.failover.equivalence_groups,
                );

                // max_attempts includes the primary, so remaining = max_attempts - 1
                let remaining = self.inner.failover.max_attempts.saturating_sub(1);
                let mut last_error = e;

                for (alt_provider, alt_model) in alternatives.into_iter().take(remaining) {
                    if !self.inner.health.is_available(&alt_provider) {
                        tracing::debug!(
                            provider = %alt_provider,
                            "skipping unhealthy provider"
                        );
                        continue;
                    }

                    let Some(alt_provider_impl) = self.inner.providers.get(&alt_provider) else {
                        continue;
                    };

                    tracing::warn!(
                        from_provider = provider_name,
                        to_provider = %alt_provider,
                        to_model = %alt_model,
                        "failing over to alternative provider"
                    );

                    let mut alt_req = request.clone();
                    alt_req.model.clone_from(&alt_model);

                    match alt_provider_impl.complete(&alt_req, context).await {
                        Ok(response) => {
                            self.inner.health.record_success(&alt_provider);
                            return Ok(response);
                        }
                        Err(e) => {
                            self.inner.health.record_failure(&alt_provider);
                            tracing::warn!(
                                provider = %alt_provider,
                                error = %e,
                                "failover provider also failed"
                            );
                            last_error = e;
                        }
                    }
                }

                Err(last_error)
            }
        }
    }

    /// Execute a streaming completion with failover support
    ///
    /// Failover is only possible before streaming starts. If the initial
    /// `complete_stream()` call returns an error, alternatives are tried.
    async fn complete_stream_with_failover(
        &self,
        request: &CompletionRequest,
        context: &RequestContext,
        provider_name: &str,
        model_id: &str,
        provider: &Arc<dyn Provider>,
    ) -> Result<(String, Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send>>), LlmError> {
        // Try primary provider
        let mut req = request.clone();
        model_id.clone_into(&mut req.model);

        let start = Instant::now();
        match provider.complete_stream(&req, context).await {
            Ok(stream) => {
                self.inner.health.record_success(provider_name);
                // Record success feedback for stream initiation
                self.inner.feedback.record(&RequestFeedback {
                    provider: provider_name.to_owned(),
                    model: model_id.to_owned(),
                    latency: start.elapsed(),
                    success: true,
                    input_tokens: None,
                    output_tokens: None,
                });
                Ok((model_id.to_owned(), stream))
            }
            Err(e) => {
                self.inner.health.record_failure(provider_name);
                self.inner.feedback.record(&RequestFeedback {
                    provider: provider_name.to_owned(),
                    model: model_id.to_owned(),
                    latency: start.elapsed(),
                    success: false,
                    input_tokens: None,
                    output_tokens: None,
                });

                if !self.inner.failover.enabled || !e.is_retryable() {
                    return Err(e);
                }

                tracing::warn!(
                    provider = provider_name,
                    model = model_id,
                    error = %e,
                    "primary provider streaming failed, attempting failover"
                );

                let alternatives = ModelRouter::find_equivalents(
                    provider_name,
                    model_id,
                    &self.inner.failover.equivalence_groups,
                );

                let remaining = self.inner.failover.max_attempts.saturating_sub(1);
                let mut last_error = e;

                for (alt_provider, alt_model) in alternatives.into_iter().take(remaining) {
                    if !self.inner.health.is_available(&alt_provider) {
                        continue;
                    }

                    let Some(alt_provider_impl) = self.inner.providers.get(&alt_provider) else {
                        continue;
                    };

                    tracing::warn!(
                        from_provider = provider_name,
                        to_provider = %alt_provider,
                        to_model = %alt_model,
                        "failing over streaming to alternative provider"
                    );

                    let mut alt_req = request.clone();
                    alt_req.model.clone_from(&alt_model);

                    match alt_provider_impl.complete_stream(&alt_req, context).await {
                        Ok(stream) => {
                            self.inner.health.record_success(&alt_provider);
                            return Ok((alt_model, stream));
                        }
                        Err(e) => {
                            self.inner.health.record_failure(&alt_provider);
                            last_error = e;
                        }
                    }
                }

                Err(last_error)
            }
        }
    }

    /// Execute a streaming cascade: buffer initial model's response, evaluate
    /// confidence, then either replay the buffer or re-request with the
    /// escalation model
    async fn complete_stream_with_cascade(
        &self,
        request: &CompletionRequest,
        context: &RequestContext,
        provider_name: &str,
        model_id: &str,
        provider: &Arc<dyn Provider>,
        cascade_config: &synapse_config::CascadeConfig,
    ) -> Result<(String, Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send>>), LlmError> {
        // Get the escalation model from the routing decision's alternatives
        let escalation = self.resolve_escalation_model(cascade_config)?;

        // Stream from the initial (cheap) model
        let (initial_model, mut stream) = self
            .complete_stream_with_failover(request, context, provider_name, model_id, provider)
            .await?;

        // Buffer stream events, collecting text content
        let mut buffered_events: Vec<StreamEvent> = Vec::new();
        let mut buffered_text = String::new();
        let mut buffer_bytes: usize = 0;
        let mut committed = false;

        let timeout = tokio::time::Duration::from_secs(cascade_config.buffer_timeout_secs);
        let deadline = tokio::time::Instant::now() + timeout;

        loop {
            let next = tokio::time::timeout_at(deadline, stream.next()).await;

            match next {
                // Timeout fired — commit to initial model
                Err(_elapsed) => {
                    tracing::debug!(
                        model = %initial_model,
                        buffered_bytes = buffer_bytes,
                        "cascade buffer timeout, committing to initial model"
                    );
                    committed = true;
                    break;
                }
                // Stream ended
                Ok(None) => break,
                // Stream error — propagate
                Ok(Some(Err(e))) => return Err(e),
                // Stream event
                Ok(Some(Ok(event))) => {
                    // Track buffer size
                    if let StreamEvent::Delta(ref delta) = event
                        && let Some(ref content) = delta.content
                    {
                        buffer_bytes += content.len();
                        buffered_text.push_str(content);
                    }

                    buffered_events.push(event.clone());

                    // If buffer limit exceeded, commit
                    if buffer_bytes >= cascade_config.max_buffer_bytes {
                        tracing::debug!(
                            model = %initial_model,
                            buffered_bytes = buffer_bytes,
                            "cascade buffer limit exceeded, committing to initial model"
                        );
                        committed = true;
                        break;
                    }

                    // Done event means stream completed
                    if matches!(event, StreamEvent::Done) {
                        break;
                    }
                }
            }
        }

        // If committed early (buffer limit or timeout), replay buffer + remaining stream
        if committed {
            let remaining = stream;
            let replay: Vec<Result<StreamEvent, LlmError>> = buffered_events.into_iter().map(Ok).collect();
            let replay_stream = futures_util::stream::iter(replay);
            let combined = replay_stream.chain(remaining);
            return Ok((initial_model, Box::pin(combined)));
        }

        // Estimate input tokens for confidence check
        let query_tokens: usize = request
            .messages
            .iter()
            .map(|m| m.content.as_text().len() / 4)
            .sum();

        // Evaluate confidence on buffered response
        let is_confident = synapse_routing::strategy::cascade::evaluate_buffered_response(
            &buffered_text,
            query_tokens,
            cascade_config.confidence_threshold,
        );

        if is_confident {
            tracing::debug!(
                model = %initial_model,
                "cascade: initial model response is confident, replaying buffer"
            );
            let replay: Vec<Result<StreamEvent, LlmError>> = buffered_events.into_iter().map(Ok).collect();
            return Ok((initial_model, Box::pin(futures_util::stream::iter(replay))));
        }

        // Not confident — escalate to stronger model
        tracing::info!(
            initial_model = %initial_model,
            escalation_provider = %escalation.0,
            escalation_model = %escalation.1,
            "cascade: escalating to stronger model"
        );

        let (esc_provider_name, esc_model_id) = &escalation;
        let esc_provider = self
            .inner
            .providers
            .get(esc_provider_name)
            .ok_or_else(|| LlmError::ProviderNotFound {
                provider: esc_provider_name.clone(),
            })?;

        self.complete_stream_with_failover(request, context, esc_provider_name, esc_model_id, esc_provider)
            .await
    }

    /// Resolve the escalation model for cascade routing
    fn resolve_escalation_model(
        &self,
        cascade_config: &synapse_config::CascadeConfig,
    ) -> Result<(String, String), LlmError> {
        if let Some(ref configured) = cascade_config.escalation_model {
            let (provider, model) = configured
                .split_once('/')
                .ok_or_else(|| LlmError::InvalidRequest("invalid escalation model format".to_owned()))?;
            return Ok((provider.to_owned(), model.to_owned()));
        }

        // Default to best quality model in registry
        let best = self
            .inner
            .model_registry
            .best_quality()
            .ok_or_else(|| LlmError::InvalidRequest("no escalation model available".to_owned()))?;
        Ok((best.provider.clone(), best.model.clone()))
    }

    /// Check if the current routing strategy is cascade
    fn is_cascade_strategy(&self, model: &str) -> bool {
        if !self.inner.routing_config.enabled || !ROUTING_CLASSES.contains(&model) {
            return false;
        }
        let config = self.map_routing_class(model);
        matches!(config.strategy, synapse_config::RoutingStrategy::Cascade)
    }
}

/// Build the LLM router with all endpoints
pub fn llm_router(state: LlmState) -> Router {
    Router::new()
        // OpenAI-compatible endpoints
        .route("/v1/chat/completions", routing::post(openai_chat_completions))
        .route("/v1/models", routing::get(openai_list_models))
        // Anthropic-compatible endpoint
        .route("/v1/messages", routing::post(anthropic_messages))
        .with_state(state)
}

// -- OpenAI-compatible handlers --

/// Handle `POST /v1/chat/completions`
async fn openai_chat_completions(
    State(state): State<LlmState>,
    axum::Extension(context): axum::Extension<RequestContext>,
    Json(wire_request): Json<OpenAiRequest>,
) -> Response {
    let is_stream = wire_request.stream.unwrap_or(false);
    let original_model = wire_request.model.clone();
    let internal_request: CompletionRequest = wire_request.into();

    let (provider_name, model_id, provider) =
        match state.resolve_provider(&internal_request.model, &internal_request).await {
            Ok(r) => r,
            Err(e) => return error_to_openai_response(e),
        };

    if is_stream {
        // Use streaming cascade when strategy is Cascade
        let result = if state.is_cascade_strategy(&original_model) {
            state
                .complete_stream_with_cascade(
                    &internal_request,
                    &context,
                    &provider_name,
                    &model_id,
                    &provider,
                    &state.inner.routing_config.cascade,
                )
                .await
        } else {
            state
                .complete_stream_with_failover(&internal_request, &context, &provider_name, &model_id, &provider)
                .await
        };

        match result {
            Ok((actual_model, stream)) => openai_stream_response(stream, actual_model).into_response(),
            Err(e) => error_to_openai_response(e),
        }
    } else {
        match state
            .complete_with_failover(&internal_request, &context, &provider_name, &model_id, &provider)
            .await
        {
            Ok(response) => {
                let wire_response: OpenAiResponse = response.into();
                Json(wire_response).into_response()
            }
            Err(e) => error_to_openai_response(e),
        }
    }
}

/// Handle `GET /v1/models`
async fn openai_list_models(State(state): State<LlmState>) -> Response {
    let models = state.inner.router.list_models().await;

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let data: Vec<OpenAiModel> = models
        .into_iter()
        .map(|(display_name, _model_id)| OpenAiModel {
            id: display_name,
            object: "model".to_owned(),
            created: now,
            owned_by: "synapse".to_owned(),
        })
        .collect();

    let response = OpenAiModelList {
        object: "list".to_owned(),
        data,
    };

    Json(response).into_response()
}

/// Build a streaming SSE response in `OpenAI` format
fn openai_stream_response(
    stream: std::pin::Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send>>,
    model: String,
) -> Sse<impl Stream<Item = Result<Event, axum::Error>>> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let response_id = format!("chatcmpl-{now}");

    let event_stream = stream.map(move |result| match result {
        Ok(StreamEvent::Delta(delta)) => {
            let chunk = convert::openai::delta_to_openai_chunk(&delta, &response_id, &model, now);
            let data = serde_json::to_string(&chunk).unwrap_or_default();
            Ok(Event::default().data(data))
        }
        Ok(StreamEvent::Usage(usage)) => {
            let chunk = convert::openai::usage_to_openai_chunk(&usage, &response_id, &model, now);
            let data = serde_json::to_string(&chunk).unwrap_or_default();
            Ok(Event::default().data(data))
        }
        Ok(StreamEvent::Done) => Ok(Event::default().data("[DONE]")),
        Err(e) => {
            let error_data = serde_json::json!({
                "error": {
                    "message": e.to_string(),
                    "type": "streaming_error"
                }
            });
            Ok(Event::default().data(error_data.to_string()))
        }
    });

    Sse::new(event_stream).keep_alive(KeepAlive::default())
}

/// Convert an LLM error to an `OpenAI`-style JSON error response
#[allow(clippy::needless_pass_by_value)]
fn error_to_openai_response(error: LlmError) -> Response {
    use synapse_core::HttpError;

    let status = error.status_code();
    let body = serde_json::json!({
        "error": {
            "message": error.client_message(),
            "type": error.error_type(),
            "code": serde_json::Value::Null,
        }
    });

    (status, Json(body)).into_response()
}

// -- Anthropic-compatible handler --

/// Handle `POST /v1/messages`
async fn anthropic_messages(
    State(state): State<LlmState>,
    axum::Extension(context): axum::Extension<RequestContext>,
    Json(wire_request): Json<AnthropicRequest>,
) -> Response {
    let is_stream = wire_request.stream.unwrap_or(false);
    let original_model = wire_request.model.clone();
    let internal_request: CompletionRequest = wire_request.into();

    let (provider_name, model_id, provider) =
        match state.resolve_provider(&internal_request.model, &internal_request).await {
            Ok(r) => r,
            Err(e) => return error_to_anthropic_response(e),
        };

    if is_stream {
        let result = if state.is_cascade_strategy(&original_model) {
            state
                .complete_stream_with_cascade(
                    &internal_request,
                    &context,
                    &provider_name,
                    &model_id,
                    &provider,
                    &state.inner.routing_config.cascade,
                )
                .await
        } else {
            state
                .complete_stream_with_failover(&internal_request, &context, &provider_name, &model_id, &provider)
                .await
        };

        match result {
            Ok((actual_model, stream)) => anthropic_stream_response(stream, actual_model).into_response(),
            Err(e) => error_to_anthropic_response(e),
        }
    } else {
        match state
            .complete_with_failover(&internal_request, &context, &provider_name, &model_id, &provider)
            .await
        {
            Ok(response) => {
                let wire_response: AnthropicResponse = response.into();
                Json(wire_response).into_response()
            }
            Err(e) => error_to_anthropic_response(e),
        }
    }
}

/// Build a streaming SSE response in Anthropic format
fn anthropic_stream_response(
    stream: std::pin::Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send>>,
    model: String,
) -> Sse<impl Stream<Item = Result<Event, axum::Error>>> {
    let response_id = format!("msg_{}", uuid_simple());

    let event_stream = stream.map(move |result| match result {
        Ok(event) => {
            let anthropic_events =
                convert::anthropic::internal_to_anthropic_stream_events(&event, &model, &response_id);

            // Return the first event as SSE
            anthropic_events.into_iter().next().map_or_else(
                || Ok(Event::default().comment("")),
                |anthropic_event| {
                    let event_type = anthropic_event_type(&anthropic_event);
                    let data = serde_json::to_string(&anthropic_event).unwrap_or_default();
                    Ok(Event::default().event(event_type).data(data))
                },
            )
        }
        Err(e) => {
            let error_data = serde_json::json!({
                "type": "error",
                "error": {
                    "type": "streaming_error",
                    "message": e.to_string()
                }
            });
            Ok(Event::default().event("error").data(error_data.to_string()))
        }
    });

    Sse::new(event_stream).keep_alive(KeepAlive::default())
}

/// Convert an LLM error to an Anthropic-style JSON error response
#[allow(clippy::needless_pass_by_value)]
fn error_to_anthropic_response(error: LlmError) -> Response {
    use synapse_core::HttpError;

    let status = error.status_code();
    let body = serde_json::json!({
        "type": "error",
        "error": {
            "type": error.error_type(),
            "message": error.client_message(),
        }
    });

    (status, Json(body)).into_response()
}

/// Get the SSE event type name for an Anthropic stream event
const fn anthropic_event_type(event: &crate::protocol::anthropic::AnthropicStreamEvent) -> &'static str {
    use crate::protocol::anthropic::AnthropicStreamEvent;

    match event {
        AnthropicStreamEvent::MessageStart { .. } => "message_start",
        AnthropicStreamEvent::ContentBlockStart { .. } => "content_block_start",
        AnthropicStreamEvent::ContentBlockDelta { .. } => "content_block_delta",
        AnthropicStreamEvent::ContentBlockStop { .. } => "content_block_stop",
        AnthropicStreamEvent::MessageDelta { .. } => "message_delta",
        AnthropicStreamEvent::MessageStop => "message_stop",
        AnthropicStreamEvent::Ping => "ping",
    }
}

/// Generate a simple unique ID
fn uuid_simple() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);

    format!("{now:x}{count:04x}")
}
