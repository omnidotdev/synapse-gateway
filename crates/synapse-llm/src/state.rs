//! Core LLM state and provider resolution logic

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use futures_util::{Stream, StreamExt};
use synapse_config::{FailoverConfig, LlmConfig, LlmProviderType, RoutingConfig};
use synapse_core::RequestContext;
use synapse_routing::{FeedbackTracker, ModelRegistry, RequestFeedback};

use crate::discovery;
use crate::error::LlmError;
use crate::health::ProviderHealthTracker;
use crate::provider::Provider;
use crate::routing::ModelRouter;
use crate::types::{CompletionRequest, CompletionResponse, StreamEvent};

/// Virtual model names that trigger smart routing
pub(crate) const ROUTING_CLASSES: &[&str] = &["auto", "fast", "best", "cheap"];

/// Shared state for LLM route handlers
#[derive(Clone)]
pub struct LlmState {
    pub(crate) inner: Arc<LlmStateInner>,
}

pub(crate) struct LlmStateInner {
    pub(crate) router: ModelRouter,
    pub(crate) providers: HashMap<String, Arc<dyn Provider>>,
    pub(crate) health: ProviderHealthTracker,
    pub(crate) failover: FailoverConfig,
    pub(crate) routing_config: RoutingConfig,
    pub(crate) model_registry: ModelRegistry,
    pub(crate) feedback: FeedbackTracker,
}

impl LlmState {
    /// Execute a non-streaming completion with automatic provider
    /// resolution, smart routing, and failover
    ///
    /// # Errors
    ///
    /// Returns an error if model resolution or all provider attempts fail
    pub async fn complete(
        &self,
        request: CompletionRequest,
        context: RequestContext,
    ) -> Result<CompletionResponse, LlmError> {
        let (provider_name, model_id, provider) =
            self.resolve_provider(&request.model, &request).await?;

        self.complete_with_failover(&request, &context, &provider_name, &model_id, &provider)
            .await
    }

    /// Execute a streaming completion with automatic provider
    /// resolution, smart routing, failover, and optional cascade
    ///
    /// # Errors
    ///
    /// Returns an error if model resolution or all provider attempts fail
    pub async fn complete_stream(
        &self,
        request: CompletionRequest,
        context: RequestContext,
    ) -> Result<(String, Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send>>), LlmError>
    {
        let original_model = request.model.clone();
        let (provider_name, model_id, provider) =
            self.resolve_provider(&request.model, &request).await?;

        if self.is_cascade_strategy(&original_model) {
            self.complete_stream_with_cascade(
                &request,
                &context,
                &provider_name,
                &model_id,
                &provider,
                &self.inner.routing_config.cascade,
            )
            .await
        } else {
            self.complete_stream_with_failover(
                &request,
                &context,
                &provider_name,
                &model_id,
                &provider,
            )
            .await
        }
    }

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

    /// List all available models across providers
    pub async fn list_models(&self) -> Vec<(String, String)> {
        self.inner.router.list_models().await
    }

    /// Resolve a model name and get the corresponding provider
    ///
    /// Handles both normal model names and virtual routing classes
    /// ("auto", "fast", "best", "cheap") when smart routing is enabled.
    pub(crate) async fn resolve_provider(
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
    pub(crate) async fn complete_with_failover(
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
    pub(crate) async fn complete_stream_with_failover(
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
    pub(crate) async fn complete_stream_with_cascade(
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
    pub(crate) fn is_cascade_strategy(&self, model: &str) -> bool {
        if !self.inner.routing_config.enabled || !ROUTING_CLASSES.contains(&model) {
            return false;
        }
        let config = self.map_routing_class(model);
        matches!(config.strategy, synapse_config::RoutingStrategy::Cascade)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Verify the public API surface compiles
    #[test]
    fn complete_method_exists() {
        // Type-level check: ensure `complete` method signature is correct
        fn _assert_complete(state: &LlmState, req: CompletionRequest, ctx: RequestContext) {
            let _fut = state.complete(req, ctx);
        }
    }

    #[test]
    fn complete_stream_method_exists() {
        fn _assert_stream(state: &LlmState, req: CompletionRequest, ctx: RequestContext) {
            let _fut = state.complete_stream(req, ctx);
        }
    }
}
