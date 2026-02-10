//! Smart model routing for Synapse
//!
//! Provides heuristic-based routing strategies adapted from LLMRouter:
//! - **Threshold**: route by complexity (HybridLLMRouter pattern)
//! - **Cost**: maximize quality within budget
//! - **Cascade**: try cheap first, escalate on low confidence (AutomixRouter pattern)

#![allow(clippy::must_use_candidate, clippy::missing_errors_doc)]

pub mod analysis;
pub mod error;
pub mod feedback;
pub mod registry;
pub mod strategy;

pub use analysis::{QueryProfile, analyze_query};
pub use error::RoutingError;
pub use feedback::{FeedbackTracker, RequestFeedback};
pub use registry::{ModelProfile, ModelRegistry};

/// The reason a particular model was selected
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoutingReason {
    /// Query classified as low complexity
    LowComplexity,
    /// Query classified as high complexity
    HighComplexity,
    /// Best quality model selected (no budget constraint)
    BestQuality,
    /// Model selected to fit within cost budget
    CostConstrained,
    /// Initial cheap model in cascade flow
    CascadeInitial,
    /// Escalated to stronger model after low-confidence response
    CascadeEscalated,
}

/// Result of a routing decision
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// Selected provider name
    pub provider: String,
    /// Selected model identifier
    pub model: String,
    /// Why this model was selected
    pub reason: RoutingReason,
    /// Alternative (provider, model) pairs for failover
    pub alternatives: Vec<(String, String)>,
}

/// Route a request using the configured strategy
pub fn route_request(
    messages: &[serde_json::Value],
    has_tools: bool,
    registry: &ModelRegistry,
    config: &synapse_config::RoutingConfig,
) -> Result<RoutingDecision, RoutingError> {
    let profile = analyze_query(messages, has_tools);

    tracing::debug!(
        task_type = ?profile.task_type,
        complexity = ?profile.complexity,
        tokens = profile.estimated_input_tokens,
        "query analyzed for routing"
    );

    let decision = match config.strategy {
        synapse_config::RoutingStrategy::Threshold => {
            strategy::threshold::route(&profile, registry, &config.threshold)?
        }
        synapse_config::RoutingStrategy::Cost => strategy::cost::route(&profile, registry, &config.cost)?,
        synapse_config::RoutingStrategy::Cascade => {
            strategy::cascade::route(&profile, registry, &config.cascade)?
        }
    };

    tracing::info!(
        provider = %decision.provider,
        model = %decision.model,
        reason = ?decision.reason,
        alternatives = decision.alternatives.len(),
        "routing decision made"
    );

    Ok(decision)
}
