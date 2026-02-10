//! Cost-constrained routing strategy
//!
//! Maximizes quality within a budget ceiling. Estimates request cost
//! from token counts and model pricing, then selects the highest-quality
//! model that fits within the per-request budget.

use synapse_config::CostConfig;

use crate::analysis::QueryProfile;
use crate::error::RoutingError;
use crate::registry::ModelRegistry;
use crate::{RoutingDecision, RoutingReason};

/// Default output-to-input token ratio for cost estimation
const DEFAULT_OUTPUT_RATIO: f64 = 0.5;

/// Route a query using cost-constrained strategy
pub fn route(
    profile: &QueryProfile,
    registry: &ModelRegistry,
    config: &CostConfig,
) -> Result<RoutingDecision, RoutingError> {
    let Some(max_cost) = config.max_cost_per_request else {
        // No budget constraint — just pick the best model
        let best = registry.best_quality().ok_or(RoutingError::NoProfiles)?;
        return Ok(RoutingDecision {
            provider: best.provider.clone(),
            model: best.model.clone(),
            reason: RoutingReason::BestQuality,
            alternatives: registry
                .profiles()
                .iter()
                .filter(|p| p.provider != best.provider || p.model != best.model)
                .map(|p| (p.provider.clone(), p.model.clone()))
                .collect(),
        });
    };

    let estimated_output = (profile.estimated_input_tokens as f64 * DEFAULT_OUTPUT_RATIO) as usize;

    // Filter models that fit within budget, then pick the highest quality
    let mut candidates: Vec<_> = registry
        .profiles()
        .iter()
        .filter(|p| p.estimate_cost(profile.estimated_input_tokens, estimated_output) <= max_cost)
        .collect();

    candidates.sort_by(|a, b| b.quality.partial_cmp(&a.quality).unwrap_or(std::cmp::Ordering::Equal));

    let selected = candidates.first().ok_or(RoutingError::NoModelAvailable {
        class: "cost".to_owned(),
    })?;

    let alternatives = candidates
        .iter()
        .skip(1)
        .map(|p| (p.provider.clone(), p.model.clone()))
        .collect();

    Ok(RoutingDecision {
        provider: selected.provider.clone(),
        model: selected.model.clone(),
        reason: RoutingReason::CostConstrained,
        alternatives,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::{Complexity, TaskType};
    use synapse_config::{ModelCapabilities, ModelProfileConfig};

    fn test_registry() -> ModelRegistry {
        ModelRegistry::from_config(&[
            ModelProfileConfig {
                provider: "expensive".to_owned(),
                model: "big-model".to_owned(),
                context_window: 200_000,
                input_per_mtok: 10.0,
                output_per_mtok: 30.0,
                quality: 0.95,
                capabilities: ModelCapabilities::default(),
            },
            ModelProfileConfig {
                provider: "cheap".to_owned(),
                model: "small-model".to_owned(),
                context_window: 32_000,
                input_per_mtok: 0.1,
                output_per_mtok: 0.3,
                quality: 0.70,
                capabilities: ModelCapabilities::default(),
            },
        ])
    }

    #[test]
    fn tight_budget_picks_cheap() {
        let registry = test_registry();
        let profile = QueryProfile {
            estimated_input_tokens: 1000,
            task_type: TaskType::General,
            complexity: Complexity::Low,
            requires_tool_use: false,
        };
        let config = CostConfig {
            max_cost_per_request: Some(0.001),
        };
        let decision = route(&profile, &registry, &config).unwrap();
        assert_eq!(decision.model, "small-model");
    }

    #[test]
    fn no_budget_picks_best() {
        let registry = test_registry();
        let profile = QueryProfile {
            estimated_input_tokens: 1000,
            task_type: TaskType::General,
            complexity: Complexity::Low,
            requires_tool_use: false,
        };
        let config = CostConfig {
            max_cost_per_request: None,
        };
        let decision = route(&profile, &registry, &config).unwrap();
        assert_eq!(decision.model, "big-model");
    }
}
