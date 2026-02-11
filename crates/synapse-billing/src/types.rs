use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Request to record usage against a meter
#[derive(Debug, Clone, Serialize)]
pub struct RecordUsageRequest {
    /// Amount to increment
    pub delta: f64,
    /// Unique key for idempotent recording
    pub idempotency_key: String,
    /// Optional metadata attached to the usage event
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
}

/// Response from recording usage
#[derive(Debug, Clone, Deserialize)]
pub struct RecordUsageResponse {
    /// Whether the recording was accepted
    pub accepted: bool,
}

/// Response from checking usage against limits
#[derive(Debug, Clone, Deserialize)]
pub struct CheckUsageResponse {
    /// Whether the additional usage is within limits
    pub allowed: bool,
    /// Current usage for this meter
    #[serde(default)]
    pub current_usage: f64,
    /// Configured limit for this meter
    #[serde(default)]
    pub limit: Option<f64>,
}

/// Response from checking a feature entitlement
#[derive(Debug, Clone, Deserialize)]
pub struct EntitlementCheckResponse {
    /// Whether access to the feature is granted
    pub has_access: bool,
    /// Entitlement version for cache invalidation
    #[serde(default)]
    pub entitlement_version: Option<u64>,
}

/// Response listing all entitlements for an entity
#[derive(Debug, Clone, Deserialize)]
pub struct EntitlementsResponse {
    /// Map of feature key → access granted
    pub entitlements: HashMap<String, bool>,
    /// Entitlement version for cache invalidation
    #[serde(default)]
    pub entitlement_version: Option<u64>,
}
