//! Entitlement webhook handler
//!
//! Receives entitlement change notifications from Aether and invalidates
//! the local cache so subsequent requests re-check against the source

use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;
use http::{HeaderMap, StatusCode};
use secrecy::{ExposeSecret, SecretString};
use serde::Deserialize;

use crate::entitlement_cache::EntitlementCache;

/// Shared state for the webhook endpoint
#[derive(Clone)]
pub struct WebhookState {
    pub cache: EntitlementCache,
    pub gateway_secret: SecretString,
}

/// Incoming entitlement change payload from Aether
#[derive(Deserialize)]
pub struct EntitlementChangePayload {
    /// Entity type (e.g. "user")
    pub entity_type: String,
    /// Entity ID
    pub entity_id: String,
    /// Specific feature key that changed, if any
    pub feature_key: Option<String>,
}

/// Handle entitlement change webhooks from Aether
///
/// Validates the gateway secret, then invalidates cached entitlements
/// for the affected entity so the next request fetches fresh data
pub async fn entitlement_webhook_handler(
    State(state): State<WebhookState>,
    headers: HeaderMap,
    Json(body): Json<EntitlementChangePayload>,
) -> impl IntoResponse {
    let secret = headers
        .get("x-gateway-secret")
        .and_then(|v| v.to_str().ok());

    if secret != Some(state.gateway_secret.expose_secret()) {
        return StatusCode::UNAUTHORIZED;
    }

    if let Some(ref feature_key) = body.feature_key {
        // Invalidate specific entitlement
        state
            .cache
            .invalidate_entitlement(&body.entity_type, &body.entity_id, feature_key);
        tracing::debug!(
            entity_type = %body.entity_type,
            entity_id = %body.entity_id,
            feature_key = %feature_key,
            "invalidated cached entitlement"
        );
    } else {
        // No specific feature key — invalidate common entitlements
        // The most critical is api_access; usage meters also get cleared
        state
            .cache
            .invalidate_entitlement(&body.entity_type, &body.entity_id, "api_access");
        state
            .cache
            .invalidate_usage(&body.entity_type, &body.entity_id, "requests");
        tracing::debug!(
            entity_type = %body.entity_type,
            entity_id = %body.entity_id,
            "invalidated all cached entitlements"
        );
    }

    StatusCode::OK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_payload_with_feature() {
        let json = r#"{"entity_type":"user","entity_id":"usr_123","feature_key":"api_access"}"#;
        let payload: EntitlementChangePayload = serde_json::from_str(json).unwrap();
        assert_eq!(payload.entity_type, "user");
        assert_eq!(payload.entity_id, "usr_123");
        assert_eq!(payload.feature_key.as_deref(), Some("api_access"));
    }

    #[test]
    fn deserialize_payload_without_feature() {
        let json = r#"{"entity_type":"user","entity_id":"usr_123"}"#;
        let payload: EntitlementChangePayload = serde_json::from_str(json).unwrap();
        assert!(payload.feature_key.is_none());
    }
}
