use std::sync::Arc;

use axum::extract::Request;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use http::StatusCode;
use synapse_billing::AetherClient;
use synapse_config::{BillingConfig, FailMode};
use synapse_core::BillingIdentity;

use crate::entitlement_cache::{CachedEntitlement, CachedUsageCheck, EntitlementCache};

/// Shared state for the entitlement middleware
#[derive(Clone)]
pub struct EntitlementState {
    client: Arc<AetherClient>,
    config: BillingConfig,
    cache: EntitlementCache,
}

impl EntitlementState {
    /// Create new entitlement state
    pub fn new(client: AetherClient, config: BillingConfig) -> Self {
        let cache = EntitlementCache::new(config.entitlement_cache_ttl_secs);

        Self {
            client: Arc::new(client),
            config,
            cache,
        }
    }
}

/// Middleware that enforces entitlements and usage limits before processing
///
/// 1. Checks `api_access` entitlement → 403 if denied
/// 2. Checks usage limit for `requests` meter → 429 if exceeded
/// 3. Passes through if all checks pass
pub async fn entitlement_middleware(state: EntitlementState, request: Request, next: Next) -> Response {
    let Some(identity) = request.extensions().get::<BillingIdentity>() else {
        // No billing identity means billing middleware didn't run or
        // no valid auth — this shouldn't happen if middleware stack is
        // correct, but fail closed
        return (StatusCode::INTERNAL_SERVER_ERROR, "missing billing identity").into_response();
    };

    let entity_type = &identity.entity_type;
    let entity_id = &identity.entity_id;

    // 1. Check api_access entitlement
    match check_entitlement(&state, entity_type, entity_id).await {
        Ok(true) => {}
        Ok(false) => {
            return (StatusCode::FORBIDDEN, "API access not granted for this account").into_response();
        }
        Err(e) => {
            return handle_aether_error(&state.config.fail_mode, e, request, next).await;
        }
    }

    // 2. Check usage limit
    match check_usage(&state, entity_type, entity_id).await {
        Ok(true) => {}
        Ok(false) => {
            return (StatusCode::TOO_MANY_REQUESTS, "usage limit exceeded").into_response();
        }
        Err(e) => {
            return handle_aether_error(&state.config.fail_mode, e, request, next).await;
        }
    }

    next.run(request).await
}

/// Check `api_access` entitlement, using cache when available
async fn check_entitlement(
    state: &EntitlementState,
    entity_type: &str,
    entity_id: &str,
) -> Result<bool, synapse_billing::BillingError> {
    let feature_key = &state.config.api_access_feature_key;

    // Check cache first
    if let Some(cached) = state.cache.get_entitlement(entity_type, entity_id, feature_key) {
        return Ok(cached.has_access);
    }

    // Cache miss — call Aether
    let response = state
        .client
        .check_entitlement(entity_type, entity_id, feature_key)
        .await?;

    // Cache the result
    state.cache.put_entitlement(
        entity_type,
        entity_id,
        feature_key,
        CachedEntitlement {
            has_access: response.has_access,
            version: response.entitlement_version,
        },
    );

    Ok(response.has_access)
}

/// Check usage limit for the requests meter, using cache when available
async fn check_usage(
    state: &EntitlementState,
    entity_type: &str,
    entity_id: &str,
) -> Result<bool, synapse_billing::BillingError> {
    let meter_key = &state.config.meters.requests;

    // Check cache first
    if let Some(cached) = state.cache.get_usage(entity_type, entity_id, meter_key) {
        return Ok(cached.allowed);
    }

    // Cache miss — call Aether
    let response = state
        .client
        .check_usage(entity_type, entity_id, meter_key, 1.0)
        .await?;

    // Cache the result
    state.cache.put_usage(
        entity_type,
        entity_id,
        meter_key,
        CachedUsageCheck {
            allowed: response.allowed,
        },
    );

    Ok(response.allowed)
}

/// Handle an Aether communication error based on fail mode
async fn handle_aether_error(
    fail_mode: &FailMode,
    error: synapse_billing::BillingError,
    request: Request,
    next: Next,
) -> Response {
    match fail_mode {
        FailMode::Open => {
            tracing::warn!(
                error = %error,
                "Aether unreachable, allowing request through (fail-open mode)"
            );
            next.run(request).await
        }
        FailMode::Closed => {
            tracing::error!(
                error = %error,
                "Aether unreachable, rejecting request (fail-closed mode)"
            );
            (StatusCode::SERVICE_UNAVAILABLE, "billing service unavailable").into_response()
        }
    }
}
