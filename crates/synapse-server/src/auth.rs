use axum::extract::Request;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use http::StatusCode;
use synapse_auth::{ApiKeyResolver, KeyMode, UsageReporter};
use synapse_core::{BillingIdentity, BillingMode};

/// Authenticate requests via API key
///
/// Extracts Bearer token from Authorization header. If it starts with
/// `sk-syn-`, resolves it via synapse-api. Skips auth for public paths
/// and passes through non-synapse tokens for existing auth flows.
pub async fn auth_middleware(
    resolver: ApiKeyResolver,
    public_paths: Vec<String>,
    usage_reporter: Option<UsageReporter>,
    request: Request,
    next: Next,
) -> Response {
    let path = request.uri().path().to_string();

    if public_paths.iter().any(|p| path.starts_with(p)) {
        return next.run(request).await;
    }

    let token = request
        .headers()
        .get(http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));

    let Some(token) = token else {
        return next.run(request).await;
    };

    if !token.starts_with("sk-syn-") {
        return next.run(request).await;
    }

    match resolver.resolve(token).await {
        Ok(resolved) => {
            let billing_identity = BillingIdentity {
                entity_type: "user".to_string(),
                entity_id: resolved.user_id.clone(),
                mode: match resolved.mode {
                    KeyMode::Byok => BillingMode::Byok,
                    KeyMode::Managed => BillingMode::Managed,
                },
            };

            let mut request = request;
            request.extensions_mut().insert(resolved);
            request.extensions_mut().insert(billing_identity);
            if let Some(reporter) = usage_reporter {
                request.extensions_mut().insert(reporter);
            }
            next.run(request).await
        }
        Err(e) => {
            tracing::warn!(error = %e, "API key authentication failed");
            (StatusCode::UNAUTHORIZED, "invalid API key").into_response()
        }
    }
}
