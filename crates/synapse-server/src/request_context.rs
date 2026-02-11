use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;
use synapse_core::{Authentication, BillingIdentity, RequestContext};

/// Middleware that constructs a `RequestContext` from the incoming request
///
/// Extracts HTTP parts and any pre-populated authentication/identity
/// extensions into a unified context for downstream handlers
pub async fn request_context_middleware(request: Request, next: Next) -> Response {
    let (parts, body) = request.into_parts();

    let api_key = None;
    let client_identity = parts.extensions.get().cloned();
    let authentication = parts.extensions.get::<Authentication>().cloned().unwrap_or_default();
    let billing_identity = parts.extensions.get::<BillingIdentity>().cloned();

    let context = RequestContext {
        parts: parts.clone(),
        api_key,
        client_identity,
        authentication,
        billing_identity,
    };

    let mut request = Request::from_parts(parts, body);
    request.extensions_mut().insert(context);

    next.run(request).await
}
