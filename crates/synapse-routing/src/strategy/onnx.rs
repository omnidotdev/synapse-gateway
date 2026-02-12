//! ONNX-based ML routing strategy
//!
//! Load a pre-trained ONNX model to classify query complexity and select
//! the optimal model. Training happens offline in Python; inference runs
//! in Rust via the `ort` crate
//!
//! Enable with the `onnx` feature flag

use crate::analysis::QueryProfile;
use crate::error::RoutingError;
use crate::feedback::FeedbackTracker;
use crate::registry::ModelRegistry;
use crate::strategy::Strategy;
use crate::RoutingDecision;

/// ONNX ML-based routing strategy
///
/// When the `onnx` feature is enabled, holds a loaded `ort::Session`
/// for inference. Otherwise acts as a placeholder that returns a
/// feature-not-available error
#[derive(Debug)]
pub struct OnnxStrategy {
    #[cfg(feature = "onnx")]
    session: ort::Session,
    #[cfg(not(feature = "onnx"))]
    _phantom: (),
}

impl OnnxStrategy {
    /// Load an ONNX model from disk
    ///
    /// # Errors
    ///
    /// Returns `RoutingError::FeatureNotAvailable` when the `onnx` feature
    /// is not enabled. Returns `RoutingError::AnalysisFailed` if the model
    /// file cannot be loaded
    #[cfg(feature = "onnx")]
    pub fn load(model_path: &str) -> Result<Self, RoutingError> {
        let session = ort::Session::builder()
            .and_then(|builder| builder.commit_from_file(model_path))
            .map_err(|e| RoutingError::AnalysisFailed(format!("failed to load ONNX model: {e}")))?;

        tracing::info!(path = %model_path, "loaded ONNX routing model");

        Ok(Self { session })
    }

    /// Load an ONNX model from disk
    ///
    /// # Errors
    ///
    /// Returns `RoutingError::FeatureNotAvailable` when the `onnx` feature
    /// is not enabled
    #[cfg(not(feature = "onnx"))]
    pub fn load(_model_path: &str) -> Result<Self, RoutingError> {
        Err(RoutingError::FeatureNotAvailable {
            feature: "onnx".to_owned(),
        })
    }
}

impl Strategy for OnnxStrategy {
    fn route(
        &self,
        _profile: &QueryProfile,
        registry: &ModelRegistry,
        _feedback: Option<&FeedbackTracker>,
    ) -> Result<RoutingDecision, RoutingError> {
        #[cfg(feature = "onnx")]
        {
            // TODO: implement ONNX inference to classify query and select model
            // For now, fall back to best quality model
            let best = registry.best_quality().ok_or(RoutingError::NoProfiles)?;

            let alternatives = registry
                .profiles()
                .iter()
                .filter(|p| p.provider != best.provider || p.model != best.model)
                .map(|p| (p.provider.clone(), p.model.clone()))
                .collect();

            return Ok(RoutingDecision {
                provider: best.provider.clone(),
                model: best.model.clone(),
                reason: crate::RoutingReason::BestQuality,
                alternatives,
            });
        }

        #[cfg(not(feature = "onnx"))]
        {
            let _ = registry;
            Err(RoutingError::FeatureNotAvailable {
                feature: "onnx".to_owned(),
            })
        }
    }

    fn name(&self) -> &'static str {
        "onnx"
    }
}
