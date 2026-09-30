"""Boundary-focused plume tracking from reduced 2-D sensor observations."""

from plume_focus.model_profiles import (
    MeasurementModelConfig,
    ModelProfile,
    ScanPolicyConfig,
    StaticCorridorConfig,
    UncertaintyModelConfig,
    synthetic_demo_v1,
    uncalibrated_video_v1,
)
from plume_focus.motion import MotionEstimate, estimate_centroid_motion, estimate_phase_motion
from plume_focus.plume_dynamics import (
    PlumeDynamicsConfig,
    propagate_plume_belief,
)
from plume_focus.predictor import (
    AttentionPrediction,
    BoundConfig,
    BoundingBox,
    SmoothAttentionPrediction,
    predict_attention_area,
    predict_smooth_attention,
)
from plume_focus.recorded_contracts import (
    DatasetQualificationLineageSpec,
    GasIdentitySpec,
    RecordedSequence,
    RecordedSequenceManifest,
    RecordedSuiteManifest,
)
from plume_focus.scanner import ScanResult, SinglePixelScanner

__all__ = [
    "AttentionPrediction",
    "BoundConfig",
    "BoundingBox",
    "DatasetQualificationLineageSpec",
    "GasIdentitySpec",
    "MeasurementModelConfig",
    "ModelProfile",
    "MotionEstimate",
    "PlumeDynamicsConfig",
    "RecordedSequence",
    "RecordedSequenceManifest",
    "RecordedSuiteManifest",
    "ScanResult",
    "ScanPolicyConfig",
    "SinglePixelScanner",
    "SmoothAttentionPrediction",
    "StaticCorridorConfig",
    "UncertaintyModelConfig",
    "estimate_centroid_motion",
    "estimate_phase_motion",
    "predict_attention_area",
    "predict_smooth_attention",
    "propagate_plume_belief",
    "synthetic_demo_v1",
    "uncalibrated_video_v1",
]
