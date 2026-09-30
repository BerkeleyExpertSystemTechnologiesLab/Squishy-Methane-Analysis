"""Shared configuration, scan, and report contracts for active-sensing runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from plume_focus.predictor import BoundingBox, bounding_box_from_mask


DEFAULT_VIDEO_OUTPUT = Path("outputs") / "moving-object-proof.mp4"
DEFAULT_REPORT_OUTPUT = Path("outputs") / "moving-object-proof.json"
DEFAULT_PLUME_VIDEO_OUTPUT = Path("outputs") / "plume-like-proof.mp4"
DEFAULT_PLUME_REPORT_OUTPUT = Path("outputs") / "plume-like-proof.json"
DEFAULT_METHANE_VIDEO_OUTPUT = Path("outputs") / "methane-leak-proof.mp4"
DEFAULT_METHANE_REPORT_OUTPUT = Path("outputs") / "methane-leak-proof.json"
DEFAULT_SCAN_RATE_HZ = 60.0
DEFAULT_RENDER_SCALE = 3
VIDEO_HEADER_HEIGHT = 52

# Relative-density cutoffs used only by rendering and offline attention
# diagnostics.  They are not calibrated confidence values and do not select
# the mechanical scan mask.
ATTENTION_INNER_RELATIVE_DENSITY = 0.45
ATTENTION_OUTER_RELATIVE_DENSITY = 0.22
ATTENTION_HEATMAP_DISPLAY_RELATIVE_DENSITY = 0.08

# Compatibility aliases retained for callers of the schema-v2 surface.
ATTENTION_INNER_THRESHOLD = ATTENTION_INNER_RELATIVE_DENSITY
ATTENTION_OUTER_THRESHOLD = ATTENTION_OUTER_RELATIVE_DENSITY
ATTENTION_HEATMAP_DISPLAY_THRESHOLD = (
    ATTENTION_HEATMAP_DISPLAY_RELATIVE_DENSITY
)

Scenario = Literal["object", "plume", "methane"]
ScanMode = Literal["point", "column"]
EvidenceMode = Literal["scan", "full-frame"]
EvaluationStatus = Literal["evaluated", "not_evaluated"]


@dataclass(frozen=True)
class RasterVideoScan:
    """Point or pixel raster scan projected onto an image grid."""

    values: NDArray[np.uint8]
    sampled_at_seconds: NDArray[np.float64]
    scan_rate_hz: float
    started_at_seconds: float
    completed_at_seconds: float
    sampled_mask: NDArray[np.bool_]
    requested_mask: NDArray[np.bool_] | None = None
    mechanical_sweep_mask: NDArray[np.bool_] | None = None
    roi_box: BoundingBox | None = None
    sample_points: int = 0
    sample_stride_px: int = 1
    scan_mode: ScanMode = "point"
    vertical_pass_seconds: float = 0.0
    horizontal_step_seconds: float = 0.0
    vertical_reference_rows: int = 0
    mean_vertical_travel_rows: float = 0.0
    mean_vertical_pass_seconds: float = 0.0

    @property
    def scan_duration_seconds(self) -> float:
        return self.completed_at_seconds - self.started_at_seconds

    @property
    def max_age_seconds(self) -> float:
        sampled = self.sampled_at_seconds[self.sampled_mask]
        if sampled.size == 0:
            return 0.0
        return self.completed_at_seconds - float(np.min(sampled))

    @property
    def sampled_pixels(self) -> int:
        return int(np.count_nonzero(self.sampled_mask))

    @property
    def sampled_area_fraction(self) -> float:
        return self.sampled_pixels / self.sampled_mask.size

    @property
    def requested_area_fraction(self) -> float:
        mask = self.requested_mask if self.requested_mask is not None else self.sampled_mask
        return float(np.count_nonzero(mask) / mask.size)

    @property
    def mechanical_sweep_area_fraction(self) -> float:
        mask = (
            self.mechanical_sweep_mask
            if self.mechanical_sweep_mask is not None
            else self.sampled_mask
        )
        return float(np.count_nonzero(mask) / mask.size)

    @property
    def effective_sample_points(self) -> int:
        return self.sample_points if self.sample_points > 0 else self.sampled_pixels

    @property
    def sampled_box(self) -> BoundingBox | None:
        if not np.any(self.sampled_mask):
            return None
        return bounding_box_from_mask(self.sampled_mask)


@dataclass(frozen=True)
class ScanTraceEntry:
    """One observer-independent scan and next-plan decision record."""

    scan_index: int
    started_at_seconds: float
    completed_at_seconds: float
    decision_at_seconds: float
    target_at_seconds: float
    requested_roi: tuple[int, int, int, int] | None
    mechanical_sweep_roi: tuple[int, int, int, int] | None
    sampled_roi: tuple[int, int, int, int] | None
    planned_next_roi: tuple[int, int, int, int]
    requested_mask_sha256: str
    mechanical_sweep_mask_sha256: str
    sampled_mask_sha256: str
    planned_next_requested_mask_sha256: str
    planned_next_mechanical_sweep_mask_sha256: str
    evidence_schedule_sha256: str
    sampled_value_sha256: str
    sampled_value_count: int
    sampled_value_min: int | None
    sampled_value_max: int | None
    requested_area_fraction: float
    mechanical_sweep_area_fraction: float
    sampled_area_fraction: float
    effective_sample_points: int
    evidence_timestamp_min_seconds: float | None
    evidence_timestamp_max_seconds: float | None
    evidence_timestamp_count: int
    next_plan_reacquisition: bool
    next_plan_converged: bool
    uncertainty_radius_px: float
    posterior_probability_sum: float | None
    predictive_probability_sum: float | None
    beliefs_finite: bool
    beliefs_nonnegative: bool
    beliefs_normalized: bool

    def to_payload(self) -> dict[str, object]:
        """Return a compact, JSON-native trace payload."""

        def roi_payload(
            roi: tuple[int, int, int, int] | None,
        ) -> list[int] | None:
            return list(roi) if roi is not None else None

        return {
            "scan_index": self.scan_index,
            "started_at_seconds": self.started_at_seconds,
            "completed_at_seconds": self.completed_at_seconds,
            "decision_at_seconds": self.decision_at_seconds,
            "target_at_seconds": self.target_at_seconds,
            "requested_roi": roi_payload(self.requested_roi),
            "mechanical_sweep_roi": roi_payload(
                self.mechanical_sweep_roi
            ),
            "sampled_roi": roi_payload(self.sampled_roi),
            "planned_next_roi": roi_payload(self.planned_next_roi),
            "requested_mask_sha256": self.requested_mask_sha256,
            "mechanical_sweep_mask_sha256": (
                self.mechanical_sweep_mask_sha256
            ),
            "sampled_mask_sha256": self.sampled_mask_sha256,
            "planned_next_requested_mask_sha256": (
                self.planned_next_requested_mask_sha256
            ),
            "planned_next_mechanical_sweep_mask_sha256": (
                self.planned_next_mechanical_sweep_mask_sha256
            ),
            "evidence_schedule_sha256": self.evidence_schedule_sha256,
            "sampled_value_sha256": self.sampled_value_sha256,
            "sampled_value_count": self.sampled_value_count,
            "sampled_value_min": self.sampled_value_min,
            "sampled_value_max": self.sampled_value_max,
            "requested_area_fraction": self.requested_area_fraction,
            "mechanical_sweep_area_fraction": (
                self.mechanical_sweep_area_fraction
            ),
            "sampled_area_fraction": self.sampled_area_fraction,
            "effective_sample_points": self.effective_sample_points,
            "evidence_timestamp_min_seconds": (
                self.evidence_timestamp_min_seconds
            ),
            "evidence_timestamp_max_seconds": (
                self.evidence_timestamp_max_seconds
            ),
            "evidence_timestamp_count": self.evidence_timestamp_count,
            "next_plan_reacquisition": self.next_plan_reacquisition,
            "next_plan_converged": self.next_plan_converged,
            "uncertainty_radius_px": self.uncertainty_radius_px,
            "posterior_probability_sum": (
                self.posterior_probability_sum
            ),
            "predictive_probability_sum": (
                self.predictive_probability_sum
            ),
            "beliefs_finite": self.beliefs_finite,
            "beliefs_nonnegative": self.beliefs_nonnegative,
            "beliefs_normalized": self.beliefs_normalized,
        }


@dataclass(frozen=True)
class BeliefIntegritySummary:
    """Aggregate numerical-integrity evidence for reported beliefs."""

    status: EvaluationStatus
    distributions_checked: int
    all_finite: bool
    all_nonnegative: bool
    all_normalized: bool
    normalization_absolute_tolerance: float
    maximum_probability_sum_error: float | None
    minimum_probability: float | None

    @classmethod
    def not_evaluated(cls) -> "BeliefIntegritySummary":
        return cls(
            status="not_evaluated",
            distributions_checked=0,
            all_finite=False,
            all_nonnegative=False,
            all_normalized=False,
            normalization_absolute_tolerance=1e-9,
            maximum_probability_sum_error=None,
            minimum_probability=None,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "status": self.status,
            "distributions_checked": self.distributions_checked,
            "all_finite": self.all_finite,
            "all_nonnegative": self.all_nonnegative,
            "all_normalized": self.all_normalized,
            "normalization_absolute_tolerance": (
                self.normalization_absolute_tolerance
            ),
            "maximum_probability_sum_error": (
                self.maximum_probability_sum_error
            ),
            "minimum_probability": self.minimum_probability,
        }


@dataclass(frozen=True)
class SampleVariationSummary:
    """Repeated-coordinate variation from scan evidence only."""

    status: EvaluationStatus
    comparisons: int
    changed_values: int
    variation_observed: bool
    changed_fraction: float | None
    mean_absolute_change: float | None
    maximum_absolute_change: int | None
    full_frame_evidence_mode: bool = False

    @classmethod
    def not_evaluated(
        cls,
        *,
        full_frame_evidence_mode: bool = False,
    ) -> "SampleVariationSummary":
        return cls(
            status="not_evaluated",
            comparisons=0,
            changed_values=0,
            variation_observed=False,
            changed_fraction=None,
            mean_absolute_change=None,
            maximum_absolute_change=None,
            full_frame_evidence_mode=full_frame_evidence_mode,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "status": self.status,
            "comparison_basis": (
                "same-coordinate values from causal sampled_mask evidence; "
                "strictly later source-clock observations only"
            ),
            "full_frame_evidence_mode": self.full_frame_evidence_mode,
            "comparisons": self.comparisons,
            "changed_values": self.changed_values,
            "variation_observed": self.variation_observed,
            "changed_fraction": self.changed_fraction,
            "mean_absolute_change": self.mean_absolute_change,
            "maximum_absolute_change": self.maximum_absolute_change,
        }


@dataclass(frozen=True)
class VideoProofReport:
    """Summary metrics for a video motion-vector proof run."""

    output_video: str
    scenario: str
    frames_processed: int
    scan_start_mode: str
    plume_mapping_mode: str
    frame_stride: int
    effective_fps: float
    mean_dx: float
    mean_dy: float
    scan_mode: ScanMode
    scan_rate_hz: float
    scan_sample_stride_px: int
    scan_sample_points: int
    scan_vertical_pass_seconds: float
    scan_horizontal_step_seconds: float
    scan_column_mass_fraction: float
    scan_row_mass_fraction: float
    scan_min_sweep_columns: int
    scan_min_sweep_rows: int
    scan_max_sweep_columns: int | None
    scan_sweep_rows: int
    scan_vertical_reference_rows: int
    scan_mean_vertical_travel_rows: float
    scan_mean_vertical_pass_seconds: float
    scan_duration_seconds: float
    scan_duration_frames: float
    scan_pixels: int
    scan_area_fraction: float
    hits: int
    evaluated_predictions: int
    mean_attention_coverage: float
    mean_attention_area_fraction: float
    attention_mask_threshold: float
    mean_uncertainty_radius: float
    mean_kl_true_to_attention: float
    mean_js_divergence: float
    initial_roi_area_fraction: float
    initial_mask_area_fraction: float
    mean_envelope_area_fraction: float
    final_envelope_area_fraction: float
    mean_mask_area_fraction: float
    final_mask_area_fraction: float
    mean_swept_area_fraction: float
    final_swept_area_fraction: float
    report_schema_version: int = 3
    evidence_mode: EvidenceMode = "scan"
    motion_source: str = "unknown"
    scans_completed: int = 0
    termination_reason: str = "max_scans_reached"
    timeline_integrity_verified: bool = False
    success_coverage_threshold: float = 0.9
    actual_sweep_successes: int = 0
    actual_sweep_evaluated_predictions: int = 0
    mean_actual_sweep_support_recall: float = 0.0
    mean_actual_sweep_mass_recall: float = 0.0
    mean_actual_sweep_interval_mass_recall: float = 0.0
    mean_actual_sweep_iou: float = 0.0
    initial_scan_duration_seconds: float = 0.0
    max_scan_duration_seconds: float = 0.0
    final_scan_duration_seconds: float = 0.0
    total_elapsed_seconds: float = 0.0
    mean_dx_px_per_second: float = 0.0
    mean_dy_px_per_second: float = 0.0
    oracle_inputs: tuple[str, ...] = ()
    policy_id: str = "adaptive-predictive"
    model_profile: dict[str, object] = field(default_factory=dict)
    data_provenance: dict[str, object] = field(default_factory=dict)
    evaluation_truth: dict[str, object] = field(default_factory=dict)
    wind_evidence: dict[str, object] = field(default_factory=dict)
    operational_trace: tuple[ScanTraceEntry, ...] = ()
    belief_integrity: BeliefIntegritySummary = field(
        default_factory=BeliefIntegritySummary.not_evaluated
    )
    sample_variation: SampleVariationSummary = field(
        default_factory=SampleVariationSummary.not_evaluated
    )

    @property
    def hit_rate(self) -> float:
        if self.evaluated_predictions == 0:
            return 0.0
        return self.hits / self.evaluated_predictions

    @property
    def actual_sweep_success_rate(self) -> float:
        if self.actual_sweep_evaluated_predictions == 0:
            return 0.0
        return self.actual_sweep_successes / self.actual_sweep_evaluated_predictions

    def to_json(self) -> str:
        """Serialize the report in a human-readable form."""

        scan_summary: dict[str, float | int] = {
            "duration_seconds": round(self.scan_duration_seconds, 4),
            "duration_video_frames": round(self.scan_duration_frames, 4),
            "covered_pixels": self.scan_pixels,
            "covered_area_fraction": round(self.scan_area_fraction, 4),
        }
        if self.scan_mode == "column":
            scan_key = "column_sweep_scan"
            scan_summary = {
                "full_height_vertical_pass_seconds": round(
                    self.scan_vertical_pass_seconds, 4
                ),
                "mean_vertical_pass_seconds": round(
                    self.scan_mean_vertical_pass_seconds, 4
                ),
                "vertical_reference_rows": self.scan_vertical_reference_rows,
                "mean_vertical_travel_rows": round(
                    self.scan_mean_vertical_travel_rows, 1
                ),
                "horizontal_step_seconds": round(
                    self.scan_horizontal_step_seconds, 4
                ),
                "column_stride_px": self.scan_sample_stride_px,
                "column_mass_fraction": round(self.scan_column_mass_fraction, 4),
                "row_mass_fraction": round(self.scan_row_mass_fraction, 4),
                "min_sweep_columns": self.scan_min_sweep_columns,
                "min_sweep_rows": self.scan_min_sweep_rows,
                "max_sweep_columns": self.scan_max_sweep_columns or 0,
                "mean_columns_scanned": self.scan_sample_points,
                "mean_rows_scanned": self.scan_sweep_rows,
                **scan_summary,
            }
        else:
            scan_key = "point_scan"
            scan_summary = {
                "sample_rate_hz": round(self.scan_rate_hz, 4),
                "sample_stride_px": self.scan_sample_stride_px,
                "mean_sample_points": self.scan_sample_points,
                **scan_summary,
            }

        payload = {
            "report_schema_version": self.report_schema_version,
            "output_video": self.output_video,
            "scenario": self.scenario,
            "frames_processed": self.frames_processed,
            "scans_completed": self.scans_completed,
            "evidence_mode": self.evidence_mode,
            "motion_source": self.motion_source,
            "scan_start_mode": self.scan_start_mode,
            "plume_mapping_mode": self.plume_mapping_mode,
            "policy_id": self.policy_id,
            "model_profile": self.model_profile,
            "data_provenance": self.data_provenance,
            "evaluation_truth": self.evaluation_truth,
            "wind_evidence": self.wind_evidence,
            "termination_reason": self.termination_reason,
            "timeline_integrity": {
                "sequential_non_overlapping_scans": self.timeline_integrity_verified,
                "total_elapsed_seconds": round(self.total_elapsed_seconds, 4),
                "prediction_target": "planned next scan completion",
            },
            "operational_trace": {
                "coordinate_convention": (
                    "inclusive [min_row, min_col, max_row, max_col]"
                ),
                "evidence_time_basis": (
                    "per-sample observation times on the source clock"
                ),
                "source_frame_indices": (
                    "not_available_at_causal_runner_boundary"
                ),
                "schedule": [
                    item.to_payload() for item in self.operational_trace
                ],
            },
            "belief_integrity": self.belief_integrity.to_payload(),
            "sample_variation": self.sample_variation.to_payload(),
            "proof_rendering": {
                "video_playback": "time-compressed update summary",
                "middle_panel": (
                    "completed scan projection; color encodes sample age, "
                    "not live head position"
                ),
                "right_panel_horizon": "planned next scan completion",
            },
            "attention_contours": {
                "field": "predictive belief divided by its frame maximum",
                "calibration": "relative density; not calibrated confidence",
                "planning_effect": "none",
                "inner_contour_relative_density": (
                    ATTENTION_INNER_RELATIVE_DENSITY
                ),
                "outer_contour_relative_density": (
                    ATTENTION_OUTER_RELATIVE_DENSITY
                ),
                "heatmap_display_relative_density": (
                    ATTENTION_HEATMAP_DISPLAY_RELATIVE_DENSITY
                ),
                "diagnostic_mask_relative_density": round(
                    self.attention_mask_threshold,
                    4,
                ),
            },
            "refresh_model": {
                "frame_stride": self.frame_stride,
                "effective_fps": round(self.effective_fps, 4),
            },
            "mean_motion_px_per_frame": {
                "dx": round(self.mean_dx, 4),
                "dy": round(self.mean_dy, 4),
            },
            "mean_motion_px_per_second": {
                "dx": round(self.mean_dx_px_per_second, 4),
                "dy": round(self.mean_dy_px_per_second, 4),
            },
            scan_key: scan_summary,
            "scan_duration_summary": {
                "initial_seconds": round(self.initial_scan_duration_seconds, 4),
                "mean_seconds": round(self.scan_duration_seconds, 4),
                "max_seconds": round(self.max_scan_duration_seconds, 4),
                "final_seconds": round(self.final_scan_duration_seconds, 4),
            },
            "actual_sweep_evaluation": {
                "success_mass_recall_threshold": round(
                    self.success_coverage_threshold, 4
                ),
                "success_rate": round(self.actual_sweep_success_rate, 4),
                "mean_support_recall": round(
                    self.mean_actual_sweep_support_recall, 4
                ),
                "mean_probability_mass_recall": round(
                    self.mean_actual_sweep_mass_recall, 4
                ),
                "mean_start_midpoint_completion_mass_recall": round(
                    self.mean_actual_sweep_interval_mass_recall, 4
                ),
                "mean_intersection_over_union": round(
                    self.mean_actual_sweep_iou, 4
                ),
                "successes": self.actual_sweep_successes,
                "evaluated_predictions": self.actual_sweep_evaluated_predictions,
            },
            "attention_diagnostics": {
                "legacy_intersection_hit_rate": round(self.hit_rate, 4),
                "mean_support_recall": round(self.mean_attention_coverage, 4),
                "mean_area_fraction": round(self.mean_attention_area_fraction, 4),
            },
            "mean_uncertainty_radius_px": round(self.mean_uncertainty_radius, 4),
            "distribution_accuracy": {
                "mean_kl_true_to_attention": round(
                    self.mean_kl_true_to_attention, 6
                ),
                "mean_js_divergence": round(self.mean_js_divergence, 6),
            },
            "iterative_roi_policy": {
                "initial_envelope_area_fraction": round(
                    self.initial_roi_area_fraction, 4
                ),
                "mean_envelope_area_fraction": round(
                    self.mean_envelope_area_fraction, 4
                ),
                "final_envelope_area_fraction": round(
                    self.final_envelope_area_fraction, 4
                ),
                "initial_mask_area_fraction": round(
                    self.initial_mask_area_fraction, 4
                ),
                "mean_mask_area_fraction": round(self.mean_mask_area_fraction, 4),
                "final_mask_area_fraction": round(self.final_mask_area_fraction, 4),
                "mean_swept_area_fraction": round(
                    self.mean_swept_area_fraction, 4
                ),
                "final_swept_area_fraction": round(
                    self.final_swept_area_fraction, 4
                ),
            },
            "oracle_inputs": list(self.oracle_inputs),
            # Compatibility keys retained for one report-schema cycle.
            "attention_hit_rate": round(self.hit_rate, 4),
            "mean_attention_coverage": round(self.mean_attention_coverage, 4),
            "mean_attention_area_fraction": round(
                self.mean_attention_area_fraction, 4
            ),
            "hits": self.hits,
            "evaluated_predictions": self.evaluated_predictions,
        }
        return json.dumps(payload, indent=2)
