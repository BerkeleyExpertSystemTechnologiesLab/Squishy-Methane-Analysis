"""Sequential active-sensing orchestration.

The video module owns OpenCV rendering and compatibility helpers.  This module
owns physical time, evidence isolation, scan planning, and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from plume_focus.bayes import (
    credible_region_mask,
    normalize_probability,
    posterior_from_likelihood,
    predict_next_prior,
    scan_likelihood,
    source_downwind_prior,
)
from plume_focus.experiment import (
    ArrayVideoTimeline,
    FunctionalTimeline,
    MaskCoverage,
    ScanEvent,
    TimelineExhausted,
    TimelineSource,
    distribution_centroid_motion,
    evaluate_scan_mask,
)
from plume_focus.experiment_contracts import (
    ATTENTION_OUTER_RELATIVE_DENSITY,
    BeliefIntegritySummary,
    DEFAULT_SCAN_RATE_HZ,
    EvidenceMode,
    RasterVideoScan,
    SampleVariationSummary,
    ScanMode,
    ScanTraceEntry,
    Scenario,
    VideoProofReport,
)
from plume_focus.experiment_metrics import (
    estimate_opencv_motion,
    foreground_mask,
    js_divergence,
    kl_divergence,
    normalize_distribution,
    plume_truth_distribution,
)
from plume_focus.motion import MotionEstimate
from plume_focus.model_profiles import (
    ModelProfile,
    ResolvedUncertaintyConfig,
    synthetic_demo_v1,
    uncalibrated_video_v1,
)
from plume_focus.plume_dynamics import PlumeDynamicsConfig
from plume_focus.predictor import BoundConfig, BoundingBox, bounding_box_from_mask
from plume_focus.recorded_contracts import (
    CausalMotionProvider,
    EvaluationTruthSource,
    RecordedSequence,
)
from plume_focus.scan_execution import (
    budget_column_sweep_mask,
    column_sweep_scan_moving_frames,
    raster_scan_moving_frames,
)
from plume_focus.scan_policies import (
    AdaptivePredictivePolicy,
    PolicyObservation,
    ScanBudget,
    ScanPolicy,
    allocate_exact_column_mask,
)
from plume_focus.scanner import SinglePixelScanner
from plume_focus.video_sources import (
    SyntheticMethaneLeakConfig,
    SyntheticPlumeVideoConfig,
    SyntheticVideoConfig,
    default_methane_roi,
    generate_synthetic_methane_frame,
    generate_synthetic_methane_values,
    generate_synthetic_object_frame,
    generate_synthetic_object_values,
    generate_synthetic_plume_frame,
    generate_synthetic_plume_values,
    load_video_frames,
    methane_scan_roi,
)

MethaneMotionMode = Literal["generator_truth", "provided", "scan_derived"]
_BELIEF_NORMALIZATION_ABSOLUTE_TOLERANCE = 1e-9
_SAMPLE_TIME_ABSOLUTE_TOLERANCE_SECONDS = 1e-12


@dataclass(frozen=True)
class ExperimentRequest:
    """User-supplied configuration for one headless experiment."""

    input_video: Path | None = None
    recorded_sequence: RecordedSequence | None = None
    max_frames: int = 48
    max_scans: int | None = None
    scenario: Scenario = "object"
    evidence_mode: EvidenceMode = "scan"
    scan_mode: ScanMode | None = None
    scan_rate_hz: float | None = None
    sample_stride_px: int | None = None
    vertical_pass_seconds: float | None = None
    horizontal_step_seconds: float | None = None
    column_stride_px: int | None = None
    column_mass_fraction: float | None = None
    row_mass_fraction: float | None = None
    min_sweep_columns: int | None = None
    min_sweep_rows: int | None = None
    max_sweep_columns: int | None = None
    methane_wind_dy: float | None = None
    methane_wind_dx: float | None = None
    methane_config: SyntheticMethaneLeakConfig | None = None
    methane_motion_mode: MethaneMotionMode = "generator_truth"
    planner_motion: MotionEstimate | None = None
    scan_budget: ScanBudget | None = None
    initial_scan_budget: ScanBudget | None = None
    model_profile: ModelProfile | None = None
    plume_dynamics_config: PlumeDynamicsConfig | None = None
    start_full_scene: bool = False
    source_free: bool = False
    frame_stride: int = 1
    success_coverage_threshold: float = 0.9


@dataclass(frozen=True)
class ExperimentContext:
    """Stable dimensions and timing supplied to experiment observers."""

    shape: tuple[int, int]
    render_fps: float
    scenario: Scenario
    evidence_mode: EvidenceMode


@dataclass(frozen=True)
class ExperimentStep:
    """One completed causal decision exposed to optional observers."""

    current_frame: NDArray[np.uint8]
    scan: RasterVideoScan
    target_frame: NDArray[np.uint8]
    target_support: NDArray[np.bool_]
    attention_field: NDArray[np.float64]
    next_scan_mask: NDArray[np.bool_]
    motion_px_per_second: MotionEstimate


@dataclass(frozen=True)
class ExperimentResult:
    """Headless aggregate result."""

    report: VideoProofReport


class ExperimentObserver(Protocol):
    """Optional side-effect adapter for experiment lifecycle events."""

    def start(self, context: ExperimentContext) -> None:
        """Initialize resources after the runtime is prepared."""

    def observe(self, step: ExperimentStep) -> None:
        """Consume one completed, evaluated experiment step."""

    def close(self) -> None:
        """Release resources after success or failure."""


@dataclass(frozen=True)
class _RuntimeSetup:
    timeline: TimelineSource
    render_fps: float
    scan_mode: ScanMode
    scan_rate_hz: float
    sample_stride_px: int
    column_stride_px: int
    vertical_pass_seconds: float
    horizontal_step_seconds: float
    column_mass_fraction: float
    row_mass_fraction: float
    min_sweep_columns: int
    min_sweep_rows: int
    max_sweep_columns: int | None
    foreground_threshold_ratio: float
    attention_diagnostic_relative_density: float
    uncertainty: ResolvedUncertaintyConfig
    initial_roi: BoundingBox
    source_point: tuple[float, float] | None
    known_motion: MotionEstimate | None
    motion_provider: CausalMotionProvider | None
    truth_source: EvaluationTruthSource | None
    sensor_derived_truth_allowed: bool
    scan_start_mode: str
    plume_mapping_mode: str
    oracle_inputs: tuple[str, ...]
    data_provenance: dict[str, object]
    plume_dynamics_config: PlumeDynamicsConfig
    model_profile: ModelProfile


@dataclass(frozen=True)
class _PlannedScan:
    requested_mask: NDArray[np.bool_]
    mechanical_mask: NDArray[np.bool_]
    roi: BoundingBox
    predictive: NDArray[np.float64]
    duration_seconds: float
    uncertainty_radius: float
    reacquisition: bool
    converged: bool


@dataclass(frozen=True)
class _BeliefCheck:
    finite: bool
    nonnegative: bool
    normalized: bool
    probability_sum: float | None
    minimum_probability: float | None
    probability_sum_error: float | None


@dataclass
class _SampleVariationAccumulator:
    last_values: NDArray[np.uint8]
    last_times: NDArray[np.float64]
    comparisons: int = 0
    changed_values: int = 0
    absolute_change_sum: int = 0
    maximum_absolute_change: int = 0

    @classmethod
    def for_shape(
        cls,
        shape: tuple[int, int],
    ) -> "_SampleVariationAccumulator":
        return cls(
            last_values=np.zeros(shape, dtype=np.uint8),
            last_times=np.full(shape, np.nan, dtype=float),
        )

    def observe(self, scan: RasterVideoScan) -> None:
        sampled = np.asarray(scan.sampled_mask, dtype=bool)
        sample_times = np.asarray(scan.sampled_at_seconds, dtype=float)
        strictly_later = (
            sampled
            & np.isfinite(self.last_times)
            & (
                sample_times
                > self.last_times
                + _SAMPLE_TIME_ABSOLUTE_TOLERANCE_SECONDS
            )
        )
        if np.any(strictly_later):
            current = np.asarray(
                scan.values[strictly_later],
                dtype=np.int16,
            )
            previous = np.asarray(
                self.last_values[strictly_later],
                dtype=np.int16,
            )
            differences = np.abs(current - previous)
            self.comparisons += int(differences.size)
            self.changed_values += int(np.count_nonzero(differences))
            self.absolute_change_sum += int(differences.sum())
            self.maximum_absolute_change = max(
                self.maximum_absolute_change,
                int(differences.max()),
            )
        self.last_values[sampled] = scan.values[sampled]
        self.last_times[sampled] = sample_times[sampled]

    def summary(
        self,
        *,
        full_frame_evidence_mode: bool,
    ) -> SampleVariationSummary:
        if self.comparisons == 0:
            return SampleVariationSummary.not_evaluated(
                full_frame_evidence_mode=full_frame_evidence_mode
            )
        return SampleVariationSummary(
            status="evaluated",
            comparisons=self.comparisons,
            changed_values=self.changed_values,
            variation_observed=self.changed_values > 0,
            changed_fraction=(self.changed_values / self.comparisons),
            mean_absolute_change=(
                self.absolute_change_sum / self.comparisons
            ),
            maximum_absolute_change=self.maximum_absolute_change,
            full_frame_evidence_mode=full_frame_evidence_mode,
        )


def _check_belief_integrity(
    distribution: NDArray[np.float64],
) -> _BeliefCheck:
    values = np.asarray(distribution, dtype=float)
    finite = bool(np.all(np.isfinite(values)))
    if not finite:
        return _BeliefCheck(
            finite=False,
            nonnegative=False,
            normalized=False,
            probability_sum=None,
            minimum_probability=None,
            probability_sum_error=None,
        )
    probability_sum = float(values.sum())
    minimum_probability = float(values.min()) if values.size else 0.0
    nonnegative = minimum_probability >= 0.0
    probability_sum_error = abs(probability_sum - 1.0)
    return _BeliefCheck(
        finite=True,
        nonnegative=nonnegative,
        normalized=(
            nonnegative
            and probability_sum_error
            <= _BELIEF_NORMALIZATION_ABSOLUTE_TOLERANCE
        ),
        probability_sum=probability_sum,
        minimum_probability=minimum_probability,
        probability_sum_error=probability_sum_error,
    )


def _box_tuple(
    box: BoundingBox | None,
) -> tuple[int, int, int, int] | None:
    if box is None:
        return None
    return (box.min_row, box.min_col, box.max_row, box.max_col)


def _mask_box_tuple(
    mask: NDArray[np.bool_] | None,
) -> tuple[int, int, int, int] | None:
    if mask is None or not np.any(mask):
        return None
    return _box_tuple(bounding_box_from_mask(mask))


def _mask_sha256(mask: NDArray[np.bool_]) -> str:
    values = np.ascontiguousarray(mask, dtype=np.uint8)
    digest = hashlib.sha256(b"plume-focus-binary-mask-v1\0")
    for dimension in values.shape:
        digest.update(int(dimension).to_bytes(8, byteorder="big"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _evidence_schedule_sha256(scan: RasterVideoScan) -> str:
    rows, columns = np.where(scan.sampled_mask)
    times = np.asarray(
        scan.sampled_at_seconds[scan.sampled_mask],
        dtype="<f8",
    )
    digest = hashlib.sha256(b"plume-focus-sample-schedule-v1\0")
    digest.update(
        np.asarray(rows, dtype="<i8").tobytes(order="C")
    )
    digest.update(
        np.asarray(columns, dtype="<i8").tobytes(order="C")
    )
    digest.update(times.tobytes(order="C"))
    return digest.hexdigest()


def _sampled_value_sha256(scan: RasterVideoScan) -> str:
    values = np.ascontiguousarray(
        scan.values[scan.sampled_mask],
        dtype=np.uint8,
    )
    digest = hashlib.sha256(b"plume-focus-sampled-values-v1\0")
    digest.update(int(values.size).to_bytes(8, byteorder="big"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _scan_trace_entry(
    *,
    event: ScanEvent,
    scan: RasterVideoScan,
    planned: _PlannedScan,
    posterior_check: _BeliefCheck,
    predictive_check: _BeliefCheck,
) -> ScanTraceEntry:
    evidence_times = np.asarray(
        scan.sampled_at_seconds[scan.sampled_mask],
        dtype=float,
    )
    sampled_values = np.asarray(
        scan.values[scan.sampled_mask],
        dtype=np.uint8,
    )
    requested_mask = (
        scan.requested_mask
        if scan.requested_mask is not None
        else scan.sampled_mask
    )
    mechanical_mask = (
        scan.mechanical_sweep_mask
        if scan.mechanical_sweep_mask is not None
        else scan.sampled_mask
    )
    return ScanTraceEntry(
        scan_index=event.scan_index,
        started_at_seconds=event.started_at_seconds,
        completed_at_seconds=event.completed_at_seconds,
        decision_at_seconds=event.decision_at_seconds,
        target_at_seconds=event.target_at_seconds,
        requested_roi=_mask_box_tuple(requested_mask),
        mechanical_sweep_roi=_mask_box_tuple(mechanical_mask),
        sampled_roi=_box_tuple(scan.sampled_box),
        planned_next_roi=(
            planned.roi.min_row,
            planned.roi.min_col,
            planned.roi.max_row,
            planned.roi.max_col,
        ),
        requested_mask_sha256=_mask_sha256(requested_mask),
        mechanical_sweep_mask_sha256=_mask_sha256(mechanical_mask),
        sampled_mask_sha256=_mask_sha256(scan.sampled_mask),
        planned_next_requested_mask_sha256=_mask_sha256(
            planned.requested_mask
        ),
        planned_next_mechanical_sweep_mask_sha256=_mask_sha256(
            planned.mechanical_mask
        ),
        evidence_schedule_sha256=_evidence_schedule_sha256(scan),
        sampled_value_sha256=_sampled_value_sha256(scan),
        sampled_value_count=int(sampled_values.size),
        sampled_value_min=(
            int(sampled_values.min()) if sampled_values.size else None
        ),
        sampled_value_max=(
            int(sampled_values.max()) if sampled_values.size else None
        ),
        requested_area_fraction=scan.requested_area_fraction,
        mechanical_sweep_area_fraction=(
            scan.mechanical_sweep_area_fraction
        ),
        sampled_area_fraction=scan.sampled_area_fraction,
        effective_sample_points=scan.effective_sample_points,
        evidence_timestamp_min_seconds=(
            float(evidence_times.min()) if evidence_times.size else None
        ),
        evidence_timestamp_max_seconds=(
            float(evidence_times.max()) if evidence_times.size else None
        ),
        evidence_timestamp_count=(
            int(np.unique(evidence_times).size)
            if evidence_times.size
            else 0
        ),
        next_plan_reacquisition=planned.reacquisition,
        next_plan_converged=planned.converged,
        uncertainty_radius_px=planned.uncertainty_radius,
        posterior_probability_sum=posterior_check.probability_sum,
        predictive_probability_sum=predictive_check.probability_sum,
        beliefs_finite=(posterior_check.finite and predictive_check.finite),
        beliefs_nonnegative=(
            posterior_check.nonnegative and predictive_check.nonnegative
        ),
        beliefs_normalized=(
            posterior_check.normalized and predictive_check.normalized
        ),
    )


def _belief_integrity_summary(
    checks: list[_BeliefCheck],
) -> BeliefIntegritySummary:
    if not checks:
        return BeliefIntegritySummary.not_evaluated()
    errors = [
        item.probability_sum_error
        for item in checks
        if item.probability_sum_error is not None
    ]
    minima = [
        item.minimum_probability
        for item in checks
        if item.minimum_probability is not None
    ]
    return BeliefIntegritySummary(
        status="evaluated",
        distributions_checked=len(checks),
        all_finite=all(item.finite for item in checks),
        all_nonnegative=all(item.nonnegative for item in checks),
        all_normalized=all(item.normalized for item in checks),
        normalization_absolute_tolerance=(
            _BELIEF_NORMALIZATION_ABSOLUTE_TOLERANCE
        ),
        maximum_probability_sum_error=(max(errors) if errors else None),
        minimum_probability=(min(minima) if minima else None),
    )


def _prepare_runtime(
    *,
    input_video: Path | None,
    recorded_sequence: RecordedSequence | None,
    scenario: Scenario,
    frame_stride: int,
    scan_mode: ScanMode | None,
    scan_rate_hz: float | None,
    sample_stride_px: int | None,
    vertical_pass_seconds: float | None,
    horizontal_step_seconds: float | None,
    column_stride_px: int | None,
    column_mass_fraction: float | None,
    row_mass_fraction: float | None,
    min_sweep_columns: int | None,
    min_sweep_rows: int | None,
    max_sweep_columns: int | None,
    methane_wind_dy: float | None,
    methane_wind_dx: float | None,
    methane_config: SyntheticMethaneLeakConfig | None,
    methane_motion_mode: MethaneMotionMode,
    planner_motion: MotionEstimate | None,
    model_profile: ModelProfile | None,
    plume_dynamics_config: PlumeDynamicsConfig | None,
    start_full_scene: bool,
    source_free: bool,
    max_scans: int,
) -> _RuntimeSetup:
    """Build one normalized runtime configuration without hidden frame minima."""

    if (methane_wind_dy is None) != (methane_wind_dx is None):
        raise ValueError("methane_wind_dy and methane_wind_dx must be provided together.")
    if methane_motion_mode not in {"generator_truth", "provided", "scan_derived"}:
        raise ValueError(
            "methane_motion_mode must be 'generator_truth', 'provided', or "
            "'scan_derived'."
        )
    if methane_motion_mode == "provided" and planner_motion is None:
        raise ValueError("planner_motion is required when methane_motion_mode='provided'.")
    if methane_motion_mode != "provided" and planner_motion is not None:
        raise ValueError(
            "planner_motion may only be supplied when methane_motion_mode='provided'."
        )
    if methane_config is not None and scenario != "methane":
        raise ValueError("methane_config may only be used with the methane scenario.")
    if scenario != "methane" and methane_motion_mode != "generator_truth":
        raise ValueError("methane_motion_mode may only be changed for methane experiments.")
    if methane_config is not None and methane_wind_dy is not None:
        raise ValueError(
            "Configure truth wind on methane_config instead of also supplying "
            "methane_wind_dy/methane_wind_dx."
        )

    sensor_derived_truth_allowed = recorded_sequence is None
    if recorded_sequence is not None:
        manifest = recorded_sequence.manifest
        scanner = manifest.scanner
        timeline = recorded_sequence.timeline
        render_fps = timeline.fps
        effective_scan_mode = scanner.scan_mode
        effective_scan_rate = scanner.scan_rate_hz
        effective_sample_stride = scanner.sample_stride_px
        effective_column_stride = scanner.column_stride_px
        effective_vertical_pass = scanner.vertical_pass_seconds
        effective_horizontal_step = scanner.horizontal_step_seconds
        effective_column_mass = scanner.column_mass_fraction
        effective_row_mass = scanner.row_mass_fraction
        effective_min_columns = scanner.minimum_sweep_columns
        effective_min_rows = scanner.minimum_sweep_rows
        effective_max_columns = scanner.maximum_sweep_columns
        initial_roi = recorded_sequence.initial_roi
        source_point = recorded_sequence.source_point
        known_motion = None
        motion_provider = recorded_sequence.motion_provider
        truth_source = recorded_sequence.truth_source
        source_mode = manifest.source_assumption.mode
        scan_start_mode = {
            "known_source": "recorded_known_source_corridor",
            "known_corridor": "recorded_known_corridor",
            "source_free": "recorded_source_free_full_scene",
        }[source_mode]
        mapping_mode = {
            "known_source": "recorded_source_anchored_scan_evidence",
            "known_corridor": "recorded_corridor_scan_evidence",
            "source_free": "recorded_source_free_scan_evidence",
        }[source_mode]
        oracle_inputs = ()
        data_provenance = recorded_sequence.provenance.to_payload()
        resolved_profile = recorded_sequence.model_profile
    elif input_video is None and scenario == "methane":
        defaults = methane_config or SyntheticMethaneLeakConfig()
        wind = (
            defaults.wind_velocity[0] if methane_wind_dy is None else methane_wind_dy,
            defaults.wind_velocity[1] if methane_wind_dx is None else methane_wind_dx,
        )
        config = (
            SyntheticMethaneLeakConfig(frames=max(1, max_scans), wind_velocity=wind)
            if methane_config is None
            else methane_config
        )
        timeline: TimelineSource = FunctionalTimeline(
            shape=config.shape,
            fps=config.fps,
            frame_at_frame_time=lambda time: generate_synthetic_methane_frame(config, time),
            samples_at_frame_times=lambda times, rows, cols: generate_synthetic_methane_values(
                config, times, rows, cols
            ),
        )
        render_fps = config.fps / frame_stride
        effective_scan_mode: ScanMode = scan_mode or "column"
        effective_scan_rate = config.scan_rate_hz if scan_rate_hz is None else scan_rate_hz
        effective_sample_stride = config.sample_stride_px if sample_stride_px is None else sample_stride_px
        effective_column_stride = config.column_stride_px if column_stride_px is None else column_stride_px
        effective_vertical_pass = (
            config.vertical_pass_seconds if vertical_pass_seconds is None else vertical_pass_seconds
        )
        effective_horizontal_step = (
            config.horizontal_step_seconds if horizontal_step_seconds is None else horizontal_step_seconds
        )
        full_roi = BoundingBox(0, 0, config.shape[0] - 1, config.shape[1] - 1)
        initial_roi = full_roi if start_full_scene or source_free else methane_scan_roi(config)
        source_point = None if source_free else config.source
        if methane_motion_mode == "generator_truth":
            known_motion = MotionEstimate(
                dy_per_t=config.wind_velocity[0] * config.fps,
                dx_per_t=config.wind_velocity[1] * config.fps,
                method="oracle_known_wind_px_per_second",
                confidence=1.0,
            )
        elif methane_motion_mode == "provided":
            known_motion = planner_motion
        else:
            known_motion = None
        scan_start_mode = (
            "source_free_full_scene_discovery"
            if source_free
            else "full_scene_discovery"
            if start_full_scene
            else "source_corridor_prior"
        )
        mapping_mode = (
            "source_free_scan_evidence"
            if source_free
            else "source_anchored_after_discovery"
            if start_full_scene
            else "source_anchored_scan_evidence"
        )
        oracle_inputs = (
            ("exact synthetic wind vector",)
            if methane_motion_mode == "generator_truth"
            else ()
        )
        motion_provider = None
        truth_source = None
        data_provenance = {
            "source_kind": "synthetic_fixture",
            "evidence_grade": "engineering_fixture",
            "claim_eligibility": "not_final_acceptance_evidence",
        }
    elif input_video is None and scenario == "plume":
        config = SyntheticPlumeVideoConfig(frames=max(1, max_scans))
        timeline = FunctionalTimeline(
            shape=config.shape,
            fps=config.fps,
            frame_at_frame_time=lambda time: generate_synthetic_plume_frame(config, time),
            samples_at_frame_times=lambda times, rows, cols: generate_synthetic_plume_values(
                config, times, rows, cols
            ),
        )
        render_fps = config.fps / frame_stride
        effective_scan_mode = scan_mode or "point"
        effective_scan_rate = config.scan_rate_hz if scan_rate_hz is None else scan_rate_hz
        effective_sample_stride = 1 if sample_stride_px is None else sample_stride_px
        effective_column_stride = 1 if column_stride_px is None else column_stride_px
        effective_vertical_pass = 1.0 if vertical_pass_seconds is None else vertical_pass_seconds
        effective_horizontal_step = 0.0 if horizontal_step_seconds is None else horizontal_step_seconds
        initial_roi = BoundingBox(0, 0, config.shape[0] - 1, config.shape[1] - 1)
        source_point = None
        known_motion = None
        scan_start_mode = "full_scene_discovery"
        mapping_mode = "scan_evidence_tracking"
        oracle_inputs = ()
        motion_provider = None
        truth_source = None
        data_provenance = {
            "source_kind": "synthetic_fixture",
            "evidence_grade": "engineering_fixture",
            "claim_eligibility": "not_final_acceptance_evidence",
        }
    elif input_video is None:
        config = SyntheticVideoConfig(frames=max(1, max_scans))
        timeline = FunctionalTimeline(
            shape=config.shape,
            fps=config.fps,
            frame_at_frame_time=lambda time: generate_synthetic_object_frame(config, time),
            samples_at_frame_times=lambda times, rows, cols: generate_synthetic_object_values(
                config, times, rows, cols
            ),
        )
        render_fps = config.fps / frame_stride
        effective_scan_mode = scan_mode or "point"
        effective_scan_rate = config.scan_rate_hz if scan_rate_hz is None else scan_rate_hz
        effective_sample_stride = 1 if sample_stride_px is None else sample_stride_px
        effective_column_stride = 1 if column_stride_px is None else column_stride_px
        effective_vertical_pass = 1.0 if vertical_pass_seconds is None else vertical_pass_seconds
        effective_horizontal_step = 0.0 if horizontal_step_seconds is None else horizontal_step_seconds
        initial_roi = BoundingBox(0, 0, config.shape[0] - 1, config.shape[1] - 1)
        source_point = None
        known_motion = None
        scan_start_mode = "full_scene_discovery"
        mapping_mode = "scan_evidence_tracking"
        oracle_inputs = ()
        motion_provider = None
        truth_source = None
        data_provenance = {
            "source_kind": "synthetic_fixture",
            "evidence_grade": "engineering_fixture",
            "claim_eligibility": "not_final_acceptance_evidence",
        }
    else:
        loaded_frames, loaded_fps = load_video_frames(input_video, max_frames=None)
        frames = tuple(loaded_frames[::frame_stride])
        effective_fps = loaded_fps / frame_stride
        timeline = ArrayVideoTimeline(frames, fps=effective_fps)
        render_fps = effective_fps
        effective_scan_mode = scan_mode or ("column" if scenario == "methane" else "point")
        effective_scan_rate = (
            (20.0 if scenario == "methane" else DEFAULT_SCAN_RATE_HZ)
            if scan_rate_hz is None
            else scan_rate_hz
        )
        effective_sample_stride = (
            (8 if scenario == "methane" else 1)
            if sample_stride_px is None
            else sample_stride_px
        )
        effective_column_stride = 1 if column_stride_px is None else column_stride_px
        effective_vertical_pass = 1.0 if vertical_pass_seconds is None else vertical_pass_seconds
        effective_horizontal_step = (
            (0.05 if scenario == "methane" else 0.0)
            if horizontal_step_seconds is None
            else horizontal_step_seconds
        )
        full_roi = BoundingBox(0, 0, timeline.shape[0] - 1, timeline.shape[1] - 1)
        default_roi = default_methane_roi(timeline.shape) if scenario == "methane" else full_roi
        initial_roi = full_roi if start_full_scene or source_free else default_roi
        source_point = (
            None
            if source_free or scenario != "methane"
            else ((default_roi.min_row + default_roi.max_row) / 2.0, float(default_roi.min_col))
        )
        known_motion = (
            MotionEstimate(
                dy_per_t=float(methane_wind_dy) * effective_fps,
                dx_per_t=float(methane_wind_dx) * effective_fps,
                method="external_known_wind_px_per_second",
                confidence=0.85,
            )
            if scenario == "methane" and methane_wind_dy is not None
            else None
        )
        scan_start_mode = (
            "source_free_full_scene_discovery"
            if source_free
            else "full_scene_discovery"
            if start_full_scene
            else "source_corridor_prior"
            if scenario == "methane"
            else "full_scene_discovery"
        )
        mapping_mode = (
            "source_free_scan_evidence"
            if source_free
            else "source_anchored_scan_evidence"
            if scenario == "methane"
            else "scan_evidence_tracking"
        )
        oracle_inputs = ()
        motion_provider = None
        truth_source = None
        data_provenance = {
            "source_kind": "unmanifested_video",
            "evidence_grade": "legacy_unmanifested_input",
            "claim_eligibility": "not_final_acceptance_evidence",
        }

    if model_profile is not None and model_profile.scenario != scenario:
        raise ValueError(
            "model_profile scenario must match the experiment scenario."
        )
    if recorded_sequence is None:
        resolved_profile = model_profile or (
            synthetic_demo_v1(scenario)
            if input_video is None
            else uncalibrated_video_v1(scenario)
        )
    if (
        model_profile is None
        and recorded_sequence is None
        and input_video is None
        and scenario == "methane"
    ):
        resolved_profile = replace(
            resolved_profile,
            scan_policy=replace(
                resolved_profile.scan_policy,
                column_mass_fraction=config.column_mass_fraction,
                row_mass_fraction=config.row_mass_fraction,
                minimum_sweep_columns=config.min_sweep_columns,
                minimum_sweep_rows=config.min_sweep_rows,
                maximum_sweep_columns=config.max_sweep_columns,
            ),
        )
    if plume_dynamics_config is not None:
        resolved_profile = replace(
            resolved_profile,
            profile_id=f"{resolved_profile.profile_id}+dynamics_override",
            provenance=(
                f"{resolved_profile.provenance} Dynamics supplied through "
                "the legacy plume_dynamics_config compatibility field."
            ),
            dynamics=plume_dynamics_config,
        )

    policy_settings = resolved_profile.scan_policy
    if recorded_sequence is None:
        effective_column_mass = (
            policy_settings.column_mass_fraction
            if column_mass_fraction is None
            else column_mass_fraction
        )
        effective_row_mass = (
            policy_settings.row_mass_fraction
            if row_mass_fraction is None
            else row_mass_fraction
        )
        effective_min_columns = (
            policy_settings.minimum_sweep_columns
            if min_sweep_columns is None
            else min_sweep_columns
        )
        effective_min_rows = (
            policy_settings.minimum_sweep_rows
            if min_sweep_rows is None
            else min_sweep_rows
        )
        effective_max_columns = (
            policy_settings.maximum_sweep_columns
            if max_sweep_columns is None
            else max_sweep_columns
        )
    resolved_profile = replace(
        resolved_profile,
        scan_policy=replace(
            policy_settings,
            column_mass_fraction=float(effective_column_mass),
            row_mass_fraction=float(effective_row_mass),
            minimum_sweep_columns=int(effective_min_columns),
            minimum_sweep_rows=int(effective_min_rows),
            maximum_sweep_columns=effective_max_columns,
        ),
    )

    for name, value in {
        "scan_rate_hz": effective_scan_rate,
        "sample_stride_px": effective_sample_stride,
        "column_stride_px": effective_column_stride,
        "vertical_pass_seconds": effective_vertical_pass,
        "min_sweep_columns": effective_min_columns,
        "min_sweep_rows": effective_min_rows,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if effective_horizontal_step < 0:
        raise ValueError("horizontal_step_seconds must be non-negative.")
    if not 0.0 < effective_column_mass <= 1.0:
        raise ValueError("column_mass_fraction must be in (0, 1].")
    if not 0.0 < effective_row_mass <= 1.0:
        raise ValueError("row_mass_fraction must be in (0, 1].")

    resolved_uncertainty = resolved_profile.uncertainty.resolve(
        timeline.fps
    )
    return _RuntimeSetup(
        timeline=timeline,
        render_fps=render_fps,
        scan_mode=effective_scan_mode,
        scan_rate_hz=float(effective_scan_rate),
        sample_stride_px=int(effective_sample_stride),
        column_stride_px=int(effective_column_stride),
        vertical_pass_seconds=float(effective_vertical_pass),
        horizontal_step_seconds=float(effective_horizontal_step),
        column_mass_fraction=float(effective_column_mass),
        row_mass_fraction=float(effective_row_mass),
        min_sweep_columns=int(effective_min_columns),
        min_sweep_rows=int(effective_min_rows),
        max_sweep_columns=effective_max_columns,
        foreground_threshold_ratio=(
            resolved_profile.measurement.foreground_threshold_ratio
        ),
        attention_diagnostic_relative_density=(
            ATTENTION_OUTER_RELATIVE_DENSITY
        ),
        uncertainty=resolved_uncertainty,
        initial_roi=initial_roi,
        source_point=source_point,
        known_motion=known_motion,
        motion_provider=motion_provider,
        truth_source=truth_source,
        sensor_derived_truth_allowed=sensor_derived_truth_allowed,
        scan_start_mode=scan_start_mode,
        plume_mapping_mode=mapping_mode,
        oracle_inputs=oracle_inputs,
        data_provenance=data_provenance,
        plume_dynamics_config=resolved_profile.dynamics,
        model_profile=resolved_profile,
    )


def _full_mask(shape: tuple[int, int]) -> NDArray[np.bool_]:
    return np.ones(shape, dtype=bool)


def _estimate_scan_duration(
    mask: NDArray[np.bool_],
    roi: BoundingBox,
    setup: _RuntimeSetup,
) -> float:
    """Estimate the exact physical duration of a requested scan mask."""

    if setup.scan_mode == "column":
        columns = np.arange(0, mask.shape[1], setup.column_stride_px, dtype=int)
        columns = columns[(columns >= roi.min_col) & (columns <= roi.max_col)]
        columns = np.asarray(
            [col for col in columns if np.any(mask[roi.min_row : roi.max_row + 1, col])],
            dtype=int,
        )
        durations: list[float] = []
        for col in columns:
            active_rows = np.where(mask[roi.min_row : roi.max_row + 1, col])[0]
            if active_rows.size:
                travel_rows = int(active_rows.max() - active_rows.min() + 1)
                durations.append(setup.vertical_pass_seconds * travel_rows / mask.shape[0])
        if not durations:
            return 0.0
        return float(sum(durations) + setup.horizontal_step_seconds * max(0, len(durations) - 1))

    scan_shape = (roi.height, roi.width)
    coarse_rows = int(np.ceil(scan_shape[0] / setup.sample_stride_px))
    coarse_cols = int(np.ceil(scan_shape[1] / setup.sample_stride_px))
    positions = np.asarray(
        list(SinglePixelScanner((coarse_rows, coarse_cols), dwell_time=1.0, order="snake")._positions()),
        dtype=int,
    )
    positions = positions * setup.sample_stride_px + setup.sample_stride_px // 2
    positions[:, 0] = np.clip(positions[:, 0], 0, scan_shape[0] - 1) + roi.min_row
    positions[:, 1] = np.clip(positions[:, 1], 0, scan_shape[1] - 1) + roi.min_col
    selected = mask[positions[:, 0], positions[:, 1]]
    return float(np.count_nonzero(selected) / setup.scan_rate_hz)


def _execute_scan(
    setup: _RuntimeSetup,
    *,
    started_at_seconds: float,
    roi: BoundingBox,
    mask: NDArray[np.bool_],
) -> RasterVideoScan:
    expected_duration = _estimate_scan_duration(mask, roi, setup)
    expected_completion = started_at_seconds + expected_duration
    end_time = setup.timeline.end_time_seconds
    if end_time is not None and expected_completion > end_time + 1e-12:
        raise TimelineExhausted(
            f"Scan would complete at {expected_completion:.6f} seconds, "
            f"after source end {end_time:.6f} seconds."
        )

    # Scan execution needs only the frame shape when a timestamped sample
    # callback is supplied. Do not materialize an unscanned full frame.
    reference = [np.zeros(setup.timeline.shape, dtype=np.uint8)]
    if setup.scan_mode == "column":
        return column_sweep_scan_moving_frames(
            reference,
            0,
            fps=setup.timeline.fps,
            vertical_pass_seconds=setup.vertical_pass_seconds,
            horizontal_step_seconds=setup.horizontal_step_seconds,
            column_stride_px=setup.column_stride_px,
            sample_values_at_seconds=setup.timeline.samples_at_seconds,
            started_at_seconds=started_at_seconds,
            roi_box=roi,
            roi_mask=mask,
        )
    return raster_scan_moving_frames(
        reference,
        0,
        fps=setup.timeline.fps,
        scan_rate_hz=setup.scan_rate_hz,
        sample_stride_px=setup.sample_stride_px,
        sample_values_at_seconds=setup.timeline.samples_at_seconds,
        started_at_seconds=started_at_seconds,
        roi_box=roi,
        roi_mask=mask,
    )


def _instantaneous_full_frame_scan(
    timeline: TimelineSource,
    time_seconds: float,
) -> RasterVideoScan:
    frame = timeline.frame_at_seconds(time_seconds)
    mask = _full_mask(frame.shape)
    sampled_at = np.full(frame.shape, time_seconds, dtype=float)
    return RasterVideoScan(
        values=frame.copy(),
        sampled_at_seconds=sampled_at,
        scan_rate_hz=0.0,
        started_at_seconds=time_seconds,
        completed_at_seconds=time_seconds,
        sampled_mask=mask,
        requested_mask=mask,
        mechanical_sweep_mask=mask,
        roi_box=BoundingBox(0, 0, frame.shape[0] - 1, frame.shape[1] - 1),
        sample_points=frame.size,
        scan_mode="point",
    )


def _propagate_with_elapsed_uncertainty(
    distribution: NDArray[np.float64],
    motion: MotionEstimate,
    *,
    elapsed_seconds: float,
    setup: _RuntimeSetup,
) -> tuple[NDArray[np.float64], float, bool]:
    bounds = BoundConfig(
        scan_age=elapsed_seconds,
        **setup.uncertainty.bound_kwargs,
    )
    radius = bounds.uncertainty_radius(motion, delta_t=elapsed_seconds)
    reacquisition = radius >= (
        float(np.hypot(*distribution.shape))
        * setup.uncertainty.reacquisition_radius_fraction_of_frame_diagonal
    )
    if reacquisition:
        return normalize_probability(np.ones(distribution.shape, dtype=float)), radius, True
    predictive = predict_next_prior(
        distribution,
        motion,
        diffusion_sigma=max(
            setup.uncertainty.diffusion_sigma_floor_px,
            radius / setup.uncertainty.diffusion_radius_divisor,
        ),
        delta_t=elapsed_seconds,
        source=setup.source_point,
        dynamics_config=setup.plume_dynamics_config,
    )
    return predictive, radius, False


def _budget_scan_mask(
    base_mask: NDArray[np.bool_],
    belief: NDArray[np.float64],
    setup: _RuntimeSetup,
) -> NDArray[np.bool_]:
    if setup.scan_mode != "column":
        return base_mask
    return budget_column_sweep_mask(
        base_mask,
        belief,
        mass_fraction=setup.column_mass_fraction,
        min_columns=setup.min_sweep_columns,
        max_columns=setup.max_sweep_columns,
        include_col=(
            int(round(setup.source_point[1]))
            if (
                setup.source_point is not None
                and setup.model_profile.scan_policy.force_known_source_in_scan
            )
            else None
        ),
        row_mass_fraction=setup.row_mass_fraction,
        min_rows=setup.min_sweep_rows,
        include_row=(
            int(round(setup.source_point[0]))
            if (
                setup.source_point is not None
                and setup.model_profile.scan_policy.force_known_source_in_scan
            )
            else None
        ),
    )


def _plan_next_scan(
    latest_likelihood: NDArray[np.float64],
    posterior: NDArray[np.float64],
    motion: MotionEstimate,
    setup: _RuntimeSetup,
    *,
    scenario: Scenario,
    decision_time_seconds: float,
    policy: ScanPolicy | None,
    budget: ScanBudget | None,
) -> _PlannedScan:
    """Solve the scan-mask/duration dependency with a bounded fixed-point loop."""

    policy_config = setup.model_profile.scan_policy
    duration = (
        policy_config.planning_initial_horizon_frames
        / setup.render_fps
    )
    masks: list[NDArray[np.bool_]] = []
    predictive = posterior
    radius = 0.0
    reacquisition = False
    converged = False
    effective_policy = policy or AdaptivePredictivePolicy(
        dynamics_config=setup.plume_dynamics_config,
        uncertainty_config=setup.uncertainty,
    )
    likelihood_for_policy = np.array(latest_likelihood, dtype=float, copy=True)
    posterior_for_policy = np.array(posterior, dtype=float, copy=True)
    likelihood_for_policy.flags.writeable = False
    posterior_for_policy.flags.writeable = False
    observation = PolicyObservation(
        latest_likelihood=likelihood_for_policy,
        posterior=posterior_for_policy,
        motion=motion,
        decision_time_seconds=decision_time_seconds,
        source_point=setup.source_point,
        scenario=scenario,
    )

    for _attempt in range(policy_config.planning_max_iterations):
        forecast = effective_policy.forecast(
            observation,
            horizon_seconds=duration,
        )
        forecast.validate(posterior.shape)
        predictive = normalize_probability(forecast.predictive)
        radius = forecast.uncertainty_radius
        reacquisition = forecast.reacquisition
        if budget is not None:
            if setup.scan_mode != "column":
                raise ValueError("exact ScanBudget is only supported for column scans.")
            requested = allocate_exact_column_mask(
                forecast.score,
                budget,
                include_point=(
                    setup.source_point
                    if policy_config.force_known_source_in_scan
                    else None
                ),
            )
        elif reacquisition:
            requested = _full_mask(posterior.shape)
        else:
            requested = credible_region_mask(
                predictive,
                mass=policy_config.credible_mass,
                blur_sigma=policy_config.credible_blur_sigma_px,
                min_area_fraction=policy_config.minimum_area_fraction,
                include_point=(
                    setup.source_point
                    if policy_config.force_known_source_in_scan
                    else None
                ),
                include_radius=max(
                    policy_config.source_include_radius_min_px,
                    int(
                        round(
                            radius
                            / policy_config.source_include_radius_uncertainty_divisor
                        )
                    ),
                ),
            )
            requested = _budget_scan_mask(requested, predictive, setup)
        roi = bounding_box_from_mask(requested)
        next_duration = _estimate_scan_duration(requested, roi, setup)
        masks.append(requested)
        if abs(next_duration - duration) <= max(
            policy_config.convergence_absolute_seconds,
            policy_config.convergence_relative_fraction
            * max(duration, next_duration),
        ):
            duration = next_duration
            converged = True
            break
        duration = next_duration

    if not converged and masks:
        requested = (
            allocate_exact_column_mask(
                forecast.score,
                budget,
                include_point=(
                    setup.source_point
                    if policy_config.force_known_source_in_scan
                    else None
                ),
            )
            if budget is not None
            else _budget_scan_mask(
                np.logical_or.reduce(masks),
                predictive,
                setup,
            )
        )
        roi = bounding_box_from_mask(requested)
        duration = _estimate_scan_duration(requested, roi, setup)

    # Column masks are contiguous by construction; point scans only sample
    # requested footprints, so this is the actual planned mechanical surface.
    mechanical = requested.copy()
    return _PlannedScan(
        requested_mask=requested,
        mechanical_mask=mechanical,
        roi=roi,
        predictive=predictive,
        duration_seconds=duration,
        uncertainty_radius=radius,
        reacquisition=reacquisition,
        converged=converged,
    )


def _validate_recorded_request(request: ExperimentRequest) -> None:
    """Keep manifest-owned recorded settings from being silently overridden."""

    sequence = request.recorded_sequence
    assert sequence is not None
    if request.input_video is not None:
        raise ValueError(
            "recorded_sequence and input_video are mutually exclusive."
        )
    if request.scenario != sequence.manifest.scenario:
        raise ValueError(
            "recorded experiment scenario must match the manifest."
        )
    if request.max_scans is None:
        raise ValueError(
            "recorded experiments require an explicit max_scans value."
        )

    forbidden = {
        "scan_mode": request.scan_mode,
        "scan_rate_hz": request.scan_rate_hz,
        "sample_stride_px": request.sample_stride_px,
        "vertical_pass_seconds": request.vertical_pass_seconds,
        "horizontal_step_seconds": request.horizontal_step_seconds,
        "column_stride_px": request.column_stride_px,
        "column_mass_fraction": request.column_mass_fraction,
        "row_mass_fraction": request.row_mass_fraction,
        "min_sweep_columns": request.min_sweep_columns,
        "min_sweep_rows": request.min_sweep_rows,
        "max_sweep_columns": request.max_sweep_columns,
        "methane_wind_dy": request.methane_wind_dy,
        "methane_wind_dx": request.methane_wind_dx,
        "methane_config": request.methane_config,
        "planner_motion": request.planner_motion,
        "scan_budget": request.scan_budget,
        "initial_scan_budget": request.initial_scan_budget,
        "model_profile": request.model_profile,
        "plume_dynamics_config": request.plume_dynamics_config,
    }
    supplied = sorted(
        name for name, value in forbidden.items() if value is not None
    )
    if supplied:
        raise ValueError(
            "recorded manifest owns model, wind, scanner, and source "
            f"settings; remove overrides: {', '.join(supplied)}."
        )
    if request.evidence_mode != "scan":
        raise ValueError(
            "recorded evidence-contract runs require scan evidence."
        )
    if request.frame_stride != 1:
        raise ValueError(
            "recorded frame clocks cannot be changed with frame_stride."
        )
    if request.start_full_scene or request.source_free:
        raise ValueError(
            "recorded source assumptions come only from the manifest."
        )
    if request.methane_motion_mode != "generator_truth":
        raise ValueError(
            "recorded wind behavior comes only from the manifest."
        )
    if request.success_coverage_threshold != 0.9:
        raise ValueError(
            "recorded evidence runs use the preregistered 0.90 success "
            "threshold."
        )


def run_experiment(
    request: ExperimentRequest,
    *,
    observer: ExperimentObserver | None = None,
    policy: ScanPolicy | None = None,
) -> ExperimentResult:
    """Run one causal experiment without requiring output side effects."""

    input_video = request.input_video
    recorded_sequence = request.recorded_sequence
    max_frames = request.max_frames
    max_scans = request.max_scans
    scenario = request.scenario
    evidence_mode = request.evidence_mode
    scan_mode = request.scan_mode
    scan_rate_hz = request.scan_rate_hz
    sample_stride_px = request.sample_stride_px
    vertical_pass_seconds = request.vertical_pass_seconds
    horizontal_step_seconds = request.horizontal_step_seconds
    column_stride_px = request.column_stride_px
    column_mass_fraction = request.column_mass_fraction
    row_mass_fraction = request.row_mass_fraction
    min_sweep_columns = request.min_sweep_columns
    min_sweep_rows = request.min_sweep_rows
    max_sweep_columns = request.max_sweep_columns
    methane_wind_dy = request.methane_wind_dy
    methane_wind_dx = request.methane_wind_dx
    methane_config = request.methane_config
    methane_motion_mode = request.methane_motion_mode
    planner_motion = request.planner_motion
    scan_budget = request.scan_budget
    initial_scan_budget = request.initial_scan_budget
    model_profile = request.model_profile
    plume_dynamics_config = request.plume_dynamics_config
    start_full_scene = request.start_full_scene
    source_free = request.source_free
    frame_stride = request.frame_stride
    success_coverage_threshold = request.success_coverage_threshold

    if recorded_sequence is not None:
        _validate_recorded_request(request)
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive.")
    scan_limit = max_frames if max_scans is None else max_scans
    if scan_limit <= 0:
        raise ValueError("max_scans must be positive.")
    if evidence_mode not in {"scan", "full-frame"}:
        raise ValueError("evidence_mode must be 'scan' or 'full-frame'.")
    if not 0.0 < success_coverage_threshold <= 1.0:
        raise ValueError("success_coverage_threshold must be in (0, 1].")
    if scan_budget is not None and evidence_mode != "scan":
        raise ValueError("scan_budget is only valid for scan evidence.")
    if initial_scan_budget is not None and evidence_mode != "scan":
        raise ValueError("initial_scan_budget is only valid for scan evidence.")

    setup = _prepare_runtime(
        input_video=input_video,
        recorded_sequence=recorded_sequence,
        scenario=scenario,
        frame_stride=frame_stride,
        scan_mode=scan_mode,
        scan_rate_hz=scan_rate_hz,
        sample_stride_px=sample_stride_px,
        vertical_pass_seconds=vertical_pass_seconds,
        horizontal_step_seconds=horizontal_step_seconds,
        column_stride_px=column_stride_px,
        column_mass_fraction=column_mass_fraction,
        row_mass_fraction=row_mass_fraction,
        min_sweep_columns=min_sweep_columns,
        min_sweep_rows=min_sweep_rows,
        max_sweep_columns=max_sweep_columns,
        methane_wind_dy=methane_wind_dy,
        methane_wind_dx=methane_wind_dx,
        methane_config=methane_config,
        methane_motion_mode=methane_motion_mode,
        planner_motion=planner_motion,
        model_profile=model_profile,
        plume_dynamics_config=plume_dynamics_config,
        start_full_scene=start_full_scene,
        source_free=source_free,
        max_scans=scan_limit,
    )

    height, width = setup.timeline.shape
    if scan_budget is not None:
        scan_budget.validate(setup.timeline.shape)
        if setup.scan_mode != "column":
            raise ValueError("scan_budget requires column scan mode.")
    if initial_scan_budget is not None:
        initial_scan_budget.validate(setup.timeline.shape)
        if setup.scan_mode != "column":
            raise ValueError("initial_scan_budget requires column scan mode.")
    context = ExperimentContext(
        shape=setup.timeline.shape,
        render_fps=setup.render_fps,
        scenario=scenario,
        evidence_mode=evidence_mode,
    )

    full_roi = BoundingBox(0, 0, height - 1, width - 1)
    current_roi = setup.initial_roi
    current_mask = np.zeros(setup.timeline.shape, dtype=bool)
    current_mask[current_roi.to_slices()] = True
    if evidence_mode == "full-frame":
        current_roi = full_roi
        current_mask = _full_mask(setup.timeline.shape)

    zero_motion = MotionEstimate(0.0, 0.0, "insufficient_scan_history_px_per_second", confidence=0.0)
    wind_lookup_counts = {
        "available": 0,
        "missing": 0,
        "stale": 0,
    }
    wind_measurement_ages: list[float] = []
    wind_scan_fallbacks = 0
    wind_zero_fallbacks = 0
    initial_provider_motion: MotionEstimate | None = None
    if setup.motion_provider is not None:
        initial_lookup = setup.motion_provider.lookup(0.0)
        wind_lookup_counts[initial_lookup.status] += 1
        if initial_lookup.age_seconds is not None:
            wind_measurement_ages.append(initial_lookup.age_seconds)
        initial_provider_motion = initial_lookup.motion
    motion = setup.known_motion or initial_provider_motion or zero_motion
    if (
        setup.source_point is not None
        and (setup.known_motion is not None or initial_provider_motion is not None)
    ):
        prior = source_downwind_prior(
            setup.timeline.shape,
            source=setup.source_point,
            wind_motion=motion,
            roi=current_roi,
            dynamics_config=setup.plume_dynamics_config,
        )
    elif setup.source_point is not None:
        prior = source_downwind_prior(
            setup.timeline.shape,
            source=setup.source_point,
            wind_motion=MotionEstimate(
                (
                    setup.model_profile.scan_policy
                    .initial_source_corridor_dy_px_per_frame
                    * setup.render_fps
                ),
                (
                    setup.model_profile.scan_policy
                    .initial_source_corridor_dx_px_per_frame
                    * setup.render_fps
                ),
                "source_corridor_initialization_px_per_second",
                confidence=0.0,
            ),
            roi=current_roi,
            dynamics_config=setup.plume_dynamics_config,
        )
    else:
        prior = normalize_probability(np.ones(setup.timeline.shape, dtype=float))
    if initial_scan_budget is not None:
        current_mask = allocate_exact_column_mask(
            prior,
            initial_scan_budget,
            include_point=(
                setup.source_point
                if (
                    setup.model_profile.scan_policy
                    .force_known_source_in_scan
                )
                else None
            ),
        )
        current_roi = bounding_box_from_mask(current_mask)
    initial_roi = current_roi
    initial_mask = current_mask.copy()

    current_time = 0.0
    previous_posterior: NDArray[np.float64] | None = None
    previous_completion: float | None = None
    previous_full_frame: NDArray[np.uint8] | None = None
    previous_full_time: float | None = None
    previous_event: ScanEvent | None = None
    events: list[ScanEvent] = []
    operational_trace: list[ScanTraceEntry] = []
    belief_checks: list[_BeliefCheck] = []
    sample_variation = _SampleVariationAccumulator.for_shape(
        setup.timeline.shape
    )
    motions: list[MotionEstimate] = []
    scans: list[RasterVideoScan] = []
    planned_rois: list[BoundingBox] = []
    uncertainty_radii: list[float] = []
    completion_coverages: list[MaskCoverage] = []
    interval_mass_recalls: list[float] = []
    attention_coverages: list[float] = []
    attention_areas: list[float] = []
    kl_scores: list[float] = []
    js_scores: list[float] = []
    attention_hits = 0
    actual_successes = 0
    termination_reason = "max_scans_reached"
    observer_started = False

    try:
        if observer is not None:
            observer.start(context)
            observer_started = True
        for scan_index in range(scan_limit):
            try:
                current_frame = (
                    setup.timeline.frame_at_seconds(current_time)
                    if evidence_mode == "full-frame"
                    or observer is not None
                    else np.zeros(setup.timeline.shape, dtype=np.uint8)
                )
                scan = (
                    _instantaneous_full_frame_scan(setup.timeline, current_time)
                    if evidence_mode == "full-frame"
                    else _execute_scan(
                        setup,
                        started_at_seconds=current_time,
                        roi=current_roi,
                        mask=current_mask,
                    )
                )
            except TimelineExhausted:
                termination_reason = "source_exhausted_before_scan"
                break

            if evidence_mode == "full-frame":
                if previous_full_frame is not None and previous_full_time is not None:
                    elapsed = current_time - previous_full_time
                    frame_motion = estimate_opencv_motion(
                        previous_full_frame,
                        current_frame,
                        current_mask=foreground_mask(
                            current_frame,
                            threshold_ratio=(
                                setup.foreground_threshold_ratio
                            ),
                        ),
                    )
                    motion = MotionEstimate(
                        dy_per_t=frame_motion.dy_per_t / elapsed,
                        dx_per_t=frame_motion.dx_per_t / elapsed,
                        method="full_frame_opencv_px_per_second",
                        confidence=frame_motion.confidence,
                    )
                previous_full_frame = current_frame
                previous_full_time = current_time
                prior_at_completion = prior
            else:
                prior_at_completion, _observation_radius, _reacquire = (
                    _propagate_with_elapsed_uncertainty(
                        prior,
                        motion,
                        elapsed_seconds=scan.scan_duration_seconds,
                        setup=setup,
                    )
                )

            likelihood = scan_likelihood(
                scan.values,
                scan.sampled_mask,
                config=setup.model_profile.measurement,
            )
            posterior = posterior_from_likelihood(prior_at_completion, likelihood)
            if setup.motion_provider is not None:
                lookup = setup.motion_provider.lookup(
                    scan.completed_at_seconds
                )
                wind_lookup_counts[lookup.status] += 1
                if lookup.age_seconds is not None:
                    wind_measurement_ages.append(lookup.age_seconds)
                if lookup.motion is not None:
                    motion = lookup.motion
                elif (
                    evidence_mode == "scan"
                    and previous_posterior is not None
                    and previous_completion is not None
                ):
                    motion = distribution_centroid_motion(
                        previous_posterior,
                        posterior,
                        elapsed_seconds=(
                            scan.completed_at_seconds
                            - previous_completion
                        ),
                    )
                    wind_scan_fallbacks += 1
                else:
                    motion = zero_motion
                    wind_zero_fallbacks += 1
            elif (
                setup.known_motion is None
                and evidence_mode == "scan"
                and previous_posterior is not None
                and previous_completion is not None
            ):
                motion = distribution_centroid_motion(
                    previous_posterior,
                    posterior,
                    elapsed_seconds=scan.completed_at_seconds - previous_completion,
                )
            elif setup.known_motion is not None:
                motion = setup.known_motion

            if evidence_mode == "full-frame":
                policy_config = setup.model_profile.scan_policy
                target_horizon = (
                    policy_config.planning_initial_horizon_frames
                    / setup.render_fps
                )
                predictive, radius, reacquisition = _propagate_with_elapsed_uncertainty(
                    posterior,
                    motion,
                    elapsed_seconds=target_horizon,
                    setup=setup,
                )
                requested = (
                    _full_mask(setup.timeline.shape)
                    if reacquisition
                    else credible_region_mask(
                        predictive,
                        mass=policy_config.credible_mass,
                        blur_sigma=policy_config.credible_blur_sigma_px,
                        min_area_fraction=(
                            policy_config.minimum_area_fraction
                        ),
                        include_point=(
                            setup.source_point
                            if policy_config.force_known_source_in_scan
                            else None
                        ),
                        include_radius=(
                            policy_config.source_include_radius_min_px
                        ),
                    )
                )
                planned = _PlannedScan(
                    requested_mask=requested,
                    mechanical_mask=requested.copy(),
                    roi=bounding_box_from_mask(requested),
                    predictive=predictive,
                    duration_seconds=target_horizon,
                    uncertainty_radius=radius,
                    reacquisition=reacquisition,
                    converged=True,
                )
            else:
                planned = _plan_next_scan(
                    likelihood,
                    posterior,
                    motion,
                    setup,
                    scenario=scenario,
                    decision_time_seconds=scan.completed_at_seconds,
                    policy=policy,
                    budget=scan_budget,
                )

            decision_time = scan.completed_at_seconds
            target_time = decision_time + planned.duration_seconds
            event = ScanEvent(
                scan_index=scan_index,
                started_at_seconds=scan.started_at_seconds,
                completed_at_seconds=scan.completed_at_seconds,
                decision_at_seconds=decision_time,
                target_at_seconds=target_time,
            )
            event.validate(previous_event)
            events.append(event)
            posterior_check = _check_belief_integrity(posterior)
            predictive_check = _check_belief_integrity(
                planned.predictive
            )
            belief_checks.extend((posterior_check, predictive_check))
            operational_trace.append(
                _scan_trace_entry(
                    event=event,
                    scan=scan,
                    planned=planned,
                    posterior_check=posterior_check,
                    predictive_check=predictive_check,
                )
            )
            sample_variation.observe(scan)
            motions.append(motion)
            scans.append(scan)
            planned_rois.append(planned.roi)
            uncertainty_radii.append(planned.uncertainty_radius)
            previous_event = event

            try:
                timeline_end = setup.timeline.end_time_seconds
                if (
                    timeline_end is not None
                    and target_time > timeline_end + 1e-12
                ):
                    raise TimelineExhausted(
                        "Prediction target would occur after the source "
                        "timeline ends."
                    )
                spatial_truth_available = (
                    setup.truth_source is not None
                    or setup.sensor_derived_truth_allowed
                )
                target_frame = (
                    setup.timeline.frame_at_seconds(target_time)
                    if observer is not None
                    or (
                        setup.truth_source is None
                        and setup.sensor_derived_truth_allowed
                    )
                    else np.zeros(setup.timeline.shape, dtype=np.uint8)
                )
                interval_times = (
                    decision_time,
                    decision_time + planned.duration_seconds / 2.0,
                    target_time,
                )
                interval_metrics: list[MaskCoverage] = []
                if spatial_truth_available:
                    for evaluation_time in interval_times:
                        if setup.truth_source is not None:
                            evaluation_truth = (
                                setup.truth_source.truth_at_seconds(
                                    evaluation_time
                                )
                            )
                            truth_support = evaluation_truth.support
                            truth_distribution = (
                                evaluation_truth.distribution
                            )
                        else:
                            truth_frame = setup.timeline.frame_at_seconds(
                                evaluation_time
                            )
                            truth_support = foreground_mask(
                                truth_frame,
                                threshold_ratio=(
                                    setup.foreground_threshold_ratio
                                ),
                            )
                            truth_distribution = plume_truth_distribution(
                                truth_frame,
                                truth_support,
                            )
                        interval_metrics.append(
                            evaluate_scan_mask(
                                planned.mechanical_mask,
                                truth_distribution,
                                truth_support,
                            )
                        )
            except TimelineExhausted:
                termination_reason = "source_exhausted_before_prediction_target"
                break

            if evidence_mode == "scan" and interval_metrics:
                completion = interval_metrics[-1]
                completion_coverages.append(completion)
                interval_mass_recalls.append(
                    float(
                        np.mean(
                            [metric.probability_mass_recall for metric in interval_metrics]
                        )
                    )
                )
                actual_successes += int(
                    completion.probability_mass_recall >= success_coverage_threshold
                )

            attention_field = planned.predictive / max(float(planned.predictive.max()), 1e-12)
            attention_mask = (
                attention_field
                >= setup.attention_diagnostic_relative_density
            )
            if setup.truth_source is not None:
                target_truth = setup.truth_source.truth_at_seconds(
                    target_time
                )
                target_support = target_truth.support
                target_distribution = target_truth.distribution
            elif setup.sensor_derived_truth_allowed:
                target_support = foreground_mask(
                    target_frame,
                    threshold_ratio=setup.foreground_threshold_ratio,
                )
                target_distribution = plume_truth_distribution(
                    target_frame,
                    target_support,
                )
            else:
                target_support = np.zeros(
                    setup.timeline.shape,
                    dtype=bool,
                )
                target_distribution = np.zeros(
                    setup.timeline.shape,
                    dtype=float,
                )
            attention_areas.append(
                float(
                    np.count_nonzero(attention_mask)
                    / attention_mask.size
                )
            )
            if spatial_truth_available:
                support_count = int(np.count_nonzero(target_support))
                attention_overlap = int(
                    np.count_nonzero(attention_mask & target_support)
                )
                attention_hits += int(attention_overlap > 0)
                attention_coverages.append(
                    attention_overlap / support_count
                    if support_count
                    else 0.0
                )
                predicted_distribution = normalize_distribution(
                    attention_field
                )
                kl_scores.append(
                    kl_divergence(
                        target_distribution,
                        predicted_distribution,
                    )
                )
                js_scores.append(
                    js_divergence(
                        target_distribution,
                        predicted_distribution,
                    )
                )

            if observer is not None:
                observer.observe(
                    ExperimentStep(
                        current_frame=current_frame,
                        scan=scan,
                        target_frame=target_frame,
                        target_support=target_support,
                        attention_field=attention_field,
                        next_scan_mask=planned.mechanical_mask,
                        motion_px_per_second=motion,
                    )
                )

            previous_posterior = posterior
            previous_completion = scan.completed_at_seconds
            prior = posterior
            current_mask = planned.requested_mask
            current_roi = planned.roi
            current_time = (
                target_time if evidence_mode == "full-frame" else scan.completed_at_seconds
            )
    finally:
        if observer is not None and observer_started:
            observer.close()

    durations = [scan.scan_duration_seconds for scan in scans]
    swept_fractions = [scan.mechanical_sweep_area_fraction for scan in scans]
    requested_fractions = [scan.requested_area_fraction for scan in scans]
    envelope_fractions = [
        roi.area / (height * width)
        for roi in ([initial_roi] + planned_rois[:-1] if planned_rois else [initial_roi])
    ]
    sample_points = [scan.effective_sample_points for scan in scans]
    sweep_rows = [
        int(np.unique(np.where(scan.sampled_mask)[0]).size)
        if scan.scan_mode == "column"
        else 0
        for scan in scans
    ]
    vertical_travel = [scan.mean_vertical_travel_rows for scan in scans]
    vertical_passes = [scan.mean_vertical_pass_seconds for scan in scans]
    mean_motion_dx_s = float(np.mean([item.dx_per_t for item in motions])) if motions else 0.0
    mean_motion_dy_s = float(np.mean([item.dy_per_t for item in motions])) if motions else 0.0
    initial_mask_fraction = float(
        np.count_nonzero(
            _full_mask(setup.timeline.shape)
            if evidence_mode == "full-frame"
            else initial_mask
        )
        / (height * width)
    )
    timeline_verified = True
    prior_event: ScanEvent | None = None
    for item in events:
        item.validate(prior_event)
        prior_event = item

    recorded_truth_evidence: dict[str, object] = {}
    if (
        isinstance(
            setup.data_provenance.get("manifest_schema_version"),
            int,
        )
        and setup.data_provenance["manifest_schema_version"] >= 2
    ):
        evidence_contract = setup.data_provenance.get(
            "evidence_contract"
        )
        if isinstance(evidence_contract, dict):
            truth_evidence = evidence_contract.get("truth")
            if isinstance(truth_evidence, dict):
                recorded_truth_evidence = dict(truth_evidence)

    report = VideoProofReport(
        output_video="",
        scenario=scenario,
        frames_processed=len(events),
        scan_start_mode=setup.scan_start_mode,
        plume_mapping_mode=(
            "explicit_full_frame_baseline"
            if evidence_mode == "full-frame"
            else setup.plume_mapping_mode
        ),
        frame_stride=frame_stride,
        effective_fps=setup.render_fps,
        mean_dx=mean_motion_dx_s / setup.render_fps,
        mean_dy=mean_motion_dy_s / setup.render_fps,
        scan_mode=setup.scan_mode,
        scan_rate_hz=setup.scan_rate_hz,
        scan_sample_stride_px=(
            setup.column_stride_px if setup.scan_mode == "column" else setup.sample_stride_px
        ),
        scan_sample_points=int(round(float(np.mean(sample_points)))) if sample_points else 0,
        scan_vertical_pass_seconds=(
            setup.vertical_pass_seconds if setup.scan_mode == "column" else 0.0
        ),
        scan_horizontal_step_seconds=(
            setup.horizontal_step_seconds if setup.scan_mode == "column" else 0.0
        ),
        scan_column_mass_fraction=setup.column_mass_fraction,
        scan_row_mass_fraction=setup.row_mass_fraction,
        scan_min_sweep_columns=setup.min_sweep_columns if setup.scan_mode == "column" else 0,
        scan_min_sweep_rows=setup.min_sweep_rows if setup.scan_mode == "column" else 0,
        scan_max_sweep_columns=(
            setup.max_sweep_columns if setup.scan_mode == "column" else None
        ),
        scan_sweep_rows=int(round(float(np.mean(sweep_rows)))) if sweep_rows else 0,
        scan_vertical_reference_rows=height if setup.scan_mode == "column" else 0,
        scan_mean_vertical_travel_rows=float(np.mean(vertical_travel)) if vertical_travel else 0.0,
        scan_mean_vertical_pass_seconds=float(np.mean(vertical_passes)) if vertical_passes else 0.0,
        scan_duration_seconds=float(np.mean(durations)) if durations else 0.0,
        scan_duration_frames=(
            float(np.mean(durations)) * setup.render_fps if durations else 0.0
        ),
        scan_pixels=int(round(float(np.mean(swept_fractions)) * height * width))
        if swept_fractions
        else 0,
        scan_area_fraction=float(np.mean(swept_fractions)) if swept_fractions else 0.0,
        hits=attention_hits,
        evaluated_predictions=len(attention_coverages),
        mean_attention_coverage=float(np.mean(attention_coverages))
        if attention_coverages
        else 0.0,
        mean_attention_area_fraction=float(np.mean(attention_areas)) if attention_areas else 0.0,
        attention_mask_threshold=(
            setup.attention_diagnostic_relative_density
        ),
        mean_uncertainty_radius=float(np.mean(uncertainty_radii))
        if uncertainty_radii
        else 0.0,
        mean_kl_true_to_attention=float(np.mean(kl_scores)) if kl_scores else 0.0,
        mean_js_divergence=float(np.mean(js_scores)) if js_scores else 0.0,
        initial_roi_area_fraction=initial_roi.area / (height * width),
        initial_mask_area_fraction=initial_mask_fraction,
        mean_envelope_area_fraction=float(np.mean(envelope_fractions))
        if envelope_fractions
        else 0.0,
        final_envelope_area_fraction=envelope_fractions[-1] if envelope_fractions else 0.0,
        mean_mask_area_fraction=float(np.mean(requested_fractions))
        if requested_fractions
        else 0.0,
        final_mask_area_fraction=requested_fractions[-1] if requested_fractions else 0.0,
        mean_swept_area_fraction=float(np.mean(swept_fractions)) if swept_fractions else 0.0,
        final_swept_area_fraction=swept_fractions[-1] if swept_fractions else 0.0,
        evidence_mode=evidence_mode,
        motion_source=(
            setup.known_motion.method
            if setup.known_motion is not None
            else motions[-1].method
            if motions
            else zero_motion.method
        ),
        scans_completed=len(scans),
        termination_reason=termination_reason,
        timeline_integrity_verified=timeline_verified,
        success_coverage_threshold=success_coverage_threshold,
        actual_sweep_successes=actual_successes,
        actual_sweep_evaluated_predictions=len(completion_coverages),
        mean_actual_sweep_support_recall=float(
            np.mean([item.support_recall for item in completion_coverages])
        )
        if completion_coverages
        else 0.0,
        mean_actual_sweep_mass_recall=float(
            np.mean([item.probability_mass_recall for item in completion_coverages])
        )
        if completion_coverages
        else 0.0,
        mean_actual_sweep_interval_mass_recall=float(np.mean(interval_mass_recalls))
        if interval_mass_recalls
        else 0.0,
        mean_actual_sweep_iou=float(
            np.mean([item.intersection_over_union for item in completion_coverages])
        )
        if completion_coverages
        else 0.0,
        initial_scan_duration_seconds=durations[0] if durations else 0.0,
        max_scan_duration_seconds=max(durations) if durations else 0.0,
        final_scan_duration_seconds=durations[-1] if durations else 0.0,
        total_elapsed_seconds=scans[-1].completed_at_seconds if scans else 0.0,
        mean_dx_px_per_second=mean_motion_dx_s,
        mean_dy_px_per_second=mean_motion_dy_s,
        oracle_inputs=setup.oracle_inputs,
        policy_id=(
            policy.policy_id
            if policy is not None
            else "adaptive-predictive"
        ),
        model_profile=setup.model_profile.to_payload(
            frame_rate_hz=setup.timeline.fps,
        ),
        data_provenance=setup.data_provenance,
        evaluation_truth=(
            {
                "source": "hashed_dense_sidecar",
                "basis": setup.truth_source.truth_basis,
                "available_to_policy": False,
                **recorded_truth_evidence,
            }
            if setup.truth_source is not None
            else {
                "source": "derived_from_sensor_frame",
                "basis": "foreground_relative_intensity",
                "available_to_policy": False,
                "claim_eligibility": "diagnostic_only",
            }
            if setup.sensor_derived_truth_allowed
            else {
                "source": "none",
                "basis": "unavailable",
                "available_to_policy": False,
                "spatial_metrics_available": False,
                "spatial_metric_samples": 0,
                **recorded_truth_evidence,
            }
        ),
        wind_evidence=(
            {
                "provider": setup.motion_provider.method,
                "causal_lookup": "latest_measurement_not_after_decision",
                "lookup_counts": wind_lookup_counts,
                "mean_measurement_age_seconds": (
                    float(np.mean(wind_measurement_ages))
                    if wind_measurement_ages
                    else None
                ),
                "max_measurement_age_seconds": (
                    max(wind_measurement_ages)
                    if wind_measurement_ages
                    else None
                ),
                "scan_derived_fallbacks": wind_scan_fallbacks,
                "zero_motion_fallbacks": wind_zero_fallbacks,
            }
            if setup.motion_provider is not None
            else {
                "provider": (
                    setup.known_motion.method
                    if setup.known_motion is not None
                    else "scan_derived_or_full_frame"
                ),
                "causal_lookup": "legacy_runtime_behavior",
            }
        ),
        operational_trace=tuple(operational_trace),
        belief_integrity=_belief_integrity_summary(belief_checks),
        sample_variation=sample_variation.summary(
            full_frame_evidence_mode=(evidence_mode == "full-frame")
        ),
    )
    return ExperimentResult(report=report)
