"""Contracts for provenance-checked recorded plume sequences."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import math
from pathlib import PurePosixPath
from typing import Literal, Protocol

from numpy.typing import NDArray
import numpy as np

from plume_focus.experiment import TimelineSource
from plume_focus.model_profiles import ModelProfile
from plume_focus.motion import MotionEstimate
from plume_focus.predictor import BoundingBox
from plume_focus.sensor_emulation import SensorEmulationProfile


RecordedSplit = Literal["development", "held-out"]
RecordedPlannerScenario = Literal["plume", "methane"]
GasIdentity = Literal[
    "methane",
    "propane",
    "hydrocarbon_unspecified",
]
QualificationPurpose = Literal["diagnostic_raw_pilot", "evidence_suite"]
ClockMode = Literal["constant_fps", "timestamp_csv"]
NormalizationMode = Literal["uint8_identity", "fixed_linear"]
WindMode = Literal["none", "constant", "time_series_csv"]
SourceAssumptionMode = Literal[
    "known_source",
    "known_corridor",
    "source_free",
]
TruthKind = Literal["binary_support", "probability_weight"]
TruthStorageFormat = Literal["none", "legacy_video", "lossless_npz_uint8"]
CalibrationScope = Literal["development_only", "independent_pre_run"]
SensorInputKind = Literal[
    "raw_recorded",
    "structure_suppressed_emulation",
    "legacy_unclassified",
]
ScannerEvidenceKind = Literal[
    "counterfactual_virtual",
    "measured_hardware",
    "legacy_unclassified",
]
TruthEvidenceGrade = Literal[
    "none",
    "silver_pseudo",
    "gold_human_reviewed",
    "legacy_unclassified",
]
ClaimEligibility = Literal[
    "legacy_unclassified_evidence",
    "blocked_unverified_data_rights",
    "no_spatial_evaluation_truth",
    "diagnostic_only_pseudo_truth",
    "offline_counterfactual_scan_evidence",
    "sensor_emulation_requires_hardware_validation",
    "requires_frozen_suite_and_independence_review",
]
MotionLookupStatus = Literal["available", "missing", "stale"]

SEQUENCE_MANIFEST_SCHEMA_VERSION = 3
SUPPORTED_SEQUENCE_MANIFEST_SCHEMA_VERSIONS = frozenset({1, 2, 3, 4, 5})
SUITE_MANIFEST_SCHEMA_VERSION = 1


def _require_text(name: str, value: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{name} must not be empty.")


def _require_relative_path(name: str, value: str) -> None:
    _require_text(name, value)
    path = PurePosixPath(value)
    if (
        "\\" in value
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise ValueError(
            f"{name} must be a normalized relative POSIX path."
        )


def _require_finite(
    name: str,
    value: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None:
        invalid = value <= minimum if strict_minimum else value < minimum
        if invalid:
            qualifier = "greater than" if strict_minimum else "at least"
            raise ValueError(f"{name} must be {qualifier} {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}.")


@dataclass(frozen=True)
class GasIdentitySpec:
    """Declared gas identity kept separate from the planner scenario."""

    identity: GasIdentity
    provenance: str

    def __post_init__(self) -> None:
        if self.identity not in {
            "methane",
            "propane",
            "hydrocarbon_unspecified",
        }:
            raise ValueError("unknown recorded gas identity.")
        _require_text("gas identity provenance", self.provenance)


@dataclass(frozen=True)
class DatasetQualificationLineageSpec:
    """Exact purpose-aware qualification checked before schema-v5 use."""

    path: str
    sha256: str
    purpose: QualificationPurpose

    def __post_init__(self) -> None:
        _require_relative_path("dataset qualification path", self.path)
        _require_sha256("dataset qualification sha256", self.sha256)
        if self.purpose not in {
            "diagnostic_raw_pilot",
            "evidence_suite",
        }:
            raise ValueError("unknown dataset qualification purpose.")


@dataclass(frozen=True)
class MediaSpec:
    """Hashed sensor-video asset."""

    path: str
    sha256: str
    capture_provenance: str

    def __post_init__(self) -> None:
        _require_text("media path", self.path)
        _require_sha256("media sha256", self.sha256)
        _require_text(
            "media capture provenance",
            self.capture_provenance,
        )


@dataclass(frozen=True)
class MediaWindowSpec:
    """Provenance-bound, half-open window within one parent media asset."""

    requested_start_time_seconds: float
    requested_end_time_seconds: float
    source_start_frame_index: int
    source_end_frame_index_exclusive: int
    first_included_timestamp_seconds: float
    last_included_timestamp_seconds: float
    parent_media_sha256: str
    decoded_window_sha256: str
    selection_metadata_path: str
    selection_metadata_sha256: str
    selection_provenance: str

    def __post_init__(self) -> None:
        _require_finite(
            "requested_start_time_seconds",
            self.requested_start_time_seconds,
            minimum=0.0,
        )
        _require_finite(
            "requested_end_time_seconds",
            self.requested_end_time_seconds,
            minimum=0.0,
        )
        if self.requested_end_time_seconds <= self.requested_start_time_seconds:
            raise ValueError(
                "requested_end_time_seconds must be greater than "
                "requested_start_time_seconds."
            )
        for name, value in (
            ("source_start_frame_index", self.source_start_frame_index),
            (
                "source_end_frame_index_exclusive",
                self.source_end_frame_index_exclusive,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer.")
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")
        if (
            self.source_end_frame_index_exclusive
            - self.source_start_frame_index
            < 3
        ):
            raise ValueError(
                "media windows must include at least three source frames."
            )
        _require_finite(
            "first_included_timestamp_seconds",
            self.first_included_timestamp_seconds,
            minimum=0.0,
        )
        _require_finite(
            "last_included_timestamp_seconds",
            self.last_included_timestamp_seconds,
            minimum=0.0,
        )
        if (
            self.first_included_timestamp_seconds
            < self.requested_start_time_seconds
        ):
            raise ValueError(
                "first_included_timestamp_seconds must not precede the "
                "requested start."
            )
        if (
            self.last_included_timestamp_seconds
            < self.first_included_timestamp_seconds
        ):
            raise ValueError(
                "last_included_timestamp_seconds must not precede the "
                "first included timestamp."
            )
        if (
            self.last_included_timestamp_seconds
            >= self.requested_end_time_seconds
        ):
            raise ValueError(
                "last_included_timestamp_seconds must precede the "
                "exclusive requested end."
            )
        _require_sha256("parent media sha256", self.parent_media_sha256)
        _require_sha256("decoded window sha256", self.decoded_window_sha256)
        _require_text(
            "selection metadata path",
            self.selection_metadata_path,
        )
        _require_sha256(
            "selection metadata sha256",
            self.selection_metadata_sha256,
        )
        _require_text("selection provenance", self.selection_provenance)


@dataclass(frozen=True)
class DatasetLineageSpec:
    """Reference from one sequence to its immutable accession audit."""

    dataset_id: str
    accession_id: str
    audit_path: str
    audit_sha256: str
    source_asset_sha256: str

    def __post_init__(self) -> None:
        for name, value in (
            ("dataset_id", self.dataset_id),
            ("accession_id", self.accession_id),
            ("dataset audit path", self.audit_path),
        ):
            _require_text(name, value)
        _require_sha256("dataset audit sha256", self.audit_sha256)
        _require_sha256(
            "dataset source asset sha256",
            self.source_asset_sha256,
        )


@dataclass(frozen=True)
class CaptureGroupSpec:
    """Acquisition-level grouping used to prevent split leakage."""

    capture_group_id: str
    grouping_provenance: str

    def __post_init__(self) -> None:
        _require_text("capture_group_id", self.capture_group_id)
        _require_text(
            "capture group provenance",
            self.grouping_provenance,
        )


@dataclass(frozen=True)
class LeakOffCalibrationAssetSpec:
    """Separate hashed leak-off media used for fixed sensor calibration."""

    path: str
    sha256: str
    provenance: str
    calibration_scope: CalibrationScope

    def __post_init__(self) -> None:
        _require_text("leak-off calibration path", self.path)
        _require_sha256("leak-off calibration sha256", self.sha256)
        _require_text(
            "leak-off calibration provenance",
            self.provenance,
        )
        if self.calibration_scope not in {
            "development_only",
            "independent_pre_run",
        }:
            raise ValueError("unknown leak-off calibration_scope.")


@dataclass(frozen=True)
class FrozenCalibrationAssetSpec:
    """Canonical calibration plus its audited leak-off source lineage."""

    path: str
    sha256: str
    source_path: str
    source_sha256: str
    source_provenance: str
    calibration_scope: CalibrationScope
    producing_commit: str
    preparation_spec_path: str
    preparation_spec_sha256: str
    preparation_profile_path: str
    preparation_profile_sha256: str
    calibration_record_path: str
    calibration_record_sha256: str

    def __post_init__(self) -> None:
        for name, value in (
            ("frozen calibration path", self.path),
            ("calibration source path", self.source_path),
            ("calibration source provenance", self.source_provenance),
            ("preparation spec path", self.preparation_spec_path),
            ("preparation profile path", self.preparation_profile_path),
            ("calibration record path", self.calibration_record_path),
        ):
            _require_text(name, value)
        _require_sha256("frozen calibration sha256", self.sha256)
        _require_sha256(
            "calibration source sha256",
            self.source_sha256,
        )
        _require_full_git_commit(
            "calibration producing_commit",
            self.producing_commit,
        )
        _require_sha256(
            "preparation spec sha256",
            self.preparation_spec_sha256,
        )
        _require_sha256(
            "preparation profile sha256",
            self.preparation_profile_sha256,
        )
        _require_sha256(
            "calibration record sha256",
            self.calibration_record_sha256,
        )
        if self.calibration_scope not in {
            "development_only",
            "independent_pre_run",
        }:
            raise ValueError("unknown frozen calibration_scope.")
        if self.path == self.source_path:
            raise ValueError(
                "frozen calibration and its leak-off source must be "
                "separate assets."
            )
        if self.sha256.lower() == self.source_sha256.lower():
            raise ValueError(
                "frozen calibration bytes must differ from source media."
            )


@dataclass(frozen=True)
class SensorInputSpec:
    """Raw recorded values or a declared structure-suppressed emulation."""

    kind: SensorInputKind
    provenance: str
    emulation_profile: SensorEmulationProfile | None = None
    leak_off_calibration: LeakOffCalibrationAssetSpec | None = None
    frozen_calibration: FrozenCalibrationAssetSpec | None = None

    def __post_init__(self) -> None:
        _require_text("sensor input provenance", self.provenance)
        if self.kind == "raw_recorded":
            if (
                self.emulation_profile is not None
                or self.leak_off_calibration is not None
                or self.frozen_calibration is not None
            ):
                raise ValueError(
                    "raw_recorded sensor input takes no emulation profile "
                    "or leak-off calibration."
                )
            return
        if self.kind == "structure_suppressed_emulation":
            calibration_count = sum(
                value is not None
                for value in (
                    self.leak_off_calibration,
                    self.frozen_calibration,
                )
            )
            if self.emulation_profile is None or calibration_count != 1:
                raise ValueError(
                    "structure_suppressed_emulation requires a profile and "
                    "exactly one calibration asset."
                )
            if (
                self.emulation_profile.smoothing_time_constant_seconds
                is not None
            ):
                raise ValueError(
                    "sensor-emulation temporal smoothing is unsupported; "
                    "smoothing_time_constant_seconds must be null."
                )
            return
        if self.kind == "legacy_unclassified":
            if (
                self.emulation_profile is not None
                or self.leak_off_calibration is not None
                or self.frozen_calibration is not None
            ):
                raise ValueError(
                    "legacy_unclassified sensor input takes no v2 settings."
                )
            return
        raise ValueError("unknown sensor input kind.")


@dataclass(frozen=True)
class FrameClockSpec:
    """Constant-rate or sidecar timestamp clock."""

    mode: ClockMode
    provenance: str
    fps: float | None = None
    path: str | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        _require_text("frame clock provenance", self.provenance)
        if self.mode == "constant_fps":
            if self.fps is None:
                raise ValueError("constant_fps clock requires fps.")
            _require_finite(
                "frame clock fps",
                self.fps,
                minimum=0.0,
                strict_minimum=True,
            )
            if self.path is not None or self.sha256 is not None:
                raise ValueError(
                    "constant_fps clock must not include a timestamp sidecar."
                )
            return
        if self.mode == "timestamp_csv":
            if self.fps is not None:
                raise ValueError(
                    "timestamp_csv clock derives fps and must not declare it."
                )
            if self.path is None or self.sha256 is None:
                raise ValueError(
                    "timestamp_csv clock requires path and sha256."
                )
            _require_text("timestamp path", self.path)
            _require_sha256("timestamp sha256", self.sha256)
            return
        raise ValueError("unknown frame clock mode.")


@dataclass(frozen=True)
class NormalizationSpec:
    """Sequence-fixed sensor normalization."""

    mode: NormalizationMode
    provenance: str
    raw_min: float | None = None
    raw_max: float | None = None

    def __post_init__(self) -> None:
        _require_text(
            "normalization provenance",
            self.provenance,
        )
        if self.mode == "uint8_identity":
            if self.raw_min is not None or self.raw_max is not None:
                raise ValueError(
                    "uint8_identity normalization takes no raw bounds."
                )
            return
        if self.mode == "fixed_linear":
            if self.raw_min is None or self.raw_max is None:
                raise ValueError(
                    "fixed_linear normalization requires raw_min and raw_max."
                )
            _require_finite("normalization raw_min", self.raw_min)
            _require_finite("normalization raw_max", self.raw_max)
            if self.raw_min >= self.raw_max:
                raise ValueError("normalization raw_min must be below raw_max.")
            return
        raise ValueError("unknown normalization mode.")


@dataclass(frozen=True)
class SensorCalibrationSpec:
    """Sensor identity and calibration provenance without a calibration claim."""

    sensor_id: str
    signal_units: str
    provenance: str

    def __post_init__(self) -> None:
        _require_text("sensor_id", self.sensor_id)
        _require_text("signal_units", self.signal_units)
        _require_text("sensor calibration provenance", self.provenance)


@dataclass(frozen=True)
class WindSpec:
    """Causal wind/advection input declared in image-plane units."""

    mode: WindMode
    provenance: str
    conversion_provenance: str
    units: str = "px_per_second"
    dy_px_per_second: float | None = None
    dx_px_per_second: float | None = None
    path: str | None = None
    sha256: str | None = None
    max_age_seconds: float | None = None

    def __post_init__(self) -> None:
        _require_text("wind provenance", self.provenance)
        _require_text(
            "wind conversion provenance",
            self.conversion_provenance,
        )
        if self.units != "px_per_second":
            raise ValueError("recorded wind units must be px_per_second.")
        if self.mode == "none":
            if any(
                value is not None
                for value in (
                    self.dy_px_per_second,
                    self.dx_px_per_second,
                    self.path,
                    self.sha256,
                    self.max_age_seconds,
                )
            ):
                raise ValueError("wind mode none takes no samples or values.")
            return
        if self.mode == "constant":
            if (
                self.dy_px_per_second is None
                or self.dx_px_per_second is None
            ):
                raise ValueError(
                    "constant wind requires dy_px_per_second and "
                    "dx_px_per_second."
                )
            _require_finite(
                "wind dy_px_per_second",
                self.dy_px_per_second,
            )
            _require_finite(
                "wind dx_px_per_second",
                self.dx_px_per_second,
            )
            if any(
                value is not None
                for value in (
                    self.path,
                    self.sha256,
                    self.max_age_seconds,
                )
            ):
                raise ValueError(
                    "constant wind must not include a sidecar or max age."
                )
            return
        if self.mode == "time_series_csv":
            if self.path is None or self.sha256 is None:
                raise ValueError(
                    "time_series_csv wind requires path and sha256."
                )
            if self.max_age_seconds is None:
                raise ValueError(
                    "time_series_csv wind requires max_age_seconds."
                )
            _require_text("wind sidecar path", self.path)
            _require_sha256("wind sidecar sha256", self.sha256)
            _require_finite(
                "wind max_age_seconds",
                self.max_age_seconds,
                minimum=0.0,
            )
            if (
                self.dy_px_per_second is not None
                or self.dx_px_per_second is not None
            ):
                raise ValueError(
                    "time-series wind must not include constant values."
                )
            return
        raise ValueError("unknown wind mode.")


@dataclass(frozen=True)
class SourceAssumptionSpec:
    """Explicit source assumption and initial search region."""

    mode: SourceAssumptionMode
    provenance: str
    source_point: tuple[float, float] | None = None
    initial_roi: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        _require_text("source assumption provenance", self.provenance)
        if self.mode == "known_source":
            if self.source_point is None or self.initial_roi is None:
                raise ValueError(
                    "known_source requires source_point and initial_roi."
                )
            _validate_point(self.source_point)
            _validate_roi(self.initial_roi)
            return
        if self.mode == "known_corridor":
            if self.source_point is not None or self.initial_roi is None:
                raise ValueError(
                    "known_corridor requires initial_roi and no source_point."
                )
            _validate_roi(self.initial_roi)
            return
        if self.mode == "source_free":
            if self.source_point is not None or self.initial_roi is not None:
                raise ValueError(
                    "source_free takes no source_point or initial_roi."
                )
            return
        raise ValueError("unknown source assumption mode.")


@dataclass(frozen=True)
class ScannerSpec:
    """Complete recorded-run mechanical scanner settings."""

    provenance: str
    scan_mode: Literal["point", "column"]
    scan_rate_hz: float
    sample_stride_px: int
    vertical_pass_seconds: float
    horizontal_step_seconds: float
    column_stride_px: int
    column_mass_fraction: float
    row_mass_fraction: float
    minimum_sweep_columns: int
    minimum_sweep_rows: int
    maximum_sweep_columns: int | None
    evidence_kind: ScannerEvidenceKind = "legacy_unclassified"

    def __post_init__(self) -> None:
        _require_text("scanner provenance", self.provenance)
        if self.scan_mode not in {"point", "column"}:
            raise ValueError("scan_mode must be point or column.")
        for name, value in (
            ("scan_rate_hz", self.scan_rate_hz),
            ("vertical_pass_seconds", self.vertical_pass_seconds),
        ):
            _require_finite(
                name,
                value,
                minimum=0.0,
                strict_minimum=True,
            )
        _require_finite(
            "horizontal_step_seconds",
            self.horizontal_step_seconds,
            minimum=0.0,
        )
        for name, value in (
            ("sample_stride_px", self.sample_stride_px),
            ("column_stride_px", self.column_stride_px),
            ("minimum_sweep_columns", self.minimum_sweep_columns),
            ("minimum_sweep_rows", self.minimum_sweep_rows),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        for name, value in (
            ("column_mass_fraction", self.column_mass_fraction),
            ("row_mass_fraction", self.row_mass_fraction),
        ):
            _require_finite(
                name,
                value,
                minimum=0.0,
                maximum=1.0,
                strict_minimum=True,
            )
        if (
            self.maximum_sweep_columns is not None
            and self.maximum_sweep_columns < self.minimum_sweep_columns
        ):
            raise ValueError(
                "maximum_sweep_columns must be at least "
                "minimum_sweep_columns."
            )
        if self.evidence_kind not in {
            "counterfactual_virtual",
            "measured_hardware",
            "legacy_unclassified",
        }:
            raise ValueError("unknown scanner evidence kind.")


@dataclass(frozen=True)
class TruthSpec:
    """Optional or hashed dense offline truth sidecar."""

    annotation_provenance: str
    path: str | None = None
    sha256: str | None = None
    truth_kind: TruthKind | None = None
    storage_format: TruthStorageFormat = "legacy_video"
    evidence_grade: TruthEvidenceGrade = "legacy_unclassified"
    method_id: str | None = None
    protocol_path: str | None = None
    protocol_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_text(
            "truth annotation provenance",
            self.annotation_provenance,
        )
        if self.evidence_grade == "none":
            if self.storage_format != "none":
                raise ValueError(
                    "truth grade none requires storage_format none."
                )
            if any(
                value is not None
                for value in (
                    self.path,
                    self.sha256,
                    self.truth_kind,
                    self.method_id,
                    self.protocol_path,
                    self.protocol_sha256,
                )
            ):
                raise ValueError(
                    "truth grade none takes no truth or protocol assets."
                )
            return
        if (
            self.path is None
            or self.sha256 is None
            or self.truth_kind is None
        ):
            raise ValueError(
                "recorded truth requires path, sha256, and truth_kind."
            )
        _require_text("truth path", self.path)
        _require_sha256("truth sha256", self.sha256)
        if self.truth_kind not in {
            "binary_support",
            "probability_weight",
        }:
            raise ValueError("unknown truth_kind.")
        if self.storage_format not in {
            "legacy_video",
            "lossless_npz_uint8",
        }:
            raise ValueError("unknown truth storage format.")
        if self.evidence_grade == "legacy_unclassified":
            if any(
                value is not None
                for value in (
                    self.method_id,
                    self.protocol_path,
                    self.protocol_sha256,
                )
            ):
                raise ValueError(
                    "legacy truth evidence takes no v2 method or protocol."
                )
            return
        if self.evidence_grade not in {
            "silver_pseudo",
            "gold_human_reviewed",
        }:
            raise ValueError("unknown truth evidence grade.")
        if (
            self.method_id is None
            or self.protocol_path is None
            or self.protocol_sha256 is None
        ):
            raise ValueError(
                "classified truth evidence requires method_id, "
                "protocol_path, and protocol_sha256."
            )
        _require_text("truth method_id", self.method_id)
        _require_text("truth protocol_path", self.protocol_path)
        _require_sha256("truth protocol_sha256", self.protocol_sha256)


@dataclass(frozen=True)
class RecordedSequenceManifest:
    """Supported manifest for one provenance-checked recorded sequence."""

    sequence_id: str
    split: RecordedSplit
    split_provenance: str
    scenario: RecordedPlannerScenario
    media: MediaSpec
    frame_clock: FrameClockSpec
    normalization: NormalizationSpec
    calibration: SensorCalibrationSpec
    wind: WindSpec
    source_assumption: SourceAssumptionSpec
    scanner: ScannerSpec
    evaluation_truth: TruthSpec
    model_profile_id: str = "uncalibrated_video_v1"
    gas_identity: GasIdentitySpec | None = None
    dataset_qualification: DatasetQualificationLineageSpec | None = None
    dataset_lineage: DatasetLineageSpec | None = None
    capture_group: CaptureGroupSpec | None = None
    sensor_input: SensorInputSpec | None = None
    media_window: MediaWindowSpec | None = None
    manifest_schema_version: int = SEQUENCE_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text("sequence_id", self.sequence_id)
        _require_text("split provenance", self.split_provenance)
        if self.split not in {"development", "held-out"}:
            raise ValueError("split must be development or held-out.")
        if (
            self.manifest_schema_version != 5
            and self.scenario != "methane"
        ):
            raise ValueError(
                "recorded manifests support the methane scenario only."
            )
        if self.model_profile_id != "uncalibrated_video_v1":
            raise ValueError(
                "recorded manifests require uncalibrated_video_v1."
            )
        if (
            isinstance(self.manifest_schema_version, bool)
            or not isinstance(self.manifest_schema_version, int)
            or self.manifest_schema_version
            not in SUPPORTED_SEQUENCE_MANIFEST_SCHEMA_VERSIONS
        ):
            raise ValueError("unsupported sequence manifest schema version.")
        if self.manifest_schema_version < 5:
            if self.gas_identity is not None:
                raise ValueError(
                    "gas_identity requires sequence manifest schema "
                    "version 5."
                )
            if self.dataset_qualification is not None:
                raise ValueError(
                    "dataset_qualification requires sequence manifest "
                    "schema version 5."
                )
        else:
            if self.gas_identity is None:
                raise ValueError(
                    "schema-v5 manifests require gas_identity."
                )
            expected_scenario = (
                "methane"
                if self.gas_identity.identity == "methane"
                else "plume"
            )
            if self.scenario != expected_scenario:
                raise ValueError(
                    "schema-v5 planner scenario is incompatible with "
                    "gas_identity."
                )
            if self.dataset_qualification is None:
                raise ValueError(
                    "schema-v5 manifests require dataset_qualification."
                )
        if self.manifest_schema_version == 1:
            if any(
                value is not None
                for value in (
                    self.dataset_lineage,
                    self.capture_group,
                    self.sensor_input,
                    self.media_window,
                )
            ):
                raise ValueError(
                    "schema-v1 manifests must not include classified "
                    "evidence fields."
                )
            if self.scanner.evidence_kind != "legacy_unclassified":
                raise ValueError(
                    "schema-v1 scanner evidence must remain unclassified."
                )
            if (
                self.evaluation_truth.evidence_grade
                != "legacy_unclassified"
                or self.evaluation_truth.storage_format != "legacy_video"
            ):
                raise ValueError(
                    "schema-v1 truth evidence must remain legacy video."
                )
            return

        if self.manifest_schema_version in {1, 2, 3}:
            if self.media_window is not None:
                raise ValueError(
                    "media_window requires sequence manifest schema version 4."
                )
        elif self.manifest_schema_version == 4 and self.media_window is None:
            raise ValueError(
                "schema-v4 manifests require media_window."
            )

        if (
            self.dataset_lineage is None
            or self.capture_group is None
            or self.sensor_input is None
        ):
            raise ValueError(
                "classified manifests require dataset_lineage, "
                "capture_group, and sensor_input."
            )
        if (
            self.dataset_lineage.source_asset_sha256.lower()
            != self.media.sha256.lower()
        ):
            raise ValueError(
                "dataset source asset sha256 must match media sha256."
            )
        if self.scanner.evidence_kind == "legacy_unclassified":
            raise ValueError(
                "classified manifests require scanner evidence."
            )
        if self.scanner.evidence_kind != "counterfactual_virtual":
            raise ValueError(
                "recorded-video manifests support only "
                "counterfactual_virtual scanner evidence; measured_hardware "
                "is reserved for a future measured-scan adapter."
            )
        truth = self.evaluation_truth
        if truth.evidence_grade == "legacy_unclassified":
            raise ValueError(
                "classified manifests require classified truth evidence."
            )
        if truth.evidence_grade == "none":
            if self.manifest_schema_version < 3:
                raise ValueError(
                    "truth-free recorded input requires schema version 3."
                )
        else:
            if truth.storage_format != "lossless_npz_uint8":
                raise ValueError(
                    "classified truth must use lossless_npz_uint8 storage."
                )
            assert truth.path is not None and truth.sha256 is not None
            if not truth.path.lower().endswith(".npz"):
                raise ValueError(
                    "lossless_npz_uint8 truth path must end in .npz."
                )
            if (
                truth.protocol_path is None
                or truth.protocol_sha256 is None
            ):
                raise RuntimeError(
                    "validated classified truth is missing its protocol "
                    "asset."
                )
            if (
                truth.protocol_path in {truth.path, self.media.path}
                or truth.protocol_sha256.lower()
                in {truth.sha256.lower(), self.media.sha256.lower()}
            ):
                raise ValueError(
                    "truth protocol must be a separate hashed metadata "
                    "asset."
                )
        if self.sensor_input.kind == "legacy_unclassified":
            raise ValueError(
                "classified manifests require classified sensor input."
            )
        if (
            self.manifest_schema_version == 5
            and self.sensor_input.kind != "raw_recorded"
        ):
            raise ValueError(
                "schema-v5 currently supports raw_recorded input only; "
                "qualification-v2 calibration-source roles are not yet "
                "modeled."
            )
        if self.media_window is not None:
            window_schema = f"schema-v{self.manifest_schema_version}"
            if self.sensor_input.kind != "raw_recorded":
                raise ValueError(
                    f"{window_schema} media windows currently require "
                    "raw_recorded sensor input."
                )
            if self.normalization.mode != "uint8_identity":
                raise ValueError(
                    f"{window_schema} media windows currently require "
                    "uint8_identity normalization."
                )
            if truth.evidence_grade != "none":
                raise ValueError(
                    f"{window_schema} media windows currently require "
                    "truth-free "
                    "evaluation."
                )
            if self.wind.mode != "none":
                raise ValueError(
                    f"{window_schema} media windows currently require "
                    "wind mode "
                    "none because parent-time wind alignment is unsupported."
                )
            assert self.media_window is not None
            if (
                self.media_window.parent_media_sha256.lower()
                != self.media.sha256.lower()
            ):
                raise ValueError(
                    "media_window parent_media_sha256 must match media "
                    "sha256."
                )
            if self.media_window.selection_metadata_path == self.media.path:
                raise ValueError(
                    "media-window selection metadata must be a separate "
                    "asset from media."
                )
            if (
                self.media_window.selection_metadata_sha256.lower()
                == self.media.sha256.lower()
            ):
                raise ValueError(
                    "media-window selection metadata must not reuse the "
                    "parent media bytes."
                )
        if self.sensor_input.kind == "structure_suppressed_emulation":
            if self.normalization.mode != "uint8_identity":
                raise ValueError(
                    "structure-suppressed emulation currently requires "
                    "uint8_identity normalization."
                )
            if self.manifest_schema_version == 2:
                leak_off = self.sensor_input.leak_off_calibration
                if leak_off is None:
                    raise RuntimeError(
                        "schema-v2 emulation requires leak-off media."
                    )
                self._validate_calibration_separation(
                    calibration_path=leak_off.path,
                    calibration_sha256=leak_off.sha256,
                    truth=truth,
                )
            else:
                frozen = self.sensor_input.frozen_calibration
                if frozen is None:
                    raise RuntimeError(
                        "schema-v3 emulation requires frozen calibration."
                    )
                self._validate_calibration_separation(
                    calibration_path=frozen.path,
                    calibration_sha256=frozen.sha256,
                    truth=truth,
                )
                self._validate_calibration_separation(
                    calibration_path=frozen.source_path,
                    calibration_sha256=frozen.source_sha256,
                    truth=truth,
                )
                if frozen.preparation_spec_path in {
                    self.media.path,
                    frozen.path,
                    frozen.source_path,
                    truth.path,
                    truth.protocol_path,
                }:
                    raise ValueError(
                        "preparation spec must be a separate metadata "
                        "asset."
                    )
                if frozen.preparation_profile_path in {
                    self.media.path,
                    frozen.path,
                    frozen.source_path,
                    frozen.preparation_spec_path,
                    truth.path,
                    truth.protocol_path,
                }:
                    raise ValueError(
                        "preparation profile must be a separate metadata "
                        "asset."
                    )
                if frozen.calibration_record_path in {
                    self.media.path,
                    frozen.path,
                    frozen.source_path,
                    frozen.preparation_spec_path,
                    frozen.preparation_profile_path,
                    truth.path,
                    truth.protocol_path,
                }:
                    raise ValueError(
                        "calibration record must be a separate metadata "
                        "asset."
                    )
                forbidden_hashes = {
                    self.media.sha256.lower(),
                    frozen.sha256.lower(),
                    frozen.source_sha256.lower(),
                }
                if truth.sha256 is not None:
                    forbidden_hashes.add(truth.sha256.lower())
                if truth.protocol_sha256 is not None:
                    forbidden_hashes.add(truth.protocol_sha256.lower())
                for name, value in (
                    (
                        "preparation spec",
                        frozen.preparation_spec_sha256,
                    ),
                    (
                        "preparation profile",
                        frozen.preparation_profile_sha256,
                    ),
                    (
                        "calibration record",
                        frozen.calibration_record_sha256,
                    ),
                ):
                    if value.lower() in forbidden_hashes:
                        raise ValueError(
                            f"{name} must have separate bytes."
                        )
                    forbidden_hashes.add(value.lower())

    def _validate_calibration_separation(
        self,
        *,
        calibration_path: str,
        calibration_sha256: str,
        truth: TruthSpec,
    ) -> None:
        if calibration_path == self.media.path:
            raise ValueError(
                "calibration must be a separate asset from evaluation media."
            )
        if calibration_sha256.lower() == self.media.sha256.lower():
            raise ValueError(
                "calibration must not reuse evaluation media bytes."
            )
        if (
            truth.path is not None
            and calibration_path == truth.path
        ) or (
            truth.sha256 is not None
            and calibration_sha256.lower() == truth.sha256.lower()
        ):
            raise ValueError(
                "calibration must not reuse evaluation truth."
            )
        if (
            truth.protocol_path is not None
            and calibration_path == truth.protocol_path
        ) or (
            truth.protocol_sha256 is not None
            and calibration_sha256.lower()
            == truth.protocol_sha256.lower()
        ):
            raise ValueError(
                "calibration must not reuse the truth protocol."
            )

    def to_payload(self) -> dict[str, object]:
        scanner_payload = asdict(self.scanner)
        truth_payload = asdict(self.evaluation_truth)
        if self.manifest_schema_version == 1:
            scanner_payload.pop("evidence_kind")
            for key in (
                "storage_format",
                "evidence_grade",
                "method_id",
                "protocol_path",
                "protocol_sha256",
            ):
                truth_payload.pop(key)
        else:
            truth_payload = _without_none(truth_payload)
            if self.evaluation_truth.evidence_grade == "none":
                truth_payload.pop("storage_format")
        payload: dict[str, object] = {
            "manifest_schema_version": self.manifest_schema_version,
            "sequence_id": self.sequence_id,
            "split": self.split,
            "split_provenance": self.split_provenance,
            "scenario": self.scenario,
            "media": asdict(self.media),
            "frame_clock": _without_none(asdict(self.frame_clock)),
            "normalization": _without_none(asdict(self.normalization)),
            "calibration": asdict(self.calibration),
            "wind": _without_none(asdict(self.wind)),
            "source_assumption": _without_none(
                asdict(self.source_assumption)
            ),
            "scanner": scanner_payload,
            "evaluation_truth": truth_payload,
            "model_profile_id": self.model_profile_id,
        }
        if self.manifest_schema_version == 5:
            if self.gas_identity is None:
                raise RuntimeError(
                    "validated schema-v5 manifest is missing gas_identity."
                )
            payload["gas_identity"] = asdict(self.gas_identity)
            if self.dataset_qualification is None:
                raise RuntimeError(
                    "validated schema-v5 manifest is missing "
                    "dataset_qualification."
                )
            payload["dataset_qualification"] = asdict(
                self.dataset_qualification
            )
        if self.manifest_schema_version == 1:
            return payload

        if (
            self.dataset_lineage is None
            or self.capture_group is None
            or self.sensor_input is None
        ):
            raise RuntimeError(
                "validated classified manifest is missing evidence fields."
            )
        payload["dataset_lineage"] = asdict(self.dataset_lineage)
        payload["capture_group"] = asdict(self.capture_group)
        sensor_payload: dict[str, object] = {
            "kind": self.sensor_input.kind,
            "provenance": self.sensor_input.provenance,
        }
        if self.sensor_input.emulation_profile is not None:
            sensor_payload["emulation_profile"] = asdict(
                self.sensor_input.emulation_profile
            )
        if self.sensor_input.leak_off_calibration is not None:
            sensor_payload["leak_off_calibration"] = asdict(
                self.sensor_input.leak_off_calibration
            )
        if self.sensor_input.frozen_calibration is not None:
            sensor_payload["frozen_calibration"] = asdict(
                self.sensor_input.frozen_calibration
            )
        payload["sensor_input"] = sensor_payload
        if self.media_window is not None:
            payload["media_window"] = asdict(self.media_window)
        return payload


@dataclass(frozen=True)
class SequenceReference:
    """Hashed reference to a sequence manifest inside a frozen suite."""

    path: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        _require_text("sequence manifest path", self.path)
        _require_sha256(
            "sequence manifest sha256",
            self.manifest_sha256,
        )


@dataclass(frozen=True)
class RecordedSuiteManifest:
    """Frozen development or held-out sequence list."""

    suite_id: str
    split: RecordedSplit
    frozen_at_utc: str
    frozen_before_model_commit: str
    freeze_provenance: str
    sequences: tuple[SequenceReference, ...]
    suite_schema_version: int = SUITE_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text("suite_id", self.suite_id)
        _require_text("freeze provenance", self.freeze_provenance)
        _require_utc_timestamp("frozen_at_utc", self.frozen_at_utc)
        _require_git_commit(
            "frozen_before_model_commit",
            self.frozen_before_model_commit,
        )
        if self.split not in {"development", "held-out"}:
            raise ValueError("split must be development or held-out.")
        if not self.sequences:
            raise ValueError("recorded suite must include a sequence.")
        if self.suite_schema_version != SUITE_MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported suite manifest schema version.")


@dataclass(frozen=True)
class MotionLookup:
    """One causal motion lookup and its measurement age."""

    status: MotionLookupStatus
    motion: MotionEstimate | None
    age_seconds: float | None

    def __post_init__(self) -> None:
        if self.status == "available":
            if self.motion is None or self.age_seconds is None:
                raise ValueError(
                    "available motion lookup requires motion and age."
                )
            _require_finite(
                "motion age_seconds",
                self.age_seconds,
                minimum=0.0,
            )
        elif self.status == "stale":
            if self.motion is not None or self.age_seconds is None:
                raise ValueError(
                    "stale motion lookup requires age and no motion."
                )
            _require_finite(
                "stale motion age_seconds",
                self.age_seconds,
                minimum=0.0,
            )
        elif self.status == "missing":
            if self.motion is not None or self.age_seconds is not None:
                raise ValueError(
                    "missing motion lookup carries no motion or age."
                )
        else:
            raise ValueError("unknown motion lookup status.")


class CausalMotionProvider(Protocol):
    """Motion input visible only at or before a decision timestamp."""

    method: str

    def lookup(self, decision_time_seconds: float) -> MotionLookup:
        """Return an available, missing, or stale causal motion result."""


@dataclass(frozen=True)
class EvaluationTruth:
    """Offline truth support and normalized probability mass."""

    support: NDArray[np.bool_]
    distribution: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.support.shape != self.distribution.shape:
            raise ValueError(
                "truth support and distribution must have matching shapes."
            )
        if self.support.ndim != 2:
            raise ValueError("truth arrays must be 2-D.")
        if np.any(~np.isfinite(self.distribution)):
            raise ValueError(
                "truth distribution must contain only finite values."
            )
        if np.any(self.distribution < 0.0):
            raise ValueError("truth distribution must be non-negative.")
        total = float(self.distribution.sum())
        if total > 0.0 and not np.isclose(total, 1.0):
            raise ValueError(
                "non-empty truth distribution must sum to 1."
            )


class EvaluationTruthSource(Protocol):
    """Offline evaluation data kept out of policy observations."""

    shape: tuple[int, int]
    end_time_seconds: float
    truth_basis: str

    def truth_at_seconds(
        self,
        time_seconds: float,
    ) -> EvaluationTruth:
        """Return time-aligned truth for offline scoring."""


@dataclass(frozen=True)
class RecordedClaimAssessment:
    """Derived evidence eligibility; never supplied by a manifest."""

    eligibility: ClaimEligibility
    limitations: tuple[str, ...]


def assess_recorded_claim(
    manifest: RecordedSequenceManifest,
    *,
    dataset_rights_blocked: bool = False,
) -> RecordedClaimAssessment:
    """Derive the most restrictive claim label and every active limitation."""

    sensor_kind = (
        manifest.sensor_input.kind
        if manifest.sensor_input is not None
        else "legacy_unclassified"
    )
    scanner_kind = manifest.scanner.evidence_kind
    truth_grade = manifest.evaluation_truth.evidence_grade
    limitations: list[str] = []

    legacy = (
        manifest.manifest_schema_version == 1
        or sensor_kind == "legacy_unclassified"
        or scanner_kind == "legacy_unclassified"
        or truth_grade == "legacy_unclassified"
    )
    if legacy:
        limitations.append(
            "legacy_sequence_manifest_has_unclassified_evidence"
        )
    if truth_grade == "none":
        limitations.append(
            "spatial_evaluation_truth_is_unavailable"
        )
    elif truth_grade == "silver_pseudo":
        limitations.append(
            "pseudo_truth_is_diagnostic_and_not_independently_reviewed"
        )
    elif truth_grade == "gold_human_reviewed":
        limitations.append(
            "human_review_grade_requires_external_protocol_and_freeze_review"
        )
    if scanner_kind == "counterfactual_virtual":
        limitations.append(
            "scanner_cost_and_timing_are_counterfactual_not_measured"
        )
    if sensor_kind == "structure_suppressed_emulation":
        limitations.append(
            "sensor_emulation_has_not_been_validated_against_hardware"
        )
    if (
        manifest.dataset_qualification is not None
        and manifest.dataset_qualification.purpose
        == "diagnostic_raw_pilot"
    ):
        limitations.append(
            "diagnostic_raw_pilot_is_engineering_only"
        )
    if (
        manifest.gas_identity is not None
        and manifest.gas_identity.identity != "methane"
    ):
        limitations.append(
            "non_methane_recording_is_not_methane_validation"
        )
    if dataset_rights_blocked:
        limitations.append("dataset_rights_are_not_verified")
    limitations.append(
        "frozen_suite_and_independence_review_are_still_required"
    )

    if legacy:
        eligibility: ClaimEligibility = "legacy_unclassified_evidence"
    elif dataset_rights_blocked:
        eligibility = "blocked_unverified_data_rights"
    elif truth_grade == "none":
        eligibility = "no_spatial_evaluation_truth"
    elif truth_grade == "silver_pseudo":
        eligibility = "diagnostic_only_pseudo_truth"
    elif scanner_kind == "counterfactual_virtual":
        eligibility = "offline_counterfactual_scan_evidence"
    elif sensor_kind == "structure_suppressed_emulation":
        eligibility = "sensor_emulation_requires_hardware_validation"
    else:
        eligibility = (
            "requires_frozen_suite_and_independence_review"
        )
    return RecordedClaimAssessment(
        eligibility=eligibility,
        limitations=tuple(limitations),
    )


@dataclass(frozen=True)
class RecordedSequenceProvenance:
    """Resolved manifest and asset provenance serialized into reports."""

    sequence_id: str
    split: RecordedSplit
    manifest_schema_version: int
    manifest_sha256: str
    media_sha256: str
    frame_count: int
    frame_clock_mode: ClockMode
    timestamp_sha256: str | None
    wind_mode: WindMode
    wind_sha256: str | None
    truth_kind: TruthKind | None
    truth_sha256: str | None
    normalization: dict[str, object]
    calibration: dict[str, object]
    source_assumption: dict[str, object]
    evidence_contract: dict[str, object]
    claim_eligibility: ClaimEligibility
    claim_limitations: tuple[str, ...]
    manifest: dict[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "source_kind": "recorded_sequence",
            "evidence_grade": "provenance_checked_recorded_input",
            **asdict(self),
            "claim_limitations": list(self.claim_limitations),
        }


@dataclass(frozen=True)
class RecordedSequence:
    """Runtime-only recorded input consumed by the headless experiment."""

    manifest: RecordedSequenceManifest
    timeline: TimelineSource
    motion_provider: CausalMotionProvider
    truth_source: EvaluationTruthSource | None
    initial_roi: BoundingBox
    source_point: tuple[float, float] | None
    model_profile: ModelProfile
    provenance: RecordedSequenceProvenance

    def __post_init__(self) -> None:
        timeline_end = self.timeline.end_time_seconds
        if timeline_end is None:
            raise ValueError(
                "recorded sensor timelines must have a finite end."
            )
        if self.truth_source is not None:
            if self.timeline.shape != self.truth_source.shape:
                raise ValueError(
                    "recorded sensor and truth shapes must match."
                )
            if (
                abs(
                    timeline_end
                    - self.truth_source.end_time_seconds
                )
                > 1e-9
            ):
                raise ValueError(
                    "recorded sensor and truth timelines must have equal "
                    "ends."
                )
        clipped = self.initial_roi.clipped(self.timeline.shape)
        if clipped != self.initial_roi:
            raise ValueError("initial_roi must lie inside the frame.")
        if self.source_point is not None:
            row, col = self.source_point
            if not (
                0.0 <= row < self.timeline.shape[0]
                and 0.0 <= col < self.timeline.shape[1]
            ):
                raise ValueError("source_point must lie inside the frame.")
        scanner = self.manifest.scanner
        if scanner.minimum_sweep_rows > self.timeline.shape[0]:
            raise ValueError(
                "minimum_sweep_rows exceeds the recorded frame height."
            )
        if scanner.minimum_sweep_columns > self.timeline.shape[1]:
            raise ValueError(
                "minimum_sweep_columns exceeds the recorded frame width."
            )
        if (
            scanner.maximum_sweep_columns is not None
            and scanner.maximum_sweep_columns > self.timeline.shape[1]
        ):
            raise ValueError(
                "maximum_sweep_columns exceeds the recorded frame width."
            )
        if self.model_profile.scenario != self.manifest.scenario:
            raise ValueError(
                "recorded model profile scenario must match the manifest."
            )

    @property
    def source_free(self) -> bool:
        return self.manifest.source_assumption.mode == "source_free"


def _validate_point(point: tuple[float, float]) -> None:
    if len(point) != 2 or not all(math.isfinite(value) for value in point):
        raise ValueError("source_point must contain two finite coordinates.")


def _validate_roi(roi: tuple[int, int, int, int]) -> None:
    if len(roi) != 4:
        raise ValueError("initial_roi must contain four coordinates.")
    min_row, min_col, max_row, max_col = roi
    if min_row < 0 or min_col < 0:
        raise ValueError("initial_roi minimum coordinates must be non-negative.")
    if max_row < min_row or max_col < min_col:
        raise ValueError("initial_roi maximum must follow its minimum.")


def _without_none(payload: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in payload.items()
        if value is not None
    }


def _require_sha256(name: str, value: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{name} must contain 64 hexadecimal characters.")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(
            f"{name} must contain 64 hexadecimal characters."
        ) from error


def _require_git_commit(name: str, value: str) -> None:
    if not 7 <= len(value) <= 40:
        raise ValueError(
            f"{name} must contain 7 to 40 hexadecimal characters."
        )
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(
            f"{name} must contain 7 to 40 hexadecimal characters."
        ) from error


def _require_full_git_commit(name: str, value: str) -> None:
    if len(value) != 40:
        raise ValueError(
            f"{name} must contain the full 40-character commit."
        )
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(
            f"{name} must contain the full 40-character commit."
        ) from error


def _require_utc_timestamp(name: str, value: str) -> None:
    _require_text(name, value)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(
            f"{name} must be an ISO-8601 UTC timestamp."
        ) from error
    if parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must use the UTC timezone.")
