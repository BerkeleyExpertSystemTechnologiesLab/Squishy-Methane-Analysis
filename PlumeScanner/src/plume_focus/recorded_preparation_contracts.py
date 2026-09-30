"""Strict contracts for dataset-neutral recorded-data preparation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import re
from typing import Literal

from plume_focus.recorded_contracts import (
    CalibrationScope,
    FrozenCalibrationAssetSpec,
    RecordedSequenceManifest,
    SensorInputKind,
)
from plume_focus.recorded_manifest_io import (
    canonical_json_sha256,
    load_json_object,
    parse_recorded_sequence_manifest,
)
from plume_focus.sensor_emulation import SensorEmulationProfile


CaptureRole = Literal[
    "development",
    "held-out",
    "independent_calibration",
    "excluded",
]
ExclusionReason = Literal[
    "rights_restriction",
    "missing_required_timing_metadata",
    "missing_required_source_metadata",
    "out_of_scope_sensor_or_view",
    "duplicate_byte_asset",
    "predeclared_protocol_exclusion",
]

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_EXCLUSION_REASONS = {
    "rights_restriction",
    "missing_required_timing_metadata",
    "missing_required_source_metadata",
    "out_of_scope_sensor_or_view",
    "duplicate_byte_asset",
    "predeclared_protocol_exclusion",
}
FIXED_CALIBRATION_ALGORITHM_ID = (
    "per_pixel_median_normal_consistent_mad_v1"
)
FIXED_CALIBRATION_ARTIFACT_FORMAT = "plume_fixed_calibration_v1"


@dataclass(frozen=True)
class DatasetAuditReference:
    dataset_id: str
    accession_id: str
    path: str
    sha256: str

    def __post_init__(self) -> None:
        _require_text("dataset_id", self.dataset_id)
        _require_text("accession_id", self.accession_id)
        _require_relative_path("dataset audit path", self.path)
        _require_sha256("dataset audit sha256", self.sha256)

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetQualificationReference:
    """Immutable reference to the fine-grained dataset qualification."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        _require_relative_path("dataset qualification path", self.path)
        _require_sha256("dataset qualification sha256", self.sha256)

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CaptureGroupDeclaration:
    capture_group_id: str
    role: CaptureRole
    provenance: str
    asset_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier(
            "capture_group_id",
            self.capture_group_id,
        )
        if self.role not in {
            "development",
            "held-out",
            "independent_calibration",
            "excluded",
        }:
            raise ValueError("unknown capture-group role.")
        _require_text("capture-group provenance", self.provenance)
        if not self.asset_sha256s:
            raise ValueError(
                "capture group must contain at least one audited asset."
            )
        normalized = tuple(value.lower() for value in self.asset_sha256s)
        for value in normalized:
            _require_sha256("capture-group asset sha256", value)
        if len(set(normalized)) != len(normalized):
            raise ValueError(
                "capture-group asset hashes must be unique."
            )

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["asset_sha256s"] = list(self.asset_sha256s)
        return payload


@dataclass(frozen=True)
class ExclusionRecord:
    capture_group_id: str
    reason_code: ExclusionReason
    rationale: str

    def __post_init__(self) -> None:
        _require_identifier(
            "excluded capture_group_id",
            self.capture_group_id,
        )
        if self.reason_code not in _EXCLUSION_REASONS:
            raise ValueError(
                "unknown or performance-based exclusion reason."
            )
        _require_text("exclusion rationale", self.rationale)

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationSourceSpec:
    path: str
    sha256: str
    provenance: str
    calibration_scope: CalibrationScope
    capture_group_id: str

    def __post_init__(self) -> None:
        _require_relative_path("calibration source path", self.path)
        _require_sha256("calibration source sha256", self.sha256)
        _require_text(
            "calibration source provenance",
            self.provenance,
        )
        _require_identifier(
            "calibration capture_group_id",
            self.capture_group_id,
        )
        if self.calibration_scope not in {
            "development_only",
            "independent_pre_run",
        }:
            raise ValueError("unknown calibration_scope.")

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MediaWindowRequest:
    """Preregistered source-time interval resolved during preparation."""

    requested_start_time_seconds: float
    requested_end_time_seconds: float
    expected_source_start_frame_index: int
    expected_source_end_frame_index_exclusive: int
    selection_metadata_path: str
    selection_metadata_sha256: str
    selection_provenance: str

    def __post_init__(self) -> None:
        for name, value in (
            (
                "requested_start_time_seconds",
                self.requested_start_time_seconds,
            ),
            (
                "requested_end_time_seconds",
                self.requested_end_time_seconds,
            ),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if (
            self.requested_start_time_seconds
            >= self.requested_end_time_seconds
        ):
            raise ValueError(
                "requested media-window start must precede its end."
            )
        for name, value in (
            (
                "expected_source_start_frame_index",
                self.expected_source_start_frame_index,
            ),
            (
                "expected_source_end_frame_index_exclusive",
                self.expected_source_end_frame_index_exclusive,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"{name} must be a non-negative integer."
                )
        if (
            self.expected_source_end_frame_index_exclusive
            - self.expected_source_start_frame_index
            < 3
        ):
            raise ValueError(
                "expected media window must contain at least three frames."
            )
        _require_relative_path(
            "media-window selection metadata path",
            self.selection_metadata_path,
        )
        _require_sha256(
            "media-window selection metadata sha256",
            self.selection_metadata_sha256,
        )
        _require_text(
            "media-window selection provenance",
            self.selection_provenance,
        )

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SequencePreparationSpec:
    output_sequence_id: str
    manifest_template: RecordedSequenceManifest
    calibration_source: CalibrationSourceSpec | None
    media_window_request: MediaWindowRequest | None = None

    def __post_init__(self) -> None:
        _require_identifier(
            "output_sequence_id",
            self.output_sequence_id,
        )
        if self.manifest_template.manifest_schema_version not in {3, 5}:
            raise ValueError(
                "preparation requires schema-v3 or schema-v5 manifest "
                "templates."
            )
        sensor_input = self.manifest_template.sensor_input
        if sensor_input is None or sensor_input.kind != "raw_recorded":
            raise ValueError(
                "manifest templates must declare raw_recorded input; "
                "the preparation profile selects the output track."
            )
        if self.manifest_template.evaluation_truth.evidence_grade != "none":
            raise ValueError(
                "preparation templates must be truth-free; evaluation "
                "truth is attached by a separate evidence workflow."
            )
        if (
            self.media_window_request is not None
            and self.calibration_source is not None
        ):
            raise ValueError(
                "windowed preparation does not support sensor emulation."
            )

    def to_payload(
        self,
        *,
        include_media_window_request: bool | None = None,
    ) -> dict[str, object]:
        if include_media_window_request is None:
            include_media_window_request = (
                self.media_window_request is not None
            )
        payload: dict[str, object] = {
            "output_sequence_id": self.output_sequence_id,
            "manifest_template": self.manifest_template.to_payload(),
            "calibration_source": (
                self.calibration_source.to_payload()
                if self.calibration_source is not None
                else None
            ),
        }
        if include_media_window_request:
            payload["media_window_request"] = (
                self.media_window_request.to_payload()
                if self.media_window_request is not None
                else None
            )
        return payload


@dataclass(frozen=True)
class PreparationSpec:
    preparation_id: str
    producing_commit: str
    dataset_audit: DatasetAuditReference
    capture_groups: tuple[CaptureGroupDeclaration, ...]
    exclusions: tuple[ExclusionRecord, ...]
    sequences: tuple[SequencePreparationSpec, ...]
    preparation_spec_schema_version: int = 1
    dataset_qualification: DatasetQualificationReference | None = None

    def __post_init__(self) -> None:
        if self.preparation_spec_schema_version not in {1, 2, 3, 4}:
            raise ValueError(
                "unsupported preparation spec schema version."
            )
        _require_identifier("preparation_id", self.preparation_id)
        _require_full_git_commit(
            "producing_commit",
            self.producing_commit,
        )
        if not self.capture_groups:
            raise ValueError(
                "preparation spec requires capture groups."
            )
        if not self.sequences:
            raise ValueError(
                "preparation spec requires at least one sequence."
            )
        has_windows = [
            sequence.media_window_request is not None
            for sequence in self.sequences
        ]
        if self.preparation_spec_schema_version == 1 and any(has_windows):
            raise ValueError(
                "preparation spec v1 does not support media windows."
            )
        if self.preparation_spec_schema_version == 2 and not all(
            has_windows
        ):
            raise ValueError(
                "preparation spec v2 requires one media-window request "
                "per sequence."
            )
        if self.preparation_spec_schema_version == 3 and not all(
            has_windows
        ):
            raise ValueError(
                "preparation spec v3 requires one media-window request "
                "per sequence."
            )
        # Schema v4 is the qualification-bound, operation-aware form.  It
        # deliberately permits either whole assets or declared windows; the
        # referenced qualification/preregistration decides which operation is
        # authorized and preparation rechecks that exact selection.
        if self.preparation_spec_schema_version < 3:
            if self.dataset_qualification is not None:
                raise ValueError(
                    "preparation spec v1/v2 does not support a dataset "
                    "qualification reference."
                )
        elif self.dataset_qualification is None:
            raise ValueError(
                "preparation spec v3/v4 requires a dataset qualification "
                "reference."
            )

        groups = {
            group.capture_group_id: group
            for group in self.capture_groups
        }
        if len(groups) != len(self.capture_groups):
            raise ValueError("capture_group_id values must be unique.")
        all_assets: list[str] = [
            value.lower()
            for group in self.capture_groups
            for value in group.asset_sha256s
        ]
        if len(set(all_assets)) != len(all_assets):
            raise ValueError(
                "one audited asset may belong to only one capture group."
            )

        exclusions = {
            item.capture_group_id: item
            for item in self.exclusions
        }
        if len(exclusions) != len(self.exclusions):
            raise ValueError(
                "each excluded capture group takes one exclusion record."
            )
        excluded_groups = {
            group.capture_group_id
            for group in self.capture_groups
            if group.role == "excluded"
        }
        if set(exclusions) != excluded_groups:
            raise ValueError(
                "exclusion records must exactly cover excluded groups."
            )

        output_ids = [
            sequence.output_sequence_id
            for sequence in self.sequences
        ]
        if len(set(output_ids)) != len(output_ids):
            raise ValueError("output_sequence_id values must be unique.")
        for sequence in self.sequences:
            self._validate_sequence(sequence, groups)
        used_assets = {
            sequence.manifest_template.media.sha256.lower()
            for sequence in self.sequences
        }
        used_assets.update(
            sequence.calibration_source.sha256.lower()
            for sequence in self.sequences
            if sequence.calibration_source is not None
        )
        active_assets = {
            value.lower()
            for group in self.capture_groups
            if group.role != "excluded"
            for value in group.asset_sha256s
        }
        unused_active_assets = sorted(active_assets - used_assets)
        if (
            self.preparation_spec_schema_version < 3
            and unused_active_assets
        ):
            raise ValueError(
                "every non-excluded asset must be used as sequence media "
                "or a calibration source; otherwise declare an objective "
                f"exclusion: {unused_active_assets}"
            )

    def _validate_sequence(
        self,
        sequence: SequencePreparationSpec,
        groups: dict[str, CaptureGroupDeclaration],
    ) -> None:
        manifest = sequence.manifest_template
        lineage = manifest.dataset_lineage
        capture = manifest.capture_group
        if lineage is None or capture is None:
            raise RuntimeError(
                "validated schema-v3 template lacks lineage."
            )
        if (
            lineage.dataset_id != self.dataset_audit.dataset_id
            or lineage.accession_id
            != self.dataset_audit.accession_id
            or lineage.audit_path != self.dataset_audit.path
            or lineage.audit_sha256.lower()
            != self.dataset_audit.sha256.lower()
        ):
            raise ValueError(
                "manifest template dataset lineage must match the "
                "preparation audit reference."
            )
        group = groups.get(capture.capture_group_id)
        if group is None:
            raise ValueError(
                "manifest capture group is absent from the registry."
            )
        if capture.grouping_provenance != group.provenance:
            raise ValueError(
                "manifest capture-group provenance must match the registry."
            )
        expected_role = (
            "development"
            if manifest.split == "development"
            else "held-out"
        )
        if group.role != expected_role:
            raise ValueError(
                "manifest split must match its capture-group role."
            )
        if manifest.media.sha256.lower() not in {
            value.lower() for value in group.asset_sha256s
        }:
            raise ValueError(
                "manifest media must belong to its capture group."
            )

        calibration = sequence.calibration_source
        if calibration is None:
            return
        calibration_group = groups.get(
            calibration.capture_group_id
        )
        if calibration_group is None:
            raise ValueError(
                "calibration capture group is absent from the registry."
            )
        if calibration.sha256.lower() not in {
            value.lower()
            for value in calibration_group.asset_sha256s
        }:
            raise ValueError(
                "calibration source must belong to its capture group."
            )
        expected_calibration_role = (
            "development"
            if calibration.calibration_scope == "development_only"
            else "independent_calibration"
        )
        if calibration_group.role != expected_calibration_role:
            raise ValueError(
                "calibration scope does not match its capture-group role."
            )
        if (
            calibration.sha256.lower()
            == manifest.media.sha256.lower()
        ):
            raise ValueError(
                "calibration source must not reuse evaluation media."
            )

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "preparation_spec_schema_version": (
                self.preparation_spec_schema_version
            ),
            "preparation_id": self.preparation_id,
            "producing_commit": self.producing_commit,
            "dataset_audit": self.dataset_audit.to_payload(),
            "capture_groups": [
                item.to_payload() for item in self.capture_groups
            ],
            "exclusions": [
                item.to_payload() for item in self.exclusions
            ],
            "sequences": [
                item.to_payload(
                    include_media_window_request=(
                        self.preparation_spec_schema_version >= 2
                    )
                )
                for item in self.sequences
            ],
        }
        if self.preparation_spec_schema_version >= 3:
            assert self.dataset_qualification is not None
            payload["dataset_qualification"] = (
                self.dataset_qualification.to_payload()
            )
        return payload


@dataclass(frozen=True)
class PreparationProfile:
    profile_id: str
    sensor_input_kind: SensorInputKind
    sensor_input_provenance: str
    emulation_profile: SensorEmulationProfile | None
    preparation_profile_schema_version: int = 1

    def __post_init__(self) -> None:
        if self.preparation_profile_schema_version != 1:
            raise ValueError(
                "unsupported preparation profile schema version."
            )
        _require_identifier("profile_id", self.profile_id)
        _require_text(
            "sensor_input_provenance",
            self.sensor_input_provenance,
        )
        if self.sensor_input_kind == "raw_recorded":
            if self.emulation_profile is not None:
                raise ValueError(
                    "raw preparation profile takes no emulation profile."
                )
            return
        if self.sensor_input_kind != (
            "structure_suppressed_emulation"
        ):
            raise ValueError(
                "preparation profile requires a classified sensor kind."
            )
        if self.emulation_profile is None:
            raise ValueError(
                "emulated preparation profile requires settings."
            )
        if self.profile_id != self.emulation_profile.profile_id:
            raise ValueError(
                "preparation and emulation profile IDs must match."
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "preparation_profile_schema_version": (
                self.preparation_profile_schema_version
            ),
            "profile_id": self.profile_id,
            "sensor_input_kind": self.sensor_input_kind,
            "sensor_input_provenance": self.sensor_input_provenance,
            "emulation_profile": (
                asdict(self.emulation_profile)
                if self.emulation_profile is not None
                else None
            ),
        }


@dataclass(frozen=True)
class HashedPreparationInput:
    """One immutable metadata input to a preparation result."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        _require_relative_path("preparation input path", self.path)
        _require_sha256("preparation input sha256", self.sha256)


@dataclass(frozen=True)
class FrozenCalibrationResult:
    """Resolved canonical calibration artifact described by a result record."""

    path: str
    sha256: str
    rows: int
    columns: int
    sample_count: int
    format_id: str = FIXED_CALIBRATION_ARTIFACT_FORMAT

    def __post_init__(self) -> None:
        _require_relative_path("calibration artifact path", self.path)
        _require_sha256("calibration artifact sha256", self.sha256)
        for name, value in (
            ("calibration rows", self.rows),
            ("calibration columns", self.columns),
            ("calibration sample_count", self.sample_count),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer.")
        if self.sample_count < 2:
            raise ValueError(
                "calibration sample_count must be at least 2."
            )
        if self.format_id != FIXED_CALIBRATION_ARTIFACT_FORMAT:
            raise ValueError("unsupported calibration artifact format.")

    def to_payload(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "format": self.format_id,
            "shape": [self.rows, self.columns],
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True)
class CalibrationPreparationRecord:
    """Strict, non-evaluative lineage for one frozen calibration artifact."""

    preparation_id: str
    producing_commit: str
    source: CalibrationSourceSpec
    calibration_artifact: FrozenCalibrationResult
    emulation_profile_sha256: str
    preparation_spec: HashedPreparationInput
    preparation_profile: HashedPreparationInput
    fit_algorithm_id: str = FIXED_CALIBRATION_ALGORITHM_ID
    calibration_record_schema_version: int = 1
    artifact_kind: str = "fixed_calibration_preparation"
    claim_eligibility: str = "not_sequence_evidence"

    def __post_init__(self) -> None:
        if self.calibration_record_schema_version != 1:
            raise ValueError(
                "unsupported calibration record schema version."
            )
        if self.artifact_kind != "fixed_calibration_preparation":
            raise ValueError("unknown calibration record artifact kind.")
        if self.claim_eligibility != "not_sequence_evidence":
            raise ValueError(
                "calibration records cannot carry evaluation claims."
            )
        _require_identifier("preparation_id", self.preparation_id)
        _require_full_git_commit(
            "calibration producing_commit",
            self.producing_commit,
        )
        if self.fit_algorithm_id != FIXED_CALIBRATION_ALGORITHM_ID:
            raise ValueError("unsupported calibration fit algorithm.")
        _require_sha256(
            "emulation profile sha256",
            self.emulation_profile_sha256,
        )
        paths = {
            self.source.path,
            self.calibration_artifact.path,
            self.preparation_spec.path,
            self.preparation_profile.path,
        }
        if len(paths) != 4:
            raise ValueError(
                "calibration record inputs and output must be separate "
                "assets."
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "calibration_record_schema_version": (
                self.calibration_record_schema_version
            ),
            "artifact_kind": self.artifact_kind,
            "claim_eligibility": self.claim_eligibility,
            "preparation_id": self.preparation_id,
            "producing_commit": self.producing_commit,
            "fit_algorithm_id": self.fit_algorithm_id,
            "source": asdict(self.source),
            "calibration_artifact": (
                self.calibration_artifact.to_payload()
            ),
            "emulation_profile_sha256": (
                self.emulation_profile_sha256
            ),
            "preparation_spec": asdict(self.preparation_spec),
            "preparation_profile": asdict(
                self.preparation_profile
            ),
        }


@dataclass(frozen=True)
class PreparedSequenceResult:
    """One emitted manifest recorded in a preparation report."""

    sequence_id: str
    path: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        _require_identifier("sequence_id", self.sequence_id)
        _require_relative_path("prepared manifest path", self.path)
        _require_sha256(
            "prepared manifest sha256",
            self.manifest_sha256,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "sequence_id": self.sequence_id,
            "path": self.path,
            "manifest_sha256": self.manifest_sha256.lower(),
        }


@dataclass(frozen=True)
class PreparationReport:
    """Strict non-evaluative summary of deterministic preparation outputs."""

    preparation_id: str
    preparation_spec: HashedPreparationInput
    preparation_profile: HashedPreparationInput
    producing_commit: str
    sequences: tuple[PreparedSequenceResult, ...]
    report_schema_version: int = 1
    artifact_kind: str = "recorded_preparation_report"
    claim_eligibility: str = "not_sequence_evidence"

    def __post_init__(self) -> None:
        if self.report_schema_version != 1:
            raise ValueError(
                "unsupported recorded preparation report version."
            )
        if self.artifact_kind != "recorded_preparation_report":
            raise ValueError("unknown preparation report artifact kind.")
        if self.claim_eligibility != "not_sequence_evidence":
            raise ValueError(
                "preparation report claim_eligibility must remain "
                "not_sequence_evidence."
            )
        _require_identifier("preparation_id", self.preparation_id)
        _require_full_git_commit(
            "preparation report producing_commit",
            self.producing_commit,
        )
        if not self.sequences:
            raise ValueError(
                "preparation report requires emitted sequences."
            )
        sequence_ids = [item.sequence_id for item in self.sequences]
        paths = [item.path for item in self.sequences]
        manifest_hashes = [
            item.manifest_sha256.lower() for item in self.sequences
        ]
        if len(set(sequence_ids)) != len(sequence_ids):
            raise ValueError(
                "preparation report sequence_id values must be unique."
            )
        if len(set(paths)) != len(paths):
            raise ValueError(
                "preparation report manifest paths must be unique."
            )
        if len(set(manifest_hashes)) != len(manifest_hashes):
            raise ValueError(
                "preparation report manifest hashes must be unique."
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "report_schema_version": self.report_schema_version,
            "artifact_kind": self.artifact_kind,
            "claim_eligibility": self.claim_eligibility,
            "preparation_id": self.preparation_id,
            "preparation_spec": asdict(self.preparation_spec),
            "preparation_profile": asdict(
                self.preparation_profile
            ),
            "producing_commit": self.producing_commit.lower(),
            "sequences": [
                item.to_payload() for item in self.sequences
            ],
        }


def load_preparation_spec(path: Path) -> PreparationSpec:
    return parse_preparation_spec(load_json_object(Path(path)))


def load_preparation_profile(path: Path) -> PreparationProfile:
    return parse_preparation_profile(load_json_object(Path(path)))


def load_calibration_preparation_record(
    path: Path,
) -> CalibrationPreparationRecord:
    return parse_calibration_preparation_record(
        load_json_object(Path(path))
    )


def load_preparation_report(path: Path) -> PreparationReport:
    return parse_preparation_report(load_json_object(Path(path)))


def parse_preparation_spec(
    payload: dict[str, object],
) -> PreparationSpec:
    base_required = {
        "preparation_spec_schema_version",
        "preparation_id",
        "producing_commit",
        "dataset_audit",
        "capture_groups",
        "exclusions",
        "sequences",
    }
    if "preparation_spec_schema_version" not in payload:
        # Retain the legacy exact-key failure for malformed v1/v2 payloads.
        _check_keys(
            payload,
            required=base_required,
            context="preparation spec",
        )
        raise RuntimeError("unreachable preparation schema validation")
    schema_version = _integer(
        "preparation_spec_schema_version",
        payload["preparation_spec_schema_version"],
    )
    required = set(base_required)
    if schema_version >= 3:
        required.add("dataset_qualification")
    _check_keys(
        payload,
        required=required,
        context="preparation spec",
    )
    audit_payload = _object("dataset_audit", payload["dataset_audit"])
    _check_keys(
        audit_payload,
        required={"dataset_id", "accession_id", "path", "sha256"},
        context="preparation dataset audit",
    )
    groups = tuple(
        _parse_capture_group(item)
        for item in _list("capture_groups", payload["capture_groups"])
    )
    exclusions = tuple(
        _parse_exclusion(item)
        for item in _list("exclusions", payload["exclusions"])
    )
    sequences = tuple(
        _parse_sequence(item, schema_version=schema_version)
        for item in _list("sequences", payload["sequences"])
    )
    qualification: DatasetQualificationReference | None = None
    if schema_version >= 3:
        qualification_payload = _object(
            "dataset_qualification",
            payload["dataset_qualification"],
        )
        _check_keys(
            qualification_payload,
            required={"path", "sha256"},
            context="preparation dataset qualification",
        )
        qualification = DatasetQualificationReference(
            path=_string(
                "dataset qualification path",
                qualification_payload["path"],
            ),
            sha256=_string(
                "dataset qualification sha256",
                qualification_payload["sha256"],
            ),
        )
    return PreparationSpec(
        preparation_spec_schema_version=schema_version,
        preparation_id=_string(
            "preparation_id",
            payload["preparation_id"],
        ),
        producing_commit=_string(
            "producing_commit",
            payload["producing_commit"],
        ),
        dataset_audit=DatasetAuditReference(
            dataset_id=_string(
                "dataset_id",
                audit_payload["dataset_id"],
            ),
            accession_id=_string(
                "accession_id",
                audit_payload["accession_id"],
            ),
            path=_string("audit path", audit_payload["path"]),
            sha256=_string("audit sha256", audit_payload["sha256"]),
        ),
        capture_groups=groups,
        exclusions=exclusions,
        sequences=sequences,
        dataset_qualification=qualification,
    )


def parse_preparation_profile(
    payload: dict[str, object],
) -> PreparationProfile:
    _check_keys(
        payload,
        required={
            "preparation_profile_schema_version",
            "profile_id",
            "sensor_input_kind",
            "sensor_input_provenance",
            "emulation_profile",
        },
        context="preparation profile",
    )
    profile_value = payload["emulation_profile"]
    emulation_profile: SensorEmulationProfile | None = None
    if profile_value is not None:
        values = _object("emulation_profile", profile_value)
        _check_keys(
            values,
            required={
                "profile_id",
                "residual_mode",
                "provenance",
                "noise_floor_raw_units",
                "output_units_per_noise_sigma",
                "smoothing_time_constant_seconds",
                "profile_schema_version",
            },
            context="preparation emulation profile",
        )
        smoothing_value = values[
            "smoothing_time_constant_seconds"
        ]
        emulation_profile = SensorEmulationProfile(
            profile_id=_string("profile_id", values["profile_id"]),
            residual_mode=_string(  # type: ignore[arg-type]
                "residual_mode",
                values["residual_mode"],
            ),
            provenance=_string("provenance", values["provenance"]),
            noise_floor_raw_units=_number(
                "noise_floor_raw_units",
                values["noise_floor_raw_units"],
            ),
            output_units_per_noise_sigma=_number(
                "output_units_per_noise_sigma",
                values["output_units_per_noise_sigma"],
            ),
            smoothing_time_constant_seconds=(
                None
                if smoothing_value is None
                else _number(
                    "smoothing_time_constant_seconds",
                    smoothing_value,
                )
            ),
            profile_schema_version=_integer(
                "profile_schema_version",
                values["profile_schema_version"],
            ),
        )
    return PreparationProfile(
        preparation_profile_schema_version=_integer(
            "preparation_profile_schema_version",
            payload["preparation_profile_schema_version"],
        ),
        profile_id=_string("profile_id", payload["profile_id"]),
        sensor_input_kind=_string(  # type: ignore[arg-type]
            "sensor_input_kind",
            payload["sensor_input_kind"],
        ),
        sensor_input_provenance=_string(
            "sensor_input_provenance",
            payload["sensor_input_provenance"],
        ),
        emulation_profile=emulation_profile,
    )


def parse_calibration_preparation_record(
    payload: dict[str, object],
) -> CalibrationPreparationRecord:
    _check_keys(
        payload,
        required={
            "calibration_record_schema_version",
            "artifact_kind",
            "claim_eligibility",
            "preparation_id",
            "producing_commit",
            "fit_algorithm_id",
            "source",
            "calibration_artifact",
            "emulation_profile_sha256",
            "preparation_spec",
            "preparation_profile",
        },
        context="calibration preparation record",
    )
    source_payload = _object("source", payload["source"])
    _check_keys(
        source_payload,
        required={
            "path",
            "sha256",
            "provenance",
            "calibration_scope",
            "capture_group_id",
        },
        context="calibration record source",
    )
    artifact_payload = _object(
        "calibration_artifact",
        payload["calibration_artifact"],
    )
    _check_keys(
        artifact_payload,
        required={"path", "sha256", "format", "shape", "sample_count"},
        context="calibration record artifact",
    )
    shape = _list("calibration artifact shape", artifact_payload["shape"])
    if len(shape) != 2:
        raise ValueError(
            "calibration artifact shape must contain rows and columns."
        )
    spec_payload = _object(
        "preparation_spec",
        payload["preparation_spec"],
    )
    profile_payload = _object(
        "preparation_profile",
        payload["preparation_profile"],
    )
    for name, value in (
        ("preparation spec", spec_payload),
        ("preparation profile", profile_payload),
    ):
        _check_keys(
            value,
            required={"path", "sha256"},
            context=name,
        )
    return CalibrationPreparationRecord(
        calibration_record_schema_version=_integer(
            "calibration_record_schema_version",
            payload["calibration_record_schema_version"],
        ),
        artifact_kind=_string(
            "artifact_kind",
            payload["artifact_kind"],
        ),
        claim_eligibility=_string(
            "claim_eligibility",
            payload["claim_eligibility"],
        ),
        preparation_id=_string(
            "preparation_id",
            payload["preparation_id"],
        ),
        producing_commit=_string(
            "producing_commit",
            payload["producing_commit"],
        ),
        fit_algorithm_id=_string(
            "fit_algorithm_id",
            payload["fit_algorithm_id"],
        ),
        source=CalibrationSourceSpec(
            path=_string("source path", source_payload["path"]),
            sha256=_string(
                "source sha256",
                source_payload["sha256"],
            ),
            provenance=_string(
                "source provenance",
                source_payload["provenance"],
            ),
            calibration_scope=_string(  # type: ignore[arg-type]
                "source calibration_scope",
                source_payload["calibration_scope"],
            ),
            capture_group_id=_string(
                "source capture_group_id",
                source_payload["capture_group_id"],
            ),
        ),
        calibration_artifact=FrozenCalibrationResult(
            path=_string(
                "calibration artifact path",
                artifact_payload["path"],
            ),
            sha256=_string(
                "calibration artifact sha256",
                artifact_payload["sha256"],
            ),
            format_id=_string(
                "calibration artifact format",
                artifact_payload["format"],
            ),
            rows=_integer("calibration rows", shape[0]),
            columns=_integer("calibration columns", shape[1]),
            sample_count=_integer(
                "calibration sample_count",
                artifact_payload["sample_count"],
            ),
        ),
        emulation_profile_sha256=_string(
            "emulation_profile_sha256",
            payload["emulation_profile_sha256"],
        ),
        preparation_spec=HashedPreparationInput(
            path=_string(
                "preparation spec path",
                spec_payload["path"],
            ),
            sha256=_string(
                "preparation spec sha256",
                spec_payload["sha256"],
            ),
        ),
        preparation_profile=HashedPreparationInput(
            path=_string(
                "preparation profile path",
                profile_payload["path"],
            ),
            sha256=_string(
                "preparation profile sha256",
                profile_payload["sha256"],
            ),
        ),
    )


def parse_preparation_report(
    payload: dict[str, object],
) -> PreparationReport:
    _check_keys(
        payload,
        required={
            "report_schema_version",
            "artifact_kind",
            "claim_eligibility",
            "preparation_id",
            "preparation_spec",
            "preparation_profile",
            "producing_commit",
            "sequences",
        },
        context="preparation report",
    )
    spec_payload = _object(
        "preparation_spec",
        payload["preparation_spec"],
    )
    profile_payload = _object(
        "preparation_profile",
        payload["preparation_profile"],
    )
    for name, value in (
        ("preparation spec", spec_payload),
        ("preparation profile", profile_payload),
    ):
        _check_keys(
            value,
            required={"path", "sha256"},
            context=name,
        )
    return PreparationReport(
        report_schema_version=_integer(
            "report_schema_version",
            payload["report_schema_version"],
        ),
        artifact_kind=_string(
            "artifact_kind",
            payload["artifact_kind"],
        ),
        claim_eligibility=_string(
            "claim_eligibility",
            payload["claim_eligibility"],
        ),
        preparation_id=_string(
            "preparation_id",
            payload["preparation_id"],
        ),
        preparation_spec=HashedPreparationInput(
            path=_string(
                "preparation spec path",
                spec_payload["path"],
            ),
            sha256=_string(
                "preparation spec sha256",
                spec_payload["sha256"],
            ),
        ),
        preparation_profile=HashedPreparationInput(
            path=_string(
                "preparation profile path",
                profile_payload["path"],
            ),
            sha256=_string(
                "preparation profile sha256",
                profile_payload["sha256"],
            ),
        ),
        producing_commit=_string(
            "producing_commit",
            payload["producing_commit"],
        ),
        sequences=tuple(
            _parse_prepared_sequence_result(item)
            for item in _list("sequences", payload["sequences"])
        ),
    )


def validate_frozen_calibration_binding(
    preparation: PreparationSpec,
    profile: PreparationProfile,
    calibration_record: CalibrationPreparationRecord,
    manifest: RecordedSequenceManifest,
    frozen: FrozenCalibrationAssetSpec,
) -> None:
    """Cross-check a schema-v3 calibration against its frozen build inputs."""

    if (
        preparation.preparation_id
        != calibration_record.preparation_id
        or preparation.producing_commit
        != calibration_record.producing_commit
        or preparation.producing_commit != frozen.producing_commit
    ):
        raise ValueError(
            "frozen calibration identity or producing commit does not "
            "match its preparation record and spec."
        )
    if profile.sensor_input_kind != (
        "structure_suppressed_emulation"
    ) or profile.emulation_profile is None:
        raise ValueError(
            "frozen calibration requires an emulated preparation profile."
        )
    expected_profile_sha256 = canonical_json_sha256(
        asdict(profile.emulation_profile)
    )
    if (
        calibration_record.emulation_profile_sha256.lower()
        != expected_profile_sha256.lower()
    ):
        raise ValueError(
            "calibration record does not bind the exact emulation profile."
        )
    sensor_input = manifest.sensor_input
    if (
        sensor_input is None
        or sensor_input.emulation_profile != profile.emulation_profile
        or sensor_input.provenance
        != profile.sensor_input_provenance
    ):
        raise ValueError(
            "manifest sensor input does not match the frozen preparation "
            "profile."
        )
    matches = [
        item
        for item in preparation.sequences
        if item.output_sequence_id == manifest.sequence_id
    ]
    if len(matches) != 1:
        raise ValueError(
            "preparation spec must contain the output sequence exactly once."
        )
    sequence = matches[0]
    calibration = sequence.calibration_source
    if calibration is None:
        raise ValueError(
            "preparation sequence has no calibration source."
        )
    if calibration_record.source != calibration:
        raise ValueError(
            "calibration record source does not match its preparation spec."
        )
    if (
        calibration.path != frozen.source_path
        or calibration.sha256.lower()
        != frozen.source_sha256.lower()
        or calibration.provenance != frozen.source_provenance
        or calibration.calibration_scope
        != frozen.calibration_scope
    ):
        raise ValueError(
            "frozen calibration source does not match its preparation spec."
        )
    if (
        calibration_record.calibration_artifact.path != frozen.path
        or calibration_record.calibration_artifact.sha256.lower()
        != frozen.sha256.lower()
    ):
        raise ValueError(
            "frozen calibration artifact does not match its preparation "
            "record."
        )
    if (
        calibration_record.preparation_spec.path
        != frozen.preparation_spec_path
        or calibration_record.preparation_spec.sha256.lower()
        != frozen.preparation_spec_sha256.lower()
        or calibration_record.preparation_profile.path
        != frozen.preparation_profile_path
        or calibration_record.preparation_profile.sha256.lower()
        != frozen.preparation_profile_sha256.lower()
    ):
        raise ValueError(
            "frozen calibration metadata inputs do not match its "
            "preparation record."
        )

    template = sequence.manifest_template
    prepared_fields = (
        manifest.sequence_id,
        manifest.split,
        manifest.split_provenance,
        manifest.scenario,
        manifest.media,
        manifest.frame_clock,
        manifest.normalization,
        manifest.calibration,
        manifest.gas_identity,
        manifest.dataset_lineage,
        manifest.capture_group,
    )
    template_fields = (
        sequence.output_sequence_id,
        template.split,
        template.split_provenance,
        template.scenario,
        template.media,
        template.frame_clock,
        template.normalization,
        template.calibration,
        template.gas_identity,
        template.dataset_lineage,
        template.capture_group,
    )
    if prepared_fields != template_fields:
        raise ValueError(
            "prepared manifest differs from its immutable acquisition and "
            "preprocessing contract."
        )


def _parse_capture_group(value: object) -> CaptureGroupDeclaration:
    payload = _object("capture group", value)
    _check_keys(
        payload,
        required={
            "capture_group_id",
            "role",
            "provenance",
            "asset_sha256s",
        },
        context="capture group",
    )
    return CaptureGroupDeclaration(
        capture_group_id=_string(
            "capture_group_id",
            payload["capture_group_id"],
        ),
        role=_string("capture-group role", payload["role"]),  # type: ignore[arg-type]
        provenance=_string(
            "capture-group provenance",
            payload["provenance"],
        ),
        asset_sha256s=tuple(
            _string("capture-group asset sha256", item)
            for item in _list(
                "capture-group asset_sha256s",
                payload["asset_sha256s"],
            )
        ),
    )


def _parse_prepared_sequence_result(
    value: object,
) -> PreparedSequenceResult:
    payload = _object("prepared sequence", value)
    _check_keys(
        payload,
        required={"sequence_id", "path", "manifest_sha256"},
        context="prepared sequence",
    )
    return PreparedSequenceResult(
        sequence_id=_string(
            "sequence_id",
            payload["sequence_id"],
        ),
        path=_string("path", payload["path"]),
        manifest_sha256=_string(
            "manifest_sha256",
            payload["manifest_sha256"],
        ),
    )


def _parse_exclusion(value: object) -> ExclusionRecord:
    payload = _object("exclusion", value)
    _check_keys(
        payload,
        required={"capture_group_id", "reason_code", "rationale"},
        context="exclusion",
    )
    return ExclusionRecord(
        capture_group_id=_string(
            "excluded capture_group_id",
            payload["capture_group_id"],
        ),
        reason_code=_string(  # type: ignore[arg-type]
            "exclusion reason_code",
            payload["reason_code"],
        ),
        rationale=_string(
            "exclusion rationale",
            payload["rationale"],
        ),
    )


def _parse_sequence(
    value: object,
    *,
    schema_version: int,
) -> SequencePreparationSpec:
    payload = _object("preparation sequence", value)
    required = {
        "output_sequence_id",
        "manifest_template",
        "calibration_source",
    }
    if schema_version >= 2:
        required.add("media_window_request")
    _check_keys(
        payload,
        required=required,
        context="preparation sequence",
    )
    calibration_payload = payload["calibration_source"]
    calibration: CalibrationSourceSpec | None = None
    if calibration_payload is not None:
        values = _object("calibration_source", calibration_payload)
        _check_keys(
            values,
            required={
                "path",
                "sha256",
                "provenance",
                "calibration_scope",
                "capture_group_id",
            },
            context="calibration source",
        )
        calibration = CalibrationSourceSpec(
            path=_string("calibration path", values["path"]),
            sha256=_string("calibration sha256", values["sha256"]),
            provenance=_string(
                "calibration provenance",
                values["provenance"],
            ),
            calibration_scope=_string(  # type: ignore[arg-type]
                "calibration_scope",
                values["calibration_scope"],
            ),
            capture_group_id=_string(
                "calibration capture_group_id",
                values["capture_group_id"],
            ),
        )
    template_payload = _object(
        "manifest_template",
        payload["manifest_template"],
    )
    window_request: MediaWindowRequest | None = None
    if schema_version >= 2:
        raw_window_payload = payload["media_window_request"]
        if raw_window_payload is None:
            if schema_version < 4:
                raise ValueError(
                    "preparation spec v2/v3 requires a media-window request."
                )
        else:
            window_payload = _object(
                "media_window_request",
                raw_window_payload,
            )
            _check_keys(
                window_payload,
                required={
                    "requested_start_time_seconds",
                    "requested_end_time_seconds",
                    "expected_source_start_frame_index",
                    "expected_source_end_frame_index_exclusive",
                    "selection_metadata_path",
                    "selection_metadata_sha256",
                    "selection_provenance",
                },
                context="media-window request",
            )
            window_request = MediaWindowRequest(
                requested_start_time_seconds=_number(
                    "requested_start_time_seconds",
                    window_payload["requested_start_time_seconds"],
                ),
                requested_end_time_seconds=_number(
                    "requested_end_time_seconds",
                    window_payload["requested_end_time_seconds"],
                ),
                expected_source_start_frame_index=_integer(
                    "expected_source_start_frame_index",
                    window_payload["expected_source_start_frame_index"],
                ),
                expected_source_end_frame_index_exclusive=_integer(
                    "expected_source_end_frame_index_exclusive",
                    window_payload[
                        "expected_source_end_frame_index_exclusive"
                    ],
                ),
                selection_metadata_path=_string(
                    "selection_metadata_path",
                    window_payload["selection_metadata_path"],
                ),
                selection_metadata_sha256=_string(
                    "selection_metadata_sha256",
                    window_payload["selection_metadata_sha256"],
                ),
                selection_provenance=_string(
                    "selection_provenance",
                    window_payload["selection_provenance"],
                ),
            )
    return SequencePreparationSpec(
        output_sequence_id=_string(
            "output_sequence_id",
            payload["output_sequence_id"],
        ),
        manifest_template=parse_recorded_sequence_manifest(
            template_payload
        ),
        calibration_source=calibration,
        media_window_request=window_request,
    )


def _check_keys(
    payload: dict[str, object],
    *,
    required: set[str],
    context: str,
) -> None:
    keys = set(payload)
    missing = required - keys
    extra = keys - required
    if missing:
        raise ValueError(f"{context} is missing keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"{context} has unknown keys: {sorted(extra)}")


def _object(name: str, value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return value


def _list(name: str, value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array.")
    return value


def _string(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string.")
    return value


def _integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    return value


def _number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number.")
    return float(value)


def _require_text(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be empty.")


def _require_identifier(name: str, value: str) -> None:
    _require_text(name, value)
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(
            f"{name} must use letters, digits, '.', '_', or '-'."
        )


def _require_relative_path(name: str, value: str) -> None:
    _require_text(name, value)
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"{name} must be relative and contain no '..'."
        )


def _require_sha256(name: str, value: str) -> None:
    if len(value) != 64:
        raise ValueError(
            f"{name} must contain 64 hexadecimal characters."
        )
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(
            f"{name} must contain 64 hexadecimal characters."
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
