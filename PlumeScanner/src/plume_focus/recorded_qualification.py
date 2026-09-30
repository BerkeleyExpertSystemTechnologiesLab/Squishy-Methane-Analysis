"""Strict qualification records for externally recorded datasets.

Qualification sits between byte/decode auditing and sequence preparation.  It
records what the dataset owner permits and how every audited asset maps to the
paper's acquisition groups.  It never decodes media and is not sequence or
performance evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from plume_focus.recorded_artifacts import (
    canonical_json_bytes,
    write_json_artifact,
)
from plume_focus.recorded_dataset import (
    DatasetAuditReport,
    parse_dataset_audit_report,
)
from plume_focus.recorded_manifest_io import file_sha256, load_json_object


# Schema v1 is an immutable compatibility contract.  New callers must opt in
# to v2 explicitly; keeping these legacy constants at one also preserves the
# exact bytes produced by the original template/check APIs.
QUALIFICATION_SCHEMA_VERSION = 1
QUALIFICATION_SCHEMA_V2_VERSION = 2
SUPPORTED_QUALIFICATION_SCHEMA_VERSIONS = frozenset({1, 2})
QUALIFICATION_CHECK_SCHEMA_VERSION = 1
QUALIFICATION_CHECK_SCHEMA_V2_VERSION = 2
PROJECT_PREREGISTRATION_SCHEMA_VERSION = 1
QUALIFICATION_PURPOSE = "local-raw-pilot"
CLAIM_ELIGIBILITY = "not_sequence_evidence"

EvidenceKind = Literal[
    "formal_terms",
    "dataset_owner_correspondence",
    "dataset_owner_metadata",
    "primary_publication",
]
PermissionValue = Literal["unknown", "allowed", "denied"]
IntervalStatus = Literal["unresolved", "verified"]
DatasetSplit = Literal["development", "held-out"]
AssignmentStatus = Literal["unresolved", "inferred", "verified"]
ProviderGasIdentity = Literal[
    "methane", "propane", "hydrocarbon_unspecified"
]
QualificationPurposeV2 = Literal["diagnostic_raw_pilot", "evidence_suite"]
ProjectSplitRole = Literal["development", "held-out", "excluded"]
QualificationOperation = Literal[
    "decode_analysis",
    "extract_bounded_window",
    "interpret_provider_intervals",
    "construct_local_derivative",
    "render_local_media",
    "publish_metrics",
    "publish_media",
    "redistribute",
]

EVIDENCE_KINDS = frozenset(
    {
        "formal_terms",
        "dataset_owner_correspondence",
        "dataset_owner_metadata",
        "primary_publication",
    }
)
AUTHORIZING_EVIDENCE_KINDS = frozenset(
    {
        "formal_terms",
        "dataset_owner_correspondence",
    }
)
ASSIGNMENT_EVIDENCE_KINDS = frozenset(
    {
        "dataset_owner_correspondence",
        "dataset_owner_metadata",
    }
)
INTERVAL_EVIDENCE_KINDS = frozenset(
    {
        "dataset_owner_correspondence",
        "dataset_owner_metadata",
        "primary_publication",
    }
)
PERMISSION_SCOPES = (
    "local_decode_analysis",
    "bounded_interval_extraction",
    "local_derivative_construction",
    "publish_aggregate_metrics",
    "publish_frames_or_clips",
    "redistribution",
)
LOCAL_RAW_PILOT_PERMISSION_SCOPES = PERMISSION_SCOPES[:3]
QUALIFICATION_PURPOSES_V2 = frozenset(
    {"diagnostic_raw_pilot", "evidence_suite"}
)
QUALIFICATION_OPERATIONS = (
    "decode_analysis",
    "extract_bounded_window",
    "interpret_provider_intervals",
    "construct_local_derivative",
    "render_local_media",
    "publish_metrics",
    "publish_media",
    "redistribute",
)
OPERATION_PERMISSION_SCOPE = {
    "decode_analysis": "local_decode_analysis",
    "extract_bounded_window": "bounded_interval_extraction",
    "interpret_provider_intervals": "bounded_interval_extraction",
    "construct_local_derivative": "local_derivative_construction",
    "render_local_media": "local_derivative_construction",
    "publish_metrics": "publish_aggregate_metrics",
    "publish_media": "publish_frames_or_clips",
    "redistribute": "redistribution",
}


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


def _require_sha256(name: str, value: str) -> None:
    _require_text(name, value)
    if len(value) != 64:
        raise ValueError(f"{name} must contain 64 hexadecimal characters.")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(
            f"{name} must contain 64 hexadecimal characters."
        ) from error


def _require_relative_path(name: str, value: str) -> None:
    _require_text(name, value)
    path = PurePosixPath(value)
    if (
        "\\" in value
        or path.is_absolute()
        or ".." in path.parts
        or str(path) != value
    ):
        raise ValueError(f"{name} must be a normalized relative POSIX path.")


def _require_utc_timestamp(name: str, value: str) -> None:
    _require_text(name, value)
    if not value.endswith("Z"):
        raise ValueError(f"{name} must use an explicit UTC Z suffix.")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO-8601 timestamp.") from error
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"{name} must use UTC.")


def _require_unique_texts(name: str, values: tuple[str, ...]) -> None:
    for value in values:
        _require_text(name, value)
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates.")


@dataclass(frozen=True)
class AuditReference:
    """Portable, hash-bound reference to the accession audit."""

    dataset_id: str
    accession_id: str
    path: str
    sha256: str

    def __post_init__(self) -> None:
        _require_text("dataset audit dataset_id", self.dataset_id)
        _require_text("dataset audit accession_id", self.accession_id)
        _require_relative_path("dataset audit path", self.path)
        _require_sha256("dataset audit sha256", self.sha256)

    def to_payload(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "accession_id": self.accession_id,
            "path": self.path,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ProjectPreregistrationReference:
    """Exact reference to a separately frozen project-selection decision."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        _require_relative_path("project preregistration path", self.path)
        _require_sha256("project preregistration sha256", self.sha256)

    def to_payload(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class ProjectAssetAssignment:
    """One outcome-blind project role, distinct from provider metadata."""

    relative_path: str
    sha256: str
    split: ProjectSplitRole
    exclusion_reason: str | None = None

    def __post_init__(self) -> None:
        _require_relative_path(
            "project assignment relative_path", self.relative_path
        )
        _require_sha256("project assignment sha256", self.sha256)
        if self.split not in {"development", "held-out", "excluded"}:
            raise ValueError("unknown project assignment split.")
        if self.split == "excluded":
            if self.exclusion_reason is None:
                raise ValueError(
                    "excluded project assignments require an exclusion_reason."
                )
            _require_text("project exclusion_reason", self.exclusion_reason)
        elif self.exclusion_reason is not None:
            raise ValueError(
                "active project assignments cannot carry an exclusion_reason."
            )

    @property
    def identity(self) -> tuple[str, str]:
        return self.relative_path, self.sha256.lower()

    def to_payload(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "split": self.split,
            "exclusion_reason": self.exclusion_reason,
        }


@dataclass(frozen=True)
class ProjectPreregistration:
    """Content-hashed project selection created before outcome inspection."""

    preregistration_id: str
    purpose: QualificationPurposeV2
    created_at_utc: str
    dataset_audit: AuditReference
    operations: tuple[QualificationOperation, ...]
    assignments: tuple[ProjectAssetAssignment, ...]
    project_preregistration_schema_version: int = (
        PROJECT_PREREGISTRATION_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.project_preregistration_schema_version, bool)
            or not isinstance(self.project_preregistration_schema_version, int)
            or self.project_preregistration_schema_version
            != PROJECT_PREREGISTRATION_SCHEMA_VERSION
        ):
            raise ValueError("unsupported project preregistration schema version.")
        _require_text("preregistration_id", self.preregistration_id)
        if self.purpose not in QUALIFICATION_PURPOSES_V2:
            raise ValueError("unknown qualification v2 purpose.")
        _require_utc_timestamp("project created_at_utc", self.created_at_utc)
        _require_unique_texts("project operations", self.operations)
        if not self.operations:
            raise ValueError("project preregistration requires operations.")
        unknown_operations = set(self.operations) - set(QUALIFICATION_OPERATIONS)
        if unknown_operations:
            raise ValueError(
                f"unknown project operations: {sorted(unknown_operations)}"
            )
        if "decode_analysis" not in self.operations:
            raise ValueError(
                "project preregistration must include decode_analysis."
            )
        identities = [item.identity for item in self.assignments]
        if len(identities) != len(set(identities)):
            raise ValueError(
                "project assignment path/SHA-256 identities must be unique."
            )
        paths = [item.relative_path for item in self.assignments]
        if len(paths) != len(set(paths)):
            raise ValueError("project assignment paths must be unique.")
        active = [item for item in self.assignments if item.split != "excluded"]
        if not active:
            raise ValueError(
                "project preregistration requires at least one active asset."
            )
        if self.purpose == "diagnostic_raw_pilot":
            if any(item.split != "development" for item in self.assignments):
                raise ValueError(
                    "diagnostic_raw_pilot assignments must all be development; "
                    "omit unselected audit assets instead of assigning held-out "
                    "or excluded roles."
                )
        elif not any(item.split == "development" for item in active) or not any(
            item.split == "held-out" for item in active
        ):
            raise ValueError(
                "evidence_suite requires non-empty development and held-out splits."
            )

    @property
    def active_assignments(self) -> tuple[ProjectAssetAssignment, ...]:
        return tuple(item for item in self.assignments if item.split != "excluded")

    @property
    def required_permission_scopes(self) -> tuple[str, ...]:
        selected = {
            OPERATION_PERMISSION_SCOPE[operation]
            for operation in self.operations
        }
        return tuple(scope for scope in PERMISSION_SCOPES if scope in selected)

    def to_payload(self) -> dict[str, object]:
        return {
            "project_preregistration_schema_version": (
                self.project_preregistration_schema_version
            ),
            "artifact_kind": "recorded_project_preregistration",
            "claim_eligibility": CLAIM_ELIGIBILITY,
            "preregistration_id": self.preregistration_id,
            "purpose": self.purpose,
            "created_at_utc": self.created_at_utc,
            "dataset_audit": self.dataset_audit.to_payload(),
            "operations": list(self.operations),
            "assignments": [item.to_payload() for item in self.assignments],
        }


@dataclass(frozen=True)
class EvidenceRecord:
    """One stable source that supports permission or metadata decisions."""

    evidence_id: str
    kind: EvidenceKind
    reference: str
    observed_at_utc: str
    content_sha256: str | None
    summary: str

    def __post_init__(self) -> None:
        _require_text("evidence_id", self.evidence_id)
        if self.kind not in EVIDENCE_KINDS:
            raise ValueError("unknown qualification evidence kind.")
        _require_text("evidence reference", self.reference)
        _require_utc_timestamp("evidence observed_at_utc", self.observed_at_utc)
        if self.content_sha256 is not None:
            _require_sha256("evidence content_sha256", self.content_sha256)
        _require_text("evidence summary", self.summary)

    def to_payload(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "reference": self.reference,
            "observed_at_utc": self.observed_at_utc,
            "content_sha256": self.content_sha256,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class PermissionDecision:
    """Independent decision for one declared use scope."""

    decision: PermissionValue
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.decision not in {"unknown", "allowed", "denied"}:
            raise ValueError("unknown permission decision.")
        _require_unique_texts("permission evidence_ids", self.evidence_ids)
        if self.decision != "unknown" and not self.evidence_ids:
            raise ValueError(
                "allowed or denied permission decisions require evidence_ids."
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "decision": self.decision,
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True)
class PermissionSet:
    """All permission scopes; publication never implies local processing."""

    local_decode_analysis: PermissionDecision
    bounded_interval_extraction: PermissionDecision
    local_derivative_construction: PermissionDecision
    publish_aggregate_metrics: PermissionDecision
    publish_frames_or_clips: PermissionDecision
    redistribution: PermissionDecision

    def to_payload(self) -> dict[str, object]:
        return {
            scope: getattr(self, scope).to_payload()
            for scope in PERMISSION_SCOPES
        }


@dataclass(frozen=True)
class IntervalException:
    """A provenance-backed deviation from nominal native intervals."""

    relative_path: str
    description: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_relative_path("interval exception relative_path", self.relative_path)
        _require_text("interval exception description", self.description)
        _require_unique_texts(
            "interval exception evidence_ids", self.evidence_ids
        )
        if not self.evidence_ids:
            raise ValueError("interval exceptions require evidence_ids.")

    def to_payload(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "description": self.description,
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True)
class NativeIntervalSemantics:
    """Meaning and trimming convention of the dataset's native intervals."""

    status: IntervalStatus
    class_zero_meaning: str | None
    nominal_interval_seconds: float | None
    trim_start_seconds: float | None
    trim_end_seconds: float | None
    exceptions: tuple[IntervalException, ...]
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status not in {"unresolved", "verified"}:
            raise ValueError("unknown native interval status.")
        if self.class_zero_meaning is not None:
            _require_text("class_zero_meaning", self.class_zero_meaning)
        for name, value, positive in (
            ("nominal_interval_seconds", self.nominal_interval_seconds, True),
            ("trim_start_seconds", self.trim_start_seconds, False),
            ("trim_end_seconds", self.trim_end_seconds, False),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or (float(value) <= 0.0 if positive else float(value) < 0.0)
            ):
                comparison = "positive" if positive else "non-negative"
                raise ValueError(f"{name} must be {comparison} and finite or null.")
        paths = [item.relative_path for item in self.exceptions]
        if len(paths) != len(set(paths)):
            raise ValueError("interval exception paths must be unique.")
        _require_unique_texts("interval evidence_ids", self.evidence_ids)
        if self.status == "verified":
            if (
                self.class_zero_meaning is None
                or self.nominal_interval_seconds is None
                or self.trim_start_seconds is None
                or self.trim_end_seconds is None
                or not self.evidence_ids
            ):
                raise ValueError(
                    "verified interval semantics require class-zero meaning, "
                    "nominal/trim values, and evidence_ids."
                )
            if (
                self.trim_start_seconds + self.trim_end_seconds
                >= self.nominal_interval_seconds
            ):
                raise ValueError(
                    "interval trim duration must be shorter than the nominal interval."
                )

    def to_payload(self) -> dict[str, object]:
        return {
            "status": self.status,
            "class_zero_meaning": self.class_zero_meaning,
            "nominal_interval_seconds": self.nominal_interval_seconds,
            "trim_start_seconds": self.trim_start_seconds,
            "trim_end_seconds": self.trim_end_seconds,
            "exceptions": [item.to_payload() for item in self.exceptions],
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True)
class QualifiedAsset:
    """Exact audit identity plus independently evidenced dataset mapping."""

    relative_path: str
    sha256: str
    paper_group_id: str | None
    capture_group_id: str | None
    split: DatasetSplit | None
    assignment_status: AssignmentStatus
    assignment_evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_relative_path("qualified asset relative_path", self.relative_path)
        _require_sha256("qualified asset sha256", self.sha256)
        for name, value in (
            ("paper_group_id", self.paper_group_id),
            ("capture_group_id", self.capture_group_id),
        ):
            if value is not None:
                _require_text(name, value)
        if self.split not in {None, "development", "held-out"}:
            raise ValueError("qualified asset split must be development, held-out, or null.")
        if self.assignment_status not in {"unresolved", "inferred", "verified"}:
            raise ValueError(
                "asset assignment_status must be unresolved, inferred, or verified."
            )
        _require_unique_texts(
            "asset assignment_evidence_ids", self.assignment_evidence_ids
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "paper_group_id": self.paper_group_id,
            "capture_group_id": self.capture_group_id,
            "split": self.split,
            "assignment_status": self.assignment_status,
            "assignment_evidence_ids": list(self.assignment_evidence_ids),
        }


@dataclass(frozen=True)
class ProviderAssetFact:
    """Provider-declared acquisition facts, deliberately without project split."""

    relative_path: str
    sha256: str
    paper_group_id: str | None
    capture_group_id: str | None
    gas_identity: ProviderGasIdentity | None
    fact_status: AssignmentStatus
    fact_evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_relative_path("provider asset relative_path", self.relative_path)
        _require_sha256("provider asset sha256", self.sha256)
        for name, value in (
            ("paper_group_id", self.paper_group_id),
            ("capture_group_id", self.capture_group_id),
        ):
            if value is not None:
                _require_text(name, value)
        if self.gas_identity not in {
            None,
            "methane",
            "propane",
            "hydrocarbon_unspecified",
        }:
            raise ValueError("unknown provider gas identity.")
        if self.fact_status not in {"unresolved", "inferred", "verified"}:
            raise ValueError(
                "provider fact_status must be unresolved, inferred, or verified."
            )
        _require_unique_texts(
            "provider fact_evidence_ids", self.fact_evidence_ids
        )

    @property
    def identity(self) -> tuple[str, str]:
        return self.relative_path, self.sha256.lower()

    def to_payload(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "paper_group_id": self.paper_group_id,
            "capture_group_id": self.capture_group_id,
            "gas_identity": self.gas_identity,
            "fact_status": self.fact_status,
            "fact_evidence_ids": list(self.fact_evidence_ids),
        }


@dataclass(frozen=True)
class DatasetQualificationV2:
    """Purpose-aware qualification with source facts split from project roles."""

    qualification_id: str
    asset_root: str
    purpose: QualificationPurposeV2
    dataset_audit: AuditReference
    project_preregistration: ProjectPreregistrationReference
    evidence_records: tuple[EvidenceRecord, ...]
    permissions: PermissionSet
    native_interval_semantics: NativeIntervalSemantics
    provider_assets: tuple[ProviderAssetFact, ...]
    qualification_schema_version: int = QUALIFICATION_SCHEMA_V2_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.qualification_schema_version, bool)
            or not isinstance(self.qualification_schema_version, int)
            or self.qualification_schema_version
            != QUALIFICATION_SCHEMA_V2_VERSION
        ):
            raise ValueError("unsupported qualification schema version.")
        if self.purpose not in QUALIFICATION_PURPOSES_V2:
            raise ValueError("unknown qualification v2 purpose.")
        _require_text("qualification_id", self.qualification_id)
        _require_relative_path("qualification asset_root", self.asset_root)
        evidence_ids = [item.evidence_id for item in self.evidence_records]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("qualification evidence_ids must be unique.")
        evidence_by_id = {
            item.evidence_id: item for item in self.evidence_records
        }
        referenced: list[tuple[str, tuple[str, ...]]] = []
        referenced.extend(
            (f"permission {scope}", getattr(self.permissions, scope).evidence_ids)
            for scope in PERMISSION_SCOPES
        )
        referenced.append(
            ("native interval semantics", self.native_interval_semantics.evidence_ids)
        )
        referenced.extend(
            (f"interval exception {item.relative_path}", item.evidence_ids)
            for item in self.native_interval_semantics.exceptions
        )
        referenced.extend(
            (f"provider asset {item.relative_path}", item.fact_evidence_ids)
            for item in self.provider_assets
        )
        for context, ids in referenced:
            unknown = set(ids) - set(evidence_by_id)
            if unknown:
                raise ValueError(
                    f"{context} references unknown evidence_ids: {sorted(unknown)}"
                )
        for scope in PERMISSION_SCOPES:
            decision = getattr(self.permissions, scope)
            if decision.decision == "unknown":
                continue
            decision_records = [
                evidence_by_id[evidence_id]
                for evidence_id in decision.evidence_ids
            ]
            authorizing = [
                record
                for record in decision_records
                if record.kind in AUTHORIZING_EVIDENCE_KINDS
            ]
            if not authorizing:
                raise ValueError(
                    f"permission {scope} decision requires formal terms or "
                    "dataset-owner correspondence evidence."
                )
            if any(record.content_sha256 is None for record in decision_records):
                raise ValueError(
                    f"permission {scope} evidence must be content-hashed in "
                    "qualification schema v2."
                )
        provider_paths = [item.relative_path for item in self.provider_assets]
        if len(provider_paths) != len(set(provider_paths)):
            raise ValueError("provider asset paths must be unique.")
        provider_identities = [item.identity for item in self.provider_assets]
        if len(provider_identities) != len(set(provider_identities)):
            raise ValueError("provider asset identities must be unique.")
        unknown_exception_paths = {
            item.relative_path for item in self.native_interval_semantics.exceptions
        } - set(provider_paths)
        if unknown_exception_paths:
            raise ValueError(
                "interval exceptions reference unknown provider assets: "
                f"{sorted(unknown_exception_paths)}"
            )
        for item in self.provider_assets:
            if item.fact_status != "verified":
                continue
            if not item.fact_evidence_ids:
                raise ValueError(
                    f"verified provider asset {item.relative_path} requires "
                    "fact_evidence_ids."
                )
            authoritative = [
                evidence_by_id[evidence_id]
                for evidence_id in item.fact_evidence_ids
                if evidence_by_id[evidence_id].kind in ASSIGNMENT_EVIDENCE_KINDS
            ]
            if not authoritative:
                raise ValueError(
                    f"verified provider asset {item.relative_path} requires "
                    "dataset-owner metadata or correspondence evidence."
                )
            if any(record.content_sha256 is None for record in authoritative):
                raise ValueError(
                    f"verified provider asset {item.relative_path} requires "
                    "content-hashed authoritative evidence."
                )
        if self.native_interval_semantics.status == "verified":
            interval_records = [
                evidence_by_id[evidence_id]
                for evidence_id in self.native_interval_semantics.evidence_ids
                if evidence_by_id[evidence_id].kind in INTERVAL_EVIDENCE_KINDS
            ]
            if not interval_records:
                raise ValueError(
                    "verified native interval semantics require authoritative "
                    "metadata evidence."
                )
            if any(record.content_sha256 is None for record in interval_records):
                raise ValueError(
                    "verified native interval semantics require content-hashed "
                    "authoritative evidence."
                )

    @property
    def evidence_by_id(self) -> dict[str, EvidenceRecord]:
        return {item.evidence_id: item for item in self.evidence_records}

    def purpose_blockers(
        self,
        preregistration: ProjectPreregistration,
    ) -> tuple[str, ...]:
        """Return deterministic v2 policy blockers, excluding file/audit I/O."""

        blockers: list[str] = []
        evidence = self.evidence_by_id
        for scope in preregistration.required_permission_scopes:
            decision = getattr(self.permissions, scope)
            if decision.decision != "allowed":
                blockers.append(f"permission.{scope}.{decision.decision}")
                continue
            if not any(
                evidence[item].kind in AUTHORIZING_EVIDENCE_KINDS
                and evidence[item].content_sha256 is not None
                for item in decision.evidence_ids
            ):
                blockers.append(
                    f"permission.{scope}.lacks_content_hashed_authorization"
                )
        if "interpret_provider_intervals" in preregistration.operations:
            if self.native_interval_semantics.status != "verified":
                blockers.append("native_interval_semantics.unresolved")
            for exception in self.native_interval_semantics.exceptions:
                if not any(
                    evidence[item].kind in INTERVAL_EVIDENCE_KINDS
                    and evidence[item].content_sha256 is not None
                    for item in exception.evidence_ids
                ):
                    blockers.append(
                        f"interval_exception.{exception.relative_path}."
                        "lacks_content_hashed_authoritative_evidence"
                    )
        facts = {item.identity: item for item in self.provider_assets}
        for assignment in preregistration.active_assignments:
            asset = facts[assignment.identity]
            if asset.fact_status != "verified":
                blockers.append(
                    f"provider_asset.{asset.relative_path}."
                    f"facts_{asset.fact_status}"
                )
            if asset.capture_group_id is None:
                blockers.append(
                    f"provider_asset.{asset.relative_path}.capture_group_id_missing"
                )
            if asset.gas_identity is None:
                blockers.append(
                    f"provider_asset.{asset.relative_path}.gas_identity_missing"
                )
        if self.purpose == "evidence_suite":
            for group_field in ("paper_group_id", "capture_group_id"):
                group_splits: dict[str, set[str]] = {}
                for assignment in preregistration.active_assignments:
                    asset = facts[assignment.identity]
                    group_id = getattr(asset, group_field)
                    if group_id is not None:
                        group_splits.setdefault(group_id, set()).add(
                            assignment.split
                        )
                for group_id in sorted(group_splits):
                    if len(group_splits[group_id]) > 1:
                        blockers.append(
                            f"provider_{group_field}.{group_id}.crosses_splits"
                        )
        return tuple(blockers)

    def to_payload(self) -> dict[str, object]:
        return {
            "qualification_schema_version": self.qualification_schema_version,
            "artifact_kind": "recorded_dataset_qualification",
            "claim_eligibility": CLAIM_ELIGIBILITY,
            "qualification_id": self.qualification_id,
            "asset_root": self.asset_root,
            "purpose": self.purpose,
            "dataset_audit": self.dataset_audit.to_payload(),
            "project_preregistration": self.project_preregistration.to_payload(),
            "evidence_records": [
                item.to_payload() for item in self.evidence_records
            ],
            "permissions": self.permissions.to_payload(),
            "native_interval_semantics": (
                self.native_interval_semantics.to_payload()
            ),
            "provider_assets": [item.to_payload() for item in self.provider_assets],
        }


@dataclass(frozen=True)
class DatasetQualification:
    """Strict schema-v1 qualification declaration."""

    qualification_id: str
    asset_root: str
    dataset_audit: AuditReference
    evidence_records: tuple[EvidenceRecord, ...]
    permissions: PermissionSet
    native_interval_semantics: NativeIntervalSemantics
    assets: tuple[QualifiedAsset, ...]
    qualification_schema_version: int = QUALIFICATION_SCHEMA_VERSION
    purpose: str = QUALIFICATION_PURPOSE

    def __post_init__(self) -> None:
        if (
            isinstance(self.qualification_schema_version, bool)
            or not isinstance(self.qualification_schema_version, int)
            or self.qualification_schema_version != QUALIFICATION_SCHEMA_VERSION
        ):
            raise ValueError("unsupported qualification schema version.")
        if self.purpose != QUALIFICATION_PURPOSE:
            raise ValueError("qualification purpose must be local-raw-pilot.")
        _require_text("qualification_id", self.qualification_id)
        _require_relative_path("qualification asset_root", self.asset_root)
        evidence_ids = [item.evidence_id for item in self.evidence_records]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("qualification evidence_ids must be unique.")
        known_evidence = set(evidence_ids)
        referenced: list[tuple[str, tuple[str, ...]]] = []
        referenced.extend(
            (f"permission {scope}", getattr(self.permissions, scope).evidence_ids)
            for scope in PERMISSION_SCOPES
        )
        referenced.append(
            ("native interval semantics", self.native_interval_semantics.evidence_ids)
        )
        referenced.extend(
            (f"interval exception {item.relative_path}", item.evidence_ids)
            for item in self.native_interval_semantics.exceptions
        )
        referenced.extend(
            (f"asset {item.relative_path}", item.assignment_evidence_ids)
            for item in self.assets
        )
        for context, ids in referenced:
            unknown = set(ids) - known_evidence
            if unknown:
                raise ValueError(
                    f"{context} references unknown evidence_ids: {sorted(unknown)}"
                )
        evidence_by_id = {
            item.evidence_id: item for item in self.evidence_records
        }
        for scope in PERMISSION_SCOPES:
            decision = getattr(self.permissions, scope)
            if decision.decision == "unknown":
                continue
            if not any(
                evidence_by_id[item].kind in AUTHORIZING_EVIDENCE_KINDS
                for item in decision.evidence_ids
            ):
                raise ValueError(
                    f"permission {scope} decision requires formal terms or "
                    "dataset-owner correspondence evidence."
                )
        paths = [item.relative_path for item in self.assets]
        if len(paths) != len(set(paths)):
            raise ValueError("qualified asset paths must be unique.")
        asset_paths = set(paths)
        unknown_exception_paths = {
            item.relative_path
            for item in self.native_interval_semantics.exceptions
        } - asset_paths
        if unknown_exception_paths:
            raise ValueError(
                "interval exceptions reference unknown assets: "
                f"{sorted(unknown_exception_paths)}"
            )

    @property
    def evidence_by_id(self) -> dict[str, EvidenceRecord]:
        return {item.evidence_id: item for item in self.evidence_records}

    def local_raw_pilot_blockers(self) -> tuple[str, ...]:
        """Return deterministic policy/metadata blockers, excluding audit I/O."""

        blockers: list[str] = []
        evidence = self.evidence_by_id
        for scope in LOCAL_RAW_PILOT_PERMISSION_SCOPES:
            decision = getattr(self.permissions, scope)
            if decision.decision != "allowed":
                blockers.append(
                    f"permission.{scope}.{decision.decision}"
                )
                continue
            if not any(
                evidence[item].kind in AUTHORIZING_EVIDENCE_KINDS
                for item in decision.evidence_ids
            ):
                blockers.append(
                    f"permission.{scope}.lacks_dataset_owner_or_terms_evidence"
                )
        if self.native_interval_semantics.status != "verified":
            blockers.append("native_interval_semantics.unresolved")
        elif not any(
            evidence[item].kind in INTERVAL_EVIDENCE_KINDS
            for item in self.native_interval_semantics.evidence_ids
        ):
            blockers.append(
                "native_interval_semantics.lacks_authoritative_metadata_evidence"
            )
        for exception in self.native_interval_semantics.exceptions:
            if not any(
                evidence[item].kind in INTERVAL_EVIDENCE_KINDS
                for item in exception.evidence_ids
            ):
                blockers.append(
                    f"interval_exception.{exception.relative_path}."
                    "lacks_authoritative_metadata_evidence"
                )
        for asset in self.assets:
            if asset.assignment_status != "verified":
                blockers.append(
                    f"asset.{asset.relative_path}.assignment_{asset.assignment_status}"
                )
            if asset.paper_group_id is None:
                blockers.append(f"asset.{asset.relative_path}.paper_group_id_missing")
            if asset.capture_group_id is None:
                blockers.append(f"asset.{asset.relative_path}.capture_group_id_missing")
            if asset.split is None:
                blockers.append(f"asset.{asset.relative_path}.split_missing")
            if not asset.assignment_evidence_ids:
                blockers.append(
                    f"asset.{asset.relative_path}.assignment_evidence_missing"
                )
            elif not any(
                evidence[item].kind in ASSIGNMENT_EVIDENCE_KINDS
                for item in asset.assignment_evidence_ids
            ):
                blockers.append(
                    f"asset.{asset.relative_path}."
                    "assignment_lacks_authoritative_metadata_evidence"
                )
        if not any(asset.split == "development" for asset in self.assets):
            blockers.append("assets.development_split_empty")
        for group_field in ("paper_group_id", "capture_group_id"):
            group_splits: dict[str, set[str]] = {}
            for asset in self.assets:
                group_id = getattr(asset, group_field)
                if group_id is not None and asset.split is not None:
                    group_splits.setdefault(group_id, set()).add(asset.split)
            for group_id in sorted(group_splits):
                if len(group_splits[group_id]) > 1:
                    blockers.append(
                        f"{group_field}.{group_id}.crosses_splits"
                    )
        return tuple(blockers)

    @property
    def eligible_for_local_raw_pilot(self) -> bool:
        """Contract-only eligibility; use the checker for audit-bound eligibility."""

        return not self.local_raw_pilot_blockers()

    def to_payload(self) -> dict[str, object]:
        return {
            "qualification_schema_version": self.qualification_schema_version,
            "artifact_kind": "recorded_dataset_qualification",
            "claim_eligibility": CLAIM_ELIGIBILITY,
            "qualification_id": self.qualification_id,
            "asset_root": self.asset_root,
            "purpose": self.purpose,
            "dataset_audit": self.dataset_audit.to_payload(),
            "evidence_records": [
                item.to_payload() for item in self.evidence_records
            ],
            "permissions": self.permissions.to_payload(),
            "native_interval_semantics": (
                self.native_interval_semantics.to_payload()
            ),
            "assets": [item.to_payload() for item in self.assets],
        }


@dataclass(frozen=True)
class QualificationCheckReport:
    """Hash-verified qualification result, still not sequence evidence."""

    qualification_id: str
    asset_root: str
    qualification_path: str
    qualification_sha256: str
    dataset_audit: AuditReference
    asset_count: int
    development_asset_count: int
    eligible: bool
    blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text("qualification_id", self.qualification_id)
        _require_relative_path("qualification asset_root", self.asset_root)
        _require_relative_path("qualification path", self.qualification_path)
        _require_sha256("qualification sha256", self.qualification_sha256)
        for name, value in (
            ("asset_count", self.asset_count),
            ("development_asset_count", self.development_asset_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        _require_unique_texts("qualification blockers", self.blockers)
        if self.eligible != (not self.blockers):
            raise ValueError("qualification eligible status disagrees with blockers.")

    def to_payload(self) -> dict[str, object]:
        return {
            "qualification_check_schema_version": QUALIFICATION_CHECK_SCHEMA_VERSION,
            "artifact_kind": "recorded_dataset_qualification_check",
            "claim_eligibility": CLAIM_ELIGIBILITY,
            "purpose": QUALIFICATION_PURPOSE,
            "qualification_id": self.qualification_id,
            "asset_root": self.asset_root,
            "qualification": {
                "path": self.qualification_path,
                "sha256": self.qualification_sha256,
            },
            "dataset_audit": self.dataset_audit.to_payload(),
            "asset_count": self.asset_count,
            "development_asset_count": self.development_asset_count,
            "eligible": self.eligible,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class QualificationCheckReportV2:
    """Purpose-aware v2 check result with explicit selection and operations."""

    qualification_id: str
    asset_root: str
    purpose: QualificationPurposeV2
    qualification_path: str
    qualification_sha256: str
    dataset_audit: AuditReference
    project_preregistration_id: str
    project_preregistration_path: str
    project_preregistration_sha256: str
    operations: tuple[QualificationOperation, ...]
    required_permission_scopes: tuple[str, ...]
    asset_count: int
    selected_asset_count: int
    development_asset_count: int
    held_out_asset_count: int
    excluded_asset_count: int
    eligible: bool
    blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text("qualification_id", self.qualification_id)
        _require_relative_path("qualification asset_root", self.asset_root)
        if self.purpose not in QUALIFICATION_PURPOSES_V2:
            raise ValueError("unknown qualification v2 purpose.")
        _require_relative_path("qualification path", self.qualification_path)
        _require_sha256("qualification sha256", self.qualification_sha256)
        _require_text(
            "project preregistration_id", self.project_preregistration_id
        )
        _require_relative_path(
            "project preregistration path", self.project_preregistration_path
        )
        _require_sha256(
            "project preregistration sha256",
            self.project_preregistration_sha256,
        )
        _require_unique_texts("qualification operations", self.operations)
        _require_unique_texts(
            "required permission scopes", self.required_permission_scopes
        )
        if set(self.operations) - set(QUALIFICATION_OPERATIONS):
            raise ValueError("qualification check contains unknown operations.")
        if set(self.required_permission_scopes) - set(PERMISSION_SCOPES):
            raise ValueError(
                "qualification check contains unknown permission scopes."
            )
        for name, value in (
            ("asset_count", self.asset_count),
            ("selected_asset_count", self.selected_asset_count),
            ("development_asset_count", self.development_asset_count),
            ("held_out_asset_count", self.held_out_asset_count),
            ("excluded_asset_count", self.excluded_asset_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if self.selected_asset_count != (
            self.development_asset_count + self.held_out_asset_count
        ):
            raise ValueError(
                "selected asset count must equal active split counts."
            )
        _require_unique_texts("qualification blockers", self.blockers)
        if self.eligible != (not self.blockers):
            raise ValueError("qualification eligible status disagrees with blockers.")

    def to_payload(self) -> dict[str, object]:
        return {
            "qualification_check_schema_version": (
                QUALIFICATION_CHECK_SCHEMA_V2_VERSION
            ),
            "artifact_kind": "recorded_dataset_qualification_check",
            "claim_eligibility": CLAIM_ELIGIBILITY,
            "purpose": self.purpose,
            "qualification_id": self.qualification_id,
            "asset_root": self.asset_root,
            "qualification": {
                "path": self.qualification_path,
                "sha256": self.qualification_sha256,
            },
            "dataset_audit": self.dataset_audit.to_payload(),
            "project_preregistration": {
                "preregistration_id": self.project_preregistration_id,
                "path": self.project_preregistration_path,
                "sha256": self.project_preregistration_sha256,
            },
            "operations": list(self.operations),
            "required_permission_scopes": list(
                self.required_permission_scopes
            ),
            "asset_count": self.asset_count,
            "selected_asset_count": self.selected_asset_count,
            "development_asset_count": self.development_asset_count,
            "held_out_asset_count": self.held_out_asset_count,
            "excluded_asset_count": self.excluded_asset_count,
            "eligible": self.eligible,
            "blockers": list(self.blockers),
        }


def parse_dataset_qualification(
    payload: dict[str, Any],
) -> DatasetQualification | DatasetQualificationV2:
    """Strictly parse one supported qualification artifact."""

    if "qualification_schema_version" not in payload:
        raise ValueError(
            "dataset qualification is missing keys: "
            "['qualification_schema_version']"
        )
    version = _require_int(
        "qualification_schema_version", payload["qualification_schema_version"]
    )
    if version == QUALIFICATION_SCHEMA_VERSION:
        return _parse_dataset_qualification_v1(payload)
    if version == QUALIFICATION_SCHEMA_V2_VERSION:
        return _parse_dataset_qualification_v2(payload)
    raise ValueError("unsupported qualification schema version.")


def _parse_dataset_qualification_v1(
    payload: dict[str, Any],
) -> DatasetQualification:
    _check_exact_keys(
        payload,
        {
            "qualification_schema_version",
            "artifact_kind",
            "claim_eligibility",
            "qualification_id",
            "asset_root",
            "purpose",
            "dataset_audit",
            "evidence_records",
            "permissions",
            "native_interval_semantics",
            "assets",
        },
        context="dataset qualification",
    )
    if payload["artifact_kind"] != "recorded_dataset_qualification":
        raise ValueError(
            "qualification artifact_kind must be recorded_dataset_qualification."
        )
    if payload["claim_eligibility"] != CLAIM_ELIGIBILITY:
        raise ValueError(
            "qualification claim_eligibility must be not_sequence_evidence."
        )
    evidence_payloads = _require_list(
        "evidence_records", payload["evidence_records"]
    )
    asset_payloads = _require_list("assets", payload["assets"])
    return DatasetQualification(
        qualification_schema_version=QUALIFICATION_SCHEMA_VERSION,
        qualification_id=_require_string(
            "qualification_id", payload["qualification_id"]
        ),
        asset_root=_require_string("asset_root", payload["asset_root"]),
        purpose=_require_string("purpose", payload["purpose"]),
        dataset_audit=_parse_audit_reference(
            _require_object("dataset_audit", payload["dataset_audit"])
        ),
        evidence_records=tuple(
            _parse_evidence_record(
                _require_object("evidence record", item)
            )
            for item in evidence_payloads
        ),
        permissions=_parse_permissions(
            _require_object("permissions", payload["permissions"])
        ),
        native_interval_semantics=_parse_interval_semantics(
            _require_object(
                "native_interval_semantics",
                payload["native_interval_semantics"],
            )
        ),
        assets=tuple(
            _parse_qualified_asset(
                _require_object("qualified asset", item)
            )
            for item in asset_payloads
        ),
    )


def _parse_dataset_qualification_v2(
    payload: dict[str, Any],
) -> DatasetQualificationV2:
    _check_exact_keys(
        payload,
        {
            "qualification_schema_version",
            "artifact_kind",
            "claim_eligibility",
            "qualification_id",
            "asset_root",
            "purpose",
            "dataset_audit",
            "project_preregistration",
            "evidence_records",
            "permissions",
            "native_interval_semantics",
            "provider_assets",
        },
        context="dataset qualification v2",
    )
    if payload["artifact_kind"] != "recorded_dataset_qualification":
        raise ValueError(
            "qualification artifact_kind must be recorded_dataset_qualification."
        )
    if payload["claim_eligibility"] != CLAIM_ELIGIBILITY:
        raise ValueError(
            "qualification claim_eligibility must be not_sequence_evidence."
        )
    evidence_payloads = _require_list(
        "evidence_records", payload["evidence_records"]
    )
    provider_payloads = _require_list(
        "provider_assets", payload["provider_assets"]
    )
    return DatasetQualificationV2(
        qualification_schema_version=QUALIFICATION_SCHEMA_V2_VERSION,
        qualification_id=_require_string(
            "qualification_id", payload["qualification_id"]
        ),
        asset_root=_require_string("asset_root", payload["asset_root"]),
        purpose=_require_string(
            "purpose", payload["purpose"]
        ),  # type: ignore[arg-type]
        dataset_audit=_parse_audit_reference(
            _require_object("dataset_audit", payload["dataset_audit"])
        ),
        project_preregistration=_parse_project_preregistration_reference(
            _require_object(
                "project_preregistration", payload["project_preregistration"]
            )
        ),
        evidence_records=tuple(
            _parse_evidence_record(_require_object("evidence record", item))
            for item in evidence_payloads
        ),
        permissions=_parse_permissions(
            _require_object("permissions", payload["permissions"])
        ),
        native_interval_semantics=_parse_interval_semantics(
            _require_object(
                "native_interval_semantics",
                payload["native_interval_semantics"],
            )
        ),
        provider_assets=tuple(
            _parse_provider_asset(
                _require_object("provider asset", item)
            )
            for item in provider_payloads
        ),
    )


def parse_project_preregistration(
    payload: dict[str, Any],
) -> ProjectPreregistration:
    """Strictly parse one standalone project preregistration artifact."""

    _check_exact_keys(
        payload,
        {
            "project_preregistration_schema_version",
            "artifact_kind",
            "claim_eligibility",
            "preregistration_id",
            "purpose",
            "created_at_utc",
            "dataset_audit",
            "operations",
            "assignments",
        },
        context="project preregistration",
    )
    version = _require_int(
        "project_preregistration_schema_version",
        payload["project_preregistration_schema_version"],
    )
    if version != PROJECT_PREREGISTRATION_SCHEMA_VERSION:
        raise ValueError("unsupported project preregistration schema version.")
    if payload["artifact_kind"] != "recorded_project_preregistration":
        raise ValueError(
            "project preregistration artifact_kind must be "
            "recorded_project_preregistration."
        )
    if payload["claim_eligibility"] != CLAIM_ELIGIBILITY:
        raise ValueError(
            "project preregistration claim_eligibility must be "
            "not_sequence_evidence."
        )
    assignments = _require_list("project assignments", payload["assignments"])
    return ProjectPreregistration(
        project_preregistration_schema_version=version,
        preregistration_id=_require_string(
            "preregistration_id", payload["preregistration_id"]
        ),
        purpose=_require_string(
            "project purpose", payload["purpose"]
        ),  # type: ignore[arg-type]
        created_at_utc=_require_string(
            "project created_at_utc", payload["created_at_utc"]
        ),
        dataset_audit=_parse_audit_reference(
            _require_object("dataset_audit", payload["dataset_audit"])
        ),
        operations=_parse_text_tuple(
            "project operations", payload["operations"]
        ),  # type: ignore[arg-type]
        assignments=tuple(
            _parse_project_assignment(
                _require_object("project assignment", item)
            )
            for item in assignments
        ),
    )


def load_dataset_qualification(
    path: Path,
) -> DatasetQualification | DatasetQualificationV2:
    """Load and strictly validate one qualification artifact."""

    return parse_dataset_qualification(load_json_object(Path(path)))


def load_project_preregistration(path: Path) -> ProjectPreregistration:
    """Load and strictly validate one project preregistration artifact."""

    return parse_project_preregistration(load_json_object(Path(path)))


def write_project_preregistration(
    *,
    data_root: Path,
    dataset_audit_path: Path,
    preregistration_id: str,
    purpose: QualificationPurposeV2,
    created_at_utc: str,
    operations: tuple[QualificationOperation, ...],
    development_assets: tuple[str, ...],
    held_out_assets: tuple[str, ...] = (),
    excluded_assets: tuple[tuple[str, str], ...] = (),
    output_path: Path,
) -> ProjectPreregistration:
    """Freeze an explicit project selection without decoding source media."""

    root = Path(data_root).resolve()
    audit_file = _require_existing_file_under_root(
        root, Path(dataset_audit_path), label="dataset audit"
    )
    audit = parse_dataset_audit_report(load_json_object(audit_file))
    audit_reference = AuditReference(
        dataset_id=audit.accession.dataset_id,
        accession_id=audit.accession.accession_id,
        path=audit_file.relative_to(root).as_posix(),
        sha256=file_sha256(audit_file),
    )
    audited = {asset.relative_path: asset for asset in audit.assets}
    requests: list[tuple[str, ProjectSplitRole, str | None]] = [
        (path, "development", None) for path in development_assets
    ]
    requests.extend((path, "held-out", None) for path in held_out_assets)
    requests.extend(
        (path, "excluded", reason) for path, reason in excluded_assets
    )
    request_paths = [path for path, _split, _reason in requests]
    if len(request_paths) != len(set(request_paths)):
        raise ValueError("project assignment paths must be unique.")
    missing = sorted(set(request_paths) - set(audited))
    if missing:
        raise ValueError(
            f"project assignments reference unaudited assets: {missing}"
        )
    preregistration = ProjectPreregistration(
        preregistration_id=preregistration_id,
        purpose=purpose,
        created_at_utc=created_at_utc,
        dataset_audit=audit_reference,
        operations=operations,
        assignments=tuple(
            ProjectAssetAssignment(
                relative_path=path,
                sha256=audited[path].sha256.lower(),
                split=split,
                exclusion_reason=reason,
            )
            for path, split, reason in requests
        ),
    )
    if purpose == "evidence_suite" and set(request_paths) != set(audited):
        omitted = sorted(set(audited) - set(request_paths))
        raise ValueError(
            "evidence_suite preregistration must exactly partition the "
            f"dataset audit; omitted={omitted}."
        )
    output_file = _resolve_output_file_under_root(
        root,
        Path(output_path),
        label="project preregistration output",
    )
    write_json_artifact(output_file, preregistration.to_payload())
    return preregistration


def write_dataset_qualification_template(
    *,
    data_root: Path,
    dataset_audit_path: Path,
    qualification_id: str,
    asset_root: str,
    output_path: Path,
    qualification_schema_version: int = QUALIFICATION_SCHEMA_VERSION,
    purpose: QualificationPurposeV2 | None = None,
    project_preregistration_path: Path | None = None,
) -> DatasetQualification | DatasetQualificationV2:
    """Write a pending, audit-complete template without decoding any media."""

    root = Path(data_root).resolve()
    audit_file = _require_existing_file_under_root(
        root, Path(dataset_audit_path), label="dataset audit"
    )
    audit = parse_dataset_audit_report(load_json_object(audit_file))
    _resolve_declared_directory(root, asset_root, label="qualification asset_root")
    audit_reference = AuditReference(
        dataset_id=audit.accession.dataset_id,
        accession_id=audit.accession.accession_id,
        path=audit_file.relative_to(root).as_posix(),
        sha256=file_sha256(audit_file),
    )
    unknown = PermissionDecision("unknown")
    permissions = PermissionSet(
        local_decode_analysis=unknown,
        bounded_interval_extraction=unknown,
        local_derivative_construction=unknown,
        publish_aggregate_metrics=unknown,
        publish_frames_or_clips=unknown,
        redistribution=unknown,
    )
    interval_semantics = NativeIntervalSemantics(
        status="unresolved",
        class_zero_meaning=None,
        nominal_interval_seconds=None,
        trim_start_seconds=None,
        trim_end_seconds=None,
        exceptions=(),
        evidence_ids=(),
    )
    if qualification_schema_version == QUALIFICATION_SCHEMA_VERSION:
        if purpose is not None or project_preregistration_path is not None:
            raise ValueError(
                "qualification schema v1 does not accept v2 purpose or "
                "project preregistration fields."
            )
        qualification: DatasetQualification | DatasetQualificationV2 = (
            DatasetQualification(
                qualification_id=qualification_id,
                asset_root=asset_root,
                dataset_audit=audit_reference,
                evidence_records=(),
                permissions=permissions,
                native_interval_semantics=interval_semantics,
                assets=tuple(
                    QualifiedAsset(
                        relative_path=asset.relative_path,
                        sha256=asset.sha256.lower(),
                        paper_group_id=None,
                        capture_group_id=None,
                        split=None,
                        assignment_status="unresolved",
                        assignment_evidence_ids=(),
                    )
                    for asset in audit.assets
                ),
            )
        )
    elif qualification_schema_version == QUALIFICATION_SCHEMA_V2_VERSION:
        if project_preregistration_path is None:
            raise ValueError(
                "qualification schema v2 requires a project preregistration."
            )
        preregistration_file = _require_existing_file_under_root(
            root,
            Path(project_preregistration_path),
            label="project preregistration",
        )
        preregistration = load_project_preregistration(preregistration_file)
        if preregistration_file.read_bytes() != canonical_json_bytes(
            preregistration.to_payload()
        ):
            raise ValueError(
                "project preregistration must use canonical JSON serialization."
            )
        if purpose is not None and purpose != preregistration.purpose:
            raise ValueError(
                "qualification purpose must match project preregistration."
            )
        _verify_audit_references_match(
            audit_reference,
            preregistration.dataset_audit,
            context="project preregistration",
        )
        qualification = DatasetQualificationV2(
            qualification_id=qualification_id,
            asset_root=asset_root,
            purpose=preregistration.purpose,
            dataset_audit=audit_reference,
            project_preregistration=ProjectPreregistrationReference(
                path=preregistration_file.relative_to(root).as_posix(),
                sha256=file_sha256(preregistration_file),
            ),
            evidence_records=(),
            permissions=permissions,
            native_interval_semantics=interval_semantics,
            provider_assets=tuple(
                ProviderAssetFact(
                    relative_path=asset.relative_path,
                    sha256=asset.sha256.lower(),
                    paper_group_id=None,
                    capture_group_id=None,
                    gas_identity=None,
                    fact_status="unresolved",
                    fact_evidence_ids=(),
                )
                for asset in audit.assets
            ),
        )
    else:
        raise ValueError("unsupported qualification schema version.")
    output_file = _resolve_output_file_under_root(
        root,
        Path(output_path),
        label="qualification template output",
    )
    write_json_artifact(output_file, qualification.to_payload())
    return qualification


def check_dataset_qualification(
    *,
    data_root: Path,
    qualification_path: Path,
    report_path: Path | None = None,
) -> QualificationCheckReport | QualificationCheckReportV2:
    """Verify qualification/audit binding and derive purpose eligibility."""

    root = Path(data_root).resolve()
    qualification_file = _require_existing_file_under_root(
        root, Path(qualification_path), label="dataset qualification"
    )
    qualification = load_dataset_qualification(qualification_file)
    if qualification_file.read_bytes() != canonical_json_bytes(
        qualification.to_payload()
    ):
        raise ValueError(
            "dataset qualification must use canonical JSON serialization."
        )
    asset_directory = _resolve_declared_directory(
        root,
        qualification.asset_root,
        label="qualification asset_root",
    )
    audit_file = _resolve_declared_file(
        root, qualification.dataset_audit.path, label="dataset audit"
    )
    if file_sha256(audit_file).lower() != qualification.dataset_audit.sha256.lower():
        raise ValueError("dataset qualification audit SHA-256 mismatch.")
    audit = parse_dataset_audit_report(load_json_object(audit_file))
    _verify_audit_identity(qualification, audit)
    _verify_exact_asset_coverage(qualification, audit)
    _verify_asset_files(
        root=root,
        asset_directory=asset_directory,
        qualification=qualification,
    )

    if isinstance(qualification, DatasetQualification):
        blockers = list(qualification.local_raw_pilot_blockers())
        if not audit.valid:
            blockers.append("dataset_audit.invalid")
        if audit.rights_blocked:
            blockers.append("dataset_audit.rights_blocked")
        blockers = sorted(set(blockers))
        report: QualificationCheckReport | QualificationCheckReportV2 = (
            QualificationCheckReport(
                qualification_id=qualification.qualification_id,
                asset_root=qualification.asset_root,
                qualification_path=(
                    qualification_file.relative_to(root).as_posix()
                ),
                qualification_sha256=file_sha256(qualification_file),
                dataset_audit=qualification.dataset_audit,
                asset_count=len(qualification.assets),
                development_asset_count=sum(
                    asset.split == "development"
                    for asset in qualification.assets
                ),
                eligible=not blockers,
                blockers=tuple(blockers),
            )
        )
    else:
        preregistration_file = _resolve_declared_file(
            root,
            qualification.project_preregistration.path,
            label="project preregistration",
        )
        if file_sha256(preregistration_file).lower() != (
            qualification.project_preregistration.sha256.lower()
        ):
            raise ValueError("project preregistration SHA-256 mismatch.")
        preregistration = load_project_preregistration(preregistration_file)
        if preregistration_file.read_bytes() != canonical_json_bytes(
            preregistration.to_payload()
        ):
            raise ValueError(
                "project preregistration must use canonical JSON serialization."
            )
        if preregistration.purpose != qualification.purpose:
            raise ValueError(
                "qualification purpose does not match project preregistration."
            )
        _verify_audit_references_match(
            qualification.dataset_audit,
            preregistration.dataset_audit,
            context="project preregistration",
        )
        _verify_project_assignment_coverage(preregistration, audit)
        _verify_v2_evidence_snapshots(root, qualification)
        blockers = list(qualification.purpose_blockers(preregistration))
        if not audit.valid:
            blockers.append("dataset_audit.invalid")
        if audit.rights_blocked:
            blockers.append("dataset_audit.rights_blocked")
        blockers = sorted(set(blockers))
        development_count = sum(
            item.split == "development"
            for item in preregistration.assignments
        )
        held_out_count = sum(
            item.split == "held-out" for item in preregistration.assignments
        )
        excluded_count = sum(
            item.split == "excluded" for item in preregistration.assignments
        )
        report = QualificationCheckReportV2(
            qualification_id=qualification.qualification_id,
            asset_root=qualification.asset_root,
            purpose=qualification.purpose,
            qualification_path=qualification_file.relative_to(root).as_posix(),
            qualification_sha256=file_sha256(qualification_file),
            dataset_audit=qualification.dataset_audit,
            project_preregistration_id=preregistration.preregistration_id,
            project_preregistration_path=(
                preregistration_file.relative_to(root).as_posix()
            ),
            project_preregistration_sha256=file_sha256(preregistration_file),
            operations=preregistration.operations,
            required_permission_scopes=(
                preregistration.required_permission_scopes
            ),
            asset_count=len(qualification.provider_assets),
            selected_asset_count=development_count + held_out_count,
            development_asset_count=development_count,
            held_out_asset_count=held_out_count,
            excluded_asset_count=excluded_count,
            eligible=not blockers,
            blockers=tuple(blockers),
        )
    if report_path is not None:
        report_file = _resolve_output_file_under_root(
            root,
            Path(report_path),
            label="qualification check report output",
        )
        write_json_artifact(report_file, report.to_payload())
    return report


def _verify_v2_evidence_snapshots(
    root: Path,
    qualification: DatasetQualificationV2,
) -> None:
    """Re-read every content-hashed v2 evidence snapshot under data_root."""

    for record in qualification.evidence_records:
        if record.content_sha256 is None:
            continue
        snapshot = _resolve_declared_file(
            root,
            record.reference,
            label=f"qualification evidence {record.evidence_id}",
        )
        if file_sha256(snapshot).lower() != record.content_sha256.lower():
            raise ValueError(
                "qualification evidence snapshot SHA-256 mismatch: "
                f"{record.evidence_id}"
            )


def _verify_audit_identity(
    qualification: DatasetQualification | DatasetQualificationV2,
    audit: DatasetAuditReport,
) -> None:
    reference = qualification.dataset_audit
    if (
        audit.accession.dataset_id != reference.dataset_id
        or audit.accession.accession_id != reference.accession_id
    ):
        raise ValueError(
            "dataset qualification audit accession identity mismatch."
        )


def _verify_exact_asset_coverage(
    qualification: DatasetQualification | DatasetQualificationV2,
    audit: DatasetAuditReport,
) -> None:
    audited = {asset.relative_path: asset.sha256.lower() for asset in audit.assets}
    assets = (
        qualification.assets
        if isinstance(qualification, DatasetQualification)
        else qualification.provider_assets
    )
    declared = {
        asset.relative_path: asset.sha256.lower() for asset in assets
    }
    missing = sorted(set(audited) - set(declared))
    extra = sorted(set(declared) - set(audited))
    if missing or extra:
        raise ValueError(
            "qualification assets must exactly cover the dataset audit; "
            f"missing={missing}, extra={extra}."
        )
    mismatched = sorted(
        path for path in audited if audited[path] != declared[path]
    )
    if mismatched:
        raise ValueError(
            "qualified asset SHA-256 differs from the dataset audit: "
            f"{mismatched}"
        )


def _verify_asset_files(
    *,
    root: Path,
    asset_directory: Path,
    qualification: DatasetQualification | DatasetQualificationV2,
) -> None:
    assets = (
        qualification.assets
        if isinstance(qualification, DatasetQualification)
        else qualification.provider_assets
    )
    for asset in assets:
        candidate = asset_directory.joinpath(
            *PurePosixPath(asset.relative_path).parts
        )
        asset_file = _require_existing_file_under_root(
            root,
            candidate,
            label=f"qualified asset {asset.relative_path}",
        )
        try:
            asset_file.relative_to(asset_directory)
        except ValueError as error:
            raise ValueError(
                f"qualified asset {asset.relative_path} escapes asset_root."
            ) from error
        if file_sha256(asset_file).lower() != asset.sha256.lower():
            raise ValueError(
                f"qualified asset file SHA-256 mismatch: {asset.relative_path}"
            )


def _verify_audit_references_match(
    expected: AuditReference,
    actual: AuditReference,
    *,
    context: str,
) -> None:
    if expected != actual:
        raise ValueError(
            f"{context} dataset audit reference does not match qualification."
        )


def _verify_project_assignment_coverage(
    preregistration: ProjectPreregistration,
    audit: DatasetAuditReport,
) -> None:
    audited = {
        (asset.relative_path, asset.sha256.lower()) for asset in audit.assets
    }
    assigned = {item.identity for item in preregistration.assignments}
    unknown = sorted(assigned - audited)
    if unknown:
        raise ValueError(
            "project preregistration contains unaudited asset identities: "
            f"{unknown}"
        )
    if preregistration.purpose == "evidence_suite" and assigned != audited:
        missing = sorted(audited - assigned)
        raise ValueError(
            "evidence_suite project preregistration must exactly partition "
            f"the dataset audit; missing={missing}."
        )


def validate_requested_operations(
    preregistration: ProjectPreregistration,
    requested_operations: tuple[QualificationOperation, ...],
) -> tuple[str, ...]:
    """Reject output/processing operations absent from the preregistration."""

    _require_unique_texts("requested operations", requested_operations)
    unknown = set(requested_operations) - set(QUALIFICATION_OPERATIONS)
    if unknown:
        raise ValueError(f"unknown requested operations: {sorted(unknown)}")
    undeclared = set(requested_operations) - set(preregistration.operations)
    if undeclared:
        raise ValueError(
            "requested operations were not project-preregistered: "
            f"{sorted(undeclared)}"
        )
    selected = {
        OPERATION_PERMISSION_SCOPE[operation]
        for operation in requested_operations
    }
    return tuple(scope for scope in PERMISSION_SCOPES if scope in selected)


def _parse_audit_reference(payload: dict[str, Any]) -> AuditReference:
    _check_exact_keys(
        payload,
        {"dataset_id", "accession_id", "path", "sha256"},
        context="qualification dataset_audit",
    )
    return AuditReference(
        dataset_id=_require_string("dataset_id", payload["dataset_id"]),
        accession_id=_require_string("accession_id", payload["accession_id"]),
        path=_require_string("dataset audit path", payload["path"]),
        sha256=_require_string("dataset audit sha256", payload["sha256"]),
    )


def _parse_project_preregistration_reference(
    payload: dict[str, Any],
) -> ProjectPreregistrationReference:
    _check_exact_keys(
        payload,
        {"path", "sha256"},
        context="project preregistration reference",
    )
    return ProjectPreregistrationReference(
        path=_require_string(
            "project preregistration path", payload["path"]
        ),
        sha256=_require_string(
            "project preregistration sha256", payload["sha256"]
        ),
    )


def _parse_project_assignment(
    payload: dict[str, Any],
) -> ProjectAssetAssignment:
    _check_exact_keys(
        payload,
        {"relative_path", "sha256", "split", "exclusion_reason"},
        context="project assignment",
    )
    exclusion_reason = _require_optional_string(
        "project exclusion_reason", payload["exclusion_reason"]
    )
    return ProjectAssetAssignment(
        relative_path=_require_string(
            "project assignment relative_path", payload["relative_path"]
        ),
        sha256=_require_string(
            "project assignment sha256", payload["sha256"]
        ),
        split=_require_string(
            "project assignment split", payload["split"]
        ),  # type: ignore[arg-type]
        exclusion_reason=exclusion_reason,
    )


def _parse_provider_asset(payload: dict[str, Any]) -> ProviderAssetFact:
    _check_exact_keys(
        payload,
        {
            "relative_path",
            "sha256",
            "paper_group_id",
            "capture_group_id",
            "gas_identity",
            "fact_status",
            "fact_evidence_ids",
        },
        context="provider asset",
    )
    return ProviderAssetFact(
        relative_path=_require_string(
            "provider asset relative_path", payload["relative_path"]
        ),
        sha256=_require_string("provider asset sha256", payload["sha256"]),
        paper_group_id=_require_optional_string(
            "provider paper_group_id", payload["paper_group_id"]
        ),
        capture_group_id=_require_optional_string(
            "provider capture_group_id", payload["capture_group_id"]
        ),
        gas_identity=_require_optional_string(
            "provider gas_identity", payload["gas_identity"]
        ),  # type: ignore[arg-type]
        fact_status=_require_string(
            "provider fact_status", payload["fact_status"]
        ),  # type: ignore[arg-type]
        fact_evidence_ids=_parse_text_tuple(
            "provider fact_evidence_ids", payload["fact_evidence_ids"]
        ),
    )


def _parse_evidence_record(payload: dict[str, Any]) -> EvidenceRecord:
    _check_exact_keys(
        payload,
        {
            "evidence_id",
            "kind",
            "reference",
            "observed_at_utc",
            "content_sha256",
            "summary",
        },
        context="qualification evidence record",
    )
    content_hash = payload["content_sha256"]
    if content_hash is not None and not isinstance(content_hash, str):
        raise ValueError("evidence content_sha256 must be a string or null.")
    return EvidenceRecord(
        evidence_id=_require_string("evidence_id", payload["evidence_id"]),
        kind=_require_string("evidence kind", payload["kind"]),  # type: ignore[arg-type]
        reference=_require_string("evidence reference", payload["reference"]),
        observed_at_utc=_require_string(
            "evidence observed_at_utc", payload["observed_at_utc"]
        ),
        content_sha256=content_hash,
        summary=_require_string("evidence summary", payload["summary"]),
    )


def _parse_permissions(payload: dict[str, Any]) -> PermissionSet:
    _check_exact_keys(payload, set(PERMISSION_SCOPES), context="permissions")
    decisions = {
        scope: _parse_permission_decision(
            _require_object(f"permission {scope}", payload[scope])
        )
        for scope in PERMISSION_SCOPES
    }
    return PermissionSet(**decisions)  # type: ignore[arg-type]


def _parse_permission_decision(payload: dict[str, Any]) -> PermissionDecision:
    _check_exact_keys(
        payload, {"decision", "evidence_ids"}, context="permission decision"
    )
    return PermissionDecision(
        decision=_require_string("permission decision", payload["decision"]),  # type: ignore[arg-type]
        evidence_ids=_parse_text_tuple(
            "permission evidence_ids", payload["evidence_ids"]
        ),
    )


def _parse_interval_semantics(payload: dict[str, Any]) -> NativeIntervalSemantics:
    _check_exact_keys(
        payload,
        {
            "status",
            "class_zero_meaning",
            "nominal_interval_seconds",
            "trim_start_seconds",
            "trim_end_seconds",
            "exceptions",
            "evidence_ids",
        },
        context="native interval semantics",
    )
    class_zero = payload["class_zero_meaning"]
    if class_zero is not None and not isinstance(class_zero, str):
        raise ValueError("class_zero_meaning must be a string or null.")
    exceptions = _require_list("interval exceptions", payload["exceptions"])
    return NativeIntervalSemantics(
        status=_require_string("interval status", payload["status"]),  # type: ignore[arg-type]
        class_zero_meaning=class_zero,
        nominal_interval_seconds=_require_optional_number(
            "nominal_interval_seconds", payload["nominal_interval_seconds"]
        ),
        trim_start_seconds=_require_optional_number(
            "trim_start_seconds", payload["trim_start_seconds"]
        ),
        trim_end_seconds=_require_optional_number(
            "trim_end_seconds", payload["trim_end_seconds"]
        ),
        exceptions=tuple(
            _parse_interval_exception(
                _require_object("interval exception", item)
            )
            for item in exceptions
        ),
        evidence_ids=_parse_text_tuple(
            "interval evidence_ids", payload["evidence_ids"]
        ),
    )


def _parse_interval_exception(payload: dict[str, Any]) -> IntervalException:
    _check_exact_keys(
        payload,
        {"relative_path", "description", "evidence_ids"},
        context="interval exception",
    )
    return IntervalException(
        relative_path=_require_string("relative_path", payload["relative_path"]),
        description=_require_string("description", payload["description"]),
        evidence_ids=_parse_text_tuple(
            "interval exception evidence_ids", payload["evidence_ids"]
        ),
    )


def _parse_qualified_asset(payload: dict[str, Any]) -> QualifiedAsset:
    _check_exact_keys(
        payload,
        {
            "relative_path",
            "sha256",
            "paper_group_id",
            "capture_group_id",
            "split",
            "assignment_status",
            "assignment_evidence_ids",
        },
        context="qualified asset",
    )
    paper_group = _require_optional_string(
        "paper_group_id", payload["paper_group_id"]
    )
    capture_group = _require_optional_string(
        "capture_group_id", payload["capture_group_id"]
    )
    split = _require_optional_string("split", payload["split"])
    return QualifiedAsset(
        relative_path=_require_string("relative_path", payload["relative_path"]),
        sha256=_require_string("sha256", payload["sha256"]),
        paper_group_id=paper_group,
        capture_group_id=capture_group,
        split=split,  # type: ignore[arg-type]
        assignment_status=_require_string(
            "assignment_status", payload["assignment_status"]
        ),  # type: ignore[arg-type]
        assignment_evidence_ids=_parse_text_tuple(
            "assignment_evidence_ids", payload["assignment_evidence_ids"]
        ),
    )


def _check_exact_keys(
    payload: dict[str, Any], required: set[str], *, context: str
) -> None:
    missing = required - set(payload)
    extra = set(payload) - required
    if missing:
        raise ValueError(f"{context} is missing keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"{context} has unknown keys: {sorted(extra)}")


def _require_object(name: str, value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object.")
    return value


def _require_list(name: str, value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array.")
    return value


def _require_string(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string.")
    return value


def _require_optional_string(name: str, value: object) -> str | None:
    if value is None:
        return None
    return _require_string(name, value)


def _require_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    return value


def _require_optional_number(name: str, value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number or null.")
    return float(value)


def _parse_text_tuple(name: str, value: object) -> tuple[str, ...]:
    items = _require_list(name, value)
    return tuple(_require_string(name, item) for item in items)


def _require_existing_file_under_root(
    root: Path, path: Path, *, label: str
) -> Path:
    candidate = path if path.is_absolute() else root / path
    try:
        relative = candidate.absolute().relative_to(root.absolute())
    except ValueError as error:
        raise ValueError(f"{label} must be inside data_root.") from error
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} path must not contain symlinks.")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes data_root.") from error
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    return resolved


def _resolve_declared_file(root: Path, relative_path: str, *, label: str) -> Path:
    _require_relative_path(f"{label} path", relative_path)
    return _require_existing_file_under_root(
        root,
        root.joinpath(*PurePosixPath(relative_path).parts),
        label=label,
    )


def _resolve_declared_directory(
    root: Path,
    relative_path: str,
    *,
    label: str,
) -> Path:
    _require_relative_path(f"{label} path", relative_path)
    candidate = root.joinpath(*PurePosixPath(relative_path).parts)
    try:
        relative = candidate.absolute().relative_to(root.absolute())
    except ValueError as error:
        raise ValueError(f"{label} must be inside data_root.") from error
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} path must not contain symlinks.")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes data_root.") from error
    if not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {resolved}")
    return resolved


def _resolve_output_file_under_root(
    root: Path,
    path: Path,
    *,
    label: str,
) -> Path:
    candidate = path if path.is_absolute() else root / path
    try:
        relative = candidate.absolute().relative_to(root.absolute())
    except ValueError as error:
        raise ValueError(f"{label} must be inside data_root.") from error
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} path must not contain symlinks.")
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes data_root.") from error
    return resolved
