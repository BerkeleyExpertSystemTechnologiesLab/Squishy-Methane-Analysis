"""Deterministic freezing of prepared recorded-sequence suites.

The existing suite manifest remains intentionally small and backwards
compatible.  A separate, canonical freeze record binds that manifest to the
non-evidence preparation report that produced its sequence manifests.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any

from plume_focus.recorded_artifacts import (
    canonical_json_bytes,
    write_json_artifact,
)
from plume_focus.recorded_calibration import load_fixed_calibration
from plume_focus.recorded_contracts import (
    RecordedSequenceManifest,
    RecordedSplit,
    RecordedSuiteManifest,
    SensorInputSpec,
    SequenceReference,
)
from plume_focus.recorded_dataset import (
    DatasetAuditReport,
    VideoAssetAudit,
    parse_dataset_audit_report,
)
from plume_focus.recorded_manifest_io import (
    canonical_json_sha256,
    file_sha256,
    load_json_object,
    load_recorded_suite_manifest,
    parse_recorded_sequence_manifest,
)
from plume_focus.recorded_preparation_contracts import (
    HashedPreparationInput,
    PreparationProfile,
    PreparationReport,
    PreparationSpec,
    PreparedSequenceResult,
    SequencePreparationSpec,
    load_calibration_preparation_record,
    load_preparation_profile,
    load_preparation_report,
    load_preparation_spec,
    validate_frozen_calibration_binding,
)
from plume_focus.recorded_preparation import (
    resolve_prepared_media_window,
    validate_preparation_qualification_binding,
)


SUITE_FREEZE_RECORD_SCHEMA_VERSION = 1
SUITE_FREEZE_RECORD_ARTIFACT_KIND = "recorded_suite_freeze_record"
NON_EVIDENCE_CLAIM = "not_sequence_evidence"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class FrozenSuiteArtifacts:
    """Paths and hashes published by one freeze operation."""

    suite_manifest_path: Path
    suite_artifact_sha256: str
    suite_manifest_sha256: str
    freeze_record_path: Path
    freeze_record_sha256: str
    preparation_report_sha256: str
    sequence_count: int


def freeze_recorded_suite(
    *,
    data_root: Path,
    preparation_report_path: Path,
    preparation_report_sha256: str,
    suite_id: str,
    split: RecordedSplit,
    frozen_at_utc: str,
    frozen_before_model_commit: str,
    freeze_provenance: str,
    suite_output_path: Path,
    freeze_record_output_path: Path,
) -> FrozenSuiteArtifacts:
    """Validate preparation lineage and publish one immutable suite.

    All paths embedded in a preparation report are relative to ``data_root``.
    The function has no clock or Git lookup: the freeze timestamp and full
    pre-change commit are required explicit inputs.
    """

    root = Path(data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    _require_identifier("suite_id", suite_id)
    if split not in {"development", "held-out"}:
        raise ValueError("split must be development or held-out.")
    _require_utc_timestamp("frozen_at_utc", frozen_at_utc)
    _require_full_git_commit(
        "frozen_before_model_commit",
        frozen_before_model_commit,
    )
    if frozen_before_model_commit != frozen_before_model_commit.lower():
        raise ValueError(
            "frozen_before_model_commit must use lowercase hexadecimal."
        )
    _require_text("freeze_provenance", freeze_provenance)
    _require_sha256(
        "preparation_report_sha256",
        preparation_report_sha256,
    )

    report_path = _resolve_existing_under_root(
        root,
        preparation_report_path,
        label="preparation report",
    )
    actual_report_sha256 = file_sha256(report_path)
    if actual_report_sha256 != preparation_report_sha256.lower():
        raise ValueError("preparation report SHA-256 mismatch.")
    report = load_preparation_report(report_path)

    preparation, profile, audit = _load_and_verify_preparation_inputs(
        root,
        report,
    )
    prepared_sequences = _verify_prepared_sequences(
        root,
        report,
        preparation,
        profile,
        audit,
    )
    selected = [
        item
        for item in prepared_sequences
        if item.manifest.split == split
    ]
    if not selected:
        raise ValueError(
            "preparation report contains no sequences for the requested "
            "suite split."
        )
    selected.sort(key=lambda item: item.reference.sequence_id)

    suite_path = _resolve_output_under_root(root, suite_output_path)
    freeze_record_path = _resolve_output_under_root(
        root,
        freeze_record_output_path,
    )
    if suite_path == freeze_record_path:
        raise ValueError(
            "suite and freeze-record outputs must be different files."
        )

    suite_references = tuple(
        SequenceReference(
            path=_relative_path(
                suite_path.parent,
                item.path,
                label="suite sequence manifest",
            ),
            manifest_sha256=item.reference.manifest_sha256.lower(),
        )
        for item in selected
    )
    suite = RecordedSuiteManifest(
        suite_id=suite_id,
        split=split,
        frozen_at_utc=frozen_at_utc,
        frozen_before_model_commit=(
            frozen_before_model_commit.lower()
        ),
        freeze_provenance=freeze_provenance,
        sequences=suite_references,
    )
    suite_payload = _suite_payload(suite)
    suite_content = canonical_json_bytes(suite_payload)
    suite_artifact_sha256 = hashlib.sha256(suite_content).hexdigest()
    suite_manifest_sha256 = canonical_json_sha256(suite_payload)

    freeze_record_payload: dict[str, object] = {
        "freeze_record_schema_version": (
            SUITE_FREEZE_RECORD_SCHEMA_VERSION
        ),
        "artifact_kind": SUITE_FREEZE_RECORD_ARTIFACT_KIND,
        "claim_eligibility": NON_EVIDENCE_CLAIM,
        "path_base": "recorded_data_root",
        "suite_id": suite_id,
        "split": split,
        "frozen_at_utc": frozen_at_utc,
        "frozen_before_model_commit": (
            frozen_before_model_commit.lower()
        ),
        "freeze_provenance": freeze_provenance,
        "preparation_report": {
            "path": _root_relative_path(root, report_path),
            "sha256": actual_report_sha256,
        },
        "suite_manifest": {
            "path": _root_relative_path(root, suite_path),
            "sha256": suite_artifact_sha256,
            "canonical_json_sha256": suite_manifest_sha256,
        },
    }
    freeze_record_content = canonical_json_bytes(freeze_record_payload)
    _preflight_output(suite_path, suite_content)
    _preflight_output(freeze_record_path, freeze_record_content)

    written_suite_sha256 = write_json_artifact(
        suite_path,
        suite_payload,
    )
    written_record_sha256 = write_json_artifact(
        freeze_record_path,
        freeze_record_payload,
    )
    return FrozenSuiteArtifacts(
        suite_manifest_path=suite_path,
        suite_artifact_sha256=written_suite_sha256,
        suite_manifest_sha256=suite_manifest_sha256,
        freeze_record_path=freeze_record_path,
        freeze_record_sha256=written_record_sha256,
        preparation_report_sha256=actual_report_sha256,
        sequence_count=len(selected),
    )


def validate_recorded_freeze_record(
    data_root: Path,
    freeze_record_path: Path,
) -> dict[str, object]:
    """Verify a frozen suite and its complete preparation lineage.

    This is intentionally stricter than validating the suite alone.  It binds
    the canonical freeze record to the exact suite bytes, logical suite
    payload, preparation report, preparation inputs, prepared manifests, and
    split metadata that were checked when the suite was first published.
    """

    root = Path(data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    record_path = _resolve_existing_under_root(
        root,
        freeze_record_path,
        label="suite freeze record",
    )
    payload = load_json_object(record_path)
    _check_exact_keys(
        payload,
        {
            "freeze_record_schema_version",
            "artifact_kind",
            "claim_eligibility",
            "path_base",
            "suite_id",
            "split",
            "frozen_at_utc",
            "frozen_before_model_commit",
            "freeze_provenance",
            "preparation_report",
            "suite_manifest",
        },
        context="suite freeze record",
    )
    if record_path.read_bytes() != canonical_json_bytes(payload):
        raise ValueError("suite freeze record must use canonical JSON bytes.")

    version = _require_integer_value(
        "freeze_record_schema_version",
        payload["freeze_record_schema_version"],
    )
    if version != SUITE_FREEZE_RECORD_SCHEMA_VERSION:
        raise ValueError("unsupported suite freeze record schema version.")
    if (
        _require_string_value("artifact_kind", payload["artifact_kind"])
        != SUITE_FREEZE_RECORD_ARTIFACT_KIND
    ):
        raise ValueError("unknown suite freeze record artifact kind.")
    if (
        _require_string_value(
            "claim_eligibility",
            payload["claim_eligibility"],
        )
        != NON_EVIDENCE_CLAIM
    ):
        raise ValueError(
            "suite freeze records must remain not_sequence_evidence."
        )
    if (
        _require_string_value("path_base", payload["path_base"])
        != "recorded_data_root"
    ):
        raise ValueError(
            "suite freeze record path_base must be recorded_data_root."
        )

    suite_id = _require_string_value("suite_id", payload["suite_id"])
    _require_identifier("suite_id", suite_id)
    split = _require_string_value("split", payload["split"])
    if split not in {"development", "held-out"}:
        raise ValueError("split must be development or held-out.")
    frozen_at_utc = _require_string_value(
        "frozen_at_utc",
        payload["frozen_at_utc"],
    )
    _require_utc_timestamp("frozen_at_utc", frozen_at_utc)
    frozen_before_model_commit = _require_string_value(
        "frozen_before_model_commit",
        payload["frozen_before_model_commit"],
    )
    _require_full_git_commit(
        "frozen_before_model_commit",
        frozen_before_model_commit,
    )
    if frozen_before_model_commit != frozen_before_model_commit.lower():
        raise ValueError(
            "frozen_before_model_commit must use lowercase hexadecimal."
        )
    freeze_provenance = _require_string_value(
        "freeze_provenance",
        payload["freeze_provenance"],
    )
    _require_text("freeze_provenance", freeze_provenance)

    report_reference = _require_object_value(
        "preparation_report",
        payload["preparation_report"],
    )
    _check_exact_keys(
        report_reference,
        {"path", "sha256"},
        context="freeze preparation report reference",
    )
    report_relative_path = _require_relative_value(
        "preparation report path",
        report_reference["path"],
    )
    report_sha256 = _require_lower_sha256_value(
        "preparation report sha256",
        report_reference["sha256"],
    )

    suite_reference = _require_object_value(
        "suite_manifest",
        payload["suite_manifest"],
    )
    _check_exact_keys(
        suite_reference,
        {"path", "sha256", "canonical_json_sha256"},
        context="freeze suite manifest reference",
    )
    suite_relative_path = _require_relative_value(
        "suite manifest path",
        suite_reference["path"],
    )
    suite_artifact_sha256 = _require_lower_sha256_value(
        "suite manifest sha256",
        suite_reference["sha256"],
    )
    suite_manifest_sha256 = _require_lower_sha256_value(
        "suite manifest canonical_json_sha256",
        suite_reference["canonical_json_sha256"],
    )

    report_path = _resolve_existing_under_root(
        root,
        Path(report_relative_path),
        label="preparation report",
    )
    if file_sha256(report_path) != report_sha256:
        raise ValueError("preparation report SHA-256 mismatch.")
    report = load_preparation_report(report_path)
    preparation, profile, audit = _load_and_verify_preparation_inputs(
        root,
        report,
    )
    prepared_sequences = _verify_prepared_sequences(
        root,
        report,
        preparation,
        profile,
        audit,
    )

    suite_path = _resolve_existing_under_root(
        root,
        Path(suite_relative_path),
        label="suite manifest",
    )
    if file_sha256(suite_path) != suite_artifact_sha256:
        raise ValueError("suite manifest artifact SHA-256 mismatch.")
    suite_payload = load_json_object(suite_path)
    if suite_path.read_bytes() != canonical_json_bytes(suite_payload):
        raise ValueError("frozen suite manifest must use canonical JSON bytes.")
    if canonical_json_sha256(suite_payload) != suite_manifest_sha256:
        raise ValueError("suite manifest canonical JSON SHA-256 mismatch.")
    suite = load_recorded_suite_manifest(suite_path)

    if (
        suite.suite_id != suite_id
        or suite.split != split
        or suite.frozen_at_utc != frozen_at_utc
        or suite.frozen_before_model_commit.lower()
        != frozen_before_model_commit.lower()
        or suite.freeze_provenance != freeze_provenance
    ):
        raise ValueError(
            "suite metadata does not match its freeze record."
        )

    selected = [
        item
        for item in prepared_sequences
        if item.manifest.split == split
    ]
    selected.sort(key=lambda item: item.reference.sequence_id)
    expected_references = tuple(
        SequenceReference(
            path=_relative_path(
                suite_path.parent,
                item.path,
                label="suite sequence manifest",
            ),
            manifest_sha256=item.reference.manifest_sha256.lower(),
        )
        for item in selected
    )
    if not expected_references:
        raise ValueError(
            "preparation report contains no sequences for the frozen split."
        )
    if suite.sequences != expected_references:
        raise ValueError(
            "suite sequence references do not match the preparation report."
        )

    return {
        "valid": True,
        "freeze_record_path": str(record_path),
        "freeze_record_sha256": file_sha256(record_path),
        "suite_manifest_path": str(suite_path),
        "suite_artifact_sha256": suite_artifact_sha256,
        "suite_manifest_sha256": suite_manifest_sha256,
        "preparation_report_path": str(report_path),
        "preparation_report_sha256": report_sha256,
        "suite_id": suite_id,
        "split": split,
        "sequence_count": len(expected_references),
        "preparation_lineage_verified": True,
        "claim_eligibility": NON_EVIDENCE_CLAIM,
    }


@dataclass(frozen=True)
class _VerifiedPreparedSequence:
    reference: PreparedSequenceResult
    path: Path
    manifest: RecordedSequenceManifest


def _load_and_verify_preparation_inputs(
    data_root: Path,
    report: PreparationReport,
) -> tuple[PreparationSpec, PreparationProfile, DatasetAuditReport]:
    spec_path = _verify_artifact_reference(
        data_root,
        report.preparation_spec,
        label="preparation spec",
    )
    profile_path = _verify_artifact_reference(
        data_root,
        report.preparation_profile,
        label="preparation profile",
    )
    preparation = load_preparation_spec(spec_path)
    profile = load_preparation_profile(profile_path)
    if preparation.preparation_id != report.preparation_id:
        raise ValueError(
            "preparation report preparation_id does not match its spec."
        )
    if (
        preparation.producing_commit.lower()
        != report.producing_commit.lower()
    ):
        raise ValueError(
            "preparation report producing_commit does not match its spec."
        )
    audit = _load_and_verify_dataset_audit(data_root, preparation)
    validate_preparation_qualification_binding(
        root=data_root,
        preparation=preparation,
        audit=audit,
        profile=profile,
    )
    _verify_preparation_source_assets(
        data_root,
        preparation,
        audit,
    )
    return preparation, profile, audit


def _load_and_verify_dataset_audit(
    data_root: Path,
    preparation: PreparationSpec,
) -> DatasetAuditReport:
    reference = preparation.dataset_audit
    audit_path = _resolve_existing_under_root(
        data_root,
        Path(reference.path),
        label="dataset audit",
    )
    if file_sha256(audit_path) != reference.sha256.lower():
        raise ValueError("dataset audit SHA-256 mismatch.")
    audit = parse_dataset_audit_report(load_json_object(audit_path))
    if (
        audit.accession.dataset_id != reference.dataset_id
        or audit.accession.accession_id != reference.accession_id
    ):
        raise ValueError(
            "dataset audit accession does not match the preparation spec."
        )
    if not audit.valid:
        raise ValueError("dataset audit must be valid.")
    if audit.rights_blocked:
        raise ValueError(
            "dataset rights must be verified before freezing derivatives."
        )
    declared_hashes = {
        value.lower()
        for group in preparation.capture_groups
        for value in group.asset_sha256s
    }
    audited_hashes = {
        asset.sha256.lower()
        for asset in audit.assets
    }
    if declared_hashes != audited_hashes:
        raise ValueError(
            "preparation capture groups must exactly cover the dataset "
            "audit's byte identities."
        )
    return audit


def _verify_preparation_source_assets(
    data_root: Path,
    preparation: PreparationSpec,
    audit: DatasetAuditReport,
) -> None:
    audited_hashes = {
        asset.sha256.lower()
        for asset in audit.assets
        if asset.status == "valid"
    }
    verified: set[tuple[str, str]] = set()
    for sequence in preparation.sequences:
        references = [
            (
                sequence.manifest_template.media.path,
                sequence.manifest_template.media.sha256,
                "sequence media",
            )
        ]
        if sequence.calibration_source is not None:
            references.append(
                (
                    sequence.calibration_source.path,
                    sequence.calibration_source.sha256,
                    "calibration source",
                )
            )
        for path, expected_sha256, label in references:
            identity = (path, expected_sha256.lower())
            if identity in verified:
                continue
            if expected_sha256.lower() not in audited_hashes:
                raise ValueError(
                    f"{label} is absent or invalid in the dataset audit."
                )
            asset_path = _resolve_existing_under_root(
                data_root,
                Path(path),
                label=label,
            )
            if file_sha256(asset_path) != expected_sha256.lower():
                raise ValueError(f"{label} SHA-256 mismatch.")
            verified.add(identity)


def _verify_prepared_sequences(
    data_root: Path,
    report: PreparationReport,
    preparation: PreparationSpec,
    profile: PreparationProfile,
    audit: DatasetAuditReport,
) -> list[_VerifiedPreparedSequence]:
    spec_by_id = {
        item.output_sequence_id: item
        for item in preparation.sequences
    }
    report_ids = {item.sequence_id for item in report.sequences}
    if report_ids != set(spec_by_id):
        raise ValueError(
            "preparation report sequences must exactly match its spec."
        )

    verified: list[_VerifiedPreparedSequence] = []
    for reference in report.sequences:
        path = _resolve_existing_under_root(
            data_root,
            Path(reference.path),
            label="prepared sequence manifest",
        )
        payload = load_json_object(path)
        actual_manifest_hash = canonical_json_sha256(payload)
        if actual_manifest_hash != reference.manifest_sha256.lower():
            raise ValueError(
                f"prepared sequence manifest hash mismatch: {path}"
            )
        manifest = parse_recorded_sequence_manifest(payload)
        if manifest.manifest_schema_version not in {3, 4, 5}:
            raise ValueError(
                "suite freezing accepts schema-v3, schema-v4, or schema-v5 "
                "sequences only."
            )
        if manifest.sequence_id != reference.sequence_id:
            raise ValueError(
                "preparation report sequence_id does not match its "
                "manifest."
            )
        _verify_sequence_binding(
            data_root,
            report,
            spec_by_id[reference.sequence_id],
            preparation,
            profile,
            manifest,
            audit,
        )
        verified.append(
            _VerifiedPreparedSequence(
                reference=reference,
                path=path,
                manifest=manifest,
            )
        )
    return verified


def _verify_sequence_binding(
    data_root: Path,
    report: PreparationReport,
    sequence_spec: SequencePreparationSpec,
    preparation: PreparationSpec,
    profile: PreparationProfile,
    manifest: RecordedSequenceManifest,
    audit: DatasetAuditReport,
) -> None:
    if profile.sensor_input_kind == "raw_recorded":
        media_window = None
        gas_aware = (
            sequence_spec.manifest_template.manifest_schema_version == 5
        )
        manifest_schema_version = 5 if gas_aware else 3
        if sequence_spec.media_window_request is not None:
            audit_asset = _unique_audit_asset(
                audit,
                sequence_spec.manifest_template.media.sha256,
            )
            media_window = resolve_prepared_media_window(
                root=data_root,
                sequence=sequence_spec,
                audit_asset=audit_asset,
            )
            if not gas_aware:
                manifest_schema_version = 4
        expected = replace(
            sequence_spec.manifest_template,
            sequence_id=sequence_spec.output_sequence_id,
            manifest_schema_version=manifest_schema_version,
            media_window=media_window,
            sensor_input=SensorInputSpec(
                kind="raw_recorded",
                provenance=profile.sensor_input_provenance,
            ),
        )
        if manifest != expected:
            raise ValueError(
                "prepared raw manifest differs from its immutable "
                "preparation template."
            )
        return

    sensor_input = manifest.sensor_input
    if sensor_input is None or sensor_input.frozen_calibration is None:
        raise ValueError(
            "prepared emulated manifest lacks frozen calibration."
        )
    frozen = sensor_input.frozen_calibration
    if (
        frozen.preparation_spec_path
        != report.preparation_spec.path
        or frozen.preparation_spec_sha256.lower()
        != report.preparation_spec.sha256.lower()
        or frozen.preparation_profile_path
        != report.preparation_profile.path
        or frozen.preparation_profile_sha256.lower()
        != report.preparation_profile.sha256.lower()
    ):
        raise ValueError(
            "prepared emulated manifest does not bind the preparation "
            "report's exact spec and profile."
        )
    calibration_source_path = _resolve_existing_under_root(
        data_root,
        Path(frozen.source_path),
        label="frozen calibration source",
    )
    if (
        file_sha256(calibration_source_path)
        != frozen.source_sha256.lower()
    ):
        raise ValueError("frozen calibration source SHA-256 mismatch.")
    calibration_artifact_path = _resolve_existing_under_root(
        data_root,
        Path(frozen.path),
        label="frozen calibration artifact",
    )
    resolved_calibration = load_fixed_calibration(
        calibration_artifact_path,
        expected_sha256=frozen.sha256,
    )
    record_path = _resolve_existing_under_root(
        data_root,
        Path(frozen.calibration_record_path),
        label="calibration preparation record",
    )
    if file_sha256(record_path) != frozen.calibration_record_sha256.lower():
        raise ValueError(
            "calibration preparation record SHA-256 mismatch."
        )
    calibration_record = load_calibration_preparation_record(record_path)
    if (
        resolved_calibration.shape
        != (
            calibration_record.calibration_artifact.rows,
            calibration_record.calibration_artifact.columns,
        )
        or resolved_calibration.sample_count
        != calibration_record.calibration_artifact.sample_count
    ):
        raise ValueError(
            "frozen calibration bytes do not match their preparation "
            "record metadata."
        )
    validate_frozen_calibration_binding(
        preparation,
        profile,
        calibration_record,
        manifest,
        frozen,
    )
    expected = replace(
        sequence_spec.manifest_template,
        sequence_id=sequence_spec.output_sequence_id,
        sensor_input=manifest.sensor_input,
    )
    if manifest != expected:
        raise ValueError(
            "prepared emulated manifest differs from its immutable "
            "preparation template outside the declared sensor input."
        )


def _unique_audit_asset(
    audit: DatasetAuditReport,
    sha256: str,
) -> VideoAssetAudit:
    matches = [
        asset
        for asset in audit.assets
        if asset.sha256.lower() == sha256.lower()
    ]
    if len(matches) != 1:
        raise ValueError(
            "windowed sequence media must match exactly one audited asset."
        )
    return matches[0]


def _suite_payload(
    suite: RecordedSuiteManifest,
) -> dict[str, object]:
    return {
        "suite_schema_version": suite.suite_schema_version,
        "suite_id": suite.suite_id,
        "split": suite.split,
        "frozen_at_utc": suite.frozen_at_utc,
        "frozen_before_model_commit": (
            suite.frozen_before_model_commit
        ),
        "freeze_provenance": suite.freeze_provenance,
        "sequences": [
            {
                "path": reference.path,
                "manifest_sha256": (
                    reference.manifest_sha256.lower()
                ),
            }
            for reference in suite.sequences
        ],
    }


def _verify_artifact_reference(
    data_root: Path,
    reference: HashedPreparationInput,
    *,
    label: str,
) -> Path:
    path = _resolve_existing_under_root(
        data_root,
        Path(reference.path),
        label=label,
    )
    if file_sha256(path) != reference.sha256.lower():
        raise ValueError(f"{label} SHA-256 mismatch.")
    return path


def _resolve_existing_under_root(
    data_root: Path,
    path: Path,
    *,
    label: str,
) -> Path:
    lexical = _lexical_under_root(data_root, path)
    _reject_symlink_components(data_root, lexical, label=label)
    candidate = lexical.resolve()
    if candidate != data_root and data_root not in candidate.parents:
        raise ValueError("recorded freeze path escapes the data root.")
    if not candidate.is_file():
        raise FileNotFoundError(f"{label}: {candidate}")
    return candidate


def _resolve_output_under_root(data_root: Path, path: Path) -> Path:
    candidate = _lexical_under_root(data_root, path)
    if candidate == data_root:
        raise ValueError("recorded artifact output must be a file path.")
    _reject_symlink_components(
        data_root,
        candidate.parent,
        label="recorded artifact output parent",
    )
    return candidate


def _lexical_under_root(data_root: Path, path: Path) -> Path:
    value = Path(path)
    unresolved = value if value.is_absolute() else data_root / value
    candidate = Path(os.path.abspath(unresolved))
    if candidate != data_root and data_root not in candidate.parents:
        raise ValueError("recorded freeze path escapes the data root.")
    return candidate


def _reject_symlink_components(
    data_root: Path,
    path: Path,
    *,
    label: str,
) -> None:
    current = data_root
    for component in path.relative_to(data_root).parts:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{label} must not use symlink paths.")


def _root_relative_path(data_root: Path, path: Path) -> str:
    return path.relative_to(data_root).as_posix()


def _relative_path(base: Path, path: Path, *, label: str) -> str:
    try:
        relative = path.relative_to(base)
    except ValueError as error:
        raise ValueError(
            f"{label} must be below the suite manifest directory."
        ) from error
    if not relative.parts:
        raise ValueError(f"{label} must reference a different file.")
    return relative.as_posix()


def _preflight_output(path: Path, expected: bytes) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink():
        raise ValueError("recorded artifact output must not be a symlink.")
    if not path.is_file():
        raise FileExistsError(
            f"recorded artifact output is not a regular file: {path}"
        )
    if path.read_bytes() != expected:
        raise FileExistsError(
            "refusing to overwrite a different recorded artifact: "
            f"{path}"
        )


def _require_text(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be empty.")


def _require_identifier(name: str, value: str) -> None:
    _require_text(name, value)
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(
            f"{name} must use letters, digits, '.', '_', or '-'."
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


def _check_exact_keys(
    payload: dict[str, Any],
    required: set[str],
    *,
    context: str,
) -> None:
    keys = set(payload)
    if keys != required:
        missing = sorted(required - keys)
        unknown = sorted(keys - required)
        raise ValueError(
            f"{context} keys do not match the contract; "
            f"missing={missing}, unknown={unknown}."
        )


def _require_object_value(
    name: str,
    value: object,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings.")
    return value


def _require_string_value(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string.")
    _require_text(name, value)
    return value


def _require_integer_value(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    return value


def _require_lower_sha256_value(name: str, value: object) -> str:
    sha256 = _require_string_value(name, value)
    _require_sha256(name, sha256)
    if sha256 != sha256.lower():
        raise ValueError(f"{name} must use lowercase hexadecimal.")
    return sha256


def _require_relative_value(name: str, value: object) -> str:
    path = _require_string_value(name, value)
    if "\\" in path:
        raise ValueError(f"{name} must use platform-neutral '/' separators.")
    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{name} must be confined and relative.")
    if relative.as_posix() != path or path == ".":
        raise ValueError(f"{name} must use canonical relative syntax.")
    return path
