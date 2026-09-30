"""Deterministic construction of truth-free recorded sequence inputs.

Preparation verifies audited byte identities, freezes declared leak-off
calibrations, and emits legacy schema-v3/v4 or gas-aware schema-v5 manifests.
It does not import or execute the experiment runner, policies, rendering,
pseudo truth, or full-frame planning.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Callable

import cv2
import numpy as np

from plume_focus.recorded_artifacts import write_json_artifact
from plume_focus.recorded_assets import (
    resolve_data_asset,
    resolve_data_asset_without_symlinks,
    resolve_frame_clock,
    verify_file_hash,
)
from plume_focus.recorded_calibration import write_fixed_calibration
from plume_focus.recorded_contracts import (
    FrozenCalibrationAssetSpec,
    MediaWindowSpec,
    RecordedSequenceManifest,
    SensorInputSpec,
)
from plume_focus.recorded_dataset import (
    DatasetAuditReport,
    VideoAssetAudit,
    decoded_frame_content_sha256,
    opencv_decoder_environment_sha256,
    parse_dataset_audit_report,
)
from plume_focus.recorded_manifest_io import (
    canonical_json_sha256,
    file_sha256,
    load_json_object,
    parse_recorded_sequence_manifest,
)
from plume_focus.recorded_preparation_contracts import (
    CalibrationPreparationRecord,
    CalibrationSourceSpec,
    FrozenCalibrationResult,
    HashedPreparationInput,
    PreparationProfile,
    PreparationReport,
    PreparationSpec,
    PreparedSequenceResult,
    SequencePreparationSpec,
    load_preparation_profile,
    load_preparation_spec,
    validate_frozen_calibration_binding,
)
from plume_focus.sensor_emulation import (
    FixedLeakOffCalibration,
    fit_fixed_leak_off_calibration,
)
from plume_focus.video_sources import (
    load_video_frame_window,
    load_video_frames,
)

if TYPE_CHECKING:
    from plume_focus.recorded_qualification import (
        DatasetQualification,
        DatasetQualificationV2,
        ProjectPreregistration,
        QualificationOperation,
    )


__all__ = [
    "PreparationReport",
    "PreparedSequenceResult",
    "prepare_recorded_dataset",
    "resolve_prepared_media_window",
    "validate_preparation_qualification_binding",
]


def prepare_recorded_dataset(
    *,
    data_root: Path,
    spec_path: Path,
    profile_path: Path,
    output_manifests_dir: Path,
    report_path: Path,
) -> PreparationReport:
    """Build deterministic truth-free manifests from strict preparation inputs."""

    root = Path(data_root).resolve()
    if not root.is_dir():
        raise ValueError(f"data_root is not a directory: {root}")
    spec_file = _resolve_existing_input(root, spec_path, name="spec_path")
    profile_file = _resolve_existing_input(
        root,
        profile_path,
        name="profile_path",
    )
    manifests_directory = _resolve_output_directory(
        root,
        output_manifests_dir,
    )
    if (
        manifests_directory.name != "manifests"
        or manifests_directory.parent != root
    ):
        raise ValueError(
            "output_manifests_dir must be the data_root/manifests directory."
        )
    report_file = _resolve_output_path(
        root,
        report_path,
        name="report_path",
    )

    spec_sha256 = file_sha256(spec_file)
    profile_sha256 = file_sha256(profile_file)
    preparation = load_preparation_spec(spec_file)
    profile = load_preparation_profile(profile_file)
    spec_input = HashedPreparationInput(
        path=_relative_path(root, spec_file),
        sha256=spec_sha256,
    )
    profile_input = HashedPreparationInput(
        path=_relative_path(root, profile_file),
        sha256=profile_sha256,
    )

    audit = _load_and_validate_audit(root, preparation)
    validate_preparation_qualification_binding(
        root=root,
        preparation=preparation,
        audit=audit,
        profile=profile,
    )
    audited_assets = _index_audited_assets(
        audit,
        preparation=preparation,
    )
    _validate_declared_assets(
        root,
        preparation,
        audited_assets=audited_assets,
    )
    _validate_profile_inputs(preparation, profile)

    artifact_directory = (
        root / "assets" / "prepared" / preparation.preparation_id
    )
    profile_binding_sha256 = (
        canonical_json_sha256(asdict(profile.emulation_profile))
        if profile.emulation_profile is not None
        else None
    )
    calibration_cache: dict[str, FixedLeakOffCalibration] = {}
    calibration_sources: dict[str, CalibrationSourceSpec] = {}
    results: list[PreparedSequenceResult] = []

    for sequence in preparation.sequences:
        if profile.sensor_input_kind == "raw_recorded":
            manifest = _build_raw_manifest(
                root,
                sequence,
                profile,
                audit_asset=audited_assets[
                    sequence.manifest_template.media.sha256.lower()
                ],
            )
        else:
            assert (
                profile.emulation_profile is not None
                and profile_binding_sha256 is not None
                and sequence.calibration_source is not None
            )
            source = sequence.calibration_source
            source_key = source.sha256.lower()
            previous_source = calibration_sources.setdefault(
                source_key,
                source,
            )
            if previous_source != source:
                raise ValueError(
                    "one calibration source hash must have one exact "
                    "declaration."
                )
            calibration = calibration_cache.get(source_key)
            if calibration is None:
                calibration = _fit_declared_calibration(
                    root,
                    source,
                    profile=profile,
                    audit_asset=audited_assets[source_key],
                )
                calibration_cache[source_key] = calibration
            manifest = _build_emulated_manifest(
                root=root,
                artifact_directory=artifact_directory,
                preparation=preparation,
                profile=profile,
                sequence=sequence,
                calibration=calibration,
                profile_binding_sha256=profile_binding_sha256,
                spec_input=spec_input,
                profile_input=profile_input,
            )

        payload = _json_compatible(manifest.to_payload())
        reparsed = parse_recorded_sequence_manifest(payload)
        if reparsed != manifest:
            raise RuntimeError(
                "prepared manifest changed during strict serialization."
            )
        manifest_path = manifests_directory / (
            f"{sequence.output_sequence_id}.json"
        )
        if _resolve_output_path(
            root,
            manifest_path,
            name="prepared manifest path",
        ) == report_file:
            raise ValueError(
                "report_path must be separate from prepared manifests."
            )
        write_json_artifact(manifest_path, payload)
        results.append(
            PreparedSequenceResult(
                sequence_id=manifest.sequence_id,
                path=_relative_path(root, manifest_path),
                manifest_sha256=canonical_json_sha256(payload),
            )
        )

    report = PreparationReport(
        preparation_id=preparation.preparation_id,
        preparation_spec=spec_input,
        preparation_profile=profile_input,
        producing_commit=preparation.producing_commit,
        sequences=tuple(results),
    )
    write_json_artifact(report_file, report.to_payload())
    return report


def validate_preparation_qualification_binding(
    *,
    root: Path,
    preparation: PreparationSpec,
    audit: DatasetAuditReport,
    profile: PreparationProfile | None = None,
) -> None:
    """Recheck qualification before any preparation decode occurs.

    Schema-v1/v2 preparation remains governed by its original audit-only
    contract. Schema v3 preserves the original qualification-v1 windowed path.
    Schema v4 binds an operation-aware qualification-v2 preregistration to the
    exact selected media identities, project splits, and provider capture facts.
    """

    reference = preparation.dataset_qualification
    if preparation.preparation_spec_schema_version < 3:
        if reference is not None:
            raise RuntimeError(
                "legacy preparation unexpectedly has qualification lineage."
            )
        return
    if reference is None:
        raise RuntimeError(
            "schema-v3/v4 preparation lacks qualification lineage."
        )

    # Imported only for qualified use so the legacy preparation path remains
    # independent of the optional fine-grained qualification workflow.
    from plume_focus.recorded_qualification import (
        DatasetQualification,
        DatasetQualificationV2,
        ProjectPreregistration,
        QualificationCheckReport,
        QualificationCheckReportV2,
        check_dataset_qualification,
        load_dataset_qualification,
        load_project_preregistration,
        validate_requested_operations,
    )

    qualification_path = resolve_data_asset_without_symlinks(
        root,
        reference.path,
        label="dataset qualification",
    )
    verify_file_hash(qualification_path, reference.sha256)
    qualification = load_dataset_qualification(qualification_path)
    check = check_dataset_qualification(
        data_root=root,
        qualification_path=qualification_path,
    )
    if not check.eligible:
        blockers = ", ".join(check.blockers) or "unspecified blocker"
        raise ValueError(
            "dataset qualification is not eligible for its declared purpose: "
            f"{blockers}"
        )

    qualification_audit = qualification.dataset_audit
    preparation_audit = preparation.dataset_audit
    if (
        qualification_audit.dataset_id != preparation_audit.dataset_id
        or qualification_audit.accession_id
        != preparation_audit.accession_id
        or qualification_audit.path != preparation_audit.path
        or qualification_audit.sha256.lower()
        != preparation_audit.sha256.lower()
    ):
        raise ValueError(
            "dataset qualification audit reference does not match the "
            "preparation audit reference."
        )
    if (
        audit.accession.dataset_id != qualification_audit.dataset_id
        or audit.accession.accession_id
        != qualification_audit.accession_id
    ):
        raise ValueError(
            "verified dataset audit identity does not match qualification."
        )

    if preparation.preparation_spec_schema_version == 3:
        if not isinstance(qualification, DatasetQualification) or not isinstance(
            check, QualificationCheckReport
        ):
            raise ValueError(
                "preparation schema v3 requires qualification schema v1."
            )
        _validate_v3_qualification_selection(
            preparation=preparation,
            audit=audit,
            qualification=qualification,
        )
        return

    if preparation.preparation_spec_schema_version != 4:
        raise RuntimeError("unsupported qualified preparation schema version.")
    if not isinstance(qualification, DatasetQualificationV2) or not isinstance(
        check, QualificationCheckReportV2
    ):
        raise ValueError(
            "preparation schema v4 requires qualification schema v2."
        )
    if profile is None:
        raise RuntimeError(
            "preparation schema v4 requires the resolved preparation profile."
        )
    if profile.sensor_input_kind != "raw_recorded":
        raise ValueError(
            "preparation schema v4 currently supports raw_recorded input "
            "only; calibration-source project roles are not yet modeled by "
            "qualification schema v2."
        )
    preregistration_path = resolve_data_asset_without_symlinks(
        root,
        qualification.project_preregistration.path,
        label="project preregistration",
    )
    verify_file_hash(
        preregistration_path,
        qualification.project_preregistration.sha256,
    )
    preregistration = load_project_preregistration(preregistration_path)
    _validate_v4_qualification_selection(
        preparation=preparation,
        profile=profile,
        audit=audit,
        qualification=qualification,
        preregistration=preregistration,
        validate_requested_operations=validate_requested_operations,
    )


def _validate_v3_qualification_selection(
    *,
    preparation: PreparationSpec,
    audit: DatasetAuditReport,
    qualification: DatasetQualification,
) -> None:
    """Preserve the qualification-v1 selection behavior byte-for-byte."""

    asset_root = _normalized_relative_posix_path(
        "qualification asset_root",
        qualification.asset_root,
    )
    audited_by_identity = {
        (asset.relative_path, asset.sha256.lower()): asset
        for asset in audit.assets
    }
    qualified_by_selected_identity: dict[
        tuple[str, str], QualifiedAsset
    ] = {}
    for asset in qualification.assets:
        expected_data_path = (
            asset_root / PurePosixPath(asset.relative_path)
        ).as_posix()
        selected_identity = (
            expected_data_path,
            asset.sha256.lower(),
        )
        if selected_identity in qualified_by_selected_identity:
            raise ValueError(
                "dataset qualification maps multiple assets to one exact "
                "data path and SHA-256 identity."
            )
        qualified_by_selected_identity[selected_identity] = asset

    for sequence in preparation.sequences:
        template = sequence.manifest_template
        selected_path = _normalized_relative_posix_path(
            "selected sequence media path",
            template.media.path,
        ).as_posix()
        media_hash = template.media.sha256.lower()
        qualified_asset = qualified_by_selected_identity.get(
            (selected_path, media_hash)
        )
        if qualified_asset is None:
            raise ValueError(
                "selected media path and SHA-256 must exactly match its "
                "qualification asset_root and qualified relative path."
            )
        audited_asset = audited_by_identity.get(
            (
                qualified_asset.relative_path,
                qualified_asset.sha256.lower(),
            )
        )
        if audited_asset is None:
            raise ValueError(
                "selected media qualified path and SHA-256 must exactly "
                "match the dataset audit."
            )
        capture = template.capture_group
        if capture is None:
            raise RuntimeError(
                "validated preparation template lacks a capture group."
            )
        if (
            qualified_asset.capture_group_id
            != capture.capture_group_id
            or qualified_asset.split != template.split
        ):
            raise ValueError(
                "selected media capture group and split must "
                "match its eligible dataset qualification."
            )


def _validate_v4_qualification_selection(
    *,
    preparation: PreparationSpec,
    profile: PreparationProfile,
    audit: DatasetAuditReport,
    qualification: DatasetQualificationV2,
    preregistration: ProjectPreregistration,
    validate_requested_operations: Callable[
        [ProjectPreregistration, tuple[QualificationOperation, ...]],
        tuple[str, ...],
    ],
) -> None:
    """Bind operation-aware preparation to the exact v2 preregistration."""

    actual_operations: list[QualificationOperation] = ["decode_analysis"]
    if any(
        sequence.media_window_request is not None
        for sequence in preparation.sequences
    ):
        actual_operations.append("extract_bounded_window")
    validate_requested_operations(
        preregistration,
        tuple(actual_operations),
    )

    asset_root = _normalized_relative_posix_path(
        "qualification asset_root",
        qualification.asset_root,
    )
    active_assignments = {
        item.identity: item
        for item in preregistration.active_assignments
    }
    provider_assets = {
        item.identity: item
        for item in qualification.provider_assets
    }
    audited_assets = {
        (item.relative_path, item.sha256.lower()): item
        for item in audit.assets
    }
    selected_identities: set[tuple[str, str]] = set()
    qualification_reference = preparation.dataset_qualification
    if qualification_reference is None:
        raise RuntimeError(
            "preparation schema v4 lacks qualification lineage."
        )

    for sequence in preparation.sequences:
        template = sequence.manifest_template
        if template.manifest_schema_version == 5:
            lineage = template.dataset_qualification
            if lineage is None:
                raise RuntimeError(
                    "validated schema-v5 template lacks qualification lineage."
                )
            if (
                lineage.path != qualification_reference.path
                or lineage.sha256.lower()
                != qualification_reference.sha256.lower()
                or lineage.purpose != qualification.purpose
            ):
                raise ValueError(
                    "schema-v5 qualification lineage must exactly match the "
                    "preparation qualification and declared purpose."
                )
        selected_path = _normalized_relative_posix_path(
            "selected sequence media path",
            template.media.path,
        )
        try:
            provider_path = selected_path.relative_to(asset_root).as_posix()
        except ValueError as error:
            raise ValueError(
                "selected media path must be inside the qualified asset_root."
            ) from error
        identity = (provider_path, template.media.sha256.lower())
        assignment = active_assignments.get(identity)
        if assignment is None:
            raise ValueError(
                "selected media path and SHA-256 must exactly match an active "
                "project-preregistered asset."
            )
        provider = provider_assets.get(identity)
        if provider is None or identity not in audited_assets:
            raise ValueError(
                "selected media identity must match provider facts and the "
                "dataset audit."
            )
        capture = template.capture_group
        if capture is None:
            raise RuntimeError(
                "validated preparation template lacks a capture group."
            )
        if assignment.split != template.split:
            raise ValueError(
                "selected media split must match its project preregistration."
            )
        if provider.capture_group_id != capture.capture_group_id:
            raise ValueError(
                "selected media capture group must match verified provider "
                "facts."
            )
        if template.gas_identity is not None and (
            provider.gas_identity != template.gas_identity.identity
        ):
            raise ValueError(
                "schema-v5 gas identity must match verified provider facts."
            )
        selected_identities.add(identity)

    if selected_identities != set(active_assignments):
        missing = sorted(set(active_assignments) - selected_identities)
        extra = sorted(selected_identities - set(active_assignments))
        raise ValueError(
            "preparation media must exactly cover active project assignments; "
            f"missing={missing}, extra={extra}."
        )


def _normalized_relative_posix_path(
    name: str,
    value: str,
) -> PurePosixPath:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(
            f"{name} must be a normalized relative POSIX path."
        )
    return path


def _build_raw_manifest(
    root: Path,
    sequence: SequencePreparationSpec,
    profile: PreparationProfile,
    *,
    audit_asset: VideoAssetAudit,
) -> RecordedSequenceManifest:
    if sequence.calibration_source is not None:
        raise ValueError(
            "raw preparation sequences must not declare calibration sources."
        )
    media_window = (
        resolve_prepared_media_window(
            root=root,
            sequence=sequence,
            audit_asset=audit_asset,
        )
        if sequence.media_window_request is not None
        else None
    )
    if sequence.manifest_template.manifest_schema_version == 5:
        output_schema_version = 5
    elif media_window is not None:
        output_schema_version = 4
    else:
        output_schema_version = 3
    return replace(
        sequence.manifest_template,
        sequence_id=sequence.output_sequence_id,
        manifest_schema_version=output_schema_version,
        media_window=media_window,
        sensor_input=SensorInputSpec(
            kind="raw_recorded",
            provenance=profile.sensor_input_provenance,
        ),
    )


def resolve_prepared_media_window(
    *,
    root: Path,
    sequence: SequencePreparationSpec,
    audit_asset: VideoAssetAudit,
) -> MediaWindowSpec:
    """Resolve and hash one preregistered interval against its full asset."""

    request = sequence.media_window_request
    if request is None:
        raise ValueError("sequence has no media-window request.")
    template = sequence.manifest_template
    media = template.media
    if audit_asset.sha256.lower() != media.sha256.lower():
        raise ValueError(
            "media-window audit asset does not match sequence media."
        )
    if audit_asset.status != "valid":
        raise ValueError("media-window source is not valid in its audit.")
    if audit_asset.decoded_frame_count < 3:
        raise ValueError(
            "media-window source audit must contain at least three frames."
        )
    if (
        template.frame_clock.mode == "constant_fps"
        and audit_asset.nominal_fps is None
    ):
        raise ValueError(
            "constant-FPS media-window preparation requires an audited "
            "nominal FPS."
        )

    selection_path = resolve_data_asset_without_symlinks(
        root,
        request.selection_metadata_path,
        label="media-window selection metadata",
    )
    media_path = resolve_data_asset_without_symlinks(
        root,
        media.path,
        label="media-window source",
    )
    if selection_path == media_path:
        raise ValueError(
            "media-window selection metadata must resolve to a separate "
            "asset from media."
        )
    verify_file_hash(
        selection_path,
        request.selection_metadata_sha256,
    )

    timestamps, _nominal_fps, _timestamp_hash = resolve_frame_clock(
        template.frame_clock,
        data_root=root,
        frame_count=audit_asset.decoded_frame_count,
        container_fps=(audit_asset.nominal_fps or 1.0),
        verify_hashes=True,
    )
    timestamp_values = np.asarray(timestamps, dtype=float)
    if request.requested_start_time_seconds > float(timestamp_values[-1]):
        raise ValueError(
            "media-window requested start lies beyond the final source "
            "timestamp."
        )
    if template.frame_clock.mode == "constant_fps":
        assert template.frame_clock.fps is not None
        final_interval_seconds = 1.0 / template.frame_clock.fps
    else:
        final_interval_seconds = float(
            timestamp_values[-1] - timestamp_values[-2]
        )
    capture_end_seconds = (
        float(timestamp_values[-1]) + final_interval_seconds
    )
    if (
        request.requested_end_time_seconds > capture_end_seconds
        and not np.isclose(
            request.requested_end_time_seconds,
            capture_end_seconds,
            rtol=0.0,
            atol=1e-9,
        )
    ):
        raise ValueError(
            "media-window requested end lies beyond the authoritative "
            "parent capture extent."
        )
    source_start = int(
        np.searchsorted(
            timestamp_values,
            request.requested_start_time_seconds,
            side="left",
        )
    )
    source_end = int(
        np.searchsorted(
            timestamp_values,
            request.requested_end_time_seconds,
            side="left",
        )
    )
    if (
        source_start != request.expected_source_start_frame_index
        or source_end
        != request.expected_source_end_frame_index_exclusive
    ):
        raise ValueError(
            "media-window requested times do not resolve to the "
            "preregistered source frame indices."
        )
    if source_end - source_start < 3:
        raise ValueError(
            "media-window preparation requires at least three frames."
        )
    if source_end > audit_asset.decoded_frame_count:
        raise ValueError(
            "media-window exclusive end exceeds the audited frame count."
        )

    frames, decoded_container_fps = load_video_frame_window(
        media_path,
        start_frame_index=source_start,
        end_frame_index_exclusive=source_end,
    )
    # Recheck the decoder-reported rate against the authoritative clock rather
    # than trusting only the audit's stored nominal value.
    resolve_frame_clock(
        template.frame_clock,
        data_root=root,
        frame_count=audit_asset.decoded_frame_count,
        container_fps=decoded_container_fps,
        verify_hashes=True,
    )
    if len(frames) != source_end - source_start:
        raise ValueError(
            "decoded media-window frame count differs from its resolved "
            "source interval."
        )
    if audit_asset.frame_shape is None or any(
        frame.shape != audit_asset.frame_shape for frame in frames
    ):
        raise ValueError(
            "decoded media-window shape differs from its dataset audit."
        )
    return MediaWindowSpec(
        requested_start_time_seconds=(
            request.requested_start_time_seconds
        ),
        requested_end_time_seconds=request.requested_end_time_seconds,
        source_start_frame_index=source_start,
        source_end_frame_index_exclusive=source_end,
        first_included_timestamp_seconds=float(timestamps[source_start]),
        last_included_timestamp_seconds=float(timestamps[source_end - 1]),
        parent_media_sha256=media.sha256,
        decoded_window_sha256=decoded_frame_content_sha256(frames),
        selection_metadata_path=request.selection_metadata_path,
        selection_metadata_sha256=request.selection_metadata_sha256,
        selection_provenance=request.selection_provenance,
    )


def _build_emulated_manifest(
    *,
    root: Path,
    artifact_directory: Path,
    preparation: PreparationSpec,
    profile: PreparationProfile,
    sequence: SequencePreparationSpec,
    calibration: FixedLeakOffCalibration,
    profile_binding_sha256: str,
    spec_input: HashedPreparationInput,
    profile_input: HashedPreparationInput,
) -> RecordedSequenceManifest:
    emulation_profile = profile.emulation_profile
    source = sequence.calibration_source
    assert emulation_profile is not None and source is not None

    artifact_token = hashlib.sha256(
        (
            source.sha256.lower()
            + ":"
            + profile_binding_sha256.lower()
        ).encode("ascii")
    ).hexdigest()[:24]
    artifact_name = (
        f"calibration-{artifact_token}.plmcal"
    )
    artifact_path = artifact_directory / artifact_name
    artifact_sha256 = write_fixed_calibration(
        artifact_path,
        calibration,
    )
    rows, columns = calibration.shape
    artifact_result = FrozenCalibrationResult(
        path=_relative_path(root, artifact_path),
        sha256=artifact_sha256,
        rows=rows,
        columns=columns,
        sample_count=calibration.sample_count,
    )
    calibration_record = CalibrationPreparationRecord(
        preparation_id=preparation.preparation_id,
        producing_commit=preparation.producing_commit,
        source=source,
        calibration_artifact=artifact_result,
        emulation_profile_sha256=profile_binding_sha256,
        preparation_spec=spec_input,
        preparation_profile=profile_input,
    )
    record_path = artifact_directory / (
        f"{sequence.output_sequence_id}-calibration.json"
    )
    record_sha256 = write_json_artifact(
        record_path,
        calibration_record.to_payload(),
    )
    frozen = FrozenCalibrationAssetSpec(
        path=artifact_result.path,
        sha256=artifact_result.sha256,
        source_path=source.path,
        source_sha256=source.sha256,
        source_provenance=source.provenance,
        calibration_scope=source.calibration_scope,
        producing_commit=preparation.producing_commit,
        preparation_spec_path=spec_input.path,
        preparation_spec_sha256=spec_input.sha256,
        preparation_profile_path=profile_input.path,
        preparation_profile_sha256=profile_input.sha256,
        calibration_record_path=_relative_path(root, record_path),
        calibration_record_sha256=record_sha256,
    )
    manifest = replace(
        sequence.manifest_template,
        sequence_id=sequence.output_sequence_id,
        sensor_input=SensorInputSpec(
            kind="structure_suppressed_emulation",
            provenance=profile.sensor_input_provenance,
            emulation_profile=emulation_profile,
            frozen_calibration=frozen,
        ),
    )
    validate_frozen_calibration_binding(
        preparation,
        profile,
        calibration_record,
        manifest,
        frozen,
    )
    return manifest


def _fit_declared_calibration(
    root: Path,
    source: CalibrationSourceSpec,
    *,
    profile: PreparationProfile,
    audit_asset: VideoAssetAudit,
) -> FixedLeakOffCalibration:
    emulation_profile = profile.emulation_profile
    assert emulation_profile is not None
    source_path = resolve_data_asset(root, source.path)
    if source_path.is_symlink():
        raise ValueError("calibration source must not be a symlink.")
    verify_file_hash(source_path, source.sha256)
    frames, _container_fps = load_video_frames(
        source_path,
        max_frames=None,
    )
    if len(frames) != audit_asset.decoded_frame_count:
        raise ValueError(
            "calibration decode count differs from its dataset audit."
        )
    if audit_asset.frame_shape is None or any(
        frame.shape != audit_asset.frame_shape for frame in frames
    ):
        raise ValueError(
            "calibration decoded shape differs from its dataset audit."
        )
    if (
        audit_asset.decoded_content_sha256 is None
        or decoded_frame_content_sha256(frames)
        != audit_asset.decoded_content_sha256.lower()
    ):
        raise ValueError(
            "calibration decoded content differs from its dataset audit."
        )
    return fit_fixed_leak_off_calibration(
        frames,
        profile=emulation_profile,
    )


def _load_and_validate_audit(
    root: Path,
    preparation: PreparationSpec,
) -> DatasetAuditReport:
    reference = preparation.dataset_audit
    audit_path = resolve_data_asset(root, reference.path)
    if audit_path.is_symlink():
        raise ValueError("dataset audit must not be a symlink.")
    verify_file_hash(audit_path, reference.sha256)
    audit = parse_dataset_audit_report(load_json_object(audit_path))
    if not audit.valid:
        raise ValueError("dataset audit must be valid.")
    if audit.rights_blocked:
        raise ValueError(
            "dataset rights must be verified before preparing derivatives."
        )
    if (
        audit.dataset_audit_schema_version < 2
        or audit.opencv_build_sha256 is None
    ):
        raise ValueError(
            "recorded preparation requires dataset audit schema v2 "
            "with a decoder-environment hash."
        )
    current_build_sha256 = opencv_decoder_environment_sha256()
    if (
        audit.opencv_version != cv2.__version__
        or audit.opencv_build_sha256.lower()
        != current_build_sha256.lower()
    ):
        raise ValueError(
            "preparation OpenCV decoder environment must match the dataset "
            "audit."
        )
    if (
        audit.accession.dataset_id != reference.dataset_id
        or audit.accession.accession_id != reference.accession_id
    ):
        raise ValueError(
            "dataset audit accession does not match the preparation spec."
        )
    return audit


def _index_audited_assets(
    audit: DatasetAuditReport,
    *,
    preparation: PreparationSpec,
) -> dict[str, VideoAssetAudit]:
    by_hash: dict[str, list[VideoAssetAudit]] = {}
    for asset in audit.assets:
        by_hash.setdefault(asset.sha256.lower(), []).append(asset)
    duplicates = sorted(
        digest for digest, assets in by_hash.items() if len(assets) != 1
    )
    duplicate_exclusion_groups = {
        item.capture_group_id
        for item in preparation.exclusions
        if item.reason_code == "duplicate_byte_asset"
    }
    allowed_duplicates = {
        value.lower()
        for group in preparation.capture_groups
        if group.capture_group_id in duplicate_exclusion_groups
        for value in group.asset_sha256s
    }
    unexpected_duplicates = sorted(
        set(duplicates) - allowed_duplicates
    )
    if unexpected_duplicates:
        raise ValueError(
            "dataset audit contains duplicate byte identities without a "
            "duplicate_byte_asset exclusion: "
            f"{unexpected_duplicates}"
        )
    return {
        digest: sorted(
            assets,
            key=lambda item: item.relative_path,
        )[0]
        for digest, assets in by_hash.items()
    }


def _validate_declared_assets(
    root: Path,
    preparation: PreparationSpec,
    *,
    audited_assets: dict[str, VideoAssetAudit],
) -> None:
    declared_hashes = {
        value.lower()
        for group in preparation.capture_groups
        for value in group.asset_sha256s
    }
    missing = sorted(declared_hashes - set(audited_assets))
    if missing:
        raise ValueError(
            f"capture groups contain assets absent from the audit: {missing}"
        )
    unclassified = sorted(set(audited_assets) - declared_hashes)
    if unclassified:
        raise ValueError(
            "every audited video must belong to a declared capture group; "
            "unused assets require an excluded group and objective "
            f"exclusion record: {unclassified}"
        )
    for sequence in preparation.sequences:
        media = sequence.manifest_template.media
        media_key = media.sha256.lower()
        _validate_file_identity(
            root,
            path=media.path,
            sha256=media.sha256,
            audit_asset=audited_assets[media_key],
            label="sequence media",
        )
        calibration = sequence.calibration_source
        if calibration is not None:
            _validate_file_identity(
                root,
                path=calibration.path,
                sha256=calibration.sha256,
                audit_asset=audited_assets[
                    calibration.sha256.lower()
                ],
                label="calibration source",
            )
            calibration_shape = audited_assets[
                calibration.sha256.lower()
            ].frame_shape
            media_shape = audited_assets[media_key].frame_shape
            if calibration_shape != media_shape:
                raise ValueError(
                    "calibration and sequence media audited shapes must match."
                )


def _validate_file_identity(
    root: Path,
    *,
    path: str,
    sha256: str,
    audit_asset: VideoAssetAudit,
    label: str,
) -> None:
    if audit_asset.status != "valid":
        raise ValueError(f"{label} is not valid in the dataset audit.")
    resolved = resolve_data_asset(root, path)
    if resolved.is_symlink():
        raise ValueError(f"{label} must not be a symlink.")
    verify_file_hash(resolved, sha256)


def _validate_profile_inputs(
    preparation: PreparationSpec,
    profile: PreparationProfile,
) -> None:
    if profile.sensor_input_kind == "raw_recorded":
        if any(
            sequence.calibration_source is not None
            for sequence in preparation.sequences
        ):
            raise ValueError(
                "raw preparation profile cannot consume calibration sources."
            )
        return
    if profile.sensor_input_kind != "structure_suppressed_emulation":
        raise ValueError("unknown preparation sensor input kind.")
    if profile.emulation_profile is None:
        raise RuntimeError(
            "validated emulated profile has no emulation settings."
        )
    if any(
        sequence.media_window_request is not None
        for sequence in preparation.sequences
    ):
        raise ValueError(
            "windowed preparation does not support sensor emulation."
        )
    if any(
        sequence.calibration_source is None
        for sequence in preparation.sequences
    ):
        raise ValueError(
            "emulated preparation requires one calibration source per "
            "sequence."
        )
    if any(
        sequence.manifest_template.normalization.mode != "uint8_identity"
        for sequence in preparation.sequences
    ):
        raise ValueError(
            "emulated preparation requires uint8_identity normalization."
        )


def _resolve_existing_input(
    root: Path,
    path: Path,
    *,
    name: str,
) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise ValueError(f"{name} must not be a symlink.")
    resolved = candidate.resolve()
    _require_within_root(root, resolved, name=name)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _resolve_output_directory(root: Path, path: Path) -> Path:
    resolved = Path(path).resolve()
    _require_within_root(root, resolved, name="output_manifests_dir")
    if resolved.exists() and not resolved.is_dir():
        raise ValueError("output_manifests_dir must be a directory.")
    return resolved


def _resolve_output_path(
    root: Path,
    path: Path,
    *,
    name: str,
) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise ValueError(f"{name} must not be a symlink.")
    resolved = candidate.resolve()
    _require_within_root(root, resolved, name=name)
    if resolved.exists() and not resolved.is_file():
        raise ValueError(f"{name} must be a file path.")
    return resolved


def _require_within_root(root: Path, path: Path, *, name: str) -> None:
    if path != root and root not in path.parents:
        raise ValueError(f"{name} must remain within data_root.")


def _relative_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _json_compatible(
    payload: dict[str, object],
) -> dict[str, object]:
    """Resolve tuples and other JSON-compatible containers exactly as written."""

    value = json.loads(
        json.dumps(payload, ensure_ascii=False, allow_nan=False)
    )
    if not isinstance(value, dict):
        raise RuntimeError("recorded manifest did not serialize as an object.")
    return value
