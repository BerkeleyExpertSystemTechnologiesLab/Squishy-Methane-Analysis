"""Public loading façade for provenance-checked recorded sequences."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path, PurePosixPath
from typing import Iterable

import numpy as np

from plume_focus.experiment import TimestampedArrayTimeline
from plume_focus.model_profiles import uncalibrated_video_v1
from plume_focus.recorded_assets import (
    data_root_for_manifest,
    load_motion_provider,
    normalize_frame,
    resolve_data_asset,
    resolve_data_asset_without_symlinks,
    resolve_frame_clock,
    resolve_manifest_reference,
    resolve_source_assumption,
    verify_file_hash,
)
from plume_focus.recorded_calibration import load_fixed_calibration
from plume_focus.recorded_contracts import (
    EvaluationTruthSource,
    MediaWindowSpec,
    RecordedSequence,
    RecordedSequenceManifest,
    RecordedSequenceProvenance,
    assess_recorded_claim,
)
from plume_focus.recorded_dataset import (
    DatasetAuditReport,
    decoded_frame_content_sha256,
    parse_dataset_audit_report,
)
from plume_focus.recorded_manifest_io import (
    canonical_json_sha256,
    file_sha256,
    load_json_object,
    load_recorded_suite_manifest,
    parse_recorded_sequence_manifest,
)
from plume_focus.recorded_providers import (
    ConstantMotionProvider,
    DenseTruthTimeline,
    MissingMotionProvider,
    TimeSeriesMotionProvider,
)
from plume_focus.recorded_preparation_contracts import (
    load_calibration_preparation_record,
    load_preparation_profile,
    load_preparation_spec,
    validate_frozen_calibration_binding,
)
from plume_focus.recorded_role_isolation import (
    validate_cross_split_asset_roles,
)
from plume_focus.recorded_qualification import (
    DatasetQualificationV2,
    QualificationCheckReportV2,
    check_dataset_qualification,
    load_dataset_qualification,
    load_project_preregistration,
    validate_requested_operations,
)
from plume_focus.sensor_emulation import (
    FixedLeakOffCalibration,
    SensorEmulationTimeline,
    fit_fixed_leak_off_calibration,
)
from plume_focus.video_sources import (
    load_video_frame_window,
    load_video_frames,
)


__all__ = [
    "ConstantMotionProvider",
    "DenseTruthTimeline",
    "MissingMotionProvider",
    "TimeSeriesMotionProvider",
    "canonical_json_sha256",
    "file_sha256",
    "load_recorded_sequence",
    "load_recorded_suite_manifest",
    "parse_recorded_sequence_manifest",
    "require_recorded_operation",
    "validate_recorded_sequence",
    "validate_recorded_suite_pair",
    "validate_recorded_suite",
]


def load_recorded_sequence(
    manifest_path: Path,
    *,
    verify_hashes: bool = True,
) -> RecordedSequence:
    """Load, validate, and resolve one recorded sequence manifest."""

    path = Path(manifest_path).resolve()
    payload = load_json_object(path)
    manifest = parse_recorded_sequence_manifest(payload)
    data_root = data_root_for_manifest(path)
    qualification_evidence = _resolve_dataset_qualification(
        manifest,
        data_root=data_root,
    )
    dataset_audit = _resolve_dataset_audit(
        manifest,
        data_root=data_root,
        verify_hashes=verify_hashes,
    )

    media_window = manifest.media_window
    media_path = (
        resolve_data_asset(
            data_root,
            manifest.media.path,
        )
        if media_window is None
        else resolve_data_asset_without_symlinks(
            data_root,
            manifest.media.path,
            label="media-window source",
        )
    )
    if verify_hashes:
        verify_file_hash(media_path, manifest.media.sha256)
    media_window_evidence: dict[str, object] | None = None
    if media_window is None:
        frames, container_fps = load_video_frames(
            media_path,
            max_frames=None,
        )
        if verify_hashes:
            _verify_audited_decoded_content(
                dataset_audit,
                manifest.media.sha256,
                frames,
                label="sequence media",
            )
        clock_frame_count = len(frames)
    else:
        selection_metadata_path = resolve_data_asset_without_symlinks(
            data_root,
            media_window.selection_metadata_path,
            label="media-window selection metadata",
        )
        if selection_metadata_path == media_path:
            raise ValueError(
                "media-window selection metadata must resolve to a "
                "separate asset from media."
            )
        if verify_hashes:
            verify_file_hash(
                selection_metadata_path,
                media_window.selection_metadata_sha256,
            )
        clock_frame_count = _audited_parent_frame_count(
            dataset_audit,
            manifest.media.sha256,
        )
        if (
            media_window.source_end_frame_index_exclusive
            > clock_frame_count
        ):
            raise ValueError(
                "media_window ends beyond the audited parent frame count."
            )
        frames, container_fps = load_video_frame_window(
            media_path,
            start_frame_index=media_window.source_start_frame_index,
            end_frame_index_exclusive=(
                media_window.source_end_frame_index_exclusive
            ),
        )
        if verify_hashes and (
            decoded_frame_content_sha256(frames)
            != media_window.decoded_window_sha256.lower()
        ):
            raise ValueError(
                "decoded media window differs from its declared hash."
            )
    normalized_frames = tuple(
        normalize_frame(frame, manifest.normalization)
        for frame in frames
    )
    resolved_timestamps, nominal_fps, timestamp_hash = resolve_frame_clock(
        manifest.frame_clock,
        data_root=data_root,
        frame_count=clock_frame_count,
        container_fps=container_fps,
        verify_hashes=verify_hashes,
    )
    if media_window is None:
        timestamps = resolved_timestamps
    else:
        timestamps, media_window_evidence = _resolve_media_window_clock(
            media_window,
            resolved_timestamps,
            parent_frame_count=clock_frame_count,
            resolved_window_frame_count=len(normalized_frames),
        )
    raw_timeline = TimestampedArrayTimeline(
        normalized_frames,
        timestamps,
        fps=nominal_fps,
    )
    timeline = raw_timeline
    sensor_evidence: dict[str, object]
    sensor_input = manifest.sensor_input
    if (
        sensor_input is not None
        and sensor_input.kind == "structure_suppressed_emulation"
    ):
        profile = sensor_input.emulation_profile
        if profile is None:
            raise RuntimeError(
                "validated sensor emulation is missing its profile."
            )
        if manifest.manifest_schema_version == 2:
            calibration_spec = sensor_input.leak_off_calibration
            if calibration_spec is None:
                raise RuntimeError(
                    "schema-v2 sensor emulation is missing leak-off media."
                )
            _require_audited_asset(
                dataset_audit,
                calibration_spec.sha256,
                label="leak-off calibration",
            )
            calibration_path = resolve_data_asset(
                data_root,
                calibration_spec.path,
            )
            if verify_hashes:
                verify_file_hash(
                    calibration_path,
                    calibration_spec.sha256,
                )
            calibration_frames, _calibration_fps = load_video_frames(
                calibration_path,
                max_frames=None,
            )
            if verify_hashes:
                _verify_audited_decoded_content(
                    dataset_audit,
                    calibration_spec.sha256,
                    calibration_frames,
                    label="leak-off calibration",
                )
            normalized_calibration_frames = tuple(
                normalize_frame(frame, manifest.normalization)
                for frame in calibration_frames
            )
            if any(
                frame.shape != raw_timeline.shape
                for frame in normalized_calibration_frames
            ):
                raise ValueError(
                    "leak-off calibration frame shape must match sensor "
                    "media."
                )
            fixed_calibration = fit_fixed_leak_off_calibration(
                normalized_calibration_frames,
                profile=profile,
            )
            calibration_evidence = {
                **asdict(calibration_spec),
                "sample_count": fixed_calibration.sample_count,
                "resolved_calibration_sha256": (
                    _fixed_calibration_sha256(fixed_calibration)
                ),
                "resolution_mode": "decoded_at_runtime_legacy_v2",
            }
        else:
            frozen_spec = sensor_input.frozen_calibration
            if frozen_spec is None:
                raise RuntimeError(
                    "schema-v3 sensor emulation is missing frozen "
                    "calibration."
                )
            _require_audited_asset(
                dataset_audit,
                frozen_spec.source_sha256,
                label="frozen calibration source",
            )
            source_path = resolve_data_asset(
                data_root,
                frozen_spec.source_path,
            )
            preparation_spec_path = resolve_data_asset(
                data_root,
                frozen_spec.preparation_spec_path,
            )
            preparation_profile_path = resolve_data_asset(
                data_root,
                frozen_spec.preparation_profile_path,
            )
            calibration_record_path = resolve_data_asset(
                data_root,
                frozen_spec.calibration_record_path,
            )
            calibration_path = resolve_data_asset(
                data_root,
                frozen_spec.path,
            )
            if verify_hashes:
                verify_file_hash(
                    source_path,
                    frozen_spec.source_sha256,
                )
                verify_file_hash(
                    preparation_spec_path,
                    frozen_spec.preparation_spec_sha256,
                )
                verify_file_hash(
                    preparation_profile_path,
                    frozen_spec.preparation_profile_sha256,
                )
                verify_file_hash(
                    calibration_record_path,
                    frozen_spec.calibration_record_sha256,
                )
            preparation = load_preparation_spec(
                preparation_spec_path
            )
            preparation_profile = load_preparation_profile(
                preparation_profile_path
            )
            calibration_record = load_calibration_preparation_record(
                calibration_record_path
            )
            validate_frozen_calibration_binding(
                preparation,
                preparation_profile,
                calibration_record,
                manifest,
                frozen_spec,
            )
            fixed_calibration = load_fixed_calibration(
                calibration_path,
                expected_sha256=(
                    frozen_spec.sha256 if verify_hashes else None
                ),
            )
            if fixed_calibration.shape != raw_timeline.shape:
                raise ValueError(
                    "frozen calibration shape must match sensor media."
                )
            if (
                calibration_record.calibration_artifact.rows,
                calibration_record.calibration_artifact.columns,
            ) != fixed_calibration.shape:
                raise ValueError(
                    "calibration record shape does not match the decoded "
                    "artifact."
                )
            if (
                calibration_record.calibration_artifact.sample_count
                != fixed_calibration.sample_count
            ):
                raise ValueError(
                    "calibration record sample count does not match the "
                    "decoded artifact."
                )
            calibration_evidence = {
                **asdict(frozen_spec),
                "sample_count": fixed_calibration.sample_count,
                "shape": list(fixed_calibration.shape),
                "calibration_capture_group_id": (
                    calibration_record.source.capture_group_id
                ),
                "resolved_calibration_sha256": frozen_spec.sha256,
                "resolution_mode": "frozen_canonical_artifact_v1",
            }
        timeline = SensorEmulationTimeline(
            source=raw_timeline,
            calibration=fixed_calibration,
            profile=profile,
        )
        sensor_evidence = {
            "kind": sensor_input.kind,
            "provenance": sensor_input.provenance,
            "emulation_profile": profile.to_payload(),
            (
                "leak_off_calibration"
                if manifest.manifest_schema_version == 2
                else "frozen_calibration"
            ): calibration_evidence,
        }
    else:
        sensor_evidence = {
            "kind": (
                sensor_input.kind
                if sensor_input is not None
                else "legacy_unclassified"
            ),
            "provenance": (
                sensor_input.provenance
                if sensor_input is not None
                else "schema-v1 manifest has no sensor-input evidence class"
            ),
        }

    truth_spec = manifest.evaluation_truth
    if (
        manifest.manifest_schema_version >= 2
        and truth_spec.evidence_grade != "none"
    ):
        if (
            truth_spec.protocol_path is None
            or truth_spec.protocol_sha256 is None
        ):
            raise RuntimeError(
                "validated classified truth is missing its protocol asset."
            )
        protocol_path = resolve_data_asset(
            data_root,
            truth_spec.protocol_path,
        )
        if verify_hashes:
            verify_file_hash(
                protocol_path,
                truth_spec.protocol_sha256,
            )

    truth_timeline: EvaluationTruthSource | None = None
    if truth_spec.evidence_grade != "none":
        if (
            truth_spec.path is None
            or truth_spec.sha256 is None
            or truth_spec.truth_kind is None
        ):
            raise RuntimeError(
                "validated recorded truth is missing its asset fields."
            )
        truth_path = resolve_data_asset(
            data_root,
            truth_spec.path,
        )
        if verify_hashes:
            verify_file_hash(
                truth_path,
                truth_spec.sha256,
            )
        truth_frames = _load_truth_frames(
            truth_path,
            storage_format=truth_spec.storage_format,
        )
        if len(truth_frames) != len(normalized_frames):
            raise ValueError(
                "truth frame count must match the sensor frame count."
            )
        if any(frame.shape != timeline.shape for frame in truth_frames):
            raise ValueError(
                "truth frame shape must match the sensor frame shape."
            )
        truth_timeline = DenseTruthTimeline(
            TimestampedArrayTimeline(
                tuple(
                    np.asarray(frame, dtype=np.uint8)
                    for frame in truth_frames
                ),
                timestamps,
                fps=nominal_fps,
            ),
            truth_spec.truth_kind,
        )

    motion_provider, wind_hash = load_motion_provider(
        manifest.wind,
        data_root=data_root,
        verify_hashes=verify_hashes,
    )
    source_point, initial_roi = resolve_source_assumption(
        manifest.source_assumption,
        timeline.shape,
    )
    claim_assessment = assess_recorded_claim(
        manifest,
        dataset_rights_blocked=bool(
            dataset_audit.get("rights_blocked", False)
        ),
    )
    scanner_kind = manifest.scanner.evidence_kind
    evidence_contract: dict[str, object] = {
        "dataset_lineage": (
            asdict(manifest.dataset_lineage)
            if manifest.dataset_lineage is not None
            else None
        ),
        "dataset_audit": dataset_audit,
        "dataset_qualification": qualification_evidence,
        "capture_group": (
            asdict(manifest.capture_group)
            if manifest.capture_group is not None
            else None
        ),
        "sensor_input": sensor_evidence,
        "scanner": {
            "evidence_kind": scanner_kind,
            "duration_basis": (
                "modeled_counterfactual_seconds"
                if scanner_kind == "counterfactual_virtual"
                else "measured_hardware_seconds"
                if scanner_kind == "measured_hardware"
                else "legacy_unclassified"
            ),
        },
        "media_attribution": _recorded_media_attribution(
            manifest,
            dataset_audit=dataset_audit,
        ),
        "truth": {
            "evidence_grade": truth_spec.evidence_grade,
            "storage_format": truth_spec.storage_format,
            "method_id": truth_spec.method_id,
            "protocol_path": truth_spec.protocol_path,
            "protocol_sha256": truth_spec.protocol_sha256,
            "annotation_provenance": truth_spec.annotation_provenance,
            "available_to_policy": False,
            "spatial_metrics_available": truth_timeline is not None,
        },
    }
    if media_window_evidence is not None:
        evidence_contract["media_window"] = media_window_evidence
    provenance = RecordedSequenceProvenance(
        sequence_id=manifest.sequence_id,
        split=manifest.split,
        manifest_schema_version=manifest.manifest_schema_version,
        manifest_sha256=canonical_json_sha256(payload),
        media_sha256=manifest.media.sha256,
        frame_count=len(normalized_frames),
        frame_clock_mode=manifest.frame_clock.mode,
        timestamp_sha256=timestamp_hash,
        wind_mode=manifest.wind.mode,
        wind_sha256=wind_hash,
        truth_kind=manifest.evaluation_truth.truth_kind,
        truth_sha256=manifest.evaluation_truth.sha256,
        normalization=asdict(manifest.normalization),
        calibration=asdict(manifest.calibration),
        source_assumption=asdict(manifest.source_assumption),
        evidence_contract=evidence_contract,
        claim_eligibility=claim_assessment.eligibility,
        claim_limitations=claim_assessment.limitations,
        manifest=manifest.to_payload(),
    )
    return RecordedSequence(
        manifest=manifest,
        timeline=timeline,
        motion_provider=motion_provider,
        truth_source=truth_timeline,
        initial_roi=initial_roi,
        source_point=source_point,
        model_profile=uncalibrated_video_v1(manifest.scenario),
        provenance=provenance,
    )


def require_recorded_operation(
    sequence: RecordedSequence,
    operation: str,
) -> None:
    """Require one already-verified schema-v5 project operation."""

    if sequence.manifest.manifest_schema_version < 5:
        return
    qualification = sequence.provenance.evidence_contract.get(
        "dataset_qualification"
    )
    if not isinstance(qualification, dict):
        raise RuntimeError(
            "schema-v5 recorded sequence lacks verified qualification "
            "evidence."
        )
    operations = qualification.get("operations")
    if not isinstance(operations, list) or operation not in operations:
        raise ValueError(
            f"recorded operation was not project-preregistered: {operation}"
        )


def _resolve_dataset_qualification(
    manifest: RecordedSequenceManifest,
    *,
    data_root: Path,
) -> dict[str, object] | None:
    """Verify schema-v5 rights, project role, species, and decode authority."""

    reference = manifest.dataset_qualification
    if manifest.manifest_schema_version < 5:
        return None
    if reference is None:
        raise RuntimeError(
            "validated schema-v5 manifest lacks dataset qualification."
        )
    qualification_path = resolve_data_asset_without_symlinks(
        data_root,
        reference.path,
        label="dataset qualification",
    )
    verify_file_hash(qualification_path, reference.sha256)
    qualification = load_dataset_qualification(qualification_path)
    if not isinstance(qualification, DatasetQualificationV2):
        raise ValueError(
            "schema-v5 runtime requires dataset qualification schema v2."
        )
    check = check_dataset_qualification(
        data_root=data_root,
        qualification_path=qualification_path,
    )
    if not isinstance(check, QualificationCheckReportV2):
        raise RuntimeError("qualification-v2 check returned legacy evidence.")
    if not check.eligible:
        blockers = ", ".join(check.blockers) or "unspecified blocker"
        raise ValueError(
            "schema-v5 dataset qualification is not eligible: "
            f"{blockers}"
        )
    if qualification.purpose != reference.purpose:
        raise ValueError(
            "schema-v5 qualification purpose differs from its manifest."
        )
    lineage = manifest.dataset_lineage
    if lineage is None or (
        qualification.dataset_audit.dataset_id != lineage.dataset_id
        or qualification.dataset_audit.accession_id != lineage.accession_id
        or qualification.dataset_audit.path != lineage.audit_path
        or qualification.dataset_audit.sha256.lower()
        != lineage.audit_sha256.lower()
    ):
        raise ValueError(
            "schema-v5 qualification audit differs from manifest lineage."
        )
    preregistration_path = resolve_data_asset_without_symlinks(
        data_root,
        qualification.project_preregistration.path,
        label="project preregistration",
    )
    verify_file_hash(
        preregistration_path,
        qualification.project_preregistration.sha256,
    )
    preregistration = load_project_preregistration(preregistration_path)
    requested_operations = ["decode_analysis"]
    if manifest.media_window is not None:
        requested_operations.append("extract_bounded_window")
    if (
        manifest.sensor_input is not None
        and manifest.sensor_input.kind
        == "structure_suppressed_emulation"
    ):
        requested_operations.append("construct_local_derivative")
    validate_requested_operations(
        preregistration,
        tuple(requested_operations),  # type: ignore[arg-type]
    )

    asset_root = PurePosixPath(qualification.asset_root)
    media_path = PurePosixPath(manifest.media.path)
    try:
        relative_media_path = media_path.relative_to(asset_root).as_posix()
    except ValueError as error:
        raise ValueError(
            "schema-v5 media path is outside qualification asset_root."
        ) from error
    identity = (relative_media_path, manifest.media.sha256.lower())
    assignment = next(
        (
            item
            for item in preregistration.active_assignments
            if item.identity == identity
        ),
        None,
    )
    provider = next(
        (
            item
            for item in qualification.provider_assets
            if item.identity == identity
        ),
        None,
    )
    if assignment is None or provider is None:
        raise ValueError(
            "schema-v5 media identity is not an active qualified asset."
        )
    capture = manifest.capture_group
    gas_identity = manifest.gas_identity
    if assignment.split != manifest.split:
        raise ValueError(
            "schema-v5 manifest split differs from project preregistration."
        )
    if capture is None or provider.capture_group_id != capture.capture_group_id:
        raise ValueError(
            "schema-v5 capture group differs from verified provider facts."
        )
    if gas_identity is None or provider.gas_identity != gas_identity.identity:
        raise ValueError(
            "schema-v5 gas identity differs from verified provider facts."
        )
    return {
        "qualification_id": qualification.qualification_id,
        "qualification_path": reference.path,
        "qualification_sha256": reference.sha256,
        "purpose": qualification.purpose,
        "eligible": True,
        "operations": list(preregistration.operations),
        "required_permission_scopes": list(
            preregistration.required_permission_scopes
        ),
        "project_preregistration_id": preregistration.preregistration_id,
        "project_preregistration_path": (
            qualification.project_preregistration.path
        ),
        "project_preregistration_sha256": (
            qualification.project_preregistration.sha256
        ),
        "asset_relative_path": relative_media_path,
        "asset_sha256": manifest.media.sha256,
        "project_split": assignment.split,
        "provider_capture_group_id": provider.capture_group_id,
        "provider_gas_identity": provider.gas_identity,
    }


def _resolve_dataset_audit(
    manifest: RecordedSequenceManifest,
    *,
    data_root: Path,
    verify_hashes: bool,
) -> dict[str, object]:
    lineage = manifest.dataset_lineage
    if lineage is None:
        return {
            "status": "legacy_unclassified",
            "rights_blocked": False,
        }
    audit_path = resolve_data_asset(data_root, lineage.audit_path)
    if verify_hashes:
        verify_file_hash(audit_path, lineage.audit_sha256)
    audit_payload = load_json_object(audit_path)
    audit: DatasetAuditReport = parse_dataset_audit_report(
        audit_payload
    )
    if not audit.valid:
        raise ValueError("dataset audit must be valid.")
    if audit.accession.dataset_id != lineage.dataset_id:
        raise ValueError(
            "dataset audit dataset_id does not match sequence lineage."
        )
    if audit.accession.accession_id != lineage.accession_id:
        raise ValueError(
            "dataset audit accession_id does not match sequence lineage."
        )
    matching_assets = [
        item for item in audit.assets
        if item.sha256.lower() == lineage.source_asset_sha256.lower()
    ]
    if not matching_assets:
        raise ValueError(
            "dataset audit must contain a matching media hash."
        )
    return {
        "status": "hash_verified" if verify_hashes else "hash_not_verified",
        "path": lineage.audit_path,
        "sha256": lineage.audit_sha256,
        "dataset_id": lineage.dataset_id,
        "accession_id": lineage.accession_id,
        "source_asset_sha256": lineage.source_asset_sha256,
        "accession_manifest_sha256": (
            audit.accession_manifest_sha256
        ),
        "decode_scope": audit.decode_scope,
        "opencv_version": audit.opencv_version,
        "opencv_build_sha256": audit.opencv_build_sha256,
        "audited_asset_sha256s": sorted(
            {asset.sha256.lower() for asset in audit.assets}
        ),
        "decoded_content_sha256_by_asset": {
            asset.sha256.lower(): asset.decoded_content_sha256
            for asset in audit.assets
        },
        "decoded_frame_count_by_asset": {
            asset.sha256.lower(): asset.decoded_frame_count
            for asset in audit.assets
        },
        "rights_status": audit.accession.rights_status,
        "source_uri": audit.accession.source_uri,
        "rights_uri": audit.accession.rights_uri,
        "rights_provenance": audit.accession.rights_provenance,
        "rights_blocked": audit.rights_blocked,
    }


def _recorded_media_attribution(
    manifest: RecordedSequenceManifest,
    *,
    dataset_audit: dict[str, object],
) -> dict[str, object]:
    """Serialize source, licence, and analysis changes with the run."""

    scanner_kind = manifest.scanner.evidence_kind
    analysis_changes = [
        "decoded source video to single-channel uint8 using the audited "
        "OpenCV decoder environment",
        (
            "applied declared normalization mode "
            f"{manifest.normalization.mode}"
        ),
    ]
    if scanner_kind == "counterfactual_virtual":
        analysis_changes.append(
            "sampled the decoded field through a counterfactual virtual "
            "scanner; no measured hardware scan is claimed"
        )
    return {
        "source_credit": manifest.media.capture_provenance,
        "credit_basis": "manifest.media.capture_provenance",
        "dataset_id": dataset_audit.get("dataset_id"),
        "accession_id": dataset_audit.get("accession_id"),
        "source_uri": dataset_audit.get("source_uri"),
        "rights_status": dataset_audit.get("rights_status"),
        "license_uri": dataset_audit.get("rights_uri"),
        "license_provenance": dataset_audit.get(
            "rights_provenance"
        ),
        "analysis_changes": analysis_changes,
        "rendered_derivative": False,
        "rendering_changes": [],
        "distribution_condition": (
            "retain this report as the attribution and changes companion "
            "for any rendered derivative"
        ),
    }


def _audited_parent_frame_count(
    dataset_audit: dict[str, object],
    asset_sha256: str,
) -> int:
    frame_counts = dataset_audit.get("decoded_frame_count_by_asset")
    if not isinstance(frame_counts, dict):
        raise ValueError(
            "media_window requires an audited parent decoded frame count."
        )
    value = frame_counts.get(asset_sha256.lower())
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            "media_window parent decoded frame count must be a positive "
            "audited integer."
        )
    return value


def _resolve_media_window_clock(
    spec: MediaWindowSpec,
    parent_timestamps: tuple[float, ...],
    *,
    parent_frame_count: int,
    resolved_window_frame_count: int,
) -> tuple[tuple[float, ...], dict[str, object]]:
    if len(parent_timestamps) != parent_frame_count:
        raise ValueError(
            "resolved parent clock must match the audited parent frame "
            "count."
        )
    start = spec.source_start_frame_index
    end = spec.source_end_frame_index_exclusive
    timestamp_values = np.asarray(parent_timestamps, dtype=float)
    if spec.requested_start_time_seconds > float(timestamp_values[-1]):
        raise ValueError(
            "media_window requested start lies beyond the final parent "
            "timestamp."
        )
    final_interval_seconds = float(
        timestamp_values[-1] - timestamp_values[-2]
    )
    capture_end_seconds = (
        float(timestamp_values[-1]) + final_interval_seconds
    )
    if (
        spec.requested_end_time_seconds > capture_end_seconds
        and not np.isclose(
            spec.requested_end_time_seconds,
            capture_end_seconds,
            rtol=0.0,
            atol=1e-9,
        )
    ):
        raise ValueError(
            "media_window requested end lies beyond the parent capture "
            "extent."
        )
    resolved_start = int(
        np.searchsorted(
            timestamp_values,
            spec.requested_start_time_seconds,
            side="left",
        )
    )
    resolved_end = int(
        np.searchsorted(
            timestamp_values,
            spec.requested_end_time_seconds,
            side="left",
        )
    )
    if start != resolved_start or end != resolved_end:
        raise ValueError(
            "media_window requested times do not resolve to its declared "
            "source frame indices."
        )
    source_timestamps = parent_timestamps[start:end]
    if len(source_timestamps) != resolved_window_frame_count:
        raise ValueError(
            "media_window frame count differs from its resolved clock."
        )
    actual_first = source_timestamps[0]
    actual_last = source_timestamps[-1]
    if not np.isclose(
        actual_first,
        spec.first_included_timestamp_seconds,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            "media_window first included timestamp differs from the "
            "resolved parent clock."
        )
    if not np.isclose(
        actual_last,
        spec.last_included_timestamp_seconds,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            "media_window last included timestamp differs from the "
            "resolved parent clock."
        )
    rebased_timestamps = (0.0,) + tuple(
        float(timestamp - actual_first)
        for timestamp in source_timestamps[1:]
    )
    return rebased_timestamps, {
        "declared": asdict(spec),
        "resolved_parent_frame_count": parent_frame_count,
        "resolved_window_frame_count": resolved_window_frame_count,
        "resolved_source_timestamp_origin_seconds": actual_first,
        "resolved_source_last_timestamp_seconds": actual_last,
        "resolved_timeline_end_seconds": rebased_timestamps[-1],
        "timeline_clock_rebased_to_zero": True,
    }


def _require_audited_asset(
    dataset_audit: dict[str, object],
    sha256: str,
    *,
    label: str,
) -> None:
    audited_hashes = dataset_audit.get("audited_asset_sha256s")
    if not isinstance(audited_hashes, list) or (
        sha256.lower()
        not in {
            value.lower()
            for value in audited_hashes
            if isinstance(value, str)
        }
    ):
        raise ValueError(
            f"{label} must appear in the verified dataset audit."
        )


def _verify_audited_decoded_content(
    dataset_audit: dict[str, object],
    asset_sha256: str,
    frames: Iterable[np.ndarray],
    *,
    label: str,
) -> None:
    decoded_hashes = dataset_audit.get(
        "decoded_content_sha256_by_asset"
    )
    if not isinstance(decoded_hashes, dict):
        return
    expected = decoded_hashes.get(asset_sha256.lower())
    if expected is None:
        return
    if not isinstance(expected, str):
        raise ValueError(
            f"{label} decoded-content audit hash must be a string."
        )
    if decoded_frame_content_sha256(frames) != expected.lower():
        raise ValueError(
            f"{label} decoded content differs from its dataset audit."
        )


def _load_truth_frames(
    path: Path,
    *,
    storage_format: str,
) -> tuple[np.ndarray, ...]:
    if storage_format == "legacy_video":
        frames, _container_fps = load_video_frames(
            path,
            max_frames=None,
        )
        return tuple(
            np.asarray(frame, dtype=np.uint8)
            for frame in frames
        )
    if storage_format != "lossless_npz_uint8":
        raise ValueError("unknown truth storage format.")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"frames"}:
                raise ValueError(
                    "lossless truth NPZ must contain only a frames array."
                )
            values = np.asarray(archive["frames"])
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith(
            "lossless truth NPZ"
        ):
            raise
        raise ValueError(f"Invalid lossless truth NPZ: {path}") from error
    if values.dtype != np.uint8:
        raise ValueError("lossless truth frames must use uint8 dtype.")
    if values.ndim != 3 or min(values.shape) <= 0:
        raise ValueError(
            "lossless truth frames must have shape (frames, rows, cols)."
        )
    return tuple(np.array(frame, dtype=np.uint8, copy=True) for frame in values)


def _fixed_calibration_sha256(
    calibration: FixedLeakOffCalibration,
) -> str:
    """Hash resolved float arrays using a platform-independent byte order."""

    digest = hashlib.sha256()
    digest.update(b"fixed-leak-off-calibration-v1\0")
    digest.update(
        np.asarray(calibration.shape, dtype="<i8").tobytes(order="C")
    )
    digest.update(
        np.asarray([calibration.sample_count], dtype="<i8").tobytes()
    )
    for values in (
        calibration.background_raw_units,
        calibration.noise_sigma_raw_units,
    ):
        digest.update(
            np.asarray(values, dtype="<f8").tobytes(order="C")
        )
    return digest.hexdigest()


def validate_recorded_sequence(
    manifest_path: Path,
    *,
    verify_hashes: bool = True,
) -> dict[str, object]:
    """Return a compact validation payload after fully loading a sequence."""

    sequence = load_recorded_sequence(
        manifest_path,
        verify_hashes=verify_hashes,
    )
    timeline = sequence.timeline
    provenance = sequence.provenance.to_payload()
    validation: dict[str, object] = {
        "valid": True,
        "sequence_id": sequence.manifest.sequence_id,
        "split": sequence.manifest.split,
        "shape": list(timeline.shape),
        "frame_count": sequence.provenance.frame_count,
        "start_time_seconds": 0.0,
        "end_time_seconds": timeline.end_time_seconds,
        "nominal_fps": timeline.fps,
        "truth_basis": (
            sequence.truth_source.truth_basis
            if sequence.truth_source is not None
            else None
        ),
        "manifest_sha256": sequence.provenance.manifest_sha256,
        "media_sha256": sequence.provenance.media_sha256,
        "truth_sha256": sequence.provenance.truth_sha256,
        "wind_mode": sequence.provenance.wind_mode,
        "evidence_grade": provenance["evidence_grade"],
        "claim_eligibility": provenance["claim_eligibility"],
        "claim_limitations": provenance["claim_limitations"],
        "evidence_contract": provenance["evidence_contract"],
    }
    media_window = provenance["evidence_contract"].get("media_window")
    if media_window is not None:
        validation["media_window"] = media_window
    return validation


def validate_recorded_suite(
    suite_path: Path,
    *,
    verify_hashes: bool = True,
) -> dict[str, object]:
    """Validate a frozen suite and every referenced recorded sequence."""

    path = Path(suite_path).resolve()
    suite = load_recorded_suite_manifest(path)
    validations: list[dict[str, object]] = []
    for reference in suite.sequences:
        manifest_path = resolve_manifest_reference(
            path.parent,
            reference.path,
        )
        payload = load_json_object(manifest_path)
        actual_manifest_hash = canonical_json_sha256(payload)
        if actual_manifest_hash != reference.manifest_sha256.lower():
            raise ValueError(
                f"Sequence manifest hash mismatch: {manifest_path}"
            )
        validation = validate_recorded_sequence(
            manifest_path,
            verify_hashes=verify_hashes,
        )
        if validation["split"] != suite.split:
            raise ValueError(
                "every sequence split must match the recorded suite split."
            )
        validations.append(validation)
    suite_payload = load_json_object(path)
    return {
        "valid": True,
        "suite_id": suite.suite_id,
        "split": suite.split,
        "frozen_at_utc": suite.frozen_at_utc,
        "frozen_before_model_commit": (
            suite.frozen_before_model_commit
        ),
        "freeze_provenance": suite.freeze_provenance,
        "suite_manifest_sha256": canonical_json_sha256(suite_payload),
        "sequence_count": len(validations),
        "sequences": validations,
    }


def validate_recorded_suite_pair(
    development_suite_path: Path,
    heldout_suite_path: Path,
    *,
    verify_hashes: bool = True,
) -> dict[str, object]:
    """Validate both frozen splits and reject capture-group leakage."""

    development = validate_recorded_suite(
        development_suite_path,
        verify_hashes=verify_hashes,
    )
    heldout = validate_recorded_suite(
        heldout_suite_path,
        verify_hashes=verify_hashes,
    )
    if development["split"] != "development":
        raise ValueError(
            "development_suite_path must reference a development suite."
        )
    if heldout["split"] != "held-out":
        raise ValueError(
            "heldout_suite_path must reference a held-out suite."
        )
    role_isolation = validate_cross_split_asset_roles(
        development,
        heldout,
    )
    return {
        "valid": True,
        "development_suite_id": development["suite_id"],
        "development_suite_manifest_sha256": (
            development["suite_manifest_sha256"]
        ),
        "heldout_suite_id": heldout["suite_id"],
        "heldout_suite_manifest_sha256": (
            heldout["suite_manifest_sha256"]
        ),
        **role_isolation,
    }
