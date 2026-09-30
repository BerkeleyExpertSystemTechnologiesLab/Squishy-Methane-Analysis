from __future__ import annotations

import copy
from pathlib import Path

import cv2
import numpy as np
import pytest

from plume_focus.model_profiles import uncalibrated_video_v1
from plume_focus.recorded_artifacts import (
    canonical_json_bytes,
    write_json_artifact,
)
from plume_focus.recorded_dataset import (
    DatasetAccession,
    audit_dataset_directory,
)
from plume_focus.recorded_freeze import freeze_recorded_suite
from plume_focus.recorded_manifest_io import (
    file_sha256,
    load_json_object,
    load_recorded_suite_manifest,
    parse_recorded_sequence_manifest,
)
from plume_focus.recorded_preparation import (
    _build_raw_manifest,
    prepare_recorded_dataset,
)
from plume_focus.recorded_preparation_contracts import (
    load_preparation_profile,
    parse_preparation_spec,
)
from plume_focus.recorded_sources import load_recorded_sequence
from plume_focus.recorded_workflow import run_recorded_sequence


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (160, 96),
    )
    if not writer.isOpened():
        raise RuntimeError("test video writer did not open")
    try:
        for index in range(30):
            value = 20 + (index % 6) * 20
            writer.write(
                np.full((96, 160, 3), value, dtype=np.uint8)
            )
    finally:
        writer.release()


def _v5_manifest(
    *,
    media_sha256: str = "a" * 64,
    audit_sha256: str = "b" * 64,
    qualification_sha256: str = "c" * 64,
    scenario: str = "plume",
    gas_identity: str = "propane",
) -> dict[str, object]:
    return {
        "manifest_schema_version": 5,
        "sequence_id": "usu-ground-propane-template",
        "split": "development",
        "split_provenance": (
            "engineering-only project role fixed before planner execution"
        ),
        "scenario": scenario,
        "gas_identity": {
            "identity": gas_identity,
            "provenance": "provider declares controlled propane release",
        },
        "dataset_qualification": {
            "path": "metadata/dataset-qualification-v2.json",
            "sha256": qualification_sha256,
            "purpose": "diagnostic_raw_pilot",
        },
        "dataset_lineage": {
            "dataset_id": "usu-ogi-survey",
            "accession_id": "usu-ogi-survey-test-accession",
            "audit_path": "assets/dataset-audit.json",
            "audit_sha256": audit_sha256,
            "source_asset_sha256": media_sha256,
        },
        "capture_group": {
            "capture_group_id": "usu-ground-26feb2018",
            "grouping_provenance": "provider-declared capture date",
        },
        "media": {
            "path": "assets/Ground_26feb2018_propane.mp4",
            "sha256": media_sha256,
            "capture_provenance": "independently recorded GF320 video",
        },
        "sensor_input": {
            "kind": "raw_recorded",
            "provenance": "unaltered decoded grayscale intensity",
        },
        "frame_clock": {
            "mode": "constant_fps",
            "fps": 5.0,
            "provenance": "decoded container clock",
        },
        "normalization": {
            "mode": "uint8_identity",
            "provenance": "decoded uint8 values",
        },
        "calibration": {
            "sensor_id": "flir-gf320-provider-recording",
            "signal_units": "compressed_8bit_ogi_display_intensity",
            "provenance": "non-radiometric recorded input",
        },
        "wind": {
            "mode": "none",
            "units": "px_per_second",
            "provenance": "no synchronized wind supplied to planner",
            "conversion_provenance": "not applicable",
        },
        "source_assumption": {
            "mode": "source_free",
            "provenance": "no source geometry available to policy",
        },
        "scanner": {
            "provenance": "counterfactual mechanical scan fixture",
            "evidence_kind": "counterfactual_virtual",
            "scan_mode": "column",
            "scan_rate_hz": 100.0,
            "sample_stride_px": 1,
            "vertical_pass_seconds": 0.01,
            "horizontal_step_seconds": 0.0,
            "column_stride_px": 1,
            "column_mass_fraction": 0.8,
            "row_mass_fraction": 0.85,
            "minimum_sweep_columns": 2,
            "minimum_sweep_rows": 2,
            "maximum_sweep_columns": 8,
        },
        "evaluation_truth": {
            "evidence_grade": "none",
            "annotation_provenance": "no spatial truth available",
        },
        "model_profile_id": "uncalibrated_video_v1",
    }


def test_v5_propane_round_trips_as_generic_plume() -> None:
    payload = _v5_manifest()

    manifest = parse_recorded_sequence_manifest(payload)

    assert manifest.scenario == "plume"
    assert manifest.gas_identity is not None
    assert manifest.gas_identity.identity == "propane"
    assert manifest.media_window is None
    assert manifest.to_payload() == payload


@pytest.mark.parametrize(
    ("scenario", "identity"),
    (
        ("plume", "methane"),
        ("methane", "propane"),
        ("methane", "hydrocarbon_unspecified"),
    ),
)
def test_v5_rejects_incompatible_planner_and_gas_identity(
    scenario: str,
    identity: str,
) -> None:
    payload = _v5_manifest(
        scenario=scenario,
        gas_identity=identity,
    )

    with pytest.raises(ValueError, match="incompatible with gas_identity"):
        parse_recorded_sequence_manifest(payload)


def test_v5_accepts_declared_methane_and_unspecified_hydrocarbon() -> None:
    methane = parse_recorded_sequence_manifest(
        _v5_manifest(scenario="methane", gas_identity="methane")
    )
    unspecified = parse_recorded_sequence_manifest(
        _v5_manifest(gas_identity="hydrocarbon_unspecified")
    )

    assert methane.gas_identity is not None
    assert methane.gas_identity.identity == "methane"
    assert unspecified.gas_identity is not None
    assert unspecified.scenario == "plume"


def test_v5_requires_strict_gas_identity_provenance() -> None:
    missing = _v5_manifest()
    missing.pop("gas_identity")
    with pytest.raises(ValueError, match="missing keys.*gas_identity"):
        parse_recorded_sequence_manifest(missing)

    unknown_identity = _v5_manifest(gas_identity="carbon_dioxide")
    with pytest.raises(ValueError, match="unknown recorded gas identity"):
        parse_recorded_sequence_manifest(unknown_identity)

    unknown_field = _v5_manifest()
    gas_payload = unknown_field["gas_identity"]
    assert isinstance(gas_payload, dict)
    gas_payload["concentration"] = "unknown"
    with pytest.raises(ValueError, match="gas identity has unknown keys"):
        parse_recorded_sequence_manifest(unknown_field)

    missing_qualification = _v5_manifest()
    missing_qualification.pop("dataset_qualification")
    with pytest.raises(
        ValueError,
        match="missing keys.*dataset_qualification",
    ):
        parse_recorded_sequence_manifest(missing_qualification)

    invalid_purpose = _v5_manifest()
    invalid_purpose["dataset_qualification"]["purpose"] = "performance"
    with pytest.raises(ValueError, match="unknown dataset qualification purpose"):
        parse_recorded_sequence_manifest(invalid_purpose)


@pytest.mark.parametrize("legacy_version", (3, 4))
def test_legacy_manifests_reject_v5_gas_identity(
    legacy_version: int,
) -> None:
    payload = _v5_manifest()
    payload["manifest_schema_version"] = legacy_version
    payload["scenario"] = "methane"
    if legacy_version == 4:
        payload["media_window"] = {
            "requested_start_time_seconds": 0.0,
            "requested_end_time_seconds": 0.6,
            "source_start_frame_index": 0,
            "source_end_frame_index_exclusive": 3,
            "first_included_timestamp_seconds": 0.0,
            "last_included_timestamp_seconds": 0.4,
            "parent_media_sha256": "a" * 64,
            "decoded_window_sha256": "c" * 64,
            "selection_metadata_path": "metadata/window.json",
            "selection_metadata_sha256": "d" * 64,
            "selection_provenance": "predeclared test window",
        }

    with pytest.raises(ValueError, match="unknown keys.*gas_identity"):
        parse_recorded_sequence_manifest(payload)


def test_raw_preparation_preserves_v5_and_loads_plume_profile(
    tmp_path: Path,
) -> None:
    root = tmp_path / "recorded"
    assets = root / "assets"
    manifests = root / "manifests"
    metadata = root / "metadata"
    assets.mkdir(parents=True)
    manifests.mkdir()
    metadata.mkdir()
    media_path = assets / "Ground_26feb2018_propane.mp4"
    _write_video(media_path)
    accession_path = metadata / "accession.json"
    write_json_artifact(
        accession_path,
        {"source": "pytest explicitly licensed external fixture"},
    )
    accession = DatasetAccession(
        dataset_id="usu-ogi-survey",
        accession_id="usu-ogi-survey-test-accession",
        source_uri="https://example.test/usu-ogi",
        accessed_at_utc="2026-07-31T00:00:00Z",
        rights_status="verified",
        rights_provenance="pytest CC BY 4.0 declaration",
        rights_uri="https://creativecommons.org/licenses/by/4.0/",
    )
    audit = audit_dataset_directory(
        assets,
        accession,
        accession_manifest_sha256=file_sha256(accession_path),
    )
    audit_path = assets / "dataset-audit.json"
    write_json_artifact(audit_path, audit.to_payload())
    terms_snapshot = metadata / "cc-by-4.0-terms.txt"
    provider_snapshot = metadata / "provider-metadata.txt"
    terms_snapshot.write_text(
        "CC BY 4.0 fixture permits local decode and annotated derivatives.\n",
        encoding="utf-8",
    )
    provider_snapshot.write_text(
        "Ground_26feb2018_propane.mp4 is a controlled propane capture.\n",
        encoding="utf-8",
    )
    audit_reference = {
        "dataset_id": "usu-ogi-survey",
        "accession_id": "usu-ogi-survey-test-accession",
        "path": "assets/dataset-audit.json",
        "sha256": file_sha256(audit_path),
    }
    preregistration_path = metadata / "project-preregistration.json"
    write_json_artifact(
        preregistration_path,
        {
            "project_preregistration_schema_version": 1,
            "artifact_kind": "recorded_project_preregistration",
            "claim_eligibility": "not_sequence_evidence",
            "preregistration_id": "usu-propane-diagnostic-v1",
            "purpose": "diagnostic_raw_pilot",
            "created_at_utc": "2026-07-31T00:00:00Z",
            "dataset_audit": audit_reference,
            "operations": ["decode_analysis", "render_local_media"],
            "assignments": [
                {
                    "relative_path": "Ground_26feb2018_propane.mp4",
                    "sha256": file_sha256(media_path),
                    "split": "development",
                    "exclusion_reason": None,
                }
            ],
        },
    )
    qualification_path = metadata / "dataset-qualification-v2.json"
    permission_scopes = (
        "local_decode_analysis",
        "bounded_interval_extraction",
        "local_derivative_construction",
        "publish_aggregate_metrics",
        "publish_frames_or_clips",
        "redistribution",
    )
    write_json_artifact(
        qualification_path,
        {
            "qualification_schema_version": 2,
            "artifact_kind": "recorded_dataset_qualification",
            "claim_eligibility": "not_sequence_evidence",
            "qualification_id": "usu-propane-diagnostic-qualification-v2",
            "asset_root": "assets",
            "purpose": "diagnostic_raw_pilot",
            "dataset_audit": audit_reference,
            "project_preregistration": {
                "path": "metadata/project-preregistration.json",
                "sha256": file_sha256(preregistration_path),
            },
            "evidence_records": [
                {
                    "evidence_id": "terms-snapshot",
                    "kind": "formal_terms",
                    "reference": "metadata/cc-by-4.0-terms.txt",
                    "observed_at_utc": "2026-07-31T00:00:00Z",
                    "content_sha256": file_sha256(terms_snapshot),
                    "summary": "Fixture rights authorize declared operations.",
                },
                {
                    "evidence_id": "provider-snapshot",
                    "kind": "dataset_owner_metadata",
                    "reference": "metadata/provider-metadata.txt",
                    "observed_at_utc": "2026-07-31T00:00:00Z",
                    "content_sha256": file_sha256(provider_snapshot),
                    "summary": "Fixture identifies capture and propane gas.",
                },
            ],
            "permissions": {
                scope: {
                    "decision": (
                        "allowed"
                        if scope
                        in {
                            "local_decode_analysis",
                            "local_derivative_construction",
                        }
                        else "unknown"
                    ),
                    "evidence_ids": (
                        ["terms-snapshot"]
                        if scope
                        in {
                            "local_decode_analysis",
                            "local_derivative_construction",
                        }
                        else []
                    ),
                }
                for scope in permission_scopes
            },
            "native_interval_semantics": {
                "status": "unresolved",
                "class_zero_meaning": None,
                "nominal_interval_seconds": None,
                "trim_start_seconds": None,
                "trim_end_seconds": None,
                "exceptions": [],
                "evidence_ids": [],
            },
            "provider_assets": [
                {
                    "relative_path": "Ground_26feb2018_propane.mp4",
                    "sha256": file_sha256(media_path),
                    "paper_group_id": "paper-ground-controlled-release",
                    "capture_group_id": "usu-ground-26feb2018",
                    "gas_identity": "propane",
                    "fact_status": "verified",
                    "fact_evidence_ids": ["provider-snapshot"],
                }
            ],
        },
    )
    template = _v5_manifest(
        media_sha256=file_sha256(media_path),
        audit_sha256=file_sha256(audit_path),
        qualification_sha256=file_sha256(qualification_path),
    )
    spec_path = metadata / "preparation-spec.json"
    profile_path = metadata / "preparation-profile.json"
    write_json_artifact(
        spec_path,
        {
            "preparation_spec_schema_version": 4,
            "preparation_id": "usu-propane-engineering-smoke",
            "producing_commit": "a" * 40,
            "dataset_audit": {
                "dataset_id": "usu-ogi-survey",
                "accession_id": "usu-ogi-survey-test-accession",
                "path": "assets/dataset-audit.json",
                "sha256": file_sha256(audit_path),
            },
            "dataset_qualification": {
                "path": "metadata/dataset-qualification-v2.json",
                "sha256": file_sha256(qualification_path),
            },
            "capture_groups": [
                {
                    "capture_group_id": "usu-ground-26feb2018",
                    "role": "development",
                    "provenance": "provider-declared capture date",
                    "asset_sha256s": [file_sha256(media_path)],
                }
            ],
            "exclusions": [],
            "sequences": [
                {
                    "output_sequence_id": "usu-ground-propane-raw",
                    "manifest_template": copy.deepcopy(template),
                    "calibration_source": None,
                    "media_window_request": None,
                }
            ],
        },
    )
    write_json_artifact(
        profile_path,
        {
            "preparation_profile_schema_version": 1,
            "profile_id": "raw-recorded-v1",
            "sensor_input_kind": "raw_recorded",
            "sensor_input_provenance": "unaltered recorded intensity",
            "emulation_profile": None,
        },
    )

    prepare_recorded_dataset(
        data_root=root,
        spec_path=spec_path,
        profile_path=profile_path,
        output_manifests_dir=manifests,
        report_path=root / "preparation-report.json",
    )
    manifest_path = manifests / "usu-ground-propane-raw.json"
    prepared = parse_recorded_sequence_manifest(
        load_json_object(manifest_path)
    )
    loaded = load_recorded_sequence(manifest_path)

    assert prepared.manifest_schema_version == 5
    assert prepared.to_payload()["gas_identity"] == {
        "identity": "propane",
        "provenance": "provider declares controlled propane release",
    }
    assert prepared.media_window is None
    assert loaded.model_profile.scenario == "plume"
    assert loaded.model_profile == uncalibrated_video_v1("plume")
    assert loaded.provenance.to_payload()["manifest"]["gas_identity"] == {
        "identity": "propane",
        "provenance": "provider declares controlled propane release",
    }
    qualification_evidence = loaded.provenance.evidence_contract[
        "dataset_qualification"
    ]
    assert qualification_evidence["provider_gas_identity"] == "propane"
    assert "render_local_media" in qualification_evidence["operations"]
    assert "diagnostic_raw_pilot_is_engineering_only" in (
        loaded.provenance.claim_limitations
    )
    assert "non_methane_recording_is_not_methane_validation" in (
        loaded.provenance.claim_limitations
    )

    suite = freeze_recorded_suite(
        data_root=root,
        preparation_report_path=root / "preparation-report.json",
        preparation_report_sha256=file_sha256(
            root / "preparation-report.json"
        ),
        suite_id="usu-propane-engineering-development",
        split="development",
        frozen_at_utc="2026-07-31T00:00:00Z",
        frozen_before_model_commit="b" * 40,
        freeze_provenance="pytest v5 freeze fixed before execution",
        suite_output_path=Path("manifests/usu-propane-suite.json"),
        freeze_record_output_path=Path(
            "metadata/usu-propane-suite.freeze.json"
        ),
    )

    frozen_suite = load_recorded_suite_manifest(
        suite.suite_manifest_path
    )
    assert frozen_suite.split == "development"
    assert len(frozen_suite.sequences) == 1

    selection_path = metadata / "window-selection.json"
    write_json_artifact(
        selection_path,
        {"selection": "complete three-frame pytest asset"},
    )
    windowed_spec = load_json_object(spec_path)
    windowed_spec["preparation_spec_schema_version"] = 4
    sequences = windowed_spec["sequences"]
    assert isinstance(sequences, list)
    sequence_payload = sequences[0]
    assert isinstance(sequence_payload, dict)
    sequence_payload["media_window_request"] = {
        "requested_start_time_seconds": 0.0,
        "requested_end_time_seconds": 0.6,
        "expected_source_start_frame_index": 0,
        "expected_source_end_frame_index_exclusive": 3,
        "selection_metadata_path": "metadata/window-selection.json",
        "selection_metadata_sha256": file_sha256(selection_path),
        "selection_provenance": "complete fixture chosen before execution",
    }
    windowed_sequence = parse_preparation_spec(
        windowed_spec
    ).sequences[0]
    windowed_manifest = _build_raw_manifest(
        root,
        windowed_sequence,
        load_preparation_profile(profile_path),
        audit_asset=audit.assets[0],
    )

    assert windowed_manifest.manifest_schema_version == 5
    assert windowed_manifest.media_window is not None
    assert windowed_manifest.gas_identity == prepared.gas_identity
    unauthorized_window_path = manifests / "unauthorized-window.json"
    write_json_artifact(
        unauthorized_window_path,
        windowed_manifest.to_payload(),
    )
    with pytest.raises(ValueError, match="extract_bounded_window"):
        load_recorded_sequence(unauthorized_window_path)

    allowed_video = root / "outputs" / "allowed.mp4"
    allowed_report = root / "outputs" / "allowed.json"
    rendered = run_recorded_sequence(
        manifest_path,
        max_scans=2,
        report_output=allowed_report,
        output_video=allowed_video,
    )
    assert rendered.scans_completed == 2
    assert allowed_video.is_file()

    original_manifest_payload = load_json_object(manifest_path)
    forged_hash_payload = copy.deepcopy(original_manifest_payload)
    forged_hash_payload["dataset_qualification"]["sha256"] = "d" * 64
    manifest_path.write_bytes(canonical_json_bytes(forged_hash_payload))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_recorded_sequence(manifest_path)
    forged_purpose_payload = copy.deepcopy(original_manifest_payload)
    forged_purpose_payload["dataset_qualification"]["purpose"] = (
        "evidence_suite"
    )
    manifest_path.write_bytes(canonical_json_bytes(forged_purpose_payload))
    with pytest.raises(ValueError, match="purpose differs"):
        load_recorded_sequence(manifest_path)
    manifest_path.write_bytes(canonical_json_bytes(original_manifest_payload))

    preregistration_payload = load_json_object(preregistration_path)
    preregistration_payload["operations"] = ["decode_analysis"]
    preregistration_path.write_bytes(
        canonical_json_bytes(preregistration_payload)
    )
    qualification_payload = load_json_object(qualification_path)
    qualification_payload["project_preregistration"]["sha256"] = (
        file_sha256(preregistration_path)
    )
    qualification_payload["permissions"][
        "local_derivative_construction"
    ] = {"decision": "unknown", "evidence_ids": []}
    qualification_path.write_bytes(
        canonical_json_bytes(qualification_payload)
    )
    manifest_payload = load_json_object(manifest_path)
    manifest_payload["dataset_qualification"]["sha256"] = file_sha256(
        qualification_path
    )
    manifest_path.write_bytes(canonical_json_bytes(manifest_payload))
    unauthorized_video = root / "outputs" / "unauthorized.mp4"

    with pytest.raises(
        ValueError,
        match="not project-preregistered: render_local_media",
    ):
        run_recorded_sequence(
            manifest_path,
            max_scans=2,
            report_output=root / "outputs" / "unauthorized.json",
            output_video=unauthorized_video,
        )
    assert not unauthorized_video.exists()
