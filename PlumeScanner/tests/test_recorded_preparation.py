from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import pytest

from plume_focus.recorded_artifacts import (
    canonical_json_bytes,
    write_json_artifact,
)
from plume_focus.recorded_calibration import load_fixed_calibration
from plume_focus.recorded_dataset import (
    DatasetAccession,
    audit_dataset_directory,
    decoded_frame_content_sha256,
)
from plume_focus.recorded_freeze import freeze_recorded_suite
from plume_focus.recorded_manifest_io import (
    canonical_json_sha256,
    file_sha256,
    load_json_object,
    parse_recorded_sequence_manifest,
)
from plume_focus.recorded_preparation import prepare_recorded_dataset
from plume_focus.recorded_preparation_contracts import (
    load_calibration_preparation_record,
    load_preparation_spec,
)
from plume_focus.recorded_sources import (
    load_recorded_sequence,
    validate_recorded_suite,
)
from plume_focus.video_sources import load_video_frame_window


def _write_video(
    path: Path,
    values: tuple[int, ...],
    *,
    fps: float = 5.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (12, 8),
    )
    if not writer.isOpened():
        raise RuntimeError("test video writer did not open")
    try:
        for value in values:
            frame = np.full((8, 12, 3), value, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def _fixture(
    tmp_path: Path,
    *,
    emulated: bool,
    sequence_count: int = 1,
    duplicate_calibration: bool = False,
    windowed: bool = False,
) -> dict[str, object]:
    root = tmp_path / "recorded"
    assets = root / "assets"
    manifests = root / "manifests"
    metadata = root / "metadata"
    assets.mkdir(parents=True)
    manifests.mkdir()
    metadata.mkdir()
    media_path = assets / "sensor.mp4"
    calibration_path = assets / "leak-off.mp4"
    _write_video(
        media_path,
        (20, 30, 40, 50, 60, 70) if windowed else (20, 30, 40),
    )
    _write_video(calibration_path, (18, 20, 22))
    if duplicate_calibration:
        shutil.copyfile(
            calibration_path,
            assets / "duplicate-leak-off.mp4",
        )

    accession_path = metadata / "accession.json"
    accession_path.write_text("fixture accession\n", encoding="utf-8")
    audit = audit_dataset_directory(
        assets,
        DatasetAccession(
            dataset_id="candidate-fixture",
            accession_id="candidate-fixture-20260726",
            source_uri="https://example.test/dataset",
            accessed_at_utc="2026-07-26T00:00:00Z",
            rights_status="verified",
            rights_provenance="pytest rights declaration",
            rights_uri="https://example.test/terms",
        ),
        accession_manifest_sha256=file_sha256(accession_path),
    )
    audit_path = assets / "dataset-audit.json"
    write_json_artifact(audit_path, audit.to_payload())
    selection_metadata_path = metadata / "window-selection.json"
    if windowed:
        write_json_artifact(
            selection_metadata_path,
            {
                "selection_rule": "fixed pytest interval before execution",
                "requested_start_time_seconds": 0.2,
                "requested_end_time_seconds": 0.8,
            },
        )

    media_sha = file_sha256(media_path)
    calibration_sha = file_sha256(calibration_path)
    template = {
        "manifest_schema_version": 3,
        "sequence_id": "template",
        "split": "development",
        "split_provenance": "assigned before preparation",
        "scenario": "methane",
        "dataset_lineage": {
            "dataset_id": "candidate-fixture",
            "accession_id": "candidate-fixture-20260726",
            "audit_path": "assets/dataset-audit.json",
            "audit_sha256": file_sha256(audit_path),
            "source_asset_sha256": media_sha,
        },
        "capture_group": {
            "capture_group_id": "capture-source",
            "grouping_provenance": "whole acquisition is one split unit",
        },
        "media": {
            "path": "assets/sensor.mp4",
            "sha256": media_sha,
            "capture_provenance": "pytest external video",
        },
        "sensor_input": {
            "kind": "raw_recorded",
            "provenance": "truth-free preparation template",
        },
        "frame_clock": {
            "mode": "constant_fps",
            "fps": 5.0,
            "provenance": "pytest video clock",
        },
        "normalization": {
            "mode": "uint8_identity",
            "provenance": "decoded uint8 values",
        },
        "calibration": {
            "sensor_id": "pytest-ogi",
            "signal_units": "compressed_8bit_ogi_display_intensity",
            "provenance": "non-radiometric pytest fixture",
        },
        "wind": {
            "mode": "none",
            "units": "px_per_second",
            "provenance": "no synchronized wind",
            "conversion_provenance": "not applicable",
        },
        "source_assumption": {
            "mode": "known_corridor",
            "provenance": "predeclared fixture corridor",
            "initial_roi": [0, 0, 7, 11],
        },
        "scanner": {
            "provenance": "counterfactual pytest scanner",
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
            "annotation_provenance": (
                "truth is attached by a separate evidence workflow"
            ),
        },
        "model_profile_id": "uncalibrated_video_v1",
    }
    sequences = []
    for index in range(sequence_count):
        sequence_payload = {
            "output_sequence_id": f"prepared-{index + 1}",
            "manifest_template": copy.deepcopy(template),
            "calibration_source": (
                {
                    "path": "assets/leak-off.mp4",
                    "sha256": calibration_sha,
                    "provenance": "separate development leak-off clip",
                    "calibration_scope": "development_only",
                    "capture_group_id": "capture-calibration",
                }
                if emulated
                else None
            ),
        }
        if windowed:
            sequence_payload["media_window_request"] = {
                "requested_start_time_seconds": 0.2,
                "requested_end_time_seconds": 0.8,
                "expected_source_start_frame_index": 1,
                "expected_source_end_frame_index_exclusive": 4,
                "selection_metadata_path": (
                    "metadata/window-selection.json"
                ),
                "selection_metadata_sha256": file_sha256(
                    selection_metadata_path
                ),
                "selection_provenance": (
                    "fixed pytest interval before planner execution"
                ),
            }
        sequences.append(sequence_payload)
    spec_payload = {
        "preparation_spec_schema_version": 2 if windowed else 1,
        "preparation_id": "pytest-preparation-v1",
        "producing_commit": "a" * 40,
        "dataset_audit": {
            "dataset_id": "candidate-fixture",
            "accession_id": "candidate-fixture-20260726",
            "path": "assets/dataset-audit.json",
            "sha256": file_sha256(audit_path),
        },
        "capture_groups": [
            {
                "capture_group_id": "capture-source",
                "role": "development",
                "provenance": "whole acquisition is one split unit",
                "asset_sha256s": [media_sha],
            },
            {
                "capture_group_id": "capture-calibration",
                "role": (
                    "development" if emulated else "excluded"
                ),
                "provenance": "separate calibration acquisition",
                "asset_sha256s": [calibration_sha],
            },
        ],
        "exclusions": (
            []
            if emulated
            else [
                {
                    "capture_group_id": "capture-calibration",
                    "reason_code": (
                        "duplicate_byte_asset"
                        if duplicate_calibration
                        else "predeclared_protocol_exclusion"
                    ),
                    "rationale": (
                        "raw-input profile does not consume calibration"
                    ),
                }
            ]
        ),
        "sequences": sequences,
    }
    profile_payload = {
        "preparation_profile_schema_version": 1,
        "profile_id": (
            "structure-suppressed-v1" if emulated else "raw-recorded-v1"
        ),
        "sensor_input_kind": (
            "structure_suppressed_emulation"
            if emulated
            else "raw_recorded"
        ),
        "sensor_input_provenance": (
            "fixed pytest sensor emulation"
            if emulated
            else "unaltered recorded intensity"
        ),
        "emulation_profile": (
            {
                "profile_id": "structure-suppressed-v1",
                "residual_mode": "two_sided_magnitude",
                "provenance": "pytest development settings",
                "noise_floor_raw_units": 1.0,
                "output_units_per_noise_sigma": 16.0,
                "smoothing_time_constant_seconds": None,
                "profile_schema_version": 1,
            }
            if emulated
            else None
        ),
    }
    spec_path = metadata / "preparation-spec.json"
    profile_path = metadata / "preparation-profile.json"
    write_json_artifact(spec_path, spec_payload)
    write_json_artifact(profile_path, profile_payload)
    return {
        "root": root,
        "manifests": manifests,
        "report": root / "preparation-report.json",
        "spec_path": spec_path,
        "profile_path": profile_path,
        "audit_path": audit_path,
        "media_path": media_path,
        "calibration_path": calibration_path,
        "selection_metadata_path": selection_metadata_path,
    }


def _prepare(paths: dict[str, object]):
    return prepare_recorded_dataset(
        data_root=paths["root"],  # type: ignore[arg-type]
        spec_path=paths["spec_path"],  # type: ignore[arg-type]
        profile_path=paths["profile_path"],  # type: ignore[arg-type]
        output_manifests_dir=paths["manifests"],  # type: ignore[arg-type]
        report_path=paths["report"],  # type: ignore[arg-type]
    )


def _qualify_windowed_fixture(
    paths: dict[str, object],
    *,
    eligible: bool = True,
) -> None:
    """Upgrade one v2 fixture to a strict qualification-bound v3 spec."""

    audit_path = paths["audit_path"]
    assert isinstance(audit_path, Path)
    qualification_path = (
        paths["root"] / "metadata" / "dataset-qualification.json"
    )
    audit_payload = load_json_object(audit_path)
    evidence_id = "owner-metadata-v1"
    permission_evidence_id = "owner-permission-v1"
    assets = []
    for asset in audit_payload["assets"]:
        relative_path = asset["relative_path"]
        is_sensor = relative_path == "sensor.mp4"
        assets.append(
            {
                "relative_path": relative_path,
                "sha256": asset["sha256"],
                "paper_group_id": (
                    "paper-sensor" if is_sensor else "paper-calibration"
                ),
                "capture_group_id": (
                    "capture-source"
                    if is_sensor and eligible
                    else (
                        None if is_sensor else "capture-calibration"
                    )
                ),
                "split": "development",
                "assignment_status": "verified",
                "assignment_evidence_ids": [evidence_id],
            }
        )
    qualification_payload = {
        "qualification_schema_version": 1,
        "artifact_kind": "recorded_dataset_qualification",
        "claim_eligibility": "not_sequence_evidence",
        "qualification_id": "pytest-qualified-window-v1",
        "purpose": "local-raw-pilot",
        "asset_root": "assets",
        "dataset_audit": {
            "dataset_id": "candidate-fixture",
            "accession_id": "candidate-fixture-20260726",
            "path": "assets/dataset-audit.json",
            "sha256": file_sha256(audit_path),
        },
        "evidence_records": [
            {
                "evidence_id": permission_evidence_id,
                "kind": "dataset_owner_correspondence",
                "reference": "private pytest permission correspondence",
                "observed_at_utc": "2026-07-31T00:00:00Z",
                "content_sha256": "e" * 64,
                "summary": "pytest owner authorizes local processing",
            },
            {
                "evidence_id": evidence_id,
                "kind": "dataset_owner_metadata",
                "reference": "metadata/owner-response.txt",
                "observed_at_utc": "2026-07-31T00:00:00Z",
                "content_sha256": None,
                "summary": "pytest owner-authorized mapping and local use",
            }
        ],
        "permissions": {
            scope: {
                "decision": (
                    "allowed"
                    if scope
                    in {
                        "local_decode_analysis",
                        "bounded_interval_extraction",
                        "local_derivative_construction",
                    }
                    else "unknown"
                ),
                "evidence_ids": (
                    [permission_evidence_id]
                    if scope
                    in {
                        "local_decode_analysis",
                        "bounded_interval_extraction",
                        "local_derivative_construction",
                    }
                    else []
                ),
            }
            for scope in (
                "local_decode_analysis",
                "bounded_interval_extraction",
                "local_derivative_construction",
                "publish_aggregate_metrics",
                "publish_frames_or_clips",
                "redistribution",
            )
        },
        "native_interval_semantics": {
            "status": "verified" if eligible else "unresolved",
            "class_zero_meaning": (
                "initial non-leak interval" if eligible else None
            ),
            "nominal_interval_seconds": 180.0 if eligible else None,
            "trim_start_seconds": 15.0 if eligible else None,
            "trim_end_seconds": 5.0 if eligible else None,
            "exceptions": [],
            "evidence_ids": [evidence_id] if eligible else [],
        },
        "assets": assets,
    }
    write_json_artifact(qualification_path, qualification_payload)

    spec_path = paths["spec_path"]
    assert isinstance(spec_path, Path)
    spec_payload = load_json_object(spec_path)
    spec_payload["preparation_spec_schema_version"] = 3
    spec_payload["dataset_qualification"] = {
        "path": "metadata/dataset-qualification.json",
        "sha256": file_sha256(qualification_path),
    }
    # Schema v3 permits an outcome-blind strict subset of qualified active
    # assets; keep the unused calibration capture active to exercise it.
    spec_payload["capture_groups"][1]["role"] = "development"
    spec_payload["exclusions"] = []
    spec_path.write_bytes(canonical_json_bytes(spec_payload))
    paths["qualification_path"] = qualification_path


def _qualify_v2_fixture(
    paths: dict[str, object],
    *,
    operations: tuple[str, ...] = ("decode_analysis",),
) -> None:
    """Upgrade a raw fixture to operation-aware preparation schema v4."""

    root = paths["root"]
    audit_path = paths["audit_path"]
    spec_path = paths["spec_path"]
    assert isinstance(root, Path)
    assert isinstance(audit_path, Path)
    assert isinstance(spec_path, Path)
    audit_payload = load_json_object(audit_path)
    audit_reference = {
        "dataset_id": "candidate-fixture",
        "accession_id": "candidate-fixture-20260726",
        "path": "assets/dataset-audit.json",
        "sha256": file_sha256(audit_path),
    }
    sensor = next(
        item
        for item in audit_payload["assets"]
        if item["relative_path"] == "sensor.mp4"
    )
    terms_snapshot = root / "metadata" / "fixture-terms.txt"
    provider_snapshot = root / "metadata" / "fixture-provider.txt"
    terms_snapshot.write_text(
        "Fixture terms authorize the declared local operations.\n",
        encoding="utf-8",
    )
    provider_snapshot.write_text(
        "sensor.mp4 belongs to capture-source.\n",
        encoding="utf-8",
    )
    preregistration_path = root / "metadata" / "project-preregistration.json"
    preregistration_payload = {
        "project_preregistration_schema_version": 1,
        "artifact_kind": "recorded_project_preregistration",
        "claim_eligibility": "not_sequence_evidence",
        "preregistration_id": "pytest-diagnostic-v1",
        "purpose": "diagnostic_raw_pilot",
        "created_at_utc": "2026-07-31T00:00:00Z",
        "dataset_audit": audit_reference,
        "operations": list(operations),
        "assignments": [
            {
                "relative_path": "sensor.mp4",
                "sha256": sensor["sha256"],
                "split": "development",
                "exclusion_reason": None,
            }
        ],
    }
    write_json_artifact(preregistration_path, preregistration_payload)

    required_scopes = {"local_decode_analysis"}
    if "extract_bounded_window" in operations:
        required_scopes.add("bounded_interval_extraction")
    if "construct_local_derivative" in operations:
        required_scopes.add("local_derivative_construction")
    qualification_path = root / "metadata" / "dataset-qualification-v2.json"
    qualification_payload = {
        "qualification_schema_version": 2,
        "artifact_kind": "recorded_dataset_qualification",
        "claim_eligibility": "not_sequence_evidence",
        "qualification_id": "pytest-diagnostic-qualification-v2",
        "asset_root": "assets",
        "purpose": "diagnostic_raw_pilot",
        "dataset_audit": audit_reference,
        "project_preregistration": {
            "path": "metadata/project-preregistration.json",
            "sha256": file_sha256(preregistration_path),
        },
        "evidence_records": [
            {
                "evidence_id": "formal-terms-v1",
                "kind": "formal_terms",
                "reference": "metadata/fixture-terms.txt",
                "observed_at_utc": "2026-07-31T00:00:00Z",
                "content_sha256": file_sha256(terms_snapshot),
                "summary": "fixture terms allow the declared local operations",
            },
            {
                "evidence_id": "owner-metadata-v2",
                "kind": "dataset_owner_metadata",
                "reference": "metadata/fixture-provider.txt",
                "observed_at_utc": "2026-07-31T00:00:00Z",
                "content_sha256": file_sha256(provider_snapshot),
                "summary": "fixture metadata identifies the sensor capture",
            },
        ],
        "permissions": {
            scope: {
                "decision": "allowed" if scope in required_scopes else "unknown",
                "evidence_ids": (
                    ["formal-terms-v1"] if scope in required_scopes else []
                ),
            }
            for scope in (
                "local_decode_analysis",
                "bounded_interval_extraction",
                "local_derivative_construction",
                "publish_aggregate_metrics",
                "publish_frames_or_clips",
                "redistribution",
            )
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
                "relative_path": item["relative_path"],
                "sha256": item["sha256"],
                "paper_group_id": "paper-sensor" if item is sensor else None,
                "capture_group_id": (
                    "capture-source" if item is sensor else None
                ),
                "gas_identity": "methane" if item is sensor else None,
                "fact_status": "verified" if item is sensor else "unresolved",
                "fact_evidence_ids": (
                    ["owner-metadata-v2"] if item is sensor else []
                ),
            }
            for item in audit_payload["assets"]
        ],
    }
    write_json_artifact(qualification_path, qualification_payload)

    spec_payload = load_json_object(spec_path)
    spec_payload["preparation_spec_schema_version"] = 4
    spec_payload["dataset_qualification"] = {
        "path": "metadata/dataset-qualification-v2.json",
        "sha256": file_sha256(qualification_path),
    }
    for sequence in spec_payload["sequences"]:
        sequence.setdefault("media_window_request", None)
    spec_path.write_bytes(canonical_json_bytes(spec_payload))
    paths["qualification_path"] = qualification_path
    paths["preregistration_path"] = preregistration_path


def _freeze_prepared_window(
    paths: dict[str, object],
    *,
    suffix: str,
):
    return freeze_recorded_suite(
        data_root=paths["root"],  # type: ignore[arg-type]
        preparation_report_path=paths["report"],  # type: ignore[arg-type]
        preparation_report_sha256=file_sha256(  # type: ignore[arg-type]
            paths["report"]
        ),
        suite_id=f"qualified-{suffix}",
        split="development",
        frozen_at_utc="2026-07-31T00:00:00Z",
        frozen_before_model_commit="b" * 40,
        freeze_provenance="pytest qualification-bound freeze",
        suite_output_path=Path(f"manifests/{suffix}.json"),
        freeze_record_output_path=Path(f"metadata/{suffix}.freeze.json"),
    )


def test_raw_preparation_emits_idempotent_truth_free_v3_manifest(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, emulated=False)

    first = _prepare(paths)
    first_report_bytes = paths["report"].read_bytes()  # type: ignore[union-attr]
    second = _prepare(paths)
    manifest_path = paths["manifests"] / "prepared-1.json"  # type: ignore[operator]
    manifest = parse_recorded_sequence_manifest(
        load_json_object(manifest_path)
    )

    assert first == second
    assert paths["report"].read_bytes() == first_report_bytes  # type: ignore[union-attr]
    assert manifest.manifest_schema_version == 3
    assert manifest.sequence_id == "prepared-1"
    assert manifest.sensor_input is not None
    assert manifest.sensor_input.kind == "raw_recorded"
    assert manifest.evaluation_truth.evidence_grade == "none"
    assert first.to_payload()["claim_eligibility"] == (
        "not_sequence_evidence"
    )
    assert set(first.to_payload()) == {
        "report_schema_version",
        "artifact_kind",
        "claim_eligibility",
        "preparation_id",
        "preparation_spec",
        "preparation_profile",
        "producing_commit",
        "sequences",
    }
    assert not list(
        (paths["root"] / "assets").rglob("*.plmcal")  # type: ignore[operator]
    )
    preparation_payload = load_preparation_spec(
        paths["spec_path"]  # type: ignore[arg-type]
    ).to_payload()
    first_sequence = preparation_payload["sequences"][0]  # type: ignore[index]
    assert "media_window_request" not in first_sequence


def test_windowed_preparation_resolves_v2_request_and_freezes_v4(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, emulated=False, windowed=True)

    report = _prepare(paths)
    preparation_payload = load_preparation_spec(
        paths["spec_path"]  # type: ignore[arg-type]
    ).to_payload()
    assert preparation_payload["sequences"][0][  # type: ignore[index]
        "media_window_request"
    ]["expected_source_start_frame_index"] == 1
    manifest_path = paths["manifests"] / "prepared-1.json"  # type: ignore[operator]
    manifest = parse_recorded_sequence_manifest(
        load_json_object(manifest_path)
    )
    assert manifest.manifest_schema_version == 4
    assert manifest.media_window is not None
    window = manifest.media_window
    assert (
        window.source_start_frame_index,
        window.source_end_frame_index_exclusive,
    ) == (1, 4)
    assert window.first_included_timestamp_seconds == pytest.approx(0.2)
    assert window.last_included_timestamp_seconds == pytest.approx(0.6)
    assert window.parent_media_sha256 == file_sha256(  # type: ignore[arg-type]
        paths["media_path"]
    )
    frames, _fps = load_video_frame_window(
        paths["media_path"],  # type: ignore[arg-type]
        start_frame_index=1,
        end_frame_index_exclusive=4,
    )
    assert window.decoded_window_sha256 == (
        decoded_frame_content_sha256(frames)
    )

    loaded = load_recorded_sequence(manifest_path)
    assert loaded.timeline.end_time_seconds == pytest.approx(0.4)
    frozen = freeze_recorded_suite(
        data_root=paths["root"],  # type: ignore[arg-type]
        preparation_report_path=paths["report"],  # type: ignore[arg-type]
        preparation_report_sha256=file_sha256(  # type: ignore[arg-type]
            paths["report"]
        ),
        suite_id="windowed-development-v1",
        split="development",
        frozen_at_utc="2026-07-31T00:00:00Z",
        frozen_before_model_commit="b" * 40,
        freeze_provenance="pytest preregistered window freeze",
        suite_output_path=Path("manifests/windowed-development.json"),
        freeze_record_output_path=Path(
            "metadata/windowed-development.freeze.json"
        ),
    )
    assert frozen.sequence_count == 1
    assert report.sequences[0].sequence_id == "prepared-1"


def test_qualified_v3_raw_window_prepares_subset_and_freezes(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, emulated=False, windowed=True)
    _qualify_windowed_fixture(paths)

    preparation = load_preparation_spec(paths["spec_path"])
    payload = preparation.to_payload()
    assert payload["dataset_qualification"] == {
        "path": "metadata/dataset-qualification.json",
        "sha256": file_sha256(paths["qualification_path"]),
    }
    report = _prepare(paths)
    frozen = _freeze_prepared_window(paths, suffix="qualified-window")

    assert report.sequences[0].sequence_id == "prepared-1"
    assert frozen.sequence_count == 1


def test_qualified_v4_whole_asset_prepares_exact_diagnostic_selection(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, emulated=False)
    _qualify_v2_fixture(paths)

    preparation = load_preparation_spec(paths["spec_path"])
    payload = preparation.to_payload()
    assert payload["preparation_spec_schema_version"] == 4
    assert payload["sequences"][0]["media_window_request"] is None
    report = _prepare(paths)
    manifest_path = paths["manifests"] / "prepared-1.json"
    manifest = parse_recorded_sequence_manifest(
        load_json_object(manifest_path)
    )

    assert report.sequences[0].sequence_id == "prepared-1"
    assert manifest.media_window is None
    assert manifest.manifest_schema_version == 3
    frozen = _freeze_prepared_window(paths, suffix="qualified-v4-whole")
    assert frozen.sequence_count == 1


def test_v4_rejects_unpreregistered_window_extraction_before_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import plume_focus.recorded_preparation as preparation_module

    paths = _fixture(tmp_path, emulated=False, windowed=True)
    _qualify_v2_fixture(paths, operations=("decode_analysis",))
    decoded = False

    def forbidden_window_decode(*args: object, **kwargs: object):
        nonlocal decoded
        decoded = True
        raise AssertionError("operation binding must precede decode")

    monkeypatch.setattr(
        preparation_module,
        "load_video_frame_window",
        forbidden_window_decode,
    )
    with pytest.raises(ValueError, match="not project-preregistered"):
        _prepare(paths)
    assert not decoded


def test_v4_requires_qualification_v2(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, emulated=False, windowed=True)
    _qualify_windowed_fixture(paths)
    spec_path = paths["spec_path"]
    assert isinstance(spec_path, Path)
    spec = load_json_object(spec_path)
    spec["preparation_spec_schema_version"] = 4
    spec_path.write_bytes(canonical_json_bytes(spec))

    with pytest.raises(
        ValueError,
        match="preparation schema v4 requires qualification schema v2",
    ):
        _prepare(paths)


def test_v4_rejects_calibration_derivative_until_roles_are_modeled(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, emulated=True)
    _qualify_v2_fixture(
        paths,
        operations=("decode_analysis", "construct_local_derivative"),
    )

    with pytest.raises(
        ValueError,
        match="supports raw_recorded input only",
    ):
        _prepare(paths)


def test_v4_preparation_cannot_silently_drop_an_active_assignment(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, emulated=False)
    _qualify_v2_fixture(paths)
    preregistration_path = paths["preregistration_path"]
    qualification_path = paths["qualification_path"]
    spec_path = paths["spec_path"]
    audit_path = paths["audit_path"]
    assert isinstance(preregistration_path, Path)
    assert isinstance(qualification_path, Path)
    assert isinstance(spec_path, Path)
    assert isinstance(audit_path, Path)
    audit = load_json_object(audit_path)
    calibration = next(
        item
        for item in audit["assets"]
        if item["relative_path"] == "leak-off.mp4"
    )
    preregistration = load_json_object(preregistration_path)
    preregistration["assignments"].append(
        {
            "relative_path": "leak-off.mp4",
            "sha256": calibration["sha256"],
            "split": "development",
            "exclusion_reason": None,
        }
    )
    preregistration_path.write_bytes(
        canonical_json_bytes(preregistration)
    )
    qualification = load_json_object(qualification_path)
    qualification["project_preregistration"]["sha256"] = file_sha256(
        preregistration_path
    )
    calibration_fact = next(
        item
        for item in qualification["provider_assets"]
        if item["relative_path"] == "leak-off.mp4"
    )
    calibration_fact.update(
        {
            "capture_group_id": "capture-calibration",
            "gas_identity": "methane",
            "fact_status": "verified",
            "fact_evidence_ids": ["owner-metadata-v2"],
        }
    )
    qualification_path.write_bytes(canonical_json_bytes(qualification))
    spec = load_json_object(spec_path)
    spec["dataset_qualification"]["sha256"] = file_sha256(
        qualification_path
    )
    spec_path.write_bytes(canonical_json_bytes(spec))

    with pytest.raises(ValueError, match="exactly cover active"):
        _prepare(paths)


def test_v3_coarse_rights_are_insufficient_and_block_before_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import plume_focus.recorded_preparation as preparation_module

    paths = _fixture(tmp_path, emulated=False, windowed=True)
    _qualify_windowed_fixture(paths, eligible=False)
    decoded = False

    def forbidden_window_decode(*args: object, **kwargs: object):
        nonlocal decoded
        decoded = True
        raise AssertionError("qualification blockers must precede decode")

    monkeypatch.setattr(
        preparation_module,
        "load_video_frame_window",
        forbidden_window_decode,
    )
    with pytest.raises(ValueError, match="not eligible"):
        _prepare(paths)
    assert not decoded


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("capture_group_id", "different-capture"),
        ("split", "held-out"),
    ],
)
def test_v3_selected_group_and_split_must_match_qualification(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    paths = _fixture(tmp_path, emulated=False, windowed=True)
    _qualify_windowed_fixture(paths)
    qualification_path = paths["qualification_path"]
    spec_path = paths["spec_path"]
    assert isinstance(qualification_path, Path)
    assert isinstance(spec_path, Path)
    qualification = load_json_object(qualification_path)
    sensor_asset = next(
        asset
        for asset in qualification["assets"]
        if asset["relative_path"] == "sensor.mp4"
    )
    sensor_asset[field] = value
    qualification_path.write_bytes(canonical_json_bytes(qualification))
    spec = load_json_object(spec_path)
    spec["dataset_qualification"]["sha256"] = file_sha256(
        qualification_path
    )
    spec_path.write_bytes(canonical_json_bytes(spec))

    with pytest.raises(ValueError, match="capture group and split"):
        _prepare(paths)


def test_v3_rejects_same_byte_media_at_unqualified_alias_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import plume_focus.recorded_preparation as preparation_module

    paths = _fixture(tmp_path, emulated=False, windowed=True)
    _qualify_windowed_fixture(paths)
    media_path = paths["media_path"]
    spec_path = paths["spec_path"]
    assert isinstance(media_path, Path)
    assert isinstance(spec_path, Path)
    alias_path = media_path.parent / "alias" / media_path.name
    alias_path.parent.mkdir()
    shutil.copyfile(media_path, alias_path)
    spec = load_json_object(spec_path)
    spec["sequences"][0]["manifest_template"]["media"]["path"] = (
        "assets/alias/sensor.mp4"
    )
    spec_path.write_bytes(canonical_json_bytes(spec))
    decoded = False

    def forbidden_window_decode(*args: object, **kwargs: object):
        nonlocal decoded
        decoded = True
        raise AssertionError("alias binding must fail before media decode")

    monkeypatch.setattr(
        preparation_module,
        "load_video_frame_window",
        forbidden_window_decode,
    )
    with pytest.raises(ValueError, match="path and SHA-256"):
        _prepare(paths)
    assert not decoded


@pytest.mark.parametrize("target", ["audit_path", "qualification_path"])
@pytest.mark.parametrize("phase", ["prepare", "freeze"])
def test_v3_rejects_audit_and_qualification_tamper(
    tmp_path: Path,
    target: str,
    phase: str,
) -> None:
    paths = _fixture(
        tmp_path / f"{target}-{phase}",
        emulated=False,
        windowed=True,
    )
    _qualify_windowed_fixture(paths)
    if phase == "freeze":
        _prepare(paths)
    artifact = paths[target]
    assert isinstance(artifact, Path)
    artifact.write_bytes(artifact.read_bytes() + b" ")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        if phase == "prepare":
            _prepare(paths)
        else:
            _freeze_prepared_window(paths, suffix=f"tampered-{target}")


def test_v3_requires_per_sequence_selection_provenance(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, emulated=False, windowed=True)
    _qualify_windowed_fixture(paths)
    spec_path = paths["spec_path"]
    assert isinstance(spec_path, Path)
    payload = load_json_object(spec_path)
    del payload["sequences"][0]["media_window_request"][
        "selection_provenance"
    ]
    spec_path.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(ValueError, match="selection_provenance"):
        load_preparation_spec(spec_path)


def test_v1_and_v2_payloads_do_not_gain_qualification_fields(
    tmp_path: Path,
) -> None:
    for version, windowed in ((1, False), (2, True)):
        paths = _fixture(
            tmp_path / f"v{version}",
            emulated=False,
            windowed=windowed,
        )
        source_payload = load_json_object(paths["spec_path"])
        parsed_payload = load_preparation_spec(
            paths["spec_path"]
        ).to_payload()
        assert set(parsed_payload) == set(source_payload)
        assert set(parsed_payload["sequences"][0]) == set(
            source_payload["sequences"][0]
        )
        assert "dataset_qualification" not in parsed_payload


def test_windowed_preparation_uses_full_authoritative_timestamp_clock(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, emulated=False, windowed=True)
    timestamp_path = (
        paths["root"] / "metadata" / "timestamps.csv"  # type: ignore[operator]
    )
    timestamp_path.write_text(
        "frame_index,timestamp_seconds\n"
        "0,0.0\n1,0.1\n2,0.3\n3,0.6\n4,0.8\n5,1.1\n",
        encoding="utf-8",
    )
    spec_payload = load_json_object(paths["spec_path"])  # type: ignore[arg-type]
    sequence = spec_payload["sequences"][0]  # type: ignore[index]
    sequence["manifest_template"]["frame_clock"] = {
        "mode": "timestamp_csv",
        "path": "metadata/timestamps.csv",
        "sha256": file_sha256(timestamp_path),
        "provenance": "pytest authoritative nonuniform clock",
    }
    request = sequence["media_window_request"]
    request["requested_start_time_seconds"] = 0.15
    request["requested_end_time_seconds"] = 0.9
    request["expected_source_start_frame_index"] = 2
    request["expected_source_end_frame_index_exclusive"] = 5
    paths["spec_path"].write_bytes(  # type: ignore[union-attr]
        canonical_json_bytes(spec_payload)
    )

    _prepare(paths)

    manifest = parse_recorded_sequence_manifest(
        load_json_object(
            paths["manifests"] / "prepared-1.json"  # type: ignore[operator]
        )
    )
    assert manifest.media_window is not None
    assert manifest.media_window.source_start_frame_index == 2
    assert manifest.media_window.source_end_frame_index_exclusive == 5
    assert manifest.media_window.first_included_timestamp_seconds == 0.3
    assert manifest.media_window.last_included_timestamp_seconds == 0.8


def test_windowed_preparation_rejects_index_drift_before_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import plume_focus.recorded_preparation as preparation_module

    paths = _fixture(tmp_path, emulated=False, windowed=True)
    spec_payload = load_json_object(paths["spec_path"])  # type: ignore[arg-type]
    request = spec_payload["sequences"][0][  # type: ignore[index]
        "media_window_request"
    ]
    request["expected_source_start_frame_index"] = 0
    paths["spec_path"].write_bytes(  # type: ignore[union-attr]
        canonical_json_bytes(spec_payload)
    )
    decoded = False

    def forbidden_window_decode(*args: object, **kwargs: object):
        nonlocal decoded
        decoded = True
        raise AssertionError("index drift must fail before decoding")

    monkeypatch.setattr(
        preparation_module,
        "load_video_frame_window",
        forbidden_window_decode,
    )

    with pytest.raises(ValueError, match="preregistered source frame"):
        _prepare(paths)
    assert not decoded


def test_windowed_preparation_rejects_end_beyond_parent_extent(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, emulated=False, windowed=True)
    spec_payload = load_json_object(paths["spec_path"])  # type: ignore[arg-type]
    request = spec_payload["sequences"][0][  # type: ignore[index]
        "media_window_request"
    ]
    request["requested_end_time_seconds"] = 1.21
    request["expected_source_end_frame_index_exclusive"] = 6
    paths["spec_path"].write_bytes(  # type: ignore[union-attr]
        canonical_json_bytes(spec_payload)
    )

    with pytest.raises(ValueError, match="parent capture extent"):
        _prepare(paths)


def test_windowed_preparation_rejects_start_beyond_last_timestamp(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, emulated=False, windowed=True)
    spec_payload = load_json_object(paths["spec_path"])  # type: ignore[arg-type]
    request = spec_payload["sequences"][0][  # type: ignore[index]
        "media_window_request"
    ]
    request["requested_start_time_seconds"] = 1.01
    request["requested_end_time_seconds"] = 1.2
    request["expected_source_start_frame_index"] = 3
    request["expected_source_end_frame_index_exclusive"] = 6
    paths["spec_path"].write_bytes(  # type: ignore[union-attr]
        canonical_json_bytes(spec_payload)
    )

    with pytest.raises(ValueError, match="final source timestamp"):
        _prepare(paths)


def test_windowed_preparation_rejects_sensor_emulation(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, emulated=True, windowed=True)

    with pytest.raises(ValueError, match="does not support sensor emulation"):
        _prepare(paths)


def test_windowed_freeze_rejects_rewritten_resolved_window(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, emulated=False, windowed=True)
    _prepare(paths)
    manifest_path = paths["manifests"] / "prepared-1.json"  # type: ignore[operator]
    manifest_payload = load_json_object(manifest_path)
    manifest_payload["media_window"][  # type: ignore[index]
        "last_included_timestamp_seconds"
    ] = 0.4
    manifest_path.write_bytes(canonical_json_bytes(manifest_payload))
    report_payload = load_json_object(paths["report"])  # type: ignore[arg-type]
    report_payload["sequences"][0]["manifest_sha256"] = (  # type: ignore[index]
        canonical_json_sha256(manifest_payload)
    )
    paths["report"].write_bytes(  # type: ignore[union-attr]
        canonical_json_bytes(report_payload)
    )

    with pytest.raises(ValueError, match="immutable preparation template"):
        freeze_recorded_suite(
            data_root=paths["root"],  # type: ignore[arg-type]
            preparation_report_path=paths["report"],  # type: ignore[arg-type]
            preparation_report_sha256=file_sha256(  # type: ignore[arg-type]
                paths["report"]
            ),
            suite_id="tampered-window-v1",
            split="development",
            frozen_at_utc="2026-07-31T00:00:00Z",
            frozen_before_model_commit="b" * 40,
            freeze_provenance="pytest tamper rejection",
            suite_output_path=Path("manifests/tampered-suite.json"),
            freeze_record_output_path=Path(
                "metadata/tampered-suite.freeze.json"
            ),
        )


def test_emulated_preparation_decodes_shared_calibration_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import plume_focus.recorded_preparation as preparation_module

    paths = _fixture(tmp_path, emulated=True, sequence_count=2)
    decoded: list[Path] = []
    original_loader = preparation_module.load_video_frames

    def tracking_loader(
        path: Path,
        *,
        max_frames: int | None,
    ):
        decoded.append(Path(path).resolve())
        return original_loader(path, max_frames=max_frames)

    monkeypatch.setattr(
        preparation_module,
        "load_video_frames",
        tracking_loader,
    )

    report = _prepare(paths)

    assert decoded == [paths["calibration_path"].resolve()]  # type: ignore[union-attr]
    assert len(report.sequences) == 2
    assert paths["media_path"].resolve() not in decoded  # type: ignore[union-attr]
    artifacts = list(
        (paths["root"] / "assets" / "prepared").rglob("*.plmcal")  # type: ignore[operator]
    )
    records = list(
        (paths["root"] / "assets" / "prepared").rglob(  # type: ignore[operator]
            "*-calibration.json"
        )
    )
    assert len(artifacts) == 1
    assert len(records) == 2
    calibration = load_fixed_calibration(artifacts[0])
    assert calibration.sample_count == 3
    assert all(
        load_calibration_preparation_record(record).claim_eligibility
        == "not_sequence_evidence"
        for record in records
    )
    for result in report.sequences:
        manifest_path = paths["root"] / result.path  # type: ignore[operator]
        manifest = parse_recorded_sequence_manifest(
            load_json_object(manifest_path)
        )
        assert manifest.sensor_input is not None
        assert manifest.sensor_input.kind == (
            "structure_suppressed_emulation"
        )
        assert manifest.sensor_input.frozen_calibration is not None
        assert manifest.evaluation_truth.evidence_grade == "none"
        loaded = load_recorded_sequence(manifest_path)
        assert loaded.truth_source is None
        assert loaded.timeline.calibration.sample_count == 3  # type: ignore[attr-defined]


def test_preparation_rejects_audit_hash_or_unverified_rights(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, emulated=False)
    spec_payload = load_json_object(paths["spec_path"])  # type: ignore[arg-type]
    spec_payload["dataset_audit"]["sha256"] = "0" * 64  # type: ignore[index]
    for sequence in spec_payload["sequences"]:  # type: ignore[union-attr]
        sequence["manifest_template"]["dataset_lineage"][  # type: ignore[index]
            "audit_sha256"
        ] = "0" * 64
    paths["spec_path"].write_text(  # type: ignore[union-attr]
        json.dumps(spec_payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _prepare(paths)
    assert not paths["report"].exists()  # type: ignore[union-attr]


def test_preparation_refuses_different_existing_manifest(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, emulated=False)
    _prepare(paths)
    manifest_path = paths["manifests"] / "prepared-1.json"  # type: ignore[operator]
    original = manifest_path.read_bytes()
    profile_payload = load_json_object(paths["profile_path"])  # type: ignore[arg-type]
    profile_payload["sensor_input_provenance"] = "changed provenance"
    paths["profile_path"].write_text(  # type: ignore[union-attr]
        json.dumps(profile_payload),
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _prepare(paths)
    assert manifest_path.read_bytes() == original


def test_preparation_rejects_changed_audited_asset_before_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import plume_focus.recorded_preparation as preparation_module

    paths = _fixture(tmp_path, emulated=True)
    calibration_path = paths["calibration_path"]
    calibration_path.write_bytes(  # type: ignore[union-attr]
        calibration_path.read_bytes() + b"changed"  # type: ignore[union-attr]
    )
    decode_calls = 0

    def forbidden_decode(*args: object, **kwargs: object):
        nonlocal decode_calls
        decode_calls += 1
        raise AssertionError("tampered media must fail before decoding")

    monkeypatch.setattr(
        preparation_module,
        "load_video_frames",
        forbidden_decode,
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _prepare(paths)
    assert decode_calls == 0


def test_preparation_requires_objective_disposition_for_every_audited_video(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, emulated=False)
    spec_payload = load_json_object(paths["spec_path"])  # type: ignore[arg-type]
    spec_payload["capture_groups"] = [  # type: ignore[index]
        group
        for group in spec_payload["capture_groups"]  # type: ignore[index]
        if group["capture_group_id"] != "capture-calibration"
    ]
    spec_payload["exclusions"] = []
    paths["spec_path"].write_text(  # type: ignore[union-attr]
        json.dumps(spec_payload),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="every audited video must belong",
    ):
        _prepare(paths)
    assert not paths["report"].exists()  # type: ignore[union-attr]


def test_preparation_rejects_silently_unused_nonexcluded_video(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, emulated=False)
    spec_payload = load_json_object(paths["spec_path"])  # type: ignore[arg-type]
    for group in spec_payload["capture_groups"]:  # type: ignore[union-attr]
        if group["capture_group_id"] == "capture-calibration":
            group["role"] = "development"
    spec_payload["exclusions"] = []
    paths["spec_path"].write_text(  # type: ignore[union-attr]
        json.dumps(spec_payload),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="every non-excluded asset must be used",
    ):
        _prepare(paths)


def test_preparation_allows_declared_duplicate_bytes_only_when_excluded(
    tmp_path: Path,
) -> None:
    paths = _fixture(
        tmp_path,
        emulated=False,
        duplicate_calibration=True,
    )

    report = _prepare(paths)

    assert len(report.sequences) == 1


def test_preparation_report_must_remain_in_recorded_data_root(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, emulated=False)
    paths["report"] = tmp_path / "outside" / "preparation-report.json"

    with pytest.raises(ValueError, match="report_path must remain"):
        _prepare(paths)


def test_preparation_requires_the_audited_decoder_version(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, emulated=True)
    audit_payload = load_json_object(
        paths["audit_path"]  # type: ignore[arg-type]
    )
    audit_payload["opencv_version"] = "different-decoder-version"
    paths["audit_path"].write_text(  # type: ignore[union-attr]
        json.dumps(audit_payload),
        encoding="utf-8",
    )
    audit_sha256 = file_sha256(
        paths["audit_path"]  # type: ignore[arg-type]
    )
    spec_payload = load_json_object(
        paths["spec_path"]  # type: ignore[arg-type]
    )
    spec_payload["dataset_audit"]["sha256"] = audit_sha256  # type: ignore[index]
    for sequence in spec_payload["sequences"]:  # type: ignore[union-attr]
        sequence["manifest_template"]["dataset_lineage"][  # type: ignore[index]
            "audit_sha256"
        ] = audit_sha256
    paths["spec_path"].write_text(  # type: ignore[union-attr]
        json.dumps(spec_payload),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="decoder environment must match",
    ):
        _prepare(paths)


def test_preparation_verifies_exact_audited_calibration_pixels(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, emulated=True)
    audit_payload = load_json_object(
        paths["audit_path"]  # type: ignore[arg-type]
    )
    for asset in audit_payload["assets"]:  # type: ignore[union-attr]
        if asset["relative_path"] == "leak-off.mp4":
            asset["decoded_content_sha256"] = "f" * 64
    paths["audit_path"].write_bytes(  # type: ignore[union-attr]
        canonical_json_bytes(audit_payload)
    )
    audit_sha256 = file_sha256(
        paths["audit_path"]  # type: ignore[arg-type]
    )
    spec_payload = load_json_object(
        paths["spec_path"]  # type: ignore[arg-type]
    )
    spec_payload["dataset_audit"]["sha256"] = audit_sha256  # type: ignore[index]
    for sequence in spec_payload["sequences"]:  # type: ignore[union-attr]
        sequence["manifest_template"]["dataset_lineage"][  # type: ignore[index]
            "audit_sha256"
        ] = audit_sha256
    paths["spec_path"].write_bytes(  # type: ignore[union-attr]
        canonical_json_bytes(spec_payload)
    )

    with pytest.raises(
        ValueError,
        match="calibration decoded content differs",
    ):
        _prepare(paths)


def test_audit_prepare_freeze_pipeline_runs_end_to_end(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, emulated=True, sequence_count=2)
    report = _prepare(paths)

    frozen = freeze_recorded_suite(
        data_root=paths["root"],  # type: ignore[arg-type]
        preparation_report_path=paths["report"],  # type: ignore[arg-type]
        preparation_report_sha256=file_sha256(
            paths["report"]  # type: ignore[arg-type]
        ),
        suite_id="prepared-development-v1",
        split="development",
        frozen_at_utc="2026-07-26T12:00:00Z",
        frozen_before_model_commit="b" * 40,
        freeze_provenance="pytest pre-policy suite freeze",
        suite_output_path=Path(
            "manifests/prepared-development-suite.json"
        ),
        freeze_record_output_path=Path(
            "metadata/prepared-development-suite.freeze.json"
        ),
    )
    validation = validate_recorded_suite(
        frozen.suite_manifest_path
    )

    assert len(report.sequences) == 2
    assert frozen.sequence_count == 2
    assert validation["valid"] is True
    assert validation["sequence_count"] == 2
    assert all(
        sequence["claim_eligibility"]
        == "no_spatial_evaluation_truth"
        for sequence in validation["sequences"]
    )
