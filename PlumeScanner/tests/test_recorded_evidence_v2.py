from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from plume_focus.experiment import TimelineSource
from plume_focus.experiment_runner import ExperimentRequest, run_experiment
from plume_focus.recorded_calibration import write_fixed_calibration
from plume_focus.recorded_contracts import assess_recorded_claim
from plume_focus.recorded_dataset import (
    DatasetAccession,
    audit_dataset_directory,
)
from plume_focus.recorded_manifest_io import (
    canonical_json_sha256,
    file_sha256,
)
from plume_focus.recorded_sources import (
    load_recorded_sequence,
    parse_recorded_sequence_manifest,
    validate_recorded_sequence,
    validate_recorded_suite_pair,
)
from plume_focus.sensor_emulation import SensorEmulationTimeline


class _AccessLoggingTimeline:
    def __init__(self, source: TimelineSource) -> None:
        self.source = source
        self.frame_calls: list[float] = []
        self.sample_call_count = 0

    @property
    def shape(self) -> tuple[int, int]:
        return self.source.shape

    @property
    def fps(self) -> float:
        return self.source.fps

    @property
    def end_time_seconds(self) -> float | None:
        return self.source.end_time_seconds

    def frame_at_seconds(self, time_seconds: float) -> np.ndarray:
        self.frame_calls.append(time_seconds)
        return self.source.frame_at_seconds(time_seconds)

    def samples_at_seconds(
        self,
        times_seconds: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
    ) -> np.ndarray:
        self.sample_call_count += 1
        return self.source.samples_at_seconds(
            times_seconds,
            rows,
            cols,
        )


class _NoopObserver:
    def start(self, _context: object) -> None:
        pass

    def observe(self, _step: object) -> None:
        pass

    def close(self) -> None:
        pass


def _write_video(
    path: Path,
    frames: tuple[np.ndarray, ...],
    *,
    fps: float = 5.0,
) -> None:
    rows, cols = frames[0].shape
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (cols, rows),
    )
    if not writer.isOpened():
        raise RuntimeError("test video writer did not open")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    finally:
        writer.release()


def _v2_fixture(
    tmp_path: Path,
    *,
    sensor_kind: str = "structure_suppressed_emulation",
    truth_grade: str = "silver_pseudo",
    scanner_kind: str = "counterfactual_virtual",
    rights_blocked: bool = False,
    split: str = "development",
    capture_group_id: str = "fixture-acquisition-001",
    sensor_offset: int = 0,
) -> tuple[Path, dict[str, object]]:
    data_root = tmp_path / "recorded"
    asset_directory = data_root / "assets"
    manifest_directory = data_root / "manifests"
    asset_directory.mkdir(parents=True)
    manifest_directory.mkdir()

    sensor_path = asset_directory / "sensor.mp4"
    calibration_path = asset_directory / "leak-off.mp4"
    truth_path = asset_directory / "truth.npz"
    truth_protocol_path = asset_directory / "truth-protocol.json"
    sensor_frames = tuple(
        np.full(
            (12, 16),
            20 + sensor_offset + index * 12,
            dtype=np.uint8,
        )
        for index in range(4)
    )
    calibration_frames = tuple(
        np.full((12, 16), value, dtype=np.uint8)
        for value in (19, 20, 21)
    )
    truth_frames = np.zeros((4, 12, 16), dtype=np.uint8)
    for index in range(4):
        truth_frames[index, 4:8, 2 + index : 5 + index] = 255
    _write_video(sensor_path, sensor_frames)
    _write_video(calibration_path, calibration_frames)
    np.savez_compressed(truth_path, frames=truth_frames)
    truth_protocol_path.write_text(
        json.dumps(
            {
                "protocol_schema_version": 1,
                "method_id": "fixture-truth-method-v1",
                "purpose": "pytest evaluation-only truth fixture",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    media_sha256 = file_sha256(sensor_path)
    accession_directory = data_root / "accessions"
    accession_directory.mkdir()
    accession_path = accession_directory / "fixture-accession.json"
    accession_payload = {
        "dataset_accession_schema_version": 1,
        "dataset_id": "candidate-fixture",
        "accession_id": "candidate-fixture-20260726",
        "source_uri": "https://example.test/dataset",
        "accessed_at_utc": "2026-07-26T00:00:00Z",
        "rights_status": (
            "unverified" if rights_blocked else "verified"
        ),
        "rights_provenance": "pytest declaration",
        "rights_uri": (
            None if rights_blocked else "https://example.test/terms"
        ),
    }
    accession_path.write_text(
        json.dumps(accession_payload, sort_keys=True),
        encoding="utf-8",
    )
    accession = DatasetAccession(
        dataset_id="candidate-fixture",
        accession_id="candidate-fixture-20260726",
        source_uri="https://example.test/dataset",
        accessed_at_utc="2026-07-26T00:00:00Z",
        rights_status=(
            "unverified" if rights_blocked else "verified"
        ),
        rights_provenance="pytest declaration",
        rights_uri=(
            None if rights_blocked else "https://example.test/terms"
        ),
    )
    audit_path = asset_directory / "dataset-audit.json"
    audit_payload = audit_dataset_directory(
        asset_directory,
        accession,
        accession_manifest_sha256=file_sha256(accession_path),
    ).to_payload()
    audit_path.write_text(
        json.dumps(audit_payload, indent=2),
        encoding="utf-8",
    )

    if sensor_kind == "raw_recorded":
        sensor_input: dict[str, object] = {
            "kind": "raw_recorded",
            "provenance": "unaltered recorded intensity fixture",
        }
    else:
        sensor_input = {
            "kind": "structure_suppressed_emulation",
            "provenance": "pytest fixed leak-off sensor emulation",
            "emulation_profile": {
                "profile_id": "structure_suppressed_fixture_v1",
                "residual_mode": "bright",
                "provenance": "pytest declared profile",
                "noise_floor_raw_units": 1.0,
                "output_units_per_noise_sigma": 16.0,
                "smoothing_time_constant_seconds": None,
                "profile_schema_version": 1,
            },
            "leak_off_calibration": {
                "path": "assets/leak-off.mp4",
                "sha256": file_sha256(calibration_path),
                "provenance": "separate pre-run pytest leak-off clip",
                "calibration_scope": "development_only",
            },
        }

    payload: dict[str, object] = {
        "manifest_schema_version": 2,
        "sequence_id": "recorded-v2-fixture",
        "split": split,
        "split_provenance": "assigned before policy execution",
        "scenario": "methane",
        "dataset_lineage": {
            "dataset_id": "candidate-fixture",
            "accession_id": "candidate-fixture-20260726",
            "audit_path": "assets/dataset-audit.json",
            "audit_sha256": file_sha256(audit_path),
            "source_asset_sha256": media_sha256,
        },
        "capture_group": {
            "capture_group_id": capture_group_id,
            "grouping_provenance": "whole acquisition is one split unit",
        },
        "media": {
            "path": "assets/sensor.mp4",
            "sha256": media_sha256,
            "capture_provenance": "pytest external-video fixture",
        },
        "sensor_input": sensor_input,
        "frame_clock": {
            "mode": "constant_fps",
            "fps": 5.0,
            "provenance": "pytest writer clock",
        },
        "normalization": {
            "mode": "uint8_identity",
            "provenance": "decoded uint8 values",
        },
        "calibration": {
            "sensor_id": "test-ogi",
            "signal_units": "compressed_8bit_ogi_display_intensity",
            "provenance": "pytest non-radiometric fixture",
        },
        "wind": {
            "mode": "none",
            "units": "px_per_second",
            "provenance": "no synchronized wind",
            "conversion_provenance": "not applicable",
        },
        "source_assumption": {
            "mode": "known_source",
            "provenance": "source declared before evaluation",
            "source_point": [6.0, 2.0],
            "initial_roi": [2, 0, 10, 12],
        },
        "scanner": {
            "provenance": "pytest scanner evidence declaration",
            "evidence_kind": scanner_kind,
            "scan_mode": "column",
            "scan_rate_hz": 100.0,
            "sample_stride_px": 2,
            "vertical_pass_seconds": 0.01,
            "horizontal_step_seconds": 0.0,
            "column_stride_px": 1,
            "column_mass_fraction": 0.8,
            "row_mass_fraction": 0.85,
            "minimum_sweep_columns": 2,
            "minimum_sweep_rows": 4,
            "maximum_sweep_columns": 8,
        },
        "evaluation_truth": {
            "path": "assets/truth.npz",
            "sha256": file_sha256(truth_path),
            "truth_kind": "probability_weight",
            "storage_format": "lossless_npz_uint8",
            "evidence_grade": truth_grade,
            "method_id": "fixture-truth-method-v1",
            "protocol_path": "assets/truth-protocol.json",
            "protocol_sha256": file_sha256(truth_protocol_path),
            "annotation_provenance": (
                "pytest pseudo protocol"
                if truth_grade == "silver_pseudo"
                else "pytest independent human review protocol"
            ),
        },
        "model_profile_id": "uncalibrated_video_v1",
    }
    manifest_path = manifest_directory / "recorded-v2-fixture.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return manifest_path, payload


def test_v2_emulated_input_runs_and_reports_complete_evidence(
    tmp_path: Path,
) -> None:
    manifest_path, _payload = _v2_fixture(tmp_path)

    sequence = load_recorded_sequence(manifest_path)
    validation = validate_recorded_sequence(manifest_path)
    report = run_experiment(
        ExperimentRequest(
            recorded_sequence=sequence,
            scenario="methane",
            max_scans=1,
        )
    ).report

    assert isinstance(sequence.timeline, SensorEmulationTimeline)
    assert validation["claim_eligibility"] == (
        "diagnostic_only_pseudo_truth"
    )
    limitations = validation["claim_limitations"]
    assert (
        "pseudo_truth_is_diagnostic_and_not_independently_reviewed"
        in limitations
    )
    assert (
        "scanner_cost_and_timing_are_counterfactual_not_measured"
        in limitations
    )
    assert (
        "sensor_emulation_has_not_been_validated_against_hardware"
        in limitations
    )
    evidence = report.data_provenance["evidence_contract"]
    sensor = evidence["sensor_input"]
    assert sensor["kind"] == "structure_suppressed_emulation"
    assert sensor["leak_off_calibration"]["sample_count"] == 3
    assert len(
        sensor["leak_off_calibration"][
            "resolved_calibration_sha256"
        ]
    ) == 64
    assert evidence["scanner"]["duration_basis"] == (
        "modeled_counterfactual_seconds"
    )
    assert evidence["dataset_audit"]["status"] == "hash_verified"
    assert report.data_provenance["claim_eligibility"] == (
        validation["claim_eligibility"]
    )
    assert report.evaluation_truth["evidence_grade"] == "silver_pseudo"
    assert report.evaluation_truth["available_to_policy"] is False


def test_headless_scan_mode_never_materializes_full_sensor_frames(
    tmp_path: Path,
) -> None:
    manifest_path, _payload = _v2_fixture(tmp_path)
    sequence = load_recorded_sequence(manifest_path)
    timeline = _AccessLoggingTimeline(sequence.timeline)

    run_experiment(
        ExperimentRequest(
            recorded_sequence=replace(sequence, timeline=timeline),
            scenario="methane",
            max_scans=1,
        )
    )

    assert timeline.sample_call_count > 0
    assert timeline.frame_calls == []


@pytest.mark.parametrize(
    (
        "sensor_kind",
        "truth_grade",
        "scanner_kind",
        "expected",
    ),
    [
        (
            "raw_recorded",
            "silver_pseudo",
            "counterfactual_virtual",
            "diagnostic_only_pseudo_truth",
        ),
        (
            "raw_recorded",
            "gold_human_reviewed",
            "counterfactual_virtual",
            "offline_counterfactual_scan_evidence",
        ),
        (
            "structure_suppressed_emulation",
            "gold_human_reviewed",
            "counterfactual_virtual",
            "offline_counterfactual_scan_evidence",
        ),
    ],
)
def test_claim_eligibility_is_derived_from_evidence_contract(
    tmp_path: Path,
    sensor_kind: str,
    truth_grade: str,
    scanner_kind: str,
    expected: str,
) -> None:
    _manifest_path, payload = _v2_fixture(
        tmp_path,
        sensor_kind=sensor_kind,
        truth_grade=truth_grade,
        scanner_kind=scanner_kind,
    )

    manifest = parse_recorded_sequence_manifest(payload)

    assert assess_recorded_claim(manifest).eligibility == expected


def test_v2_rejects_measured_hardware_without_measured_scan_adapter(
    tmp_path: Path,
) -> None:
    _manifest_path, payload = _v2_fixture(tmp_path)
    payload["scanner"]["evidence_kind"] = "measured_hardware"  # type: ignore[index]

    with pytest.raises(ValueError, match="reserved for a future"):
        parse_recorded_sequence_manifest(payload)


def test_unverified_dataset_rights_are_a_blocking_eligibility(
    tmp_path: Path,
) -> None:
    manifest_path, _payload = _v2_fixture(
        tmp_path,
        rights_blocked=True,
    )

    validation = validate_recorded_sequence(manifest_path)

    assert validation["claim_eligibility"] == (
        "blocked_unverified_data_rights"
    )
    assert "dataset_rights_are_not_verified" in (
        validation["claim_limitations"]
    )


def test_paired_suites_reject_capture_group_overlap(
    tmp_path: Path,
) -> None:
    development_manifest, development_payload = _v2_fixture(
        tmp_path / "development",
        sensor_kind="raw_recorded",
        truth_grade="gold_human_reviewed",
        capture_group_id="shared-capture-group",
    )
    heldout_manifest, heldout_payload = _v2_fixture(
        tmp_path / "heldout",
        sensor_kind="raw_recorded",
        truth_grade="gold_human_reviewed",
        split="held-out",
        capture_group_id="shared-capture-group",
        sensor_offset=3,
    )
    heldout_payload["sequence_id"] = "recorded-v2-heldout"
    heldout_payload["split_provenance"] = (
        "held out before policy execution"
    )
    heldout_manifest.write_text(
        json.dumps(heldout_payload),
        encoding="utf-8",
    )

    development_suite = (
        development_manifest.parent / "development-suite.json"
    )
    heldout_suite = heldout_manifest.parent / "heldout-suite.json"
    development_suite_payload = {
        "suite_schema_version": 1,
        "suite_id": "development-v2-fixture",
        "split": "development",
        "frozen_at_utc": "2026-07-26T00:00:00Z",
        "frozen_before_model_commit": "abcdef0",
        "freeze_provenance": "pytest development freeze",
        "sequences": [
            {
                "path": development_manifest.name,
                "manifest_sha256": canonical_json_sha256(
                    development_payload
                ),
            }
        ],
    }
    heldout_suite_payload = {
        "suite_schema_version": 1,
        "suite_id": "heldout-v2-fixture",
        "split": "held-out",
        "frozen_at_utc": "2026-07-26T00:00:00Z",
        "frozen_before_model_commit": "abcdef0",
        "freeze_provenance": "pytest held-out freeze",
        "sequences": [
            {
                "path": heldout_manifest.name,
                "manifest_sha256": canonical_json_sha256(
                    heldout_payload
                ),
            }
        ],
    }
    development_suite.write_text(
        json.dumps(development_suite_payload),
        encoding="utf-8",
    )
    heldout_suite.write_text(
        json.dumps(heldout_suite_payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="capture groups overlap"):
        validate_recorded_suite_pair(
            development_suite,
            heldout_suite,
        )

    heldout_payload["capture_group"]["capture_group_id"] = (  # type: ignore[index]
        "fixture-acquisition-002"
    )
    heldout_manifest.write_text(
        json.dumps(heldout_payload),
        encoding="utf-8",
    )
    heldout_suite_payload["sequences"][0]["manifest_sha256"] = (  # type: ignore[index]
        canonical_json_sha256(heldout_payload)
    )
    heldout_suite.write_text(
        json.dumps(heldout_suite_payload),
        encoding="utf-8",
    )

    validation = validate_recorded_suite_pair(
        development_suite,
        heldout_suite,
    )

    assert validation["capture_group_isolation_verified"] is True
    assert validation["source_asset_isolation_verified"] is True
    assert validation["development_capture_groups"] == [
        "shared-capture-group"
    ]
    assert validation["heldout_capture_groups"] == [
        "fixture-acquisition-002"
    ]


def test_paired_suites_reject_reused_source_media_with_renamed_group(
    tmp_path: Path,
) -> None:
    development_manifest, development_payload = _v2_fixture(
        tmp_path / "development",
        sensor_kind="raw_recorded",
        truth_grade="gold_human_reviewed",
        capture_group_id="development-group",
    )
    heldout_manifest, heldout_payload = _v2_fixture(
        tmp_path / "heldout",
        sensor_kind="raw_recorded",
        truth_grade="gold_human_reviewed",
        split="held-out",
        capture_group_id="heldout-group",
    )
    heldout_payload["sequence_id"] = "heldout-reused-bytes"
    heldout_manifest.write_text(
        json.dumps(heldout_payload),
        encoding="utf-8",
    )
    development_suite = (
        development_manifest.parent / "development-suite.json"
    )
    heldout_suite = heldout_manifest.parent / "heldout-suite.json"
    development_suite.write_text(
        json.dumps(
            {
                "suite_schema_version": 1,
                "suite_id": "development-reused-bytes",
                "split": "development",
                "frozen_at_utc": "2026-07-26T00:00:00Z",
                "frozen_before_model_commit": "abcdef0",
                "freeze_provenance": "pytest development freeze",
                "sequences": [
                    {
                        "path": development_manifest.name,
                        "manifest_sha256": canonical_json_sha256(
                            development_payload
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    heldout_suite.write_text(
        json.dumps(
            {
                "suite_schema_version": 1,
                "suite_id": "heldout-reused-bytes",
                "split": "held-out",
                "frozen_at_utc": "2026-07-26T00:00:00Z",
                "frozen_before_model_commit": "abcdef0",
                "freeze_provenance": "pytest held-out freeze",
                "sequences": [
                    {
                        "path": heldout_manifest.name,
                        "manifest_sha256": canonical_json_sha256(
                            heldout_payload
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source media assets overlap"):
        validate_recorded_suite_pair(
            development_suite,
            heldout_suite,
        )


def test_v2_rejects_self_declared_claim_and_temporal_smoothing(
    tmp_path: Path,
) -> None:
    _manifest_path, payload = _v2_fixture(tmp_path)
    claim_payload = copy.deepcopy(payload)
    claim_payload["claim_eligibility"] = "supported"
    with pytest.raises(ValueError, match="unknown keys"):
        parse_recorded_sequence_manifest(claim_payload)

    smoothing_payload = copy.deepcopy(payload)
    smoothing_payload["sensor_input"]["emulation_profile"][  # type: ignore[index]
        "smoothing_time_constant_seconds"
    ] = 0.25
    with pytest.raises(
        ValueError,
        match="smoothing_time_constant_seconds is not supported",
    ):
        parse_recorded_sequence_manifest(smoothing_payload)


def test_v2_requires_lossless_truth_and_identity_before_emulation(
    tmp_path: Path,
) -> None:
    _manifest_path, payload = _v2_fixture(tmp_path)
    lossy_payload = copy.deepcopy(payload)
    lossy_payload["evaluation_truth"]["storage_format"] = (  # type: ignore[index]
        "legacy_video"
    )
    with pytest.raises(ValueError, match="lossless_npz_uint8"):
        parse_recorded_sequence_manifest(lossy_payload)

    normalized_payload = copy.deepcopy(payload)
    normalized_payload["normalization"] = {
        "mode": "fixed_linear",
        "provenance": "invalid double-normalization fixture",
        "raw_min": 0.0,
        "raw_max": 255.0,
    }
    with pytest.raises(ValueError, match="uint8_identity"):
        parse_recorded_sequence_manifest(normalized_payload)


def test_v2_cross_checks_hashed_dataset_audit_media(
    tmp_path: Path,
) -> None:
    manifest_path, payload = _v2_fixture(tmp_path)
    audit_path = manifest_path.parent.parent / "assets" / "dataset-audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    media_hash = payload["media"]["sha256"]  # type: ignore[index]
    media_asset = next(
        item for item in audit["assets"]
        if item["sha256"] == media_hash
    )
    media_asset["sha256"] = "a" * 64
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    payload["dataset_lineage"]["audit_sha256"] = file_sha256(  # type: ignore[index]
        audit_path
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="matching media hash"):
        load_recorded_sequence(manifest_path)


def test_v2_requires_complete_audit_and_calibration_lineage(
    tmp_path: Path,
) -> None:
    partial_manifest, partial_payload = _v2_fixture(tmp_path / "partial")
    partial_audit_path = (
        partial_manifest.parent.parent / "assets" / "dataset-audit.json"
    )
    partial_audit = json.loads(
        partial_audit_path.read_text(encoding="utf-8")
    )
    partial_audit.pop("accession_manifest_sha256")
    partial_audit_path.write_text(
        json.dumps(partial_audit),
        encoding="utf-8",
    )
    partial_payload["dataset_lineage"]["audit_sha256"] = file_sha256(  # type: ignore[index]
        partial_audit_path
    )
    partial_manifest.write_text(
        json.dumps(partial_payload),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing keys"):
        load_recorded_sequence(partial_manifest)

    calibration_manifest, calibration_payload = _v2_fixture(
        tmp_path / "calibration"
    )
    calibration_audit_path = (
        calibration_manifest.parent.parent
        / "assets"
        / "dataset-audit.json"
    )
    calibration_audit = json.loads(
        calibration_audit_path.read_text(encoding="utf-8")
    )
    calibration_hash = calibration_payload["sensor_input"][  # type: ignore[index]
        "leak_off_calibration"
    ]["sha256"]
    calibration_audit["assets"] = [
        item for item in calibration_audit["assets"]
        if item["sha256"] != calibration_hash
    ]
    calibration_audit["asset_count"] = len(calibration_audit["assets"])
    calibration_audit_path.write_text(
        json.dumps(calibration_audit),
        encoding="utf-8",
    )
    calibration_payload["dataset_lineage"]["audit_sha256"] = file_sha256(  # type: ignore[index]
        calibration_audit_path
    )
    calibration_manifest.write_text(
        json.dumps(calibration_payload),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="leak-off calibration"):
        load_recorded_sequence(calibration_manifest)


def test_v2_verifies_truth_protocol_asset_hash(
    tmp_path: Path,
) -> None:
    manifest_path, _payload = _v2_fixture(
        tmp_path,
        truth_grade="gold_human_reviewed",
    )
    protocol_path = (
        manifest_path.parent.parent / "assets" / "truth-protocol.json"
    )
    protocol_path.write_text(
        '{"tampered": true}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_recorded_sequence(manifest_path)


def test_v2_hash_comparisons_are_case_normalized(
    tmp_path: Path,
) -> None:
    manifest_path, payload = _v2_fixture(
        tmp_path,
        sensor_kind="raw_recorded",
    )
    payload["media"]["sha256"] = payload["media"]["sha256"].upper()  # type: ignore[index,union-attr]
    payload["dataset_lineage"]["source_asset_sha256"] = (  # type: ignore[index,union-attr]
        payload["dataset_lineage"]["source_asset_sha256"].upper()  # type: ignore[index,union-attr]
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_recorded_sequence(manifest_path).manifest.sequence_id == (
        "recorded-v2-fixture"
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "dataset_audit_schema_version",
            3,
            "dataset_audit_schema_version 1 or 2",
        ),
        (
            "artifact_kind",
            "sequence_evidence",
            "artifact_kind",
        ),
        (
            "claim_eligibility",
            "supported",
            "not_sequence_evidence",
        ),
        (
            "rights_blocked",
            True,
            "rights status and rights_blocked disagree",
        ),
    ],
)
def test_v2_rejects_inconsistent_dataset_audit_contract(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    manifest_path, payload = _v2_fixture(tmp_path)
    audit_path = manifest_path.parent.parent / "assets" / "dataset-audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit[field] = value
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    payload["dataset_lineage"]["audit_sha256"] = file_sha256(  # type: ignore[index]
        audit_path
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_recorded_sequence(manifest_path)


def test_v2_verified_dataset_audit_requires_rights_uri(
    tmp_path: Path,
) -> None:
    manifest_path, payload = _v2_fixture(tmp_path)
    audit_path = manifest_path.parent.parent / "assets" / "dataset-audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["accession"]["rights_uri"] = None
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    payload["dataset_lineage"]["audit_sha256"] = file_sha256(  # type: ignore[index]
        audit_path
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="rights_uri"):
        load_recorded_sequence(manifest_path)


def test_v2_rejects_non_uint8_truth_npz(
    tmp_path: Path,
) -> None:
    manifest_path, payload = _v2_fixture(
        tmp_path,
        sensor_kind="raw_recorded",
    )
    truth_path = manifest_path.parent.parent / "assets" / "truth.npz"
    np.savez_compressed(
        truth_path,
        frames=np.zeros((4, 12, 16), dtype=np.float32),
    )
    payload["evaluation_truth"]["sha256"] = file_sha256(  # type: ignore[index]
        truth_path
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="uint8 dtype"):
        load_recorded_sequence(manifest_path)


def test_v3_truth_free_run_has_no_spatial_metrics_or_full_frame_access(
    tmp_path: Path,
) -> None:
    manifest_path, payload = _v2_fixture(
        tmp_path,
        sensor_kind="raw_recorded",
    )
    payload["manifest_schema_version"] = 3
    payload["evaluation_truth"] = {
        "evidence_grade": "none",
        "annotation_provenance": (
            "no spatial labels are available for this causal run"
        ),
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    parsed = parse_recorded_sequence_manifest(payload)
    assert json.loads(json.dumps(parsed.to_payload())) == payload
    sequence = load_recorded_sequence(manifest_path)
    assert sequence.truth_source is None
    validation = validate_recorded_sequence(manifest_path)
    assert validation["truth_basis"] is None
    assert validation["claim_eligibility"] == (
        "no_spatial_evaluation_truth"
    )

    timeline = _AccessLoggingTimeline(sequence.timeline)
    result = run_experiment(
        ExperimentRequest(
            recorded_sequence=replace(sequence, timeline=timeline),
            scenario="methane",
            max_scans=1,
        )
    )

    assert timeline.sample_call_count > 0
    assert timeline.frame_calls == []
    assert result.report.evaluation_truth["source"] == "none"
    assert result.report.evaluation_truth[
        "spatial_metrics_available"
    ] is False
    assert result.report.evaluation_truth["spatial_metric_samples"] == 0


def test_v2_cannot_self_remove_required_evaluation_truth(
    tmp_path: Path,
) -> None:
    _manifest_path, payload = _v2_fixture(
        tmp_path,
        sensor_kind="raw_recorded",
    )
    payload["evaluation_truth"] = {
        "evidence_grade": "none",
        "annotation_provenance": "not available",
    }

    with pytest.raises(ValueError, match="missing keys"):
        parse_recorded_sequence_manifest(payload)


def test_v3_truth_free_timeline_termination_is_observer_independent(
    tmp_path: Path,
) -> None:
    manifest_path, payload = _v2_fixture(
        tmp_path,
        sensor_kind="raw_recorded",
    )
    payload["manifest_schema_version"] = 3
    payload["evaluation_truth"] = {
        "evidence_grade": "none",
        "annotation_provenance": "no spatial truth",
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    sequence = load_recorded_sequence(manifest_path)
    request = ExperimentRequest(
        recorded_sequence=sequence,
        scenario="methane",
        max_scans=10,
    )

    headless = run_experiment(request).report
    observed = run_experiment(
        request,
        observer=_NoopObserver(),
    ).report

    assert observed.scans_completed == headless.scans_completed
    assert observed.termination_reason == headless.termination_reason
    assert observed.total_elapsed_seconds == headless.total_elapsed_seconds
    assert observed.scan_duration_seconds == headless.scan_duration_seconds


def test_v3_loads_frozen_calibration_without_redecoding_leak_off_media(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import plume_focus.recorded_sources as recorded_sources_module

    manifest_path, payload = _v2_fixture(tmp_path)
    legacy_sequence = load_recorded_sequence(manifest_path)
    assert isinstance(legacy_sequence.timeline, SensorEmulationTimeline)

    data_root = manifest_path.parent.parent
    calibration_path = data_root / "assets" / "fixed-calibration.plmcal"
    calibration_sha256 = write_fixed_calibration(
        calibration_path,
        legacy_sequence.timeline.calibration,
    )
    legacy_calibration = payload["sensor_input"][  # type: ignore[index]
        "leak_off_calibration"
    ]
    emulation_profile = payload["sensor_input"][  # type: ignore[index]
        "emulation_profile"
    ]
    template_payload = copy.deepcopy(payload)
    template_payload["manifest_schema_version"] = 3
    template_payload["sensor_input"] = {
        "kind": "raw_recorded",
        "provenance": "raw template before fixed preparation",
    }
    template_payload["evaluation_truth"] = {
        "evidence_grade": "none",
        "annotation_provenance": (
            "truth is attached by a separate evidence workflow"
        ),
    }
    producing_commit = "a" * 40
    preparation_spec_path = (
        data_root / "assets" / "preparation-spec.json"
    )
    preparation_spec_payload = {
        "preparation_spec_schema_version": 1,
        "preparation_id": "pytest-preparation-v1",
        "producing_commit": producing_commit,
        "dataset_audit": {
            "dataset_id": payload["dataset_lineage"]["dataset_id"],  # type: ignore[index]
            "accession_id": payload["dataset_lineage"]["accession_id"],  # type: ignore[index]
            "path": payload["dataset_lineage"]["audit_path"],  # type: ignore[index]
            "sha256": payload["dataset_lineage"]["audit_sha256"],  # type: ignore[index]
        },
        "capture_groups": [
            {
                "capture_group_id": "fixture-acquisition-001",
                "role": "development",
                "provenance": "whole acquisition is one split unit",
                "asset_sha256s": [payload["media"]["sha256"]],  # type: ignore[index]
            },
            {
                "capture_group_id": "fixture-calibration-001",
                "role": "development",
                "provenance": "separate development calibration capture",
                "asset_sha256s": [legacy_calibration["sha256"]],
            },
        ],
        "exclusions": [],
        "sequences": [
            {
                "output_sequence_id": payload["sequence_id"],
                "manifest_template": template_payload,
                "calibration_source": {
                    "path": legacy_calibration["path"],
                    "sha256": legacy_calibration["sha256"],
                    "provenance": (
                        "prepared from audited pytest leak-off media"
                    ),
                    "calibration_scope": legacy_calibration[
                        "calibration_scope"
                    ],
                    "capture_group_id": "fixture-calibration-001",
                },
            }
        ],
    }
    preparation_spec_path.write_text(
        json.dumps(preparation_spec_payload, sort_keys=True),
        encoding="utf-8",
    )
    preparation_profile_path = (
        data_root / "assets" / "preparation-profile.json"
    )
    preparation_profile_payload = {
        "preparation_profile_schema_version": 1,
        "profile_id": emulation_profile["profile_id"],
        "sensor_input_kind": "structure_suppressed_emulation",
        "sensor_input_provenance": (
            "pytest frozen calibration input"
        ),
        "emulation_profile": emulation_profile,
    }
    preparation_profile_path.write_text(
        json.dumps(preparation_profile_payload, sort_keys=True),
        encoding="utf-8",
    )
    calibration_record_path = (
        data_root / "assets" / "calibration-record.json"
    )
    calibration_shape = legacy_sequence.timeline.calibration.shape
    calibration_record_payload = {
        "calibration_record_schema_version": 1,
        "artifact_kind": "fixed_calibration_preparation",
        "claim_eligibility": "not_sequence_evidence",
        "preparation_id": "pytest-preparation-v1",
        "producing_commit": producing_commit,
        "fit_algorithm_id": (
            "per_pixel_median_normal_consistent_mad_v1"
        ),
        "source": {
            "path": legacy_calibration["path"],
            "sha256": legacy_calibration["sha256"],
            "provenance": (
                "prepared from audited pytest leak-off media"
            ),
            "calibration_scope": legacy_calibration[
                "calibration_scope"
            ],
            "capture_group_id": "fixture-calibration-001",
        },
        "calibration_artifact": {
            "path": "assets/fixed-calibration.plmcal",
            "sha256": calibration_sha256,
            "format": "plume_fixed_calibration_v1",
            "shape": list(calibration_shape),
            "sample_count": (
                legacy_sequence.timeline.calibration.sample_count
            ),
        },
        "emulation_profile_sha256": canonical_json_sha256(
            emulation_profile
        ),
        "preparation_spec": {
            "path": "assets/preparation-spec.json",
            "sha256": file_sha256(preparation_spec_path),
        },
        "preparation_profile": {
            "path": "assets/preparation-profile.json",
            "sha256": file_sha256(preparation_profile_path),
        },
    }
    calibration_record_path.write_text(
        json.dumps(calibration_record_payload, sort_keys=True),
        encoding="utf-8",
    )
    payload["manifest_schema_version"] = 3
    payload["sensor_input"] = {
        "kind": "structure_suppressed_emulation",
        "provenance": "pytest frozen calibration input",
        "emulation_profile": emulation_profile,
        "frozen_calibration": {
            "path": "assets/fixed-calibration.plmcal",
            "sha256": calibration_sha256,
            "source_path": legacy_calibration["path"],
            "source_sha256": legacy_calibration["sha256"],
            "source_provenance": (
                "prepared from audited pytest leak-off media"
            ),
            "calibration_scope": legacy_calibration[
                "calibration_scope"
            ],
            "producing_commit": "a" * 40,
            "preparation_spec_path": (
                "assets/preparation-spec.json"
            ),
            "preparation_spec_sha256": file_sha256(
                preparation_spec_path
            ),
            "preparation_profile_path": (
                "assets/preparation-profile.json"
            ),
            "preparation_profile_sha256": file_sha256(
                preparation_profile_path
            ),
            "calibration_record_path": (
                "assets/calibration-record.json"
            ),
            "calibration_record_sha256": file_sha256(
                calibration_record_path
            ),
        },
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    parsed = parse_recorded_sequence_manifest(payload)
    assert json.loads(json.dumps(parsed.to_payload())) == payload

    decoded_paths: list[Path] = []
    original_loader = recorded_sources_module.load_video_frames

    def tracking_loader(
        path: Path,
        *,
        max_frames: int | None,
    ) -> tuple[list[np.ndarray], float]:
        decoded_paths.append(Path(path).resolve())
        return original_loader(path, max_frames=max_frames)

    monkeypatch.setattr(
        recorded_sources_module,
        "load_video_frames",
        tracking_loader,
    )
    sequence = load_recorded_sequence(manifest_path)

    assert isinstance(sequence.timeline, SensorEmulationTimeline)
    assert (data_root / legacy_calibration["path"]).resolve() not in (
        decoded_paths
    )
    np.testing.assert_array_equal(
        sequence.timeline.frame_at_seconds(0.0),
        legacy_sequence.timeline.frame_at_seconds(0.0),
    )
    validation = validate_recorded_sequence(manifest_path)
    sensor_evidence = validation["evidence_contract"]["sensor_input"]
    assert sensor_evidence["frozen_calibration"]["resolution_mode"] == (
        "frozen_canonical_artifact_v1"
    )

    policy_variant = copy.deepcopy(payload)
    policy_variant["wind"] = {
        "mode": "constant",
        "units": "px_per_second",
        "provenance": "policy-only variant",
        "conversion_provenance": "pytest declared pixels",
        "dy_px_per_second": 0.0,
        "dx_px_per_second": 0.5,
    }
    policy_variant["source_assumption"]["source_point"] = [  # type: ignore[index]
        5.5,
        2.5,
    ]
    policy_variant["scanner"]["row_mass_fraction"] = 0.8  # type: ignore[index]
    manifest_path.write_text(
        json.dumps(policy_variant),
        encoding="utf-8",
    )
    load_recorded_sequence(manifest_path)

    acquisition_variant = copy.deepcopy(payload)
    acquisition_variant["media"]["capture_provenance"] = (  # type: ignore[index]
        "undeclared acquisition mutation"
    )
    manifest_path.write_text(
        json.dumps(acquisition_variant),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="acquisition and preprocessing contract",
    ):
        load_recorded_sequence(manifest_path)

    changed_profile = copy.deepcopy(preparation_profile_payload)
    changed_profile["emulation_profile"]["residual_mode"] = (  # type: ignore[index]
        "dark"
    )
    preparation_profile_path.write_text(
        json.dumps(changed_profile, sort_keys=True),
        encoding="utf-8",
    )
    rehashed_record = copy.deepcopy(calibration_record_payload)
    rehashed_record["emulation_profile_sha256"] = (  # type: ignore[index]
        canonical_json_sha256(changed_profile["emulation_profile"])
    )
    rehashed_record["preparation_profile"]["sha256"] = (  # type: ignore[index]
        file_sha256(preparation_profile_path)
    )
    calibration_record_path.write_text(
        json.dumps(rehashed_record, sort_keys=True),
        encoding="utf-8",
    )
    profile_variant = copy.deepcopy(payload)
    frozen_variant = profile_variant["sensor_input"][  # type: ignore[index]
        "frozen_calibration"
    ]
    frozen_variant["preparation_profile_sha256"] = file_sha256(
        preparation_profile_path
    )
    frozen_variant["calibration_record_sha256"] = file_sha256(
        calibration_record_path
    )
    manifest_path.write_text(
        json.dumps(profile_variant),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="manifest sensor input does not match",
    ):
        load_recorded_sequence(manifest_path)

    preparation_profile_path.write_text(
        json.dumps(preparation_profile_payload, sort_keys=True),
        encoding="utf-8",
    )
    changed_source_record = copy.deepcopy(calibration_record_payload)
    changed_source_record["source"]["capture_group_id"] = (  # type: ignore[index]
        "undeclared-calibration-group"
    )
    calibration_record_path.write_text(
        json.dumps(changed_source_record, sort_keys=True),
        encoding="utf-8",
    )
    source_variant = copy.deepcopy(payload)
    source_variant["sensor_input"]["frozen_calibration"][  # type: ignore[index]
        "calibration_record_sha256"
    ] = file_sha256(calibration_record_path)
    manifest_path.write_text(
        json.dumps(source_variant),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="record source does not match",
    ):
        load_recorded_sequence(manifest_path)

    calibration_record_path.write_text(
        json.dumps(calibration_record_payload, sort_keys=True),
        encoding="utf-8",
    )
    mismatched_record = copy.deepcopy(calibration_record_payload)
    mismatched_record["calibration_artifact"]["sha256"] = "0" * 64
    calibration_record_path.write_text(
        json.dumps(mismatched_record, sort_keys=True),
        encoding="utf-8",
    )
    payload["sensor_input"]["frozen_calibration"][  # type: ignore[index]
        "calibration_record_sha256"
    ] = file_sha256(calibration_record_path)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="artifact does not match its preparation record",
    ):
        load_recorded_sequence(manifest_path)

    calibration_record_path.write_text(
        json.dumps(calibration_record_payload, sort_keys=True),
        encoding="utf-8",
    )
    payload["sensor_input"]["frozen_calibration"][  # type: ignore[index]
        "calibration_record_sha256"
    ] = file_sha256(calibration_record_path)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    preparation_profile_path.write_text(
        '{"tampered": true}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_recorded_sequence(manifest_path)
