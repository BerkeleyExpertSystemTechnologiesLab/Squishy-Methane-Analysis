from __future__ import annotations

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from plume_focus.recorded_artifacts import (
    canonical_json_bytes,
    write_json_artifact,
)
from plume_focus.recorded_cli import main as recorded_main
from plume_focus.recorded_calibration import write_fixed_calibration
from plume_focus.recorded_dataset import (
    opencv_decoder_environment_sha256,
)
from plume_focus.recorded_freeze import (
    freeze_recorded_suite,
    load_preparation_report,
    validate_recorded_freeze_record,
)
from plume_focus.recorded_manifest_io import (
    canonical_json_sha256,
    file_sha256,
    load_recorded_suite_manifest,
)
from plume_focus.sensor_emulation import FixedLeakOffCalibration


_PRODUCING_COMMIT = "a" * 40
_MODEL_COMMIT = "b" * 40
_FREEZE_TIME = "2026-07-26T12:34:56Z"


def _manifest_payload(
    sequence_id: str,
    *,
    media_sha256: str,
    capture_group_id: str,
    audit_sha256: str,
    split: str = "development",
    schema_version: int = 3,
) -> dict[str, object]:
    return {
        "manifest_schema_version": schema_version,
        "sequence_id": sequence_id,
        "split": split,
        "split_provenance": "fixed before planner execution",
        "scenario": "methane",
        "dataset_lineage": {
            "dataset_id": "freeze-fixture",
            "accession_id": "freeze-fixture-20260726",
            "audit_path": "metadata/dataset-audit.json",
            "audit_sha256": audit_sha256,
            "source_asset_sha256": media_sha256,
        },
        "capture_group": {
            "capture_group_id": capture_group_id,
            "grouping_provenance": "whole acquisition fixture",
        },
        "media": {
            "path": f"assets/{sequence_id}.mp4",
            "sha256": media_sha256,
            "capture_provenance": "pytest external-video fixture",
        },
        "sensor_input": {
            "kind": "raw_recorded",
            "provenance": "unaltered recorded intensity fixture",
        },
        "frame_clock": {
            "mode": "constant_fps",
            "fps": 5.0,
            "provenance": "declared fixture clock",
        },
        "normalization": {
            "mode": "uint8_identity",
            "provenance": "decoded uint8 values",
        },
        "calibration": {
            "sensor_id": "fixture-ogi",
            "signal_units": "compressed_8bit_ogi_display_intensity",
            "provenance": "non-radiometric fixture",
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
            "provenance": "counterfactual fixture scanner",
            "evidence_kind": "counterfactual_virtual",
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
            "evidence_grade": "none",
            "annotation_provenance": (
                "truth intentionally unavailable during preparation"
            ),
        },
        "model_profile_id": "uncalibrated_video_v1",
    }


def _write_freeze_fixture(
    tmp_path: Path,
    *,
    schema_version: int = 3,
) -> dict[str, object]:
    root = tmp_path / "recorded"
    asset_directory = root / "assets"
    manifest_directory = root / "manifests"
    metadata_directory = root / "metadata"
    asset_directory.mkdir(parents=True)
    manifest_directory.mkdir(parents=True)
    metadata_directory.mkdir()

    for sequence_id in ("sequence-b", "sequence-a"):
        (asset_directory / f"{sequence_id}.mp4").write_bytes(
            f"immutable fixture media: {sequence_id}\n".encode("utf-8")
        )
    sequence_details = tuple(
        (
            sequence_id,
            file_sha256(asset_directory / f"{sequence_id}.mp4"),
            capture_group_id,
        )
        for sequence_id, capture_group_id in (
            ("sequence-b", "capture-b"),
            ("sequence-a", "capture-a"),
        )
    )
    audit_payload = {
        "dataset_audit_schema_version": 2,
        "artifact_kind": "dataset_accession_audit",
        "claim_eligibility": "not_sequence_evidence",
        "valid": True,
        "rights_blocked": False,
        "asset_count": len(sequence_details),
        "decode_scope": "full_sequential_decode",
        "opencv_version": cv2.__version__,
        "opencv_build_sha256": opencv_decoder_environment_sha256(),
        "accession_manifest_sha256": "0" * 64,
        "accession": {
            "dataset_id": "freeze-fixture",
            "accession_id": "freeze-fixture-20260726",
            "source_uri": "https://example.test/freeze-fixture",
            "accessed_at_utc": "2026-07-26T00:00:00Z",
            "rights_status": "verified",
            "rights_provenance": "pytest fixture rights declaration",
            "rights_uri": "https://example.test/fixture-terms",
        },
        "assets": [
            {
                "relative_path": f"{sequence_id}.mp4",
                "sha256": media_sha256,
                "byte_count": (
                    asset_directory / f"{sequence_id}.mp4"
                ).stat().st_size,
                "status": "valid",
                "decoded_frame_count": 3,
                "container_frame_count": 3,
                "frame_shape": [12, 16],
                "nominal_fps": 5.0,
                "nominal_duration_seconds": 0.6,
                "container_fourcc": "TEST",
                "issue": None,
                "decoded_content_sha256": media_sha256,
            }
            for sequence_id, media_sha256, _capture_group_id
            in sequence_details
        ],
    }
    audit_path = metadata_directory / "dataset-audit.json"
    write_json_artifact(audit_path, audit_payload)
    audit_sha256 = file_sha256(audit_path)
    manifest_payloads: dict[str, dict[str, object]] = {}
    manifest_paths: dict[str, Path] = {}
    for sequence_id, media_sha256, capture_group_id in sequence_details:
        payload = _manifest_payload(
            sequence_id,
            media_sha256=media_sha256,
            capture_group_id=capture_group_id,
            audit_sha256=audit_sha256,
            schema_version=schema_version,
        )
        path = manifest_directory / f"{sequence_id}.json"
        write_json_artifact(path, payload)
        manifest_payloads[sequence_id] = payload
        manifest_paths[sequence_id] = path

    spec_payload: dict[str, object] = {
        "preparation_spec_schema_version": 1,
        "preparation_id": "freeze-fixture-v1",
        "producing_commit": _PRODUCING_COMMIT,
        "dataset_audit": {
            "dataset_id": "freeze-fixture",
            "accession_id": "freeze-fixture-20260726",
            "path": "metadata/dataset-audit.json",
            "sha256": audit_sha256,
        },
        "capture_groups": [
            {
                "capture_group_id": capture_group_id,
                "role": "development",
                "provenance": "whole acquisition fixture",
                "asset_sha256s": [media_sha256],
            }
            for _sequence_id, media_sha256, capture_group_id
            in sequence_details
        ],
        "exclusions": [],
        "sequences": [
            {
                "output_sequence_id": sequence_id,
                "manifest_template": manifest_payloads[sequence_id],
                "calibration_source": None,
            }
            for sequence_id, _media_sha256, _capture_group_id
            in sequence_details
        ],
    }
    spec_path = metadata_directory / "preparation-spec.json"
    write_json_artifact(spec_path, spec_payload)

    profile_payload: dict[str, object] = {
        "preparation_profile_schema_version": 1,
        "profile_id": "raw-recorded-fixture-v1",
        "sensor_input_kind": "raw_recorded",
        "sensor_input_provenance": (
            "unaltered recorded intensity fixture"
        ),
        "emulation_profile": None,
    }
    profile_path = metadata_directory / "preparation-profile.json"
    write_json_artifact(profile_path, profile_payload)

    report_payload: dict[str, object] = {
        "report_schema_version": 1,
        "artifact_kind": "recorded_preparation_report",
        "claim_eligibility": "not_sequence_evidence",
        "preparation_id": "freeze-fixture-v1",
        "preparation_spec": {
            "path": "metadata/preparation-spec.json",
            "sha256": file_sha256(spec_path),
        },
        "preparation_profile": {
            "path": "metadata/preparation-profile.json",
            "sha256": file_sha256(profile_path),
        },
        "producing_commit": _PRODUCING_COMMIT,
        "sequences": [
            {
                "sequence_id": sequence_id,
                "path": f"manifests/{sequence_id}.json",
                "manifest_sha256": canonical_json_sha256(
                    manifest_payloads[sequence_id]
                ),
            }
            for sequence_id, _media_sha256, _capture_group_id
            in sequence_details
        ],
    }
    report_path = metadata_directory / "preparation-report.json"
    write_json_artifact(report_path, report_payload)
    return {
        "root": root,
        "manifest_payloads": manifest_payloads,
        "manifest_paths": manifest_paths,
        "audit_path": audit_path,
        "spec_path": spec_path,
        "profile_path": profile_path,
        "report_path": report_path,
        "report_payload": report_payload,
    }


def _convert_fixture_to_emulated(
    fixture: dict[str, object],
) -> dict[str, object]:
    root = fixture["root"]
    audit_path = fixture["audit_path"]
    spec_path = fixture["spec_path"]
    profile_path = fixture["profile_path"]
    report_path = fixture["report_path"]
    leak_off_path = root / "assets" / "leak-off.mp4"
    leak_off_path.write_bytes(b"immutable leak-off fixture media\n")
    leak_off_sha256 = file_sha256(leak_off_path)
    audit_payload = json.loads(audit_path.read_bytes())
    audit_payload["assets"].append(
        {
            "relative_path": "leak-off.mp4",
            "sha256": leak_off_sha256,
            "byte_count": leak_off_path.stat().st_size,
            "status": "valid",
            "decoded_frame_count": 3,
            "container_frame_count": 3,
            "frame_shape": [12, 16],
            "nominal_fps": 5.0,
            "nominal_duration_seconds": 0.6,
            "container_fourcc": "TEST",
            "issue": None,
            "decoded_content_sha256": leak_off_sha256,
        }
    )
    audit_payload["asset_count"] = len(audit_payload["assets"])
    audit_path.write_bytes(canonical_json_bytes(audit_payload))
    audit_sha256 = file_sha256(audit_path)

    spec_payload = json.loads(spec_path.read_bytes())
    spec_payload["dataset_audit"]["sha256"] = audit_sha256
    calibration_source = {
        "path": "assets/leak-off.mp4",
        "sha256": leak_off_sha256,
        "provenance": "separate development leak-off fixture",
        "calibration_scope": "development_only",
        "capture_group_id": "capture-calibration",
    }
    spec_payload["capture_groups"].append(
        {
            "capture_group_id": "capture-calibration",
            "role": "development",
            "provenance": "separate calibration acquisition",
            "asset_sha256s": [leak_off_sha256],
        }
    )
    for item in spec_payload["sequences"]:
        item["calibration_source"] = calibration_source
        item["manifest_template"]["dataset_lineage"][
            "audit_sha256"
        ] = audit_sha256
    spec_path.write_bytes(canonical_json_bytes(spec_payload))

    emulation_profile = {
        "profile_id": "structure-suppressed-fixture-v1",
        "residual_mode": "two_sided_magnitude",
        "provenance": "pytest development-only settings",
        "noise_floor_raw_units": 1.0,
        "output_units_per_noise_sigma": 16.0,
        "smoothing_time_constant_seconds": None,
        "profile_schema_version": 1,
    }
    profile_payload = {
        "preparation_profile_schema_version": 1,
        "profile_id": "structure-suppressed-fixture-v1",
        "sensor_input_kind": "structure_suppressed_emulation",
        "sensor_input_provenance": "fixed pytest sensor emulation",
        "emulation_profile": emulation_profile,
    }
    profile_path.write_bytes(canonical_json_bytes(profile_payload))
    spec_sha256 = file_sha256(spec_path)
    profile_sha256 = file_sha256(profile_path)
    emulation_profile_sha256 = canonical_json_sha256(
        emulation_profile
    )
    artifact_path = "assets/prepared/calibration-shared.plmcal"
    artifact_sha256 = write_fixed_calibration(
        root / artifact_path,
        FixedLeakOffCalibration(
            background_raw_units=np.zeros((12, 16), dtype=np.float64),
            noise_sigma_raw_units=np.ones((12, 16), dtype=np.float64),
            sample_count=3,
        ),
    )

    record_paths: dict[str, Path] = {}
    manifest_payloads: dict[str, dict[str, object]] = {}
    for sequence in spec_payload["sequences"]:
        sequence_id = sequence["output_sequence_id"]
        record_payload = {
            "calibration_record_schema_version": 1,
            "artifact_kind": "fixed_calibration_preparation",
            "claim_eligibility": "not_sequence_evidence",
            "preparation_id": "freeze-fixture-v1",
            "producing_commit": _PRODUCING_COMMIT,
            "fit_algorithm_id": (
                "per_pixel_median_normal_consistent_mad_v1"
            ),
            "source": calibration_source,
            "calibration_artifact": {
                "path": artifact_path,
                "sha256": artifact_sha256,
                "format": "plume_fixed_calibration_v1",
                "shape": [12, 16],
                "sample_count": 3,
            },
            "emulation_profile_sha256": (
                emulation_profile_sha256
            ),
            "preparation_spec": {
                "path": "metadata/preparation-spec.json",
                "sha256": spec_sha256,
            },
            "preparation_profile": {
                "path": "metadata/preparation-profile.json",
                "sha256": profile_sha256,
            },
        }
        record_path = (
            root / "metadata" / f"{sequence_id}-calibration.json"
        )
        record_path.write_bytes(canonical_json_bytes(record_payload))
        record_paths[sequence_id] = record_path

        manifest_payload = copy.deepcopy(
            sequence["manifest_template"]
        )
        manifest_payload["sequence_id"] = sequence_id
        manifest_payload["sensor_input"] = {
            "kind": "structure_suppressed_emulation",
            "provenance": "fixed pytest sensor emulation",
            "emulation_profile": emulation_profile,
            "frozen_calibration": {
                "path": artifact_path,
                "sha256": artifact_sha256,
                "source_path": calibration_source["path"],
                "source_sha256": calibration_source["sha256"],
                "source_provenance": calibration_source[
                    "provenance"
                ],
                "calibration_scope": calibration_source[
                    "calibration_scope"
                ],
                "producing_commit": _PRODUCING_COMMIT,
                "preparation_spec_path": (
                    "metadata/preparation-spec.json"
                ),
                "preparation_spec_sha256": spec_sha256,
                "preparation_profile_path": (
                    "metadata/preparation-profile.json"
                ),
                "preparation_profile_sha256": profile_sha256,
                "calibration_record_path": (
                    f"metadata/{sequence_id}-calibration.json"
                ),
                "calibration_record_sha256": file_sha256(
                    record_path
                ),
            },
        }
        manifest_path = fixture["manifest_paths"][sequence_id]
        manifest_path.write_bytes(
            canonical_json_bytes(manifest_payload)
        )
        manifest_payloads[sequence_id] = manifest_payload

    report_payload = {
        "report_schema_version": 1,
        "artifact_kind": "recorded_preparation_report",
        "claim_eligibility": "not_sequence_evidence",
        "preparation_id": "freeze-fixture-v1",
        "preparation_spec": {
            "path": "metadata/preparation-spec.json",
            "sha256": spec_sha256,
        },
        "preparation_profile": {
            "path": "metadata/preparation-profile.json",
            "sha256": profile_sha256,
        },
        "producing_commit": _PRODUCING_COMMIT,
        "sequences": [
            {
                "sequence_id": sequence_id,
                "path": f"manifests/{sequence_id}.json",
                "manifest_sha256": canonical_json_sha256(
                    manifest_payloads[sequence_id]
                ),
            }
            for sequence_id in ("sequence-b", "sequence-a")
        ],
    }
    report_path.write_bytes(canonical_json_bytes(report_payload))
    fixture["manifest_payloads"] = manifest_payloads
    fixture["report_payload"] = report_payload
    fixture["record_paths"] = record_paths
    return fixture


def _freeze(
    fixture: dict[str, object],
    **overrides: object,
) -> object:
    arguments: dict[str, object] = {
        "data_root": fixture["root"],
        "preparation_report_path": fixture["report_path"],
        "preparation_report_sha256": file_sha256(
            fixture["report_path"]  # type: ignore[arg-type]
        ),
        "suite_id": "development-suite-v1",
        "split": "development",
        "frozen_at_utc": _FREEZE_TIME,
        "frozen_before_model_commit": _MODEL_COMMIT,
        "freeze_provenance": "explicit pytest preregistration",
        "suite_output_path": Path("manifests/development-suite.json"),
        "freeze_record_output_path": Path(
            "metadata/development-suite.freeze.json"
        ),
    }
    arguments.update(overrides)
    return freeze_recorded_suite(**arguments)  # type: ignore[arg-type]


def test_freeze_is_canonical_idempotent_and_persists_lineage(
    tmp_path: Path,
) -> None:
    fixture = _write_freeze_fixture(tmp_path)

    first = _freeze(fixture)
    suite_bytes = first.suite_manifest_path.read_bytes()
    record_bytes = first.freeze_record_path.read_bytes()
    second = _freeze(fixture)

    assert second == first
    assert first.sequence_count == 2
    assert first.suite_manifest_path.read_bytes() == suite_bytes
    assert first.freeze_record_path.read_bytes() == record_bytes

    suite_payload = json.loads(suite_bytes)
    assert suite_bytes == canonical_json_bytes(suite_payload)
    assert [
        item["path"] for item in suite_payload["sequences"]
    ] == ["sequence-a.json", "sequence-b.json"]
    loaded = load_recorded_suite_manifest(first.suite_manifest_path)
    assert loaded.frozen_at_utc == _FREEZE_TIME
    assert loaded.frozen_before_model_commit == _MODEL_COMMIT

    freeze_record = json.loads(record_bytes)
    assert freeze_record["claim_eligibility"] == (
        "not_sequence_evidence"
    )
    assert freeze_record["preparation_report"] == {
        "path": "metadata/preparation-report.json",
        "sha256": file_sha256(fixture["report_path"]),
    }
    assert freeze_record["suite_manifest"] == {
        "path": "manifests/development-suite.json",
        "sha256": file_sha256(first.suite_manifest_path),
        "canonical_json_sha256": canonical_json_sha256(
            suite_payload
        ),
    }
    assert first.suite_artifact_sha256 == file_sha256(
        first.suite_manifest_path
    )
    assert first.suite_manifest_sha256 == canonical_json_sha256(
        suite_payload
    )


def test_freeze_record_validator_checks_complete_lineage(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _write_freeze_fixture(tmp_path)
    frozen = _freeze(fixture)

    validation = validate_recorded_freeze_record(
        fixture["root"],  # type: ignore[arg-type]
        frozen.freeze_record_path,
    )
    assert validation["valid"] is True
    assert validation["preparation_lineage_verified"] is True
    assert validation["sequence_count"] == 2
    assert validation["claim_eligibility"] == "not_sequence_evidence"

    assert recorded_main(
        [
            "validate",
            "--freeze-record",
            str(frozen.freeze_record_path),
            "--root",
            str(fixture["root"]),
        ]
    ) == 0
    cli_payload = json.loads(capsys.readouterr().out)
    assert cli_payload["freeze_record_sha256"] == (
        validation["freeze_record_sha256"]
    )


def test_freeze_record_validator_rejects_noncanonical_or_changed_lineage(
    tmp_path: Path,
) -> None:
    canonical_fixture = _write_freeze_fixture(
        tmp_path / "noncanonical"
    )
    canonical_frozen = _freeze(canonical_fixture)
    record_payload = json.loads(
        canonical_frozen.freeze_record_path.read_bytes()
    )
    canonical_frozen.freeze_record_path.write_text(
        json.dumps(record_payload),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical JSON bytes"):
        validate_recorded_freeze_record(
            canonical_fixture["root"],  # type: ignore[arg-type]
            canonical_frozen.freeze_record_path,
        )

    lineage_fixture = _write_freeze_fixture(tmp_path / "lineage")
    lineage_frozen = _freeze(lineage_fixture)
    lineage_fixture["profile_path"].write_text(
        '{"tampered":true}',
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="preparation profile SHA-256 mismatch",
    ):
        validate_recorded_freeze_record(
            lineage_fixture["root"],  # type: ignore[arg-type]
            lineage_frozen.freeze_record_path,
        )


def test_freeze_record_validator_cross_checks_suite_metadata(
    tmp_path: Path,
) -> None:
    fixture = _write_freeze_fixture(tmp_path)
    frozen = _freeze(fixture)
    payload = json.loads(frozen.freeze_record_path.read_bytes())
    payload["suite_id"] = "different-suite"
    frozen.freeze_record_path.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(ValueError, match="suite metadata does not match"):
        validate_recorded_freeze_record(
            fixture["root"],  # type: ignore[arg-type]
            frozen.freeze_record_path,
        )


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("audit", "dataset audit SHA-256 mismatch"),
        ("media", "sequence media SHA-256 mismatch"),
        ("calibration", "fixed-calibration artifact SHA-256 mismatch"),
    ],
)
def test_freeze_record_validator_rechecks_source_and_derived_bytes(
    tmp_path: Path,
    target: str,
    message: str,
) -> None:
    fixture = _write_freeze_fixture(tmp_path)
    if target == "calibration":
        fixture = _convert_fixture_to_emulated(fixture)
    frozen = _freeze(fixture)

    if target == "audit":
        fixture["audit_path"].write_text(  # type: ignore[union-attr]
            '{"tampered":true}',
            encoding="utf-8",
        )
    elif target == "media":
        (
            fixture["root"] / "assets" / "sequence-a.mp4"  # type: ignore[operator]
        ).write_bytes(b"tampered media")
    else:
        (
            fixture["root"]  # type: ignore[operator]
            / "assets"
            / "prepared"
            / "calibration-shared.plmcal"
        ).write_bytes(b"tampered calibration")

    with pytest.raises(ValueError, match=message):
        validate_recorded_freeze_record(
            fixture["root"],  # type: ignore[arg-type]
            frozen.freeze_record_path,
        )


def test_freeze_record_validator_rejects_rewritten_suite_membership(
    tmp_path: Path,
) -> None:
    fixture = _write_freeze_fixture(tmp_path)
    frozen = _freeze(fixture)
    suite_payload = json.loads(frozen.suite_manifest_path.read_bytes())
    suite_payload["sequences"].reverse()
    frozen.suite_manifest_path.write_bytes(
        canonical_json_bytes(suite_payload)
    )
    record_payload = json.loads(frozen.freeze_record_path.read_bytes())
    record_payload["suite_manifest"]["sha256"] = file_sha256(
        frozen.suite_manifest_path
    )
    record_payload["suite_manifest"]["canonical_json_sha256"] = (
        canonical_json_sha256(suite_payload)
    )
    frozen.freeze_record_path.write_bytes(
        canonical_json_bytes(record_payload)
    )

    with pytest.raises(
        ValueError,
        match="references do not match the preparation report",
    ):
        validate_recorded_freeze_record(
            fixture["root"],  # type: ignore[arg-type]
            frozen.freeze_record_path,
        )


def test_freeze_record_validator_and_cli_reject_noncanonical_inputs(
    tmp_path: Path,
) -> None:
    fixture = _write_freeze_fixture(tmp_path)
    frozen = _freeze(fixture)
    payload = json.loads(frozen.freeze_record_path.read_bytes())
    payload["preparation_report"]["path"] = (
        "./metadata/preparation-report.json"
    )
    frozen.freeze_record_path.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(ValueError, match="canonical relative syntax"):
        validate_recorded_freeze_record(
            fixture["root"],  # type: ignore[arg-type]
            frozen.freeze_record_path,
        )
    with pytest.raises(SystemExit):
        recorded_main(
            [
                "validate",
                "--freeze-record",
                str(frozen.freeze_record_path),
            ]
        )
    with pytest.raises(SystemExit):
        recorded_main(
            [
                "validate",
                "--manifest",
                "sequence.json",
                "--root",
                str(fixture["root"]),
            ]
        )


def test_freeze_requires_explicit_utc_time_and_full_commit(
    tmp_path: Path,
) -> None:
    fixture = _write_freeze_fixture(tmp_path)

    with pytest.raises(ValueError, match="full 40-character commit"):
        _freeze(
            fixture,
            frozen_before_model_commit="b" * 12,
        )
    with pytest.raises(ValueError, match="UTC timezone"):
        _freeze(
            fixture,
            frozen_at_utc="2026-07-26T12:34:56+08:00",
        )

    assert not (
        fixture["root"] / "manifests/development-suite.json"
    ).exists()


def test_freeze_rejects_sequence_manifest_before_v3(
    tmp_path: Path,
) -> None:
    fixture = _write_freeze_fixture(tmp_path)
    manifest_path = fixture["manifest_paths"]["sequence-a"]
    payload = copy.deepcopy(
        fixture["manifest_payloads"]["sequence-a"]
    )
    payload["manifest_schema_version"] = 2
    payload["evaluation_truth"] = {
        "path": "assets/truth.npz",
        "sha256": "4" * 64,
        "truth_kind": "probability_weight",
        "storage_format": "lossless_npz_uint8",
        "evidence_grade": "silver_pseudo",
        "method_id": "fixture-method-v1",
        "protocol_path": "metadata/truth-protocol.json",
        "protocol_sha256": "5" * 64,
        "annotation_provenance": "evaluation-only fixture",
    }
    manifest_path.write_bytes(canonical_json_bytes(payload))

    report_payload = copy.deepcopy(fixture["report_payload"])
    for reference in report_payload["sequences"]:
        if reference["sequence_id"] == "sequence-a":
            reference["manifest_sha256"] = canonical_json_sha256(payload)
    fixture["report_path"].write_bytes(
        canonical_json_bytes(report_payload)
    )

    with pytest.raises(
        ValueError,
        match="schema-v3, schema-v4, or schema-v5 sequences only",
    ):
        _freeze(fixture)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "sequence_id",
            "sequence-b",
            "sequence_id values must be unique",
        ),
        (
            "manifest_sha256",
            None,
            "manifest hashes must be unique",
        ),
    ],
)
def test_preparation_report_rejects_duplicate_sequence_identity(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    fixture = _write_freeze_fixture(tmp_path)
    payload = copy.deepcopy(fixture["report_payload"])
    first, second = payload["sequences"]
    second[field] = (
        first["manifest_sha256"] if value is None else value
    )
    fixture["report_path"].write_bytes(canonical_json_bytes(payload))

    with pytest.raises(ValueError, match=message):
        load_preparation_report(fixture["report_path"])


def test_freeze_rejects_hash_and_lineage_mismatches(
    tmp_path: Path,
) -> None:
    fixture = _write_freeze_fixture(tmp_path)

    with pytest.raises(
        ValueError,
        match="preparation report SHA-256 mismatch",
    ):
        _freeze(
            fixture,
            preparation_report_sha256="f" * 64,
        )

    payload = copy.deepcopy(fixture["report_payload"])
    payload["preparation_id"] = "different-preparation"
    fixture["report_path"].write_bytes(canonical_json_bytes(payload))
    with pytest.raises(
        ValueError,
        match="preparation_id does not match",
    ):
        _freeze(fixture)


def test_freeze_verifies_profile_and_sequence_manifest_hashes(
    tmp_path: Path,
) -> None:
    profile_fixture = _write_freeze_fixture(
        tmp_path / "profile-case"
    )
    profile_fixture["profile_path"].write_text(
        '{"tampered":true}',
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="preparation profile SHA-256 mismatch",
    ):
        _freeze(profile_fixture)

    manifest_fixture = _write_freeze_fixture(
        tmp_path / "manifest-case"
    )
    manifest_fixture["manifest_paths"]["sequence-a"].write_text(
        '{"tampered":true}',
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="prepared sequence manifest hash mismatch",
    ):
        _freeze(manifest_fixture)


def test_emulated_freeze_preserves_truth_free_template(
    tmp_path: Path,
) -> None:
    fixture = _convert_fixture_to_emulated(
        _write_freeze_fixture(tmp_path)
    )
    frozen = _freeze(fixture)
    assert frozen.sequence_count == 2

    manifest_payload = copy.deepcopy(
        fixture["manifest_payloads"]["sequence-a"]
    )
    manifest_payload["evaluation_truth"] = {
        "path": "assets/pseudo-truth.npz",
        "sha256": "8" * 64,
        "truth_kind": "probability_weight",
        "storage_format": "lossless_npz_uint8",
        "evidence_grade": "silver_pseudo",
        "method_id": "unapproved-pseudo-method",
        "protocol_path": "metadata/pseudo-protocol.json",
        "protocol_sha256": "9" * 64,
        "annotation_provenance": "not part of preparation",
    }
    manifest_path = fixture["manifest_paths"]["sequence-a"]
    manifest_path.write_bytes(
        canonical_json_bytes(manifest_payload)
    )
    report_payload = copy.deepcopy(fixture["report_payload"])
    for item in report_payload["sequences"]:
        if item["sequence_id"] == "sequence-a":
            item["manifest_sha256"] = canonical_json_sha256(
                manifest_payload
            )
    fixture["report_path"].write_bytes(
        canonical_json_bytes(report_payload)
    )

    with pytest.raises(
        ValueError,
        match="differs from its immutable preparation template",
    ):
        _freeze(
            fixture,
            suite_output_path=Path("manifests/second-suite.json"),
            freeze_record_output_path=Path(
                "metadata/second-suite.freeze.json"
            ),
        )


def test_emulated_freeze_binds_report_spec_and_profile_exactly(
    tmp_path: Path,
) -> None:
    fixture = _convert_fixture_to_emulated(
        _write_freeze_fixture(tmp_path)
    )
    alternate_profile_path = (
        fixture["root"] / "metadata" / "alternate-profile.json"
    )
    alternate_profile_path.write_bytes(
        fixture["profile_path"].read_bytes()
    )

    sequence_id = "sequence-a"
    record_path = fixture["record_paths"][sequence_id]
    record_payload = json.loads(record_path.read_bytes())
    record_payload["preparation_profile"]["path"] = (
        "metadata/alternate-profile.json"
    )
    record_path.write_bytes(canonical_json_bytes(record_payload))

    manifest_payload = copy.deepcopy(
        fixture["manifest_payloads"][sequence_id]
    )
    frozen_calibration = manifest_payload["sensor_input"][
        "frozen_calibration"
    ]
    frozen_calibration["preparation_profile_path"] = (
        "metadata/alternate-profile.json"
    )
    frozen_calibration["calibration_record_sha256"] = file_sha256(
        record_path
    )
    manifest_path = fixture["manifest_paths"][sequence_id]
    manifest_path.write_bytes(
        canonical_json_bytes(manifest_payload)
    )

    report_payload = copy.deepcopy(fixture["report_payload"])
    for item in report_payload["sequences"]:
        if item["sequence_id"] == sequence_id:
            item["manifest_sha256"] = canonical_json_sha256(
                manifest_payload
            )
    fixture["report_path"].write_bytes(
        canonical_json_bytes(report_payload)
    )

    with pytest.raises(
        ValueError,
        match="exact spec and profile",
    ):
        _freeze(fixture)


def test_freeze_never_accepts_claim_elevation(
    tmp_path: Path,
) -> None:
    fixture = _write_freeze_fixture(tmp_path)
    payload = copy.deepcopy(fixture["report_payload"])
    payload["claim_eligibility"] = "supported"
    fixture["report_path"].write_bytes(canonical_json_bytes(payload))

    with pytest.raises(
        ValueError,
        match="must remain not_sequence_evidence",
    ):
        _freeze(fixture)


def test_freeze_refuses_to_overwrite_different_outputs(
    tmp_path: Path,
) -> None:
    fixture = _write_freeze_fixture(tmp_path)
    frozen = _freeze(fixture)
    suite_before = frozen.suite_manifest_path.read_bytes()
    record_before = frozen.freeze_record_path.read_bytes()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _freeze(
            fixture,
            frozen_at_utc="2026-07-26T12:35:00Z",
        )

    assert frozen.suite_manifest_path.read_bytes() == suite_before
    assert frozen.freeze_record_path.read_bytes() == record_before
