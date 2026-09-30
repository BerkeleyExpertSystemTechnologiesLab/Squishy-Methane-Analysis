from __future__ import annotations

import copy

import pytest

from plume_focus.recorded_contracts import (
    SEQUENCE_MANIFEST_SCHEMA_VERSION,
)
from plume_focus.recorded_manifest_io import (
    parse_recorded_sequence_manifest,
)


_MEDIA_SHA256 = "a" * 64


def _truth_free_raw_manifest(*, schema_version: int = 3) -> dict[str, object]:
    payload: dict[str, object] = {
        "manifest_schema_version": schema_version,
        "sequence_id": "recorded-window-fixture",
        "split": "development",
        "split_provenance": "assigned before policy execution",
        "scenario": "methane",
        "dataset_lineage": {
            "dataset_id": "recorded-window-dataset",
            "accession_id": "recorded-window-accession",
            "audit_path": "assets/dataset-audit.json",
            "audit_sha256": "d" * 64,
            "source_asset_sha256": _MEDIA_SHA256,
        },
        "capture_group": {
            "capture_group_id": "recorded-window-capture",
            "grouping_provenance": "whole acquisition is one split unit",
        },
        "media": {
            "path": "assets/parent.mp4",
            "sha256": _MEDIA_SHA256,
            "capture_provenance": "externally recorded parent video",
        },
        "sensor_input": {
            "kind": "raw_recorded",
            "provenance": "unaltered decoded grayscale intensity",
        },
        "frame_clock": {
            "mode": "constant_fps",
            "fps": 5.0,
            "provenance": "decoded parent-video clock",
        },
        "normalization": {
            "mode": "uint8_identity",
            "provenance": "decoded uint8 values",
        },
        "calibration": {
            "sensor_id": "unverified-ogi",
            "signal_units": "compressed_8bit_ogi_display_intensity",
            "provenance": "non-radiometric recorded input",
        },
        "wind": {
            "mode": "none",
            "units": "px_per_second",
            "provenance": "no synchronized wind",
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
            "annotation_provenance": "no spatial truth available",
        },
        "model_profile_id": "uncalibrated_video_v1",
    }
    if schema_version == 4:
        payload["media_window"] = {
            "requested_start_time_seconds": 224.0,
            "requested_end_time_seconds": 384.0,
            "source_start_frame_index": 1120,
            "source_end_frame_index_exclusive": 1920,
            "first_included_timestamp_seconds": 224.0,
            "last_included_timestamp_seconds": 383.8,
            "parent_media_sha256": _MEDIA_SHA256,
            "decoded_window_sha256": "b" * 64,
            "selection_metadata_path": "metadata/window-selection.csv",
            "selection_metadata_sha256": "c" * 64,
            "selection_provenance": "predeclared provider timing metadata",
        }
    return payload


def _window(payload: dict[str, object]) -> dict[str, object]:
    window = payload["media_window"]
    assert isinstance(window, dict)
    return window


def test_default_schema_remains_v3_and_v3_round_trip_is_unchanged() -> None:
    payload = _truth_free_raw_manifest()

    parsed = parse_recorded_sequence_manifest(payload)

    assert SEQUENCE_MANIFEST_SCHEMA_VERSION == 3
    assert parsed.manifest_schema_version == 3
    assert parsed.media_window is None
    assert parsed.to_payload() == payload


def test_v4_media_window_strictly_round_trips() -> None:
    payload = _truth_free_raw_manifest(schema_version=4)

    parsed = parse_recorded_sequence_manifest(payload)

    assert parsed.media_window is not None
    assert parsed.media_window.source_start_frame_index == 1120
    assert parsed.media_window.source_end_frame_index_exclusive == 1920
    assert parsed.to_payload() == payload


def test_v4_requires_media_window_and_v3_rejects_it() -> None:
    missing = _truth_free_raw_manifest(schema_version=4)
    missing.pop("media_window")
    with pytest.raises(ValueError, match="missing keys.*media_window"):
        parse_recorded_sequence_manifest(missing)

    v3 = _truth_free_raw_manifest()
    v3["media_window"] = _window(
        _truth_free_raw_manifest(schema_version=4)
    )
    with pytest.raises(ValueError, match="unknown keys.*media_window"):
        parse_recorded_sequence_manifest(v3)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "requested_start_time_seconds",
            -0.1,
            "requested_start_time_seconds must be at least 0.0",
        ),
        (
            "requested_end_time_seconds",
            224.0,
            "requested_end_time_seconds must be greater",
        ),
        (
            "source_start_frame_index",
            True,
            "source_start_frame_index must be an integer",
        ),
        (
            "source_end_frame_index_exclusive",
            1122,
            "at least three source frames",
        ),
        (
            "first_included_timestamp_seconds",
            223.8,
            "must not precede the requested start",
        ),
        (
            "last_included_timestamp_seconds",
            384.0,
            "must precede the exclusive requested end",
        ),
        (
            "decoded_window_sha256",
            "not-a-hash",
            "decoded window sha256 must contain 64 hexadecimal",
        ),
        (
            "selection_metadata_path",
            "  ",
            "selection metadata path must not be empty",
        ),
        (
            "selection_provenance",
            "",
            "selection provenance must not be empty",
        ),
    ),
)
def test_v4_rejects_invalid_media_window_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _truth_free_raw_manifest(schema_version=4)
    _window(payload)[field] = value

    with pytest.raises(ValueError, match=message):
        parse_recorded_sequence_manifest(payload)


def test_v4_rejects_unknown_media_window_fields() -> None:
    payload = _truth_free_raw_manifest(schema_version=4)
    _window(payload)["untracked_selection_note"] = "not allowed"

    with pytest.raises(ValueError, match="media window has unknown keys"):
        parse_recorded_sequence_manifest(payload)


def test_v4_binds_window_to_parent_media_and_separate_metadata() -> None:
    mismatched_parent = _truth_free_raw_manifest(schema_version=4)
    _window(mismatched_parent)["parent_media_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="must match media sha256"):
        parse_recorded_sequence_manifest(mismatched_parent)

    reused_path = _truth_free_raw_manifest(schema_version=4)
    media = reused_path["media"]
    assert isinstance(media, dict)
    _window(reused_path)["selection_metadata_path"] = media["path"]
    with pytest.raises(ValueError, match="must be a separate asset"):
        parse_recorded_sequence_manifest(reused_path)

    reused_bytes = _truth_free_raw_manifest(schema_version=4)
    _window(reused_bytes)["selection_metadata_sha256"] = _MEDIA_SHA256
    with pytest.raises(ValueError, match="must not reuse the parent media"):
        parse_recorded_sequence_manifest(reused_bytes)


def test_v4_is_truth_free_only() -> None:
    payload = _truth_free_raw_manifest(schema_version=4)
    payload["evaluation_truth"] = {
        "path": "assets/truth.npz",
        "sha256": "e" * 64,
        "truth_kind": "probability_weight",
        "storage_format": "lossless_npz_uint8",
        "evidence_grade": "silver_pseudo",
        "method_id": "pseudo-truth-v1",
        "protocol_path": "metadata/truth-protocol.json",
        "protocol_sha256": "f" * 64,
        "annotation_provenance": "diagnostic pseudo truth",
    }

    with pytest.raises(ValueError, match="require truth-free evaluation"):
        parse_recorded_sequence_manifest(payload)


def test_v4_is_raw_recorded_only() -> None:
    payload = _truth_free_raw_manifest(schema_version=4)
    payload["sensor_input"] = {
        "kind": "structure_suppressed_emulation",
        "provenance": "fixed leak-off emulation fixture",
        "emulation_profile": {
            "profile_id": "structure_suppressed_fixture_v1",
            "residual_mode": "bright",
            "provenance": "declared test profile",
            "noise_floor_raw_units": 1.0,
            "output_units_per_noise_sigma": 16.0,
            "smoothing_time_constant_seconds": None,
            "profile_schema_version": 1,
        },
        "frozen_calibration": {
            "path": "assets/calibration.npz",
            "sha256": "1" * 64,
            "source_path": "assets/leak-off.mp4",
            "source_sha256": "2" * 64,
            "source_provenance": "independent leak-off recording",
            "calibration_scope": "development_only",
            "producing_commit": "3" * 40,
            "preparation_spec_path": "metadata/preparation-spec.json",
            "preparation_spec_sha256": "4" * 64,
            "preparation_profile_path": "metadata/preparation-profile.json",
            "preparation_profile_sha256": "5" * 64,
            "calibration_record_path": "metadata/calibration-record.json",
            "calibration_record_sha256": "6" * 64,
        },
    }

    with pytest.raises(ValueError, match="require raw_recorded sensor input"):
        parse_recorded_sequence_manifest(payload)


def test_v4_rejects_fixed_linear_normalization() -> None:
    payload = _truth_free_raw_manifest(schema_version=4)
    payload["normalization"] = {
        "mode": "fixed_linear",
        "provenance": "invalid window-specific rescaling",
        "raw_min": 0.0,
        "raw_max": 255.0,
    }

    with pytest.raises(
        ValueError,
        match="require uint8_identity normalization",
    ):
        parse_recorded_sequence_manifest(payload)


def test_v4_rejects_unaligned_parent_time_wind() -> None:
    payload = _truth_free_raw_manifest(schema_version=4)
    payload["wind"] = {
        "mode": "constant",
        "units": "px_per_second",
        "provenance": "parent-timeline wind",
        "conversion_provenance": "already in image-plane units",
        "dy_px_per_second": 0.0,
        "dx_px_per_second": 1.5,
    }

    with pytest.raises(ValueError, match="require wind mode none"):
        parse_recorded_sequence_manifest(payload)


def test_manifest_parser_does_not_mutate_v4_payload() -> None:
    payload = _truth_free_raw_manifest(schema_version=4)
    original = copy.deepcopy(payload)

    parse_recorded_sequence_manifest(payload)

    assert payload == original
