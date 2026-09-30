import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from plume_focus.experiment_runner import (
    ExperimentContext,
    ExperimentRequest,
    ExperimentStep,
    run_experiment,
)
from plume_focus.recorded_cli import main as recorded_main
from plume_focus.recorded_contracts import (
    MotionLookup,
    RecordedSuiteManifest,
    SequenceReference,
)
from plume_focus.recorded_dataset import (
    DatasetAccession,
    audit_dataset_directory,
    decoded_frame_content_sha256,
)
from plume_focus.recorded_sources import (
    DenseTruthTimeline,
    TimeSeriesMotionProvider,
    canonical_json_sha256,
    file_sha256,
    load_recorded_sequence,
    parse_recorded_sequence_manifest,
    validate_recorded_sequence,
    validate_recorded_suite,
)
from plume_focus.recorded_workflow import run_recorded_sequence
from plume_focus.experiment import TimestampedArrayTimeline
from plume_focus.video_sources import (
    load_video_frame_window,
    load_video_frames,
)


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


def _recorded_fixture(
    tmp_path: Path,
    *,
    split: str = "development",
    truth_start_col: int = 2,
) -> tuple[Path, dict[str, object]]:
    data_root = tmp_path / "recorded"
    asset_directory = data_root / "assets"
    manifest_directory = data_root / "manifests"
    asset_directory.mkdir(parents=True)
    manifest_directory.mkdir()

    sensor_path = asset_directory / "sensor.mp4"
    truth_path = asset_directory / "truth.mp4"
    sensor_frames = tuple(
        np.full((12, 16), 20 + index * 20, dtype=np.uint8)
        for index in range(4)
    )
    truth_frames = []
    for index in range(4):
        frame = np.zeros((12, 16), dtype=np.uint8)
        start_col = truth_start_col + index
        frame[4:8, start_col : start_col + 3] = 255
        truth_frames.append(frame)
    _write_video(sensor_path, sensor_frames)
    _write_video(truth_path, tuple(truth_frames))

    manifest: dict[str, object] = {
        "manifest_schema_version": 1,
        "sequence_id": "test-recording-001",
        "split": split,
        "split_provenance": "pytest assignment before policy execution",
        "scenario": "methane",
        "media": {
            "path": "assets/sensor.mp4",
            "sha256": file_sha256(sensor_path),
            "capture_provenance": "pytest engineering fixture",
        },
        "frame_clock": {
            "mode": "constant_fps",
            "fps": 5.0,
            "provenance": "pytest writer configuration",
        },
        "normalization": {
            "mode": "uint8_identity",
            "provenance": "pytest decoded uint8 fixture",
        },
        "calibration": {
            "sensor_id": "test-sensor",
            "signal_units": "uncalibrated_intensity",
            "provenance": "pytest engineering fixture",
        },
        "wind": {
            "mode": "constant",
            "units": "px_per_second",
            "provenance": "pytest engineering fixture",
            "conversion_provenance": "already in image-plane units",
            "dy_px_per_second": 0.0,
            "dx_px_per_second": 1.5,
        },
        "source_assumption": {
            "mode": "known_source",
            "provenance": "pytest engineering fixture",
            "source_point": [6.0, 2.0],
            "initial_roi": [2, 0, 10, 12],
        },
        "scanner": {
            "provenance": "pytest mechanical fixture",
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
            "path": "assets/truth.mp4",
            "sha256": file_sha256(truth_path),
            "truth_kind": "binary_support",
            "annotation_provenance": "pytest engineering fixture",
        },
        "model_profile_id": "uncalibrated_video_v1",
    }
    manifest_path = manifest_directory / "test-recording-001.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest_path, manifest


def _recorded_window_fixture(
    tmp_path: Path,
) -> tuple[Path, dict[str, object], tuple[np.ndarray, ...]]:
    manifest_path, payload = _recorded_fixture(tmp_path)
    data_root = manifest_path.parent.parent
    asset_directory = data_root / "assets"
    sensor_path = asset_directory / "sensor.mp4"
    selection_metadata_path = asset_directory / "selection-metadata.json"
    selection_metadata_path.write_text(
        json.dumps(
            {
                "selection": "pytest fixed half-open window",
                "requested_interval_seconds": [0.2, 0.8],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    accession_directory = data_root / "accessions"
    accession_directory.mkdir()
    accession_path = accession_directory / "fixture-accession.json"
    accession_path.write_text("pytest accession\n", encoding="utf-8")
    accession = DatasetAccession(
        dataset_id="window-fixture",
        accession_id="window-fixture-20260731",
        source_uri="https://example.test/window-fixture",
        accessed_at_utc="2026-07-31T00:00:00Z",
        rights_status="verified",
        rights_provenance="pytest rights declaration",
        rights_uri="https://example.test/terms",
    )
    audit_path = asset_directory / "dataset-audit.json"
    audit_payload = audit_dataset_directory(
        asset_directory,
        accession,
        accession_manifest_sha256=file_sha256(accession_path),
    ).to_payload()
    audit_path.write_text(
        json.dumps(audit_payload, sort_keys=True),
        encoding="utf-8",
    )

    window_frames, _ = load_video_frame_window(
        sensor_path,
        start_frame_index=1,
        end_frame_index_exclusive=4,
    )
    media_sha256 = file_sha256(sensor_path)
    payload.update(
        {
            "manifest_schema_version": 4,
            "dataset_lineage": {
                "dataset_id": "window-fixture",
                "accession_id": "window-fixture-20260731",
                "audit_path": "assets/dataset-audit.json",
                "audit_sha256": file_sha256(audit_path),
                "source_asset_sha256": media_sha256,
            },
            "capture_group": {
                "capture_group_id": "window-fixture-capture",
                "grouping_provenance": (
                    "whole pytest acquisition is one split unit"
                ),
            },
            "sensor_input": {
                "kind": "raw_recorded",
                "provenance": "unaltered decoded pytest intensities",
            },
            "wind": {
                "mode": "none",
                "units": "px_per_second",
                "provenance": "no synchronized wind in window fixture",
                "conversion_provenance": "not applicable",
            },
            "media_window": {
                "requested_start_time_seconds": 0.2,
                "requested_end_time_seconds": 0.8,
                "source_start_frame_index": 1,
                "source_end_frame_index_exclusive": 4,
                "first_included_timestamp_seconds": 0.2,
                "last_included_timestamp_seconds": 0.6,
                "parent_media_sha256": media_sha256,
                "decoded_window_sha256": (
                    decoded_frame_content_sha256(window_frames)
                ),
                "selection_metadata_path": (
                    "assets/selection-metadata.json"
                ),
                "selection_metadata_sha256": file_sha256(
                    selection_metadata_path
                ),
                "selection_provenance": (
                    "pytest interval fixed before runtime decoding"
                ),
            },
            "evaluation_truth": {
                "evidence_grade": "none",
                "annotation_provenance": (
                    "truth intentionally absent from runtime fixture"
                ),
            },
        }
    )
    scanner = payload["scanner"]
    assert isinstance(scanner, dict)
    scanner["evidence_kind"] = "counterfactual_virtual"
    manifest_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return manifest_path, payload, tuple(window_frames)


class _MaskObserver:
    def __init__(self) -> None:
        self.masks: list[np.ndarray] = []

    def start(self, _context: ExperimentContext) -> None:
        pass

    def observe(self, step: ExperimentStep) -> None:
        self.masks.append(step.next_scan_mask.copy())

    def close(self) -> None:
        pass


def test_time_series_wind_is_causal_and_reports_staleness() -> None:
    provider = TimeSeriesMotionProvider(
        timestamps_seconds=(1.0, 2.0),
        dy_px_per_second=(2.0, 4.0),
        dx_px_per_second=(3.0, 5.0),
        max_age_seconds=0.5,
    )

    assert provider.lookup(0.9).status == "missing"
    available = provider.lookup(1.4)
    assert available.status == "available"
    assert available.age_seconds == pytest.approx(0.4)
    assert available.motion is not None
    assert available.motion.dy_per_t == 2.0
    assert available.motion.dx_per_t == 3.0
    assert provider.lookup(1.6).status == "stale"
    assert provider.lookup(2.0).motion is not None
    assert provider.lookup(2.0).motion.dx_per_t == 5.0


def test_motion_lookup_rejects_unknown_runtime_status() -> None:
    with pytest.raises(ValueError, match="unknown motion lookup status"):
        MotionLookup("future", None, None)  # type: ignore[arg-type]


def test_recorded_manifest_loads_hashed_media_and_uniform_binary_truth(
    tmp_path: Path,
) -> None:
    manifest_path, payload = _recorded_fixture(tmp_path)

    sequence = load_recorded_sequence(manifest_path)
    truth = sequence.truth_source.truth_at_seconds(0.41)
    validation = validate_recorded_sequence(manifest_path)

    assert sequence.timeline.frame_at_seconds(0.39)[0, 0] < 60
    assert sequence.manifest.split == "development"
    assert truth.support.sum() > 0
    assert truth.distribution.sum() == pytest.approx(1.0)
    assert np.unique(truth.distribution[truth.support]).size == 1
    assert validation["valid"] is True
    assert validation["truth_basis"] == (
        "uniform_probability_over_annotated_binary_support"
    )
    assert validation["claim_eligibility"] == (
        "legacy_unclassified_evidence"
    )
    assert canonical_json_sha256(sequence.manifest.to_payload()) == (
        canonical_json_sha256(payload)
    )
    assert "dataset_lineage" not in sequence.manifest.to_payload()
    assert "media_window" not in validation


def test_recorded_media_window_decodes_exact_frames_and_rebases_clock(
    tmp_path: Path,
) -> None:
    manifest_path, _payload, expected_frames = _recorded_window_fixture(
        tmp_path
    )

    sequence = load_recorded_sequence(manifest_path)
    validation = validate_recorded_sequence(manifest_path)
    window_evidence = validation["media_window"]

    assert sequence.provenance.frame_count == 3
    assert sequence.timeline.end_time_seconds == pytest.approx(0.4)
    assert np.array_equal(
        sequence.timeline.frame_at_seconds(0.0),
        expected_frames[0],
    )
    assert np.array_equal(
        sequence.timeline.frame_at_seconds(0.4),
        expected_frames[-1],
    )
    assert isinstance(window_evidence, dict)
    assert window_evidence["resolved_parent_frame_count"] == 4
    assert window_evidence["resolved_window_frame_count"] == 3
    assert window_evidence["resolved_source_timestamp_origin_seconds"] == (
        pytest.approx(0.2)
    )
    assert window_evidence["resolved_source_last_timestamp_seconds"] == (
        pytest.approx(0.6)
    )
    assert window_evidence["timeline_clock_rebased_to_zero"] is True
    assert (
        sequence.provenance.evidence_contract["media_window"]
        == window_evidence
    )
    attribution = sequence.provenance.evidence_contract[
        "media_attribution"
    ]
    assert attribution["source_credit"] == "pytest engineering fixture"
    assert attribution["source_uri"] == (
        "https://example.test/window-fixture"
    )
    assert attribution["license_uri"] == "https://example.test/terms"
    assert attribution["rights_status"] == "verified"
    assert attribution["rendered_derivative"] is False
    assert attribution["analysis_changes"]


def test_v4_truth_free_source_free_window_runs_two_causal_scans(
    tmp_path: Path,
) -> None:
    manifest_path, payload, _expected_frames = _recorded_window_fixture(
        tmp_path
    )
    payload["source_assumption"] = {
        "mode": "source_free",
        "provenance": "no source geometry available to the policy",
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    sequence = load_recorded_sequence(manifest_path)
    report = run_experiment(
        ExperimentRequest(
            recorded_sequence=sequence,
            scenario="methane",
            max_scans=2,
        )
    ).report

    assert report.scans_completed == 2
    assert report.scan_start_mode == "recorded_source_free_full_scene"
    assert report.plume_mapping_mode == (
        "recorded_source_free_scan_evidence"
    )
    assert report.initial_roi_area_fraction == 1.0
    assert report.initial_mask_area_fraction == 1.0
    assert sequence.source_point is None
    assert sequence.truth_source is None
    assert sequence.manifest.wind.mode == "none"
    assert report.wind_evidence["lookup_counts"]["available"] == 0
    assert report.wind_evidence["lookup_counts"]["missing"] > 0
    assert report.evaluation_truth["source"] == "none"
    assert report.evaluation_truth["available_to_policy"] is False
    assert report.evaluation_truth["spatial_metrics_available"] is False
    assert report.evaluation_truth["spatial_metric_samples"] == 0
    assert report.actual_sweep_evaluated_predictions == 0
    assert report.oracle_inputs == ()

    provenance = report.data_provenance
    manifest_window = payload["media_window"]
    assert isinstance(manifest_window, dict)
    assert provenance["media_sha256"] == manifest_window[
        "parent_media_sha256"
    ]
    assert provenance["manifest"]["media_window"] == manifest_window
    resolved_window = provenance["evidence_contract"]["media_window"]
    assert resolved_window["declared"]["parent_media_sha256"] == (
        manifest_window["parent_media_sha256"]
    )
    assert resolved_window["declared"]["decoded_window_sha256"] == (
        manifest_window["decoded_window_sha256"]
    )


def test_recorded_media_window_slices_full_timestamp_clock_before_rebase(
    tmp_path: Path,
) -> None:
    manifest_path, payload, expected_frames = _recorded_window_fixture(
        tmp_path
    )
    timestamp_path = (
        manifest_path.parent.parent / "assets" / "timestamps.csv"
    )
    timestamp_path.write_text(
        "frame_index,timestamp_seconds\n"
        "0,0.0\n"
        "1,0.1\n"
        "2,0.35\n"
        "3,0.9\n",
        encoding="utf-8",
    )
    payload["frame_clock"] = {
        "mode": "timestamp_csv",
        "path": "assets/timestamps.csv",
        "sha256": file_sha256(timestamp_path),
        "provenance": "pytest full-parent timestamp sidecar",
    }
    window = payload["media_window"]
    assert isinstance(window, dict)
    window.update(
        {
            "requested_start_time_seconds": 0.1,
            "requested_end_time_seconds": 1.0,
            "first_included_timestamp_seconds": 0.1,
            "last_included_timestamp_seconds": 0.9,
        }
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    sequence = load_recorded_sequence(manifest_path)

    assert sequence.timeline.end_time_seconds == pytest.approx(0.8)
    assert np.array_equal(
        sequence.timeline.frame_at_seconds(0.24),
        expected_frames[0],
    )
    assert np.array_equal(
        sequence.timeline.frame_at_seconds(0.25),
        expected_frames[1],
    )


def test_recorded_media_window_rejects_content_and_metadata_hash_mismatch(
    tmp_path: Path,
) -> None:
    manifest_path, payload, _expected_frames = _recorded_window_fixture(
        tmp_path
    )
    window = payload["media_window"]
    assert isinstance(window, dict)
    window["decoded_window_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="decoded media window differs",
    ):
        load_recorded_sequence(manifest_path)

    window["decoded_window_sha256"] = decoded_frame_content_sha256(
        load_video_frame_window(
            manifest_path.parent.parent / "assets" / "sensor.mp4",
            start_frame_index=1,
            end_frame_index_exclusive=4,
        )[0]
    )
    window["selection_metadata_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_recorded_sequence(manifest_path)


def test_recorded_media_window_rejects_end_beyond_audited_parent(
    tmp_path: Path,
) -> None:
    manifest_path, payload, _expected_frames = _recorded_window_fixture(
        tmp_path
    )
    window = payload["media_window"]
    assert isinstance(window, dict)
    window.update(
        {
            "source_end_frame_index_exclusive": 5,
            "last_included_timestamp_seconds": 0.7,
        }
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="beyond the audited parent frame count",
    ):
        load_recorded_sequence(manifest_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "first_included_timestamp_seconds",
            0.21,
            "first included timestamp differs",
        ),
        (
            "last_included_timestamp_seconds",
            0.61,
            "last included timestamp differs",
        ),
    ],
)
def test_recorded_media_window_rejects_timestamp_declaration_mismatch(
    tmp_path: Path,
    field: str,
    value: float,
    message: str,
) -> None:
    manifest_path, payload, _expected_frames = _recorded_window_fixture(
        tmp_path
    )
    window = payload["media_window"]
    assert isinstance(window, dict)
    window[field] = value
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_recorded_sequence(manifest_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "requested_start_time_seconds",
            0.0,
            "do not resolve to its declared source frame indices",
        ),
        (
            "requested_end_time_seconds",
            10.0,
            "requested end lies beyond the parent capture extent",
        ),
    ],
)
def test_recorded_media_window_recomputes_declared_time_boundaries(
    tmp_path: Path,
    field: str,
    value: float,
    message: str,
) -> None:
    manifest_path, payload, _expected_frames = _recorded_window_fixture(
        tmp_path
    )
    window = payload["media_window"]
    assert isinstance(window, dict)
    window[field] = value
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_recorded_sequence(manifest_path)


def test_recorded_media_window_rejects_metadata_media_path_alias(
    tmp_path: Path,
) -> None:
    manifest_path, payload, _expected_frames = _recorded_window_fixture(
        tmp_path
    )
    window = payload["media_window"]
    assert isinstance(window, dict)
    media = payload["media"]
    assert isinstance(media, dict)
    window["selection_metadata_path"] = "assets/./sensor.mp4"
    window["selection_metadata_sha256"] = "e" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="resolve to a separate asset"):
        load_recorded_sequence(manifest_path, verify_hashes=False)


def test_recorded_media_window_rejects_symlinked_selection_metadata(
    tmp_path: Path,
) -> None:
    manifest_path, payload, _expected_frames = _recorded_window_fixture(
        tmp_path
    )
    data_root = manifest_path.parent.parent
    target = data_root / "assets" / "selection-metadata.json"
    link = data_root / "assets" / "selection-link.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("file symlinks are unavailable on this platform")
    window = payload["media_window"]
    assert isinstance(window, dict)
    window["selection_metadata_path"] = "assets/selection-link.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must not use symlink paths"):
        load_recorded_sequence(manifest_path)


def test_recorded_manifest_rejects_unknown_keys_and_asset_hash_mismatch(
    tmp_path: Path,
) -> None:
    manifest_path, payload = _recorded_fixture(tmp_path)
    payload["unexpected"] = "not allowed"
    with pytest.raises(ValueError, match="unknown keys"):
        parse_recorded_sequence_manifest(payload)

    clean_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    clean_payload["media"]["sha256"] = "0" * 64
    manifest_path.write_text(
        json.dumps(clean_payload),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_recorded_sequence(manifest_path)


def test_recorded_manifest_rejects_asset_path_traversal(
    tmp_path: Path,
) -> None:
    manifest_path, payload = _recorded_fixture(tmp_path)
    payload["media"]["path"] = "../outside.mp4"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="contain no '..'"):
        load_recorded_sequence(manifest_path)


def test_frozen_suite_checks_manifest_hash_and_split(tmp_path: Path) -> None:
    manifest_path, manifest_payload = _recorded_fixture(tmp_path)
    suite_path = manifest_path.parent / "development-suite.json"
    suite_payload = {
        "suite_schema_version": 1,
        "suite_id": "development-fixture-v1",
        "split": "development",
        "frozen_at_utc": "2026-07-26T00:00:00Z",
        "frozen_before_model_commit": "fbfbb78",
        "freeze_provenance": "pytest suite fixture",
        "sequences": [
            {
                "path": manifest_path.name,
                "manifest_sha256": canonical_json_sha256(manifest_payload),
            }
        ],
    }
    suite_path.write_text(json.dumps(suite_payload), encoding="utf-8")

    validation = validate_recorded_suite(suite_path)

    assert validation["valid"] is True
    assert validation["sequence_count"] == 1
    assert validation["frozen_before_model_commit"] == "fbfbb78"


def test_frozen_suite_requires_utc_time_and_git_commit() -> None:
    reference = SequenceReference("sequence.json", "0" * 64)
    with pytest.raises(ValueError, match="UTC timezone"):
        RecordedSuiteManifest(
            suite_id="invalid-time",
            split="held-out",
            frozen_at_utc="2026-07-26T08:00:00+08:00",
            frozen_before_model_commit="fbfbb78",
            freeze_provenance="pytest invalid fixture",
            sequences=(reference,),
        )
    with pytest.raises(ValueError, match="hexadecimal"):
        RecordedSuiteManifest(
            suite_id="invalid-commit",
            split="held-out",
            frozen_at_utc="2026-07-26T00:00:00Z",
            frozen_before_model_commit="not-a-commit",
            freeze_provenance="pytest invalid fixture",
            sequences=(reference,),
        )


def test_recorded_sequence_runs_with_isolated_truth_and_causal_wind(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _recorded_fixture(tmp_path)
    sequence = load_recorded_sequence(manifest_path)

    report = run_experiment(
        ExperimentRequest(
            recorded_sequence=sequence,
            scenario="methane",
            max_scans=2,
        )
    ).report

    assert report.scans_completed == 2
    assert report.data_provenance["sequence_id"] == "test-recording-001"
    assert report.data_provenance["evidence_grade"] == (
        "provenance_checked_recorded_input"
    )
    manifest_provenance = report.data_provenance["manifest"]
    assert manifest_provenance["media"]["capture_provenance"] == (
        "pytest engineering fixture"
    )
    assert manifest_provenance["frame_clock"]["provenance"] == (
        "pytest writer configuration"
    )
    assert manifest_provenance["normalization"]["provenance"] == (
        "pytest decoded uint8 fixture"
    )
    assert manifest_provenance["scanner"]["provenance"] == (
        "pytest mechanical fixture"
    )
    assert report.evaluation_truth == {
        "source": "hashed_dense_sidecar",
        "basis": "uniform_probability_over_annotated_binary_support",
        "available_to_policy": False,
    }
    assert report.wind_evidence["lookup_counts"] == {
        "available": 3,
        "missing": 0,
        "stale": 0,
    }
    assert report.oracle_inputs == ()


def test_recorded_sequence_rejects_hidden_runtime_overrides(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _recorded_fixture(tmp_path)
    sequence = load_recorded_sequence(manifest_path)

    with pytest.raises(ValueError, match="remove overrides: scan_rate_hz"):
        run_experiment(
            ExperimentRequest(
                recorded_sequence=sequence,
                scenario="methane",
                max_scans=1,
                scan_rate_hz=200.0,
            )
        )
    with pytest.raises(ValueError, match="explicit max_scans"):
        run_experiment(
            ExperimentRequest(
                recorded_sequence=sequence,
                scenario="methane",
            )
        )


def test_recorded_cli_validates_and_writes_headless_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path, _ = _recorded_fixture(tmp_path)
    report_path = tmp_path / "outputs" / "recorded-report.json"

    assert recorded_main(
        ["validate", "--manifest", str(manifest_path)]
    ) == 0
    assert '"valid": true' in capsys.readouterr().out

    assert recorded_main(
        [
            "run",
            "--manifest",
            str(manifest_path),
            "--max-scans",
            "2",
            "--report",
            str(report_path),
        ]
    ) == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["data_provenance"]["sequence_id"] == (
        "test-recording-001"
    )
    assert payload["evaluation_truth"]["available_to_policy"] is False
    assert payload["wind_evidence"]["lookup_counts"]["available"] == 3


def test_recorded_cli_validates_suite_pair(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = {
        "valid": True,
        "capture_group_isolation_verified": True,
    }
    monkeypatch.setattr(
        "plume_focus.recorded_cli.validate_recorded_suite_pair",
        lambda development, held_out: {
            **expected,
            "development_path": str(development),
            "held_out_path": str(held_out),
        },
    )

    assert recorded_main(
        [
            "validate-pair",
            "--development-suite",
            "development.json",
            "--held-out-suite",
            "held-out.json",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["capture_group_isolation_verified"] is True
    assert payload["development_path"].endswith("development.json")
    assert payload["held_out_path"].endswith("held-out.json")


def test_recorded_truth_sidecar_cannot_change_scan_planning(
    tmp_path: Path,
) -> None:
    left_manifest, _ = _recorded_fixture(
        tmp_path / "left",
        truth_start_col=1,
    )
    right_manifest, _ = _recorded_fixture(
        tmp_path / "right",
        truth_start_col=10,
    )
    left_observer = _MaskObserver()
    right_observer = _MaskObserver()

    for manifest, observer in (
        (left_manifest, left_observer),
        (right_manifest, right_observer),
    ):
        sequence = load_recorded_sequence(manifest)
        run_experiment(
            ExperimentRequest(
                recorded_sequence=sequence,
                scenario="methane",
                max_scans=2,
            ),
            observer=observer,
        )

    assert len(left_observer.masks) == len(right_observer.masks) == 2
    for left, right in zip(
        left_observer.masks,
        right_observer.masks,
        strict=True,
    ):
        assert np.array_equal(left, right)


def test_timestamp_clock_and_stale_time_series_wind_are_causal(
    tmp_path: Path,
) -> None:
    manifest_path, payload = _recorded_fixture(tmp_path)
    asset_directory = manifest_path.parent.parent / "assets"
    timestamp_path = asset_directory / "timestamps.csv"
    timestamp_path.write_text(
        "frame_index,timestamp_seconds\n"
        "0,0.0\n"
        "1,0.2\n"
        "2,0.4\n"
        "3,0.6\n",
        encoding="utf-8",
    )
    wind_path = asset_directory / "wind.csv"
    wind_path.write_text(
        "timestamp_seconds,dy_px_per_second,dx_px_per_second\n"
        "0.0,0.0,1.0\n"
        "0.4,0.0,9.0\n",
        encoding="utf-8",
    )
    payload["frame_clock"] = {
        "mode": "timestamp_csv",
        "path": "assets/timestamps.csv",
        "sha256": file_sha256(timestamp_path),
        "provenance": "pytest timestamp sidecar",
    }
    payload["wind"] = {
        "mode": "time_series_csv",
        "units": "px_per_second",
        "provenance": "pytest wind sidecar",
        "conversion_provenance": "already image-plane motion",
        "path": "assets/wind.csv",
        "sha256": file_sha256(wind_path),
        "max_age_seconds": 0.15,
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    sequence = load_recorded_sequence(manifest_path)
    assert sequence.timeline.frame_at_seconds(0.39)[0, 0] < (
        sequence.timeline.frame_at_seconds(0.4)[0, 0]
    )
    assert sequence.motion_provider.lookup(0.39).status == "stale"
    at_measurement = sequence.motion_provider.lookup(0.4)
    assert at_measurement.status == "available"
    assert at_measurement.motion is not None
    assert at_measurement.motion.dx_per_t == 9.0

    report = run_experiment(
        ExperimentRequest(
            recorded_sequence=sequence,
            scenario="methane",
            max_scans=2,
        )
    ).report
    assert report.wind_evidence["lookup_counts"] == {
        "available": 2,
        "missing": 0,
        "stale": 1,
    }
    assert report.wind_evidence["scan_derived_fallbacks"] == 1
    assert report.wind_evidence["zero_motion_fallbacks"] == 0


def test_missing_recorded_wind_uses_explicit_zero_then_scan_fallback(
    tmp_path: Path,
) -> None:
    manifest_path, payload = _recorded_fixture(tmp_path)
    payload["wind"] = {
        "mode": "none",
        "units": "px_per_second",
        "provenance": "no wind instrument available",
        "conversion_provenance": "not applicable",
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    sequence = load_recorded_sequence(manifest_path)

    report = run_experiment(
        ExperimentRequest(
            recorded_sequence=sequence,
            scenario="methane",
            max_scans=2,
        )
    ).report

    assert report.wind_evidence["lookup_counts"] == {
        "available": 0,
        "missing": 3,
        "stale": 0,
    }
    assert report.wind_evidence["zero_motion_fallbacks"] == 1
    assert report.wind_evidence["scan_derived_fallbacks"] == 1


def test_fixed_normalization_uses_declared_bounds(
    tmp_path: Path,
) -> None:
    manifest_path, payload = _recorded_fixture(tmp_path)
    payload["normalization"] = {
        "mode": "fixed_linear",
        "provenance": "pytest fixed sensor range",
        "raw_min": 10.0,
        "raw_max": 110.0,
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    raw_frames, _ = load_video_frames(
        manifest_path.parent.parent / "assets" / "sensor.mp4",
        max_frames=None,
    )

    sequence = load_recorded_sequence(manifest_path)
    expected = np.clip(
        (raw_frames[0].astype(float) - 10.0) / 100.0 * 255.0,
        0.0,
        255.0,
    ).astype(np.uint8)

    assert np.array_equal(
        sequence.timeline.frame_at_seconds(0.0),
        expected,
    )


def test_weighted_truth_preserves_relative_annotation_mass() -> None:
    frame = np.zeros((3, 4), dtype=np.uint8)
    frame[1, 1] = 64
    frame[1, 2] = 192
    source = DenseTruthTimeline(
        TimestampedArrayTimeline(
            frames=(frame,),
            timestamps_seconds=(0.0,),
            fps=1.0,
        ),
        truth_kind="probability_weight",
    )

    truth = source.truth_at_seconds(0.0)

    assert truth.support.sum() == 2
    assert truth.distribution[1, 2] == pytest.approx(
        3.0 * truth.distribution[1, 1]
    )
    assert truth.distribution.sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("source_assumption", "expected_source", "expected_full_scene"),
    [
        (
            {
                "mode": "known_corridor",
                "provenance": "pytest corridor",
                "initial_roi": [2, 0, 10, 12],
            },
            None,
            False,
        ),
        (
            {
                "mode": "source_free",
                "provenance": "pytest source-free input",
            },
            None,
            True,
        ),
    ],
)
def test_recorded_source_modes_are_manifest_authoritative(
    tmp_path: Path,
    source_assumption: dict[str, object],
    expected_source: None,
    expected_full_scene: bool,
) -> None:
    manifest_path, payload = _recorded_fixture(tmp_path)
    payload["source_assumption"] = source_assumption
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    sequence = load_recorded_sequence(manifest_path)

    assert sequence.source_point is expected_source
    assert sequence.source_free is expected_full_scene
    assert (
        sequence.initial_roi.area == sequence.timeline.shape[0]
        * sequence.timeline.shape[1]
    ) is expected_full_scene


def test_recorded_workflow_optionally_writes_video(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _recorded_fixture(tmp_path)
    report_path = tmp_path / "outputs" / "report.json"
    video_path = tmp_path / "outputs" / "proof.mp4"

    report = run_recorded_sequence(
        manifest_path,
        max_scans=2,
        report_output=report_path,
        output_video=video_path,
    )

    assert report.scans_completed == 2
    assert report_path.stat().st_size > 0
    assert video_path.stat().st_size > 0
    attribution = report.data_provenance["evidence_contract"][
        "media_attribution"
    ]
    assert attribution["rendered_derivative"] is True
    assert attribution["rendering_changes"] == [
        "created a time-compressed annotated three-panel proof video "
        "from causal experiment steps"
    ]
    assert "retain this report" in attribution[
        "distribution_condition"
    ]


def test_recorded_constant_clock_rejects_container_fps_mismatch(
    tmp_path: Path,
) -> None:
    manifest_path, payload = _recorded_fixture(tmp_path)
    payload["frame_clock"]["fps"] = 7.0
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="container FPS differs"):
        load_recorded_sequence(manifest_path)
