from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import plume_focus.recorded_dataset_cli as dataset_cli
from plume_focus.recorded_dataset import (
    DatasetAccession,
    audit_dataset_directory,
    decoded_frame_content_sha256,
    load_dataset_accession,
    parse_dataset_audit_report,
)
from plume_focus.recorded_dataset_cli import main as dataset_main
from plume_focus.video_sources import load_video_frames


def _write_video(
    path: Path,
    value: int,
    *,
    fps: float = 5.0,
    frame_count: int = 3,
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
        for _ in range(frame_count):
            frame = np.full((8, 12, 3), value, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def _accession(*, rights_status: str = "verified") -> DatasetAccession:
    return DatasetAccession(
        dataset_id="candidate-fixture",
        accession_id="candidate-fixture-20260726",
        source_uri="https://example.test/dataset",
        accessed_at_utc="2026-07-26T00:00:00Z",
        rights_status=rights_status,  # type: ignore[arg-type]
        rights_provenance="pytest declaration",
        rights_uri=(
            "https://example.test/terms"
            if rights_status == "verified"
            else None
        ),
    )


def _write_accession(
    path: Path,
    *,
    rights_status: str = "verified",
) -> None:
    path.write_text(
        json.dumps(
            {
                "dataset_accession_schema_version": 1,
                "dataset_id": "candidate-fixture",
                "accession_id": "candidate-fixture-20260726",
                "source_uri": "https://example.test/dataset",
                "accessed_at_utc": "2026-07-26T00:00:00Z",
                "rights_status": rights_status,
                "rights_provenance": "pytest declaration",
                "rights_uri": (
                    "https://example.test/terms"
                    if rights_status == "verified"
                    else None
                ),
            }
        ),
        encoding="utf-8",
    )


def test_dataset_audit_is_sorted_hashed_and_fully_decoded(
    tmp_path: Path,
) -> None:
    _write_video(tmp_path / "z.mp4", 30)
    _write_video(tmp_path / "nested" / "a.mp4", 80)
    (tmp_path / "ignored.txt").write_text("not media", encoding="utf-8")

    report = audit_dataset_directory(
        tmp_path,
        _accession(),
        accession_manifest_sha256="0" * 64,
    )
    payload = report.to_payload()

    assert report.valid is True
    assert report.rights_blocked is False
    assert [asset.relative_path for asset in report.assets] == [
        "nested/a.mp4",
        "z.mp4",
    ]
    assert all(asset.decoded_frame_count == 3 for asset in report.assets)
    assert all(asset.container_frame_count == 3 for asset in report.assets)
    assert all(asset.frame_shape == (8, 12) for asset in report.assets)
    assert all(len(asset.sha256) == 64 for asset in report.assets)
    audited_by_path = {
        asset.relative_path: asset
        for asset in report.assets
    }
    for relative_path in ("nested/a.mp4", "z.mp4"):
        frames, _fps = load_video_frames(tmp_path / relative_path)
        assert (
            audited_by_path[relative_path].decoded_content_sha256
            == decoded_frame_content_sha256(frames)
        )
    assert payload["claim_eligibility"] == "not_sequence_evidence"
    assert "dataset_root" not in payload
    reparsed = parse_dataset_audit_report(payload)
    assert reparsed.to_payload() == payload


def test_dataset_audit_requires_assets_and_blocks_unverified_rights(
    tmp_path: Path,
) -> None:
    report = audit_dataset_directory(
        tmp_path,
        _accession(rights_status="unverified"),
        accession_manifest_sha256="0" * 64,
    )

    assert report.valid is False
    assert report.rights_blocked is True
    assert report.assets == ()


def test_dataset_audit_rejects_video_too_short_for_runtime(
    tmp_path: Path,
) -> None:
    _write_video(tmp_path / "short.mp4", 20, frame_count=2)

    report = audit_dataset_directory(
        tmp_path,
        _accession(),
        accession_manifest_sha256="0" * 64,
    )

    assert report.valid is False
    assert report.assets[0].status == "insufficient_frames"
    assert "three frames" in (report.assets[0].issue or "")


def test_verified_accession_requires_rights_uri() -> None:
    with pytest.raises(ValueError, match="rights_uri"):
        DatasetAccession(
            dataset_id="fixture",
            accession_id="fixture-v1",
            source_uri="https://example.test/data",
            accessed_at_utc="2026-07-26T00:00:00Z",
            rights_status="verified",
            rights_provenance="pytest",
        )


def test_dataset_audit_requires_accession_manifest_hash(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="64 hexadecimal"):
        audit_dataset_directory(
            tmp_path,
            _accession(),
            accession_manifest_sha256="not-a-hash",
        )


def test_dataset_audit_parser_rejects_partial_artifact(
    tmp_path: Path,
) -> None:
    report = audit_dataset_directory(
        tmp_path,
        _accession(rights_status="unverified"),
        accession_manifest_sha256="0" * 64,
    )
    payload = report.to_payload()
    payload.pop("decode_scope")

    with pytest.raises(ValueError, match="missing keys"):
        parse_dataset_audit_report(payload)


def test_dataset_audit_v1_remains_loadable_without_decoder_build_hash(
    tmp_path: Path,
) -> None:
    _write_video(tmp_path / "clip.mp4", 20)
    payload = audit_dataset_directory(
        tmp_path,
        _accession(),
        accession_manifest_sha256="0" * 64,
    ).to_payload()
    payload["dataset_audit_schema_version"] = 1
    payload.pop("opencv_build_sha256")
    for asset in payload["assets"]:
        asset.pop("decoded_content_sha256")

    parsed = parse_dataset_audit_report(payload)

    assert parsed.dataset_audit_schema_version == 1
    assert parsed.opencv_build_sha256 is None
    assert parsed.to_payload() == payload


def test_dataset_audit_cli_writes_a_non_evidence_report(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "candidate"
    accession_path = tmp_path / "accession.json"
    report_path = tmp_path / "outputs" / "audit.json"
    _write_video(data_root / "clip.mp4", 42)
    _write_accession(accession_path, rights_status="unverified")

    exit_code = dataset_main(
        [
            "--root",
            str(data_root),
            "--accession-manifest",
            str(accession_path),
            "--report",
            str(report_path),
        ]
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["valid"] is True
    assert payload["rights_blocked"] is True
    assert payload["claim_eligibility"] == "not_sequence_evidence"
    assert len(payload["accession_manifest_sha256"]) == 64


def test_accession_manifest_is_strict_and_versioned(
    tmp_path: Path,
) -> None:
    accession_path = tmp_path / "accession.json"
    _write_accession(accession_path)
    payload = json.loads(accession_path.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    accession_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown keys"):
        load_dataset_accession(accession_path)


def test_dataset_audit_rejects_symlinked_video_assets(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.mp4"
    data_root = tmp_path / "candidate"
    data_root.mkdir()
    _write_video(outside, 10)
    linked = data_root / "linked.mp4"
    try:
        linked.symlink_to(outside)
    except OSError:
        pytest.skip("video symlinks are unavailable on this platform")

    with pytest.raises(ValueError, match="must not be symlinks"):
        audit_dataset_directory(
            data_root,
            _accession(),
            accession_manifest_sha256="0" * 64,
        )


def test_recorded_data_cli_dispatches_prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_prepare(**kwargs: object) -> SimpleNamespace:
        observed.update(kwargs)
        return SimpleNamespace(
            sequences=("one", "two"),
            preparation_id="fixture-preparation",
        )

    monkeypatch.setattr(
        dataset_cli,
        "prepare_recorded_dataset",
        fake_prepare,
    )

    exit_code = dataset_main(
        [
            "prepare",
            "--root",
            str(tmp_path),
            "--spec",
            str(tmp_path / "spec.json"),
            "--profile",
            str(tmp_path / "profile.json"),
            "--output-manifests",
            str(tmp_path / "manifests"),
            "--report",
            str(tmp_path / "preparation-report.json"),
        ]
    )

    assert exit_code == 0
    assert observed["data_root"] == tmp_path
    assert observed["output_manifests_dir"] == tmp_path / "manifests"


def test_recorded_data_cli_dispatches_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, object] = {}

    def fake_freeze(**kwargs: object) -> SimpleNamespace:
        observed.update(kwargs)
        return SimpleNamespace(
            sequence_count=3,
            suite_manifest_path=tmp_path / "suite.json",
            freeze_record_path=tmp_path / "suite.freeze.json",
        )

    monkeypatch.setattr(
        dataset_cli,
        "freeze_recorded_suite",
        fake_freeze,
    )

    exit_code = dataset_main(
        [
            "freeze",
            "--root",
            str(tmp_path),
            "--preparation-report",
            str(tmp_path / "preparation-report.json"),
            "--preparation-report-sha256",
            "a" * 64,
            "--suite-id",
            "heldout-v1",
            "--split",
            "held-out",
            "--frozen-at-utc",
            "2026-07-26T12:00:00Z",
            "--pre-change-model-commit",
            "b" * 40,
            "--freeze-provenance",
            "pytest explicit freeze",
            "--output",
            str(tmp_path / "suite.json"),
            "--freeze-record",
            str(tmp_path / "suite.freeze.json"),
        ]
    )

    assert exit_code == 0
    assert observed["split"] == "held-out"
    assert observed["frozen_before_model_commit"] == "b" * 40
    output = capsys.readouterr().out
    assert "Freeze record claim eligibility: not_sequence_evidence" in output
    assert "Suite publication does not establish a performance claim" in output
