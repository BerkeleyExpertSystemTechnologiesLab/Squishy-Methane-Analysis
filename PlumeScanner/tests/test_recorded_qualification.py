from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import plume_focus.recorded_dataset as recorded_dataset
from plume_focus.recorded_artifacts import (
    canonical_json_bytes,
    write_json_artifact,
)
from plume_focus.recorded_dataset import (
    DatasetAccession,
    DatasetAuditReport,
    VideoAssetAudit,
)
from plume_focus.recorded_dataset_cli import main as dataset_main
from plume_focus.recorded_manifest_io import file_sha256
from plume_focus.recorded_qualification import (
    DatasetQualificationV2,
    QualificationCheckReportV2,
    check_dataset_qualification,
    load_dataset_qualification,
    load_project_preregistration,
    parse_dataset_qualification,
    validate_requested_operations,
    write_dataset_qualification_template,
    write_project_preregistration,
)


def _valid_audit(
    root: Path,
    *,
    rights_status: str = "verified",
    asset_count: int = 2,
) -> Path:
    asset_root = root / "source-media"
    asset_root.mkdir(parents=True, exist_ok=True)
    for index in range(1, asset_count + 1):
        (asset_root / f"clip-{index}.mp4").write_bytes(
            f"qualification fixture clip {index}\n".encode("utf-8")
        )
    audit_path = root / "metadata" / "dataset-audit.json"
    accession = DatasetAccession(
        dataset_id="qualification-fixture",
        accession_id="qualification-fixture-20260731",
        source_uri="https://example.test/fixture",
        accessed_at_utc="2026-07-31T00:00:00Z",
        rights_status=rights_status,  # type: ignore[arg-type]
        rights_provenance="pytest rights declaration",
        rights_uri=(
            "https://example.test/fixture/terms"
            if rights_status == "verified"
            else None
        ),
    )
    assets = tuple(
        VideoAssetAudit(
            relative_path=f"clip-{index}.mp4",
            sha256=file_sha256(asset_root / f"clip-{index}.mp4"),
            byte_count=100 + index,
            status="valid",
            decoded_frame_count=10,
            container_frame_count=10,
            frame_shape=(8, 12),
            nominal_fps=5.0,
            nominal_duration_seconds=2.0,
            container_fourcc="mp4v",
            decoded_content_sha256=(str(index + 2) * 64),
        )
        for index in range(1, asset_count + 1)
    )
    report = DatasetAuditReport(
        accession=accession,
        accession_manifest_sha256="a" * 64,
        assets=assets,
        opencv_build_sha256="b" * 64,
    )
    write_json_artifact(audit_path, report.to_payload())
    return audit_path


def _pending_qualification(root: Path) -> tuple[Path, Path]:
    audit_path = _valid_audit(root)
    qualification_path = root / "metadata" / "qualification.json"
    write_dataset_qualification_template(
        data_root=root,
        dataset_audit_path=audit_path,
        qualification_id="qualification-fixture-v1",
        asset_root="source-media",
        output_path=qualification_path,
    )
    return audit_path, qualification_path


def _eligible_payload(qualification_path: Path) -> dict[str, object]:
    payload = json.loads(qualification_path.read_text(encoding="utf-8"))
    payload["evidence_records"] = [
        {
            "evidence_id": "owner-terms",
            "kind": "dataset_owner_correspondence",
            "reference": "private correspondence retained outside repository",
            "observed_at_utc": "2026-07-31T01:00:00Z",
            "content_sha256": "c" * 64,
            "summary": "Owner permits the three local processing scopes.",
        },
        {
            "evidence_id": "owner-metadata",
            "kind": "dataset_owner_metadata",
            "reference": "owner-supplied filename mapping",
            "observed_at_utc": "2026-07-31T01:01:00Z",
            "content_sha256": "d" * 64,
            "summary": "Owner confirms interval and group assignments.",
        },
    ]
    permissions = payload["permissions"]
    assert isinstance(permissions, dict)
    for scope in (
        "local_decode_analysis",
        "bounded_interval_extraction",
        "local_derivative_construction",
    ):
        permissions[scope] = {
            "decision": "allowed",
            "evidence_ids": ["owner-terms"],
        }
    # These scopes are deliberately independent of local-pilot eligibility.
    permissions["publish_aggregate_metrics"] = {
        "decision": "denied",
        "evidence_ids": ["owner-terms"],
    }
    permissions["publish_frames_or_clips"] = {
        "decision": "denied",
        "evidence_ids": ["owner-terms"],
    }
    permissions["redistribution"] = {
        "decision": "denied",
        "evidence_ids": ["owner-terms"],
    }
    payload["native_interval_semantics"] = {
        "status": "verified",
        "class_zero_meaning": "no controlled gas release",
        "nominal_interval_seconds": 180.0,
        "trim_start_seconds": 15.0,
        "trim_end_seconds": 5.0,
        "exceptions": [],
        "evidence_ids": ["owner-metadata"],
    }
    assets = payload["assets"]
    assert isinstance(assets, list)
    for index, asset in enumerate(assets):
        assert isinstance(asset, dict)
        asset["paper_group_id"] = f"pad-{index + 1}"
        asset["capture_group_id"] = f"capture-{index + 1}"
        asset["split"] = "development" if index == 0 else "held-out"
        asset["assignment_status"] = "verified"
        asset["assignment_evidence_ids"] = ["owner-metadata"]
    return payload


def _pending_v2_qualification(
    root: Path,
    *,
    operations: tuple[str, ...] = ("decode_analysis",),
) -> tuple[Path, Path, Path]:
    audit_path = _valid_audit(root)
    preregistration_path = root / "metadata" / "project-preregistration.json"
    write_project_preregistration(
        data_root=root,
        dataset_audit_path=audit_path,
        preregistration_id="qualification-fixture-diagnostic-v1",
        purpose="diagnostic_raw_pilot",
        created_at_utc="2026-07-31T03:00:00Z",
        operations=operations,  # type: ignore[arg-type]
        development_assets=("clip-1.mp4",),
        output_path=preregistration_path,
    )
    qualification_path = root / "metadata" / "qualification-v2.json"
    write_dataset_qualification_template(
        data_root=root,
        dataset_audit_path=audit_path,
        qualification_id="qualification-fixture-v2",
        asset_root="source-media",
        output_path=qualification_path,
        qualification_schema_version=2,
        purpose="diagnostic_raw_pilot",
        project_preregistration_path=preregistration_path,
    )
    return audit_path, preregistration_path, qualification_path


def _eligible_v2_payload(qualification_path: Path) -> dict[str, object]:
    root = qualification_path.parents[1]
    terms_snapshot = root / "metadata" / "fixture-terms.txt"
    provider_snapshot = root / "metadata" / "fixture-provider-metadata.txt"
    terms_snapshot.write_text(
        "Fixture terms authorize the declared local operations.\n",
        encoding="utf-8",
    )
    provider_snapshot.write_text(
        "clip-1.mp4 belongs to provider-capture-1.\n",
        encoding="utf-8",
    )
    payload = json.loads(qualification_path.read_text(encoding="utf-8"))
    payload["evidence_records"] = [
        {
            "evidence_id": "cc-terms-snapshot",
            "kind": "formal_terms",
            "reference": "metadata/fixture-terms.txt",
            "observed_at_utc": "2026-07-31T03:01:00Z",
            "content_sha256": file_sha256(terms_snapshot),
            "summary": "Content-hashed terms authorize local use.",
        },
        {
            "evidence_id": "provider-metadata-snapshot",
            "kind": "dataset_owner_metadata",
            "reference": "metadata/fixture-provider-metadata.txt",
            "observed_at_utc": "2026-07-31T03:02:00Z",
            "content_sha256": file_sha256(provider_snapshot),
            "summary": "Provider identifies the selected capture.",
        },
    ]
    permissions = payload["permissions"]
    assert isinstance(permissions, dict)
    preregistration_path = qualification_path.parent / "project-preregistration.json"
    preregistration = load_project_preregistration(preregistration_path)
    for scope in preregistration.required_permission_scopes:
        permissions[scope] = {
            "decision": "allowed",
            "evidence_ids": ["cc-terms-snapshot"],
        }
    provider_assets = payload["provider_assets"]
    assert isinstance(provider_assets, list)
    selected = next(
        asset
        for asset in provider_assets
        if asset["relative_path"] == "clip-1.mp4"
    )
    selected["capture_group_id"] = "provider-capture-1"
    selected["gas_identity"] = "methane"
    selected["fact_status"] = "verified"
    selected["fact_evidence_ids"] = ["provider-metadata-snapshot"]
    return payload


def test_pending_template_is_strict_roundtrippable_and_blocked(
    tmp_path: Path,
) -> None:
    _audit_path, qualification_path = _pending_qualification(tmp_path)

    qualification = load_dataset_qualification(qualification_path)
    payload = qualification.to_payload()
    reparsed = parse_dataset_qualification(payload)
    report = check_dataset_qualification(
        data_root=tmp_path,
        qualification_path=qualification_path,
    )

    assert reparsed.to_payload() == payload
    assert qualification.asset_root == "source-media"
    assert qualification.dataset_audit.path == "metadata/dataset-audit.json"
    assert len(qualification.assets) == 2
    assert all(asset.split is None for asset in qualification.assets)
    assert all(
        asset.assignment_status == "unresolved"
        for asset in qualification.assets
    )
    assert report.eligible is False
    assert "permission.local_decode_analysis.unknown" in report.blockers
    assert report.development_asset_count == 0


def test_template_uses_explicit_asset_root_and_rejects_escape(
    tmp_path: Path,
) -> None:
    audit_path = _valid_audit(tmp_path)
    for asset_root in (
        "metadata",
        "",
        "../outside",
        "source\\media",
        "source-media/../source-media",
        "./source-media",
    ):
        if asset_root == "metadata":
            # Existing, but deliberately not the declared source-media root;
            # the API accepts it because it cannot infer intent from location.
            qualification = write_dataset_qualification_template(
                data_root=tmp_path,
                dataset_audit_path=audit_path,
                qualification_id="explicit-non-inferred-root-v1",
                asset_root=asset_root,
                output_path=tmp_path / "explicit-root.json",
            )
            assert qualification.asset_root == "metadata"
            continue
        with pytest.raises(ValueError, match="asset_root"):
            write_dataset_qualification_template(
                data_root=tmp_path,
                dataset_audit_path=audit_path,
                qualification_id="escaping-root-v1",
                asset_root=asset_root,
                output_path=tmp_path / "escaping-root.json",
            )


def test_rights_blocked_audit_cannot_become_pilot_eligible(
    tmp_path: Path,
) -> None:
    audit_path = _valid_audit(tmp_path, rights_status="unverified")
    qualification_path = tmp_path / "metadata" / "pending-rights.json"
    write_dataset_qualification_template(
        data_root=tmp_path,
        dataset_audit_path=audit_path,
        qualification_id="qualification-pending-rights-v1",
        asset_root="source-media",
        output_path=qualification_path,
    )
    eligible_payload = _eligible_payload(qualification_path)
    eligible_path = tmp_path / "metadata" / "rights-blocked-eligible.json"
    write_json_artifact(eligible_path, eligible_payload)

    report = check_dataset_qualification(
        data_root=tmp_path,
        qualification_path=eligible_path,
    )

    assert report.eligible is False
    assert "dataset_audit.rights_blocked" in report.blockers


def test_eligible_local_pilot_ignores_publication_and_redistribution_denials(
    tmp_path: Path,
) -> None:
    _audit_path, qualification_path = _pending_qualification(tmp_path)
    write_json_artifact(
        tmp_path / "metadata" / "eligible.json",
        _eligible_payload(qualification_path),
    )

    report = check_dataset_qualification(
        data_root=tmp_path,
        qualification_path=tmp_path / "metadata" / "eligible.json",
        report_path=tmp_path / "outputs" / "qualification-check.json",
    )

    assert report.eligible is True
    assert report.blockers == ()
    check_payload = json.loads(
        (tmp_path / "outputs" / "qualification-check.json").read_text(
            encoding="utf-8"
        )
    )
    assert check_payload["claim_eligibility"] == "not_sequence_evidence"
    assert check_payload["qualification_id"] == "qualification-fixture-v1"
    assert check_payload["asset_root"] == "source-media"
    assert check_payload["eligible"] is True


def test_primary_publication_cannot_authorize_local_processing(
    tmp_path: Path,
) -> None:
    _audit_path, qualification_path = _pending_qualification(tmp_path)
    payload = _eligible_payload(qualification_path)
    records = payload["evidence_records"]
    assert isinstance(records, list)
    records[0]["kind"] = "primary_publication"
    eligible_path = tmp_path / "metadata" / "paper-only.json"
    write_json_artifact(eligible_path, payload)

    with pytest.raises(ValueError, match="requires formal terms"):
        check_dataset_qualification(
            data_root=tmp_path,
            qualification_path=eligible_path,
        )


def test_provider_metadata_alone_cannot_authorize_local_processing(
    tmp_path: Path,
) -> None:
    _audit_path, qualification_path = _pending_qualification(tmp_path)
    payload = _eligible_payload(qualification_path)
    records = payload["evidence_records"]
    assert isinstance(records, list)
    records[0]["kind"] = "dataset_owner_metadata"
    metadata_only = tmp_path / "metadata" / "metadata-only-rights.json"
    write_json_artifact(metadata_only, payload)

    with pytest.raises(ValueError, match="requires formal terms"):
        check_dataset_qualification(
            data_root=tmp_path,
            qualification_path=metadata_only,
        )


def test_nonlocal_permission_decisions_require_rights_evidence(
    tmp_path: Path,
) -> None:
    _audit_path, qualification_path = _pending_qualification(tmp_path)
    payload = _eligible_payload(qualification_path)
    permissions = payload["permissions"]
    assert isinstance(permissions, dict)
    permissions["publish_frames_or_clips"] = {
        "decision": "allowed",
        "evidence_ids": ["owner-metadata"],
    }

    with pytest.raises(ValueError, match="publish_frames_or_clips"):
        parse_dataset_qualification(payload)


def test_unknown_publication_and_redistribution_do_not_block_local_pilot(
    tmp_path: Path,
) -> None:
    _audit_path, qualification_path = _pending_qualification(tmp_path)
    payload = _eligible_payload(qualification_path)
    permissions = payload["permissions"]
    assert isinstance(permissions, dict)
    for scope in (
        "publish_aggregate_metrics",
        "publish_frames_or_clips",
        "redistribution",
    ):
        permissions[scope] = {"decision": "unknown", "evidence_ids": []}
    independent = tmp_path / "metadata" / "independent-scopes.json"
    write_json_artifact(independent, payload)

    report = check_dataset_qualification(
        data_root=tmp_path,
        qualification_path=independent,
    )

    assert report.eligible is True


def test_formal_terms_alone_cannot_establish_asset_metadata(
    tmp_path: Path,
) -> None:
    _audit_path, qualification_path = _pending_qualification(tmp_path)
    payload = _eligible_payload(qualification_path)
    records = payload["evidence_records"]
    assert isinstance(records, list)
    records[0]["kind"] = "formal_terms"
    assets = payload["assets"]
    assert isinstance(assets, list)
    assets[0]["assignment_evidence_ids"] = ["owner-terms"]
    terms_only = tmp_path / "metadata" / "terms-only-assignment.json"
    write_json_artifact(terms_only, payload)

    report = check_dataset_qualification(
        data_root=tmp_path,
        qualification_path=terms_only,
    )

    assert report.eligible is False
    assert (
        "asset.clip-1.mp4."
        "assignment_lacks_authoritative_metadata_evidence"
    ) in report.blockers


def test_primary_publication_alone_cannot_verify_exact_asset_assignments(
    tmp_path: Path,
) -> None:
    _audit_path, qualification_path = _pending_qualification(tmp_path)
    payload = _eligible_payload(qualification_path)
    records = payload["evidence_records"]
    assert isinstance(records, list)
    records[1]["kind"] = "primary_publication"
    paper_only = tmp_path / "metadata" / "paper-only-assignment.json"
    write_json_artifact(paper_only, payload)

    report = check_dataset_qualification(
        data_root=tmp_path,
        qualification_path=paper_only,
    )

    assert report.eligible is False
    assert (
        "asset.clip-1.mp4."
        "assignment_lacks_authoritative_metadata_evidence"
    ) in report.blockers


def test_inferred_asset_mappings_remain_blocked(tmp_path: Path) -> None:
    _audit_path, qualification_path = _pending_qualification(tmp_path)
    payload = _eligible_payload(qualification_path)
    assets = payload["assets"]
    assert isinstance(assets, list)
    assets[0]["assignment_status"] = "inferred"
    inferred = tmp_path / "metadata" / "inferred-assignment.json"
    write_json_artifact(inferred, payload)

    report = check_dataset_qualification(
        data_root=tmp_path,
        qualification_path=inferred,
    )

    assert report.eligible is False
    assert "asset.clip-1.mp4.assignment_inferred" in report.blockers


@pytest.mark.parametrize(
    "kind",
    [
        "formal_terms",
        "dataset_owner_correspondence",
        "dataset_owner_metadata",
        "primary_publication",
    ],
)
def test_all_declared_evidence_kinds_roundtrip(
    tmp_path: Path,
    kind: str,
) -> None:
    _audit_path, qualification_path = _pending_qualification(tmp_path)
    payload = _eligible_payload(qualification_path)
    records = payload["evidence_records"]
    assert isinstance(records, list)
    records.append(
        {
            "evidence_id": "kind-under-test",
            "kind": kind,
            "reference": "standalone evidence-kind fixture",
            "observed_at_utc": "2026-07-31T02:00:00Z",
            "content_sha256": None,
            "summary": "Exercises strict evidence-kind parsing.",
        }
    )

    parsed = parse_dataset_qualification(payload)

    assert parsed.evidence_records[-1].kind == kind


def test_parser_rejects_unknown_keys_duplicate_assets_and_evidence(
    tmp_path: Path,
) -> None:
    _audit_path, qualification_path = _pending_qualification(tmp_path)
    payload = json.loads(qualification_path.read_text(encoding="utf-8"))
    with_extra = copy.deepcopy(payload)
    with_extra["unexpected"] = True
    with pytest.raises(ValueError, match="unknown keys"):
        parse_dataset_qualification(with_extra)

    duplicate_asset = copy.deepcopy(payload)
    duplicate_asset["assets"].append(copy.deepcopy(duplicate_asset["assets"][0]))
    with pytest.raises(ValueError, match="paths must be unique"):
        parse_dataset_qualification(duplicate_asset)

    unknown_evidence = copy.deepcopy(payload)
    unknown_evidence["permissions"]["local_decode_analysis"] = {
        "decision": "allowed",
        "evidence_ids": ["missing"],
    }
    with pytest.raises(ValueError, match="unknown evidence_ids"):
        parse_dataset_qualification(unknown_evidence)


def test_checker_rejects_missing_extra_or_hash_changed_assets(
    tmp_path: Path,
) -> None:
    _audit_path, qualification_path = _pending_qualification(tmp_path)
    original = json.loads(qualification_path.read_text(encoding="utf-8"))

    missing = copy.deepcopy(original)
    missing["assets"].pop()
    missing_path = tmp_path / "metadata" / "missing.json"
    write_json_artifact(missing_path, missing)
    with pytest.raises(ValueError, match="exactly cover"):
        check_dataset_qualification(
            data_root=tmp_path, qualification_path=missing_path
        )

    extra = copy.deepcopy(original)
    extra_asset = copy.deepcopy(extra["assets"][0])
    extra_asset["relative_path"] = "extra.mp4"
    extra_asset["sha256"] = "e" * 64
    extra["assets"].append(extra_asset)
    extra_path = tmp_path / "metadata" / "extra.json"
    write_json_artifact(extra_path, extra)
    with pytest.raises(ValueError, match="exactly cover"):
        check_dataset_qualification(
            data_root=tmp_path, qualification_path=extra_path
        )

    changed = copy.deepcopy(original)
    changed["assets"][0]["sha256"] = "f" * 64
    changed_path = tmp_path / "metadata" / "changed.json"
    write_json_artifact(changed_path, changed)
    with pytest.raises(ValueError, match="differs"):
        check_dataset_qualification(
            data_root=tmp_path, qualification_path=changed_path
        )


def test_checker_rejects_tampered_audit_binding(tmp_path: Path) -> None:
    audit_path, qualification_path = _pending_qualification(tmp_path)
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_payload["opencv_version"] = "tampered"
    audit_path.write_text(json.dumps(audit_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="audit SHA-256 mismatch"):
        check_dataset_qualification(
            data_root=tmp_path,
            qualification_path=qualification_path,
        )


def test_checker_rejects_noncanonical_qualification_bytes(
    tmp_path: Path,
) -> None:
    _audit_path, qualification_path = _pending_qualification(tmp_path)
    payload = json.loads(qualification_path.read_text(encoding="utf-8"))
    pretty_path = tmp_path / "metadata" / "pretty-qualification.json"
    pretty_path.write_text(
        json.dumps(payload, indent=4, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical JSON"):
        check_dataset_qualification(
            data_root=tmp_path,
            qualification_path=pretty_path,
        )


def test_checker_rejects_missing_or_changed_asset_files(
    tmp_path: Path,
) -> None:
    _audit_path, qualification_path = _pending_qualification(tmp_path)
    missing_asset = tmp_path / "source-media" / "clip-1.mp4"
    missing_asset.unlink()
    with pytest.raises(ValueError, match="not a regular file"):
        check_dataset_qualification(
            data_root=tmp_path,
            qualification_path=qualification_path,
        )

    missing_asset.write_bytes(b"changed asset bytes\n")
    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        check_dataset_qualification(
            data_root=tmp_path,
            qualification_path=qualification_path,
        )


def test_checker_rejects_symlinked_asset_alias(tmp_path: Path) -> None:
    _audit_path, qualification_path = _pending_qualification(tmp_path)
    asset = tmp_path / "source-media" / "clip-1.mp4"
    alias_target = tmp_path / "alias-target.mp4"
    alias_target.write_bytes(asset.read_bytes())
    asset.unlink()
    try:
        asset.symlink_to(alias_target)
    except OSError:
        pytest.skip("file symlinks are unavailable on this platform")

    with pytest.raises(ValueError, match="must not contain symlinks"):
        check_dataset_qualification(
            data_root=tmp_path,
            qualification_path=qualification_path,
        )


def test_template_is_idempotent_no_overwrite_and_never_decodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_path = _valid_audit(tmp_path)
    output = tmp_path / "metadata" / "qualification.json"

    def fail_decode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("qualification template must not decode media")

    monkeypatch.setattr(recorded_dataset.cv2, "VideoCapture", fail_decode)
    first = write_dataset_qualification_template(
        data_root=tmp_path,
        dataset_audit_path=audit_path,
        qualification_id="qualification-fixture-v1",
        asset_root="source-media",
        output_path=output,
    )
    second = write_dataset_qualification_template(
        data_root=tmp_path,
        dataset_audit_path=audit_path,
        qualification_id="qualification-fixture-v1",
        asset_root="source-media",
        output_path=output,
    )
    assert first == second

    output.write_text("different\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_dataset_qualification_template(
            data_root=tmp_path,
            dataset_audit_path=audit_path,
            qualification_id="qualification-fixture-v1",
            asset_root="source-media",
            output_path=output,
        )


def test_qualification_outputs_must_remain_under_data_root(
    tmp_path: Path,
) -> None:
    audit_path, qualification_path = _pending_qualification(tmp_path)

    with pytest.raises(ValueError, match="data_root"):
        write_dataset_qualification_template(
            data_root=tmp_path,
            dataset_audit_path=audit_path,
            qualification_id="escaping-output-v1",
            asset_root="source-media",
            output_path=tmp_path.parent / "escaping-qualification.json",
        )
    with pytest.raises(ValueError, match="data_root"):
        check_dataset_qualification(
            data_root=tmp_path,
            qualification_path=qualification_path,
            report_path=tmp_path.parent / "escaping-check.json",
        )


def test_qualification_outputs_reject_symlinked_parent(
    tmp_path: Path,
) -> None:
    audit_path, qualification_path = _pending_qualification(tmp_path)
    real_output = tmp_path / "real-output"
    real_output.mkdir()
    linked_output = tmp_path / "linked-output"
    try:
        linked_output.symlink_to(real_output, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this platform")

    with pytest.raises(ValueError, match="must not contain symlinks"):
        write_dataset_qualification_template(
            data_root=tmp_path,
            dataset_audit_path=audit_path,
            qualification_id="symlink-output-v1",
            asset_root="source-media",
            output_path=linked_output / "qualification.json",
        )
    with pytest.raises(ValueError, match="must not contain symlinks"):
        check_dataset_qualification(
            data_root=tmp_path,
            qualification_path=qualification_path,
            report_path=linked_output / "check.json",
        )


def test_qualification_cli_exit_codes(tmp_path: Path) -> None:
    audit_path = _valid_audit(tmp_path)
    qualification_path = tmp_path / "metadata" / "qualification.json"
    assert dataset_main(
        [
            "qualification-template",
            "--root",
            str(tmp_path),
            "--dataset-audit",
            str(audit_path),
            "--qualification-id",
            "qualification-fixture-v1",
            "--asset-root",
            "source-media",
            "--output",
            "metadata/qualification.json",
        ]
    ) == 0
    assert dataset_main(
        [
            "qualification-check",
            "--root",
            str(tmp_path),
            "--qualification",
            "metadata/qualification.json",
            "--report",
            "outputs/pending-check.json",
        ]
    ) == 2

    invalid_path = tmp_path / "metadata" / "invalid.json"
    invalid_path.write_text("{}", encoding="utf-8")
    assert dataset_main(
        [
            "qualification-check",
            "--root",
            str(tmp_path),
            "--qualification",
            "metadata/invalid.json",
            "--report",
            "outputs/invalid-check.json",
        ]
    ) == 1

    eligible_path = tmp_path / "metadata" / "eligible.json"
    write_json_artifact(eligible_path, _eligible_payload(qualification_path))
    assert dataset_main(
        [
            "qualification-check",
            "--root",
            str(tmp_path),
            "--qualification",
            "metadata/eligible.json",
            "--report",
            "outputs/eligible-check.json",
        ]
    ) == 0


def test_group_cannot_cross_splits(tmp_path: Path) -> None:
    _audit_path, qualification_path = _pending_qualification(tmp_path)
    payload = _eligible_payload(qualification_path)
    assets = payload["assets"]
    assert isinstance(assets, list)
    assets[1]["capture_group_id"] = assets[0]["capture_group_id"]
    crossed = tmp_path / "metadata" / "crossed.json"
    write_json_artifact(crossed, payload)

    report = check_dataset_qualification(
        data_root=tmp_path,
        qualification_path=crossed,
    )

    assert report.eligible is False
    assert "capture_group_id.capture-1.crosses_splits" in report.blockers


def test_qualification_hash_is_exact_file_hash(tmp_path: Path) -> None:
    _audit_path, qualification_path = _pending_qualification(tmp_path)

    report = check_dataset_qualification(
        data_root=tmp_path,
        qualification_path=qualification_path,
    )

    assert report.qualification_sha256 == file_sha256(qualification_path)


def test_v1_template_and_check_payloads_do_not_gain_v2_fields(
    tmp_path: Path,
) -> None:
    _audit_path, qualification_path = _pending_qualification(tmp_path)
    qualification_payload = json.loads(
        qualification_path.read_text(encoding="utf-8")
    )
    report = check_dataset_qualification(
        data_root=tmp_path,
        qualification_path=qualification_path,
    )
    report_payload = report.to_payload()

    assert qualification_payload["qualification_schema_version"] == 1
    assert qualification_payload["purpose"] == "local-raw-pilot"
    assert "assets" in qualification_payload
    assert "provider_assets" not in qualification_payload
    assert "project_preregistration" not in qualification_payload
    assert report_payload["qualification_check_schema_version"] == 1
    assert report_payload["purpose"] == "local-raw-pilot"
    assert "operations" not in report_payload
    assert "selected_asset_count" not in report_payload


def test_v2_diagnostic_checks_selected_asset_without_suite_completeness(
    tmp_path: Path,
) -> None:
    _audit, preregistration_path, qualification_path = (
        _pending_v2_qualification(
            tmp_path,
            operations=(
                "decode_analysis",
                "extract_bounded_window",
                "render_local_media",
            ),
        )
    )
    eligible_path = tmp_path / "metadata" / "eligible-v2.json"
    write_json_artifact(
        eligible_path,
        _eligible_v2_payload(qualification_path),
    )

    qualification = load_dataset_qualification(eligible_path)
    report = check_dataset_qualification(
        data_root=tmp_path,
        qualification_path=eligible_path,
        report_path=tmp_path / "outputs" / "v2-check.json",
    )

    assert isinstance(qualification, DatasetQualificationV2)
    assert isinstance(report, QualificationCheckReportV2)
    assert report.eligible is True
    assert report.asset_count == 2
    assert report.selected_asset_count == 1
    assert report.development_asset_count == 1
    assert report.held_out_asset_count == 0


@pytest.mark.parametrize("failure", ("missing", "tampered"))
def test_v2_checker_re_reads_content_hashed_evidence_snapshots(
    tmp_path: Path,
    failure: str,
) -> None:
    _audit, _preregistration, qualification_path = _pending_v2_qualification(
        tmp_path
    )
    eligible_path = tmp_path / "metadata" / "eligible-v2.json"
    write_json_artifact(
        eligible_path,
        _eligible_v2_payload(qualification_path),
    )
    terms_snapshot = tmp_path / "metadata" / "fixture-terms.txt"
    if failure == "missing":
        terms_snapshot.unlink()
        expected = "is not a regular file"
    else:
        terms_snapshot.write_text("changed terms\n", encoding="utf-8")
        expected = "snapshot SHA-256 mismatch"

    with pytest.raises(ValueError, match=expected):
        check_dataset_qualification(
            data_root=tmp_path,
            qualification_path=eligible_path,
        )


def test_v2_checker_rejects_symlinked_evidence_snapshot(
    tmp_path: Path,
) -> None:
    _audit, _preregistration, qualification_path = _pending_v2_qualification(
        tmp_path
    )
    payload = _eligible_v2_payload(qualification_path)
    real_snapshot = tmp_path / "metadata" / "fixture-terms.txt"
    linked_snapshot = tmp_path / "metadata" / "linked-terms.txt"
    try:
        linked_snapshot.symlink_to(real_snapshot)
    except OSError:
        pytest.skip("file symlinks are unavailable on this platform")
    payload["evidence_records"][0]["reference"] = (
        "metadata/linked-terms.txt"
    )
    eligible_path = tmp_path / "metadata" / "eligible-v2-symlink.json"
    write_json_artifact(eligible_path, payload)

    with pytest.raises(ValueError, match="must not contain symlinks"):
        check_dataset_qualification(
            data_root=tmp_path,
            qualification_path=eligible_path,
        )
    assert report.operations == (
        "decode_analysis",
        "extract_bounded_window",
        "render_local_media",
    )
    assert report.required_permission_scopes == (
        "local_decode_analysis",
        "bounded_interval_extraction",
        "local_derivative_construction",
    )
    assert report.project_preregistration_sha256 == file_sha256(
        preregistration_path
    )
    assert qualification.provider_assets[1].fact_status == "unresolved"
    assert report.to_payload()["purpose"] == "diagnostic_raw_pilot"


def test_v2_diagnostic_requires_selected_provider_fact_but_not_unselected(
    tmp_path: Path,
) -> None:
    _audit, _preregistration, qualification_path = _pending_v2_qualification(
        tmp_path
    )
    pending_rights = _eligible_v2_payload(qualification_path)
    provider_assets = pending_rights["provider_assets"]
    assert isinstance(provider_assets, list)
    selected = next(
        asset
        for asset in provider_assets
        if asset["relative_path"] == "clip-1.mp4"
    )
    selected["fact_status"] = "unresolved"
    selected["capture_group_id"] = None
    selected["fact_evidence_ids"] = []
    pending_path = tmp_path / "metadata" / "pending-provider-fact.json"
    write_json_artifact(pending_path, pending_rights)

    report = check_dataset_qualification(
        data_root=tmp_path,
        qualification_path=pending_path,
    )

    assert report.eligible is False
    assert "provider_asset.clip-1.mp4.facts_unresolved" in report.blockers
    assert (
        "provider_asset.clip-1.mp4.capture_group_id_missing"
        in report.blockers
    )
    assert not any("clip-2.mp4" in blocker for blocker in report.blockers)


def test_v2_evidence_suite_requires_full_partition_and_verified_active_facts(
    tmp_path: Path,
) -> None:
    audit_path = _valid_audit(tmp_path, asset_count=3)
    preregistration_path = tmp_path / "metadata" / "project-preregistration.json"
    write_project_preregistration(
        data_root=tmp_path,
        dataset_audit_path=audit_path,
        preregistration_id="qualification-fixture-suite-v1",
        purpose="evidence_suite",
        created_at_utc="2026-07-31T03:00:00Z",
        operations=("decode_analysis",),
        development_assets=("clip-1.mp4",),
        held_out_assets=("clip-2.mp4",),
        excluded_assets=(("clip-3.mp4", "predeclared protocol exclusion"),),
        output_path=preregistration_path,
    )
    qualification_path = tmp_path / "metadata" / "qualification-v2.json"
    write_dataset_qualification_template(
        data_root=tmp_path,
        dataset_audit_path=audit_path,
        qualification_id="qualification-fixture-suite-v2",
        asset_root="source-media",
        output_path=qualification_path,
        qualification_schema_version=2,
        purpose="evidence_suite",
        project_preregistration_path=preregistration_path,
    )
    payload = _eligible_v2_payload(qualification_path)
    provider_assets = payload["provider_assets"]
    assert isinstance(provider_assets, list)
    held_out = next(
        asset
        for asset in provider_assets
        if asset["relative_path"] == "clip-2.mp4"
    )
    held_out["capture_group_id"] = "provider-capture-2"
    held_out["gas_identity"] = "methane"
    held_out["fact_status"] = "verified"
    held_out["fact_evidence_ids"] = ["provider-metadata-snapshot"]
    eligible_path = tmp_path / "metadata" / "eligible-suite-v2.json"
    write_json_artifact(eligible_path, payload)

    report = check_dataset_qualification(
        data_root=tmp_path,
        qualification_path=eligible_path,
    )

    assert isinstance(report, QualificationCheckReportV2)
    assert report.eligible is True
    assert report.asset_count == 3
    assert report.selected_asset_count == 2
    assert report.development_asset_count == 1
    assert report.held_out_asset_count == 1
    assert report.excluded_asset_count == 1


def test_v2_provider_facts_cannot_contain_project_split(
    tmp_path: Path,
) -> None:
    _audit, _preregistration, qualification_path = _pending_v2_qualification(
        tmp_path
    )
    payload = _eligible_v2_payload(qualification_path)
    provider_assets = payload["provider_assets"]
    assert isinstance(provider_assets, list)
    provider_assets[0]["split"] = "development"

    with pytest.raises(ValueError, match="provider asset has unknown keys"):
        parse_dataset_qualification(payload)


def test_v2_requires_content_hash_for_eligibility_changing_evidence(
    tmp_path: Path,
) -> None:
    _audit, _preregistration, qualification_path = _pending_v2_qualification(
        tmp_path
    )
    permission_payload = _eligible_v2_payload(qualification_path)
    permission_payload["evidence_records"][0]["content_sha256"] = None
    with pytest.raises(ValueError, match="content-hashed"):
        parse_dataset_qualification(permission_payload)

    provider_payload = _eligible_v2_payload(qualification_path)
    provider_payload["evidence_records"][1]["content_sha256"] = None
    with pytest.raises(ValueError, match="content-hashed authoritative"):
        parse_dataset_qualification(provider_payload)


def test_v2_operations_drive_permissions_and_undeclared_outputs_fail(
    tmp_path: Path,
) -> None:
    _audit, preregistration_path, qualification_path = (
        _pending_v2_qualification(
            tmp_path,
            operations=("decode_analysis", "publish_media"),
        )
    )
    payload = _eligible_v2_payload(qualification_path)
    permissions = payload["permissions"]
    assert isinstance(permissions, dict)
    permissions["publish_frames_or_clips"] = {
        "decision": "unknown",
        "evidence_ids": [],
    }
    blocked_path = tmp_path / "metadata" / "blocked-output-v2.json"
    write_json_artifact(blocked_path, payload)

    report = check_dataset_qualification(
        data_root=tmp_path,
        qualification_path=blocked_path,
    )
    preregistration = load_project_preregistration(preregistration_path)

    assert report.eligible is False
    assert (
        "permission.publish_frames_or_clips.unknown" in report.blockers
    )
    with pytest.raises(ValueError, match="not project-preregistered"):
        validate_requested_operations(
            preregistration,
            ("decode_analysis", "render_local_media"),
        )


def test_diagnostic_preregistration_cannot_silently_become_suite(
    tmp_path: Path,
) -> None:
    audit_path = _valid_audit(tmp_path)
    with pytest.raises(ValueError, match="must all be development"):
        write_project_preregistration(
            data_root=tmp_path,
            dataset_audit_path=audit_path,
            preregistration_id="invalid-diagnostic",
            purpose="diagnostic_raw_pilot",
            created_at_utc="2026-07-31T03:00:00Z",
            operations=("decode_analysis",),
            development_assets=("clip-1.mp4",),
            held_out_assets=("clip-2.mp4",),
            output_path=tmp_path / "metadata" / "invalid-diagnostic.json",
        )

    suite_root = tmp_path / "suite"
    suite_audit = _valid_audit(suite_root, asset_count=3)
    with pytest.raises(ValueError, match="exactly partition"):
        write_project_preregistration(
            data_root=suite_root,
            dataset_audit_path=suite_audit,
            preregistration_id="incomplete-suite",
            purpose="evidence_suite",
            created_at_utc="2026-07-31T03:00:00Z",
            operations=("decode_analysis",),
            development_assets=("clip-1.mp4",),
            held_out_assets=("clip-2.mp4",),
            output_path=suite_root / "metadata" / "incomplete-suite.json",
        )


def test_v2_checker_rejects_preregistration_tamper_and_identity_change(
    tmp_path: Path,
) -> None:
    _audit, preregistration_path, qualification_path = _pending_v2_qualification(
        tmp_path
    )
    eligible_path = tmp_path / "metadata" / "eligible-before-tamper.json"
    qualification_payload = _eligible_v2_payload(qualification_path)
    write_json_artifact(eligible_path, qualification_payload)

    preregistration_payload = json.loads(
        preregistration_path.read_text(encoding="utf-8")
    )
    preregistration_payload["assignments"][0]["sha256"] = "f" * 64
    preregistration_path.write_bytes(canonical_json_bytes(preregistration_payload))
    with pytest.raises(ValueError, match="preregistration SHA-256 mismatch"):
        check_dataset_qualification(
            data_root=tmp_path,
            qualification_path=eligible_path,
        )

    qualification_payload["project_preregistration"]["sha256"] = file_sha256(
        preregistration_path
    )
    changed_identity_path = (
        tmp_path / "metadata" / "changed-preregistration-identity.json"
    )
    write_json_artifact(changed_identity_path, qualification_payload)
    with pytest.raises(ValueError, match="unaudited asset identities"):
        check_dataset_qualification(
            data_root=tmp_path,
            qualification_path=changed_identity_path,
        )


def test_v2_cli_builds_separate_preregistration_and_pending_qualification(
    tmp_path: Path,
) -> None:
    audit_path = _valid_audit(tmp_path)
    assert dataset_main(
        [
            "project-preregistration-template",
            "--root",
            str(tmp_path),
            "--dataset-audit",
            str(audit_path),
            "--preregistration-id",
            "cli-diagnostic-v1",
            "--purpose",
            "diagnostic_raw_pilot",
            "--created-at-utc",
            "2026-07-31T03:00:00Z",
            "--operation",
            "decode_analysis",
            "--development-asset",
            "clip-1.mp4",
            "--output",
            "metadata/cli-project-preregistration.json",
        ]
    ) == 0
    assert dataset_main(
        [
            "qualification-template",
            "--root",
            str(tmp_path),
            "--dataset-audit",
            str(audit_path),
            "--qualification-id",
            "cli-qualification-v2",
            "--asset-root",
            "source-media",
            "--schema-version",
            "2",
            "--purpose",
            "diagnostic_raw_pilot",
            "--project-preregistration",
            "metadata/cli-project-preregistration.json",
            "--output",
            "metadata/cli-qualification-v2.json",
        ]
    ) == 0
    assert dataset_main(
        [
            "qualification-check",
            "--root",
            str(tmp_path),
            "--qualification",
            "metadata/cli-qualification-v2.json",
            "--report",
            "outputs/cli-qualification-v2.check.json",
        ]
    ) == 2

    qualification = load_dataset_qualification(
        tmp_path / "metadata" / "cli-qualification-v2.json"
    )
    assert isinstance(qualification, DatasetQualificationV2)
    assert qualification.purpose == "diagnostic_raw_pilot"
    assert len(qualification.provider_assets) == 2
