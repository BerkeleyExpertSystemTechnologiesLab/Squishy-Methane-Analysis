"""CLI for auditing, preparing, and freezing recorded-video inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from plume_focus.recorded_artifacts import write_json_artifact
from plume_focus.recorded_dataset import (
    audit_dataset_directory,
    load_dataset_accession,
)
from plume_focus.recorded_freeze import freeze_recorded_suite
from plume_focus.recorded_manifest_io import file_sha256
from plume_focus.recorded_preparation import prepare_recorded_dataset
from plume_focus.recorded_qualification import (
    QUALIFICATION_OPERATIONS,
    check_dataset_qualification,
    write_dataset_qualification_template,
    write_project_preregistration,
)


_COMMANDS = frozenset(
    {
        "audit",
        "project-preregistration-template",
        "qualification-template",
        "qualification-check",
        "prepare",
        "freeze",
    }
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit external plume videos, prepare truth-free recorded "
            "inputs, or freeze a prepared suite. None of these commands "
            "creates a performance claim."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser(
        "audit",
        help="Create a deterministic hash/decode accession audit.",
    )
    audit.add_argument("--root", type=Path, required=True)
    audit.add_argument(
        "--accession-manifest",
        type=Path,
        required=True,
        help="Strict JSON declaration of source, accession, and rights.",
    )
    audit.add_argument("--report", type=Path, required=True)
    audit.set_defaults(handler=_run_audit)

    preregistration = commands.add_parser(
        "project-preregistration-template",
        help=(
            "Freeze an outcome-blind project asset selection and operation "
            "plan without decoding media."
        ),
    )
    preregistration.add_argument("--root", type=Path, required=True)
    preregistration.add_argument(
        "--dataset-audit",
        "--audit",
        dest="dataset_audit",
        type=Path,
        required=True,
    )
    preregistration.add_argument("--preregistration-id", required=True)
    preregistration.add_argument(
        "--purpose",
        choices=("diagnostic_raw_pilot", "evidence_suite"),
        required=True,
    )
    preregistration.add_argument("--created-at-utc", required=True)
    preregistration.add_argument(
        "--operation",
        action="append",
        choices=QUALIFICATION_OPERATIONS,
        required=True,
    )
    preregistration.add_argument(
        "--development-asset",
        action="append",
        default=[],
        metavar="RELATIVE_PATH",
    )
    preregistration.add_argument(
        "--held-out-asset",
        action="append",
        default=[],
        metavar="RELATIVE_PATH",
    )
    preregistration.add_argument(
        "--excluded-asset",
        action="append",
        default=[],
        metavar="RELATIVE_PATH=NON_PERFORMANCE_REASON",
    )
    preregistration.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Data-root-relative output path, or an absolute path inside root.",
    )
    preregistration.set_defaults(handler=_run_project_preregistration)

    qualification_template = commands.add_parser(
        "qualification-template",
        help=(
            "Create a pending rights/metadata qualification from one audit "
            "without decoding media."
        ),
    )
    qualification_template.add_argument("--root", type=Path, required=True)
    qualification_template.add_argument(
        "--dataset-audit",
        "--audit",
        dest="dataset_audit",
        type=Path,
        required=True,
    )
    qualification_template.add_argument(
        "--qualification-id",
        required=True,
    )
    qualification_template.add_argument(
        "--asset-root",
        required=True,
        help="Normalized data-root-relative directory containing audit assets.",
    )
    qualification_template.add_argument(
        "--schema-version",
        type=int,
        choices=(1, 2),
        default=1,
        help="Schema v1 is the compatibility default; v2 is purpose-aware.",
    )
    qualification_template.add_argument(
        "--purpose",
        choices=("diagnostic_raw_pilot", "evidence_suite"),
        help="Required by v2 and must match the preregistration.",
    )
    qualification_template.add_argument(
        "--project-preregistration",
        type=Path,
        help="Required content-hashed preregistration input for schema v2.",
    )
    qualification_template.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Data-root-relative output path, or an absolute path inside root.",
    )
    qualification_template.set_defaults(handler=_run_qualification_template)

    qualification_check = commands.add_parser(
        "qualification-check",
        help=(
            "Validate qualification provenance and derive declared-purpose "
            "eligibility."
        ),
    )
    qualification_check.add_argument("--root", type=Path, required=True)
    qualification_check.add_argument(
        "--qualification", type=Path, required=True
    )
    qualification_check.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Data-root-relative report path, or an absolute path inside root.",
    )
    qualification_check.set_defaults(handler=_run_qualification_check)

    prepare = commands.add_parser(
        "prepare",
        help="Build deterministic truth-free classified recorded manifests.",
    )
    prepare.add_argument("--root", type=Path, required=True)
    prepare.add_argument("--spec", type=Path, required=True)
    prepare.add_argument("--profile", type=Path, required=True)
    prepare.add_argument(
        "--output-manifests",
        type=Path,
        required=True,
        help="Must resolve to <root>/manifests.",
    )
    prepare.add_argument("--report", type=Path, required=True)
    prepare.set_defaults(handler=_run_prepare)

    freeze = commands.add_parser(
        "freeze",
        help="Freeze one split from an exact preparation report.",
    )
    freeze.add_argument("--root", type=Path, required=True)
    freeze.add_argument(
        "--preparation-report",
        type=Path,
        required=True,
    )
    freeze.add_argument(
        "--preparation-report-sha256",
        required=True,
    )
    freeze.add_argument("--suite-id", required=True)
    freeze.add_argument(
        "--split",
        choices=("development", "held-out"),
        required=True,
    )
    freeze.add_argument("--frozen-at-utc", required=True)
    freeze.add_argument(
        "--pre-change-model-commit",
        required=True,
    )
    freeze.add_argument("--freeze-provenance", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument(
        "--freeze-record",
        type=Path,
        required=True,
    )
    freeze.set_defaults(handler=_run_freeze)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    normalized = _with_legacy_audit_command(arguments)
    args = build_parser().parse_args(normalized)
    return int(args.handler(args))


def _with_legacy_audit_command(argv: list[str]) -> list[str]:
    """Keep the original flat audit invocation reproducible."""

    if not argv or argv[0] in _COMMANDS or argv[0] in {"-h", "--help"}:
        return argv
    return ["audit", *argv]


def _run_audit(args: argparse.Namespace) -> int:
    accession = load_dataset_accession(args.accession_manifest)
    report = audit_dataset_directory(
        args.root,
        accession,
        accession_manifest_sha256=file_sha256(
            args.accession_manifest
        ),
    )
    write_json_artifact(args.report, report.to_payload())
    print(
        f"Audited {len(report.assets)} video asset(s); "
        f"valid={report.valid}; rights_blocked={report.rights_blocked}"
    )
    print("Claim eligibility: not_sequence_evidence")
    print(f"Wrote dataset audit to {args.report}")
    return 0 if report.valid else 1


def _run_prepare(args: argparse.Namespace) -> int:
    report = prepare_recorded_dataset(
        data_root=args.root,
        spec_path=args.spec,
        profile_path=args.profile,
        output_manifests_dir=args.output_manifests,
        report_path=args.report,
    )
    print(
        f"Prepared {len(report.sequences)} truth-free sequence(s) "
        f"for {report.preparation_id}."
    )
    print("Claim eligibility: not_sequence_evidence")
    print(f"Wrote preparation report to {args.report}")
    return 0


def _run_project_preregistration(args: argparse.Namespace) -> int:
    try:
        excluded_assets = tuple(
            _parse_excluded_asset(value) for value in args.excluded_asset
        )
        preregistration = write_project_preregistration(
            data_root=args.root,
            dataset_audit_path=args.dataset_audit,
            preregistration_id=args.preregistration_id,
            purpose=args.purpose,
            created_at_utc=args.created_at_utc,
            operations=tuple(args.operation),
            development_assets=tuple(args.development_asset),
            held_out_assets=tuple(args.held_out_asset),
            excluded_assets=excluded_assets,
            output_path=args.output,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"Invalid project preregistration: {error}", file=sys.stderr)
        return 1
    print(
        f"Wrote {preregistration.purpose} preregistration with "
        f"{len(preregistration.active_assignments)} selected asset(s) to "
        f"{args.output}."
    )
    print("Claim eligibility: not_sequence_evidence")
    return 0


def _parse_excluded_asset(value: str) -> tuple[str, str]:
    path, separator, reason = value.partition("=")
    if not separator or not path.strip() or not reason.strip():
        raise ValueError(
            "--excluded-asset must use RELATIVE_PATH=NON_PERFORMANCE_REASON."
        )
    return path, reason


def _run_qualification_template(args: argparse.Namespace) -> int:
    try:
        qualification = write_dataset_qualification_template(
            data_root=args.root,
            dataset_audit_path=args.dataset_audit,
            qualification_id=args.qualification_id,
            asset_root=args.asset_root,
            output_path=args.output,
            qualification_schema_version=args.schema_version,
            purpose=args.purpose,
            project_preregistration_path=args.project_preregistration,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"Invalid qualification template request: {error}", file=sys.stderr)
        return 1
    assets = getattr(qualification, "assets", None)
    if assets is None:
        assets = qualification.provider_assets
    print(
        f"Wrote pending qualification for {len(assets)} "
        f"audited asset(s) to {args.output}."
    )
    print("Claim eligibility: not_sequence_evidence")
    return 0


def _run_qualification_check(args: argparse.Namespace) -> int:
    try:
        report = check_dataset_qualification(
            data_root=args.root,
            qualification_path=args.qualification,
            report_path=args.report,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"Invalid dataset qualification: {error}", file=sys.stderr)
        return 1
    print(
        f"Checked {report.asset_count} qualified asset(s); "
        f"eligible={report.eligible}."
    )
    print("Claim eligibility: not_sequence_evidence")
    print(f"Wrote qualification check to {args.report}")
    return 0 if report.eligible else 2


def _run_freeze(args: argparse.Namespace) -> int:
    artifacts = freeze_recorded_suite(
        data_root=args.root,
        preparation_report_path=args.preparation_report,
        preparation_report_sha256=(
            args.preparation_report_sha256
        ),
        suite_id=args.suite_id,
        split=args.split,
        frozen_at_utc=args.frozen_at_utc,
        frozen_before_model_commit=(
            args.pre_change_model_commit
        ),
        freeze_provenance=args.freeze_provenance,
        suite_output_path=args.output,
        freeze_record_output_path=args.freeze_record,
    )
    print(
        f"Froze {artifacts.sequence_count} sequence(s) into "
        f"{artifacts.suite_manifest_path}."
    )
    print("Freeze record claim eligibility: not_sequence_evidence")
    print("Suite publication does not establish a performance claim.")
    print(f"Wrote freeze record to {artifacts.freeze_record_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
