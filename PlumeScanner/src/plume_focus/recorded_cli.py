"""Command-line validation and execution for recorded plume sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plume_focus.recorded_freeze import (
    validate_recorded_freeze_record,
)
from plume_focus.recorded_sources import (
    validate_recorded_sequence,
    validate_recorded_suite,
    validate_recorded_suite_pair,
)
from plume_focus.recorded_workflow import run_recorded_sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate provenance-checked recorded plume inputs or run one "
            "causal recorded-sequence experiment."
        )
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate all declared hashes and manifest contracts.",
    )
    validate_target = validate_parser.add_mutually_exclusive_group(
        required=True
    )
    validate_target.add_argument(
        "--manifest",
        type=Path,
        help="One sequence manifest under a recorded data root.",
    )
    validate_target.add_argument(
        "--suite",
        type=Path,
        help="One frozen development or held-out suite manifest.",
    )
    validate_target.add_argument(
        "--freeze-record",
        type=Path,
        help=(
            "One canonical suite freeze record whose preparation lineage "
            "must also be verified."
        ),
    )
    validate_parser.add_argument(
        "--root",
        type=Path,
        help=(
            "Recorded data root; required with --freeze-record and invalid "
            "for the other validation targets."
        ),
    )

    pair_parser = subparsers.add_parser(
        "validate-pair",
        help=(
            "Validate frozen development and held-out suites together and "
            "reject capture-group or source-media overlap."
        ),
    )
    pair_parser.add_argument(
        "--development-suite",
        type=Path,
        required=True,
    )
    pair_parser.add_argument(
        "--held-out-suite",
        type=Path,
        required=True,
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run one validated sequence through the causal scanner.",
    )
    run_parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Sequence manifest under a recorded data root.",
    )
    run_parser.add_argument(
        "--max-scans",
        type=int,
        required=True,
        help="Maximum number of completed mechanical scans.",
    )
    run_parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Destination JSON evidence report.",
    )
    run_parser.add_argument(
        "--output-video",
        type=Path,
        default=None,
        help="Optional annotated MP4 evidence artifact.",
    )
    run_parser.add_argument(
        "--render-scale",
        type=int,
        default=1,
        help="Optional video pixel scaling factor.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate":
        if args.freeze_record is not None:
            if args.root is None:
                parser.error("--root is required with --freeze-record.")
            validation = validate_recorded_freeze_record(
                args.root,
                args.freeze_record,
            )
        else:
            if args.root is not None:
                parser.error(
                    "--root is valid only with --freeze-record."
                )
            validation = (
                validate_recorded_sequence(args.manifest)
                if args.manifest is not None
                else validate_recorded_suite(args.suite)
            )
        print(json.dumps(validation, indent=2))
        return 0

    if args.command == "validate-pair":
        validation = validate_recorded_suite_pair(
            args.development_suite,
            args.held_out_suite,
        )
        print(json.dumps(validation, indent=2))
        return 0

    if args.max_scans <= 0:
        parser.error("--max-scans must be positive.")
    if args.render_scale <= 0:
        parser.error("--render-scale must be positive.")
    report = run_recorded_sequence(
        args.manifest,
        max_scans=args.max_scans,
        report_output=args.report,
        output_video=args.output_video,
        render_scale=args.render_scale,
    )
    print(
        "Recorded input grade: "
        f"{report.data_provenance['evidence_grade']}"
    )
    print(
        "Claim eligibility: "
        f"{report.data_provenance['claim_eligibility']}"
    )
    print(f"Wrote recorded report to {args.report}")
    if args.output_video is not None:
        print(f"Wrote recorded video to {args.output_video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
