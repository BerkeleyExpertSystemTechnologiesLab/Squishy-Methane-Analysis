"""Command-line adapter for the headless methane benchmark."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from plume_focus.benchmark_cases import make_benchmark_suite
from plume_focus.benchmark_runner import run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the deterministic, cost-matched methane scan benchmark."
        )
    )
    parser.add_argument(
        "--suite",
        choices=("development", "held-out"),
        default="development",
        help="Episode manifest to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON report path (defaults under outputs/).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run one short development case to verify the harness; "
            "this is not benchmark evidence."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    suite = make_benchmark_suite(args.suite)
    if args.smoke:
        if args.suite != "development":
            parser.error("--smoke is only valid for the development suite.")
        suite = replace(
            suite,
            suite_id=f"{suite.suite_id}-smoke",
            episodes=suite.episodes[:1],
            budgets=suite.budgets[:1],
            scans_per_run=2,
        )
    output = (
        args.output
        if args.output is not None
        else Path("outputs") / f"methane-fair-benchmark-{args.suite}-v1.json"
    )

    def progress(completed: int, total: int, label: str) -> None:
        if completed == total or completed % 10 == 0:
            print(f"[{completed}/{total}] {label}")

    report = run_benchmark(suite, progress=progress)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.to_json() + "\n", encoding="utf-8")
    print(f"Claim status: {report.claim_status}")
    print(f"Wrote benchmark report to {output}")
    return 2 if report.claim_status == "invalid" else 0


if __name__ == "__main__":
    raise SystemExit(main())
