"""Closeout verification of the frozen evidence-gated focus artifacts.

Recomputes the predeclared criteria directly from the 104 execution records, and
records the one known reporting defect in the study driver: the report's
continuous_operation_preserved status label is overwritten by the aggregate's own
status key, so it reads "unsupported" while its recorded detail shows exact
preservation. This script is additive and never rewrites a frozen artifact.
"""
import argparse
import hashlib
import json
from pathlib import Path

INCREASE_LIMIT = .05
COVERAGE_LIMIT_POINTS = .05


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(root):
    declaration = read(root / "declaration.json")
    records = {}
    for stage in ("development", "heldout"):
        for job in declaration[stage]:
            path = root / "runs" / f"{job['name']}.json"
            records[job["name"]] = (job, read(path)) if path.exists() else (job, None)
    return declaration, records


def pairs(records, stage, profile, cases):
    groups = {}
    for job, record in records.values():
        if record is None or job["stage"] != stage or job["profile"] != profile or job["case"] not in cases:
            continue
        groups.setdefault((job["case"], job["seed"], job["noise"]), {})[job["policy"]] = record
    return [group for group in groups.values() if set(group) == {"control", "candidate"}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/evidence-gated-focus"))
    args = parser.parse_args()
    root = args.root
    declaration, records = load(root)
    report = read(root / "report.json")
    present = sum(1 for _, record in records.values() if record is not None)
    results, contradictions = {}, []

    preservation = {}
    for profile in ("coarse", "double"):
        selection = pairs(records, "heldout", profile, ("continuous", "wind_error"))
        control = sum(group["control"]["J_over_T"] for group in selection)
        candidate = sum(group["candidate"]["J_over_T"] for group in selection)
        coverage = [(group["control"]["metrics"]["focus_core_geometric_coverage"],
                     group["candidate"]["metrics"]["focus_core_geometric_coverage"]) for group in selection]
        defined = all(left is not None and right is not None for left, right in coverage)
        change = (candidate - control) / control if control else None
        points = (sum(right for _, right in coverage) / len(coverage)
                  - sum(left for left, _ in coverage) / len(coverage)) if defined and coverage else None
        passes = bool(change is not None and change <= INCREASE_LIMIT and defined
                      and points is not None and points >= -COVERAGE_LIMIT_POINTS)
        preservation[profile] = {"records": len(selection), "control_J_over_T": control,
                                 "candidate_J_over_T": candidate, "relative_change": change,
                                 "coverage_defined": defined, "coverage_difference_points": points,
                                 "passes": passes}
    results["continuous_operation_preserved"] = {
        "status": "supported" if preservation and all(value["passes"] for value in preservation.values())
                  else "unavailable" if not preservation else "unsupported",
        "rule": {"max_relative_J_over_T_increase": INCREASE_LIMIT,
                 "max_coverage_decrease_points": COVERAGE_LIMIT_POINTS,
                 "missing_coverage_does_not_pass": True},
        "detail": preservation}

    blank = [group for group in pairs(records, "heldout", "coarse", ("blank",))
             + pairs(records, "heldout", "double", ("blank",))]
    false_focus = sum(group["candidate"]["metrics"]["false_focus_decisions"] for group in blank)
    results["no_blank_false_focus_action"] = {"status": "supported" if blank and false_focus == 0 else "unsupported",
                                             "detail": {"pairs": len(blank), "candidate_false_focus": false_focus}}

    cessation, intermittent = [], []
    for job, record in records.values():
        if record is None or job["policy"] != "candidate" or job["stage"] != "heldout":
            continue
        if job["case"] == "cessation":
            cessation.append((job, record))
        if job["case"] == "intermittent":
            intermittent.append((job, record))
    completed = [entry for _, record in cessation for entry in record["change_metrics"]
                 if entry["kind"] == "cessation" and entry["sufficient_remaining_time"]]
    results["cessation_completes_post_change_full"] = {
        "status": "supported" if completed and all(entry["time_to_completed_full_seconds"] is not None
                                                   for entry in completed) else "unsupported",
        "detail": {"sufficient_cases": len(completed),
                   "latency_seconds": [entry["time_to_completed_full_seconds"] for entry in completed]}}
    coarse_resume = [entry for job, record in intermittent if job["profile"] == "coarse"
                     for entry in record["change_metrics"] if entry["kind"] == "resume"]
    doubled_resume = [entry for job, record in intermittent if job["profile"] == "double"
                      for entry in record["change_metrics"] if entry["kind"] == "resume"]
    results["intermittent_reacquires_signal"] = {
        "status": "supported" if coarse_resume and all(entry["reacquisition_latency_seconds"] is not None
                                                       for entry in coarse_resume) else "unsupported",
        "detail": {"coarse_latency_seconds": [entry["reacquisition_latency_seconds"] for entry in coarse_resume],
                   "doubled_sufficient_remaining_time": [entry["sufficient_remaining_time"] for entry in doubled_resume],
                   "doubled_reason": [entry["unavailable_reason"] for entry in doubled_resume]}}

    for name, value in results.items():
        frozen = report["criteria"].get(name, {}).get("status")
        if frozen != value["status"]:
            contradictions.append({"criterion": name, "report_artifact_status": frozen,
                                   "recomputed_status": value["status"],
                                   "explanation": ("report-phase label collision: the aggregate status key "
                                                   "overwrites the criterion status; the recorded detail still "
                                                   "shows the correct values" if name == "continuous_operation_preserved"
                                                   else "unexplained difference; inspect before reuse")})
    result = {"artifact_type": "evidence_gated_focus_criteria_check_v1",
              "report_sha256": sha(root / "report.json"), "records_present": present,
              "criteria": results, "differences_from_report_artifact": contradictions,
              "note": "Recomputed from the 104 execution records; additive closeout check."}
    target = root / "checks" / "predeclared-criteria.json"
    target.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"records": present, "criteria": {k: v["status"] for k, v in results.items()},
                      "differences": contradictions}, indent=1))
    unexplained = [row for row in contradictions if "unexplained" in row["explanation"]]
    if unexplained:
        raise SystemExit("Unexplained criterion difference")
    if results["continuous_operation_preserved"]["status"] != "supported":
        raise SystemExit("Continuous operation does not satisfy the predeclared tolerance")


if __name__ == "__main__":
    main()
