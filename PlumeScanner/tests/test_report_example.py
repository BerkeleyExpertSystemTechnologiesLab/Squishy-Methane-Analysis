import json
from pathlib import Path


def test_canonical_report_demonstrates_the_documented_smoke_contract() -> None:
    report_path = (
        Path(__file__).resolve().parents[1]
        / "outputs"
        / "examples"
        / "causal-methane.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    sweep = report["actual_sweep_evaluation"]
    policy = report["iterative_roi_policy"]

    assert report["report_schema_version"] == 3
    assert report["evidence_mode"] == "scan"
    assert report["timeline_integrity"]["sequential_non_overlapping_scans"]
    assert len(report["operational_trace"]["schedule"]) == 8
    assert report["belief_integrity"]["status"] == "evaluated"
    assert report["belief_integrity"]["all_finite"]
    assert report["belief_integrity"]["all_normalized"]
    assert report["sample_variation"]["comparisons"] > 0
    assert report["sample_variation"]["changed_values"] > 0
    assert report["sample_variation"]["variation_observed"]
    assert report["sample_variation"]["full_frame_evidence_mode"] is False
    assert report["termination_reason"] == "max_scans_reached"
    assert sweep["success_mass_recall_threshold"] == 0.9
    assert sweep["mean_probability_mass_recall"] >= 0.9
    assert sweep["success_rate"] == 1.0
    assert policy["mean_swept_area_fraction"] < 1.0
