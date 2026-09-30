"""Descriptive paired summaries; distinguishes estimator and allocation effects."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from plume_focus.sensing_cli_v1 import write_json


def available_mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args=parser.parse_args(); root=args.directory
    summary=json.loads((root/"study-summary.json").read_text())
    groups={}; motion=[]; historical=[]; details=[]; configurations={}; executions={}
    old_root=Path(__file__).resolve().parents[1]/"outputs/mechanism-revision/development"
    for item in summary["results"]:
        report=json.loads((root/"development"/item["report"]).read_text())
        name=item["case_name"]; group=name.rsplit("-seed-",1)[0]
        configurations.setdefault(name,{})[item["estimator"]]=report["configuration"]
        trace=json.loads((root/"development"/item["name"]/"trace.json").read_text())
        executions.setdefault(name,{})[item["estimator"]]={
            "delivery_and_receipts":[e for e in trace["events"] if e["operation"]!="plan_next"],
            "actual_segments":[e["executed_plan"]["segments"] for e in trace["executions"]]}
        groups.setdefault((group,item["estimator"]),[]).append({**item,
            "signal_index_squared_error_mean":report["reconstruction"]["signal_index_squared_error_mean"]})
        rows = report["scan_schedule"]
        recall = report["mean_interval_mass_recall"]
        bins=[]
        horizon=report["reconstruction"]["horizon_seconds"]
        for start in np.arange(0,horizon,25.):
            end=min(start+25.,horizon)
            loss=sum(max(0.,min(end,q["time_seconds"]+q["duration_seconds"])-max(start,q["time_seconds"]))*q["map_mse"]
                     for q in report["reconstruction"]["quadrature_samples"])
            bins.append({"start_seconds":float(start),"end_seconds":float(end),
                         "J":float(loss) if report["reconstruction"]["J"] is not None else None,
                         "J_over_interval":float(loss/(end-start)) if report["reconstruction"]["J"] is not None else None})
        details.append({"case":name,"estimator":item["estimator"],
            "post_hoc_25_second_loss_bins":bins,
            "mean_interval_missed_signal_fraction":1-recall if recall is not None else None,
            "mean_decision_background_allocation":available_mean(r["predicted_background_mass_fraction"] for r in rows),
            "mean_receipt_reportable_signal_recall":available_mean(r["actual_reportable_signal_recall"] for r in rows),
            "mean_uncertainty_excess_fraction":available_mean(r["uncertainty_excess_fraction"] for r in rows),
            "actual_time_components_seconds":report["actual_time_components_seconds"],
            "visited_fraction":report["visited_fraction"],
            "final_observed_age_quantiles_seconds":rows[-1]["observed_age_quantiles_seconds"] if rows else None,
            "final_stale_or_unvisited_fraction":rows[-1]["stale_or_unvisited_fraction"] if rows else None,
            "tracking_shortfall_count":report["tracking_shortfall_count"],"no_action_count":report["no_action_count"],
            "compute_seconds":report["compute_seconds"],"producing_revision":report.get("producing_revision")})
        if item["role"]=="main" and item["estimator"]=="baseline":
            old=json.loads((old_root/name/"report.json").read_text())
            historical.append({"case":name,"same_execution":old["operational_hash"]==item["operational_hash"],
                               "J_difference":item["J"]-old["reconstruction"]["J"]})
        if item["estimator"]=="motion":
            source=report["configuration"]["source"]
            nominal=np.asarray(report["configuration"]["wind"]["velocity_yx"])
            source_velocity=np.asarray(source["wind_velocity"])*source["fps"]/(np.asarray(source["shape"])-1)
            actual_correction=source_velocity-nominal
            posterior=rows[-1]["posterior_motion"] if rows else None
            motion.append({"case":name,"nominal_source_wind_correction_yx":actual_correction.tolist(),
                           "posterior_bias_yx":posterior["inferred_bias_yx"] if posterior else None,
                           "hypothesis_count":posterior["hypothesis_count"] if posterior else None,
                           "zero_bias_prior":not bool(np.any(report["configuration"]["session"]["motion"]["bias_covariance_yx"])),
                           "final_effective_hypotheses":posterior["effective_hypothesis_count"] if posterior else None,
                           "final_maximum_weight":posterior["maximum_weight"] if posterior else None,
                           "rows_with_max_weight_above_99_percent":sum(r["posterior_motion"]["maximum_weight"]>.99 for r in rows),
                           "mean_between_hypothesis_variance":float(np.mean([r["prediction_between_hypothesis_variance"] for r in rows])) if rows else None})
    table=[]
    for (name,estimator),items in sorted(groups.items()):
        table.append({"case":name,"estimator":estimator,"seeds":len(items),
                      **{key:float(np.mean([r[key] for r in items])) for key in ("J_over_T","actual_scan_seconds","tracking_count","fallback_count","signed_mean_enhancement_error_mean","signal_index_squared_error_mean")}})
    primary=[p for p in summary["pairs"] if p["role"]=="main" and "-wind-error-threshold90_segment_v2-" in p["case"]]
    improving=[p for p in primary if p["motion_minus_baseline"] < -max(1e-10,.01*abs(p["baseline_J_over_T"]))]
    worsening=[p for p in primary if p["motion_minus_baseline"] > max(1e-10,.01*abs(p["baseline_J_over_T"]))]
    classification=("numerically_unresolved" if summary["numerical_status"]!="converged" else
                    "supported_in_tested_fixtures" if len(improving)==len(primary) else
                    "mixed" if improving else "unsupported_in_tested_fixtures")
    matching=[]; execution_checks=[]
    for name,pair in configurations.items():
        candidate=deepcopy(pair["motion"])
        candidate["message_type"]=pair["baseline"]["message_type"]
        candidate["session"]["formulation_id"]=pair["baseline"]["session"]["formulation_id"]
        candidate["session"].pop("motion")
        if candidate != pair["baseline"]:
            raise ValueError(f"Unexpected non-motion configuration difference: {name}")
        matching.append(name)
        same=executions[name]["baseline"]==executions[name]["motion"]
        zero=not bool(np.any(pair["motion"]["session"]["motion"]["bias_covariance_yx"]))
        fixed=pair["baseline"]["session"]["policy"]["policy_id"] in ("static_corridor_segment_v2","age_segment_v2")
        if (zero or fixed) and not same:
            raise ValueError(f"Expected identical acquisitions for this control: {name}")
        execution_checks.append({"case":name,"same_readings_delivery_and_motion":same,
                                 "expected_identical":zero or fixed})
    result={"hypothesis_classification":classification,"claim_status":"not_demonstrated", "means":table,"motion":motion,"per_run_diagnostics":details,
            "configuration_pair_checks":{"passed":True,"pairs":matching,"allowed_differences":["run transport message type","formulation identifier","declared motion configuration"]},
            "acquisition_pair_checks":execution_checks,
            "historical_baseline_checks":historical,"primary_improving_count":len(improving),"primary_worsening_count":len(worsening),
            "supplemental_coverage_definition":"mean_interval_missed_signal_fraction is one minus the inherited whole-executed-sweep-mask recall, evaluated at five equally spaced truth times per scan and averaged equally across scans. It is a geometric diagnostic, not contemporaneous acquisition coverage or a physical-time-weighted capture fraction. Background allocation is expected-excess background mass averaged per decision. Causal J/T remains the common-physical-horizon reconstruction outcome.",
            "interpretation":"Descriptive primary comparison is biased-wind adaptive estimator pairs; differences within 1% are numerically small. The post-hoc 25-second bins aid video inspection and do not replace the frozen full-horizon loss. No independent plume replicates or superiority claim. Source-wind correction is evaluator-only nominal generator transport, not calibrated physical wind truth; blank signal cannot identify wind."}
    write_json(root/"descriptive-summary.json",result)
    for row in table:
        print(row,flush=True)
    print("Hypothesis classification:",classification,flush=True)


if __name__=="__main__":
    main()
