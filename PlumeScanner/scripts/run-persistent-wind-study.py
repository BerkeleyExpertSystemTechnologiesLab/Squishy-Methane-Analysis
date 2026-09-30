"""Declared persistent-wind comparisons, convergence and provenance adapter."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

from plume_focus.sensing_cli_v1 import _run_job, _convergence_job, config_from_dict, load_config, write_json, artifact_manifest
from plume_focus.sensing_execution_v3 import with_persistent_motion
from plume_focus.sensing_contracts import CausalWind
from plume_focus.sensing_session import configuration_hash


ROOT = Path(__file__).resolve().parents[1]


def script_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def configurations(manifest, order):
    old = script_module("v2_suite", "run-sensing-v2-development.py")
    _, originals, _ = old.declared_configs(ROOT / manifest["base_suite"])
    result = {}
    for name, baseline in originals.items():
        cov = manifest["wind_bias_covariance_yx"] if "-wind-error-" in name else manifest["known_wind_covariance_yx"]
        for estimator, config in (("baseline", baseline), ("motion", with_persistent_motion(baseline, cov, order))):
            result[f"{name}-{estimator}"] = {"role": "main", "estimator": estimator, "case_name": name, "configuration": config.to_dict()}
    base = load_config(ROOT / "configs/sensing-v2/coarse-full-startup.json")
    suite = json.loads((ROOT / manifest["base_suite"]).read_text())
    for scale in manifest["diagnostic_signal_scales"]:
        for seed in manifest["diagnostic_seeds"]:
            baseline = replace(base, seed=seed, source=replace(base.source, signal_scale=scale),
                session=replace(base.session, startup_mode="prior", source_position=None,
                    policy=replace(base.session.policy, corridor_yxyx=tuple(suite["source_free_corridor_yxyx"]))),
                wind=replace(base.wind, velocity_yx=tuple(a+b for a,b in zip(base.wind.velocity_yx, (.012, -.015))), error_speed=.02))
            name = f"diagnostic-source-free-biased-scale-{scale:g}-seed-{seed}"
            for estimator, config in (("baseline", baseline), ("motion", with_persistent_motion(baseline, manifest["wind_bias_covariance_yx"], order))):
                result[f"{name}-{estimator}"] = {"role": "diagnostic", "estimator": estimator, "case_name": name, "configuration": config.to_dict()}
    assert len(result) == 84
    return result


def compare_numbers(coarse, fine, manifest):
    if coarse is None or fine is None:
        return {"coarse": coarse, "fine": fine, "passed": False}
    difference = abs(coarse-fine)
    tolerance = max(manifest["absolute_tolerance"], manifest["relative_tolerance"]*abs(fine))
    return {"coarse": coarse, "fine": fine, "difference": difference, "tolerance": tolerance, "passed": difference <= tolerance}


def run_jobs(jobs, output, workers, grid=None):
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {}
        for name, job in jobs.items():
            saved = output / name / "report.json"
            expected = config_from_dict(job["configuration"])
            if grid is not None:
                from plume_focus.sensing_execution_v1 import ReconstructionGridV1
                expected = replace(expected, evaluation=ReconstructionGridV1.from_dict(grid))
            if saved.exists():
                report = json.loads(saved.read_text())
                if configuration_hash(report["configuration"]) != configuration_hash(expected.to_dict()):
                    raise ValueError(f"Existing result has a different configuration: {name}")
                results.append(record(name, job, report)); continue
            print("Starting", name, flush=True)
            pending[pool.submit(_run_job, name, job["configuration"], str(output), grid)] = (name, job)
        for future in as_completed(pending):
            name, job = pending[future]; future.result()
            report = json.loads((output/name/"report.json").read_text())
            item = record(name, job, report); results.append(item)
            print("Completed", name, "J/T", item["J_over_T"], "tracking", item["tracking_count"], flush=True)
            write_json(output / "progress.json", {"completed_this_invocation": len(results), "requested":len(jobs), "last":item})
    return sorted(results, key=lambda x: x["name"])


def record(name, job, report):
    loss = report["reconstruction"]
    return {"name":name, "role":job["role"], "estimator":job["estimator"], "case_name":job["case_name"],
            "J":loss["J"], "J_over_T":loss["J_over_T"], "expected_excess_integral":loss["expected_excess_integral"],
            "signed_mean_enhancement_error_mean":loss["signed_mean_enhancement_error_mean"],
            "actual_scan_seconds":report["total_scan_seconds"], "tracking_count":report["successful_tracking_count"],
            "fallback_count":report["reacquisition_count"], "operational_hash":report["operational_hash"],
            "configuration_hash":configuration_hash(report["configuration"]), "report":f"{name}/report.json"}


def convergence(manifest, output, workers):
    diagnostics = script_module("motion_diagnostics", "diagnose-persistent-wind.py")
    controls = {str(order): diagnostics.controlled_forecasts(order) for order in manifest["orders"]}
    write_json(output / "controlled-forecasts.json", controls)
    results, checks, selected, status = {}, [], manifest["orders"][-1], "incomplete"
    for order in manifest["orders"]:
        all_jobs = configurations(manifest, order)
        jobs = {name+"-motion":all_jobs[name+"-motion"] for name in manifest["representative_cases"]}
        results[order] = run_jobs(jobs, output / f"order-{order}", workers)
        if order == manifest["orders"][0]:
            continue
        previous = manifest["orders"][manifest["orders"].index(order)-1]
        arithmetic = []
        for a, b in zip(results[previous], results[order]):
            arithmetic.append({"case":a["case_name"], "J":compare_numbers(a["J"], b["J"], manifest),
                               "expected_excess_integral":compare_numbers(a["expected_excess_integral"],b["expected_excess_integral"],manifest)})
        controlled = [{"case":a["case"], "horizon_seconds":a["horizon_seconds"],
                       **compare_numbers(a["expected_excess"],b["expected_excess"],manifest)}
                      for a,b in zip(controls[str(previous)],controls[str(order)])]
        passed = all(row[key]["passed"] for row in arithmetic for key in ("J","expected_excess_integral")) and all(r["passed"] for r in controlled)
        checks.append({"coarse_order":previous,"fine_order":order,"runs":arithmetic,"controlled":controlled,"passed":passed})
        write_json(output / "hypothesis-checks.json", checks)
        if passed:
            selected, status = previous, "converged"; break
    jobs = configurations(manifest, selected)
    spatial = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(_convergence_job, name, jobs[name+"-motion"]["configuration"],
                   {"absolute_tolerance":manifest["absolute_tolerance"],"relative_tolerance":manifest["relative_tolerance"],
                    "maximum_refinements":manifest["maximum_evaluation_refinements"]}, str(output)):name
                   for name in manifest["representative_cases"]}
        for future in as_completed(pending):
            item=future.result(); spatial.append(item)
            print("Spatial/time refinement",item["fixture"],item["status"],flush=True)
    spatial_complete = all(r["accepted_level"] is not None for r in spatial)
    level = max(r["accepted_level"] if r["accepted_level"] is not None else manifest["maximum_evaluation_refinements"] for r in spatial)
    grid = config_from_dict(next(iter(jobs.values()))["configuration"]).evaluation
    grid = replace(grid,shape=tuple(n*2**level for n in grid.shape),step_seconds=grid.step_seconds/2**level)
    freeze = {"hypothesis_status":status,"hypothesis_order":selected,"hypothesis_checks":checks,
              "evaluation_status":"converged" if spatial_complete else "incomplete","evaluation":grid.to_dict(),
              "evaluation_checks":spatial,"status":"converged" if status=="converged" and spatial_complete else "incomplete",
              "limitation":"Two representative paths plus controlled fixtures, not universal convergence. If incomplete, finest tested approximation remains provisional."}
    write_json(output.parent / "numerical-freeze.json",freeze)
    return freeze


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--phase",choices=("declare","diagnose","convergence","run","all"),default="all")
    parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args()
    manifest_path=ROOT/"configs/sensing-motion-v3/study.json"
    manifest=json.loads(manifest_path.read_text())
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    declaration={"manifest":manifest,"manifest_sha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                 "initial_configurations":configurations(manifest,3)}
    declaration=json.loads(json.dumps(declaration,allow_nan=False))
    path=output/"declared-study.json"
    if path.exists():
        if json.loads(path.read_text())!=declaration:
            raise ValueError("Declaration changed; use a new output directory.")
    else:
        write_json(path,declaration)
    if args.phase in ("all","diagnose"):
        diagnostics=script_module("motion_diagnostics","diagnose-persistent-wind.py")
        target=output/"diagnosis"/"fixed-evidence.json"
        if not target.exists():
            write_json(target,diagnostics.fixed_evidence(ROOT/"outputs/mechanism-revision/development",manifest["representative_cases"]))
    if args.phase in ("all","convergence"):
        convergence(manifest,output/"convergence",args.workers)
    if args.phase in ("all","run"):
        freeze=json.loads((output/"numerical-freeze.json").read_text())
        jobs=configurations(manifest,freeze["hypothesis_order"])
        write_json(output/"execution-configurations.json",jobs)
        results=run_jobs(jobs,output/"development",args.workers,freeze["evaluation"])
        groups={}
        for row in results:
            groups.setdefault(row["case_name"],{})[row["estimator"]]=row
        pairs=[]
        for name, group in groups.items():
            a,b=group["baseline"],group["motion"]
            pairs.append({"case":name,"role":a["role"],"baseline_J_over_T":a["J_over_T"],"motion_J_over_T":b["J_over_T"],
                          "motion_minus_baseline":b["J_over_T"]-a["J_over_T"],
                          "baseline_scan_seconds":a["actual_scan_seconds"],"motion_scan_seconds":b["actual_scan_seconds"]})
        write_json(output/"study-summary.json",{"engineering_execution":"complete","claim_status":"not_demonstrated",
            "numerical_status":freeze["status"],"results":results,"pairs":pairs,
            "limitations":["Descriptive two-noise-seed evidence, not independent plume realizations.","Common ceilings are not equal expenditure.",
                           "Changing local drift remains required and deferred."]})
    artifact_manifest(output)


if __name__=="__main__":
    main()
