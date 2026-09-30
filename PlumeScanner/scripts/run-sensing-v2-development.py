"""Run predeclared v2 comparisons and mechanism stress cases, preserving v1 artifacts."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path

from plume_focus.sensing_cli_v1 import _run_job, _convergence_job, artifact_manifest, load_config, write_json
from plume_focus.sensing_contracts import CausalWind


def declared_configs(path):
    suite = json.loads(path.read_text())
    if suite["artifact_type"] != "mechanism_revision_development_v2":
        raise ValueError("Wrong development suite version.")
    configs = {}
    for profile in suite["profiles"]:
        base = load_config(path.parent / profile)
        for case in suite["cases"]:
            for policy in suite["policies"]:
                for seed in suite["seeds"]:
                    session = replace(base.session, source_position=None if case["source_free"] else base.session.source_position,
                        policy=replace(base.session.policy, policy_id=policy,
                            corridor_yxyx=tuple(suite["source_free_corridor_yxyx"]) if case["source_free"] else base.session.policy.corridor_yxyx))
                    wind = CausalWind(velocity_yx=tuple(a+b for a,b in zip(base.wind.velocity_yx,case["wind_bias_yx"])), error_speed=case["wind_error_speed"])
                    name = f"{Path(profile).stem}-{case['name']}-{policy}-seed-{seed}"
                    configs[name] = replace(base, session=session, wind=wind, seed=seed)
    base = load_config(path.parent / "coarse-full-startup.json")
    supplemental = {
        "diagnostic-zero-threshold-segments": replace(base, session=replace(base.session, policy=replace(base.session.policy, policy_id="mass90_segment_v2"))),
        "diagnostic-threshold-2sigma": replace(base, session=replace(base.session, policy=replace(base.session.policy, reportable_threshold=.10,
            threshold_provenance="Predeclared 2-sigma sensitivity diagnostic; not chosen on evaluation loss."))),
        "diagnostic-threshold-4sigma": replace(base, session=replace(base.session, policy=replace(base.session.policy, reportable_threshold=.20,
            threshold_provenance="Predeclared 4-sigma sensitivity diagnostic; not chosen on evaluation loss."))),
        "diagnostic-full-height-surveillance": replace(base, session=replace(base.session, policy=replace(base.session.policy, survey_height_fov=1.))),
        "diagnostic-double-full-startup": load_config(path.parent / "double-full-startup.json"),
        "diagnostic-weak-source-free": replace(base, source=replace(base.source, signal_scale=.1), session=replace(base.session, source_position=None)),
        "diagnostic-blank-source-free": replace(base, source=replace(base.source, signal_scale=0.), session=replace(base.session, source_position=None)),
    }
    if len(configs) != 36:
        raise ValueError("Expected 36 main comparisons.")
    return suite, configs, supplemental


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("configs/sensing-v2/development.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a fresh output directory to preserve previous evidence.")
    suite, configs, supplemental = declared_configs(args.suite)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "declared-suite.json", {"suite":suite,"main":{k:v.to_dict() for k,v in configs.items()},
                                                    "supplementary":{k:v.to_dict() for k,v in supplemental.items()}})
    fixtures = ["coarse-full-startup-known-wind-threshold90_segment_v2-seed-0",
                "double-prior-startup-known-wind-threshold90_segment_v2-seed-0"]
    convergence = {"absolute_tolerance":1e-10,"relative_tolerance":.01,"maximum_refinements":2}
    checks = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(_convergence_job, name, configs[name].to_dict(), convergence, str(args.output)):name for name in fixtures}
        for future in as_completed(pending):
            result=future.result(); checks.append(result)
            print("Convergence", result["fixture"], result["status"], "level", result["accepted_level"], flush=True)
    if any(c["accepted_level"] is None for c in checks):
        write_json(args.output / "evaluation-freeze.json", {"status":"incomplete","checks":checks})
        artifact_manifest(args.output)
        raise ValueError("Numerical convergence incomplete; no accepted comparison table.")
    level = max(c["accepted_level"] for c in checks)
    grid = next(iter(configs.values())).evaluation
    grid = replace(grid, shape=tuple(n*2**level for n in grid.shape), step_seconds=grid.step_seconds/2**level)
    write_json(args.output / "evaluation-freeze.json", {"status":"converged","selected_grid":grid.to_dict(),"checks":checks,
        "scope":"Two representative new-policy development paths; not a universal convergence guarantee."})
    jobs = {**configs, **supplemental}
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(_run_job,name,c.to_dict(),str(args.output),grid.to_dict()):name for name,c in jobs.items()}
        for future in as_completed(pending):
            item=future.result()
            report=json.loads((args.output/item["report"]).read_text())
            item.update(role="main" if item["name"] in configs else "diagnostic",
                        successful_tracking_count=report["successful_tracking_count"], visited_fraction=report["visited_fraction"])
            results.append(item)
            write_json(args.output/"progress.json",{"completed":len(results),"total":len(jobs),"last":item})
            print(f"{len(results)}/{len(jobs)} {item['name']}: J/T={item['J_over_T']:.7f}, tracking={item['successful_tracking_count']}, time={item['total_scan_seconds']:.3f}",flush=True)
    write_json(args.output/"development-summary.json",{"status":"complete","claim_status":"not_demonstrated",
        "results":sorted(results,key=lambda r:r["name"]),"evaluation":grid.to_dict(),
        "limitations":["Descriptive development evidence; adaptive superiority not established.",
                       "Coarse uses full startup; doubled profile uses declared prior startup. Compare policies within each profile/case, not across those startup assumptions.",
                       "Common ceilings, different actual expenditure.","Synthetic reporting threshold is not physical calibration."]})
    artifact_manifest(args.output)


if __name__ == "__main__":
    main()
