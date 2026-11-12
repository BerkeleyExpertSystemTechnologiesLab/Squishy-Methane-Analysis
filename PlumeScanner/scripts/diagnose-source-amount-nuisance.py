"""Run the frozen fixed-path source amount/wind/decay diagnostic locally."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess

from plume_focus.full_fov_execution import decode_config, run_full_fov, verify_replay
from plume_focus.source_amount_nuisance import (
    NuisanceEnvironment, dynamics_pair, refinement_checks, run_level,
    save_json, select_histories, sha256, reading_schedule_sha256, ARCHIVED_SCHEDULE_SHA256,
)


def condition_run(condition, cfg, root):
    run = decode_config(json.loads(Path(cfg["baseline_config"]).read_text()))
    model = replace(run.model, grid_shape=tuple(cfg["grid_shape"]))
    nominal, truth = dynamics_pair(model, run.wind.velocity_yx, run.session.half_life_seconds,
                                   cfg["clock_tick_seconds"], condition)
    out = Path(root) / condition["name"]
    out.mkdir()
    save_json(out / "condition.json", condition)
    record = json.loads(Path(cfg["record"]).read_text())
    env = NuisanceEnvironment(truth, run, cfg, record, out / "environment")
    results, comparisons = [], []
    for level in cfg["levels"]:
        results.append(run_level(nominal, truth, env, run, cfg, level,
                                 out / f"source-{level[0]}-amount-{level[1]}"))
    check = refinement_checks(cfg, results[-2], results[-1])
    comparisons.append(dict(coarse=results[-2]["level"], fine=results[-1]["level"], **check))
    save_json(out / "refinement.json", comparisons)
    for level in cfg["fallback_levels"]:
        unresolved = sorted({r["history"] for r in check["checks"] if not r["passed"]})
        if not unresolved:
            break
        refined = run_level(nominal, truth, env, run, cfg, level,
                            out / f"source-{level[0]}-amount-{level[1]}", indices=unresolved)
        check = refinement_checks(cfg, results[-1], refined)
        comparisons.append(dict(coarse=results[-1]["level"], fine=refined["level"], **check))
        results.append(refined)
        save_json(out / "refinement.json", comparisons)
    selected = select_histories(results)
    unresolved = sorted({r["history"] for r in check["checks"] if not r["passed"]})
    # Earlier histories passed their last comparison; later comparisons contain only failures.
    verification = dict(condition=condition, histories=len(selected),
                        unresolved_histories=unresolved, all_numerical_checks_passed=not unresolved,
                        record_unchanged=sha256(cfg["record"]) == cfg["record_sha256"],
                        fixed_evidence_shared=all(r["evidence_sha256"] == results[0]["evidence_sha256"] for r in results),
                        levels=[r["level"] for r in results])
    save_json(out / "selected-results.json", selected)
    save_json(out / "verification.json", verification)
    manifest = {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}
    save_json(out / "sha256.json", manifest)
    print(json.dumps(dict(condition_complete=condition["name"], verification=verification)), flush=True)
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fresh-schedule", action="store_true",
                        help="Regenerate and verify the archived physical schedule without local archive files")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if cfg["true_amount"] != 1 or cfg["amount_support"] != [.25, 1.75]:
        raise ValueError("This named experiment fixes Q0 and the inherited amount support")
    if args.jobs < 1:
        raise ValueError("Positive worker count required")
    if args.smoke:
        cfg["grid_shape"] = [4, 6]
        cfg["levels"] = [[17, 16], [33, 32]]
        cfg["fallback_levels"] = [[65, 64]]
        cfg["conditions"] = [c for c in cfg["conditions"] if c["name"] in ("matched", "wind_150", "half_life_050")]
        cfg["scenes"] = cfg["scenes"][:2]
        for scene in cfg["scenes"]:
            scene["seeds"] = scene["seeds"][:2]
    if not args.fresh_schedule and sha256(cfg["record"]) != cfg["record_sha256"]:
        raise ValueError("Archived schedule hash mismatch")
    args.output.mkdir(parents=True)
    if list(args.output.iterdir()):
        raise FileExistsError("Use a new empty output destination")
    if args.fresh_schedule:
        record = run_full_fov(decode_config(json.loads(Path(cfg["baseline_config"]).read_text())))
        verify_replay(record)
        physical_hash = reading_schedule_sha256(record)
        if physical_hash != ARCHIVED_SCHEDULE_SHA256:
            raise ValueError("Regenerated acquisition schedule differs from the frozen 708-reading contract")
        fresh_record = args.output / "fresh-schedule.json"
        save_json(fresh_record, record)
        cfg["archived_record_sha256"] = cfg["record_sha256"]
        cfg["record"] = str(fresh_record)
        cfg["record_sha256"] = sha256(fresh_record)
        cfg["schedule_origin"] = "regenerated baseline, physical acquisition contract verified"
        cfg["physical_schedule_sha256"] = physical_hash
    save_json(args.output / "config.json", cfg)
    producers = [args.config, Path(__file__), Path("src/plume_focus/source_amount_nuisance.py"),
                 Path("src/plume_focus/source_amount_validation.py"), Path("src/plume_focus/source_input.py"),
                 Path("docs/specs/source-amount-nuisance-v1.md"), Path(cfg["baseline_config"]), Path(cfg["record"])]
    for producer in producers:
        destination = args.output / producer.name
        if producer.resolve() != destination.resolve():
            shutil.copy2(producer, destination)
    save_json(args.output / "provenance.json", dict(
        head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        branch=subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        status=subprocess.check_output(["git", "status", "--short"], text=True),
        smoke_only=args.smoke, producer_hashes={str(p): sha256(p) for p in producers}))
    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(condition_run, c, cfg, str(args.output)) for c in cfg["conditions"]]
        for future in as_completed(futures):
            results.append(future.result())
            save_json(args.output / "completed-conditions.json", results)
    results.sort(key=lambda r: [c["name"] for c in cfg["conditions"]].index(r["condition"]["name"]))
    save_json(args.output / "verification.json", dict(
        smoke_only=args.smoke, conditions=results,
        all_numerical_checks_passed=all(r["all_numerical_checks_passed"] for r in results),
        record_unchanged=sha256(cfg["record"]) == cfg["record_sha256"]))


if __name__ == "__main__":
    main()
