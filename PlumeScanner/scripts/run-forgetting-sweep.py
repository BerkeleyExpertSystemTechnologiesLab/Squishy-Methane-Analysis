"""Log-odds forgetting sweep (v2 candidate) over the B1 case matrix.

For each (case, noise, log_odds_half_life) it runs the forgetting candidate
alone (the control/gated references are already frozen in the B1 study at
outputs/source-activity-study/) and reports the bidirectional activity metrics
and the mixture reconstruction loss. The truth field is shared with the B1
driver through the same truth_key, so no new truth is built.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parent / "src"))

from plume_focus.source_activity_forgetting import (
    SourceActivityForgettingRunConfig,
    SourceActivityForgettingSession,
)
from plume_focus.full_fov_execution import run_full_fov, decode_config
from plume_focus.source_activity_execution import activity_trace, activity_metrics, mixture_J, ground_truth_activity

import importlib.util
_driver_spec = importlib.util.spec_from_file_location("b1_driver", str(_here / "run-source-activity-study.py"))
_b1 = importlib.util.module_from_spec(_driver_spec)
_driver_spec.loader.exec_module(_b1)

STUDY = Path("configs/evidence-gated-focus/study.json")
B1_RUNS = Path("outputs/source-activity-study/runs")

POLICY_ID = "full_fov_source_activity_loop_v2_forgetting_candidate"


def build_forgetting_config(document, profile, case, seed, noise, tau):
    base = decode_config(_b1.read(document["reference_configs"][profile]))
    case_settings = document["cases"][case]
    source = base.source
    if seed:
        phases = np.random.default_rng(seed).uniform(0, 2 * np.pi, 3)
        from dataclasses import replace
        source = replace(source, lateral_phase=float(phases[0]),
                         amplitude_phase=float(phases[1]), shimmer_phase=float(phases[2]))
    from dataclasses import replace
    source = replace(source, signal_scale=float(case_settings.get("signal_scale", 1.0)))
    wind = base.wind
    if "wind_delta_yx" in case_settings:
        wind = replace(wind, velocity_yx=tuple(
            np.asarray(wind.velocity_yx) + np.asarray(case_settings["wind_delta_yx"])))
    cessation = None if case_settings.get("cessation_seconds") is None else float(case_settings["cessation_seconds"])
    windows = tuple(tuple(float(v) for v in w) for w in case_settings.get("active_windows", ())) or None
    settings = base.session.settings
    common = dict(source=source, wind=wind, duration_seconds=float(document["duration_seconds"]),
                  simulated_noise_std=float(noise), cessation_seconds=cessation, seed=int(seed),
                  delivery="stream", evaluation_shape=tuple(document["evaluation"]["shape"]),
                  evaluation_step_seconds=float(document["evaluation"]["step_seconds"]), active_windows=windows)
    session = SourceActivityForgettingSession(base.session.instrument, base.session.model, settings,
        base.session.source_position, base.session.home_yx,
        half_life_seconds=base.session.half_life_seconds, prior_active_probability=0.5,
        log_odds_half_life_seconds=tau)
    return SourceActivityForgettingRunConfig(session, **common)


def control_J(profile, case, seed, noise):
    path = B1_RUNS / f"{profile}-{case}-seed{seed}-noise{noise:g}.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    return float(row["policies"]["control"]["J_over_T"])


def run_case(document, profile, case, seed, noise, tau, truth):
    config = build_forgetting_config(document, profile, case, seed, noise, tau)
    record = run_full_fov(config, truth=truth)
    schedule = ground_truth_activity(config)
    duration = float(config.duration_seconds)
    trace = activity_trace(record)
    metrics = activity_metrics(trace, schedule, duration)
    ctl = control_J(profile, case, seed, noise)
    cand = float(mixture_J(record, truth))
    return {"profile": profile, "case": case, "seed": int(seed), "noise": float(noise),
            "log_odds_half_life_seconds": (None if tau is None else float(tau)),
            "policy_id": config.session.policy_id,
            "configuration": config.to_dict(), "configuration_hash": config.digest,
            "truth_key": _b1.truth_key(config),
            "activity_metrics": metrics,
            "J_control": ctl, "J_candidate_mixture": cand,
            "J_ratio": (cand / ctl if ctl else None)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", default="coarse")
    parser.add_argument("--cases", default="continuous,blank,cessation,intermittent,wind_error,weak")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noises", default="0")
    parser.add_argument("--taus", default="None,120,60,30,15,7.5")
    parser.add_argument("--root", type=Path, default=Path("outputs/forgetting-sweep"))
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--no-aggregate", action="store_true")
    args = parser.parse_args()

    document = _b1.read(STUDY)
    profiles = args.profiles.split(",")
    cases = args.cases.split(",")
    noises = [float(n) for n in args.noises.split(",")]
    taus = [None if t == "None" else float(t) for t in args.taus.split(",")]

    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    outdir = root / "runs"
    outdir.mkdir(parents=True, exist_ok=True)

    truth_root = Path("outputs/source-activity-study")  # shared B1 truth cache
    for profile in profiles:
        for case in cases:
            truth = _b1.load_or_build_truth(truth_root, build_forgetting_config(document, profile, case, args.seed, 0.0, None))
            for noise in noises:
                for tau in taus:
                    if args.shard is not None:
                        # Shard by the (case, noise, tau) triple, flattened.
                        flat = cases.index(case) * len(noises) * len(taus) + noises.index(noise) * len(taus) + taus.index(tau)
                        if flat % args.shards != args.shard:
                            continue
                    tag = "none" if tau is None else f"{tau:g}"
                    name = f"{profile}-{case}-seed{args.seed}-noise{noise:g}-tau{tag}"
                    path = outdir / f"{name}.json"
                    if path.exists():
                        continue
                    row = run_case(document, profile, case, args.seed, noise, tau, truth)
                    tmp = path.with_suffix(".partial")
                    tmp.write_text(json.dumps(row, indent=2), encoding="utf-8")
                    tmp.replace(path)
                    m = row["activity_metrics"]
                    print(f"{name}: stop={m.get('stop_latency_seconds')} resume={m.get('resume_latency_seconds')} "
                          f"final={m.get('final_p_active')} min={m.get('min_p_active')} max={m.get('max_p_active')} "
                          f"inact_mean={m.get('inactive_mean_p')} act_below={m.get('active_fraction_below_half')} "
                          f"J_ratio={row['J_ratio']}", flush=True)

    # Aggregate.
    if not args.no_aggregate:
        all_runs = sorted(outdir.glob("*.json"))
        rows = [json.loads(p.read_text(encoding="utf-8")) for p in all_runs]
        lines = ["# Forgetting sweep (v2)", "",
                 "| case | noise | tau | stop lat | resume lat | p_final | min p(active) | max p | inactive mean | active frac below 0.5 | J ratio |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            m = r["activity_metrics"]
            def f(v, d=3):
                return "—" if v is None else f"{v:.{d}f}"
            lines.append(f"| {r['case']} | {r['noise']:g} | {f(r['log_odds_half_life_seconds'])} | {f(m.get('stop_latency_seconds'))} | "
                         f"{f(m.get('resume_latency_seconds'))} | {f(m.get('final_p_active'))} | {f(m.get('min_p_active'))} | "
                         f"{f(m.get('max_p_active'))} | {f(m.get('inactive_mean_p'))} | {f(m.get('active_fraction_below_half'))} | {f(r['J_ratio'], 4)} |")
        (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (root / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(json.dumps({"records": len(rows), "root": str(root)}))


if __name__ == "__main__":
    main()
