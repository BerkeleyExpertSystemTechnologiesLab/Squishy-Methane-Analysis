"""Source-activity comparison matrix driver (headless, evaluator-side).

Reuses the frozen study.json scenarios and reference configs. For each
(profile, case, seed, noise) it runs three scanners on one shared truth field --
the calibrated control (single decay belief), the source-activity candidate
(two-model belief, same full-FoV policy), and the evidence-gated reference --
then reports reconstruction loss and the candidate's activity-inference
trajectory against the ground-truth schedule.

The loss truth uses the exact separable `clean_grid` evaluation (fast, grid-only,
same as the evidence-gated study); readings still use `signal_at`. Outputs land
under outputs/source-activity-study/ (gitignored).
"""

import argparse
import json
import os
import sys
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parent / "src"))

from calibration_truth_grid import clean_grid

from plume_focus.full_fov_execution import run_full_fov, decode_config
from plume_focus.calibration_execution import CalibrationRunConfig
from plume_focus.evidence_gated_execution import EvidenceGatedRunConfig
from plume_focus.evidence_gated_loop import EvidenceGatedSession, EvidenceGateSettings
from plume_focus.calibrated_scanner import CalibrationSession, CalibrationSettings
from plume_focus.source_activity import SourceActivitySession
from plume_focus.source_activity_execution import (
    SourceActivityRunConfig, activity_trace, activity_metrics, mixture_J, ground_truth_activity,
)
from plume_focus.sensing_risk import projection_gram
from plume_focus.sensing_session import configuration_hash
from plume_focus.signal_belief import grid_positions, interpolation_stencil

STUDY = Path("configs/evidence-gated-focus/study.json")

POLICIES = ("control", "source_activity", "gated")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_config(document, profile, case, seed, noise, policy):
    base = decode_config(read(document["reference_configs"][profile]))
    case_settings = document["cases"][case]
    source = base.source
    if seed:
        phases = np.random.default_rng(seed).uniform(0, 2 * np.pi, 3)
        source = replace(source, lateral_phase=float(phases[0]),
                         amplitude_phase=float(phases[1]), shimmer_phase=float(phases[2]))
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
    if policy == "source_activity":
        session = SourceActivitySession(base.session.instrument, base.session.model, settings,
            base.session.source_position, base.session.home_yx,
            half_life_seconds=base.session.half_life_seconds, prior_active_probability=0.5)
        return SourceActivityRunConfig(session, **common)
    if policy == "gated":
        gate = document["gate"]
        candidate_settings = EvidenceGateSettings(
            **{f.name: getattr(settings, f.name) for f in fields(CalibrationSettings)},
            alpha=float(gate["alpha"]), entry_level=float(gate["entry_level"]),
            retain_level=float(gate["retain_level"]))
        session = EvidenceGatedSession(base.session.instrument, base.session.model, candidate_settings,
            base.session.source_position, base.session.home_yx,
            half_life_seconds=base.session.half_life_seconds)
        return EvidenceGatedRunConfig(session, **common)
    session = CalibrationSession(base.session.instrument, base.session.model, settings,
        base.session.source_position, base.session.home_yx,
        half_life_seconds=base.session.half_life_seconds)
    return CalibrationRunConfig(session, **common)


def build_truth(config):
    """Exact separable loss quadrature (same sufficient statistics as OnlineLoss)."""
    shape = config.evaluation_shape
    latent = config.model.grid_shape
    times = np.arange(int(np.ceil(config.duration_seconds / config.evaluation_step_seconds))) * config.evaluation_step_seconds
    durations = np.diff(np.append(times, config.duration_seconds))
    points = grid_positions(shape)
    ids, weights = interpolation_stencil(points, latent)
    gram = projection_gram(latent, shape)
    mean_operator = np.zeros(int(np.prod(latent)))
    np.add.at(mean_operator, ids.ravel(), weights.ravel() / len(points))
    cross, square, average = [], [], []
    for time in times:
        values = clean_grid(config, float(time), shape).ravel()
        projected = np.zeros(int(np.prod(latent)))
        np.add.at(projected, ids.ravel(), (weights * values[:, None]).ravel() / len(points))
        cross.append(projected)
        square.append(float(np.mean(values ** 2)))
        average.append(float(np.mean(values)))
    return SimpleNamespace(times=times, durations=durations, gram=gram, cross=np.array(cross),
                           square=np.array(square), average=np.array(average), mean_operator=mean_operator)


def truth_key(config):
    """Policy/estimator/noise-independent identity of the common truth field."""
    return configuration_hash({"source": config.source.to_dict(), "cessation": config.cessation_seconds,
        "windows": config.active_windows, "shape": config.evaluation_shape,
        "latent": config.model.grid_shape, "lattice": config.session.instrument.lattice_shape,
        "step": config.evaluation_step_seconds, "duration": config.duration_seconds})


def load_or_build_truth(root, config):
    """Disk-cached exact truth; one field is shared across policies and noises."""
    key = truth_key(config)
    path = Path(root) / "truth-cache" / f"{key}.npz"
    if path.exists():
        with np.load(path) as z:
            return SimpleNamespace(times=z["times"], durations=z["durations"], gram=z["gram"],
                                   cross=z["cross"], square=z["square"], average=z["average"],
                                   mean_operator=z["mean_operator"])
    q = build_truth(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + f"-{os.getpid()}.partial.npz")
    np.savez_compressed(temporary, times=q.times, durations=q.durations, gram=q.gram, cross=q.cross,
                        square=q.square, average=q.average, mean_operator=q.mean_operator)
    temporary.replace(path)
    return q


def run_one(document, profile, case, seed, noise, truth):
    configs = {policy: build_config(document, profile, case, seed, noise, policy) for policy in POLICIES}
    schedule = ground_truth_activity(configs["control"])
    duration = float(configs["control"].duration_seconds)
    out = {"profile": profile, "case": case, "seed": int(seed), "noise": float(noise),
           "schedule": schedule, "duration_seconds": duration,
           "truth_key": truth_key(configs["control"]), "policies": {}}
    for policy, config in configs.items():
        record = run_full_fov(config, truth=truth)
        entry = {"configuration": config.to_dict(), "configuration_hash": config.digest,
                 "policy_id": config.session.policy_id,
                 "J_over_T": float(record["J_over_T"]), "termination": record["termination"],
                 "scans": len(record["scans"]),
                 "completed_full_scans": record["completed_full_scans"],
                 "spent_seconds": float(record["spent_seconds"]),
                 "offline_seconds": float(record["offline_seconds"])}
        if policy == "source_activity":
            trace = activity_trace(record)
            entry["activity_trace"] = trace
            entry["activity_metrics"] = activity_metrics(trace, schedule, duration)
            entry["mixture_J_over_T"] = float(mixture_J(record, truth))
        out["policies"][policy] = entry
    return out


def criteria(row):
    """Predeclared criteria for one source-activity run (see 对照矩阵_实验记录.md §4)."""
    case = row["case"]
    m = row["policies"]["source_activity"]["activity_metrics"]
    checks = {}
    if case == "continuous":
        checks["continuous_p_active_high"] = m["final_p_active"] is not None and m["final_p_active"] >= 0.9
    elif case == "cessation":
        checks["cessation_drops"] = m["final_p_active"] is not None and m["final_p_active"] <= 0.1
        checks["cessation_latency_seconds"] = m.get("stop_latency_seconds")
    elif case == "intermittent":
        checks["intermittent_recovers"] = m["final_p_active"] is not None and m["final_p_active"] >= 0.9
        checks["stop_latency_seconds"] = m.get("stop_latency_seconds")
        checks["resume_latency_seconds"] = m.get("resume_latency_seconds")
    elif case == "blank":
        checks["blank_never_high"] = m["max_p_active"] is not None and m["max_p_active"] <= 0.5
        checks["blank_drops"] = m["final_p_active"] is not None and m["final_p_active"] <= 0.1
    elif case == "weak":
        checks["weak_keeps_high"] = m["final_p_active"] is not None and m["final_p_active"] >= 0.5
    elif case == "wind_error":
        checks["wind_error_keeps_high"] = m["final_p_active"] is not None and m["final_p_active"] >= 0.5
    # Reconstruction preservation vs control.
    ctl = row["policies"]["control"]["J_over_T"]
    cand = row["policies"]["source_activity"]["mixture_J_over_T"]
    if case == "continuous":
        checks["continuous_J_preserved"] = (cand - ctl) / ctl <= 0.05 if ctl else None
    else:
        checks["J_no_worse"] = cand <= ctl + 1e-12
    return checks


def summarize(results):
    lines = ["# Source-activity comparison matrix", "",
             "| profile | case | seed | noise | J/T control | J/T candidate(mix) | J/T gated | p_final | "
             "active mean p | inactive mean p | stop lat | resume lat | max p |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in results:
        p = row["policies"]
        m = p["source_activity"]["activity_metrics"]

        def fmt(v, digits=3):
            return "—" if v is None else f"{v:.{digits}f}"
        lines.append(f"| {row['profile']} | {row['case']} | {row['seed']} | {row['noise']:g} | "
                     f"{p['control']['J_over_T']:.6f} | {p['source_activity']['mixture_J_over_T']:.6f} | "
                     f"{p['gated']['J_over_T']:.6f} | {fmt(m['final_p_active'])} | "
                     f"{fmt(m['active_mean_p'])} | {fmt(m['inactive_mean_p'])} | {fmt(m.get('stop_latency_seconds'))} | "
                     f"{fmt(m.get('resume_latency_seconds'))} | {fmt(m['max_p_active'])} |")
    lines += ["", "## Criteria", ""]
    for row in results:
        lines.append(f"### {row['profile']} / {row['case']} / seed{row['seed']} / noise{row['noise']:g}")
        for name, value in criteria(row).items():
            lines.append(f"- {name}: {value}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", default="coarse")
    parser.add_argument("--cases", default="continuous,blank,cessation,intermittent,wind_error,weak")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noises", default="0,0.05")
    parser.add_argument("--root", type=Path, default=Path("outputs/source-activity-study"))
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()

    document = read(STUDY)
    profiles = args.profiles.split(",")
    cases = args.cases.split(",")
    if args.shard is not None:
        cases = [case for index, case in enumerate(cases) if index % args.shards == args.shard]
    noises = [float(n) for n in args.noises.split(",")]
    root = args.root
    root.mkdir(parents=True, exist_ok=True)

    results = []
    for profile in profiles:
        for case in cases:
            # Truth depends only on source/cessation/windows (seed-fixed here); cached on disk.
            truth = load_or_build_truth(root, build_config(document, profile, case, args.seed, 0.0, "control"))
            for noise in noises:
                name = f"{profile}-{case}-seed{args.seed}-noise{noise:g}"
                path = root / "runs" / f"{name}.json"
                if path.exists():
                    row = read(path)
                else:
                    row = run_one(document, profile, case, args.seed, noise, truth)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(".partial")
                    temporary.write_text(json.dumps(row, indent=2), encoding="utf-8")
                    temporary.replace(path)
                results.append(row)
                p = row["policies"]
                m = p["source_activity"]["activity_metrics"]
                print(f"{name}: J ctl={p['control']['J_over_T']:.4f} "
                      f"cand(mix)={p['source_activity']['mixture_J_over_T']:.4f} "
                      f"p_final={m['final_p_active']}", flush=True)

    # Aggregate from every completed run record (shards share this directory).
    all_runs = sorted((root / "runs").glob("*.json"))
    results = [read(path) for path in all_runs]
    (root / "summary.md").write_text(summarize(results), encoding="utf-8")
    (root / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(results), "root": str(root)}))


if __name__ == "__main__":
    main()
