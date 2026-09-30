"""Filesystem/CLI adapter for the recovered single-detector demonstrator."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import time
import sys

from plume_focus.sensing_diagnostics import render_matched_diagnostics, measurement_replay
from plume_focus.sensing_evaluation import summarize_run
from plume_focus.sensing_rendering import RenderSettings, render_sensing_video
from plume_focus.sensing_simulation import CleanMethaneSource, SensingSimulationConfig, run_sensing_simulation


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _runtime():
    import numpy
    import cv2
    return {"python": platform.python_version(), "numpy": numpy.__version__, "opencv": cv2.__version__,
            "system": platform.system(), "processor": platform.processor(),
            "compute_timing_scope": "headless simulation plus evaluation, excluding video rendering"}


def run_development_suite(output_dir: Path, *, max_scans=5, duration=75., seeds=(0, 1)) -> dict:
    """Frozen development conditions, not a held-out advantage test."""
    cases = (("known-wind", False, (0., 0.), .005),
             ("wind-error", False, (.012, -.015), .02),
             ("source-free", True, (0., 0.), .005))
    manifest = {"role": "development", "seeds": list(seeds), "max_scans": max_scans, "duration_seconds": duration,
                "cases": [{"name": n, "source_free": f, "wind_bias_yx": b, "wind_error_speed": e} for n, f, b, e in cases],
                "policies": ["adaptive", "static_corridor", "rescan"],
                "comparison": "Common duration limit and per-scan ceilings; actual durations and scan counts can differ."}
    _write(output_dir / "development-manifest.json", manifest)
    records = []
    for name, source_free, bias, error in cases:
        for seed in seeds:
            for policy in manifest["policies"]:
                config = SensingSimulationConfig(max_scans=max_scans, duration_seconds=duration, seed=seed,
                                                 source_free=source_free, wind_bias_yx=bias, wind_error_speed=error, policy=policy)
                source = CleanMethaneSource()
                start = time.perf_counter()
                run = run_sensing_simulation(config, source=source)
                report = summarize_run(run, source)
                report["compute_seconds"] = time.perf_counter() - start
                report["runtime"] = _runtime()
                report["source_configuration"] = asdict(source.config)
                report["case"] = name
                filename = f"{name}-{policy}-seed-{seed}.json"
                _write(output_dir / filename, report)
                records.append({"case": name, "policy": policy, "seed": seed,
                                "mean_mass_recall": report["mean_mass_recall"],
                                "mean_interval_mass_recall": report["mean_interval_mass_recall"],
                                "mean_tracking_mass_recall": report["mean_tracking_mass_recall"],
                                "total_scan_seconds": report["total_scan_seconds"],
                                "scans_completed": report["scans_completed"], "report": filename})
                print(f"{name} / {policy} / seed {seed}: recall={report['mean_mass_recall']:.3f}, seconds={report['total_scan_seconds']:.3f}", flush=True)
    result = {"claim_status": "not_demonstrated", "manifest": manifest, "results": records,
              "limitations": ["Descriptive development comparison; no inferential advantage claim.",
                              "Policies may use different actual time and finish different numbers of scans.",
                              "The source-free static rectangle is predeclared, not supplied the true source."]}
    _write(output_dir / "development-summary.json", result)
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sensing-recovery"))
    parser.add_argument("--max-scans", type=int, default=5)
    parser.add_argument("--duration", type=float, default=75.)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", choices=("adaptive", "static_corridor", "rescan"), default="adaptive")
    parser.add_argument("--source-free", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--fps", type=float, default=12.)
    parser.add_argument("--playback-speed", type=float, default=1.)
    configuration = parser.add_mutually_exclusive_group()
    configuration.add_argument("--run-config", type=Path)
    configuration.add_argument("--suite-config", type=Path)
    parser.add_argument("--workers", type=int, default=2, help="V1 suite worker processes; does not change physical settings")
    args = parser.parse_args(argv)
    if args.run_config or args.suite_config:
        arguments = sys.argv[1:] if argv is None else argv
        overrides = {"--max-scans", "--duration", "--seed", "--policy", "--source-free", "--suite"}
        if any(token.split("=")[0] in overrides for token in arguments):
            parser.error("Explicit v1 configuration cannot be combined with legacy physical/policy overrides.")
        from plume_focus.sensing_cli_v1 import run_config_file, run_suite_file
        if args.suite_config:
            run_suite_file(args.suite_config, args.output_dir, workers=args.workers, render=args.render,
                           fps=args.fps, playback_speed=args.playback_speed)
        else:
            run_config_file(args.run_config, args.output_dir, render=args.render, fps=args.fps, playback_speed=args.playback_speed)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.suite:
        run_development_suite(args.output_dir, max_scans=args.max_scans, duration=args.duration)
    else:
        source = CleanMethaneSource()
        config = SensingSimulationConfig(max_scans=args.max_scans, duration_seconds=args.duration, seed=args.seed,
                                         policy=args.policy, source_free=args.source_free)
        start = time.perf_counter()
        run = run_sensing_simulation(config, source=source)
        report = summarize_run(run, source)
        report["compute_seconds"] = time.perf_counter() - start
        report["runtime"] = _runtime()
        report["source_configuration"] = asdict(source.config)
        _write(args.output_dir / "report.json", report)
        _write(args.output_dir / "trace.json", run.operational_payload())
        replay, _ = measurement_replay(run, source)
        _write(args.output_dir / "measurement-replay.json", replay)
        if args.render:
            videos = []
            for mode in ("physical", "summary"):
                videos.append(render_sensing_video(run, args.output_dir / f"{mode}.mp4", source=source,
                              settings=RenderSettings(mode=mode, fps=args.fps, playback_speed=args.playback_speed)))
            matched = render_matched_diagnostics(run, args.output_dir, source)
            _write(args.output_dir / "video-manifest.json", videos + matched["videos"])
        print(json.dumps({"scans": len(run.scans), "mean_mass_recall": report["mean_mass_recall"],
                          "total_scan_seconds": report["total_scan_seconds"], "operational_hash": run.operational_hash}))
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=False).stdout.strip())
    artifacts = []
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file() and path.name != "artifact-manifest.json":
            artifacts.append({"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
    _write(args.output_dir / "artifact-manifest.json", {"producing_revision": revision, "working_tree_dirty": dirty, "artifacts": artifacts})


if __name__ == "__main__":
    main()
