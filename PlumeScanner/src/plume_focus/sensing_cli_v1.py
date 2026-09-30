"""File/configuration adapter for explicit frozen-v1 development experiments."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

from plume_focus.sensing_contracts import CausalWind
from plume_focus.sensing_execution_v1 import ReconstructionGridV1, SensingRunConfigV1, run_sensing_v1
from plume_focus.sensing_execution_v2 import SensingRunConfigV2
from plume_focus.sensing_execution_v3 import SensingRunConfigV3
from plume_focus.sensing_execution_v4 import SensingRunConfigV4, SensingMotionRunConfigV4
from plume_focus.sensing_reconstruction import reconstruction_loss, summarize_v1
from plume_focus.sensing_rendering import RenderSettings, render_sensing_video
from plume_focus.sensing_session import configuration_hash


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")


def load_config(path: Path) -> SensingRunConfigV1:
    return config_from_dict(json.loads(path.read_text(encoding="utf-8")))


def config_from_dict(payload):
    kind = payload.get("message_type")
    if kind=="TransportBaselineRunConfigV2":
        from plume_focus.sensing_execution_v5 import TransportBaselineRunConfigV2
        return TransportBaselineRunConfigV2.from_dict(payload)
    if kind in ("SensingRunConfigV5","SensingMotionRunConfigV5"):
        from plume_focus.sensing_execution_v5 import SensingRunConfigV5, SensingMotionRunConfigV5
        return (SensingRunConfigV5 if kind=="SensingRunConfigV5" else SensingMotionRunConfigV5).from_dict(payload)
    if kind in ("TransportDiagnosticRunConfig", "MotionTransportDiagnosticRunConfig"):
        from plume_focus.sensing_transport_diagnostic import TransportDiagnosticRunConfig, MotionTransportDiagnosticRunConfig
        cls = TransportDiagnosticRunConfig if kind == "TransportDiagnosticRunConfig" else MotionTransportDiagnosticRunConfig
        return cls.from_dict(payload)
    if kind == "SensingRunConfigV4":
        return SensingRunConfigV4.from_dict(payload)
    if kind == "SensingMotionRunConfigV4":
        return SensingMotionRunConfigV4.from_dict(payload)
    if kind == "SensingRunConfigV3":
        return SensingRunConfigV3.from_dict(payload)
    if kind == "SensingRunConfigV2":
        return SensingRunConfigV2.from_dict(payload)
    return SensingRunConfigV1.from_dict(payload)


def artifact_manifest(output: Path):
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True).stdout.strip())
    derived = {"artifact-manifest.json", "inspection-manifest.json", "visual-review.json"}
    artifacts = [{"path": str(p.relative_to(output)).replace("\\", "/"), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                  "bytes": p.stat().st_size} for p in sorted(output.rglob("*"))
                 if p.is_file() and p.name not in derived and "inspection-frames" not in p.relative_to(output).parts]
    import numpy
    import cv2
    import scipy
    write_json(output / "artifact-manifest.json", {"producing_revision": revision, "tracked_worktree_dirty": dirty,
               "runtime": {"python": platform.python_version(), "numpy": numpy.__version__, "opencv": cv2.__version__, "scipy": scipy.__version__,
                           "system": platform.system(), "processor": platform.processor()}, "artifacts": artifacts})


def _run_job(name: str, payload: dict, output: str, grid_payload: dict | None = None):
    producing_revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    config = config_from_dict(payload)
    if grid_payload is not None:
        config = replace(config, evaluation=ReconstructionGridV1.from_dict(grid_payload))
    directory = Path(output) / name
    start = time.perf_counter()
    run = run_sensing_v1(config)
    simulated = time.perf_counter()
    loss = reconstruction_loss(run, config.source.build())
    evaluated = time.perf_counter()
    report = summarize_v1(run, config.source.build(), loss=loss)
    report["producing_revision"] = producing_revision
    report["compute_seconds"] = {"simulation": simulated - start, "causal_replay_evaluation": evaluated - simulated}
    write_json(directory / "configuration.json", config.to_dict())
    write_json(directory / "trace.json", run.operational_payload())
    write_json(directory / "report.json", report)
    return {"name": name, "profile": config.session.instrument.profile_id, "policy": config.policy, "seed": config.seed,
            "J": loss["J"], "J_over_T": loss["J_over_T"], "reconstruction_status": loss["status"],
            "scans_completed": len(run.scans), "startup_complete": run.startup_complete,
            "reacquisition_count": report["reacquisition_count"], "tracking_shortfall_count": report["tracking_shortfall_count"],
            "total_scan_seconds": report["total_scan_seconds"], "mean_mass_recall": report["mean_mass_recall"],
            "operational_hash": run.operational_hash, "report": f"{name}/report.json"}


def _convergence_job(name, payload, convergence, output):
    config = config_from_dict(payload)
    run = run_sensing_v1(config)
    grid = config.evaluation
    previous = reconstruction_loss(run, config.source.build(), grid=grid, distribution_diagnostics=False)
    records = []
    accepted_level = None
    for level in range(convergence["maximum_refinements"]):
        refined = replace(grid, shape=tuple(n * 2 for n in grid.shape), step_seconds=grid.step_seconds / 2)
        current = reconstruction_loss(run, config.source.build(), grid=refined, distribution_diagnostics=False)
        difference = abs(previous["J"] - current["J"]) if previous["J"] is not None and current["J"] is not None else None
        tolerance = max(convergence["absolute_tolerance"], convergence["relative_tolerance"] * abs(current["J"])) if current["J"] is not None else None
        passed = difference is not None and difference <= tolerance
        records.append({"coarse_grid": grid.to_dict(), "fine_grid": refined.to_dict(), "coarse_J": previous["J"],
                        "fine_J": current["J"], "absolute_difference": difference, "tolerance": tolerance, "passed": passed})
        if passed:
            # The coarser mesh is accepted only after its agreement with refinement
            # is checked. This preserves the predeclared initial mesh when adequate.
            accepted_level = level
            break
        grid, previous = refined, current
    result = {"fixture": name, "accepted_level": accepted_level, "status": "converged" if accepted_level is not None else "incomplete",
              "checks": records, "operational_hash": run.operational_hash}
    write_json(Path(output) / "convergence" / f"{name}.json", result)
    return result


def suite_configs(path: Path):
    suite = json.loads(path.read_text(encoding="utf-8"))
    expected = {"schema_version", "artifact_type", "profiles", "seeds", "policies", "cases", "source_free_corridor_yxyx", "convergence"}
    if set(suite) != expected or suite["schema_version"] != 1 or suite["artifact_type"] != "sensing_development_suite_v1":
        raise ValueError("Unsupported or incomplete development suite manifest.")
    configs = {}
    for profile in suite["profiles"]:
        base = load_config(path.parent / profile)
        prefix = Path(profile).stem
        for case in suite["cases"]:
            for policy in suite["policies"]:
                for seed in suite["seeds"]:
                    session = replace(base.session, source_position=None if case["source_free"] else base.session.source_position,
                                      policy=replace(base.session.policy, policy_id=policy,
                                                     corridor_yxyx=tuple(suite["source_free_corridor_yxyx"]) if case["source_free"] else base.session.policy.corridor_yxyx))
                    wind = CausalWind(velocity_yx=tuple(a + b for a, b in zip(base.wind.velocity_yx, case["wind_bias_yx"])),
                                      error_speed=case["wind_error_speed"])
                    config = replace(base, session=session, wind=wind, seed=seed)
                    name = f"{prefix}-{case['name']}-{policy}-seed-{seed}"
                    configs[name] = config
    if len(configs) != 36 or any(name not in configs for name in suite["convergence"]["fixtures"]):
        raise ValueError("The frozen development suite requires 36 unique runs and valid convergence fixtures.")
    return suite, configs


def _render(config, directory, fps, playback_speed):
    run = run_sensing_v1(config)
    manifest = [render_sensing_video(run, directory / f"{mode}.mp4", source=config.source.build(),
                                    settings=RenderSettings(mode=mode, fps=fps, playback_speed=playback_speed))
                for mode in ("physical", "summary")]
    write_json(directory / "video-manifest.json", manifest)
    return manifest


def run_config_file(path, output, *, render=False, fps=12., playback_speed=1.):
    config = load_config(Path(path))
    result = _run_job("run", config.to_dict(), str(output))
    if render:
        _render(config, Path(output) / "run", fps, playback_speed)
    artifact_manifest(Path(output))
    print(json.dumps(result), flush=True)


def run_suite_file(path, output, *, workers=2, render=False, fps=12., playback_speed=1.):
    suite, configs = suite_configs(Path(path))
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be positive.")
    output = Path(output)
    write_json(output / "declared-suite.json", suite)
    checks, results = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_convergence_job, name, configs[name].to_dict(), suite["convergence"], str(output)): name
                   for name in suite["convergence"]["fixtures"]}
        for future in as_completed(futures):
            result = future.result()
            checks.append(result)
            print(f"Quadrature {result['fixture']}: {result['status']}; level {result['accepted_level']}", flush=True)
        complete = all(c["accepted_level"] is not None for c in checks)
        level = max(c["accepted_level"] if c["accepted_level"] is not None else suite["convergence"]["maximum_refinements"] for c in checks)
        base_grid = next(iter(configs.values())).evaluation
        grid = replace(base_grid, shape=tuple(n * 2**level for n in base_grid.shape), step_seconds=base_grid.step_seconds / 2**level)
        write_json(output / "evaluation-freeze.json", {"status": "converged" if complete else "incomplete",
                   "selected_grid": grid.to_dict(), "evaluation_hash": configuration_hash(grid.to_dict()),
                   "checks": sorted(checks, key=lambda c: c["fixture"]), "selection": "Common coarsest tested mesh accepted by all declared fixtures"})
        jobs = {pool.submit(_run_job, name, c.to_dict(), str(output), grid.to_dict()): name for name, c in configs.items()}
        for future in as_completed(jobs):
            result = future.result()
            results.append(result)
            print(f"{len(results)}/36 {result['name']}: J={result['J']:.8g}; charged={result['total_scan_seconds']:.3f}s; startup={result['startup_complete']}", flush=True)
            write_json(output / "progress.json", sorted(results, key=lambda r: r["name"]))
    summary = {"claim_status": "not_demonstrated", "role": "descriptive_development", "quadrature_status": "converged" if complete else "incomplete",
               "common_evaluation": grid.to_dict(), "suite_manifest_hash": configuration_hash(suite),
               "comparison": "Common instrument/model/startup and time ceilings within each profile/case; actual expenditure differs.",
               "results": sorted(results, key=lambda r: r["name"]),
               "limitations": ["Two noise seeds do not support an inferential superiority claim.",
                               "Narrow full-scene startup exceeds the monitoring horizon.",
                               "Uncalibrated synthetic retrieval and persistent Gaussian transport remain approximations."]}
    write_json(output / "development-summary.json", summary)
    if render:
        for prefix in ("coarse", "narrow"):
            name = f"{prefix}-known-wind-mass90_column_v1-seed-0"
            _render(replace(configs[name], evaluation=grid), output / name, fps, playback_speed)
            print(f"Rendered physical/summary views: {prefix}", flush=True)
        delayed = replace(configs["coarse-known-wind-mass90_column_v1-seed-0"], delivery="batch",
                          delivery_delay_seconds=.5, completion_overrun_seconds=.25, evaluation=grid)
        _run_job("delayed-inspection", delayed.to_dict(), str(output))
        _render(delayed, output / "delayed-inspection", fps, playback_speed)
    artifact_manifest(output)
    return summary
