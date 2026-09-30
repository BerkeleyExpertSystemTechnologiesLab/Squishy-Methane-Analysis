"""Fixed-evidence/path audit of frozen-v1 sensing failure mechanisms.

Synthetic truth is evaluator-only. These interventions are descriptive mechanism
tests, not hardware calibrations or replacement performance benchmarks.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from plume_focus.physical_scan import compile_profile_scan
from plume_focus.sensing_execution_v1 import SensingRunConfigV1, ControllerCallV1, CausalReplayV1
from plume_focus.sensing_policy_v1 import select_mass_prefix
from plume_focus.sensing_simulation import FakeScanner
from plume_focus.signal_belief import GaussianSignalBelief, grid_positions, positive_expectation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.trace.read_text())
    config = SensingRunConfigV1.from_dict(payload["configuration"])
    calls = tuple(ControllerCallV1.from_dict(e) for e in payload["events"])
    decisions = [e for e in calls if e.operation == "plan_next" and e.plan.samples]
    startup, next_decision = decisions[:2]
    pose = next_decision.position_yx
    now, horizon = next_decision.at_seconds, next_decision.request.max_duration_seconds
    readings = tuple(r for e in calls if e.operation == "ingest_readings" for r in e.readings if r.scan_id == startup.plan.scan_id)
    source = config.source.build()
    clean = FakeScanner(source, noise_std=0., seed=config.seed, gain=config.model.gain, offset=config.model.offset).execute(startup.plan).readings
    clean_by_id = {r.reading_id: r.value for r in clean}
    shape = config.capabilities.lattice_shape
    points = grid_positions(shape)
    weights = np.asarray(config.session.physical_quadrature_weights).reshape(shape)
    result = {"artifact_type": "fixed_path_mechanism_diagnosis_v1",
              "producing_revision": subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(),
              "source_trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(),
              "source_run_hash": payload["run_configuration_hash"],
              "decision_time": now, "forecast_horizon": horizon,
              "background_definition": "truth <= 1% of instantaneous scene peak, evaluator only",
              "interventions": []}
    variants = [
        ("recorded", config.model, 1., True),
        ("zero_injected_noise_same_assumed_R", config.model, 0., True),
        ("tenfold_lower_injected_and_assumed_noise", replace(config.model, sensor_noise_std=config.model.sensor_noise_std / 10), .1, True),
        ("zero_process_noise_same_evidence", replace(config.model, process_std_per_sqrt_second=0.), 1., True),
        ("quarter_prior_std_same_evidence", replace(config.model, initial_std=config.model.initial_std / 4), 1., True),
        ("no_source_input_same_evidence", config.model, 1., False),
    ]
    replay = CausalReplayV1(config.session, calls).advance_to(now)
    for label, model, noise_scale, inject_source in variants:
        belief = GaussianSignalBelief(model)
        declared_source = config.session.source_position if inject_source else None
        for reading in sorted(readings, key=lambda r: r.acquired_at_seconds):
            clean_value = clean_by_id[reading.reading_id]
            value = clean_value + noise_scale * (reading.value - clean_value)
            belief.assimilate(replace(reading, value=value), config.wind, declared_source)
        belief.advance(now, config.wind, declared_source)
        if label == "recorded":
            actual, _ = replay.predictive_moments(now, shape)
            np.testing.assert_allclose(actual, belief.moments_on(shape)[0], atol=1e-12, rtol=0)
        row = {"label": label, "model": model.to_dict(), "injected_noise_scale": noise_scale,
               "source_input_enabled": inject_source, "forecasts": []}
        for offset in (0., horizon):
            predicted = belief.clone()
            predicted.advance(now + offset, config.wind, declared_source)
            mean, variance = predicted.moments_on(shape)
            positive = positive_expectation(mean, variance, model.background_threshold)
            mean_positive = np.maximum(mean - model.background_threshold, 0)
            total = float(np.sum(weights * positive))
            truth = source.signal_at(np.full(len(points), now + offset), points).reshape(shape)
            bg = truth <= .01 * truth.max()
            requested = select_mass_prefix(weights * positive / total, .9, 4, 1e-10)
            plan = compile_profile_scan(requested, config.session.instrument, scan_id="diagnostic", started_at_seconds=now, current_position=pose)
            row["forecasts"].append({"at_seconds": now + offset,
                "expected_signal_total": total, "positive_mean_signal": float(np.sum(weights * mean_positive)),
                "uncertainty_excess_fraction": float(np.sum(weights * (positive - mean_positive)) / total),
                "background_cell_fraction": float(bg.mean()), "background_mass_fraction": float(np.sum(weights[bg] * positive[bg]) / total),
                "mean_map_squared_error": float(np.mean((mean - truth)**2)),
                "sigma_quantiles": np.quantile(np.sqrt(variance), [0, .5, 1]).tolist(),
                "requested_cell_fraction": float(requested.mean()), "column_closed_cell_fraction": len(plan.swept_cells) / requested.size,
                "compiled_seconds": plan.duration_seconds})
        result["interventions"].append(row)
        print(label, row["forecasts"][-1], flush=True)
    result["executed_geometry"] = []
    for decision in decisions:
        plan = decision.plan
        sweep_heights = [abs(s.end_yx[0] - s.start_yx[0]) for s in plan.segments if s.kind == "sweep"]
        result["executed_geometry"].append({"scan_id": plan.scan_id, "status": plan.status,
            "duration_seconds": plan.duration_seconds, "columns": len(sweep_heights),
            "mean_vertical_extent": float(np.mean(sweep_heights)), "max_vertical_extent": max(sweep_heights),
            "tracking_shortfall": plan.tracking_shortfall})
    result["interpretation_limits"] = [
        "Fixed startup path and readings: separates interventions before subsequent policy paths diverge.",
        "Uncertainty excess is an algebraic decomposition, not a causal attribution to measurement noise.",
        "Background includes model transport/source errors; 1% threshold is evaluator-only.",
        "Zero noise and removed source/process are diagnostics, not proposed calibrated operating settings."]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
