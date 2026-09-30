"""Read-only evidence replay and controlled transport diagnostics; I/O adapter."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from plume_focus.motion_belief import PersistentWindBelief
from plume_focus.motion_config import PersistentMotionConfig
from plume_focus.sensing_cli_v1 import config_from_dict, write_json
from plume_focus.sensing_execution_v1 import ControllerCallV1, CausalReplayV1
from plume_focus.sensing_execution_v3 import with_persistent_motion
from plume_focus.sensing_policy_v2 import plan_scan_v2, survey_blocks
from plume_focus.sensing_contracts import CausalWind
from plume_focus.signal_belief import GaussianSignalBelief, SignalModelConfig, grid_positions
from plume_focus.signal_distribution import PredictiveDistribution


def distribution(belief, shape):
    if isinstance(belief, PersistentWindBelief):
        return belief.distribution_on(shape)
    mu, var = belief.moments_on(shape)
    return PredictiveDistribution(belief.time_seconds, np.ones(1), mu[None], var[None])


def spatial_summary(d, threshold=.15):
    density = d.expected_excess(threshold)
    total = float(density.mean())
    coordinates = grid_positions(density.shape)
    centroid = covariance = None
    if density.sum() > 1e-12:
        p = density.ravel() / density.sum()
        center = p @ coordinates
        delta = coordinates - center
        centroid = center.tolist(); covariance = ((delta.T * p) @ delta).tolist()
    return {"expected_excess": total, "centroid_yx": centroid, "spatial_covariance_yx": covariance,
            "within_variance": float(d.within_variance.mean()), "between_variance": float(d.between_variance.mean())}


def controlled_forecasts(order):
    model = SignalModelConfig(grid_shape=(12, 16), source_rate=0, diffusivity=0,
                              process_std_per_sqrt_second=0)
    points = grid_positions(model.grid_shape)
    cases = {"zero": (((0., 0.), (0., 0.)), (.5, .45), (0., .02)),
             "vertical": (((.0001, 0.), (0., 0.)), (.5, .45), (.02, 0.)),
             "horizontal": (((0., 0.), (0., .0001)), (.5, .45), (0., .02)),
             "correlated": (((.0001, .00008), (.00008, .0001)), (.5, .45), (.01, .02)),
             "fov_exit": (((.0001, 0.), (0., .0001)), (.5, .85), (0., .06))}
    rows = []
    for name, (cov, center, velocity) in cases.items():
        state = PersistentWindBelief(model, PersistentMotionConfig(cov, "session_constant", order, "Controlled transport verification"))
        initial = np.exp(-np.sum((points - center)**2, axis=1) / (2 * .06**2))
        for component in state.components:
            component.mean[:] = initial; component.covariance[:] = 0.
        for horizon in (0., 1., 3., 6., 12.):
            future = state.clone(); future.advance(horizon, CausalWind(velocity_yx=velocity))
            d = future.distribution_on(model.grid_shape)
            rows.append({"case": name, "horizon_seconds": horizon, **spatial_summary(d),
                         "signed_field_integral": float(d.mean.mean()),
                         "field_integral_ratio": float(d.mean.sum() / initial.sum()),
                         "weight_change": float(np.max(np.abs(future.weights - state.weights)))})
    return rows


def fixed_evidence_state(config, calls, now):
    """Replay acquisition AND completion propagation boundaries on a forced trace.

    This is an inference diagnostic, not a controller execution of another plan.
    Keeping receipt/decision boundaries matters to the inherited numerical solver.
    """
    belief = PersistentWindBelief(config.model, config.session.motion) if hasattr(config.session, "motion") else GaussianSignalBelief(config.model)
    wind, base, pending = CausalWind(), None, {}
    for call in calls:
        if call.at_seconds > now:
            break
        if call.operation == "plan_next":
            belief.advance(call.at_seconds, wind, config.session.source_position)
            wind = call.wind if call.wind is not None else wind
            base, pending = belief.clone(), {}
        elif call.operation == "ingest_readings":
            new = [r for r in call.readings if r.reading_id not in pending]
            if not new:
                continue
            replay = bool(pending) and min(r.acquired_at_seconds for r in new) < belief.time_seconds - 1e-10
            pending.update({r.reading_id:r for r in new})
            if replay:
                belief = base.clone()
            work = pending.values() if replay else new
            for reading in sorted(work, key=lambda r:(r.acquired_at_seconds,r.reading_id)):
                belief.assimilate(reading, wind, config.session.source_position)
        else:
            belief.advance(call.receipt.received_at_seconds, wind, config.session.source_position)
    belief.advance(now, wind, config.session.source_position)
    return belief


def fixed_evidence(root, names, order=3):
    rows = []
    for name in names:
        path = root / name / "trace.json"
        payload = json.loads(path.read_text())
        base = config_from_dict(payload["configuration"])
        calls = tuple(ControllerCallV1.from_dict(e) for e in payload["events"])
        decisions = [e for e in calls if e.operation == "plan_next" and e.plan.samples]
        selected = sorted({1, len(decisions)//2, len(decisions)-1} - {0})
        for index in selected:
            if index >= len(decisions):
                continue
            event = decisions[index]; now = event.at_seconds
            view = CausalReplayV1(base.session, calls).advance_to(now)
            evidence = {r.reading_id: r for c in calls if c.operation == "ingest_readings" and c.at_seconds <= now
                        for r in c.readings}
            complete_ids = {c.receipt.scan_id for c in calls if c.operation == "finish_scan" and c.at_seconds <= now}
            last_survey = max((c.at_seconds for c in calls if c.operation == "finish_scan" and c.at_seconds <= now
                              and next(d for d in decisions if d.plan.scan_id == c.receipt.scan_id).plan.status in ("reacquisition", "startup_full_scene")), default=0.)
            motion = with_persistent_motion(base, ((.0001, 0.), (0., .0001)), order)
            shape = base.capabilities.lattice_shape; points = grid_positions(shape)
            for label, c in (("baseline", base), ("motion", motion)):
                belief = fixed_evidence_state(c, calls, now)
                if label == "baseline":
                    np.testing.assert_allclose(belief.moments_on(shape)[0], view.predictive_moments(now, shape)[0], atol=1e-10, rtol=0)
                last = max(evidence.values(), key=lambda r: r.acquired_at_seconds, default=None)
                plan = plan_scan_v2(belief.clone(), c.session, event.request, c.wind, view.visit_times,
                    scan_id="hypothetical", now=now, pose=event.position_yx,
                    rotation=len(complete_ids) % len(survey_blocks(shape, c.session.policy.survey_height_fov)),
                    startup_complete=view.startup_complete, last_survey_completed=last_survey,
                    last_reading_at_seconds=last.acquired_at_seconds if last else None,
                    last_integration_seconds=last.integration_seconds if last else None)
                forecasts = []
                for offset in (0., 1., 3., 6., 12.):
                    predicted = belief.clone(); predicted.advance(now + offset, c.wind, c.session.source_position)
                    d = distribution(predicted, shape); density = d.expected_excess(.15)
                    truth = c.source.build().signal_at(np.full(len(points), now+offset), points).reshape(shape)
                    background = truth <= .01 * truth.max()
                    forecasts.append({"offset_seconds": offset, **spatial_summary(d),
                        "map_mse": float(np.mean((d.mean-truth)**2)),
                        "background_excess_fraction": float(density[background].sum()/density.sum()) if truth.max()>0 and density.sum()>1e-12 else None,
                        "mean": d.mean.tolist(), "std": np.sqrt(d.variance).tolist(), "excess_density": density.tolist()})
                rows.append({"source_run": name, "source_trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "decision_index": index, "at_seconds": now, "estimator": label, "evidence_count": len(evidence),
                    "motion": belief.diagnostics() if label == "motion" else None,
                    "hypothetical_plan": plan.to_dict(), "forecasts": forecasts})
    return {"artifact_type": "persistent_wind_fixed_evidence_v3", "quadrature_order": order,
            "interpretation": "Same delivered evidence; hypothetical predictions/plans, not new executions. Existing interpolation numerically spreads signal.", "rows": rows}


def boundary_probe():
    """Post-hoc isolated check of inherited inflow clipping, not a study fixture."""
    model = SignalModelConfig(grid_shape=(12, 16), source_rate=0., diffusivity=0.,
                              process_std_per_sqrt_second=0.)
    initial = GaussianSignalBelief(model)
    initial.mean[:] = 0.; initial.mean.reshape(model.grid_shape)[:, 0] = 1.
    initial.covariance[:] = 0.
    wind = CausalWind(velocity_yx=(0., .04))
    rows = []
    for horizon in (0., 1., 6., 12., 30.):
        state = initial.clone(); state.advance(horizon, wind)
        rows.append({"horizon_seconds":horizon,"field_mean_integral":float(state.mean.mean()),
                     "inflow_cell_mean":float(state.mean.reshape(model.grid_shape)[:,0].mean()),
                     "mean":state.mean.reshape(model.grid_shape).tolist()})
    return {"artifact_type":"inherited_boundary_clamping_probe_v1","model":model.to_dict(),"wind":wind.to_dict(),
            "initial_state":"First physical-inference column one, remaining cells zero; zero covariance; no observations or source.",
            "scope":"Post-hoc engineering diagnostic prompted by persistent signal and concentrated wind weights; outside the predeclared 84 comparisons.",
            "finding":"Clipping between the FOV boundary and first cell center pins incoming-cell signal. This can supply artificial persistent inflow; it is not signal renormalization.",
            "action":"Record the inherited limitation; do not change transport, diffusion, process noise or sources in the frozen comparison.","rows":rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--boundary-probe", action="store_true")
    args = parser.parse_args()
    if args.boundary_probe:
        write_json(args.output / "boundary-clamping-probe.json", boundary_probe())
        return
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "configs/sensing-motion-v3/study.json").read_text())
    write_json(args.output / "fixed-evidence.json", fixed_evidence(root / "outputs/mechanism-revision/development", manifest["representative_cases"], args.order))
    write_json(args.output / "controlled-forecasts.json", {str(n): controlled_forecasts(n) for n in manifest["orders"]})


if __name__ == "__main__":
    main()
