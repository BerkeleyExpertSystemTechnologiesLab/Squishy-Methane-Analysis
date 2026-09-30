"""Evaluator-only causal reconstruction loss over a common physical horizon."""

from __future__ import annotations

import math

import numpy as np

from plume_focus.sensing_evaluation import summarize_run
from plume_focus.sensing_execution_v1 import CausalReplayV1, ReconstructionGridV1, SensingRunV1
from plume_focus.sensing_session import configuration_hash
from plume_focus.sensing_simulation import SensorSource, SourceExhausted
from plume_focus.signal_belief import grid_positions, positive_expectation
from plume_focus.physical_scan import coverage_mask


def reconstruction_loss(run: SensingRunV1, truth: SensorSource | None, *,
                        grid: ReconstructionGridV1 | None = None, distribution_diagnostics=True) -> dict:
    grid = grid or run.config.evaluation
    horizon = run.config.duration_seconds
    replay = CausalReplayV1(run.config.session, run.events)
    complete = truth is not None and (truth.end_time_seconds is None or truth.end_time_seconds >= horizon - 1e-10)
    result = {"horizon_seconds": horizon, "evaluation": grid.to_dict(),
              "evaluation_hash": configuration_hash(grid.to_dict()), "truth_available": truth is not None,
              "status": "complete" if complete else "truth_unavailable" if truth is None else "incomplete_truth_horizon",
              "J": None, "J_over_T": None, "signal_index_squared_error_integral": None,
              "signal_index_squared_error_mean": None, "posterior_pointwise_95_coverage": None,
              "signed_mean_enhancement_error_integral": None, "signed_mean_enhancement_error_mean": None,
              "expected_excess_integral": None,
              "coverage_band_method": "component_envelope" if hasattr(run.config.session, "motion") else "gaussian_marginal",
              "quadrature_samples": [], "units": run.config.model.signal_units + "^2"}
    if not complete:
        # Still validate the recorded controller sequence. No fabricated or
        # truncated reconstruction score is returned for unavailable truth.
        replay.advance_to(max((e.at_seconds for e in run.events), default=0.))
        return result
    points = grid_positions(grid.shape)
    times = np.arange(int(math.ceil(horizon / grid.step_seconds)), dtype=float) * grid.step_seconds
    times = times[times < horizon]
    durations = np.diff(np.append(times, horizon))
    loss = index_loss = calibration = signed_index = expected_integral = 0.
    mixture = hasattr(run.config.session, "motion")
    threshold = getattr(run.config.session, "allocation_threshold", run.config.model.background_threshold)
    for t, duration in zip(times, durations):
        controller = replay.advance_to(float(t))
        distribution = controller.predictive_distribution(float(t), grid.shape, fast_projection=True) if mixture and distribution_diagnostics else None
        if distribution is not None:
            mean, variance = distribution.mean, distribution.variance
        elif mixture:
            mean, variance = controller.predictive_aggregate_moments(float(t), grid.shape)
        else:
            mean, variance = controller.predictive_moments(float(t), grid.shape)
        try:
            values = truth.signal_at(np.full(len(points), t), points).reshape(grid.shape)
        except SourceExhausted:
            result.update(status="incomplete_truth_horizon", quadrature_samples=[])
            return result
        if np.any(~np.isfinite(values)):
            raise ValueError("Evaluator truth must be finite or explicitly unavailable.")
        error = mean - values
        mse = float(np.mean(error**2))
        index_error = float(np.mean(error)**2)
        if not distribution_diagnostics:
            covered = expected = 0.
        elif distribution is not None:
            lower, upper = distribution.envelope(1.959963984540054)
            covered = float(np.mean((values >= lower) & (values <= upper)))
            expected = float(distribution.expected_excess_vectorized(threshold).mean())
        else:
            covered = float(np.mean(np.abs(error) <= 1.959963984540054 * np.sqrt(variance)))
            expected = float(positive_expectation(mean, variance, threshold).mean())
        loss += duration * mse
        index_loss += duration * index_error
        calibration += duration * covered
        signed_index += duration * float(np.mean(error))
        expected_integral += duration * expected
        result["quadrature_samples"].append({"time_seconds": float(t), "duration_seconds": float(duration),
                                             "map_mse": mse, "signal_index_squared_error": index_error,
                                             "signed_mean_enhancement_error": float(np.mean(error)),
                                             "expected_excess": expected,
                                             "pointwise_95_coverage": covered})
    # Check remaining receipts as well, even if actual execution overran T.
    replay.advance_to(max(horizon, max((e.at_seconds for e in run.events), default=0.)))
    result.update(J=float(loss), J_over_T=float(loss / horizon),
                  signal_index_squared_error_integral=float(index_loss),
                  signal_index_squared_error_mean=float(index_loss / horizon),
                  posterior_pointwise_95_coverage=float(calibration / horizon),
                  coverage_band_method="component_envelope" if mixture else "gaussian_marginal",
                  signed_mean_enhancement_error_integral=float(signed_index),
                  signed_mean_enhancement_error_mean=float(signed_index / horizon),
                  expected_excess_integral=float(expected_integral))
    result["distribution_diagnostics_computed"] = distribution_diagnostics
    if not distribution_diagnostics:
        result["posterior_pointwise_95_coverage"] = result["expected_excess_integral"] = None
        for row in result["quadrature_samples"]:
            row["pointwise_95_coverage"] = row["expected_excess"] = None
    return result


def summarize_v1(run: SensingRunV1, truth: SensorSource | None = None, *,
                 grid: ReconstructionGridV1 | None = None, loss: dict | None = None) -> dict:
    grid = grid or run.config.evaluation
    report = summarize_run(run, truth, evaluation_shape=grid.shape, background_threshold=run.config.model.background_threshold)
    delivery_times = {}
    for event in run.events:
        for reading in event.readings:
            delivery_times.setdefault(reading.reading_id, event.at_seconds)
    totals = dict(travel=0., acquisition=0., settling=0., detector_wait=0., unclassified_execution=0., receipt_latency=0.)
    deviations = []
    for trace, row in zip(run.scans, report["scan_schedule"]):
        plan, execution = trace.plan, trace.execution
        parts = dict(travel=0., acquisition=0., settling=0., detector_wait=0., unclassified_execution=0.)
        kinds = {"move": "travel", "sweep": "acquisition", "dwell": "acquisition", "settle": "settling", "wait": "detector_wait"}
        for segment in execution.executed_plan.segments:
            parts[kinds[segment.kind]] += segment.completed_at_seconds - segment.started_at_seconds
        parts["unclassified_execution"] = max(0., execution.executed_plan.duration_seconds - sum(parts.values()))
        for key, value in parts.items():
            totals[key] += value
        latency = execution.receipt.received_at_seconds - execution.receipt.completed_at_seconds
        totals["receipt_latency"] += latency
        readings = execution.readings
        acquisition_times = [r.acquired_at_seconds for r in readings]
        delivered_lags = [delivery_times[r.reading_id] - r.acquired_at_seconds for r in readings]
        integration_deviations = sum(not math.isclose(r.integration_seconds, run.config.capabilities.integration_seconds, abs_tol=1e-10) for r in readings)
        footprint_deviations = sum(r.footprint != run.config.capabilities.footprint for r in readings)
        pointing_deviations = sum(r.position_yx != p.position_yx for r, p in zip(readings, plan.samples))
        overrun = execution.receipt.completed_at_seconds - plan.completed_at_seconds
        if integration_deviations or footprint_deviations or pointing_deviations or abs(overrun) > 1e-8:
            deviations.append({"scan_id": plan.scan_id, "completion_deviation_seconds": overrun,
                               "integration_deviations": integration_deviations, "footprint_deviations": footprint_deviations,
                               "pointing_deviations": pointing_deviations})
        row.update(time_components_seconds=parts, receipt_latency_seconds=latency,
                   mean_reading_delivery_lag_seconds=float(np.mean(delivered_lags)) if delivered_lags else None,
                   acquisition_time_spread_seconds=max(acquisition_times) - min(acquisition_times) if acquisition_times else None,
                   startup_complete_before_scan=plan.startup_complete,
                   mass_target_is_execution_requirement=plan.status == "target_met",
                   tracking_shortfall=list(plan.tracking_shortfall), planning_termination=plan.termination,
                   candidates=[c.to_dict() for c in plan.candidates],
                   confidence=plan.confidence.to_dict() if plan.confidence else None)
    last = run.scans[-1].execution.receipt.received_at_seconds if run.scans else 0.
    tracking = [r["actual_sweep_mass_recall"] for r in report["scan_schedule"]
                if r["planning_status"] in ("target_met", "static_corridor") and r["actual_sweep_mass_recall"] is not None]
    full_refresh_lower_bound = math.prod(run.config.capabilities.lattice_shape) / run.config.session.instrument.acquisition.cadence_hz
    report.update(schema_version=2, report_type=run.config.session.formulation_id, profile=run.config.session.instrument.profile_id,
                  policy=run.config.policy, configuration=run.config.to_dict(), configuration_hashes=run.config.session.hashes,
                  reconstruction=loss if loss is not None else reconstruction_loss(run, truth, grid=grid),
                  startup_complete=run.startup_complete, visited_fraction=run.visited_fraction,
                  actual_time_components_seconds=totals, idle_tail_seconds=max(0., run.config.duration_seconds - last),
                  profile_deviations=deviations,
                  reacquisition_count=sum(s.plan.status == "reacquisition" for s in run.scans),
                  tracking_shortfall_count=sum(bool(s.plan.tracking_shortfall) for s in run.scans),
                  no_action_count=sum(e.plan is not None and not e.plan.samples for e in run.events),
                  mean_tracking_mass_recall=float(np.mean(tracking)) if tracking else None,
                  stale_threshold_below_scene_acquisition_lower_bound=run.config.session.policy.stale_after_seconds < full_refresh_lower_bound)
    report["limitations"].extend([
        "No decay term was added: excessive modeled signal persistence remains a possible failure.",
        "Posterior bands are model conditional; pointwise coverage is a descriptive calibration check.",
        "Spatial mean enhancement is a signal index, not an emission rate.",
        "A stale threshold below achievable scene-refresh time can force continual reacquisition."])
    if hasattr(run.config.session, "allocation_threshold"):
        threshold = run.config.session.allocation_threshold
        shape = run.config.capabilities.lattice_shape
        weights = np.asarray(run.config.session.physical_quadrature_weights).reshape(shape)
        points = grid_positions(shape)
        for trace, row in zip(run.scans, report["scan_schedule"]):
            mu, var = trace.predicted.mean, trace.predicted.std**2
            distribution = trace.predicted_distribution
            density = distribution.expected_excess(threshold) if distribution is not None else positive_expectation(mu, var, threshold)
            total = float(np.sum(weights * density))
            zero = distribution.expected_excess(run.config.model.background_threshold) if distribution is not None else positive_expectation(mu, var, run.config.model.background_threshold)
            sweeps = [s for s in trace.execution.executed_plan.segments if s.kind == "sweep"]
            row.update(allocation_threshold=threshold,
                expected_reportable_excess=total,
                expected_zero_threshold_excess=float(np.sum(weights * zero)),
                uncertainty_excess_fraction=float(np.sum(weights * (density - np.maximum(mu - threshold, 0))) / total) if total > run.config.session.policy.mass_floor else None,
                mean_acquiring_vertical_extent=float(np.mean([abs(s.end_yx[0] - s.start_yx[0]) for s in sweeps])) if sweeps else None,
                predicted_background_mass_fraction=None, actual_reportable_signal_recall=None)
            if distribution is not None:
                row.update(prediction_within_hypothesis_variance=float(distribution.within_variance.mean()),
                           prediction_between_hypothesis_variance=float(distribution.between_variance.mean()),
                           posterior_motion=trace.motion_diagnostics,
                           coverage_band_method="component_envelope_cellwise_union_bound")
            t = trace.plan.completed_at_seconds
            if truth is not None and (truth.end_time_seconds is None or t <= truth.end_time_seconds):
                values = truth.signal_at(np.full(len(points), t), points).reshape(shape)
                background = values <= .01 * values.max()
                if total > run.config.session.policy.mass_floor:
                    row["predicted_background_mass_fraction"] = float(np.sum(weights[background] * density[background]) / total)
            t = trace.execution.receipt.completed_at_seconds
            if truth is not None and (truth.end_time_seconds is None or t <= truth.end_time_seconds):
                values = np.maximum(truth.signal_at(np.full(len(points), t), points).reshape(shape) - threshold, 0)
                swept = coverage_mask(trace.execution.executed_plan, shape, run.config.capabilities)
                if values.sum() > 1e-15:
                    row["actual_reportable_signal_recall"] = float(values[swept].sum() / values.sum())
        report.update(allocation_threshold=threshold, simulated_noise_std=run.config.simulated_noise_std,
                      assumed_observation_noise_std=run.config.model.sensor_noise_std,
                      successful_tracking_count=sum(t.plan.status == "target_met" for t in run.scans),
                      full_scene_ever_visited=run.visited_fraction == 1.)
        report["limitations"][-1] = "V2 periodic surveillance is a service interval, not a guarantee of full-scene refresh."
        report["limitations"].append("90% refers to expected excess above the reportable threshold; full-signal recall and full-map loss remain separate.")
        report["focusing_audit"] = focusing_audit(run, report["scan_schedule"])
    if hasattr(run.config.session, "motion"):
        report["motion_model"] = run.config.session.motion.to_dict()
        report["limitations"].extend([
            "Finite quadrature approximation to session-constant wind bias; no changing local drift.",
            "Mixture bands use conservative component envelopes, not an aggregate Gaussian or calibrated physical guarantee.",
            "Unchanged process noise may overlap with the newly represented motion uncertainty."])
    if run.scans and hasattr(run.scans[0].plan, "risk_candidates"):
        report.update(value_selected_count=sum(s.plan.status == "value_selected" for s in run.scans),
                      estimator_id=run.config.session.estimator_id, selection_criterion="expected_physical_time_map_risk")
        for trace, row in zip(run.scans, report["scan_schedule"]):
            plan = trace.plan
            shape = run.config.capabilities.lattice_shape
            visited = {int(min(int(r.position_yx[0]*shape[0]),shape[0]-1)*shape[1] +
                           min(int(r.position_yx[1]*shape[1]),shape[1]-1)) for r in trace.execution.readings}
            completed = bool(plan.service_required_cells) and set(plan.service_required_cells) <= visited
            realized = None
            if report["reconstruction"]["J"] is not None:
                start, end = plan.started_at_seconds, plan.started_at_seconds+plan.lookahead_seconds
                realized = sum(max(0., min(end,q["time_seconds"]+q["duration_seconds"])-max(start,q["time_seconds"]))*q["map_mse"]
                               for q in report["reconstruction"]["quadrature_samples"])
            row.update(risk_candidates=[c.to_dict() for c in plan.risk_candidates], selected_candidate=plan.selected_candidate,
                       lookahead_seconds=plan.lookahead_seconds, expected_integrated_map_risk=plan.expected_integrated_map_risk,
                       idle_integrated_map_risk=plan.idle_integrated_map_risk, risk_history_count=plan.history_count,
                       risk_step_seconds=plan.risk_step_seconds, risk_sampling_seed=plan.risk_sampling_seed,
                       service_status=plan.service_status, service_required_cells=list(plan.service_required_cells),
                       service_completed=completed, actual_followon_integrated_map_error=realized,
                       risk_comparison_note="Predicted repeats versus actual later replanning; not a matched-continuation calibration test.")
        report["service_required_count"] = sum(bool(s.plan.service_required_cells) for s in run.scans)
        report["service_completed_count"] = sum(r["service_completed"] for r in report["scan_schedule"])
        report["service_unreachable_count"] = sum(s.plan.service_status == "unmet_no_reachable_block" for s in run.scans)
        report["limitations"] = [s for s in report["limitations"] if not s.startswith("V2 periodic") and not s.startswith("90% refers")]
        report["limitations"].extend([
            "Risk selection is over at most 14 repeated-pattern candidates, not a globally optimal or adaptive rollout.",
            "Coverage is diagnostic, not a 90% execution requirement; small service visits do not guarantee full refresh.",
            "Predicted risk is model conditional; lower risk need not imply lower true error under model mismatch."])
    return report


def focusing_audit(run, rows):
    """Receipt-time actual visits and area changes, independent of raster display."""
    shape = run.config.capabilities.lattice_shape
    visits = np.full(shape, -1.)
    last_tracking = None
    tracking_sizes = []
    outside_service_readings = 0
    for trace, row in zip(run.scans, rows):
        outside = 0
        for reading in trace.execution.readings:
            cell = np.minimum((np.array(reading.position_yx) * shape).astype(int), np.array(shape) - 1)
            flat = int(np.ravel_multi_index(tuple(cell), shape))
            visits[tuple(cell)] = max(visits[tuple(cell)], reading.acquired_at_seconds)
            if trace.plan.status == "reacquisition" and last_tracking is not None and flat not in last_tracking:
                outside += 1
        now = trace.execution.receipt.received_at_seconds
        observed = visits >= 0
        ages = now - visits[observed]
        # The existing swept_fraction uses executed segments/footprints. A copied
        # compiled cell list can still describe the full plan after early completion.
        row.update(unvisited_fraction=float(np.mean(~observed)),
                   stale_or_unvisited_fraction=float(np.mean((~observed) | (now - visits > run.config.session.policy.stale_after_seconds))),
                   observed_age_quantiles_seconds=dict(zip(("p0", "p50", "p90", "p100"), np.quantile(ages, (0., .5, .9, 1.)).tolist())) if ages.size else None,
                   surveillance_readings_outside_last_tracking=outside if last_tracking is not None else None)
        outside_service_readings += outside
        if trace.plan.status in ("target_met", "value_selected"):
            last_tracking = set(trace.plan.swept_cells)
            tracking_sizes.append(len(trace.plan.requested_cells))
    differences = np.diff(tracking_sizes)
    return {"scope": "Actual cell-center visits at receipt time; unvisited cells are separate from observed-age quantiles.",
            "tracking_requested_cell_sequence": tracking_sizes,
            "tracking_expansions": int(np.sum(differences > 0)),
            "tracking_contractions": int(np.sum(differences < 0)),
            "tracking_unchanged_sizes": int(np.sum(differences == 0)),
            "tracking_at_minimum_requested_cells": sum(n <= run.config.session.policy.minimum_samples for n in tracking_sizes),
            "surveillance_readings_outside_last_tracking": outside_service_readings,
            "limitation": "Finite-run area changes and surveillance are diagnostics, not a guarantee against self-reinforcing missed signal."}
