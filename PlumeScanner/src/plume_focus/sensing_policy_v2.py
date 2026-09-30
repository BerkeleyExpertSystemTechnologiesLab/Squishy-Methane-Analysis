"""Experimental reportable-excess tracking and charged two-axis surveillance."""

from dataclasses import replace
import math
import numpy as np

from plume_focus.physical_scan import compile_profile_scan, coverage_mask
from plume_focus.sensing_messages_v2 import PlanningCandidate
from plume_focus.sensing_policy_v1 import coverage_diagnostic, select_mass_prefix
from plume_focus.signal_belief import ExpectedSignalMass, grid_positions, positive_expectation


def survey_blocks(shape, height_fov):
    """Fixed vertical blocks; splitting a pass never changes beam or sample pitch."""
    rows, cols = shape
    height = max(1, int(math.floor(height_fov * rows + 1e-10)))
    return tuple((lo, min(rows, lo + height), col) for lo in range(0, rows, height) for col in range(cols))


def plan_scan_v2(belief, session, request, wind, last_sampled, *, scan_id, now, pose,
                 rotation, startup_complete, last_survey_completed,
                 last_reading_at_seconds=None, last_integration_seconds=None):
    profile, cfg = session.instrument, session.policy
    shape, budget = profile.lattice_shape, request.max_duration_seconds
    weights = np.asarray(session.physical_quadrature_weights).reshape(shape)
    threshold = session.allocation_threshold
    cache, distributions, records = {}, {}, []

    def compile(mask):
        return compile_profile_scan(mask, profile, scan_id=scan_id, started_at_seconds=now,
            current_position=pose, last_reading_at_seconds=last_reading_at_seconds,
            last_integration_seconds=last_integration_seconds, target_mass=cfg.gamma)

    def forecast(duration):
        duration = float(duration)
        if duration not in cache:
            state = belief.clone()
            state.advance(now + duration, wind, session.source_position)
            if hasattr(state, "distribution_on"):
                distribution = state.distribution_on(shape)
                distributions[duration] = distribution
                mu, var = distribution.mean, distribution.variance
                density = distribution.expected_excess(threshold)
            else:
                mu, var = state.moments_on(shape)
                density = positive_expectation(mu, var, threshold)
            mass = weights * density
            total = float(mass.sum())
            supported = cfg.policy_id == "mass90_segment_v2" or bool(np.any(mu > threshold))
            cache[duration] = ExpectedSignalMass(mu, var, density, mass, total,
                                                 mass / total if total > cfg.mass_floor and supported else None)
        return cache[duration]

    def result(plan, status, reason, termination, failures=(), converged=False):
        fields = dict(status=status, policy_id=cfg.policy_id, reason=reason, termination=termination,
                      tracking_shortfall=tuple(failures), candidates=tuple(records),
                      converged=converged, startup_complete=startup_complete)
        if plan.samples:
            moments = forecast(plan.duration_seconds)
            retained = coverage_mask(plan, shape, profile.capabilities())
            recall = None if moments.normalized is None else min(1., float(moments.normalized[retained].sum()))
            fields.update(predicted_mass_recall=recall, expected_signal_total=moments.total,
                confidence=coverage_diagnostic(moments, retained, weights, threshold=threshold,
                    delta=cfg.confidence_delta, mass_floor=cfg.mass_floor, gamma=cfg.gamma,
                    envelope=distributions[plan.duration_seconds].envelope if plan.duration_seconds in distributions else None))
        else:
            fields["status"] = "no_legal_action"
        return replace(plan, **fields)

    def survey(reason, termination="survey_trigger", failures=()):
        blocks = survey_blocks(shape, cfg.survey_height_fov)
        order = sorted(range(len(blocks)), key=lambda i: (
            float(np.min(last_sampled[blocks[i][0]:blocks[i][1], blocks[i][2]])),
            (i - rotation) % len(blocks)))
        selected = np.zeros(shape, bool)
        chosen = compile(selected)
        for i in order:
            lo, hi, col = blocks[i]
            trial = selected.copy()
            trial[lo:hi, col] = True
            candidate = compile(trial)
            if candidate.duration_seconds <= budget + cfg.time_tolerance_seconds:
                selected, chosen = trial, candidate
        if len(chosen.samples) < cfg.minimum_samples:
            chosen = compile(np.zeros(shape, bool))
            reason += "; no block action fits minimum samples"
        return result(chosen, "reacquisition", reason + "; charged two-axis block surveillance", termination, failures)

    if budget <= 0 or cfg.minimum_samples > math.prod(shape):
        return result(compile(np.zeros(shape, bool)), "no_legal_action", "No available action capacity", "invalid_action_capacity")
    if session.startup_mode == "full_scene" and not startup_complete:
        complete = compile(np.ones(shape, bool))
        if complete.duration_seconds <= budget + cfg.time_tolerance_seconds:
            return result(complete, "startup_full_scene", "Full-scene startup; actual visits required", "startup")
        return survey("Full-scene startup remains incomplete", "startup_budget")
    initial = forecast(budget)
    reasons = []
    fixed = cfg.policy_id == "static_corridor_segment_v2"
    if initial.normalized is None and not fixed:
        reasons.append("no_reportable_signal")
    elif initial.normalized is not None and not fixed:
        average = float(np.sum(weights * initial.density))
        cv = math.sqrt(float(np.sum(weights * (initial.density - average)**2))) / average
        if cv <= cfg.uniform_density_tolerance:
            reasons.append("uniform_expected_density")
    if now - last_survey_completed >= cfg.survey_interval_seconds:
        reasons.append("periodic_surveillance_due")
    if wind.error_speed * budget >= cfg.max_prediction_error_fov:
        reasons.append("wind_error_displacement")
    if cfg.policy_id == "age_segment_v2":
        reasons.append("age_segment_policy")
    if reasons:
        return survey(", ".join(reasons))
    if cfg.policy_id == "static_corridor_segment_v2":
        points = grid_positions(shape).reshape(*shape, 2)
        y0, x0, y1, x1 = cfg.corridor_yxyx
        selected = (points[..., 0] >= y0) & (points[..., 0] <= y1) & (points[..., 1] >= x0) & (points[..., 1] <= x1)
        plan = compile(selected)
        failed = [name for name, bad in (("duration_budget", plan.duration_seconds > budget + cfg.time_tolerance_seconds),
                                         ("minimum_samples", len(plan.samples) < cfg.minimum_samples)) if bad]
        return survey("Fixed corridor infeasible", "corridor_constraints", failed) if failed else result(
            plan, "static_corridor", "Full declared corridor; reportable-mass target does not apply", "fixed_geometry")
    horizon, history, valid = budget, [], []
    termination = "iteration_limit"
    for _ in range(cfg.max_planning_iterations):
        predicted = forecast(horizon)
        if predicted.normalized is None:
            return survey("Forecast has no reportable signal", "mass_floor", ("mass_floor",))
        selected = select_mass_prefix(predicted.normalized, cfg.gamma, cfg.minimum_samples, cfg.mass_tolerance)
        plan = compile(selected)
        own = forecast(plan.duration_seconds)
        retained = coverage_mask(plan, shape, profile.capabilities())
        recall = None if own.normalized is None else float(own.normalized[retained].sum())
        failures = [name for name, bad in (("duration_budget", plan.duration_seconds > budget + cfg.time_tolerance_seconds),
            ("minimum_samples", len(plan.samples) < cfg.minimum_samples), ("mass_floor", recall is None),
            ("coverage_target", recall is not None and recall < cfg.gamma - cfg.mass_tolerance)) if bad]
        records.append(PlanningCandidate(horizon, plan.duration_seconds, plan.requested_cells, recall, tuple(failures)))
        if not failures:
            valid.append(plan)
        signature = (plan.requested_cells, plan.duration_seconds)
        if history and signature[0] == history[-1][0] and abs(signature[1] - history[-1][1]) <= cfg.time_tolerance_seconds:
            termination = "identical_candidate"
            break
        if abs(plan.duration_seconds - horizon) <= cfg.time_tolerance_seconds:
            termination = "time_fixed_point"
            break
        if any(signature[0] == cells and abs(signature[1] - duration) <= cfg.time_tolerance_seconds for cells, duration in history[:-1]):
            termination = "cycle"
            break
        history.append(signature)
        horizon = plan.duration_seconds
    if valid:
        chosen = min(valid, key=lambda p: (p.duration_seconds, p.requested_cells))
        own = forecast(chosen.duration_seconds)
        recomputed = tuple(np.flatnonzero(select_mass_prefix(own.normalized, cfg.gamma, cfg.minimum_samples, cfg.mass_tolerance)).tolist())
        return result(chosen, "target_met", f"90% expected excess above reportable threshold {threshold:g}; full selected geometry", termination,
                      converged=recomputed == chosen.requested_cells)
    failures = tuple(sorted({f for record in records for f in record.failed_constraints}))
    return survey("Tracking selection infeasible; no tracking cells trimmed", termination, failures)
