"""Frozen mass90 column rule: preserve selection, forecast, check, or reacquire."""

from __future__ import annotations

from dataclasses import replace
from statistics import NormalDist
import math

import numpy as np

from plume_focus.physical_scan import compile_profile_scan, coverage_mask
from plume_focus.sensing_contracts import CausalWind, Position
from plume_focus.sensing_messages_v2 import CoverageDiagnostic, PlanningCandidate, ScanPlanV2, ScanRequestV1
from plume_focus.sensing_session import SensingSessionV1
from plume_focus.signal_belief import ExpectedSignalMass, GaussianSignalBelief, grid_positions


def select_mass_prefix(mass: np.ndarray, gamma: float, minimum_samples: int, tolerance: float) -> np.ndarray:
    """Smallest descending prefix with row-major ties and a minimum-count extension."""
    ordered = np.argsort(-mass.ravel(), kind="stable")
    count = int(np.searchsorted(np.cumsum(mass.ravel()[ordered]), gamma - tolerance, side="left")) + 1
    count = min(mass.size, max(count, minimum_samples))
    mask = np.zeros(mass.shape, bool)
    mask.ravel()[ordered[:count]] = True
    return mask


def coverage_diagnostic(moments: ExpectedSignalMass, retained: np.ndarray, weights: np.ndarray,
                        *, threshold: float, delta: float, mass_floor: float, gamma: float,
                        envelope=None) -> CoverageDiagnostic:
    n = moments.mean.size
    kappa = NormalDist().inv_cdf(1 - delta / (2 * n))
    sigma = np.sqrt(moments.variance)
    lo, hi = (moments.mean - kappa * sigma, moments.mean + kappa * sigma) if envelope is None else envelope(kappa)
    lower = weights * np.maximum(lo - threshold, 0)
    upper = weights * np.maximum(hi - threshold, 0)
    numerator = float(lower[retained].sum())
    denominator = numerator + float(upper[~retained].sum())
    bound = numerator / denominator if denominator > 0 else None
    total_lower = float(lower.sum())
    meaningful = total_lower > mass_floor
    status = "certified" if meaningful and bound is not None and bound >= gamma else "not_certified"
    return CoverageDiagnostic(delta, n, kappa, bound, total_lower, meaningful, status)


def plan_scan_v1(belief: GaussianSignalBelief, session: SensingSessionV1, request: ScanRequestV1,
                 wind: CausalWind, last_sampled: np.ndarray, *, scan_id: str, now: float,
                 pose: Position, rotation: int, startup_complete: bool,
                 last_reading_at_seconds: float | None = None,
                 last_integration_seconds: float | None = None) -> ScanPlanV2:
    profile, config = session.instrument, session.policy
    shape, budget = profile.lattice_shape, request.max_duration_seconds
    weights = np.asarray(session.physical_quadrature_weights).reshape(shape)
    caps = profile.capabilities()
    cache: dict[float, ExpectedSignalMass] = {}
    records: list[PlanningCandidate] = []

    def compile(mask):
        return compile_profile_scan(mask, profile, scan_id=scan_id, started_at_seconds=now,
                                    current_position=pose, last_reading_at_seconds=last_reading_at_seconds,
                                    last_integration_seconds=last_integration_seconds,
                                    target_mass=config.gamma)

    def forecast(duration):
        key = float(duration)
        if key not in cache:
            predicted = belief.clone()
            predicted.advance(now + key, wind, session.source_position)
            cache[key] = predicted.expected_mass_on(shape, weights, config.mass_floor)
        return cache[key]

    def selection(moments):
        if config.policy_id == "static_corridor_with_reacquisition_v1":
            y0, x0, y1, x1 = config.corridor_yxyx
            points = grid_positions(shape).reshape(*shape, 2)
            return ((points[..., 0] >= y0) & (points[..., 0] <= y1)
                    & (points[..., 1] >= x0) & (points[..., 1] <= x1))
        return select_mass_prefix(moments.normalized, config.gamma, config.minimum_samples, config.mass_tolerance)

    def result(plan, status, reason, termination, *, converged=False, shortfall=()):
        if not plan.samples:
            return replace(plan, status="no_legal_action", reason=reason, termination=termination,
                           policy_id=config.policy_id, candidates=tuple(records),
                           tracking_shortfall=tuple(shortfall), startup_complete=startup_complete)
        moments = forecast(plan.duration_seconds)
        retained = coverage_mask(plan, shape, caps)
        recall = None if moments.normalized is None else min(1., float(moments.normalized[retained].sum()))
        diagnostic = coverage_diagnostic(moments, retained, weights, threshold=belief.config.background_threshold,
                                         delta=config.confidence_delta, mass_floor=config.mass_floor, gamma=config.gamma)
        return replace(plan, predicted_mass_recall=recall, expected_signal_total=moments.total,
                       status=status, reason=reason, termination=termination, converged=converged,
                       candidates=tuple(records), tracking_shortfall=tuple(shortfall), confidence=diagnostic,
                       policy_id=config.policy_id, startup_complete=startup_complete)

    def reacquire(trigger, termination="reacquisition_trigger", shortfall=()):
        chosen = np.zeros(shape, bool)
        plan = compile(chosen)
        oldest = np.min(last_sampled, axis=0)
        order = sorted(range(shape[1]), key=lambda col: (oldest[col], (col - rotation) % shape[1]))
        for column in order:
            trial = chosen.copy()
            trial[:, column] = True
            candidate = compile(trial)
            if candidate.duration_seconds <= budget + config.time_tolerance_seconds:
                chosen, plan = trial, candidate
        if len(plan.samples) < config.minimum_samples:
            return result(compile(np.zeros(shape, bool)), "no_legal_action", trigger + "; no whole-column action fits minimum samples",
                          termination, shortfall=shortfall)
        return result(plan, "reacquisition", trigger, termination, shortfall=shortfall)

    if budget <= 0 or config.minimum_samples > math.prod(shape):
        return result(compile(np.zeros(shape, bool)), "no_legal_action", "Nonpositive budget or minimum samples exceed lattice capacity",
                      "invalid_action_capacity")

    if session.startup_mode == "full_scene" and not startup_complete:
        complete = compile(np.ones(shape, bool))
        if complete.duration_seconds <= budget + config.time_tolerance_seconds:
            return result(complete, "startup_full_scene", "Full-scene startup; completion depends on actual valid visits", "startup")
        return reacquire("Full-scene startup incomplete; executing a charged whole-column portion", "startup_budget")

    initial = forecast(budget)
    triggers = []
    if initial.normalized is None:
        triggers.append("no_meaningful_signal")
    else:
        density_mean = float(np.sum(weights * initial.density))
        cv = math.sqrt(float(np.sum(weights * (initial.density - density_mean)**2))) / density_mean
        if cv <= config.uniform_density_tolerance:
            triggers.append("uniform_expected_density")
    ages = np.where(last_sampled < 0, now, now - last_sampled)
    if np.any(ages >= config.stale_after_seconds):
        triggers.append("overdue_cell")
    if wind.error_speed * budget >= config.max_prediction_error_fov:
        triggers.append("wind_error_displacement")
    if config.policy_id == "age_rescan_v1":
        triggers.append("age_rescan_policy")
    if triggers:
        return reacquire(", ".join(triggers))

    if config.policy_id == "static_corridor_with_reacquisition_v1":
        # A non-adaptive baseline executes its declared region, irrespective of
        # the adaptive mass target. It shares physical, startup and age rules.
        plan = compile(selection(initial))
        failures = []
        if plan.duration_seconds > budget + config.time_tolerance_seconds:
            failures.append("duration_budget")
        if len(plan.samples) < config.minimum_samples:
            failures.append("minimum_samples")
        if failures:
            return reacquire("Fixed corridor does not fit the physical constraints", "corridor_constraints", tuple(failures))
        return result(plan, "static_corridor", "Fixed declared corridor; 90% adaptive target does not apply", "fixed_geometry")

    horizon = budget
    visited: list[tuple[tuple[int, ...], float]] = []
    valid: list[ScanPlanV2] = []
    termination = "iteration_limit"
    for _ in range(config.max_planning_iterations):
        predicted = forecast(horizon)
        if predicted.normalized is None:
            return reacquire("Forecast fell below meaningful signal floor", "no_meaningful_forecast", ("mass_floor",))
        plan = compile(selection(predicted))
        own = forecast(plan.duration_seconds)
        recall = None if own.normalized is None else float(own.normalized[coverage_mask(plan, shape, caps)].sum())
        failed = []
        if plan.duration_seconds > budget + config.time_tolerance_seconds:
            failed.append("duration_budget")
        if len(plan.samples) < config.minimum_samples:
            failed.append("minimum_samples")
        if recall is None:
            failed.append("mass_floor")
        elif recall < config.gamma - config.mass_tolerance:
            failed.append("coverage_target")
        records.append(PlanningCandidate(horizon, plan.duration_seconds, plan.requested_cells, recall, tuple(failed)))
        if not failed:
            valid.append(plan)
        if own.normalized is None:
            return reacquire("Own-completion forecast below meaningful signal floor", "no_meaningful_forecast", ("mass_floor",))
        signature = (plan.requested_cells, plan.duration_seconds)
        same_previous = bool(visited) and signature[0] == visited[-1][0] and abs(signature[1] - visited[-1][1]) <= config.time_tolerance_seconds
        repeated = any(signature[0] == cells and abs(signature[1] - duration) <= config.time_tolerance_seconds for cells, duration in visited[:-1])
        if same_previous:
            termination = "identical_candidate"
            break
        if abs(plan.duration_seconds - horizon) <= config.time_tolerance_seconds:
            termination = "time_fixed_point"
            break
        if repeated:
            termination = "cycle"
            break
        visited.append(signature)
        horizon = plan.duration_seconds
    if valid:
        chosen = min(valid, key=lambda p: (p.duration_seconds, p.requested_cells))
        recomputed = tuple(np.flatnonzero(selection(forecast(chosen.duration_seconds))).tolist())
        converged = recomputed == chosen.requested_cells
        return result(chosen, "target_met", "Shortest valid visited candidate; no global optimality claim", termination, converged=converged)
    failures = tuple(sorted({item for record in records for item in record.failed_constraints}))
    return reacquire("Tracking shortfall; executing a separately charged age schedule", termination, failures)
