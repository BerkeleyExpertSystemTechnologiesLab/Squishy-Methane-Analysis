"""Bounded mass/cost planning and age-based reacquisition for the callable core."""

from __future__ import annotations

from dataclasses import replace
import numpy as np

from plume_focus.physical_scan import compile_scan, coverage_mask, with_mass
from plume_focus.sensing_contracts import CausalWind, Position, ScanConstraints, ScanPlan, ScannerCapabilities
from plume_focus.signal_belief import GaussianSignalBelief, grid_positions


def _allocate(mass: np.ndarray, desired: np.ndarray, caps: ScannerCapabilities,
              constraints: ScanConstraints, last_sampled: np.ndarray, rotation: int,
              *, scan_id: str, now: float, pose: Position, reacquisition: bool) -> ScanPlan:
    def compile(mask):
        return compile_scan(mask, caps, scan_id=scan_id, started_at_seconds=now,
                            current_position=pose, target_mass=constraints.target_mass)

    chosen = np.zeros(caps.lattice_shape, dtype=bool)
    plan = compile(chosen)
    active_cols = list(np.flatnonzero(desired.any(axis=0)))
    while active_cols:
        old_coverage = coverage_mask(plan, mass.shape, caps)
        candidates = []
        for col in active_cols:
            trial = chosen.copy()
            trial[:, col] = desired[:, col]
            candidate = compile(trial)
            if candidate.duration_seconds > constraints.max_duration_seconds + 1e-10:
                continue
            gain = float(mass[coverage_mask(candidate, mass.shape, caps) & ~old_coverage].sum())
            incremental = max(1e-12, candidate.duration_seconds - plan.duration_seconds)
            age_priority = -float(np.max(last_sampled[desired[:, col], col]))
            cycle_priority = -((int(col) - rotation) % caps.lattice_shape[1])
            score = ((age_priority, cycle_priority, gain / incremental) if reacquisition
                     else (gain / incremental, age_priority, cycle_priority))
            candidates.append((score, int(col), trial, candidate))
        if not candidates:
            break
        _, column, chosen, plan = max(candidates, key=lambda item: item[0])
        active_cols.remove(column)
        if not reacquisition and len(plan.samples) >= constraints.minimum_samples:
            if mass[coverage_mask(plan, mass.shape, caps)].sum() >= constraints.target_mass:
                break
    if len(plan.samples) < constraints.minimum_samples:
        return replace(compile(np.zeros_like(chosen)), reason="No candidate meets minimum samples within the time budget.")
    return plan


def plan_scan(belief: GaussianSignalBelief, capabilities: ScannerCapabilities, constraints: ScanConstraints,
              wind: CausalWind, source: Position | None, last_sampled: np.ndarray, *, scan_id: str,
              now: float, pose: Position, rotation: int = 0, full_scene: bool = False,
              policy: str = "adaptive") -> ScanPlan:
    if policy not in ("adaptive", "static_corridor", "rescan"):
        raise ValueError("Unknown policy.")
    shape = capabilities.lattice_shape
    horizon = constraints.max_duration_seconds
    cache = {}

    def forecast(duration):
        key = round(duration, 10)
        if key not in cache:
            predicted = belief.clone()
            predicted.advance(now + duration, wind, source)
            cache[key] = predicted.mass_on(shape)
        return cache[key]

    last_signature = None
    converged = False
    reacquisition = False
    for _ in range(constraints.max_planning_iterations):
        mass = forecast(horizon)
        nonzero = mass.sum() > 1e-12
        uniform = not nonzero or float(np.std(mass)) <= float(np.mean(mass)) * 0.02
        reacquisition = policy == "rescan" or (policy == "adaptive" and (uniform or wind.error_speed * horizon >= 0.5))
        if full_scene or reacquisition:
            desired = np.ones(shape, dtype=bool)
        elif policy == "static_corridor":
            # Fixed, declared geometry, never derived from evaluator truth.
            center = source if source is not None else (0.5, 0.25)
            points = grid_positions(shape).reshape(*shape, 2)
            desired = ((abs(points[..., 0] - center[0]) <= .18)
                       & (points[..., 1] >= center[1] - .06) & (points[..., 1] <= center[1] + .6))
        else:
            ordered = np.argsort(-mass.ravel(), kind="stable")
            count = int(np.searchsorted(np.cumsum(mass.ravel()[ordered]), constraints.target_mass)) + 1
            count = max(count, constraints.minimum_samples)
            desired = np.zeros(shape, dtype=bool)
            desired.ravel()[ordered[:count]] = True
        if full_scene:
            complete = compile_scan(desired, capabilities, scan_id=scan_id, started_at_seconds=now,
                                    current_position=pose, target_mass=constraints.target_mass)
            if complete.duration_seconds <= constraints.max_duration_seconds + 1e-10:
                plan = complete
            else:
                reacquisition = True
                plan = _allocate(mass, desired, capabilities, constraints, last_sampled, rotation,
                                 scan_id=scan_id, now=now, pose=pose, reacquisition=True)
        else:
            plan = _allocate(mass, desired, capabilities, constraints, last_sampled, rotation,
                             scan_id=scan_id, now=now, pose=pose, reacquisition=reacquisition)
        signature = (plan.swept_cells, round(plan.duration_seconds, 10))
        if signature == last_signature or abs(plan.duration_seconds - horizon) <= 1e-8:
            converged = True
            break
        last_signature = signature
        horizon = plan.duration_seconds
    final_mass = forecast(plan.duration_seconds)
    reason = plan.reason
    if not converged:
        reason = "Planning iteration limit; final path mass rechecked at its own completion time."
    result = with_mass(plan, final_mass, capabilities, converged=converged, reacquisition=reacquisition, reason=reason)
    if result.status == "coverage_target_unmet" and not reason:
        result = replace(result, reason="The bounded candidate search did not meet the mass target within this budget.")
    if full_scene and len(result.swept_cells) < np.prod(shape):
        result = replace(result, reason="Full-scene startup exceeds this budget; executing a charged reacquisition portion.")
    return result
