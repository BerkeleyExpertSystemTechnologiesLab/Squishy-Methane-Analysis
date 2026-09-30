"""Bounded reconstruction-risk acquisition; physical paths remain immutable."""

from dataclasses import dataclass
from collections import OrderedDict
import hashlib
import math

import numpy as np

from plume_focus.physical_scan import compile_profile_scan, coverage_mask
from plume_focus.sensing_messages_v3 import RiskCandidate, risk_plan
from plume_focus.sensing_policy_v2 import plan_scan_v2, survey_blocks
from plume_focus.sensing_risk import ForecastSweep, covariance_proposal, receipt_time, score_pattern
from plume_focus.signal_belief import grid_positions, positive_expectation


@dataclass(frozen=True)
class CandidatePath:
    name: str
    core_cells: tuple[int, ...]
    plan: object


def cell_mask(shape, cells):
    mask = np.zeros(math.prod(shape), bool)
    mask[list(cells)] = True
    return mask.reshape(shape)


def compile_for(session, mask, scan_id, now, pose, last_reading=None, last_integration=None):
    return compile_profile_scan(mask, session.instrument, scan_id=scan_id, started_at_seconds=now,
        current_position=pose, last_reading_at_seconds=last_reading,
        last_integration_seconds=last_integration, target_mass=session.policy.gamma)


def service_requirement(session, ages, rotation, last_service, *, now, pose, budget,
                        last_reading=None, last_integration=None):
    cfg, shape = session.policy, session.instrument.lattice_shape
    if now-last_service < cfg.survey_interval_seconds - cfg.time_tolerance_seconds:
        return (), "not_due"
    blocks = survey_blocks(shape, cfg.survey_height_fov)
    order = sorted(range(len(blocks)), key=lambda i: (float(np.min(ages[blocks[i][0]:blocks[i][1], blocks[i][2]])),
                                                    (i-rotation) % len(blocks)))
    for i in order:
        lo, hi, col = blocks[i]
        cells = tuple(row*shape[1]+col for row in range(lo, hi))
        plan = compile_for(session, cell_mask(shape, cells), "service-check", now, pose, last_reading, last_integration)
        if plan.duration_seconds <= budget + cfg.time_tolerance_seconds:
            return cells, "required"
    return (), "unmet_no_reachable_block"


def repeated_pattern(candidate, session, ages, rotation, last_service, required, *, horizon,
                     last_reading=None, last_integration=None):
    cfg, shape = session.policy, session.instrument.lattice_shape
    ages = ages.copy()
    plan = candidate.plan
    end = plan.started_at_seconds+horizon
    result = []
    while plan.samples:
        receipt = receipt_time(plan, cfg)
        result.append(ForecastSweep(plan, receipt))
        for sample in plan.samples:
            iy, ix = np.minimum((np.asarray(sample.position_yx)*shape).astype(int), np.asarray(shape)-1)
            ages[iy, ix] = sample.acquired_at_seconds
        if required:
            last_service = receipt
        rotation = (rotation+1) % len(survey_blocks(shape, cfg.survey_height_fov))
        if receipt >= end - cfg.time_tolerance_seconds:
            break
        last_reading, last_integration = plan.samples[-1].acquired_at_seconds, plan.samples[-1].integration_seconds
        required, _ = service_requirement(session, ages, rotation, last_service, now=receipt,
            pose=plan.final_position_yx, budget=end-receipt, last_reading=last_reading, last_integration=last_integration)
        mask = cell_mask(shape, candidate.core_cells)
        mask.ravel()[list(required)] = True
        next_plan = compile_for(session, mask, f"forecast-repeat-{len(result):04d}", receipt, plan.final_position_yx, last_reading, last_integration)
        if (len(next_plan.samples) < cfg.minimum_samples or
                receipt_time(next_plan, cfg) > end + cfg.time_tolerance_seconds):
            break
        plan = next_plan
    return tuple(result)


def candidate_paths(belief, session, request, wind, ages, *, scan_id, now, pose, rotation,
                    last_survey_completed, startup_complete, last_reading_at_seconds=None,
                    last_integration_seconds=None):
    cfg, shape = session.policy, session.instrument.lattice_shape
    budget = request.max_duration_seconds
    required, service = service_requirement(session, ages, rotation, last_survey_completed,
        now=now, pose=pose, budget=budget, last_reading=last_reading_at_seconds, last_integration=last_integration_seconds)
    midpoint = belief.clone(); midpoint.advance(now+budget/2, wind, session.source_position)
    if hasattr(midpoint, "distribution_on"):
        density = midpoint.distribution_on(shape).expected_excess(cfg.reportable_threshold)
    else:
        mu, var = midpoint.moments_on(shape)
        density = positive_expectation(mu, var, cfg.reportable_threshold)
    mass = density * np.asarray(session.physical_quadrature_weights).reshape(shape)
    gain = covariance_proposal(midpoint, session)
    indices = np.arange(math.prod(shape))
    orders = [("expected_excess", np.argsort(-mass.ravel(), kind="stable")),
              ("covariance_gain_proxy", np.argsort(-gain.ravel(), kind="stable")),
              ("visit_age", np.lexsort(((indices-rotation) % len(indices), ages.ravel())))]
    candidates, keys = [], set()

    def compile(mask):
        return compile_for(session, mask, scan_id, now, pose, last_reading_at_seconds, last_integration_seconds)

    def add(name, core, plan):
        if len(plan.samples) < cfg.minimum_samples or plan.duration_seconds > budget + cfg.time_tolerance_seconds:
            return
        key = (plan.swept_cells, tuple((s.position_yx, s.acquired_at_seconds) for s in plan.samples))
        if key not in keys:
            keys.add(key)
            candidates.append(CandidatePath(name, tuple(np.flatnonzero(core).tolist()), plan))

    for name, order in orders:
        # Evaluate each prefix once; never assume route length is monotonic.
        ceilings = [budget*f for f in (.25, .5, .75, 1.)]
        best = [None]*4
        core = np.zeros(shape, bool)
        for index in order:
            core.ravel()[index] = True
            mask = core.copy(); mask.ravel()[list(required)] = True
            occupied = np.flatnonzero(mask.any(axis=0))
            count = sum(int(np.flatnonzero(mask[:, j])[-1]-np.flatnonzero(mask[:, j])[0]+1) for j in occupied)
            if count/session.instrument.acquisition.cadence_hz > budget + cfg.time_tolerance_seconds:
                break
            plan = compile(mask)
            if len(plan.samples) >= cfg.minimum_samples:
                for k, ceiling in enumerate(ceilings):
                    if plan.duration_seconds <= ceiling + cfg.time_tolerance_seconds:
                        best[k] = (core.copy(), plan)
        for k, value in enumerate(best):
            if value is not None:
                add(f"{name}:{(.25,.5,.75,1.)[k]:g}", *value)
    legacy = plan_scan_v2(belief, session.legacy_session(), request, wind, ages,
        scan_id=scan_id, now=now, pose=pose, rotation=rotation, startup_complete=startup_complete,
        last_survey_completed=last_survey_completed, last_reading_at_seconds=last_reading_at_seconds,
        last_integration_seconds=last_integration_seconds)
    core = cell_mask(shape, legacy.requested_cells)
    mask = core.copy(); mask.ravel()[list(required)] = True
    add("legacy_executable", core, compile(mask))
    points = grid_positions(shape).reshape(*shape, 2)
    y0, x0, y1, x1 = cfg.corridor_yxyx
    core = (points[...,0]>=y0)&(points[...,0]<=y1)&(points[...,1]>=x0)&(points[...,1]<=x1)
    mask = core.copy(); mask.ravel()[list(required)] = True
    add("declared_corridor", core, compile(mask))
    return tuple(candidates), required, service


def _plan_scan_v4(belief, session, request, wind, last_sampled, *, scan_id, now, pose, rotation,
                 startup_complete, last_survey_completed, last_reading_at_seconds=None, last_integration_seconds=None):
    cfg, shape = session.policy, session.instrument.lattice_shape
    common = dict(scan_id=scan_id, now=now, pose=pose, rotation=rotation, startup_complete=startup_complete,
                  last_survey_completed=last_survey_completed, last_reading_at_seconds=last_reading_at_seconds,
                  last_integration_seconds=last_integration_seconds)
    budget = request.max_duration_seconds
    if session.startup_mode == "full_scene" and not startup_complete and budget > 0:
        old = plan_scan_v2(belief, session.legacy_session(), request, wind, last_sampled, **common)
        return risk_plan(old, policy_id=cfg.policy_id, lookahead_seconds=budget,
                         service_required_cells=old.swept_cells, service_status="startup_actual_visits")
    if budget <= 0:
        candidates, required, service = (), (), "not_due"
    else:
        candidates, required, service = candidate_paths(belief, session, request, wind, last_sampled, **common)
    seed = int.from_bytes(hashlib.sha256(f"{cfg.random_seed}:{scan_id}:{now:.17g}".encode()).digest()[:8], "little")
    if not candidates:
        empty = compile_for(session, np.zeros(shape,bool), scan_id, now, pose, last_reading_at_seconds, last_integration_seconds)
        return risk_plan(empty, policy_id=cfg.policy_id, status="no_legal_action", reason="No complete legal candidate fits physical constraints",
                         lookahead_seconds=max(0.,budget), service_status=service, service_required_cells=required)
    scores, records = [], []
    for candidate in candidates:
        sweeps = repeated_pattern(candidate, session, last_sampled, rotation, last_survey_completed, required,
                                  horizon=budget, last_reading=last_reading_at_seconds, last_integration=last_integration_seconds)
        score = score_pattern(belief, session, wind, sweeps, budget, seed=seed)
        scores.append(score)
        records.append(RiskCandidate(candidate.name, candidate.core_cells, candidate.plan.requested_cells,
            candidate.plan.duration_seconds, score["risk"], score["within"], score["between"], score["standard_error"], len(sweeps)))
    idle = score_pattern(belief, session, wind, (), budget, seed=seed)["risk"]
    best = min(s["risk"] for s in scores)
    tolerance = max(cfg.risk_absolute_tolerance, cfg.risk_relative_tolerance*idle)
    selected = min((i for i,s in enumerate(scores) if s["risk"] <= best+tolerance),
                   key=lambda i:(candidates[i].plan.duration_seconds, candidates[i].plan.requested_cells))
    chosen = candidates[selected].plan
    predicted = belief.clone(); predicted.advance(chosen.completed_at_seconds, wind, session.source_position)
    if hasattr(predicted, "distribution_on"):
        density = predicted.distribution_on(shape).expected_excess(cfg.reportable_threshold)
    else:
        mu, var = predicted.moments_on(shape); density = positive_expectation(mu,var,cfg.reportable_threshold)
    mass = density*np.asarray(session.physical_quadrature_weights).reshape(shape)
    total = float(mass.sum())
    retained = coverage_mask(chosen, shape, session.instrument.capabilities())
    coverage = min(1.,float(mass[retained].sum()/total)) if total > cfg.mass_floor else None
    return risk_plan(chosen, policy_id=cfg.policy_id, status="value_selected", predicted_mass_recall=coverage,
        expected_signal_total=total, reason="Minimum predicted common-horizon map risk; mass coverage is diagnostic",
        termination="bounded_candidate_comparison", startup_complete=startup_complete,
        risk_candidates=tuple(records), selected_candidate=selected, lookahead_seconds=budget,
        expected_integrated_map_risk=scores[selected]["risk"], idle_integrated_map_risk=idle,
        risk_step_seconds=cfg.risk_step_seconds, history_count=scores[selected]["histories"], risk_sampling_seed=seed,
        service_required_cells=required, service_status=service, expected_delivery=cfg.expected_delivery,
        expected_delivery_delay_seconds=cfg.expected_delivery_delay_seconds,
        expected_completion_overrun_seconds=cfg.expected_completion_overrun_seconds,
        lookahead_wind_yx=wind.velocity_yx)


# Public-call replay evaluates exactly the same detached decision. Cache only
# immutable results, keyed by every estimator, physical and lifecycle input.
# This does not bypass evidence replay, plan comparison or causal validation.
_decision_cache = OrderedDict()


def plan_scan_v4(belief, session, request, wind, last_sampled, **lifecycle):
    lifecycle = {**lifecycle, "pose": tuple(lifecycle["pose"])}
    digest = hashlib.sha256()
    digest.update(np.asarray([belief.time_seconds], dtype=np.float64).tobytes())
    for component in getattr(belief, "components", (belief,)):
        digest.update(np.asarray(component.mean, dtype=np.float64).tobytes())
        digest.update(np.asarray(component.covariance, dtype=np.float64).tobytes())
    for name in ("offsets", "log_weights"):
        if hasattr(belief, name):
            digest.update(np.asarray(getattr(belief,name), dtype=np.float64).tobytes())
    digest.update(np.asarray(last_sampled, dtype=np.float64).tobytes())
    key = (session, request, wind, tuple(sorted(lifecycle.items())), digest.digest())
    if key in _decision_cache:
        _decision_cache.move_to_end(key)
        return _decision_cache[key]
    result = _plan_scan_v4(belief, session, request, wind, last_sampled, **lifecycle)
    _decision_cache[key] = result
    if len(_decision_cache) > 64:
        _decision_cache.popitem(last=False)
    return result
