"""Truth-free active discrimination and a prequential two-source discriminator.

No simulator, source object, seed, label, filesystem or future observation enters
route selection. Values are dimensionless toy concentration-path signal.
"""
from dataclasses import dataclass
from itertools import product
import time

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import lsq_linear
from scipy.special import logsumexp

from plume_focus.c2_r2_mechanics import compile_directed_columns, route_record
from plume_focus.signal_belief import footprint_points


def pose_at(plan, at):
    """Interpolate only the proposed physical path, without simulator access."""
    for segment in plan.segments:
        if at <= segment.completed_at_seconds:
            dt = segment.completed_at_seconds - segment.started_at_seconds
            fraction = np.clip((at - segment.started_at_seconds) / max(dt, 1e-15), 0, 1)
            return tuple((1 - fraction) * np.asarray(segment.start_yx) + fraction * np.asarray(segment.end_yx))
    return plan.final_position_yx


def plume(points, source, wind=0.):
    """Declared steady 2-D toy kernel; additive signals, no emission-rate units."""
    points = np.asarray(points)
    dy, dx = points[..., 0] - source[0], points[..., 1] - source[1]
    width = .065 + .14 * np.maximum(dx, 0)
    return np.exp(-.5 * ((dy - wind * dx / .5) / width)**2) * np.exp(-np.maximum(dx, 0) / .65) * np.exp(-.5 * (np.minimum(dx, 0) / .045)**2)


def exposure_response(plan, source, wind=0.):
    """Same moving footprint and Simpson exposure quadrature as FakeScanner."""
    values = []
    for s in plan.samples:
        q = [np.mean(plume(footprint_points(pose_at(plan, s.acquired_at_seconds + a * s.integration_seconds), s.footprint), source, wind)) for a in (-.5, 0., .5)]
        values.append((q[0] + 4 * q[1] + q[2]) / 6)
    return np.asarray(values)


def reading_response(reading, source, wind=0.):
    # Natural-history target is a midpoint approximation. Route forecasts use
    # full moving-exposure quadrature above; this small history mismatch stays.
    return float(np.mean(plume(footprint_points(reading.position_yx, reading.footprint), source, wind)))


def covariance(n, wind, cfg):
    """Each wind component has its own declared joint predictive covariance."""
    tau = cfg['common_bias_std'] * (1 + cfg['wind_bias_covariance_multiplier'] * abs(wind))
    return np.eye(n) * cfg['noise_std']**2 + np.ones((n, n)) * tau**2


def bounded_discrimination(s, g, cov, bounds):
    """One shared, bounded nuisance vector for the whole delivered prefix."""
    if len(s) == 0:
        return 0.
    chol = np.linalg.cholesky(cov)
    sw = solve_triangular(chol, s, lower=True)
    gw = solve_triangular(chol, g, lower=True)
    fit = lsq_linear(gw, sw, bounds=tuple(np.asarray(bounds)), tol=1e-10)
    if not fit.success:
        raise RuntimeError('Bounded nuisance fit failed')
    return float(np.sum((sw - gw @ fit.x)**2))


def time_average(times, values, t0, horizon):
    """Left-continuous delivered evidence, zero before first delivery."""
    times = np.asarray(times, float)
    values = np.asarray(values, float)
    if np.any(np.diff(times) < 0) or len(times) != len(values):
        raise ValueError('Expected chronological prefixes')
    keep = (times >= t0) & (times <= t0 + horizon)
    ts, vs = times[keep], values[keep]
    return float(np.sum(vs * np.diff(np.r_[ts, t0 + horizon])) / horizon) if len(ts) else 0.


@dataclass(frozen=True)
class VisibleState:
    t0: float
    pose: tuple
    last_acquired: float
    last_integration: float
    suspect: tuple
    supplied_wind: float = 0.


def candidates(state, cfg):
    return tuple((float(np.clip(state.suspect[0] + d, .05, .95)), float(np.clip(state.suspect[1] - cfg['source_upwind_offset'], .05, .95))) for d in cfg['candidate_y_offsets'])


def compile_candidates(state, profile, cfg, budget):
    """Fixed reference is at old-source row; no truth-dependent reference."""
    rows, cols = profile.lattice_shape
    sr = min(rows - 1, int(state.suspect[0] * rows))
    sc = min(cols - 1, int(state.suspect[1] * cols))
    rc = max(0, sc - 2)
    if rc == sc:
        rc = min(cols - 1, sc + 2)
    rr = min(rows - 1, int(cfg['old_source_yx'][0] * rows))
    masks = {}
    pair = np.zeros((rows, cols), bool)
    pair[max(0, sr - 1):min(rows, sr + 2), sc] = True
    pair[max(0, rr - 1):min(rows, rr + 2), rc] = True
    masks['suspect_reference'] = (pair, (sc, rc))
    masks['reference_suspect'] = (pair, (rc, sc))
    only = pair.copy(); only[:, rc] = False
    masks['suspect_only'] = (only, (sc,))
    cross = np.zeros_like(pair)
    order = tuple(range(max(0, sc - 1), min(cols, sc + 2)))
    cross[max(0, sr - 1):min(rows, sr + 2), list(order)] = True
    masks['crosscut'] = (cross, order)
    result = {}
    for name in cfg['route_order']:
        mask, order = masks[name]
        # Fixed upward sweep; routes differ in support/order, never in readings.
        down = (False,) * len(order)
        kw = dict(scan_id='rescan-' + name, started_at_seconds=state.t0,
                  current_position=state.pose, last_reading_at_seconds=state.last_acquired,
                  last_integration_seconds=state.last_integration)
        plan = compile_directed_columns(mask, profile, column_order=order, directions_down=down, **kw)
        receipt = plan.completed_at_seconds + cfg['delivery_delay_seconds']
        if receipt <= state.t0 + min(budget, cfg['coverage_deadline_after_t0_seconds']) + 1e-9:
            result[name] = (plan, route_record(plan, mask, profile, kw, order, down))
    return result


def route_scores(plan, state, cfg, *, endpoint_only=False):
    times = np.array([s.acquired_at_seconds + s.integration_seconds / 2 + cfg['delivery_delay_seconds'] for s in plan.samples])
    # Equal frozen weights over candidate source locations and nuisance wind
    # components. A different wind is explicitly whitened by its own covariance.
    by_model = []
    for source, wind in product(candidates(state, cfg), cfg['wind_biases']):
        s = exposure_response(plan, source, state.supplied_wind + wind)
        old = exposure_response(plan, cfg['old_source_yx'], state.supplied_wind + wind)
        eps = .001
        derivative = (exposure_response(plan, cfg['old_source_yx'], state.supplied_wind + wind + eps) - exposure_response(plan, cfg['old_source_yx'], state.supplied_wind + wind - eps)) / (2 * eps)
        g = np.column_stack((old, derivative))
        prefixes = [len(s)] if endpoint_only else range(1, len(s) + 1)
        by_model.append([bounded_discrimination(s[:n], g[:n], covariance(n, wind, cfg), cfg['nuisance_bounds']) for n in prefixes])
    prefix = np.mean(by_model, axis=0)
    return dict(endpoint=float(prefix[-1]), early=None if endpoint_only else time_average(times, prefix, state.t0, cfg['common_window_seconds']),
                delivery_times=times.tolist(), prefix_d=prefix.tolist())


def choose_routes(routes, state, cfg, timings=None):
    start = time.perf_counter()
    endpoint = {name: route_scores(p, state, cfg, endpoint_only=True)['endpoint'] for name, (p, _) in routes.items()}
    end_b = time.perf_counter()
    scores = {name: route_scores(p, state, cfg) for name, (p, _) in routes.items()}
    end_c = time.perf_counter()
    if timings is not None:
        timings.update(A=0., B=end_b - start, C=end_c - end_b)
    if not routes:
        return {}, scores
    # Dict order is frozen; exact ties go to the first legal route.
    chosen = {'A': 'suspect_reference' if 'suspect_reference' in routes else None,
              'B': max(endpoint, key=endpoint.get),
              'C': max(scores, key=lambda n: scores[n]['early'])}
    return chosen, scores


class PredictiveBank:
    """Finite H0/H1 mixture with per-component shared Gaussian residual offset.

    H1 means OLD + NEW simultaneously. Each H1 old-source nuisance component
    exists in H0, with identical nuisance prior and sensor assumptions.
    Predictive likelihood is evaluated BEFORE updating weights or offset state.
    """
    def __init__(self, cfg, locations):
        self.cfg = cfg
        self.components = []
        for h in (0, 1):
            for strength, wind in product(cfg['old_strengths'], cfg['wind_biases']):
                for source, amp in (product(locations, cfg['new_strengths']) if h else [(None, 0.)]):
                    self.components.append((h, strength, wind, source, amp))
        self.h = np.array([c[0] for c in self.components])
        self.logw = np.array([-np.log(np.sum(self.h == h)) for h in self.h])
        self.bias = np.zeros(len(self.h))
        self.var = np.array([covariance(1, c[2], cfg)[0, 0] - cfg['noise_std']**2 for c in self.components])
        self.evidence = 0.
        self.seen = set()
        self.audit = []

    def means(self, old_response, new_response):
        return np.array([strength * old_response[wind] + (amp * new_response[(source, wind)] if h else 0.) for h, strength, wind, source, amp in self.components])

    def update(self, reading_id, value, means, *, count=True):
        if reading_id in self.seen:
            raise ValueError('Duplicate evidence')
        oldw, oldbias = self.logw.copy(), self.bias.copy()
        variance = self.var + self.cfg['noise_std']**2
        residual = value - means - oldbias
        loglike = -.5 * (np.log(2 * np.pi * variance) + residual**2 / variance)
        predictive = [float(logsumexp(oldw[self.h == h] + loglike[self.h == h])) for h in (0, 1)]
        if count:
            self.evidence += predictive[1] - predictive[0]
        for h in (0, 1):
            self.logw[self.h == h] = oldw[self.h == h] + loglike[self.h == h] - predictive[h]
        gain = self.var / variance
        self.bias += gain * residual
        self.var *= 1 - gain
        self.seen.add(reading_id)
        self.audit.append(dict(reading_id=reading_id, log_predictive_h0=predictive[0], log_predictive_h1=predictive[1],
                               counted=count, cumulative_log_bf=self.evidence, order='predict_then_assimilate'))
        return self.evidence

    def map_mean(self, points):
        means = np.array([strength * plume(points, self.cfg['old_source_yx'], wind) +
                          (amp * plume(points, source, wind) if h else 0.) + b
                          for (h, strength, wind, source, amp), b in zip(self.components, self.bias)])
        # Fixed equal H priors; log BF is only new evidence since t0.
        logp = np.where(self.h == 1, min(self.evidence, 700.), 0.) + self.logw
        w = np.exp(logp - logsumexp(logp))
        return w @ means
