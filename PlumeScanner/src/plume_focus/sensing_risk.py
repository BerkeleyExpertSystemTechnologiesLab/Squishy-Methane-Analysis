"""Detached physical-time Bayes-risk calculation for fixed repeated scan patterns.

Gaussian covariance is deterministic for a schedule. The mixture's expected
within-component contribution is analytic; only posterior component disagreement
requires possible-reading histories. No simulator, controller, truth or I/O.
"""

from dataclasses import dataclass, replace
from collections import OrderedDict
import copy
import hashlib
from functools import lru_cache
import math

import numpy as np
from scipy.sparse import csr_matrix, eye as sparse_eye, block_diag
from scipy.special import logsumexp

from plume_focus.signal_belief import grid_positions, interpolation_stencil, observation_weights, advection_stencil


@lru_cache(maxsize=16)
def projection_gram(latent_shape, output_shape):
    """G.T W G for exactly the evaluator's uniform bilinear spatial projection."""
    ids, weights = interpolation_stencil(grid_positions(output_shape), latent_shape)
    gram = np.zeros((math.prod(latent_shape),) * 2)
    for i in range(4):
        for j in range(4):
            np.add.at(gram, (ids[:, i], ids[:, j]), weights[:, i] * weights[:, j] / len(ids))
    gram.setflags(write=False)
    return gram


@lru_cache(maxsize=128)
def linear_transition(config, velocity, source, elapsed, transport_id="legacy_clipped_v1"):
    """Linear mean map reproducing GaussianSignalBelief.advance's time partition.

Keep this separate from runtime inference; reference tests guard drift. The
Historical callers retain their boundary convention; diagnostics opt in explicitly.
"""
    n = math.prod(config.grid_shape)
    a, b, noise = sparse_eye(n,format="csr"), np.zeros(n), np.zeros((n,n))
    if elapsed <= 1e-12:
        return a, b, noise
    shape = config.grid_shape
    speed = np.max(np.abs(velocity) * np.array(shape))
    limit = min(config.max_transition_seconds, .5 / max(speed, 1e-12),
                .45 / max(config.diffusivity * sum(x*x for x in shape), 1e-12))
    count = max(1, int(math.ceil(elapsed / limit)))
    dt = elapsed / count
    points = grid_positions(shape)
    ids, weights = advection_stencil(points - np.asarray(velocity) * dt, shape, transport_id)
    stencils = [(ids, weights)]
    if config.diffusivity:
        grid = np.arange(n).reshape(shape)
        offsets = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
        wy, wx = config.diffusivity * dt * shape[0]**2, config.diffusivity * dt * shape[1]**2
        heat_ids = np.stack([np.roll(grid, (dy, dx), axis=(0, 1)).ravel() for dy, dx in offsets], axis=1)
        heat = np.broadcast_to([1-2*wy-2*wx, wy, wy, wx, wx], heat_ids.shape).copy()
        y, x = np.indices(shape)
        for k, (dy, dx) in enumerate(offsets):
            heat[:, k] *= ((y-dy >= 0) & (y-dy < shape[0]) & (x-dx >= 0) & (x-dx < shape[1])).ravel()
        stencils.append((heat_ids, heat))
    emission = np.zeros(n) if source is None else config.source_rate * np.exp(
        -np.sum((points-source)**2, axis=1)/(2*config.source_sigma**2))
    step = sparse_eye(n,format="csr")
    for ids, weights in stencils:
        operator = csr_matrix((weights.ravel(), (np.repeat(np.arange(n),ids.shape[1]),ids.ravel())),shape=(n,n))
        step = operator @ step
    process = config.process_std_per_sqrt_second**2 * dt * np.exp(
        -np.sum((points[:,None]-points[None,:])**2,axis=-1)/(2*config.correlation_length**2))
    for _ in range(count):
        a = step @ a
        noise = (step @ (step @ noise).T).T + process
        for ids, weights in stencils:
            b = np.sum(b[ids] * weights, axis=1)
        b += emission * dt
    b.setflags(write=False); noise.setflags(write=False)
    return a, b, noise


def mean_transition(config, velocity, source, elapsed, transport_id="legacy_clipped_v1"):
    a,b,_ = linear_transition(config,velocity,source,elapsed,transport_id)
    return a.toarray(),b


@lru_cache(maxsize=8)
def combined_transition(config, velocities, source, elapsed, transport_id="legacy_clipped_v1"):
    operators = [linear_transition(config,velocity,source,elapsed,transport_id) for velocity in velocities]
    return (block_diag([a for a,_,_ in operators],format="csr"),
            np.stack([b for _,b,_ in operators]), np.stack([q for _,_,q in operators]))


@lru_cache(maxsize=8)
def projected_risk_operator(config, velocities, source, elapsed, output_shape, transport_id="legacy_clipped_v1"):
    """Trace(G A P A.T) = trace(A.T G A P), avoiding query-only dense P copies."""
    gram = csr_matrix(projection_gram(config.grid_shape, output_shape))
    n, count = math.prod(config.grid_shape), len(velocities)
    operator, _, process = combined_transition(config, velocities, source, elapsed, transport_id)
    projected = (operator.T @ block_diag([gram]*count, format="csr") @ operator).tocoo()
    component = projected.row//n
    i, j = projected.row % n, projected.col % n
    noise_risk = np.sum(process * gram.toarray()[None], axis=(1,2))
    return component, i, j, projected.data, noise_risk


@dataclass(frozen=True)
class ForecastSweep:
    plan: object
    receipt_at_seconds: float


def receipt_time(plan, config):
    return plan.completed_at_seconds + config.expected_completion_overrun_seconds + config.expected_delivery_delay_seconds


def covariance_proposal(belief, session):
    """Single-reading reduction proxy; aggregate-mixture use is proposal-only."""
    shape = session.instrument.lattice_shape
    gram = projection_gram(session.model.grid_shape, session.policy.risk_spatial_shape)
    footprint = session.instrument.capabilities().footprint
    h = np.stack([observation_weights(session.model.grid_shape, tuple(p), footprint) for p in grid_positions(shape)])
    cross = belief.covariance @ h.T
    noise = (session.model.sensor_noise_std / session.model.gain)**2
    denominator = np.sum(h.T * cross, axis=0) + noise
    return (np.sum(cross * (gram @ cross), axis=0) / denominator).reshape(shape)


def _score_pattern(belief, session, wind, sweeps, horizon, *, seed=0, include_history_samples=False):
    """Expected integrated map risk; all state and fake evidence remain detached."""
    cfg, model = session.policy, session.model
    transport_id = getattr(session, "transport_id", "legacy_clipped_v1")
    if getattr(belief, "transport_id", "legacy_clipped_v1") != transport_id:
        raise ValueError("Risk and inference transport identifiers differ.")
    start, end = belief.time_seconds, belief.time_seconds + horizon
    components = getattr(belief, "components", (belief,))
    covariance = np.stack([s.covariance for s in components])
    inference_time = belief.time_seconds
    prior = np.asarray(getattr(belief, "weights", np.ones(1)))
    offsets = getattr(belief, "offsets", np.zeros((1, 2)))
    winds = [replace(wind, velocity_yx=tuple(np.asarray(wind.velocity_yx) + offset)) for offset in offsets]
    velocities = tuple(w.velocity_yx for w in winds)
    count = len(components)
    gram = projection_gram(model.grid_shape, cfg.risk_spatial_shape)
    gi, gj = np.nonzero(gram)
    gv = gram[gi, gj]
    sparse_gram = csr_matrix(gram)
    histories = cfg.history_count if count > 1 else 0
    # Without hypothetical evidence all histories have identical component
    # moments and unchanged weights; one copy gives the exact disagreement.
    if histories and not sweeps:
        histories = 1
    means = np.repeat(np.stack([s.mean for s in components])[:, None, :], histories, axis=1) if histories else None
    initial_logs = getattr(belief, "log_weights", np.zeros(1))
    logs = np.repeat(np.asarray(initial_logs)[:, None], histories, axis=1) if histories else None
    rng = np.random.default_rng(seed)
    generating = np.minimum(np.searchsorted(np.cumsum(prior), rng.random(128)), count-1)[:histories]
    # Fixed maximum history dimension makes 32/64/128 nested, and scores do not
    # depend on the number/order of other candidate paths.
    reading_count = sum(len(sweep.plan.samples) for sweep in sweeps)
    normals = rng.standard_normal((reading_count, 128))[:, :histories]
    events = []
    for sweep in sweeps:
        plan, receipt = sweep.plan, sweep.receipt_at_seconds
        for sample in plan.samples:
            delivered = receipt if cfg.expected_delivery == "batch" else (
                sample.acquired_at_seconds + sample.integration_seconds/2 + cfg.expected_delivery_delay_seconds)
            events.append((delivered, 0, sample))
        events.append((receipt, 1, None))
    events.sort(key=lambda e: (e[0], e[1]))
    times = sorted(set([start, end] + [start + float(x) for x in np.arange(0, horizon, cfg.risk_step_seconds)]
                       + [t for t, _, _ in events if start <= t <= end]))
    within_total, between_total = 0., np.zeros(histories)
    event_index = reading_index = 0
    noise = (model.sensor_noise_std/model.gain)**2

    def advance(target):
        nonlocal means, covariance, inference_time
        # Canonicalize subtraction roundoff, far below the causal tolerance.
        elapsed = round(target-inference_time, 12)
        if elapsed < -1e-9:
            raise ValueError("Hypothetical evidence has an invalid causal sequence.")
        if elapsed > 1e-12:
            n = covariance.shape[-1]
            a, b, process = combined_transition(model, velocities, session.source_position, elapsed, transport_id)
            left = (a @ covariance.reshape(count*n,n)).reshape(count,n,n)
            covariance = (a @ left.transpose(0,2,1).reshape(count*n,n)).reshape(count,n,n).transpose(0,2,1)+process
            covariance = (covariance+covariance.transpose(0,2,1))*.5
            if means is not None:
                means = (a @ means.transpose(0,2,1).reshape(count*n,histories)).reshape(count,n,histories).transpose(0,2,1)+b[:,None,:]
        inference_time = target

    def quadratic(values):
        flat = values.reshape(-1, values.shape[-1])
        return np.sum(flat * (sparse_gram @ flat.T).T, axis=1).reshape(values.shape[:-1])

    for t, next_t in zip(times[:-1], times[1:]):
        while event_index < len(events) and events[event_index][0] <= t + 1e-10:
            at, kind, sample = events[event_index]
            if kind:
                advance(at)
            else:
                advance(sample.acquired_at_seconds)
                h = observation_weights(model.grid_shape, sample.position_yx, sample.footprint)
                crosses = covariance @ h
                variances = crosses @ h + noise
                if histories:
                    predicted = means @ h
                    y = predicted[generating, np.arange(histories)] + np.sqrt(variances[generating]) * normals[reading_index]
                    residual = y[None, :] - predicted
                    logs += -.5 * (np.log(2*np.pi*variances[:, None]) + residual**2/variances[:, None])
                    logs -= logsumexp(logs, axis=0)
                    means += (crosses/variances[:, None])[:, None, :] * residual[..., None]
                covariance -= crosses[:,:,None]*crosses[:,None,:]/variances[:,None,None]
                covariance = (covariance+covariance.transpose(0,2,1))*.5
                reading_index += 1
            event_index += 1
        elapsed = round(t-inference_time, 12)
        if elapsed > 1e-12:
            k, i, j, coefficients, process_risk = projected_risk_operator(
                model, velocities, session.source_position, elapsed, cfg.risk_spatial_shape, transport_id)
            within = np.sum(prior[k]*coefficients*covariance[k,i,j])+prior@process_risk
            if histories:
                a,b,_ = combined_transition(model,velocities,session.source_position,elapsed,transport_id)
                n = covariance.shape[-1]
                projected_means = (a @ means.transpose(0,2,1).reshape(count*n,histories)).reshape(count,n,histories).transpose(0,2,1)+b[:,None,:]
        else:
            within = np.sum(prior[:,None]*covariance[:,gi,gj]*gv[None])
            projected_means = means
        within_total += (next_t-t) * max(0., float(within))
        if histories:
            weights = np.exp(logs)
            average = np.sum(weights[..., None] * projected_means, axis=0)
            disagreement = np.sum(weights * quadratic(projected_means-average[None]), axis=0)
            between_total += (next_t-t)*np.maximum(disagreement, 0.)
    between = float(between_total.mean()) if histories else 0.
    result = {"risk": within_total+between, "within": within_total, "between": between,
            "standard_error": float(between_total.std(ddof=1)/math.sqrt(histories)) if histories > 1 else 0.,
            "hypothetical_sweeps": len(sweeps), "histories": histories}
    if include_history_samples:
        result["between_history_integrals"] = between_total.tolist()
    return result


_gaussian_risk_cache = OrderedDict()


def score_pattern(belief, session, wind, sweeps, horizon, *, seed=0, include_history_samples=False):
    """Exact reuse for fixed-Gaussian schedules, independent of observed values.

    Mixtures retain their original scorer. Include physical evidence/receipt
    times and full covariance, configuration, wind and transport identity.
    Returned dictionaries never alias cached results.
    """
    if hasattr(belief, "components"):
        return _score_pattern(belief,session,wind,sweeps,horizon,seed=seed,include_history_samples=include_history_samples)
    transport=getattr(session,"transport_id","legacy_clipped_v1")
    if belief.transport_id != transport:
        raise ValueError("Risk and inference transport identifiers differ.")
    physical=tuple((s.receipt_at_seconds,tuple((r.acquired_at_seconds,r.position_yx,r.footprint,r.integration_seconds)
        for r in s.plan.samples)) for s in sweeps)
    key=(session,wind,belief.time_seconds,float(horizon),physical,include_history_samples,
         hashlib.sha256(belief.covariance.tobytes()).digest())
    if key not in _gaussian_risk_cache:
        _gaussian_risk_cache[key]=_score_pattern(belief,session,wind,sweeps,horizon,
            seed=seed,include_history_samples=include_history_samples)
        if len(_gaussian_risk_cache)>128:
            _gaussian_risk_cache.popitem(last=False)
    _gaussian_risk_cache.move_to_end(key)
    return copy.deepcopy(_gaussian_risk_cache[key])
