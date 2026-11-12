"""Evaluator-only fixed-path nuisance diagnostic; never imported by the controller."""
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from plume_focus.signal_belief import grid_positions, observation_weights
from plume_focus.sensing_contracts import SensorReading
from plume_focus.source_amount_validation import (
    ClockAffineFilter, ClockDynamics, posterior_inputs,
)
from plume_focus.source_input import gaussian_source_input

METHODS = ("fixed", "joint", "correct_dynamics")


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dynamics_pair(model, velocity, half_life, tick, condition):
    """Only A/source_dt differ; initial and per-tick innovation covariance stay fixed."""
    factor, multiplier = condition["factor"], condition["multiplier"]
    if factor not in ("matched", "wind_speed", "half_life"):
        raise ValueError("Unknown nuisance factor")
    if not np.isfinite(multiplier) or multiplier <= 0 or (factor == "matched" and multiplier != 1):
        raise ValueError("Invalid nuisance multiplier")
    nominal = ClockDynamics(model, velocity, tick=tick, half_life=half_life)
    truth_velocity = tuple(v * (multiplier if factor == "wind_speed" else 1) for v in velocity)
    truth_half_life = half_life * (multiplier if factor == "half_life" else 1)
    truth = ClockDynamics(model, truth_velocity, tick=tick, half_life=truth_half_life)
    for key in ("Q", "noise_factor", "P0", "initial_factor"):
        setattr(truth, key, getattr(nominal, key).copy())
    return nominal, truth


def history_metadata(config):
    return [dict(scene=s["name"], scene_index=i, split=s["split"],
                 source_yx=s["source_yx"], seed=seed, true_amount=1.0)
            for i, s in enumerate(config["scenes"]) for seed in s["seeds"]]


def sensor_noise(seed, tick, position, std):
    if seed is None:
        return 0.0
    entropy = [seed, tick, *(int(round(p * 1e6)) for p in position)]
    return float(np.random.default_rng(np.random.SeedSequence(entropy)).normal(0, std))


def clone_filter(f):
    result = copy.copy(f)
    for key in ("m", "P", "B", "L", "K"):
        setattr(result, key, getattr(f, key).copy())
    return result


def amount_diagnostics(posterior, support=(.25, 1.75)):
    """Marginal amount diagnostics of the actual finite quadrature mixture."""
    nodes = posterior["nodes"].ravel()
    weights = (posterior["w"][:, None] * posterior["conditional"]).ravel()
    order = np.argsort(nodes)
    cumulative = np.cumsum(weights[order])
    interval = [float(nodes[order[min(np.searchsorted(cumulative, q), len(order) - 1)]])
                for q in (.025, .975)]
    mean = float(posterior["w"] @ posterior["a1"])
    return dict(amount_mean=mean,
                amount_std=float(np.sqrt(max(0., posterior["w"] @ posterior["a2"] - mean**2))),
                amount_interval95=interval,
                true_amount_inside95=bool(interval[0] <= 1. <= interval[1]),
                amount_edge_mass=float(weights[(nodes <= support[0] + .05) |
                                               (nodes >= support[1] - .05)].sum()))


class NuisanceEnvironment:
    """Immutable evaluator-owned full-clock truth and common scalar evidence."""
    def __init__(self, dynamics, run, config, record, output):
        self.d = dynamics
        self.config = config
        self.metadata = history_metadata(config)
        self.seeds = sorted({m["seed"] for m in self.metadata if m["seed"] is not None})
        self.seed_ids = {s: i for i, s in enumerate(self.seeds)}
        self.emissions = gaussian_source_input(
            dynamics.model.grid_shape, [s["source_yx"] for s in config["scenes"]],
            source_sigma=dynamics.model.source_sigma,
            total_rate=config["reference_total_per_second"], formulation_id=config["source_input_id"])
        output = Path(output)
        output.mkdir()
        self.output = output
        end = dynamics.tick_of(config["fixed_duration_seconds"] + max(config["forecast_horizons_seconds"]))
        n = len(dynamics.P0)
        self.deterministic = np.lib.format.open_memmap(
            output / "deterministic.npy", mode="w+", dtype="float64",
            shape=(end + 1, len(config["scenes"]), n))
        self.noise = np.lib.format.open_memmap(
            output / "noise.npy", mode="w+", dtype="float64", shape=(end + 1, len(self.seeds), n))
        self.deterministic[0] = dynamics.model.initial_mean
        rngs = [np.random.default_rng(np.random.SeedSequence([s, 2])) for s in self.seeds]
        self.noise[0] = np.stack([
            dynamics.initial_factor @ np.random.default_rng(np.random.SeedSequence([s, 1])).standard_normal(n)
            for s in self.seeds])
        for tick in range(1, end + 1):
            self.deterministic[tick] = dynamics.means(self.deterministic[tick - 1], self.emissions)
            z = np.stack([rng.standard_normal(n) for rng in rngs])
            self.noise[tick] = (dynamics.A @ self.noise[tick - 1].T).T + z @ dynamics.noise_factor.T
        self.deterministic.flush()
        self.noise.flush()
        self.schedule = []
        values = []
        for event in record["events"]:
            if event["operation"] != "ingest_readings":
                continue
            if len(event["readings"]) != 1:
                raise ValueError("Expected singleton chronological readings")
            r = SensorReading.from_dict(event["readings"][0])
            if r.available_at_seconds > event["at_seconds"] + 1e-9:
                raise ValueError("Reading is not available at delivery")
            at = dynamics.tick_of(r.acquired_at_seconds)
            h = observation_weights(dynamics.model.grid_shape, r.position_yx, r.footprint)
            y = self.truth(at) @ h * run.model.gain + run.model.offset
            y += [sensor_noise(m["seed"], at, r.position_yx, run.model.sensor_noise_std)
                  for m in self.metadata]
            values.append(y)
            self.schedule.append(r)
        self.readings = np.asarray(values)
        np.save(output / "readings.npy", self.readings)
        np.save(output / "source-inputs.npy", self.emissions)
        save_json(output / "schedule.json", [r.to_dict() for r in self.schedule])
        self.hashes = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
        save_json(output / "environment.json", dict(
            metadata=self.metadata, seeds=self.seeds, hashes=self.hashes,
            velocity_yx=list(dynamics.velocity), tick_seconds=dynamics.tick,
            source_dt=dynamics.source_dt,
            input_totals=self.emissions.mean(axis=1).tolist(),
            noise_convention=config["noise_convention"]))
        for array in (self.deterministic, self.noise, self.readings, self.emissions):
            array.flags.writeable = False

    def truth(self, tick, indices=None):
        indices = range(len(self.metadata)) if indices is None else indices
        result = []
        for i in indices:
            m = self.metadata[i]
            value = self.deterministic[tick, m["scene_index"]].copy()
            if m["seed"] is not None:
                value += self.noise[tick, self.seed_ids[m["seed"]]]
            result.append(value)
        return np.asarray(result)


def run_level(nominal, truth_dynamics, environment, run, config, level, output, indices=None):
    """Paired streaming filters with full-time causal scoring and frozen evidence."""
    output = Path(output)
    output.mkdir()
    started = time.monotonic()
    indices = list(range(len(environment.metadata))) if indices is None else list(indices)
    metas = [environment.metadata[i] for i in indices]
    nh, n = len(metas), len(nominal.P0)
    source_order, amount_order = level
    sources = grid_positions((source_order, source_order))
    # Finer fallback grids use a disk-backed table to bound resident construction memory.
    if source_order > 257:
        emissions = np.lib.format.open_memmap(output / "source-table.npy", mode="w+",
                                              dtype="float64", shape=(len(sources), n))
        for start in range(0, len(sources), 4096):
            emissions[start:start + 4096] = gaussian_source_input(
                nominal.model.grid_shape, sources[start:start + 4096],
                source_sigma=nominal.model.source_sigma, total_rate=config["reference_total_per_second"],
                formulation_id=config["source_input_id"])
        emissions.flush()
    else:
        emissions = gaussian_source_input(
            nominal.model.grid_shape, sources, source_sigma=nominal.model.source_sigma,
            total_rate=config["reference_total_per_second"], formulation_id=config["source_input_id"])
    f = ClockAffineFilter(nominal, emissions, nh)
    oracle = ClockAffineFilter(truth_dynamics, emissions, nh)
    means = np.full((nh, 3, n), nominal.model.initial_mean)
    inputs = np.zeros_like(means)
    current = 0
    end = nominal.tick_of(config["fixed_duration_seconds"])
    errors = np.zeros((end, nh, 3))
    snapshots, diagnostics = {}, []
    maxbound = 0.
    posteriors = None

    def one_posterior(i):
        return (f.posterior(i, 1., amount_order, analytic_moments=True),
                f.posterior(i, None, amount_order, analytic_moments=True),
                oracle.posterior(i, 1., amount_order, analytic_moments=True))

    with ThreadPoolExecutor(max_workers=2) as pool:
        def refresh():
            nonlocal means, inputs, posteriors, maxbound
            posteriors = list(pool.map(one_posterior, range(nh)))
            maxbound = max(maxbound, max(p["omitted_bound"] for ps in posteriors for p in ps))
            inputs = posterior_inputs(emissions, [p for ps in posteriors for p in ps]).reshape(nh, 3, n)
            means[:, :2] = f.m[:, None] + inputs[:, :2] @ f.B.T
            means[:, 2] = oracle.m + inputs[:, 2] @ oracle.B.T

        def snapshot(tick):
            ff, oo = clone_filter(f), clone_filter(oracle)
            ff.advance(tick)
            oo.advance(tick)
            reference = environment.truth(tick, indices)
            for i, meta in enumerate(metas):
                positions = config["global_footprints_yx"] + [meta["source_yx"],
                    [meta["source_yx"][0], min(1., meta["source_yx"][1] + .1)]]
                H = np.stack([observation_weights(nominal.model.grid_shape, p, run.capabilities.footprint)
                              for p in positions])
                for method, model, fixed in zip(METHODS, (ff, ff, oo), (1., None, 1.)):
                    p = model.posterior(i, fixed, amount_order, components=True)
                    v = model.validation(i, p, H, H @ reference[i])
                    location = p["w"] @ sources[p["ids"]]
                    row = dict(history=indices[i], method=method, time=tick * nominal.tick,
                               source_mean_yx=location.tolist(),
                               source_error_fov=float(np.linalg.norm(location - meta["source_yx"])),
                               positions_yx=positions, pit=v["pit"].tolist(),
                               inside95=((v["pit"] >= .025) & (v["pit"] <= .975)).tolist(),
                               prediction_mean=v["mean"].tolist(), prediction_variance=v["variance"].tolist(),
                               squared_error=v["squared_error"].tolist(),
                               **amount_diagnostics(p, config["amount_support"]))
                    diagnostics.append(row)
            snapshots[str(tick * nominal.tick)] = means.copy()

        def advance_means():
            nonlocal means
            means[:, :2] = nominal.means(means[:, :2].reshape(nh * 2, n),
                                         inputs[:, :2].reshape(nh * 2, n)).reshape(nh, 2, n)
            means[:, 2] = truth_dynamics.means(means[:, 2], inputs[:, 2])

        def score_until(target):
            nonlocal current
            while current < target:
                if current * nominal.tick in config["uncertainty_times_seconds"]:
                    snapshot(current)
                errors[current] = np.mean((means - environment.truth(current, indices)[:, None])**2, axis=2)
                advance_means()
                current += 1

        refresh()
        for ri, reading in enumerate(environment.schedule):
            acquired = nominal.tick_of(reading.acquired_at_seconds)
            available = nominal.tick_of(reading.available_at_seconds)
            if available > end:
                raise ValueError("Schedule extends past monitoring horizon")
            score_until(available)
            h = observation_weights(nominal.model.grid_shape, reading.position_yx, reading.footprint)
            for model in (f, oracle):
                model.advance(acquired)
                model.condition(environment.readings[ri, indices], h)
                model.advance(available)
            refresh()
            if (ri + 1) % 200 == 0:
                print(json.dumps(dict(level=level, readings=ri + 1,
                                      output=str(output), elapsed=time.monotonic() - started)), flush=True)
        score_until(end)
        f.advance(end)
        oracle.advance(end)
        snapshot(end)
        for name, model in (("nominal", f), ("correct", oracle)):
            np.savez_compressed(output / f"final-{name}-filter.npz",
                                **{key: getattr(model, key) for key in ("m", "P", "B", "L", "K")})
        forecasts = []
        for horizon in config["forecast_horizons_seconds"]:
            target = nominal.tick_of(config["fixed_duration_seconds"] + horizon)
            while current < target:
                advance_means()
                current += 1
            reference = environment.truth(target, indices)
            forecasts.append(np.mean((means - reference[:, None])**2, axis=2).tolist())
            snapshot(target)
    np.savez_compressed(output / "fields.npz", **snapshots)
    np.save(output / "errors.npy", errors)
    result = dict(indices=indices, metadata=metas, level=list(level), methods=list(METHODS),
                  JT=errors.mean(axis=0).tolist(), forecasts=forecasts,
                  JT_subsample_025=errors[::2].mean(axis=0).tolist(),
                  JT_subsample_050=errors[::4].mean(axis=0).tolist(), diagnostics=diagnostics,
                  max_omitted_bound=maxbound, wall_seconds=time.monotonic() - started,
                  evidence_sha256=sha256(environment.output / "readings.npy"))
    save_json(output / "results.json", result)
    return result


def refinement_checks(config, coarse, fine):
    """History-level gates including interval endpoints and membership, not only means."""
    tol = config["refinement"]
    rows = []
    coarse_ids = {hi: i for i, hi in enumerate(coarse["indices"])}
    fine_ids = {hi: i for i, hi in enumerate(fine["indices"])}
    dc = {(r["history"], r["method"], r["time"]): r for r in coarse["diagnostics"]}
    df = {(r["history"], r["method"], r["time"]): r for r in fine["diagnostics"]}
    for hi in fine["indices"]:
        ci, fi = coarse_ids[hi], fine_ids[hi]
        for mi, method in enumerate(METHODS):
            failures, maxima = [], dict(source_shift=0., amount_shift=0., endpoint_shift=0., pit_shift=0.)
            for label, a, b in [("JT", coarse["JT"], fine["JT"])] + [
                (f"forecast_{j}", a, b) for j, (a, b) in enumerate(zip(coarse["forecasts"], fine["forecasts"]))]:
                if abs(a[ci][mi] - b[fi][mi]) > max(tol["mse_absolute"], tol["mse_relative"] * b[fi][mi]):
                    failures.append(label)
            for t in config["uncertainty_times_seconds"]:
                a, b = dc[hi, method, t], df[hi, method, t]
                values = dict(source_shift=float(np.linalg.norm(np.array(a["source_mean_yx"]) - b["source_mean_yx"])),
                              amount_shift=abs(a["amount_mean"] - b["amount_mean"]),
                              endpoint_shift=float(np.max(np.abs(np.array(a["amount_interval95"]) - b["amount_interval95"]))),
                              pit_shift=float(np.max(np.abs(np.array(a["pit"]) - b["pit"]))))
                for key, value in values.items():
                    maxima[key] = max(maxima[key], value)
                if a["inside95"] != b["inside95"] or a["true_amount_inside95"] != b["true_amount_inside95"]:
                    failures.append(f"membership_{t}")
            for key, limit in (("source_shift", tol["source_shift_fov"]),
                               ("amount_shift", tol["amount_shift"]),
                               ("endpoint_shift", tol["amount_shift"]), ("pit_shift", tol["pit_absolute"])):
                if maxima[key] > limit:
                    failures.append(key)
            rows.append(dict(history=hi, method=method, passed=not failures, failures=failures, **maxima))
    return dict(passed=all(r["passed"] for r in rows), checks=rows)


def select_histories(level_results):
    """Keep each history's finest completed result, with exact level provenance."""
    selected = {}
    for result in level_results:
        for local, hi in enumerate(result["indices"]):
            selected[hi] = dict(history=hi, metadata=result["metadata"][local], level=result["level"],
                                JT=result["JT"][local],
                                forecasts=[v[local] for v in result["forecasts"]],
                                diagnostics=[r for r in result["diagnostics"] if r["history"] == hi])
    return [selected[hi] for hi in sorted(selected)]
