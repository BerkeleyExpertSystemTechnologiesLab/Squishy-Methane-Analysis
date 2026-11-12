"""Verify saved evidence and replay final filter states of the nuisance study."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil

import numpy as np

from plume_focus.full_fov_execution import decode_config
from plume_focus.signal_belief import grid_positions, observation_weights
from plume_focus.sensing_contracts import SensorReading
from plume_focus.source_amount_validation import ClockAffineFilter
from plume_focus.source_input import gaussian_source_input
from plume_focus.source_amount_nuisance import dynamics_pair, sensor_noise, sha256, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--conditions", nargs="+", help="Audit a completed subset while other conditions execute")
    args = parser.parse_args()
    cfg = json.loads((args.study / "config.json").read_text())
    conditions = cfg["conditions"]
    if args.conditions:
        known = {c["name"] for c in conditions}
        if not set(args.conditions) <= known:
            raise ValueError("Unknown requested condition")
        conditions = [c for c in conditions if c["name"] in args.conditions]
    run = decode_config(json.loads((args.study / Path(cfg["baseline_config"]).name).read_text()))
    model = replace(run.model, grid_shape=tuple(cfg["grid_shape"]))
    args.output.mkdir(parents=True, exist_ok=False)
    checks = []
    for condition in conditions:
        root = args.study / condition["name"]
        hashes = json.loads((root / "sha256.json").read_text())
        hash_failures = [name for name, digest in hashes.items() if sha256(root / name) != digest]
        nominal, truth = dynamics_pair(model, run.wind.velocity_yx, run.session.half_life_seconds,
                                       cfg["clock_tick_seconds"], condition)
        selected = json.loads((root / "selected-results.json").read_text())
        env = root / "environment"
        einfo = json.loads((env / "environment.json").read_text())
        det, noise = [np.load(env / f"{name}.npy", mmap_mode="r") for name in ("deterministic", "noise")]
        readings = np.load(env / "readings.npy")
        schedule = [SensorReading.from_dict(r) for r in json.loads((env / "schedule.json").read_text())]
        def field(tick, hi):
            meta = einfo["metadata"][hi]
            value = det[tick, meta["scene_index"]].copy()
            if meta["seed"] is not None:
                value += noise[tick, einfo["seeds"].index(meta["seed"])]
            return value
        evidence_error = 0.
        for ri, reading in enumerate(schedule):
            h = observation_weights(model.grid_shape, reading.position_yx, reading.footprint)
            tick = truth.tick_of(reading.acquired_at_seconds)
            for hi, meta in enumerate(einfo["metadata"]):
                expected = field(tick, hi) @ h * model.gain + model.offset
                expected += sensor_noise(meta["seed"], tick, reading.position_yx, model.sensor_noise_std)
                evidence_error = max(evidence_error, abs(float(expected - readings[ri, hi])))
        replay_error, map_error, matched_error, score_error = 0., 0., 0., 0.
        for level in sorted({tuple(r["level"]) for r in selected}):
            folder = root / f"source-{level[0]}-amount-{level[1]}"
            result = json.loads((folder / "results.json").read_text())
            ids = result["indices"]
            if (folder / "source-table.npy").exists():
                E = np.load(folder / "source-table.npy", mmap_mode="r")
            else:
                E = gaussian_source_input(model.grid_shape, grid_positions((level[0], level[0])),
                    source_sigma=model.source_sigma, total_rate=cfg["reference_total_per_second"],
                    formulation_id=cfg["source_input_id"])
            filters = [ClockAffineFilter(d, E, len(ids)) for d in (nominal, truth)]
            for ri, reading in enumerate(schedule):
                h = observation_weights(model.grid_shape, reading.position_yx, reading.footprint)
                for f in filters:
                    f.advance(nominal.tick_of(reading.acquired_at_seconds))
                    f.condition(readings[ri, ids], h)
                    f.advance(nominal.tick_of(reading.available_at_seconds))
            for name, f in zip(("nominal", "correct"), filters):
                f.advance(nominal.tick_of(40.))
                stored = np.load(folder / f"final-{name}-filter.npz")
                for key in ("m", "P", "B", "L", "K"):
                    replay_error = max(replay_error, float(np.max(np.abs(getattr(f, key) - stored[key]))))
            maps = np.load(folder / "fields.npz")
            for t in (40., 46., 52.):
                for f in filters:
                    f.advance(nominal.tick_of(t))
                reconstructed = []
                for i, hi in enumerate(ids):
                    values = []
                    for f, fixed in ((filters[0], 1.), (filters[0], None), (filters[1], 1.)):
                        posterior = f.posterior(i, fixed, level[1], components=True)
                        values.append(f.summary(i, posterior)[0])
                    reconstructed.append(values)
                reconstructed = np.asarray(reconstructed)
                map_error = max(map_error, float(np.max(np.abs(reconstructed - maps[str(t)]))))
                if condition["factor"] == "matched":
                    matched_error = max(matched_error, float(np.max(np.abs(reconstructed[:, 0] - reconstructed[:, 2]))))
            # Score-at-20 cross-check is independent of saved error summaries.
            errors = np.load(folder / "errors.npy", mmap_mode="r")
            at20 = maps["20.0"]
            expected20 = np.mean((at20 - np.stack([field(nominal.tick_of(20.), hi) for hi in ids])[:, None])**2, axis=2)
            score_error = max(score_error, float(np.max(np.abs(expected20 - errors[nominal.tick_of(20.)]))))
            del filters, E
        row = dict(condition=condition["name"], hash_failures=hash_failures,
                   regenerated_evidence_max_error=evidence_error, final_filter_replay_max_error=replay_error,
                   snapshot_reconstruction_max_error=map_error, score_crosscheck_max_error=score_error,
                   matched_fixed_correct_max_error=matched_error,
                   passed=not hash_failures and evidence_error < 1e-12 and replay_error < 1e-10
                          and map_error < 1e-8 and score_error < 1e-12 and matched_error < 1e-12)
        checks.append(row)
        save_json(args.output / "audit.json", dict(conditions=[c["name"] for c in conditions],
                   checks=checks, complete=len(checks) == len(conditions), passed=all(r["passed"] for r in checks)))
        print(json.dumps(row), flush=True)
    shutil.copy2(__file__, args.output / Path(__file__).name)
    save_json(args.output / "sha256.json", {p.name: sha256(p) for p in sorted(args.output.iterdir()) if p.is_file()})


if __name__ == "__main__":
    main()
