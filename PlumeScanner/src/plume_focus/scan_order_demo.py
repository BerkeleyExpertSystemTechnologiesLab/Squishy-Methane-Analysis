"""Portable b00_L adapter around the original C2 P0-V implementation.

No controller registration, default changes, route search, or stored observations.
"""
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np

from plume_focus.c2_p0 import common_grid, direction_of, evaluate
from plume_focus.c2_p0v import construct_rule, make_pairs, prepare_variant
from plume_focus.c2_r2_dynamic import R2TruthGrid
from plume_focus.c2_r2_mechanics import route_record, verify_route_record
from plume_focus.evidence_gated_execution import EvidenceGatedRunConfig
from plume_focus.sensing_session import configuration_hash


def prepare_demo(package_root):
    """Generate the preselected first world and its actual complete FULL prefix."""
    root = Path(package_root)
    spec = json.loads((root / "configs/scan-order-study/demo.json").read_text())
    base = json.loads((root / spec["base_config"]).read_text())
    world = make_pairs(base, spec["world_seed_start"], spec["noise_seed_start"])[0]["variants"]["L"]
    if world["id"] != spec["variant"]:
        raise ValueError("Unexpected demo variant")
    prepared = prepare_variant(world)
    if not prepared["eligible"]:
        raise ValueError("Preselected world has no eligible original FOCUS; do not substitute a winner")
    if prepared["budget"] != spec["action_budget_seconds"] or prepared["T_e"] - prepared["t0"] != spec["evaluation_horizon_seconds"]:
        raise ValueError("Demo clock differs from P0-V")
    return spec, prepared


def compile_routes(spec, prepared):
    """Same READY state, requested support and pose; pay before recompiling."""
    cfg = EvidenceGatedRunConfig.from_dict(prepared["world"]["config"])
    lag = cfg.session.settings.delivery_delay_seconds + cfg.session.settings.completion_overrun_seconds
    records, decisions = {}, {}
    # M/R_up are compiled only to retain the original four-route common
    # quadrature partition. Only B/L_up are executed and evaluated in this demo.
    for name, rule in (("B", "B"), ("M", "M"), ("L_up", "F"), ("R_up", "C")):
        fee = spec["planning_fee_seconds"][name]
        kw = deepcopy(prepared["kwargs"])
        kw["current_position"] = tuple(kw["current_position"])
        kw["started_at_seconds"] = prepared["t0"] + fee
        plan, meta = construct_rule(prepared["mask"], cfg.session.instrument, kw, rule,
                                    prepared["budget"] - fee - lag)
        records[name] = route_record(plan, prepared["mask"], cfg.session.instrument, kw, *direction_of(plan)) if plan else None
        decisions[name] = dict(meta, planning_fee_seconds=fee, receipt_lag_seconds=lag)
    return records, decisions


def run_demo(package_root, output):
    """Causal time-averaged reconstruction MSE, including planning and idle."""
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    spec, prepared = prepare_demo(package_root)
    cfg = EvidenceGatedRunConfig.from_dict(prepared["world"]["config"])
    records, decisions = compile_routes(spec, prepared)
    times, weights = common_grid(prepared, [r for r in records.values() if r], spec["evaluation_step_seconds"])
    truth = R2TruthGrid(cfg.build_source(), times, weights, cfg.model.grid_shape, spec["evaluation_shape"])
    initial_hash = configuration_hash(prepared["saved"])
    results = {}
    arrays = dict(times=times, duration_weights=weights)
    for name in ("B", "L_up"):
        rec = records[name]
        result = evaluate(prepared["saved"], rec, times, weights, truth=truth,
                          source=cfg.build_source(), noise_std=spec["simulated_noise_std"], seed=0, save=True)
        if configuration_hash(prepared["saved"]) != initial_hash:
            raise ValueError("Shared READY state was mutated")
        plan = verify_route_record(rec) if rec else None
        if result["receipt"] and result["receipt"]["received_at_seconds"] > prepared["t0"] + prepared["budget"] + 1e-9:
            raise ValueError("Receipt exceeded original paid action budget")
        results[name] = dict(mse=result["value"], readings=result["n_readings"],
                             column_order=list(rec["column_order"]) if rec else [],
                             directions_down=list(rec["directions_down"]) if rec else [],
                             action_seconds=result["receipt"]["received_at_seconds"] - prepared["t0"] if result["receipt"] else None,
                             scan_seconds=plan.duration_seconds if plan else 0., decision=decisions[name])
        arrays[name + "_mse"] = result["values"]
        (output / (name + "_route.json")).write_text(json.dumps(rec, indent=2) + "\n")
        (output / (name + "_readings.json")).write_text(json.dumps(result["trace"], indent=2) + "\n")
    summary = dict(schema="scan_order_demo_result_v1", variant=spec["variant"],
                   window_seconds=[prepared["t0"], prepared["T_e"]], action_budget_seconds=prepared["budget"],
                   mse_units="synthetic signal squared", initial_state_hash=initial_hash,
                   configuration_hash=configuration_hash(prepared["world"]["config"]),
                   results=results, gain=results["B"]["mse"] - results["L_up"]["mse"],
                   elapsed_compute_seconds=time.perf_counter() - started)
    (output / "RESULTS.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output / "PREFIX.json").write_text(json.dumps(prepared, indent=2, allow_nan=False) + "\n")
    (output / "DEMO_CONFIG.json").write_text(json.dumps(spec, indent=2) + "\n")
    np.savez_compressed(output / "evaluation.npz", **arrays)
    return summary
