"""Resumable, hash-checked study driver for the evidence-gated focus candidate.

All filesystem I/O for this study lives here; runtime modules stay headless.
Phases: declare, verify, shadow, development, freeze, heldout, report, refine.
Generated evidence stays under the ignored outputs/evidence-gated-focus/ tree.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import fields, replace
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_truth_grid import clean_grid

from plume_focus.calibrated_scanner import (CalibratedScanner, CalibrationSession, CalibrationSettings,
                                             DecayGaussianBelief)
from plume_focus.calibration_execution import CalibrationRunConfig
from plume_focus.evidence_gate import (EVIDENCE_GATE_POLICY_ID, EvidenceGateState, bound_multiplier,
    build_action_evidence, calibrated_value, classify_bounds, entry_decision, full_action_alpha, normal_quantile,
    observed_value_cutoff, reading_bounds, renewal_decision)
from plume_focus.evidence_gated_execution import EvidenceGatedRunConfig
from plume_focus.evidence_gated_loop import (EvidenceGateSettings, EvidenceGatedScanner, EvidenceGatedSession,
    gate_reading_evidence, gate_slot_map)
from plume_focus.focus_extent_diagnostics import EvidenceReplay
from plume_focus.full_fov_execution import decode_config, decode_plan, run_full_fov, verify_replay
from plume_focus.physical_scan import compile_physical_scan
from plume_focus.sensing_contracts import ExecutionReceipt, SensorReading
from plume_focus.sensing_counterfactual import OnlineLoss
from plume_focus.sensing_risk import projection_gram
from plume_focus.sensing_session import configuration_hash
from plume_focus.signal_belief import grid_positions, interpolation_stencil

STUDY = Path("configs/evidence-gated-focus/study.json")
PLANNING_MODULES = ("evidence_gate.py", "evidence_gated_loop.py", "evidence_gated_execution.py")
ISOLATED_MODULES = ("sensing_counterfactual", "focus_extent_diagnostics", "calibration_evaluation",
                    "sensing_simulation", "sensing_rendering", "video_sources")


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def revision(): return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Refusing to write a non-finite number; use explicit null instead")
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def code_hash():
    paths = [Path(__file__), Path("scripts/calibration_truth_grid.py"), *sorted(Path("src/plume_focus").glob("*.py"))]
    return hashlib.sha256(b"".join(p.name.encode() + p.read_bytes().replace(b"\r\n", b"\n") for p in paths)).hexdigest()


def study(): return read(STUDY)


def full_scan_seconds(config):
    """Declared physical duration of one complete full-FoV action, never the focus ceiling."""
    mask = np.ones(config.session.instrument.lattice_shape, bool)
    plan = compile_physical_scan(mask, config.session.instrument, scan_id="declared-full",
                                 started_at_seconds=0., current_position=(0., 0.))
    return plan.duration_seconds


def build_config(document, profile, case, seed, noise, policy):
    """Expand one declared record into an immutable run configuration."""
    base = decode_config(read(document["reference_configs"][profile]))
    case_settings = document["cases"][case]
    source = base.source
    if seed:
        phases = np.random.default_rng(seed).uniform(0, 2 * np.pi, 3)
        source = replace(source, lateral_phase=float(phases[0]), amplitude_phase=float(phases[1]),
                         shimmer_phase=float(phases[2]))
    source = replace(source, signal_scale=float(case_settings.get("signal_scale", 1.)))
    wind = base.wind
    if "wind_delta_yx" in case_settings:
        wind = replace(wind, velocity_yx=tuple(np.asarray(wind.velocity_yx) + np.asarray(case_settings["wind_delta_yx"])))
    cessation = None if case_settings.get("cessation_seconds") is None else float(case_settings["cessation_seconds"])
    windows = tuple(tuple(float(v) for v in w) for w in case_settings.get("active_windows", ())) or None
    settings = base.session.settings
    common = dict(source=source, wind=wind, duration_seconds=float(document["duration_seconds"]),
                  simulated_noise_std=float(noise), cessation_seconds=cessation, seed=int(seed), delivery="stream",
                  evaluation_shape=tuple(document["evaluation"]["shape"]),
                  evaluation_step_seconds=float(document["evaluation"]["step_seconds"]), active_windows=windows)
    if policy == "candidate":
        gate = document["gate"]
        candidate_settings = EvidenceGateSettings(
            **{f.name: getattr(settings, f.name) for f in fields(CalibrationSettings)},
            alpha=float(gate["alpha"]), entry_level=float(gate["entry_level"]), retain_level=float(gate["retain_level"]))
        session = EvidenceGatedSession(base.session.instrument, base.session.model, candidate_settings,
            base.session.source_position, base.session.home_yx, half_life_seconds=base.session.half_life_seconds)
        return EvidenceGatedRunConfig(session, **common)
    session = CalibrationSession(base.session.instrument, base.session.model, settings,
        base.session.source_position, base.session.home_yx, half_life_seconds=base.session.half_life_seconds)
    return CalibrationRunConfig(session, **common)


def job_name(stage, profile, case, seed, noise, policy):
    return f"{stage}-{profile}-{case}-seed{seed}-noise{noise:g}-{policy}"


def planned_jobs(document, stage):
    section = document[stage]
    if stage == "development":
        for profile in section["profiles"]:
            for case in section["cases"]:
                for noise in section["noise"]:
                    for policy in ("control", "candidate"):
                        yield {"stage": stage, "profile": profile, "case": case, "seed": int(section["seed"]),
                               "noise": float(noise), "policy": policy}
    else:
        for profile in section["profiles"]:
            for case in section["cases"]:
                for seed, noise in section["seed_noise_pairs"]:
                    for policy in ("control", "candidate"):
                        yield {"stage": stage, "profile": profile, "case": case, "seed": int(seed),
                               "noise": float(noise), "policy": policy}


def declare(root):
    root = Path(root)
    document = study()
    if (root / "declaration.json").exists():
        declaration = read(root / "declaration.json")
        runs = sorted((root / "runs").glob("*.json")) if (root / "runs").exists() else []
        if declaration["study_sha256"] == sha(STUDY):
            for job in declaration["development"] + declaration["heldout"]:
                assert sha(root / job["config"]) == job["config_sha256"], f"Declared config changed: {job['name']}"
            print("Declaration already present and hash-consistent")
            return
        if runs:
            raise SystemExit("Study manifest changed after outcomes exist; declare a new study root explicitly")
        print("Study manifest changed before any outcome; re-declaring")
        (root / "declaration.json").unlink()
        if (root / "heldout-manifest.json").exists():
            (root / "heldout-manifest.json").unlink()
    if (root / "runs").exists() and any((root / "runs").glob("*.json")):
        raise SystemExit("Refusing to declare after outcomes exist")
    groups, reference = {"development": [], "heldout": []}, {}
    for profile, path in document["reference_configs"].items():
        reference[profile] = {"path": path, "sha256": sha(path)}
    for stage in ("development", "heldout"):
        for job in planned_jobs(document, stage):
            name = job_name(stage, job["profile"], job["case"], job["seed"], job["noise"], job["policy"])
            config = build_config(document, job["profile"], job["case"], job["seed"], job["noise"], job["policy"])
            relative = f"configs/{name}.json"
            write(root / relative, config.to_dict())
            groups[stage].append({**job, "name": name, "config": relative, "config_sha256": sha(root / relative),
                                  "configuration_hash": config.digest, "policy_id": config.session.policy_id,
                                  "full_scan_seconds": full_scan_seconds(config)})
    duration = {profile: full_scan_seconds(build_config(document, profile, "continuous", 0, 0., "control"))
                for profile in document["reference_configs"]}
    declaration = {"artifact_type": "evidence_gated_focus_declaration_v1", "revision": revision(),
                   "study_path": STUDY.as_posix(), "study_sha256": sha(STUDY), "reference_configs": reference,
                   "declared_full_scan_seconds": duration, "gate": document["gate"], "code_sha256": code_hash(),
                   **groups}
    write(root / "declaration.json", declaration)
    write(root / "heldout-manifest.json", {"artifact_type": "evidence_gated_focus_heldout_manifest_v1",
        "declared_before_development": True, "declaration_sha256": sha(root / "declaration.json"),
        "records": [{"name": j["name"], "config": j["config"], "config_sha256": j["config_sha256"]}
                    for j in groups["heldout"]]})
    print(f"Declared {len(groups['development'])} development and {len(groups['heldout'])} held-out records")

def load_declaration(root):
    root = Path(root)
    declaration = read(root / "declaration.json")
    for job in declaration["development"] + declaration["heldout"]:
        assert sha(root / job["config"]) == job["config_sha256"], f"Declared config changed: {job['name']}"
    assert declaration["study_sha256"] == sha(STUDY), "Study manifest changed after declaration"
    return declaration


def configured(root, job):
    return decode_config(read(Path(root) / job["config"]))


def recorded_scans(record):
    """Group recorded events into physical actions with their delivered readings."""
    scans, current = [], None
    for event in record["events"]:
        if event["operation"] == "plan_next":
            current = None
            if event["plan"]["samples"]:
                current = {"plan": event["plan"], "readings": [], "receipt": None}
                scans.append(current)
        elif event["operation"] == "ingest_readings" and current is not None:
            current["readings"].extend(event["readings"])
        elif event["operation"] == "finish_scan" and current is not None:
            current["receipt"] = event["receipt"]
            current = None
    return scans


def verify(root):
    root = Path(root)
    document = study()
    declaration = load_declaration(root)
    context_path = Path("docs/handoffs/evidence-gated-focus-context.json")
    context = read(context_path)
    results = []

    def check(name, function):
        try:
            results.append({"name": name, "status": "passed", "detail": function()})
        except Exception as error:
            results.append({"name": name, "status": "failed", "detail": f"{type(error).__name__}: {error}"})

    def context_package_hash():
        digest = sha(context_path)
        documented = "61a97b0916e094bf92760a968a5ff3dfdc51d02a8eb23b5b038494a08eaa53d6"
        assert digest == documented, f"Context package bytes changed: {digest}"
        return {"sha256": digest, "bytes": context_path.stat().st_size, "documented": documented}

    def reference_configuration_hashes():
        rows = {}
        for profile, entry in context["reference_runs"].items():
            local = Path(entry["repository_path"])
            digest = sha(local)
            assert digest == entry["file_sha256"], f"Frozen reference changed: {profile}"
            assert local.as_posix() == document["reference_configs"][profile]
            rows[profile] = {"path": local.as_posix(), "sha256": digest}
        return rows

    def historical_summary_hashes():
        rows, unavailable = {}, []
        for name, relative in (("summary", "outputs/scanner-calibration/summary.json"),
                               ("freeze", "outputs/scanner-calibration/freeze.json"),
                               ("bound_heldout", "outputs/scanner-calibration/bound-heldout.json")):
            path = Path(relative)
            if not path.exists():
                unavailable.append(name)
                continue
            expected = context["summary_source"][{"summary": "sha256", "freeze": "freeze_sha256",
                                                  "bound_heldout": "bound_heldout_sha256"}[name]]
            assert sha(path) == expected, f"Historical evidence changed: {name}"
            rows[name] = expected
        return {"verified": rows, "unavailable": unavailable}

    def declared_full_scan_durations():
        expected = {"coarse": 33.575, "double": 129.5875}
        rows = {}
        for profile, value in declaration["declared_full_scan_seconds"].items():
            assert abs(value - expected[profile]) < 1e-9, f"Declared full scan changed: {profile}"
            assert abs(full_scan_seconds(decode_config(read(document["reference_configs"][profile]))) - expected[profile]) < 1e-9
            rows[profile] = value
        return rows

    def declaration_round_trip():
        rows = {"development": 0, "heldout": 0}
        for job in declaration["development"] + declaration["heldout"]:
            config = configured(root, job)
            assert config.digest == job["configuration_hash"], f"Configuration hash changed: {job['name']}"
            restored = type(config).from_dict(json.loads(json.dumps(config.to_dict())))
            assert restored == config and restored.digest == config.digest, f"Round trip changed: {job['name']}"
            assert config.session.policy_id == job["policy_id"]
            assert abs(full_scan_seconds(config) - job["full_scan_seconds"]) < 1e-9
            rows[job["stage"]] += 1
        return rows

    def estimator_identity():
        rows = []
        for profile in document["reference_configs"]:
            base = decode_config(read(document["reference_configs"][profile])).session
            control = build_config(document, profile, "continuous", 0, 0., "control").session
            candidate = build_config(document, profile, "continuous", 0, 0., "candidate").session
            assert control.to_dict() == base.to_dict(), "Control session differs from the frozen calibrated session"
            assert candidate.model.to_dict() == base.model.to_dict(), "Estimator model changed"
            assert candidate.half_life_seconds == base.half_life_seconds == 6.
            assert candidate.formulation_id == base.formulation_id == "beam_reconstruction_decay_v1_candidate"
            assert candidate.instrument.to_dict() == base.instrument.to_dict()
            assert candidate.transport_id == base.transport_id == "zero_inflow_v1"
            assert candidate.settings.vertical_margin_fov == 0. and candidate.settings.full_scan_only is False
            assert candidate.settings.minimum_readings == control.settings.minimum_readings
            assert candidate.settings.focus_seconds == control.settings.focus_seconds == 12.
            assert (candidate.settings.alpha, candidate.settings.entry_level, candidate.settings.retain_level) == (.05, .15, .10)
            rows.append({"profile": profile, "control_session": control.digest, "candidate_session": candidate.digest})
        return {"pairs": rows, "note": "identical estimator; policy identity and gate settings differ only"}

    def controller_dispatch():
        from plume_focus.controller import SensingController
        rows = {}
        for policy, expected in (("control", CalibratedScanner), ("candidate", EvidenceGatedScanner)):
            config = build_config(document, "coarse", "continuous", 0, 0., policy)
            controller = SensingController.from_session(config.session)
            assert type(controller) is expected, f"Dispatch changed for {policy}"
            rows[policy] = type(controller).__name__
        return rows

    def gate_arithmetic():
        independent = NormalDist().inv_cdf
        worst = 0.
        for probability in (1 - 1e-9, 1 - .001, .5, .9, .95, 1 - .05 / (2 * 4), 1 - .025 / (2 * 640)):
            worst = max(worst, abs(normal_quantile(probability) - independent(probability)))
        assert worst < 1e-12, "Normal quantile disagrees with an independent implementation"
        cutoffs = {m: observed_value_cutoff(.15, gain=1., offset=0., sensor_noise_std=.05, alpha_action=.025,
                                             scheduled_slots=m) for m in (640, 2560)}
        assert abs(cutoffs[640] - .35564779584685524) < 1e-12
        assert abs(cutoffs[2560] - .37111494274450096) < 1e-12
        totals = [sum(full_action_alpha(p) for p in range(1, P + 1)) for P in (1, 2, 5, 50)]
        assert all(total <= .05 + 1e-15 for total in totals)
        assert abs(totals[0] - .025) < 1e-15 and totals[-1] < .05
        assert classify_bounds([.15], [.15], .15) == "inconclusive"
        assert classify_bounds([.1500000001], [.9], .15) == "supported"
        assert classify_bounds([-.2], [.1499], .15) == "below_level_on_observed_footprints"
        assert classify_bounds([], [], .15) == "unavailable"
        scaled = reading_bounds((0., -1.), gain=2., offset=.5, sensor_noise_std=.1, alpha_action=.025, scheduled_slots=4)[0]
        assert abs(scaled[0] + .25 + (normal_quantile(1 - .025 / 8) * .05)) < 1e-12
        larger = bound_multiplier(.025, 2560) - bound_multiplier(.025, 640)
        assert larger > 0, "Multiplicity correction must grow with the frozen schedule"
        return {"max_quantile_difference": worst, "observed_value_cutoffs": cutoffs, "epoch_alpha_totals": totals}

    def planning_import_isolation():
        rows = {}
        for name in PLANNING_MODULES:
            text = (Path("src/plume_focus") / name).read_text(encoding="utf-8")
            imports = [line.strip() for line in text.splitlines() if line.strip().startswith(("import ", "from "))]
            forbidden = sorted({module for module in ISOLATED_MODULES for line in imports if module in line})
            assert not forbidden, f"Evaluator or simulator import in {name}: {forbidden}"
            rows[name] = len(imports)
        return rows

    def gate_state_round_trip():
        evidence = build_action_evidence(scan_id="scan-000000", epoch=1, action_index=1, mode="FULL", alpha_action=.025,
            scheduled_slots=2, readings=tuple(), completed_at_seconds=1., received_at_seconds=1.,
            plan_completed_at_seconds=1., contract_notes=("unmappable_reading:x",))
        state = EvidenceGateState(policy_id=EVIDENCE_GATE_POLICY_ID, epoch=1, action_index=1,
            epoch_alpha_allocated=.025, focus_allowed=False, contract_violations=1, focus_evidence=None,
            last_decision=None, full_epoch_evidence=(evidence,), transitions=())
        restored = EvidenceGateState.from_dict(json.loads(json.dumps(state.to_dict())))
        assert restored == state
        return {"fields": len(state.to_dict()), "round_trip": True}

    def truth_identity_and_lineage():
        rows = []
        for profile, case in (("coarse", "continuous"), ("double", "blank")):
            control = configured(root, {"config": f"configs/development-{profile}-{case}-seed0-noise0-control.json"})
            candidate = configured(root, {"config": f"configs/development-{profile}-{case}-seed0-noise0-candidate.json"})
            key = truth_key(control)
            assert key == truth_key(candidate), "Truth identity depends on the policy"
            historical = Path("outputs/scanner-calibration/truth-cache") / f"{key}.npz"
            entry = {"profile": profile, "case": case, "truth_key": key,
                     "historical_cache": historical.exists()}
            if historical.exists():
                with np.load(historical) as cached:
                    times = cached["times"][:4]
                    for index, time in enumerate(times):
                        values = clean_grid(control, float(time), tuple(control.evaluation_shape))
                        assert abs(float(np.mean(values ** 2)) - float(cached["square"][index])) < 1e-12
                        assert abs(float(np.mean(values)) - float(cached["average"][index])) < 1e-12
                entry["historical_samples_checked"] = int(len(times))
            rows.append(entry)
        return rows

    check("context_package_hash", context_package_hash)
    check("reference_configuration_hashes", reference_configuration_hashes)
    check("historical_summary_hashes", historical_summary_hashes)
    check("declared_full_scan_durations", declared_full_scan_durations)
    check("declaration_round_trip", declaration_round_trip)
    check("estimator_identity", estimator_identity)
    check("controller_dispatch", controller_dispatch)
    check("gate_arithmetic_independent", gate_arithmetic)
    check("planning_import_isolation", planning_import_isolation)
    check("gate_state_round_trip", gate_state_round_trip)
    check("truth_identity_and_lineage", truth_identity_and_lineage)
    status = "passed" if all(row["status"] == "passed" for row in results) else "failed"
    write(root / "checks" / "verify.json", {"artifact_type": "evidence_gated_focus_verification_v1",
        "revision": revision(), "code_sha256": code_hash(), "status": status, "checks": results})
    print(json.dumps({"status": status, "checks": {row["name"]: row["status"] for row in results}}, indent=1))
    if status != "passed":
        raise SystemExit("Verification failed; see checks/verify.json")


def metrics(record, q, config, replay):
    """Evaluator-side metrics; truth is used only for scoring, never by the gate."""
    lattice = tuple(config.session.instrument.lattice_shape)
    horizon, cells = float(config.duration_seconds), int(np.prod(lattice))
    times, durations = q.times, q.durations
    core = np.asarray(q.core.todense())
    declared_full = full_scan_seconds(config)
    delivered = [event for event in record["events"] if event["operation"] == "ingest_readings"]
    visits = np.full(cells, -1.)
    travel = acquiring = waiting = latency = 0.
    focus_supported_seconds = focus_core_captured = focus_core_total = 0.
    whole_core_total = whole_core_captured = 0.
    ever_integral = trailing_integral = 0.
    age_integral = unknown_integral = 0.
    read_core = read_background = 0
    false_focus = missed_focus = refocus_after_background = repeated_focus_without_support = 0
    return_home_seconds = 0.
    scheduled_full = scheduled_focus = actual_full = actual_focus = 0.
    epochs = set()
    previous_mode = None
    startup = False
    last_background_full = False
    mean_positive = mean_positive_unsupported = supported_mean_negative = unavailable = 0
    max_gate_mean = 0.
    seen = 0
    index = 0
    for scan in record["scans"]:
        plan, receipt = scan["plan"], scan["receipt"]
        at, end = plan["started_at_seconds"], receipt["received_at_seconds"]
        selected = np.flatnonzero((times >= at - 1e-10) & (times < end - 1e-10))
        visible = float(np.interp(at, times, q.peak)) > .15
        scheduled = plan["completed_at_seconds"] - at
        actual = receipt["completed_at_seconds"] - at
        mean_gate = float(plan["gate_max_mean"])
        gate_state = plan.get("gate_evidence_state")
        bound = gate_state is not None
        max_gate_mean = max(max_gate_mean, mean_gate)
        seen += 1
        mean_positive += mean_gate > float(plan["threshold"])
        unavailable += gate_state == "unavailable"
        support = gate_state == "supported"
        mean_positive_unsupported += bool(bound and (mean_gate > float(plan["threshold"])) and not support)
        supported_mean_negative += bool(bound and support and mean_gate <= float(plan["threshold"]))
        if plan["mode"] == "FOCUS":
            scheduled_focus += scheduled
            actual_focus += actual
            false_focus += not visible
            refocus_after_background += last_background_full
            repeated_focus_without_support += bool(previous_mode == "FOCUS" and bound and not support)
            if len(selected):
                total = core[selected].sum(axis=1)
                inside = core[selected][:, list(plan["swept_cells"])].sum(axis=1)
                valid = total > 0
                focus_core_captured += float((inside[valid] * durations[selected][valid]).sum())
                focus_core_total += float((total[valid] * durations[selected][valid]).sum())
                focus_supported_seconds += float(durations[selected][valid].sum())
        else:
            scheduled_full += scheduled
            actual_full += actual
            epochs.add(int(plan["full_epoch"]))
            missed_focus += startup and visible
            if plan["return_home_until_seconds"] > at + 1e-9:
                return_home_seconds += plan["return_home_until_seconds"] - at
        if len(selected):
            whole_core_total += float(core[selected].sum(axis=1) @ durations[selected])
            whole_core_captured += float(core[selected][:, list(plan["swept_cells"])].sum(axis=1) @ durations[selected])
        for target, clean in zip(plan["samples"], scan["clean_readings"]):
            peak = float(np.interp(target["acquired_at_seconds"], times, q.peak))
            read_core += clean > max(.15, .5 * peak)
            read_background += clean <= .15
        for segment in plan["segments"]:
            duration = segment["completed_at_seconds"] - segment["started_at_seconds"]
            if segment["kind"] == "sweep":
                acquiring += duration
            elif segment["kind"] == "move":
                travel += duration
            else:
                waiting += duration
        latency += receipt["received_at_seconds"] - receipt["completed_at_seconds"]
        last_background_full = (plan["mode"] == "FULL" and not scan["missing_full_cells"]
                                and max(scan["clean_readings"], default=0.) <= .15)
        startup = scan["startup_complete"]
        previous_mode = plan["mode"]
    observed = None
    for i, (time, duration) in enumerate(zip(times, durations)):
        while index < len(delivered) and delivered[index]["at_seconds"] <= time + 1e-10:
            for reading in delivered[index]["readings"]:
                cell = tuple(np.minimum((np.array(reading["position_yx"]) * lattice).astype(int), np.array(lattice) - 1))
                flat = int(np.ravel_multi_index(cell, lattice))
                visits[flat] = max(visits[flat], reading["acquired_at_seconds"])
            index += 1
        observed = visits >= 0
        if observed.any():
            age_integral += float(np.mean(time - visits[observed])) * duration
        unknown_integral += float((~observed).mean()) * duration
        total = float(core[i].sum())
        if total > 0:
            ever_integral += float(core[i][observed].sum()) / total * duration
            fresh = observed & ((time - np.maximum(visits, 0.)) <= declared_full)
            trailing_integral += float(core[i][fresh].sum()) / total * duration
    state = replay.gate_state() if hasattr(replay, "gate_state") else None
    shape = tuple(config.model.grid_shape)
    mean, _ = replay.predictive_moments(horizon, shape)
    final_peak = float(np.max(mean))
    final_visible = float(np.interp(horizon, times, q.peak)) > .15
    return {
        "spent_seconds": actual_full + actual_focus,
        "scheduled_seconds": scheduled_full + scheduled_focus,
        "actual_full_seconds": actual_full, "actual_focus_seconds": actual_focus,
        "scheduled_full_seconds": scheduled_full, "scheduled_focus_seconds": scheduled_focus,
        "travel_seconds": travel, "acquiring_pass_seconds": acquiring, "waiting_seconds": waiting,
        "receipt_latency_seconds": latency, "return_home_charged_seconds": return_home_seconds,
        "declared_full_scan_seconds": declared_full,
        "focus_scans": sum(scan["plan"]["mode"] == "FOCUS" for scan in record["scans"]),
        "full_scans": sum(scan["plan"]["mode"] == "FULL" for scan in record["scans"]),
        "full_epochs_started": len(epochs), "completed_full_epochs": record["completed_full_scans"],
        "incomplete_full_scan_budget": record["termination"] == "incomplete_full_scan_budget",
        "focus_core_geometric_coverage": (focus_core_captured / focus_core_total) if focus_core_total > 0 else None,
        "focus_core_supported_seconds": focus_supported_seconds,
        "whole_horizon_core_captured_fraction": (whole_core_captured / whole_core_total) if whole_core_total > 0 else None,
        "ever_sampled_core_coverage": ever_integral / horizon,
        "trailing_full_scan_core_coverage": trailing_integral / horizon,
        "actual_core_readings": int(read_core), "actual_background_readings": int(read_background),
        "actual_core_reference": "clean exposure response vs .15 and half interpolated instantaneous truth peak",
        "false_focus_decisions": int(false_focus), "missed_focus_decisions": int(missed_focus),
        "refocus_after_background_full_scan": int(refocus_after_background),
        "focus_after_focus_without_support": int(repeated_focus_without_support),
        "mean_observed_sample_age_seconds": age_integral / horizon, "mean_unknown_fraction": unknown_integral / horizon,
        "gate_decisions": seen, "gate_switches": len(state.transitions) if state is not None else None,
        "mean_gate_positive_decisions": int(mean_positive),
        "mean_positive_evidence_unsupported_decisions": int(mean_positive_unsupported) if state is not None else None,
        "evidence_supported_mean_negative_decisions": int(supported_mean_negative) if state is not None else None,
        "evidence_unavailable_decisions": int(unavailable) if state is not None else None,
        "max_gate_max_mean": max_gate_mean,
        "gate_contract_violations": int(state.contract_violations) if state is not None else None,
        "final_posterior_max_mean": final_peak,
        "final_posterior_exceeds_reporting_threshold": bool(final_peak > .15),
        "final_posterior_false_peak": bool(final_peak > .15 and not final_visible),
        "gate_transitions": [transition.to_dict() for transition in state.transitions] if state is not None else None,
    }


def change_metrics(record, q, config):
    """Cessation and intermittency behaviour: latencies, availability and reacquisition."""
    changes = []
    if config.cessation_seconds is not None:
        changes.append({"kind": "cessation", "at_seconds": float(config.cessation_seconds)})
    elif config.active_windows:
        changes.append({"kind": "stop", "at_seconds": float(config.active_windows[0][1])})
        changes.append({"kind": "resume", "at_seconds": float(config.active_windows[-1][0])})
    rows = []
    for change in changes:
        at = change["at_seconds"]
        straddling = [scan for scan in record["scans"] if scan["plan"]["mode"] == "FULL"
                      and scan["plan"]["started_at_seconds"] < at - 1e-9 < scan["receipt"]["received_at_seconds"]]
        started = [scan for scan in record["scans"] if scan["plan"]["mode"] == "FULL"
                   and scan["plan"]["started_at_seconds"] >= at - 1e-9]
        completed = [scan for scan in started if not scan["missing_full_cells"]]
        background_epochs = [scan for scan in started if max(scan["clean_readings"], default=0.) <= .15]
        row = {"kind": change["kind"], "at_seconds": at,
               "full_straddling_change": len(straddling),
               "time_to_first_full_start_seconds": (started[0]["plan"]["started_at_seconds"] - at) if started else None,
               "time_to_completed_full_seconds": (completed[0]["receipt"]["received_at_seconds"] - at) if completed else None,
               "first_all_background_full_epoch_start_seconds":
                   (background_epochs[0]["plan"]["started_at_seconds"] - at) if background_epochs else None,
               "remaining_seconds_at_change": float(config.duration_seconds) - at,
               "sufficient_remaining_time": (float(config.duration_seconds) - at) >= full_scan_seconds(config),
               "unavailable_reason": None if completed else (
                   "incomplete_full_scan_budget" if record["termination"] == "incomplete_full_scan_budget"
                   else "horizon_before_completion" if started else "no_full_started"),
               "first_background_full_refocuses": None}
        row["straddling_post_change_reading_slots"] = sum(
            target["acquired_at_seconds"] >= at
            for scan in straddling for target in scan["plan"]["samples"][:len(scan["clean_readings"])])
        if background_epochs:
            first = background_epochs[0]
            following = next((scan for scan in record["scans"]
                              if scan["plan"]["started_at_seconds"] >= first["receipt"]["received_at_seconds"] - 1e-9), None)
            row["first_background_full_refocuses"] = bool(following is not None and following["plan"]["mode"] == "FOCUS")
        if change["kind"] == "resume":
            latency, genuine = None, False
            for scan in record["scans"]:
                start = scan["plan"]["started_at_seconds"]
                core_cells = set(np.flatnonzero(np.asarray(q.core[int(np.argmin(np.abs(q.times - start)))].todense()).ravel()
                                                > 0).tolist())
                if start >= at and scan["plan"]["mode"] == "FOCUS" and core_cells & set(scan["plan"]["requested_cells"]):
                    genuine = True
                for target, clean in zip(scan["plan"]["samples"], scan["clean_readings"]):
                    time = target["acquired_at_seconds"]
                    if time >= at and clean > max(.15, .5 * float(np.interp(time, q.times, q.peak))):
                        latency = time - at
                        break
                if latency is not None:
                    break
            row["reacquisition_latency_seconds"] = latency
            row["reacquisition_focus_plan"] = genuine
            # Preserve the predeclared full-restart eligibility above. An ongoing
            # scan can still offer readings even when a fresh full scan cannot fit.
            row["reacquisition_opportunity"] = bool(row["sufficient_remaining_time"]
                                                     or row["straddling_post_change_reading_slots"])
        rows.append(row)
    return rows


def truth_key(config):
    """Policy-, estimator- and noise-independent identity of the common truth field."""
    return configuration_hash({"source": config.source.to_dict(), "cessation": config.cessation_seconds,
        "windows": config.active_windows, "shape": config.evaluation_shape,
        "latent": config.model.grid_shape, "lattice": config.session.instrument.lattice_shape,
        "step": config.evaluation_step_seconds, "duration": config.duration_seconds})


@lru_cache(maxsize=2)
def cached_truth(root, config_json):
    config = decode_config(json.loads(config_json))
    source = config.build_source()
    shape, latent, lattice = config.evaluation_shape, config.model.grid_shape, config.session.instrument.lattice_shape
    key = truth_key(config)
    path = Path(root) / "truth-cache" / f"{key}.npz"
    if path.exists():
        with np.load(path) as z:
            q = SimpleNamespace(**{k: z[k] for k in ("times", "durations", "gram", "cross", "square", "average",
                                                     "mean_operator", "peak")})
            q.core = csr_matrix((z["data"], z["indices"], z["indptr"]), shape=(len(q.times), int(np.prod(lattice))))
            return q
    times = np.arange(int(np.ceil(config.duration_seconds / config.evaluation_step_seconds))) * config.evaluation_step_seconds
    durations = np.diff(np.append(times, config.duration_seconds))
    points = grid_positions(shape)
    ids, weights = interpolation_stencil(points, latent)
    gram = projection_gram(latent, shape)
    mapping = np.ravel_multi_index(tuple((points * np.array(lattice)).astype(int).T), lattice)
    mean_operator = np.zeros(int(np.prod(latent)))
    np.add.at(mean_operator, ids.ravel(), weights.ravel() / len(points))
    cross, square, average, peak, data, indices, indptr = [], [], [], [], [], [], [0]
    for time in times:
        values = clean_grid(config, float(time), shape).ravel()
        projected = np.zeros(int(np.prod(latent)))
        np.add.at(projected, ids.ravel(), (weights * values[:, None]).ravel() / len(points))
        cross.append(projected)
        square.append(float(np.mean(values ** 2)))
        average.append(float(np.mean(values)))
        peak.append(float(values.max()))
        core = values > max(.15, .5 * values.max())
        counts = np.bincount(mapping[core], minlength=int(np.prod(latent)))
        nonzero = np.flatnonzero(counts)
        indices.extend(nonzero)
        data.extend(counts[nonzero] / len(points))
        indptr.append(len(data))
    arrays = dict(times=times, durations=durations, gram=gram, cross=np.array(cross), square=np.array(square),
                  average=np.array(average), mean_operator=mean_operator, peak=np.array(peak), data=np.array(data),
                  indices=np.array(indices, dtype=int), indptr=np.array(indptr, dtype=int))
    path.parent.mkdir(parents=True, exist_ok=True)
    # Workers may produce the same deterministic cache concurrently; each writes a
    # distinct temporary file before atomically replacing the shared entry.
    temporary = path.with_name(path.stem + f"-{os.getpid()}.partial.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    q = SimpleNamespace(**{k: arrays[k] for k in ("times", "durations", "gram", "cross", "square", "average",
                                                  "mean_operator", "peak")})
    q.core = csr_matrix((arrays["data"], arrays["indices"], arrays["indptr"]),
                        shape=(len(times), int(np.prod(lattice))))
    return q


def truth(root, config):
    return cached_truth(str(root), json.dumps(config.to_dict(), sort_keys=True))


def candidate_session(document, session):
    """The same estimator binding with the evidence-gated policy identity."""
    gate = document["gate"]
    settings = EvidenceGateSettings(**{f.name: getattr(session.settings, f.name) for f in fields(CalibrationSettings)},
        alpha=float(gate["alpha"]), entry_level=float(gate["entry_level"]), retain_level=float(gate["retain_level"]))
    return EvidenceGatedSession(session.instrument, session.model, settings, session.source_position, session.home_yx,
                                half_life_seconds=session.half_life_seconds)


def execute(root, job, stage):
    root = Path(root)
    path = root / "runs" / f"{job['name']}.json"
    config = configured(root, job)
    freeze_file_sha = None
    if stage == "heldout":
        freeze_document = read(root / "freeze.json")
        assert freeze_document["code_sha256"] == code_hash(), "Held-out execution requires the frozen code revision"
        freeze_file_sha = sha(root / "freeze.json")
    if path.exists():
        saved = read(path)
        if (saved.get("configuration_hash") == config.digest and saved.get("code_sha256") == code_hash()
                and saved.get("study_sha256") == sha(STUDY) and saved.get("freeze_file_sha256") == freeze_file_sha):
            return job["name"], {"status": "reused", "J_over_T": saved["J_over_T"]}
    q = truth(root, config)
    result = run_full_fov(config, truth=q)
    replay = verify_replay(result, truth=q)
    state = replay.gate_state() if hasattr(replay, "gate_state") else None
    result.update(job=job, stage=stage, producing_revision=revision(), code_sha256=code_hash(),
                  study_sha256=sha(STUDY), declaration_sha256=sha(root / "declaration.json"),
                  truth_key=truth_key(config), freeze_file_sha256=freeze_file_sha,
                  gate=None if state is None else {
                      "policy_id": state.policy_id, "alpha": config.session.settings.alpha,
                      "entry_level": config.session.settings.entry_level,
                      "retain_level": config.session.settings.retain_level,
                      "epoch": state.epoch, "action_index": state.action_index,
                      "epoch_alpha_allocated": state.epoch_alpha_allocated, "focus_allowed": state.focus_allowed,
                      "contract_violations": state.contract_violations,
                      "last_decision": None if state.last_decision is None else state.last_decision.to_dict(),
                      "transitions": [transition.to_dict() for transition in state.transitions]})
    result["metrics"] = metrics(result, q, config, replay)
    result["change_metrics"] = change_metrics(result, q, config)
    write(path, result)
    return job["name"], {"status": "completed", "J_over_T": result["J_over_T"],
                         "focus_scans": result["metrics"]["focus_scans"]}


def shadow_audit(record, path, document):
    """Fixed-evidence attribution: identical readings, old mean gate against the new statistics."""
    config = decode_config(record["configuration"])
    assert config.session.policy_id == "full_fov_margin_loop_v2_candidate"
    session = config.session
    model = config.model
    threshold = float(config.session.settings.threshold)
    gate = document["gate"]
    scans = recorded_scans(record)
    assert len(scans) == len(record["scans"]), "Recorded scans and events disagree"
    for row, saved in zip(scans, record["scans"]):
        row["record"] = saved
    rows, decisions = [], []
    epoch, action_index, epoch_alpha, epoch_evidence = None, 0, 0., []
    shadow_next, divergences = [], []
    for position, entry in enumerate(scans):
        plan = decode_plan(entry["plan"])
        readings = [SensorReading.from_dict(item) for item in entry["readings"]]
        receipt = ExecutionReceipt.from_dict(entry["receipt"])
        slots, notes = gate_slot_map(plan, readings)
        if plan.mode == "FULL":
            if plan.full_epoch != epoch:
                epoch, action_index, epoch_alpha, epoch_evidence = int(plan.full_epoch), 0, 0., []
            action_index += 1
            alpha_action = full_action_alpha(action_index, float(gate["alpha"]))
            epoch_alpha += alpha_action
        else:
            alpha_action = float(gate["alpha"])
        evidence = build_action_evidence(scan_id=plan.scan_id, epoch=int(plan.full_epoch), action_index=action_index,
            mode=plan.mode, alpha_action=alpha_action, scheduled_slots=len(plan.samples),
            readings=tuple(gate_reading_evidence(reading, slots[reading.reading_id], gain=model.gain,
                offset=model.offset, sensor_noise_std=model.sensor_noise_std, alpha_action=alpha_action,
                scheduled_slots=len(plan.samples)) for reading in readings if reading.reading_id in slots),
            completed_at_seconds=receipt.completed_at_seconds, received_at_seconds=receipt.received_at_seconds,
            plan_completed_at_seconds=plan.completed_at_seconds, contract_notes=notes)
        complete_full_epoch = bool(plan.mode == "FULL" and not entry["record"]["missing_full_cells"])
        decision = None
        if plan.mode == "FULL":
            epoch_evidence.append(evidence)
            if complete_full_epoch:
                decision = entry_decision(tuple(epoch_evidence), level=float(gate["entry_level"]),
                    decided_at_seconds=receipt.received_at_seconds, offset=model.offset, gain=model.gain,
                    sensor_noise_std=model.sensor_noise_std)
        else:
            decision = renewal_decision(evidence, level=float(gate["retain_level"]),
                decided_at_seconds=receipt.received_at_seconds, offset=model.offset, gain=model.gain,
                sensor_noise_std=model.sensor_noise_std)
        if decision is not None:
            decisions.append({**{k: v for k, v in decision.to_dict().items() if k not in ("schema_version",)},
                              "scan_id": plan.scan_id, "historical_mode": plan.mode})
            shadow_next.append("FOCUS" if decision.focus_allowed else "FULL")
        else:
            shadow_next.append(plan.mode if plan.mode == "FULL" else "FOCUS")
        rows.append({"scan_id": plan.scan_id, "mode": plan.mode, "epoch": int(plan.full_epoch),
                     "action_index": action_index if plan.mode == "FULL" else None, "alpha_action": alpha_action,
                     "epoch_alpha_allocated": epoch_alpha if plan.mode == "FULL" else None,
                     "started_at_seconds": plan.started_at_seconds,
                     "received_at_seconds": receipt.received_at_seconds,
                     "scheduled_slots": len(plan.samples), "delivered_slots": len(slots),
                     "contract_notes": list(notes), "fully_executed": evidence.fully_executed,
                     "complete_full_epoch": complete_full_epoch,
                     "mean_gate_value": float(plan.gate_max_mean), "mean_gate_positive": bool(plan.gate_max_mean > threshold),
                     "state_at_entry_level": evidence.state(float(gate["entry_level"])),
                     "state_at_retain_level": evidence.state(float(gate["retain_level"])),
                     "max_lower_bound": evidence.max_lower_bound, "max_upper_bound": evidence.max_upper_bound,
                     "entry_cutoff": observed_value_cutoff(float(gate["entry_level"]), gain=model.gain,
                         offset=model.offset, sensor_noise_std=model.sensor_noise_std, alpha_action=alpha_action,
                         scheduled_slots=len(plan.samples)),
                     "retain_cutoff": observed_value_cutoff(float(gate["retain_level"]), gain=model.gain,
                         offset=model.offset, sensor_noise_std=model.sensor_noise_std, alpha_action=alpha_action,
                         scheduled_slots=len(plan.samples)),
                     "supporting_readings": [reading.reading_id for reading in evidence.supporting(float(gate["entry_level"]))]})
    for position in range(len(scans) - 1):
        historical_next = scans[position + 1]["plan"]["mode"]
        if shadow_next[position] != historical_next:
            divergences.append({"after_scan": rows[position]["scan_id"],
                                "at_seconds": rows[position]["received_at_seconds"],
                                "historical_next_mode": historical_next, "shadow_next_mode": shadow_next[position],
                                "mean_gate_value": rows[position]["mean_gate_value"],
                                "evidence_state": rows[position]["state_at_entry_level"]})
    summary = {
        "scans": len(scans),
        "full_scans": sum(row["mode"] == "FULL" for row in rows),
        "focus_scans": sum(row["mode"] == "FOCUS" for row in rows),
        "complete_full_epochs": sum(row["complete_full_epoch"] for row in rows),
        "contract_violations": sum(bool(row["contract_notes"]) for row in rows),
        "mean_gate_positive_decisions": sum(row["mean_gate_positive"] for row in rows),
        "mean_positive_evidence_unsupported": sum(row["mean_gate_positive"]
            and row["state_at_entry_level"] != "supported" for row in rows),
        "evidence_supported_mean_negative": sum(not row["mean_gate_positive"]
            and row["state_at_entry_level"] == "supported" for row in rows),
        "entry_decisions_supported": sum(decision["kind"] == "full_epoch_entry" and decision["focus_allowed"]
                                         for decision in decisions),
        "entry_decisions": sum(decision["kind"] == "full_epoch_entry" for decision in decisions),
        "renewal_decisions_supported": sum(decision["kind"] == "focus_renewal" and decision["focus_allowed"]
                                           for decision in decisions),
        "renewal_decisions": sum(decision["kind"] == "focus_renewal" for decision in decisions),
        "prospective_divergences": len(divergences),
        "first_divergence_seconds": divergences[0]["at_seconds"] if divergences else None,
    }
    return {"artifact_type": "evidence_gated_focus_shadow_v1", "history": Path(path).as_posix(),
            "history_sha256": sha(path), "configuration_hash": record["configuration_hash"],
            "policy_id": config.session.policy_id, "scans": rows, "decisions": decisions,
            "divergences": divergences, "summary": summary,
            "belief_equivalence": belief_equivalence(record, session, candidate_session(document, session))}


def belief_equivalence(record, control_session, candidate_session_value):
    """The gate must not change belief on identical evidence."""
    left, right = EvidenceReplay(control_session), EvidenceReplay(candidate_session_value)
    for controller, session in ((left, control_session), (right, candidate_session_value)):
        controller._belief = DecayGaussianBelief(session.model, half_life_seconds=session.half_life_seconds,
                                                 transport_id=session.transport_id)
    maximum, mismatches = 0., []
    for index, event in enumerate(record["events"]):
        left.event(event)
        right.event(event)
        difference = max(float(np.max(np.abs(left._belief.mean - right._belief.mean))),
                         float(np.max(np.abs(left._belief.covariance - right._belief.covariance))))
        maximum = max(maximum, difference)
        if difference > 0 and len(mismatches) < 5:
            mismatches.append(index)
    return {"events": len(record["events"]), "max_abs_difference": maximum, "mismatch_events": mismatches,
            "identical": maximum == 0.}


def shadow(root):
    root = Path(root)
    document = study()
    results = []
    for profile in document["reference_configs"]:
        for case in document["shadow"]["cases"]:
            path = Path(document["shadow"]["path_template"].format(profile=profile, case=case))
            if not path.exists():
                results.append({"profile": profile, "case": case, "status": "unavailable", "path": path.as_posix()})
                continue
            record = read(path)
            target = root / "shadow" / f"{profile}-{case}.json"
            audit = None
            if target.exists():
                saved = read(target)
                if saved.get("history_sha256") == sha(path) and saved.get("code_sha256") == code_hash():
                    audit = saved
            if audit is None:
                audit = shadow_audit(record, path, document)
                audit["code_sha256"] = code_hash()
                audit["revision"] = revision()
                write(target, audit)
            results.append({"profile": profile, "case": case, "status": "completed",
                            "history_sha256": audit["history_sha256"], **audit["summary"]})
            print(profile, case, audit["summary"]["mean_positive_evidence_unsupported"], "unsupported decisions",
                  audit["summary"]["prospective_divergences"], "divergences", flush=True)
    status = "completed" if all(row["status"] == "completed" for row in results) else "incomplete"
    write(root / "shadow" / "index.json", {"artifact_type": "evidence_gated_focus_shadow_index_v1",
        "revision": revision(), "code_sha256": code_hash(), "status": status, "histories": results})
    print(json.dumps({"status": status, "histories": len(results)}))

def freeze(root, revise=False):
    root = Path(root)
    declaration = load_declaration(root)
    runs = sorted((root / "runs").glob("heldout-*.json")) if (root / "runs").exists() else []
    path = root / "freeze.json"
    if path.exists():
        existing = read(path)
        if existing["code_sha256"] == code_hash() and existing["study_sha256"] == sha(STUDY):
            print("Freeze already records the current code and study configuration")
            return
        if not revise:
            raise SystemExit("Operational code or configuration changed after the freeze; "
                             "rerun with --revise-freeze to invalidate held-out records explicitly")
    elif runs:
        raise SystemExit("Held-out records exist without a freeze; refusing to freeze retroactively")
    value = {"artifact_type": "evidence_gated_focus_freeze_v1", "revision": revision(), "code_sha256": code_hash(),
             "study_path": STUDY.as_posix(), "study_sha256": sha(STUDY),
             "declaration_sha256": sha(root / "declaration.json"),
             "heldout_manifest_sha256": sha(root / "heldout-manifest.json"),
             "reference_configs": declaration["reference_configs"], "gate": declaration["gate"],
             "policies": {"control": "full_fov_margin_loop_v2_candidate",
                          "candidate": EVIDENCE_GATE_POLICY_ID},
             "frozen_before_heldout_execution": not runs,
             "supersedes": ({"code_sha256": read(path)["code_sha256"], "revision": read(path)["revision"]}
                            if path.exists() else None),
             "invalidated_records": [item.name for item in runs] if path.exists() else []}
    write(path, value)
    print(json.dumps({k: value[k] for k in ("revision", "code_sha256", "frozen_before_heldout_execution")}))


def paired_rows(items):
    groups = {}
    for item in items:
        job = item["job"]
        groups.setdefault((job["stage"], job["profile"], job["case"], job["seed"], job["noise"]), {})[job["policy"]] = item
    rows = []
    for key in sorted(groups):
        stage, profile, case, seed, noise = key
        pair = groups[key]
        row = {"stage": stage, "profile": profile, "case": case, "seed": seed, "noise": noise}
        if set(pair) != {"control", "candidate"}:
            row.update({"status": "incomplete", "present": sorted(pair)})
            rows.append(row)
            continue
        control, candidate = pair["control"]["record"], pair["candidate"]["record"]
        left, right = control["metrics"], candidate["metrics"]

        def relative(first, second):
            return None if not first else (second - first) / first

        def difference(first, second):
            return None if first is None or second is None else second - first

        row.update({"status": "complete",
                    "J_over_T": {"control": control["J_over_T"], "candidate": candidate["J_over_T"],
                                 "relative_change": relative(control["J_over_T"], candidate["J_over_T"])},
                    "focus_core_geometric_coverage": {
                        "control": left["focus_core_geometric_coverage"],
                        "candidate": right["focus_core_geometric_coverage"],
                        "difference_points": difference(left["focus_core_geometric_coverage"],
                                                        right["focus_core_geometric_coverage"])},
                    "trailing_full_scan_core_coverage": {
                        "control": left["trailing_full_scan_core_coverage"],
                        "candidate": right["trailing_full_scan_core_coverage"],
                        "difference_points": difference(left["trailing_full_scan_core_coverage"],
                                                        right["trailing_full_scan_core_coverage"])},
                    "ever_sampled_core_coverage": {"control": left["ever_sampled_core_coverage"],
                                                   "candidate": right["ever_sampled_core_coverage"]},
                    "spent_seconds": {"control": left["spent_seconds"], "candidate": right["spent_seconds"],
                                      "relative_change": relative(left["spent_seconds"], right["spent_seconds"])},
                    "focus_scans": {"control": left["focus_scans"], "candidate": right["focus_scans"]},
                    "full_scans": {"control": left["full_scans"], "candidate": right["full_scans"]},
                    "gate_switches": {"control": left["gate_switches"], "candidate": right["gate_switches"]},
                    "actual_core_readings": {"control": left["actual_core_readings"],
                                             "candidate": right["actual_core_readings"]},
                    "false_focus_decisions": {"control": left["false_focus_decisions"],
                                              "candidate": right["false_focus_decisions"]},
                    "return_home_charged_seconds": {"control": left["return_home_charged_seconds"],
                                                    "candidate": right["return_home_charged_seconds"]},
                    "final_posterior_max_mean": {"control": left["final_posterior_max_mean"],
                                                 "candidate": right["final_posterior_max_mean"]},
                    "mean_positive_evidence_unsupported_decisions": {
                        "control": left["mean_positive_evidence_unsupported_decisions"],
                        "candidate": right["mean_positive_evidence_unsupported_decisions"]},
                    "change_metrics": {"control": control["change_metrics"], "candidate": candidate["change_metrics"]}})
        rows.append(row)
    return rows


def aggregate(rows, stage, profile, cases):
    selection = [row for row in rows if row["status"] == "complete" and row["stage"] == stage
                 and row["profile"] == profile and row["case"] in cases]
    if not selection:
        return {"records": 0, "status": "unavailable"}
    control_J = sum(row["J_over_T"]["control"] for row in selection)
    candidate_J = sum(row["J_over_T"]["candidate"] for row in selection)
    coverages = [(row["focus_core_geometric_coverage"]["control"], row["focus_core_geometric_coverage"]["candidate"])
                 for row in selection]
    defined = all(first is not None and second is not None for first, second in coverages)
    return {"records": len(selection), "status": "complete",
            "J_over_T": {"control": control_J, "candidate": candidate_J,
                         "relative_change": (candidate_J - control_J) / control_J if control_J else None},
            "focus_core_geometric_coverage": {
                "control": float(np.mean([first for first, _ in coverages])) if defined else None,
                "candidate": float(np.mean([second for _, second in coverages])) if defined else None,
                "difference_points": float(np.mean([second - first for first, second in coverages])) if defined else None,
                "all_defined": defined},
            "trailing_full_scan_core_coverage": {
                "control": float(np.mean([row["trailing_full_scan_core_coverage"]["control"] for row in selection])),
                "candidate": float(np.mean([row["trailing_full_scan_core_coverage"]["candidate"] for row in selection]))},
            "spent_seconds": {"control": sum(row["spent_seconds"]["control"] for row in selection),
                              "candidate": sum(row["spent_seconds"]["candidate"] for row in selection)},
            "case_breakdown": {case: {
                "J_over_T": {"control": sum(row["J_over_T"]["control"] for row in selection if row["case"] == case),
                             "candidate": sum(row["J_over_T"]["candidate"] for row in selection if row["case"] == case)}}
                for case in cases}}


def criteria(document, rows, items):
    tolerances = document["criteria"]
    results = {}
    development = [row for row in rows if row["stage"] == "development"]
    heldout = [row for row in rows if row["stage"] == "heldout"]

    blank_free = [row for row in development if row["case"] == "blank" and row["noise"] == 0]
    blank_free_focus = [row["focus_scans"]["candidate"] for row in blank_free if row["status"] == "complete"]
    results["noise_free_blank_never_focuses"] = {
        "status": ("supported" if blank_free_focus and all(count == 0 for count in blank_free_focus)
                   else "unsupported" if blank_free_focus else "unavailable"),
        "detail": {"records": len(blank_free_focus), "candidate_focus_scans": blank_free_focus}}

    unsupported = [row for row in development if row["status"] == "complete"]
    repeated = sum(item["record"]["metrics"]["focus_after_focus_without_support"] for item in items
                   if item["job"]["stage"] == "development" and item["job"]["policy"] == "candidate")
    results["unsupported_focus_forces_full"] = {
        "status": ("supported" if repeated == 0 else "unsupported") if unsupported else "unavailable",
        "detail": {"focus_after_focus_without_support": repeated,
                   "checked_records": len(unsupported)}}

    background = [row for row in development if row["case"] == "blank" and row["noise"] == 0 and row["status"] == "complete"]
    refocus = [item["record"]["metrics"]["refocus_after_background_full_scan"] for item in items
               if item["job"]["stage"] == "development" and item["job"]["policy"] == "candidate"
               and item["job"]["case"] == "blank" and item["job"]["noise"] == 0]
    results["completed_background_full_does_not_refocus"] = {
        "status": "supported" if refocus and all(count == 0 for count in refocus) else
                  "unsupported" if refocus else "unavailable",
        "detail": {"records": len(refocus), "refocus_counts": refocus}}

    noisy_blank = [row for row in heldout if row["case"] == "blank" and row["status"] == "complete"]
    noisy_blank_focus = {f"{row['profile']}-seed{row['seed']}-noise{row['noise']:g}": row["focus_scans"]["candidate"]
                         for row in noisy_blank}
    results["no_blank_false_focus_action"] = {
        "status": "supported" if noisy_blank_focus and all(count == 0 for count in noisy_blank_focus.values())
                  else "mixed" if noisy_blank_focus else "unavailable",
        "detail": noisy_blank_focus}

    cessation = [row for row in heldout if row["case"] == "cessation" and row["status"] == "complete"]
    cessation_detail = {}
    for row in cessation:
        entry = row["change_metrics"]["candidate"][0] if row["change_metrics"]["candidate"] else None
        key = f"{row['profile']}-seed{row['seed']}-noise{row['noise']:g}"
        cessation_detail[key] = None if entry is None else {
            "sufficient_remaining_time": entry["sufficient_remaining_time"],
            "time_to_completed_full_seconds": entry["time_to_completed_full_seconds"],
            "unavailable_reason": entry["unavailable_reason"]}
    expected = {key: value for key, value in cessation_detail.items() if value and value["sufficient_remaining_time"]}
    results["cessation_completes_post_change_full"] = {
        "status": ("supported" if expected and all(value["time_to_completed_full_seconds"] is not None
                                                   for value in expected.values())
                   else "unsupported" if expected else "unavailable"),
        "detail": cessation_detail}

    intermittent = [row for row in heldout if row["case"] == "intermittent" and row["status"] == "complete"]
    intermittent_detail = {}
    for row in intermittent:
        entry = next((item for item in row["change_metrics"]["candidate"] if item["kind"] == "resume"), None)
        key = f"{row['profile']}-seed{row['seed']}-noise{row['noise']:g}"
        intermittent_detail[key] = None if entry is None else {
            "sufficient_remaining_time": entry["sufficient_remaining_time"],
            "reacquisition_latency_seconds": entry["reacquisition_latency_seconds"],
            "reacquisition_focus_plan": entry["reacquisition_focus_plan"]}
    expected = {key: value for key, value in intermittent_detail.items() if value and value["sufficient_remaining_time"]}
    results["intermittent_reacquires_signal"] = {
        "status": ("supported" if expected and all(value["reacquisition_latency_seconds"] is not None
                                                   for value in expected.values())
                   else "mixed" if expected and any(value["reacquisition_latency_seconds"] is not None
                                                    for value in expected.values())
                   else "unsupported" if expected else "unavailable"),
        "detail": intermittent_detail}

    aggregates = {profile: aggregate(rows, "heldout", profile, ("continuous", "wind_error"))
                  for profile in document["reference_configs"]}
    preservation = {}
    for profile, value in aggregates.items():
        if value["status"] != "complete":
            preservation[profile] = {"status": "unavailable"}
            continue
        change = value["J_over_T"]["relative_change"]
        coverage_change = value["focus_core_geometric_coverage"]["difference_points"]
        if change is None or not value["focus_core_geometric_coverage"]["all_defined"] or coverage_change is None:
            preservation[profile] = {**value, "status": "unavailable"}
            continue
        passes = (change is not None and change <= tolerances["continuous_j_over_t_max_increase"]
                  and value["focus_core_geometric_coverage"]["all_defined"]
                  and coverage_change is not None
                  and coverage_change >= -tolerances["continuous_core_coverage_max_decrease_points"])
        preservation[profile] = {**value, "status": "supported" if passes else "unsupported"}
    results["continuous_operation_preserved"] = {"status": (
        "unsupported" if any(value["status"] == "unsupported" for value in preservation.values())
        else "supported" if preservation and all(value["status"] == "supported" for value in preservation.values())
        else "unavailable"), "detail": preservation}

    weak = [row for row in heldout if row["case"] == "weak" and row["status"] == "complete"]
    results["weak_signal_reported_separately"] = {"status": "complete", "detail": {
        f"{row['profile']}-seed{row['seed']}-noise{row['noise']:g}": {
            "J_over_T": row["J_over_T"], "focus_scans": row["focus_scans"],
            "focus_core_geometric_coverage": row["focus_core_geometric_coverage"]} for row in weak}}

    improved, equivalent, worse = 0, 0, 0
    for row in heldout:
        if row["status"] != "complete":
            continue
        baseline, candidate = row["J_over_T"]["control"], row["J_over_T"]["candidate"]
        tolerance = max(tolerances["numerical_tolerance"], tolerances["reconstruction_comparator_fraction"] * abs(baseline))
        if candidate < baseline - tolerance:
            improved += 1
        elif candidate > baseline + tolerance:
            worse += 1
        else:
            equivalent += 1
    results["reconstruction_classification"] = {"status": ("improved" if improved and not worse and not equivalent else
                                                            "mixed" if improved or equivalent else "worse")
                                                if improved or worse or equivalent else "unavailable",
                                                "detail": {"improved": improved, "equivalent": equivalent, "worse": worse,
                                                           "tolerance_rule": tolerances}}
    return results


def inspection_targets(document, job):
    section = document["inspection"]
    return (job["stage"] == section["stage"] and job["policy"] == "candidate"
            and job["noise"] == section["noise"]
            and [job["profile"], job["case"]] in [list(pair) for pair in section["pairs"]])


def verify_record(root, job):
    """Independent verification row for one declared record; resumable and hash-checked."""
    root = Path(root)
    path = root / "runs" / f"{job['name']}.json"
    if not path.exists():
        return {"name": job["name"], "stage": job["stage"], "status": "missing"}
    digest = sha(path)
    target = root / "checks" / "records" / f"{job['name']}.json"
    if target.exists():
        saved = read(target)
        if saved.get("run_sha256") == digest and saved.get("code_sha256") == code_hash():
            return saved
    row = {"name": job["name"], "stage": job["stage"], "sha256": digest, "code_sha256": code_hash(),
           "revision": revision()}
    try:
        config = configured(root, job)
        record = read(path)
        assert type(config).from_dict(record["configuration"]) == config, "Serialized configuration differs"
        assert record["configuration_hash"] == config.digest, "Configuration hash differs"
        assert record["code_sha256"] == code_hash(), "Producing code revision differs"
        assert record["study_sha256"] == sha(STUDY), "Study manifest differs"
        assert record["declaration_sha256"] == sha(root / "declaration.json"), "Declaration lineage differs"
        if job["stage"] == "heldout":
            assert record["freeze_file_sha256"] == sha(root / "freeze.json"), "Held-out freeze lineage differs"
        q = truth(root, config)
        replay = verify_replay(record, truth=q)
        recomputed = metrics(record, q, config, replay)
        assert recomputed == record["metrics"], "Stored metrics differ from a fresh replay"
        assert change_metrics(record, q, config) == record["change_metrics"], "Stored change metrics differ"
        if inspection_targets(study(), job):
            state = replay.gate_state()
            write(root / "traces" / f"{job['name']}.json",
                  {"artifact_type": "evidence_gated_focus_gate_trace_v1", "name": job["name"],
                   "configuration_hash": record["configuration_hash"], "gate_state": state.to_dict()})
        row.update({"status": "verified", "truth_key": record["truth_key"], "J_over_T": record["J_over_T"],
                    "metrics": recomputed, "change_metrics": record["change_metrics"]})
    except Exception as error:
        row.update({"status": "failed", "detail": f"{type(error).__name__}: {error}"})
    write(target, row)
    return row


def verify_records_phase(root, shard=None, shards=1):
    declaration = load_declaration(root)
    for index, job in enumerate(declaration["development"] + declaration["heldout"]):
        if shard is not None and index % shards != shard:
            continue
        row = verify_record(root, job)
        print(row["name"], row["status"], flush=True)


def run_records_verification(root, workers):
    if workers <= 1:
        verify_records_phase(root)
        return
    processes = [subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--phase", "verify-records",
                                   "--root", str(root), "--shard", str(shard), "--shards", str(workers)])
                 for shard in range(workers)]
    codes = [process.wait() for process in processes]
    if any(code != 0 for code in codes):
        raise SystemExit(f"Verification processes failed: {codes}")


def report(root, workers=1):
    root = Path(root)
    document = study()
    declaration = load_declaration(root)
    run_records_verification(root, workers)
    inventory, items = [], []
    for stage in ("development", "heldout"):
        for job in declaration[stage]:
            target = root / "checks" / "records" / f"{job['name']}.json"
            if not target.exists():
                inventory.append({"name": job["name"], "stage": stage, "status": "missing"})
                continue
            row = read(target)
            inventory.append(row)
            if row["status"] == "verified":
                items.append({"job": job, "record": read(root / "runs" / f"{job['name']}.json")})
    rows = paired_rows(items)
    expected = len(declaration["development"]) + len(declaration["heldout"])
    counts = {status: sum(row["status"] == status for row in inventory)
              for status in ("verified", "missing", "failed")}
    aggregates = {"development": {profile: aggregate(rows, "development", profile, tuple(document["development"]["cases"]))
                                 for profile in document["reference_configs"]},
                  "heldout": {profile: aggregate(rows, "heldout", profile, tuple(document["heldout"]["cases"]))
                              for profile in document["reference_configs"]},
                  "heldout_continuous": {profile: aggregate(rows, "heldout", profile, ("continuous", "wind_error"))
                                         for profile in document["reference_configs"]}}
    checks = criteria(document, rows, items)
    targets = [item["job"]["name"] for item in items if inspection_targets(document, item["job"])]
    behavior = {"noise_free_blank_never_focuses": checks["noise_free_blank_never_focuses"]["status"],
                "no_blank_false_focus_action": checks["no_blank_false_focus_action"]["status"],
                "cessation_completes_post_change_full": checks["cessation_completes_post_change_full"]["status"],
                "intermittent_reacquires_signal": checks["intermittent_reacquires_signal"]["status"]}
    result = {"artifact_type": "evidence_gated_focus_report_v1", "revision": revision(), "code_sha256": code_hash(),
              "study_sha256": sha(STUDY), "freeze": read(root / "freeze.json") if (root / "freeze.json").exists() else None,
              "gate_settings": declaration["gate"], "declared_full_scan_seconds": declaration["declared_full_scan_seconds"],
              "record_inventory": {"expected": expected, **counts,
                                   "missing": [row["name"] for row in inventory if row["status"] == "missing"],
                                   "failed": [row for row in inventory if row["status"] == "failed"]},
              "paired": rows, "aggregates": aggregates, "criteria": checks,
              "claims": {"implementation": "verified" if counts["verified"] == expected else "incomplete",
                         "behavior": behavior,
                         "reconstruction": checks["reconstruction_classification"]["status"],
                         "continuous_operation": checks["continuous_operation_preserved"]["status"]},
              "gate_traces": targets}
    write(root / "report.json", result)
    write(root / "hash-inventory.json", hash_inventory(root, declaration, inventory))
    write(root / "report.md", report_markdown(result, rows, inventory))
    print(json.dumps({"records": counts, "behavior": behavior,
                      "reconstruction": result["claims"]["reconstruction"]}, indent=1))
    if counts["verified"] != expected:
        raise SystemExit("Reporting found missing or failed records; see report.json")


def hash_inventory(root, declaration, inventory):
    rows = {"declaration": {"path": "declaration.json", "sha256": sha(root / "declaration.json")},
            "heldout_manifest": {"path": "heldout-manifest.json", "sha256": sha(root / "heldout-manifest.json")},
            "study": {"path": STUDY.as_posix(), "sha256": sha(STUDY)},
            "configs": [{"name": job["name"], "sha256": job["config_sha256"]}
                        for job in declaration["development"] + declaration["heldout"]],
            "runs": [{"name": row["name"], "stage": row["stage"], "sha256": row["sha256"],
                      "J_over_T": row.get("J_over_T"), "truth_key": row.get("truth_key")}
                     for row in inventory if row["status"] == "verified"]}
    for name in ("freeze.json", "checks/verify.json", "shadow/index.json", "report.json"):
        path = root / name
        rows[name] = {"sha256": sha(path)} if path.exists() else None
    return {"artifact_type": "evidence_gated_focus_hash_inventory_v1", "revision": revision(),
            "code_sha256": code_hash(), **rows}


def report_markdown(result, rows, inventory):
    lines = ["# Evidence-gated focus study report", "",
             f"- Producing revision: {result['revision']}", f"- Code hash: {result['code_sha256']}",
             f"- Study manifest hash: {result['study_sha256']}",
             f"- Verified records: {result['record_inventory']['verified']} of "
             f"{result['record_inventory']['expected']}",
             f"- Gate: alpha {result['gate_settings']['alpha']}, entry {result['gate_settings']['entry_level']}, "
             f"retain {result['gate_settings']['retain_level']}", "",
             "## Paired comparisons", "",
             "| stage | profile | case | seed | noise | J/T control | J/T candidate | change | coverage control | "
             "coverage candidate | focus scans (control/candidate) |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        if row["status"] != "complete":
            lines.append(f"| {row['stage']} | {row['profile']} | {row['case']} | {row['seed']} | {row['noise']:g} | "
                         "incomplete | | | | | |")
            continue
        coverage_left = row["focus_core_geometric_coverage"]["control"]
        coverage_right = row["focus_core_geometric_coverage"]["candidate"]
        lines.append(f"| {row['stage']} | {row['profile']} | {row['case']} | {row['seed']} | {row['noise']:g} | "
                     f"{row['J_over_T']['control']:.6f} | {row['J_over_T']['candidate']:.6f} | "
                     f"{row['J_over_T']['relative_change']:+.2%} | "
                     f"{'unavailable' if coverage_left is None else format(coverage_left, '.3f')} | "
                     f"{'unavailable' if coverage_right is None else format(coverage_right, '.3f')} | "
                     f"{row['focus_scans']['control']}/{row['focus_scans']['candidate']} |")
    lines += ["", "## Predeclared criteria", ""]
    for name, value in result["criteria"].items():
        lines.append(f"- {name}: {value['status']}")
    lines += ["", "## Behavior", ""]
    for name, value in result["claims"]["behavior"].items():
        lines.append(f"- {name}: {value}")
    return "\n".join(lines) + "\n"


def refine_loss(root, document, config, record, factor, job):
    shape = tuple(n * factor for n in config.evaluation_shape)
    step = config.evaluation_step_seconds / factor
    probe = replace(config, evaluation_shape=shape, evaluation_step_seconds=step)
    coarse = build_config(document, "coarse", job["case"], job["seed"], job["noise"], job["policy"])
    if tuple(config.session.instrument.lattice_shape) != tuple(coarse.session.instrument.lattice_shape):
        # Loss quadrature is a property of the common field, not scanner mechanics.
        assert (coarse.source == config.source and coarse.cessation_seconds == config.cessation_seconds
                and coarse.active_windows == config.active_windows and coarse.duration_seconds == config.duration_seconds
                and coarse.model.grid_shape == config.model.grid_shape), "Profile variants must share one truth field"
        probe = replace(coarse, evaluation_shape=shape, evaluation_step_seconds=step)
    else:
        assert coarse.source == config.source and coarse.model.grid_shape == config.model.grid_shape
    q = truth(root, probe)
    loss = OnlineLoss(q)
    controller = EvidenceReplay(config.session)
    controller._belief = DecayGaussianBelief(config.model, half_life_seconds=config.session.half_life_seconds,
                                             transport_id=config.session.transport_id)
    for event in record["events"]:
        loss.before(controller, event["at_seconds"])
        controller.event(event)
    loss.before(controller, config.duration_seconds + 1e-8)
    return loss.result()["J"]


def refinement(root, job, config, record, document):
    rows = [{"shape": list(config.evaluation_shape), "step": config.evaluation_step_seconds, "J": record["J"]}]
    accepted = False
    for factor in document["refinement"]["factors"]:
        value = refine_loss(root, document, config, record, factor, job)
        shape = tuple(n * factor for n in config.evaluation_shape)
        change = abs(value - rows[-1]["J"])
        tolerance = max(1e-10, .01 * abs(value))
        rows.append({"shape": list(shape), "step": config.evaluation_step_seconds / factor, "J": value,
                     "change": change, "tolerance": tolerance})
        if change <= tolerance:
            accepted = True
            break
    return {"name": job["name"], "stage": job["stage"], "profile": job["profile"], "case": job["case"],
            "policy": job["policy"], "input_sha256": sha(root / "runs" / f"{job['name']}.json"),
            "rows": rows, "status": "converged" if accepted else "numerically_unresolved",
            "code_sha256": code_hash(), "revision": revision()}


def refine(root, workers):
    root = Path(root)
    document = study()
    declaration = load_declaration(root)
    section = document["refinement"]
    targets = [job for job in declaration[section["stage"]]
               if job["case"] in section["cases"] and job["noise"] == section["noise"] and job["policy"] == section["policy"]]
    report_path = root / "report.json"
    near_tolerance = []
    if section.get("change_case_near_tolerance") and report_path.exists():
        for row in read(report_path)["paired"]:
            if (row["status"] == "complete" and row["case"] in section["change_cases"]
                    and abs(row["J_over_T"]["relative_change"]) <= section["near_tolerance_fraction"]):
                for job in declaration[row["stage"]]:
                    if (job["profile"], job["case"], job["seed"], job["noise"]) == (
                            row["profile"], row["case"], row["seed"], row["noise"]):
                        near_tolerance.append({**job, "reason": "change_case_near_tolerance"})
    jobs = targets + near_tolerance
    results = []
    for job in jobs:
        path = root / "refinement" / f"{job['name']}.json"
        if path.exists():
            saved = read(path)
            if saved["code_sha256"] == code_hash() and saved["input_sha256"] == sha(root / "runs" / f"{job['name']}.json"):
                results.append(saved)
                continue
        config = configured(root, job)
        record = read(root / "runs" / f"{job['name']}.json")
        assert record["configuration_hash"] == config.digest
        value = refinement(root, job, config, record, document)
        if job.get("reason"):
            value["reason"] = job["reason"]
        write(path, value)
        results.append(value)
        print(job["name"], value["status"], flush=True)
    write(root / "refinement" / "index.json", {"artifact_type": "evidence_gated_focus_refinement_index_v1",
        "revision": revision(), "code_sha256": code_hash(), "records": results})


def operational_hash(record):
    """Public-call operational content of a record: plans, readings, receipts and wind."""
    payload = {"configuration_hash": record["configuration_hash"],
               "events": [event for event in record["events"]]}
    return hashlib.sha256(json.dumps(jsonable(payload), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def render_video(root, name, *, fps, speed, variant):
    command = [sys.executable, "scripts/render-full-fov-demo.py", "--output", str(root), "--name", name,
               "--fps", str(fps), "--speed", str(speed)]
    if variant:
        command += ["--variant", variant]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise SystemExit(f"Renderer failed for {name}: {completed.stdout}{completed.stderr}")
    return completed.stdout.strip()


def media(root):
    root = Path(root)
    document = study()
    declaration = load_declaration(root)
    section = document["inspection"]
    records = []
    for profile, case in section["pairs"]:
        name = job_name(section["stage"], profile, case, int(document[section["stage"]]["seed"]),
                        float(section["noise"]), "candidate")
        path = root / "runs" / f"{name}.json"
        if not path.exists():
            records.append({"name": name, "status": "missing_execution"})
            continue
        before = operational_hash(read(path))
        digest_before = sha(path)
        render_video(root, name, fps=float(section["fps"]), speed=1., variant="")
        after = operational_hash(read(path))
        manifest = read(root / "inspection" / name / "manifest.json")
        frames = sum(item["frames"] for item in manifest["files"])
        records.append({"name": name, "profile": profile, "case": case, "status": "rendered",
                        "execution_sha256": digest_before, "execution_sha256_after": sha(path),
                        "operational_hash": before, "operational_hash_after": after,
                        "operational_hash_preserved": before == after,
                        "videos": manifest["files"], "decoded_frames": frames,
                        "keyframes": manifest["keyframes"],
                        "manifest": (root / "inspection" / name / "manifest.json").as_posix(),
                        "review_status": "pending_visual_review"})
        print(name, "rendered", len(manifest["files"]), "videos", flush=True)
    playback = None
    if records and records[0].get("status") == "rendered":
        name = records[0]["name"]
        before = operational_hash(read(root / "runs" / f"{name}.json"))
        render_video(root, name, fps=float(section["playback_fps"]), speed=float(section["playback_speed"]),
                     variant=f"playback-{section['playback_speed']:g}x")
        after = operational_hash(read(root / "runs" / f"{name}.json"))
        manifest = read(root / "inspection" / name / f"playback-{section['playback_speed']:g}x" / "manifest.json")
        playback = {"name": name, "fps": section["playback_fps"], "speed": section["playback_speed"],
                    "operational_hash": before, "operational_hash_after": after,
                    "operational_hash_preserved": before == after,
                    "videos": manifest["files"], "keyframes": manifest["keyframes"]}
        print(name, "alternate playback rendered", flush=True)
    write(root / "media" / "index.json", {"artifact_type": "evidence_gated_focus_media_index_v1",
        "revision": revision(), "code_sha256": code_hash(), "records": records, "alternate_playback": playback,
        "renderer": {"script": "scripts/render-full-fov-demo.py", "script_sha256": sha("scripts/render-full-fov-demo.py"),
                     "module_sha256": sha("src/plume_focus/sensing_rendering.py")}})

def run_stage(root, stage, shard=None, shards=1):
    """Records sharing one truth field stay in one shard, so each field is built once."""
    jobs = load_declaration(root)[stage]
    groups = {}
    for job in jobs:
        groups.setdefault((job["profile"], job["case"], job["seed"], job["noise"]), []).append(job)
    for index, key in enumerate(sorted(groups)):
        if shard is not None and index % shards != shard:
            continue
        for job in groups[key]:
            print(execute(str(root), job, stage), flush=True)


def run_stage_bounded(root, stage, workers):
    """Bounded parallelism without pipes: child processes inherit this console."""
    if workers <= 1:
        run_stage(root, stage)
        return
    processes = [subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--phase", stage,
                                   "--root", str(root), "--shard", str(shard), "--shards", str(workers)])
                 for shard in range(workers)]
    codes = [process.wait() for process in processes]
    if any(code != 0 for code in codes):
        raise SystemExit(f"Worker processes failed: {codes}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True,
                        choices=["declare", "verify", "shadow", "development", "freeze", "heldout", "report", "refine",
                                 "media", "verify-records"])
    parser.add_argument("--root", type=Path, default=Path("outputs/evidence-gated-focus"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--revise-freeze", action="store_true")
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    root = args.root
    if args.phase == "declare":
        declare(root)
    elif args.phase == "verify":
        verify(root)
    elif args.phase == "shadow":
        shadow(root)
    elif args.phase == "freeze":
        freeze(root, revise=args.revise_freeze)
    elif args.phase == "report":
        report(root, args.workers)
    elif args.phase == "verify-records":
        verify_records_phase(root, shard=args.shard, shards=args.shards)
    elif args.phase == "refine":
        refine(root, args.workers)
    elif args.phase == "media":
        media(root)
    elif args.shard is not None and args.phase in ("development", "heldout"):
        run_stage(root, args.phase, shard=args.shard, shards=args.shards)
    else:
        run_stage_bounded(root, args.phase, args.workers)


if __name__ == "__main__":
    main()
