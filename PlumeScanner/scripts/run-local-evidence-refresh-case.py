"""Compare two actions from the same saved C3 READY prefix."""

import argparse
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plume_focus.evidence_refresh import EvidenceRefreshRequest
from plume_focus.full_fov_execution import decode_config, decode_plan
from plume_focus.local_evidence_refresh import LocalSourceActivityRefreshScanner
from plume_focus.physical_scan import PhysicalPlan
from plume_focus.sensing_contracts import CausalWind, ExecutionReceipt, SensorReading
from plume_focus.sensing_messages_v2 import ScanRequestV1
from plume_focus.sensing_simulation import FakeScanner
from plume_focus.source_activity_refresh import SourceActivityRefreshScanner


def replay_startup(scanner, raw):
    """Feed recorded deliveries up to the first complete native FULL receipt."""
    opening = raw["events"][0]
    recorded = decode_plan(opening["plan"])
    plan = scanner.plan_next(
        opening["at_seconds"], tuple(opening["position_yx"]),
        ScanRequestV1(opening["remaining_seconds"]), wind=CausalWind.from_dict(opening["wind"]),
    )
    if plan.mode != "FULL" or any(
        getattr(plan, field.name) != getattr(recorded, field.name) for field in fields(PhysicalPlan)
    ):
        raise ValueError("Common startup physical plan does not match the saved C3 prefix.")
    for event in raw["events"][1:]:
        if event["operation"] == "finish_scan":
            receipt = ExecutionReceipt.from_dict(event["receipt"])
            scanner.finish_scan(receipt)
            if not scanner.startup_complete or not scanner.gate_focus_allowed:
                raise ValueError("Saved startup prefix does not support the required READY gate state.")
            return receipt
        if event["operation"] != "ingest_readings":
            raise ValueError("Unexpected operation before the first FULL receipt.")
        scanner.ingest_readings(
            [SensorReading.from_dict(row) for row in event["readings"]],
            available_at_seconds=event["at_seconds"],
        )
    raise ValueError("No startup completion receipt in the saved C3 case.")


def execute(scanner, simulator, plan, request, planning_seconds):
    """Evaluate the existing emitter at this path's own exposure times."""
    settings = scanner.session.settings
    run = simulator.execute(
        plan, delivery="stream", delivery_delay=settings.delivery_delay_seconds,
        completion_overrun=settings.completion_overrun_seconds,
    )
    targets = []
    for reading in run.readings:
        exposure_start = reading.acquired_at_seconds - reading.integration_seconds / 2
        is_target = (scanner._physical_cell(reading.position_yx) == request.cells[0]
                     and exposure_start >= request.fresh_since_seconds - 1e-10)
        before = scanner.source_activity_state() if is_target else None
        events = scanner.ingest_readings([reading], available_at_seconds=reading.available_at_seconds)
        if is_target:
            if tuple(event.reading_id for event in events) != (reading.reading_id,):
                raise ValueError("Fresh target reading did not enter the native B1 filter exactly once.")
            targets.append(dict(
                reading_id=reading.reading_id, exposure_start_seconds=exposure_start,
                accepted_at_seconds=reading.available_at_seconds,
                value=reading.value, b1_before=asdict(before),
                b1_after=asdict(scanner.source_activity_state()),
            ))
    scanner.finish_scan(run.receipt)
    return run.receipt, dict(
        plan=plan.to_dict(), readings=[reading.to_dict() for reading in run.readings],
        receipt=run.receipt.to_dict(), targets=targets, planning_wall_seconds=planning_seconds,
        focus_allowed_after_receipt=scanner.gate_focus_allowed,
        completed_full_scans=scanner.completed_full_scans,
    )


def terminal_full(scanner, receipt, horizon, preparation, wind):
    """Request the independently required native FULL at the current receipt."""
    now = receipt.received_at_seconds
    rows, columns = scanner.capabilities.lattice_shape
    coverage = EvidenceRefreshRequest(
        "terminal-full", scanner.session.instrument.digest, tuple(range(rows * columns)), now, horizon,
    )
    decision = scanner.plan_evidence_refresh(
        coverage, now, receipt.final_position_yx, ScanRequestV1(horizon - now),
        preparation_seconds=preparation, wind=wind,
    )
    if decision.plan is None or decision.plan.mode != "FULL":
        raise ValueError(f"Required terminal FULL could not open: {decision.outcome}.")
    return decision.plan


def run_arm(scanner, simulator, prefix, contract, wind, *, use_local):
    ready = prefix.received_at_seconds
    prep = contract["preparation_seconds_per_opened_action"]
    horizon = contract["session_horizon_seconds"]
    request = scanner.source_activity_request(
        "case-A-source", fresh_since_seconds=contract["fresh_since_seconds"],
        deadline_seconds=contract["observation_deadline_seconds"],
    )
    if request.cells != (contract["target_cell"],):
        raise ValueError("Declared source cell differs from the frozen request.")
    args = (request, ready, prefix.final_position_yx, ScanRequestV1(horizon - ready))
    begin = perf_counter()
    if use_local:
        decision = scanner.plan_local_evidence_refresh(*args, terminal="home_full", preparation_seconds=prep, wind=wind)
    else:
        decision = scanner.plan_evidence_refresh(*args, preparation_seconds=prep, wind=wind)
    # The four-second evidence request and the mandatory mission FULL are
    # independent. A missed request still allows the required coverage action.
    first_plan = decision.plan if decision.plan is not None else terminal_full(scanner, prefix, horizon, prep, wind)
    receipt, first = execute(scanner, simulator, first_plan, request, perf_counter() - begin)
    phases = [first]
    if first_plan.mode == "REFRESH":
        if scanner.gate_focus_allowed or scanner.completed_full_scans != 1:
            raise ValueError("Local action changed native FOCUS permission or FULL coverage credit.")
        begin = perf_counter()
        recovery_plan = terminal_full(scanner, receipt, horizon, prep, wind)
        receipt, recovery = execute(scanner, simulator, recovery_plan, request, perf_counter() - begin)
        phases.append(recovery)
    if scanner.completed_full_scans != 2 or scanner.full_scan_missing or scanner.active_plan is not None:
        raise ValueError("Arm did not reach a complete native FULL terminal receipt.")
    if receipt.received_at_seconds > horizon + 1e-9:
        raise ValueError("Arm exceeded the shared remaining-session budget.")
    target = min((row for phase in phases for row in phase["targets"]), key=lambda row: row["accepted_at_seconds"])
    status = scanner.evidence_status(request, now_seconds=receipt.received_at_seconds)
    summary = dict(
        request_outcome=decision.outcome, local_selection_reason=decision.local_reason,
        estimated_acceptance_seconds=decision.estimated_last_acceptance_seconds,
        reserved_duration_seconds=decision.required_duration_seconds,
        actual_request_status=status.state, first_target=target,
        seconds_to_first_target=target["accepted_at_seconds"] - ready,
        seconds_to_terminal_receipt=receipt.received_at_seconds - ready,
        terminal_pose_yx=receipt.final_position_yx, terminal_focus_allowed=scanner.gate_focus_allowed,
        terminal_full_cells=len(phases[-1]["readings"]),
        planning_within_allowance=all(phase["planning_wall_seconds"] <= prep for phase in phases),
        action_modes=[phase["plan"]["mode"] for phase in phases],
        readings_per_action=[len(phase["readings"]) for phase in phases],
    )
    return summary, phases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    results_path, trace_path = args.output / "RESULTS.json", args.output / "TRACES.json"
    if results_path.exists() or trace_path.exists():
        raise FileExistsError("Use a fresh result directory for another execution.")

    case_bytes = args.case.read_bytes()
    contract = json.loads(case_bytes)
    data = (args.project_root / contract["raw_path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != contract["raw_sha256"]:
        raise ValueError("Saved C3 input hash differs from the frozen case.")
    raw = json.loads(data)
    config = decode_config(raw["configuration"])
    if (raw["c3_scene"] != "A" or raw["c3_arm"] != "adaptive" or config.digest != contract["raw_configuration_hash"]
            or config.delivery != "stream" or config.seed != contract["seed"]
            or config.simulated_noise_std != contract["noise_std"]):
        raise ValueError("The fixed C3 comparison requires the frozen A / adaptive streaming case.")
    # Scene A's existing continuous emitter has no additional source or emission windows.
    source = config.source.build().source
    simulator = FakeScanner(source, noise_std=config.simulated_noise_std, seed=config.seed,
                            gain=config.model.gain, offset=config.model.offset)

    baseline = SourceActivityRefreshScanner(config.session, delivery=config.delivery)
    candidate = LocalSourceActivityRefreshScanner(config.session, delivery=config.delivery)
    baseline_prefix = replay_startup(baseline, raw)
    candidate_prefix = replay_startup(candidate, raw)
    if (baseline_prefix != candidate_prefix or baseline_prefix.received_at_seconds != contract["ready_at_seconds"]
            or list(baseline_prefix.final_position_yx) != contract["ready_pose_yx"]
            or baseline.source_activity_state() != candidate.source_activity_state()
            or baseline.gate_state() != candidate.gate_state()):
        raise ValueError("The two arms must start with identical delivered evidence and READY state.")

    full, full_trace = run_arm(baseline, simulator, baseline_prefix, contract, config.wind, use_local=False)
    local, local_trace = run_arm(candidate, simulator, candidate_prefix, contract, config.wind, use_local=True)
    if any(full[key] != local[key] for key in ("terminal_pose_yx", "terminal_focus_allowed", "terminal_full_cells")):
        raise ValueError("The two arms did not reach the same required terminal state.")
    package = Path(__file__).resolve().parents[1]
    module_paths = [package / "src" / "plume_focus" / f"{name}.py" for name in (
        "evidence_refresh", "local_evidence_refresh", "source_activity",
        "source_activity_refresh", "c3_refresh", "physical_scan", "sensing_simulation",
    )]
    code_hashes = {path.relative_to(package).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in (*module_paths, Path(__file__).resolve())}
    summary = dict(
        scope=contract["scope"], contract_sha256=hashlib.sha256(case_bytes).hexdigest(),
        raw_sha256=contract["raw_sha256"], python=platform.python_version(), code_sha256=code_hashes,
        same_terminal=True, full_only=full, local_candidate=local,
        target_seconds_earlier=full["seconds_to_first_target"] - local["seconds_to_first_target"],
        terminal_seconds_saved=full["seconds_to_terminal_receipt"] - local["seconds_to_terminal_receipt"],
    )
    args.output.mkdir(exist_ok=True, parents=True)
    with trace_path.open("x", encoding="utf-8") as stream:
        json.dump({"full_only": full_trace, "local_candidate": local_trace}, stream, indent=2, allow_nan=False)
    with results_path.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    print(json.dumps({key: summary[key] for key in (
        "same_terminal", "target_seconds_earlier", "terminal_seconds_saved",
    )}, indent=2))


if __name__ == "__main__":
    main()
