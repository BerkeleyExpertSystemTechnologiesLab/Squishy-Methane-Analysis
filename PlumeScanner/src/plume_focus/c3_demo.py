"""One predeclared synthetic integration fixture, never a performance study."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np

from plume_focus.active_rescan import plume
from plume_focus.active_rescan_study import make_config
from plume_focus.c2_r2_dynamic import R2Counterfactual, ready_payload
from plume_focus.c3_active_sensing import (
    FullHistoryBank, GuardedTwoStagePlanner, ReadyContext, ServiceContract,
)
from plume_focus.sensing_contracts import CausalWind
from plume_focus.sensing_messages_v2 import ScanRequestV1
from plume_focus.sensing_simulation import FakeScanner


def demonstration_config(root):
    cfg = json.loads((root / 'configs/active-rescan/protocol.json').read_text(encoding='utf-8'))
    cfg.update(source_grid_y=[.3, .7], source_grid_x=[.5], onset_grid_count=2,
               forecast_draws=4, old_strengths=[.6, 1.4], wind_biases=[0.], new_strengths=[.6])
    return cfg


class DemoSignal:
    """Fixed, disclosed source; only FakeScanner sees this object."""
    end_time_seconds = None

    def signal_at(self, times, positions_yx):
        return (plume(positions_yx, (.5, .15))
                + .6 * plume(positions_yx, (.3, .5)) * (np.asarray(times) >= 2.5))


def run_demo(package_root, output):
    root, output = Path(package_root), Path(output)
    # Reject overwrite before generating any fixture or deciding a route.
    output.mkdir(parents=True, exist_ok=False)
    cfg = demonstration_config(root)
    run = make_config(root, cfg)
    native = run.make_controller()
    history_plan = native.plan_next(0., run.session.home_yx, ScanRequestV1(100.),
                                   wind=CausalWind(velocity_yx=(0., .02)))
    signal = DemoSignal()
    history = FakeScanner(signal, noise_std=cfg['noise_std'], seed=74100).execute(
        history_plan, delivery_delay=cfg['delivery_delay_seconds'])
    for reading in history.readings:
        native.ingest_readings([reading], available_at_seconds=reading.available_at_seconds)
    native.finish_scan(history.receipt)
    saved = ready_payload(native, provenance={'kind': 'fixed_C3_software_fixture'})
    t0 = history.receipt.received_at_seconds
    bank = FullHistoryBank.from_history(cfg, [(history_plan, history.readings)], t0)
    before = dict(history_log_odds=bank.history_log_odds, new_log_bf=bank.evidence,
                  label_probabilities=bank.label_probabilities().tolist())
    contract = ServiceContract(deadline=t0 + 15., coverage_due=t0 + 12., terminal_reserve=3.,
                               first_planning_fee=.25, second_planning_fee=.25)
    service = GuardedTwoStagePlanner(bank, run.session.instrument, contract)
    last = history.readings[-1]
    context = ReadyContext(t0, tuple(history.receipt.final_position_yx), last.acquired_at_seconds,
                           last.integration_seconds, gate_allows_service=bool(native.gate_focus_allowed),
                           active_scan=native.active_plan is not None,
                           full_obligation=native._full_missing is not None)
    counter = R2Counterfactual(saved)
    stages = []
    for stage in range(2):
        decision = service.choose(context)
        entry = dict(stage=stage + 1, reason=decision.reason, charged_fee=decision.charged_fee,
                     planner_wall_seconds=decision.planner_wall_seconds)
        if decision.candidate is None:
            stages.append(entry)
            break
        plan = decision.candidate['plan']
        # Use the existing research executor; no production factory/default edits.
        counter._belief.advance(plan.started_at_seconds, counter._wind, counter.source_position)
        counter._last_available = plan.started_at_seconds
        counter.begin_record(decision.record)
        execution = FakeScanner(signal, noise_std=cfg['noise_std'], seed=74101 + stage).execute(
            plan, delivery_delay=cfg['delivery_delay_seconds'])
        for reading in execution.readings:
            counter.ingest_readings([reading], available_at_seconds=reading.available_at_seconds)
            service.ingest(reading, available_at=reading.available_at_seconds)
        counter.finish_scan(execution.receipt)
        context = service.finish(execution.receipt, received_at=execution.receipt.received_at_seconds)
        # Same one fixture permission, not an assertion of real receiver acceptance.
        entry.update(name=decision.candidate['name'], target=decision.candidate['target'],
                     started_at=plan.started_at_seconds, receipt_at=context.at,
                     n_readings=len(execution.readings), new_log_bf=bank.evidence,
                     record=decision.record, readings=[r.to_dict() for r in execution.readings],
                     receipt=execution.receipt.to_dict())
        stages.append(entry)
    source = Path(__file__).with_name('c3_active_sensing.py')
    result = dict(schema='c3_software_fixture_v1', fixture='fixed_north_74100',
                  scope='Software integration only. No new performance/adoption/false-alarm claim.',
                  research_executor='R2Counterfactual; native controller integration pending',
                  posterior_mode='full_history_design_with_separate_new_confirmation_evidence',
                  terminal_reserve_note='Illustrative 3s reservation, not a measured HOME+FULL bound; not executed.',
                  completion_note='Two local actions do not grant FOCUS or discharge native FULL obligations.',
                  ready_at=t0, native_initial_gate_allowed=bool(native.gate_focus_allowed),
                  contract=asdict(contract), initial=before, stages=stages,
                  history_plan=history_plan.to_dict(), history_readings=[r.to_dict() for r in history.readings],
                  history_receipt=history.receipt.to_dict(), cfg=cfg,
                  planner_audit=service.audit, final_label_probabilities=bank.label_probabilities().tolist(),
                  environment=dict(python=platform.python_version(),
                                   numpy=importlib.metadata.version('numpy'),
                                   scipy=importlib.metadata.version('scipy')),
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    (output / 'RESULTS.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/c3-software-demo'))
    args = parser.parse_args()
    result = run_demo(Path(__file__).resolve().parents[2], args.output)
    print('C3 fixed synthetic software fixture; not a performance study.')
    for stage in result['stages']:
        print(f"Stage {stage['stage']}: {stage['reason']}; start={stage.get('started_at')}; receipt={stage.get('receipt_at')}")
    print(f"Evidence: {args.output / 'RESULTS.json'}")


if __name__ == '__main__':
    main()
