"""JSON round trips through plan -> readings -> completion -> next plan.

Run: .venv/Scripts/python examples/sensing_controller_v1.py --run-config configs/sensing-v1/coarse.json
The configuration is synthetic, not a hardware profile.
"""

import argparse
import json
from pathlib import Path

from plume_focus.controller import SensingController
from plume_focus.sensing_cli_v1 import load_config
from plume_focus.sensing_messages_v2 import ScanRequestV1
from plume_focus.sensing_simulation import FakeScanner


def transport(message):
    return type(message).from_dict(json.loads(json.dumps(message.to_dict(), allow_nan=False)))


def execute_example(config, count=2):
    config = transport(config)
    controller = SensingController.from_session(transport(config.session))
    scanner = FakeScanner(config.source.build(), noise_std=getattr(config, "simulated_noise_std", config.model.sensor_noise_std),
                          gain=config.model.gain, offset=config.model.offset, seed=config.seed)
    now, pose, results = 0., config.session.initial_position_yx, []
    for index in range(count):
        request = ScanRequestV1(min(config.initial_budget_seconds if index == 0 else config.budget_seconds,
                                    config.duration_seconds - now))
        plan = controller.plan_next(now, pose, transport(request), wind=transport(config.wind))
        if not plan.samples:
            break
        actual = scanner.execute(transport(plan), delivery=config.delivery, delivery_delay=config.delivery_delay_seconds,
                                 completion_overrun=config.completion_overrun_seconds)
        if config.delivery == "batch":
            controller.ingest_readings([transport(r) for r in actual.readings], available_at_seconds=actual.receipt.received_at_seconds)
        else:
            for reading in actual.readings:
                controller.ingest_readings([transport(reading)], available_at_seconds=reading.available_at_seconds)
        view = controller.finish_scan(transport(actual.receipt))
        results.append((plan, view))
        now, pose = actual.receipt.received_at_seconds, actual.receipt.final_position_yx
        if actual.exhausted:
            break
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.run_config)
    for plan, view in execute_example(config):
        print(json.dumps({"scan_id": plan.scan_id, "status": plan.status, "duration_seconds": plan.duration_seconds,
                          "predicted_mass_recall": plan.predicted_mass_recall, "tracking_shortfall": plan.tracking_shortfall,
                          "posterior_time_seconds": view.time_seconds}))
