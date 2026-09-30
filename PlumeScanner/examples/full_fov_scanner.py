"""Two public-API cycles using a declared full-FoV demo configuration."""
import argparse
import json
from pathlib import Path

from plume_focus.controller import SensingController
from plume_focus.full_fov_execution import decode_config
from plume_focus.sensing_messages_v2 import ScanRequestV1
from plume_focus.sensing_simulation import FakeScanner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = decode_config(json.loads(args.config.read_text(encoding="utf-8")))
    controller = SensingController.from_session(config.session)
    scanner = FakeScanner(config.build_source(), noise_std=config.simulated_noise_std,
                          seed=config.seed, gain=config.model.gain, offset=config.model.offset)
    now, pose = 0., config.session.home_yx
    for _ in range(2):
        plan = controller.plan_next(now, pose, ScanRequestV1(config.duration_seconds-now), wind=config.wind)
        print(plan.mode, plan.reason, len(plan.samples), plan.completed_at_seconds)
        if not plan.samples:
            break
        execution = scanner.execute(plan, delivery=config.delivery,
            delivery_delay=config.session.settings.delivery_delay_seconds,
            completion_overrun=config.session.settings.completion_overrun_seconds)
        # A completed batch preserves each reading's acquisition/availability time.
        controller.ingest_readings(execution.readings,
                                   available_at_seconds=execution.receipt.received_at_seconds)
        controller.finish_scan(execution.receipt)
        now, pose = execution.receipt.received_at_seconds, execution.receipt.final_position_yx


if __name__ == "__main__":
    main()
