"""Two scans through the same serializable interface a hardware adapter can use.

Run: .venv/Scripts/python examples/sensing_controller.py
"""

import json
import numpy as np

from plume_focus.controller import (
    CausalWind, ExecutionReceipt, ScanConstraints, ScanPlan, ScannerCapabilities,
    SensorReading, SensingController, SignalModelConfig,
)
from plume_focus.sensing_simulation import CleanMethaneSource, FakeScanner


def transport(message):
    return type(message).from_dict(json.loads(json.dumps(message.to_dict(), allow_nan=False)))


def main():
    source = CleanMethaneSource()
    source_position = tuple(np.array(source.config.source) / (np.array(source.config.shape) - 1))
    velocity = tuple(np.array(source.config.wind_velocity) * source.config.fps / (np.array(source.config.shape) - 1))
    wind = CausalWind(velocity_yx=velocity, error_speed=.005)
    model = SignalModelConfig()
    controller = SensingController(transport(ScannerCapabilities()), transport(model), source_position=source_position)
    adapter = FakeScanner(source, seed=0, noise_std=model.sensor_noise_std)
    now, pose = 0., (0., 0.)
    for scan_index in range(2):
        plan = controller.plan_next(now, pose, ScanConstraints(36 if scan_index == 0 else 12), wind=transport(wind))
        if not plan.samples:
            print(plan.reason)
            break
        execution = adapter.execute(transport(plan))
        for reading in execution.readings:
            controller.ingest_readings([transport(reading)], available_at_seconds=reading.available_at_seconds)
        view = controller.finish_scan(transport(execution.receipt))
        print(json.dumps({"scan_id": plan.scan_id, "readings": len(execution.readings),
                          "duration_seconds": plan.duration_seconds, "status": plan.status,
                          "predicted_mass_recall": plan.predicted_mass_recall, "posterior_time": view.time_seconds}))
        now, pose = execution.receipt.received_at_seconds, execution.receipt.final_position_yx


if __name__ == "__main__":
    main()
