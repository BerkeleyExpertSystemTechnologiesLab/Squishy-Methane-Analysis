"""Regenerate the two declared synthetic run fixtures; no hardware defaults."""

from dataclasses import asdict, replace
import json
from pathlib import Path

from plume_focus.sensing_contracts import CausalWind
from plume_focus.sensing_execution_v1 import ReconstructionGridV1, SensingRunConfigV1, SyntheticSourceV1
from plume_focus.sensing_session import InstrumentProfileV1, PolicyConfigV1, SensingSessionV1
from plume_focus.signal_belief import SignalModelConfig


def configurations():
    source = SyntheticSourceV1()
    source_position = tuple(v / (n - 1) for v, n in zip(source.source, source.shape))
    wind = CausalWind(velocity_yx=tuple(v * source.fps / (n - 1) for v, n in zip(source.wind_velocity, source.shape)), error_speed=.005)
    center_y, center_x = source_position
    corridor = (max(0., center_y - .18), max(0., center_x - .06), min(1., center_y + .18), min(1., center_x + .6))
    policy = PolicyConfigV1("mass90_column_v1", .90, 4, 5, 1e-8, 1e-10, .02, 1e-12, 60., .5, .05, corridor)
    root = Path(__file__).resolve().parents[1]
    template = json.loads((root / "docs/specs/instrument-profile-v1.example.json").read_text())
    for name, shape in (("coarse", (20, 32)), ("narrow", (220, 320))):
        payload = json.loads(json.dumps(template))
        payload["profile_id"] = f"synthetic_{name}_20hz_v1"
        payload["provenance"]["note"] = "Declared synthetic verification profile; not May hardware or a reference instrument."
        payload["lattice_shape"] = list(shape)
        payload["footprint"].update(height=1 / shape[0], width=1 / shape[1])
        payload["mechanics"].update(sweep_vertical_fov_per_second=20 / shape[0], move_vertical_fov_per_second=20 / shape[0],
                                    move_horizontal_fov_per_second=.625, settling_seconds_per_visit=0.)
        payload["acquisition"].update(cadence_hz=20., detector_capacity_hz=20., integration_seconds=.025)
        instrument = InstrumentProfileV1.from_dict(payload)
        session = SensingSessionV1("beam_reconstruction_v1", instrument,
                                   SignalModelConfig(signal_units="synthetic_concentration_path"), policy,
                                   source_position, "full_scene", (0., 0.), "seconds from synthetic session start",
                                   "Declared wind proxy in normalized FOV/s; source pixel velocity times fps divided by shape minus one",
                                   (1 / (shape[0] * shape[1]),) * (shape[0] * shape[1]))
        yield name, SensingRunConfigV1(session, source, wind, 0, 75., 12., 36., "stream", 0., 0.,
                                      ReconstructionGridV1((120, 160), .05, "uniform_cell_area", "left_endpoint"))


def main():
    output = Path(__file__).resolve().parents[1] / "configs/sensing-v1"
    output.mkdir(parents=True, exist_ok=True)
    for name, config in configurations():
        # Compact explicit quadrature arrays keep checked-in manifests manageable.
        (output / f"{name}.json").write_text(json.dumps(config.to_dict(), separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    suite = {"schema_version": 1, "artifact_type": "sensing_development_suite_v1",
             "profiles": ["coarse.json", "narrow.json"], "seeds": [0, 1],
             "policies": ["mass90_column_v1", "static_corridor_with_reacquisition_v1", "age_rescan_v1"],
             "cases": [{"name": "known-wind", "source_free": False, "wind_bias_yx": [0., 0.], "wind_error_speed": .005},
                       {"name": "wind-error", "source_free": False, "wind_bias_yx": [.012, -.015], "wind_error_speed": .02},
                       {"name": "source-free", "source_free": True, "wind_bias_yx": [0., 0.], "wind_error_speed": .005}],
             "source_free_corridor_yxyx": [.32, .19, .68, .85],
             "convergence": {"absolute_tolerance": 1e-10, "relative_tolerance": .01, "maximum_refinements": 2,
                             "fixtures": ["coarse-known-wind-mass90_column_v1-seed-0",
                                          "coarse-known-wind-static_corridor_with_reacquisition_v1-seed-0",
                                          "coarse-known-wind-age_rescan_v1-seed-0",
                                          "coarse-wind-error-mass90_column_v1-seed-0",
                                          "coarse-source-free-mass90_column_v1-seed-0",
                                          "narrow-known-wind-mass90_column_v1-seed-0"]}}
    (output / "development.json").write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
