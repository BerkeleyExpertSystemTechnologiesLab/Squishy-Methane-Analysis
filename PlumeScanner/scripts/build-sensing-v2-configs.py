"""Declare experimental v2 fixtures before testing; no hardware calibration claim."""

from dataclasses import asdict
import json
from pathlib import Path

from plume_focus.sensing_cli_v1 import load_config
from plume_focus.sensing_execution_v2 import SensingRunConfigV2, SyntheticSourceV2
from plume_focus.sensing_session import InstrumentProfileV1
from plume_focus.sensing_session_v2 import PolicyConfigV2, SensingSessionV2


def main():
    root = Path(__file__).resolve().parents[1]
    base = load_config(root / "configs/sensing-v1/coarse.json")
    output = root / "configs/sensing-v2"
    output.mkdir(parents=True, exist_ok=True)
    p = asdict(base.session.policy)
    p.update(policy_id="threshold90_segment_v2", reportable_threshold=.15,
             threshold_provenance="Predeclared synthetic blank-response convention: three times 0.05 scalar std; not a hardware or plume-detection calibration.",
             survey_height_fov=.2, survey_interval_seconds=30.)
    policy = PolicyConfigV2(**p)
    for name, shape, startup in (("coarse-full-startup", (20, 32), "full_scene"),
                                  ("double-prior-startup", (40, 64), "prior"),
                                  ("double-full-startup", (40, 64), "full_scene")):
        instrument = base.session.instrument.to_dict()
        instrument["profile_id"] = "synthetic_coarse_20hz_v1" if shape == (20, 32) else "tentative_40x64_20hz"
        instrument["lattice_shape"] = shape
        instrument["footprint"].update(height=1/shape[0], width=1/shape[1])
        instrument["mechanics"].update(sweep_vertical_fov_per_second=20/shape[0], move_vertical_fov_per_second=20/shape[0])
        if shape == (40, 64):
            instrument["provenance"]["note"] = "User-requested tentative doubled physical lattice; same 20 Hz detector, explicit 0.5 FOV/s sweep; unconfirmed hardware."
        s = asdict(base.session)
        s.update(formulation_id="beam_reconstruction_v2_candidate", instrument=InstrumentProfileV1.from_dict(instrument),
                 model=base.model, policy=policy, startup_mode=startup,
                 physical_quadrature_weights=(1/(shape[0]*shape[1]),)*(shape[0]*shape[1]))
        session = SensingSessionV2(**s)
        c = asdict(base)
        c.update(session=session, source=SyntheticSourceV2(**asdict(base.source)), wind=base.wind, evaluation=base.evaluation, simulated_noise_std=.05,
                 simulated_noise_provenance="Independent Gaussian retrieved-scalar error, seed/time/pose coupled across policies; unchanged from the v1 fixture.")
        config = SensingRunConfigV2(**c)
        (output / f"{name}.json").write_text(json.dumps(config.to_dict(), separators=(",", ":")) + "\n", encoding="utf-8")
    suite = {"artifact_type":"mechanism_revision_development_v2", "profiles":["coarse-full-startup.json", "double-prior-startup.json"],
             "seeds":[0,1], "policies":["threshold90_segment_v2", "static_corridor_segment_v2", "age_segment_v2"],
             "cases":[{"name":"known-wind","source_free":False,"wind_bias_yx":[0.,0.],"wind_error_speed":.005},
                      {"name":"wind-error","source_free":False,"wind_bias_yx":[.012,-.015],"wind_error_speed":.02},
                      {"name":"source-free","source_free":True,"wind_bias_yx":[0.,0.],"wind_error_speed":.005}],
             "source_free_corridor_yxyx":[.32,.19,.68,.85],
             "evaluation":"Use existing accepted 120x160 / 0.05 s mesh; check representative v2 refinements before accepting comparisons.",
             "supplementary":["coarse zero-threshold segment ablation", "coarse 2-sigma threshold", "coarse 4-sigma threshold",
                              "coarse full-height surveillance",
                              "double full-startup resource check", "source-free weak signal x0.1", "source-free blank signal x0"]}
    (output / "development.json").write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
