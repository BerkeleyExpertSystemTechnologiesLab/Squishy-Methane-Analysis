from dataclasses import replace
import json

import pytest

from plume_focus.experiment_runner import ExperimentRequest, run_experiment
from plume_focus.model_profiles import (
    MeasurementModelConfig,
    ModelProfile,
    synthetic_demo_v1,
)


def test_synthetic_methane_profile_serializes_declared_and_resolved_values() -> None:
    profile = synthetic_demo_v1("methane")

    payload = profile.to_payload(frame_rate_hz=12.0)

    assert payload["profile_id"] == "synthetic_demo_v1"
    assert payload["calibration_status"] == "uncalibrated"
    assert payload["measurement"]["foreground_threshold_ratio"] == 0.22
    assert payload["scan_policy"]["credible_mass"] == 0.93
    assert (
        payload["uncertainty"]["resolved"][
            "wind_error_speed_px_per_second"
        ]
        == 3.0
    )
    assert (
        payload["uncertainty"]["resolved"][
            "plume_spread_speed_px_per_second"
        ]
        == pytest.approx(5.4)
    )


def test_model_profile_rejects_invalid_measurement_and_scenario() -> None:
    with pytest.raises(ValueError, match="below"):
        MeasurementModelConfig(
            foreground_threshold_ratio=0.2,
            intensity_lower_percentile=99.0,
            intensity_upper_percentile=5.0,
        )

    base = synthetic_demo_v1("methane")
    with pytest.raises(ValueError, match="scenario"):
        ModelProfile(
            profile_id="invalid",
            scenario="invalid",  # type: ignore[arg-type]
            intended_use=base.intended_use,
            provenance=base.provenance,
            dynamics=base.dynamics,
            measurement=base.measurement,
            uncertainty=base.uncertainty,
            scan_policy=base.scan_policy,
        )


def test_experiment_report_contains_the_complete_resolved_profile() -> None:
    base = synthetic_demo_v1("methane")
    profile = replace(base, profile_id="test-profile")

    report = run_experiment(
        ExperimentRequest(
            scenario="methane",
            max_scans=1,
            vertical_pass_seconds=0.005,
            horizontal_step_seconds=0.0,
            model_profile=profile,
        )
    ).report
    payload = json.loads(report.to_json())

    assert report.model_profile["profile_id"] == "test-profile"
    assert payload["model_profile"]["dynamics"][
        "source_emission_rate_per_second"
    ] == 0.24
    assert payload["model_profile"]["measurement"][
        "neutral_likelihood"
    ] == 0.35
    assert payload["model_profile"]["scan_policy"][
        "force_known_source_in_scan"
    ]
    assert payload["attention_contours"]["planning_effect"] == "none"
    assert (
        payload["attention_contours"]["calibration"]
        == "relative density; not calibrated confidence"
    )


def test_experiment_rejects_a_profile_for_another_scenario() -> None:
    with pytest.raises(ValueError, match="scenario must match"):
        run_experiment(
            ExperimentRequest(
                scenario="object",
                max_scans=1,
                model_profile=synthetic_demo_v1("methane"),
            )
        )
