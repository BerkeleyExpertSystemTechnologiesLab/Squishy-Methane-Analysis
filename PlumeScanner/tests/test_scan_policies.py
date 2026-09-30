from dataclasses import dataclass, field

import numpy as np

from plume_focus.experiment_runner import ExperimentRequest, run_experiment
from plume_focus.motion import MotionEstimate
from plume_focus.scan_policies import (
    AdaptivePredictivePolicy,
    PolicyForecast,
    PolicyObservation,
    ScanBudget,
    StaticSourceCorridorPolicy,
    allocate_exact_column_mask,
)
from plume_focus.video_sources import SyntheticMethaneLeakConfig


@dataclass
class RecordingPolicy:
    policy_id: str = "recording"
    observations: list[PolicyObservation] = field(default_factory=list)

    def forecast(
        self,
        observation: PolicyObservation,
        *,
        horizon_seconds: float,
    ) -> PolicyForecast:
        assert horizon_seconds >= 0.0
        self.observations.append(observation)
        predictive = np.asarray(observation.posterior, dtype=float)
        return PolicyForecast(score=predictive, predictive=predictive)


def test_exact_allocator_preserves_budget_and_required_source() -> None:
    score = np.arange(30 * 40, dtype=float).reshape(30, 40)
    budget = ScanBudget("test", columns=7, rows=9)
    source = (3.0, 2.0)

    mask = allocate_exact_column_mask(
        score,
        budget,
        include_point=source,
    )

    assert np.count_nonzero(mask) == 7 * 9
    assert np.unique(np.where(mask)[1]).size == 7
    assert np.unique(np.where(mask)[0]).size == 9
    assert mask[3, 2]


def test_policy_receives_read_only_causal_data_and_provided_motion() -> None:
    policy = RecordingPolicy()
    budget = ScanBudget("tiny", columns=5, rows=8)
    provided_motion = MotionEstimate(
        dy_per_t=2.5,
        dx_per_t=7.5,
        method="test_measured_motion",
        confidence=0.4,
    )

    run_experiment(
        ExperimentRequest(
            scenario="methane",
            max_scans=1,
            methane_config=SyntheticMethaneLeakConfig(),
            methane_motion_mode="provided",
            planner_motion=provided_motion,
            scan_budget=budget,
            initial_scan_budget=budget,
        ),
        policy=policy,
    )

    assert policy.observations
    assert set(PolicyObservation.__dataclass_fields__) == {
        "latest_likelihood",
        "posterior",
        "motion",
        "decision_time_seconds",
        "source_point",
        "scenario",
    }
    assert all(not item.posterior.flags.writeable for item in policy.observations)
    assert all(
        not item.latest_likelihood.flags.writeable
        for item in policy.observations
    )
    assert all(item.motion == provided_motion for item in policy.observations)


def test_source_free_policy_observation_does_not_leak_source() -> None:
    policy = RecordingPolicy()

    run_experiment(
        ExperimentRequest(
            scenario="methane",
            max_scans=1,
            source_free=True,
            vertical_pass_seconds=0.005,
            horizontal_step_seconds=0.0,
        ),
        policy=policy,
    )

    assert policy.observations
    assert all(item.source_point is None for item in policy.observations)


def test_static_corridor_uses_a_fixed_downwind_band() -> None:
    shape = (60, 100)
    source = (30.0, 15.0)
    uniform = np.full(shape, 1.0 / (shape[0] * shape[1]))
    observation = PolicyObservation(
        latest_likelihood=uniform,
        posterior=uniform,
        motion=MotionEstimate(0.0, -99.0, "ignored_motion"),
        decision_time_seconds=1.0,
        source_point=source,
        scenario="methane",
    )
    forecast = StaticSourceCorridorPolicy().forecast(
        observation,
        horizon_seconds=500.0,
    )
    mask = allocate_exact_column_mask(
        forecast.score,
        ScanBudget("corridor", columns=30, rows=24),
        include_point=source,
    )
    selected_columns = np.unique(np.where(mask)[1])

    assert mask[int(source[0]), int(source[1])]
    assert selected_columns.mean() > source[1] + 15.0
    assert np.count_nonzero(mask) == 30 * 24


def test_default_policy_remains_behavior_compatible() -> None:
    request = ExperimentRequest(
        scenario="methane",
        max_scans=2,
        vertical_pass_seconds=0.005,
        horizontal_step_seconds=0.0,
    )

    default_report = run_experiment(request).report
    explicit_report = run_experiment(
        request,
        policy=AdaptivePredictivePolicy(
            {
                "wind_error_speed": 3.0,
                "model_error_radius": 2.0,
                "plume_spread_speed": 5.4,
            }
        ),
    ).report

    assert explicit_report.to_json() == default_report.to_json()
