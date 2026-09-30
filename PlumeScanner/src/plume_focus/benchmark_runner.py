"""Headless execution of the fair, cost-matched methane benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from plume_focus.benchmark_contracts import (
    BenchmarkIntegrity,
    BenchmarkReport,
    BenchmarkRunResult,
    BenchmarkSuite,
    ClaimStatus,
    MethaneEpisodeSpec,
)
from plume_focus.benchmark_statistics import compare_adaptive_to_baselines
from plume_focus.experiment_runner import (
    ExperimentContext,
    ExperimentObserver,
    ExperimentRequest,
    ExperimentStep,
    run_experiment,
)
from plume_focus.model_profiles import ModelProfile, synthetic_demo_v1
from plume_focus.scan_policies import (
    AdaptivePredictivePolicy,
    ConstantVelocityPolicy,
    LastObservedLikelihoodPolicy,
    PolicyObservation,
    ScanBudget,
    ScanPolicy,
    StaticSourceCorridorPolicy,
    ZeroMotionPosteriorPolicy,
)


ADAPTIVE_POLICY_ID = "adaptive-predictive"
ORACLE_POLICY_ID = "oracle-wind-adaptive"
BASELINE_POLICY_IDS = (
    "static-source-corridor",
    "last-observed-likelihood",
    "zero-motion-posterior",
    "constant-velocity",
)
COMPETITIVE_POLICY_IDS = BASELINE_POLICY_IDS + (ADAPTIVE_POLICY_ID,)
POLICY_IDS = COMPETITIVE_POLICY_IDS + (ORACLE_POLICY_ID,)

BenchmarkProgress = Callable[[int, int, str], None]


@dataclass
class _BenchmarkObserver(ExperimentObserver):
    support_precisions: list[float]

    def __init__(self) -> None:
        self.support_precisions = []

    def start(self, context: ExperimentContext) -> None:
        del context

    def observe(self, step: ExperimentStep) -> None:
        selected = np.asarray(step.next_scan_mask, dtype=bool)
        support = np.asarray(step.target_support, dtype=bool)
        selected_count = int(np.count_nonzero(selected))
        intersection = int(np.count_nonzero(selected & support))
        self.support_precisions.append(
            intersection / selected_count if selected_count else 0.0
        )

    def close(self) -> None:
        return None


def run_benchmark(
    suite: BenchmarkSuite,
    *,
    progress: BenchmarkProgress | None = None,
) -> BenchmarkReport:
    """Run every policy on fresh, matched episode/budget timelines."""

    suite.validate()
    model_profile = synthetic_demo_v1("methane")
    total_runs = len(suite.episodes) * len(suite.budgets) * len(POLICY_IDS)
    results: list[BenchmarkRunResult] = []
    completed = 0
    for episode in suite.episodes:
        for budget in suite.budgets:
            for policy_id in POLICY_IDS:
                result = _run_policy_case(
                    episode,
                    budget=budget,
                    scans_per_run=suite.scans_per_run,
                    policy_id=policy_id,
                    model_profile=model_profile,
                )
                results.append(result)
                completed += 1
                if progress is not None:
                    progress(
                        completed,
                        total_runs,
                        (
                            f"{episode.episode_id}/{budget.budget_id}/"
                            f"{policy_id}"
                        ),
                    )

    result_tuple = tuple(results)
    integrity = _evaluate_integrity(suite, result_tuple)
    comparisons = compare_adaptive_to_baselines(
        result_tuple,
        adaptive_policy_id=ADAPTIVE_POLICY_ID,
        baseline_policy_ids=BASELINE_POLICY_IDS,
        budget_ids=tuple(budget.budget_id for budget in suite.budgets),
    )
    gates_pass = all(
        comparison.practical_improvement_met
        and comparison.confidence_interval_positive
        and comparison.no_negative_budget
        for comparison in comparisons
    )
    valid = (
        integrity.causal_policy_contract
        and integrity.exact_cost_match
        and integrity.complete_result_matrix
        and integrity.timeline_integrity
    )
    claim_status: ClaimStatus = (
        "invalid"
        if not valid
        else "supported"
        if gates_pass
        else "not_demonstrated"
    )

    methane_config = suite.episodes[0].methane_config()
    shape = methane_config.shape
    full_scene_duration = (
        shape[1] * methane_config.vertical_pass_seconds
        + max(0, shape[1] - 1)
        * methane_config.horizontal_step_seconds
    )
    return BenchmarkReport(
        suite=suite,
        policy_ids=POLICY_IDS,
        competitive_policy_ids=COMPETITIVE_POLICY_IDS,
        results=result_tuple,
        comparisons=comparisons,
        integrity=integrity,
        claim_status=claim_status,
        full_scene_reference_area_fraction=1.0,
        full_scene_reference_duration_seconds=full_scene_duration,
        model_profile=model_profile.to_payload(
            frame_rate_hz=methane_config.fps,
        ),
    )


def _run_policy_case(
    episode: MethaneEpisodeSpec,
    *,
    budget: ScanBudget,
    scans_per_run: int,
    policy_id: str,
    model_profile: ModelProfile,
) -> BenchmarkRunResult:
    oracle = policy_id == ORACLE_POLICY_ID
    planner_motion = (
        episode.truth_motion()
        if oracle
        else episode.planner_motion()
    )
    motion_mode = "scan_derived" if planner_motion is None else "provided"
    methane_config = episode.methane_config()
    policy = _policy_for_id(
        policy_id,
        model_profile=model_profile,
        frame_rate_hz=methane_config.fps,
    )
    observer = _BenchmarkObserver()
    result = run_experiment(
        ExperimentRequest(
            max_frames=scans_per_run,
            max_scans=scans_per_run,
            scenario="methane",
            evidence_mode="scan",
            methane_config=methane_config,
            methane_motion_mode=motion_mode,
            planner_motion=planner_motion,
            scan_budget=budget,
            initial_scan_budget=budget,
            model_profile=model_profile,
        ),
        observer=observer,
        policy=policy,
    )
    report = result.report
    return BenchmarkRunResult(
        episode_id=episode.episode_id,
        budget_id=budget.budget_id,
        policy_id=policy_id,
        stress=episode.stress,
        scans_completed=report.scans_completed,
        termination_reason=report.termination_reason,
        timeline_integrity_verified=report.timeline_integrity_verified,
        initial_area_fraction=report.initial_mask_area_fraction,
        mean_swept_area_fraction=report.mean_swept_area_fraction,
        mean_scan_duration_seconds=report.scan_duration_seconds,
        total_elapsed_seconds=report.total_elapsed_seconds,
        interval_mass_recall=report.mean_actual_sweep_interval_mass_recall,
        completion_mass_recall=report.mean_actual_sweep_mass_recall,
        completion_support_recall=report.mean_actual_sweep_support_recall,
        support_precision=(
            float(np.mean(observer.support_precisions))
            if observer.support_precisions
            else 0.0
        ),
        intersection_over_union=report.mean_actual_sweep_iou,
        success_rate=report.actual_sweep_success_rate,
        oracle_inputs=(
            ("exact synthetic wind vector",)
            if oracle
            else report.oracle_inputs
        ),
    )


def _policy_for_id(
    policy_id: str,
    *,
    model_profile: ModelProfile,
    frame_rate_hz: float,
) -> ScanPolicy:
    if policy_id == "static-source-corridor":
        return StaticSourceCorridorPolicy(
            config=model_profile.scan_policy.static_corridor,
        )
    if policy_id == "last-observed-likelihood":
        return LastObservedLikelihoodPolicy()
    if policy_id == "zero-motion-posterior":
        return ZeroMotionPosteriorPolicy()
    if policy_id == "constant-velocity":
        return ConstantVelocityPolicy()
    if policy_id == ADAPTIVE_POLICY_ID:
        return AdaptivePredictivePolicy(
            dynamics_config=model_profile.dynamics,
            uncertainty_config=model_profile.uncertainty.resolve(
                frame_rate_hz
            ),
        )
    if policy_id == ORACLE_POLICY_ID:
        return AdaptivePredictivePolicy(
            policy_id=ORACLE_POLICY_ID,
            dynamics_config=model_profile.dynamics,
            uncertainty_config=model_profile.uncertainty.resolve(
                frame_rate_hz
            ),
        )
    raise ValueError(f"Unknown benchmark policy: {policy_id}")


def _evaluate_integrity(
    suite: BenchmarkSuite,
    results: tuple[BenchmarkRunResult, ...],
) -> BenchmarkIntegrity:
    expected = {
        (episode.episode_id, budget.budget_id, policy_id)
        for episode in suite.episodes
        for budget in suite.budgets
        for policy_id in POLICY_IDS
    }
    actual = {
        (result.episode_id, result.budget_id, result.policy_id)
        for result in results
    }

    max_area_difference = 0.0
    max_duration_difference = 0.0
    exact_cost_match = True
    for episode in suite.episodes:
        for budget in suite.budgets:
            matched = [
                result
                for result in results
                if result.episode_id == episode.episode_id
                and result.budget_id == budget.budget_id
            ]
            if len(matched) != len(POLICY_IDS):
                exact_cost_match = False
                continue
            areas = [result.mean_swept_area_fraction for result in matched]
            durations = [result.mean_scan_duration_seconds for result in matched]
            initial_areas = [result.initial_area_fraction for result in matched]
            area_difference = max(areas) - min(areas)
            duration_difference = max(durations) - min(durations)
            initial_difference = max(initial_areas) - min(initial_areas)
            max_area_difference = max(
                max_area_difference,
                area_difference,
                initial_difference,
            )
            max_duration_difference = max(
                max_duration_difference,
                duration_difference,
            )
            expected_area = budget.area_fraction(
                episode.methane_config().shape
            )
            if (
                abs(areas[0] - expected_area) > 1e-12
                or abs(initial_areas[0] - expected_area) > 1e-12
                or area_difference > 1e-12
                or initial_difference > 1e-12
                or duration_difference > 1e-12
            ):
                exact_cost_match = False

    policy_fields = set(PolicyObservation.__dataclass_fields__)
    causal_contract = not any(
        forbidden in field
        for field in policy_fields
        for forbidden in ("truth", "target", "future", "timeline")
    )
    return BenchmarkIntegrity(
        causal_policy_contract=causal_contract,
        exact_cost_match=exact_cost_match,
        complete_result_matrix=actual == expected and len(results) == len(expected),
        timeline_integrity=all(
            result.timeline_integrity_verified
            for result in results
        ),
        max_area_difference=max_area_difference,
        max_duration_difference_seconds=max_duration_difference,
    )
