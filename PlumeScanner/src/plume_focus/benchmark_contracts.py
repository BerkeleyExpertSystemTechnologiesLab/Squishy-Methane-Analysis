"""Stable contracts for the cost-matched methane benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Literal

from plume_focus.motion import MotionEstimate
from plume_focus.scan_policies import ScanBudget
from plume_focus.video_sources import SyntheticMethaneLeakConfig


BenchmarkSplit = Literal["development", "held-out"]
WindCondition = Literal["noisy", "biased", "missing", "near-zero", "reversed"]
ClaimStatus = Literal["supported", "not_demonstrated", "invalid"]

BENCHMARK_SCHEMA_VERSION = 1
FROZEN_MODEL_COMMIT = "104f55c"
POLICY_VERSION = "2"

DEFAULT_BUDGETS = (
    ScanBudget("small", columns=30, rows=56),
    ScanBudget("medium", columns=60, rows=80),
    ScanBudget("large", columns=120, rows=104),
)


@dataclass(frozen=True)
class MethaneEpisodeSpec:
    """One deterministic generator truth and separately disclosed planner input."""

    episode_id: str
    seed: int
    split: BenchmarkSplit
    wind_condition: WindCondition
    stress: bool
    source: tuple[float, float]
    truth_wind_per_frame: tuple[float, float]
    planner_motion_px_per_second: tuple[float, float] | None
    emission_interval: int
    base_sigma: float
    diffusion_per_frame: float
    lateral_phase: float
    amplitude_phase: float
    shimmer_phase: float

    def methane_config(self) -> SyntheticMethaneLeakConfig:
        """Build the pure synthetic generator configuration for this episode."""

        return SyntheticMethaneLeakConfig(
            source=self.source,
            wind_velocity=self.truth_wind_per_frame,
            emission_interval=self.emission_interval,
            base_sigma=self.base_sigma,
            diffusion_per_frame=self.diffusion_per_frame,
            lateral_phase=self.lateral_phase,
            amplitude_phase=self.amplitude_phase,
            shimmer_phase=self.shimmer_phase,
        )

    def planner_motion(self) -> MotionEstimate | None:
        """Return the non-oracle motion exposed to competitive policies."""

        if self.planner_motion_px_per_second is None:
            return None
        return MotionEstimate(
            dy_per_t=self.planner_motion_px_per_second[0],
            dx_per_t=self.planner_motion_px_per_second[1],
            method=f"{self.wind_condition}_measured_wind_px_per_second",
            confidence=0.75 if self.wind_condition == "noisy" else 0.6,
        )

    def truth_motion(self) -> MotionEstimate:
        """Return true generator motion for the explicitly labeled oracle policy."""

        config = self.methane_config()
        return MotionEstimate(
            dy_per_t=config.wind_velocity[0] * config.fps,
            dx_per_t=config.wind_velocity[1] * config.fps,
            method="oracle_true_wind_px_per_second",
            confidence=1.0,
        )

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["source"] = list(self.source)
        payload["truth_wind_per_frame"] = list(self.truth_wind_per_frame)
        if self.planner_motion_px_per_second is not None:
            payload["planner_motion_px_per_second"] = list(
                self.planner_motion_px_per_second
            )
        return payload


@dataclass(frozen=True)
class BenchmarkSuite:
    """A frozen episode manifest and exact scan budgets."""

    suite_id: str
    split: BenchmarkSplit
    episodes: tuple[MethaneEpisodeSpec, ...]
    budgets: tuple[ScanBudget, ...] = DEFAULT_BUDGETS
    scans_per_run: int = 12

    def validate(self) -> None:
        if not self.suite_id:
            raise ValueError("suite_id must not be empty.")
        if not self.episodes:
            raise ValueError("benchmark suite must include at least one episode.")
        if not self.budgets:
            raise ValueError("benchmark suite must include at least one budget.")
        if self.scans_per_run <= 0:
            raise ValueError("scans_per_run must be positive.")
        episode_ids = [episode.episode_id for episode in self.episodes]
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError("episode_id values must be unique.")
        budget_ids = [budget.budget_id for budget in self.budgets]
        if len(set(budget_ids)) != len(budget_ids):
            raise ValueError("budget_id values must be unique.")
        for episode in self.episodes:
            if episode.split != self.split:
                raise ValueError("all episodes must match the suite split.")
            for budget in self.budgets:
                budget.validate(episode.methane_config().shape)

    @property
    def manifest_hash(self) -> str:
        payload = {
            "suite_id": self.suite_id,
            "split": self.split,
            "scans_per_run": self.scans_per_run,
            "budgets": [asdict(budget) for budget in self.budgets],
            "episodes": [episode.to_payload() for episode in self.episodes],
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class BenchmarkRunResult:
    """Episode-level metrics for one policy and one exact budget."""

    episode_id: str
    budget_id: str
    policy_id: str
    stress: bool
    scans_completed: int
    termination_reason: str
    timeline_integrity_verified: bool
    initial_area_fraction: float
    mean_swept_area_fraction: float
    mean_scan_duration_seconds: float
    total_elapsed_seconds: float
    interval_mass_recall: float
    completion_mass_recall: float
    completion_support_recall: float
    support_precision: float
    intersection_over_union: float
    success_rate: float
    oracle_inputs: tuple[str, ...]

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["oracle_inputs"] = list(self.oracle_inputs)
        return payload


@dataclass(frozen=True)
class PairedComparison:
    """Adaptive-minus-baseline result over the frozen primary episodes."""

    baseline_policy_id: str
    mean_delta: float
    confidence_interval_95: tuple[float, float]
    per_budget_mean_delta: dict[str, float]
    practical_improvement_met: bool
    confidence_interval_positive: bool
    no_negative_budget: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "baseline_policy_id": self.baseline_policy_id,
            "mean_delta": self.mean_delta,
            "confidence_interval_95": list(self.confidence_interval_95),
            "per_budget_mean_delta": self.per_budget_mean_delta,
            "practical_improvement_met": self.practical_improvement_met,
            "confidence_interval_positive": self.confidence_interval_positive,
            "no_negative_budget": self.no_negative_budget,
        }


@dataclass(frozen=True)
class BenchmarkIntegrity:
    """Causality, cost, and timeline checks that gate interpretation."""

    causal_policy_contract: bool
    exact_cost_match: bool
    complete_result_matrix: bool
    timeline_integrity: bool
    max_area_difference: float
    max_duration_difference_seconds: float


@dataclass(frozen=True)
class BenchmarkReport:
    """Reproducible benchmark report independent of the video proof schema."""

    suite: BenchmarkSuite
    policy_ids: tuple[str, ...]
    competitive_policy_ids: tuple[str, ...]
    results: tuple[BenchmarkRunResult, ...]
    comparisons: tuple[PairedComparison, ...]
    integrity: BenchmarkIntegrity
    claim_status: ClaimStatus
    full_scene_reference_area_fraction: float
    full_scene_reference_duration_seconds: float
    model_profile: dict[str, object]
    report_schema_version: int = BENCHMARK_SCHEMA_VERSION
    frozen_model_commit: str = FROZEN_MODEL_COMMIT
    policy_version: str = POLICY_VERSION

    def to_payload(self) -> dict[str, object]:
        return {
            "report_schema_version": self.report_schema_version,
            "benchmark_id": self.suite.suite_id,
            "split": self.suite.split,
            "manifest_hash": self.suite.manifest_hash,
            "frozen_model_commit": self.frozen_model_commit,
            "policy_version": self.policy_version,
            "model_profile": self.model_profile,
            "scans_per_run": self.suite.scans_per_run,
            "budgets": [
                {
                    **asdict(budget),
                    "area_fraction": budget.area_fraction(
                        self.suite.episodes[0].methane_config().shape
                    ),
                }
                for budget in self.suite.budgets
            ],
            "episodes": [
                episode.to_payload()
                for episode in self.suite.episodes
            ],
            "policy_ids": list(self.policy_ids),
            "competitive_policy_ids": list(self.competitive_policy_ids),
            "primary_metric": "mean_start_midpoint_completion_mass_recall",
            "claim_gate": {
                "minimum_absolute_improvement": 0.02,
                "paired_bootstrap_confidence": 0.95,
                "requires_positive_lower_bound_against_every_baseline": True,
                "requires_no_negative_budget_mean": True,
                "oracle_and_stress_cases_excluded": True,
            },
            "integrity": asdict(self.integrity),
            "claim_status": self.claim_status,
            "recall_cost_curve": self._recall_cost_curve(),
            "stress_diagnostics": self._stress_diagnostics(),
            "comparisons": [
                comparison.to_payload()
                for comparison in self.comparisons
            ],
            "full_scene_mechanical_cost_reference": {
                "area_fraction": self.full_scene_reference_area_fraction,
                "duration_seconds": self.full_scene_reference_duration_seconds,
                "competitive": False,
            },
            "results": [result.to_payload() for result in self.results],
        }

    def _recall_cost_curve(self) -> dict[str, list[dict[str, float | str]]]:
        curves: dict[str, list[dict[str, float | str]]] = {}
        shape = self.suite.episodes[0].methane_config().shape
        for policy_id in self.policy_ids:
            points: list[dict[str, float | str]] = []
            for budget in self.suite.budgets:
                matched = [
                    result
                    for result in self.results
                    if not result.stress
                    and result.policy_id == policy_id
                    and result.budget_id == budget.budget_id
                ]
                if not matched:
                    continue
                points.append(
                    {
                        "budget_id": budget.budget_id,
                        "area_fraction": budget.area_fraction(shape),
                        "mean_duration_seconds": sum(
                            result.mean_scan_duration_seconds
                            for result in matched
                        )
                        / len(matched),
                        "mean_interval_mass_recall": sum(
                            result.interval_mass_recall
                            for result in matched
                        )
                        / len(matched),
                        "mean_completion_mass_recall": sum(
                            result.completion_mass_recall
                            for result in matched
                        )
                        / len(matched),
                    }
                )
            curves[policy_id] = points
        return curves

    def _stress_diagnostics(self) -> dict[str, dict[str, float | int]]:
        diagnostics: dict[str, dict[str, float | int]] = {}
        for policy_id in self.policy_ids:
            matched = [
                result
                for result in self.results
                if result.stress and result.policy_id == policy_id
            ]
            if not matched:
                diagnostics[policy_id] = {
                    "runs": 0,
                    "mean_interval_mass_recall": 0.0,
                }
                continue
            diagnostics[policy_id] = {
                "runs": len(matched),
                "mean_interval_mass_recall": sum(
                    result.interval_mass_recall
                    for result in matched
                )
                / len(matched),
            }
        return diagnostics

    def to_json(self) -> str:
        return json.dumps(
            self.to_payload(),
            indent=2,
            sort_keys=True,
        )
