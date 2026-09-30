"""Causal scan-policy contracts and cost-matched policy implementations.

Policies receive completed scan evidence only.  They never receive a timeline,
future frame, or ground-truth support.  Mechanical mask construction is shared
so benchmark policies cannot trade more scan area or time for higher recall.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from plume_focus.bayes import normalize_probability, predict_next_prior
from plume_focus.experiment_contracts import Scenario
from plume_focus.model_profiles import (
    ResolvedUncertaintyConfig,
    StaticCorridorConfig,
    UncertaintyModelConfig,
)
from plume_focus.motion import MotionEstimate
from plume_focus.plume_dynamics import PlumeDynamicsConfig
from plume_focus.predictor import BoundConfig


@dataclass(frozen=True)
class ScanBudget:
    """Exact mechanical budget for one column-sweep decision."""

    budget_id: str
    columns: int
    rows: int

    def validate(self, shape: tuple[int, int]) -> None:
        """Reject budgets that cannot be represented exactly in ``shape``."""

        if not self.budget_id:
            raise ValueError("budget_id must not be empty.")
        if not 1 <= self.columns <= shape[1]:
            raise ValueError("columns must be within the scan shape.")
        if not 1 <= self.rows <= shape[0]:
            raise ValueError("rows must be within the scan shape.")

    def area_fraction(self, shape: tuple[int, int]) -> float:
        """Return the exact selected-pixel fraction."""

        self.validate(shape)
        return self.columns * self.rows / float(shape[0] * shape[1])


@dataclass(frozen=True)
class PolicyObservation:
    """Information available to a policy after one scan has completed."""

    latest_likelihood: NDArray[np.float64]
    posterior: NDArray[np.float64]
    motion: MotionEstimate
    decision_time_seconds: float
    source_point: tuple[float, float] | None
    scenario: Scenario

    @property
    def shape(self) -> tuple[int, int]:
        return self.posterior.shape

    def validate(self) -> None:
        """Validate causal policy arrays without modifying them."""

        if self.latest_likelihood.shape != self.posterior.shape:
            raise ValueError("latest_likelihood and posterior must have matching shapes.")
        if self.posterior.ndim != 2:
            raise ValueError("policy arrays must be 2-D.")
        if self.decision_time_seconds < 0.0:
            raise ValueError("decision_time_seconds must be non-negative.")
        if not np.all(np.isfinite(self.latest_likelihood)):
            raise ValueError("latest_likelihood must contain only finite values.")
        if not np.all(np.isfinite(self.posterior)):
            raise ValueError("posterior must contain only finite values.")


@dataclass(frozen=True)
class PolicyForecast:
    """A policy score surface and its predictive interpretation."""

    score: NDArray[np.float64]
    predictive: NDArray[np.float64]
    uncertainty_radius: float = 0.0
    reacquisition: bool = False

    def validate(self, shape: tuple[int, int]) -> None:
        """Validate a forecast before it reaches the mechanical allocator."""

        if self.score.shape != shape or self.predictive.shape != shape:
            raise ValueError("policy forecast arrays must match the experiment shape.")
        if np.any(~np.isfinite(self.score)) or np.any(self.score < 0.0):
            raise ValueError("policy score must be finite and non-negative.")
        if np.any(~np.isfinite(self.predictive)) or np.any(self.predictive < 0.0):
            raise ValueError("predictive field must be finite and non-negative.")
        if not np.isfinite(self.uncertainty_radius) or self.uncertainty_radius < 0.0:
            raise ValueError("uncertainty_radius must be finite and non-negative.")


class ScanPolicy(Protocol):
    """Strategy that scores where a mechanically constrained scanner looks next."""

    policy_id: str

    def forecast(
        self,
        observation: PolicyObservation,
        *,
        horizon_seconds: float,
    ) -> PolicyForecast:
        """Return a causal score surface for the proposed scan horizon."""


@dataclass(frozen=True)
class AdaptivePredictivePolicy:
    """Current uncertainty-expanded Bayesian predictive policy."""

    bound_kwargs: dict[str, float] | None = None
    policy_id: str = "adaptive-predictive"
    dynamics_config: PlumeDynamicsConfig = field(
        default_factory=PlumeDynamicsConfig
    )
    uncertainty_config: ResolvedUncertaintyConfig | None = None

    def forecast(
        self,
        observation: PolicyObservation,
        *,
        horizon_seconds: float,
    ) -> PolicyForecast:
        observation.validate()
        if horizon_seconds < 0.0:
            raise ValueError("horizon_seconds must be non-negative.")

        uncertainty = self._resolved_uncertainty()
        bounds = BoundConfig(
            scan_age=horizon_seconds,
            **uncertainty.bound_kwargs,
        )
        radius = bounds.uncertainty_radius(
            observation.motion,
            delta_t=horizon_seconds,
        )
        reacquisition = radius >= (
            float(np.hypot(*observation.shape))
            * uncertainty.reacquisition_radius_fraction_of_frame_diagonal
        )
        if reacquisition:
            predictive = normalize_probability(
                np.ones(observation.shape, dtype=float)
            )
        else:
            predictive = predict_next_prior(
                observation.posterior,
                observation.motion,
                diffusion_sigma=max(
                    uncertainty.diffusion_sigma_floor_px,
                    radius / uncertainty.diffusion_radius_divisor,
                ),
                delta_t=horizon_seconds,
                source=observation.source_point,
                dynamics_config=self.dynamics_config,
            )
        return PolicyForecast(
            score=predictive,
            predictive=predictive,
            uncertainty_radius=radius,
            reacquisition=reacquisition,
        )

    def _resolved_uncertainty(self) -> ResolvedUncertaintyConfig:
        if self.uncertainty_config is not None:
            return self.uncertainty_config
        return UncertaintyModelConfig.from_legacy_bound_kwargs(
            self.bound_kwargs or {},
        )


@dataclass(frozen=True)
class StaticSourceCorridorPolicy:
    """Fixed rightward source corridor that ignores new scan evidence."""

    policy_id: str = "static-source-corridor"
    config: StaticCorridorConfig = field(
        default_factory=StaticCorridorConfig
    )

    def forecast(
        self,
        observation: PolicyObservation,
        *,
        horizon_seconds: float,
    ) -> PolicyForecast:
        del horizon_seconds
        observation.validate()
        if observation.source_point is None:
            score = np.ones(observation.shape, dtype=float)
        else:
            rows, cols = np.indices(observation.shape, dtype=float)
            source_row, source_col = observation.source_point
            downwind_distance = cols - source_col
            crosswind = np.exp(
                -((rows - source_row) ** 2)
                / (2.0 * self.config.crosswind_sigma_px**2)
            )
            longitudinal = np.exp(
                -(
                    (
                        downwind_distance
                        - self.config.downwind_center_px
                    )
                    ** 2
                )
                / (2.0 * self.config.downwind_sigma_px**2)
            )
            upwind_penalty = np.where(
                downwind_distance >= 0.0,
                1.0,
                np.exp(
                    downwind_distance
                    / self.config.upwind_decay_length_px
                ),
            )
            score = crosswind * longitudinal * upwind_penalty
        predictive = normalize_probability(score)
        return PolicyForecast(score=predictive, predictive=predictive)


@dataclass(frozen=True)
class LastObservedLikelihoodPolicy:
    """Rescan the strongest values in the latest completed scan."""

    policy_id: str = "last-observed-likelihood"

    def forecast(
        self,
        observation: PolicyObservation,
        *,
        horizon_seconds: float,
    ) -> PolicyForecast:
        del horizon_seconds
        observation.validate()
        predictive = normalize_probability(observation.latest_likelihood)
        return PolicyForecast(score=predictive, predictive=predictive)


@dataclass(frozen=True)
class ZeroMotionPosteriorPolicy:
    """Use the completed posterior without advection."""

    policy_id: str = "zero-motion-posterior"

    def forecast(
        self,
        observation: PolicyObservation,
        *,
        horizon_seconds: float,
    ) -> PolicyForecast:
        del horizon_seconds
        observation.validate()
        predictive = normalize_probability(observation.posterior)
        return PolicyForecast(score=predictive, predictive=predictive)


@dataclass(frozen=True)
class ConstantVelocityPolicy:
    """Translate the posterior using measured motion without uncertainty growth."""

    policy_id: str = "constant-velocity"

    def forecast(
        self,
        observation: PolicyObservation,
        *,
        horizon_seconds: float,
    ) -> PolicyForecast:
        observation.validate()
        if horizon_seconds < 0.0:
            raise ValueError("horizon_seconds must be non-negative.")
        predictive = predict_next_prior(
            observation.posterior,
            observation.motion,
            diffusion_sigma=0.0,
            delta_t=horizon_seconds,
            source=None,
        )
        return PolicyForecast(score=predictive, predictive=predictive)


def allocate_exact_column_mask(
    score: NDArray[np.float64],
    budget: ScanBudget,
    *,
    include_point: tuple[float, float] | None = None,
) -> NDArray[np.bool_]:
    """Select exact column and contiguous-row counts from a policy score."""

    values = np.maximum(np.asarray(score, dtype=float), 0.0)
    if values.ndim != 2 or np.any(~np.isfinite(values)):
        raise ValueError("score must be a finite 2-D array.")
    budget.validate(values.shape)

    column_scores = values.sum(axis=0)
    selected_columns = _top_indices(column_scores, budget.columns)
    if include_point is not None:
        source_col = int(np.clip(round(include_point[1]), 0, values.shape[1] - 1))
        selected_columns = _force_index(
            selected_columns,
            source_col,
            column_scores,
        )

    selected_scores = values[:, selected_columns].sum(axis=1)
    row_start = _best_window_start(selected_scores, budget.rows)
    if include_point is not None:
        source_row = int(np.clip(round(include_point[0]), 0, values.shape[0] - 1))
        row_start = int(
            np.clip(
                row_start,
                source_row - budget.rows + 1,
                source_row,
            )
        )
        row_start = int(np.clip(row_start, 0, values.shape[0] - budget.rows))

    mask = np.zeros(values.shape, dtype=bool)
    mask[
        row_start : row_start + budget.rows,
        selected_columns,
    ] = True
    expected_pixels = budget.columns * budget.rows
    if int(np.count_nonzero(mask)) != expected_pixels:
        raise RuntimeError("exact column allocator did not satisfy its budget.")
    return mask


def _top_indices(values: NDArray[np.float64], count: int) -> NDArray[np.int_]:
    order = np.argsort(-values, kind="stable")
    return np.sort(order[:count].astype(int))


def _force_index(
    selected: NDArray[np.int_],
    required: int,
    scores: NDArray[np.float64],
) -> NDArray[np.int_]:
    if np.any(selected == required):
        return selected
    weakest_position = min(
        range(selected.size),
        key=lambda index: (float(scores[selected[index]]), -int(selected[index])),
    )
    updated = selected.copy()
    updated[weakest_position] = required
    return np.sort(updated)


def _best_window_start(values: NDArray[np.float64], length: int) -> int:
    if length == values.size:
        return 0
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=float)))
    totals = cumulative[length:] - cumulative[:-length]
    return int(np.argmax(totals))
