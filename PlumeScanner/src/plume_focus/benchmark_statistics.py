"""Paired statistics for the deterministic benchmark report."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from numpy.typing import NDArray

from plume_focus.benchmark_contracts import (
    BenchmarkRunResult,
    PairedComparison,
)


BOOTSTRAP_SEED = 20260725
BOOTSTRAP_SAMPLES = 10_000
MINIMUM_PRACTICAL_IMPROVEMENT = 0.02


def paired_bootstrap_confidence_interval(
    deltas: NDArray[np.float64],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Return a deterministic percentile interval over paired episode deltas."""

    values = np.asarray(deltas, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("deltas must be a non-empty 1-D array.")
    if np.any(~np.isfinite(values)):
        raise ValueError("deltas must contain only finite values.")
    if samples <= 0:
        raise ValueError("samples must be positive.")
    if values.size == 1:
        value = float(values[0])
        return value, value

    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        values.size,
        size=(samples, values.size),
    )
    means = values[indices].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def compare_adaptive_to_baselines(
    results: tuple[BenchmarkRunResult, ...],
    *,
    adaptive_policy_id: str,
    baseline_policy_ids: tuple[str, ...],
    budget_ids: tuple[str, ...],
) -> tuple[PairedComparison, ...]:
    """Compare matched primary episodes after equal averaging across budgets."""

    primary = [result for result in results if not result.stress]
    by_key = {
        (result.policy_id, result.episode_id, result.budget_id): result
        for result in primary
    }
    adaptive_keys = {
        (result.episode_id, result.budget_id)
        for result in primary
        if result.policy_id == adaptive_policy_id
    }
    if not adaptive_keys:
        raise ValueError("adaptive policy has no primary results.")

    comparisons: list[PairedComparison] = []
    for comparison_index, baseline_id in enumerate(baseline_policy_ids):
        episode_deltas: dict[str, list[float]] = defaultdict(list)
        per_budget: dict[str, list[float]] = defaultdict(list)
        for episode_id, budget_id in sorted(adaptive_keys):
            adaptive = by_key.get((adaptive_policy_id, episode_id, budget_id))
            baseline = by_key.get((baseline_id, episode_id, budget_id))
            if adaptive is None or baseline is None:
                raise ValueError(
                    f"Missing paired result for {baseline_id}, "
                    f"{episode_id}, {budget_id}."
                )
            delta = (
                adaptive.interval_mass_recall
                - baseline.interval_mass_recall
            )
            episode_deltas[episode_id].append(delta)
            per_budget[budget_id].append(delta)

        if any(len(values) != len(budget_ids) for values in episode_deltas.values()):
            raise ValueError("Every primary episode must contain every budget.")
        paired_values = np.asarray(
            [
                float(np.mean(episode_deltas[episode_id]))
                for episode_id in sorted(episode_deltas)
            ],
            dtype=float,
        )
        interval = paired_bootstrap_confidence_interval(
            paired_values,
            seed=BOOTSTRAP_SEED + comparison_index,
        )
        per_budget_means = {
            budget_id: float(np.mean(per_budget[budget_id]))
            for budget_id in budget_ids
        }
        mean_delta = float(np.mean(paired_values))
        comparisons.append(
            PairedComparison(
                baseline_policy_id=baseline_id,
                mean_delta=mean_delta,
                confidence_interval_95=interval,
                per_budget_mean_delta=per_budget_means,
                practical_improvement_met=(
                    mean_delta >= MINIMUM_PRACTICAL_IMPROVEMENT
                ),
                confidence_interval_positive=interval[0] > 0.0,
                no_negative_budget=all(
                    delta >= 0.0
                    for delta in per_budget_means.values()
                ),
            )
        )
    return tuple(comparisons)
