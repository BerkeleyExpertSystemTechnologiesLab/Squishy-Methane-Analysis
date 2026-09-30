"""Frozen deterministic episode manifests for benchmark development and evaluation."""

from __future__ import annotations

import math

import numpy as np

from plume_focus.benchmark_contracts import (
    BenchmarkSplit,
    BenchmarkSuite,
    MethaneEpisodeSpec,
    WindCondition,
)


DEVELOPMENT_PRIMARY_SEEDS = tuple(range(6))
DEVELOPMENT_STRESS_SEEDS = (50, 51)
HELD_OUT_PRIMARY_SEEDS = tuple(range(100, 124))
HELD_OUT_STRESS_SEEDS = tuple(range(200, 206))


def make_benchmark_suite(
    split: BenchmarkSplit,
    *,
    scans_per_run: int = 12,
) -> BenchmarkSuite:
    """Build the committed development or untouched held-out manifest."""

    if split == "development":
        primary_seeds = DEVELOPMENT_PRIMARY_SEEDS
        stress_seeds = DEVELOPMENT_STRESS_SEEDS
        suite_id = "methane-fair-benchmark-development-v1"
    elif split == "held-out":
        primary_seeds = HELD_OUT_PRIMARY_SEEDS
        stress_seeds = HELD_OUT_STRESS_SEEDS
        suite_id = "methane-fair-benchmark-held-out-v1"
    else:
        raise ValueError("split must be 'development' or 'held-out'.")

    episodes = tuple(
        _primary_episode(split, seed, index)
        for index, seed in enumerate(primary_seeds)
    ) + tuple(
        _stress_episode(split, seed, index)
        for index, seed in enumerate(stress_seeds)
    )
    suite = BenchmarkSuite(
        suite_id=suite_id,
        split=split,
        episodes=episodes,
        scans_per_run=scans_per_run,
    )
    suite.validate()
    return suite


def _primary_episode(
    split: BenchmarkSplit,
    seed: int,
    index: int,
) -> MethaneEpisodeSpec:
    rng = np.random.default_rng(seed)
    condition: WindCondition = ("noisy", "biased", "missing")[index % 3]
    source = (
        float(rng.uniform(82.0, 138.0)),
        float(rng.uniform(38.0, 62.0)),
    )
    truth_wind = (
        float(rng.uniform(-0.18, 0.18)),
        float(rng.uniform(0.72, 1.28)),
    )
    planner_motion = _planner_motion(
        rng,
        truth_wind,
        condition,
        fps=12.0,
        index=index,
    )
    return _episode(
        split=split,
        seed=seed,
        index=index,
        condition=condition,
        stress=False,
        source=source,
        truth_wind=truth_wind,
        planner_motion=planner_motion,
        rng=rng,
    )


def _stress_episode(
    split: BenchmarkSplit,
    seed: int,
    index: int,
) -> MethaneEpisodeSpec:
    rng = np.random.default_rng(seed)
    condition: WindCondition = "near-zero" if index % 2 == 0 else "reversed"
    if condition == "near-zero":
        source = (
            float(rng.uniform(82.0, 138.0)),
            float(rng.uniform(112.0, 208.0)),
        )
        truth_wind = (
            float(rng.uniform(-0.12, 0.12)),
            float(rng.uniform(0.04, 0.16)),
        )
    else:
        source = (
            float(rng.uniform(82.0, 138.0)),
            float(rng.uniform(258.0, 282.0)),
        )
        truth_wind = (
            float(rng.uniform(-0.18, 0.18)),
            float(rng.uniform(-1.28, -0.72)),
        )
    planner_motion = _noisy_motion(rng, truth_wind, fps=12.0)
    return _episode(
        split=split,
        seed=seed,
        index=index,
        condition=condition,
        stress=True,
        source=source,
        truth_wind=truth_wind,
        planner_motion=planner_motion,
        rng=rng,
    )


def _episode(
    *,
    split: BenchmarkSplit,
    seed: int,
    index: int,
    condition: WindCondition,
    stress: bool,
    source: tuple[float, float],
    truth_wind: tuple[float, float],
    planner_motion: tuple[float, float] | None,
    rng: np.random.Generator,
) -> MethaneEpisodeSpec:
    group = "stress" if stress else "primary"
    return MethaneEpisodeSpec(
        episode_id=f"{split}-{group}-{index:03d}-seed-{seed}",
        seed=seed,
        split=split,
        wind_condition=condition,
        stress=stress,
        source=source,
        truth_wind_per_frame=truth_wind,
        planner_motion_px_per_second=planner_motion,
        emission_interval=int(rng.integers(2, 5)),
        base_sigma=float(rng.uniform(1.4, 2.3)),
        diffusion_per_frame=float(rng.uniform(0.05, 0.11)),
        lateral_phase=float(rng.uniform(-math.pi, math.pi)),
        amplitude_phase=float(rng.uniform(-math.pi, math.pi)),
        shimmer_phase=float(rng.uniform(-math.pi, math.pi)),
    )


def _planner_motion(
    rng: np.random.Generator,
    truth_wind: tuple[float, float],
    condition: WindCondition,
    *,
    fps: float,
    index: int,
) -> tuple[float, float] | None:
    if condition == "missing":
        return None
    if condition == "noisy":
        return _noisy_motion(rng, truth_wind, fps=fps)

    dy, dx = truth_wind
    angle = math.radians(12.0 if index % 2 == 0 else -12.0)
    scale = 0.75
    biased_dx = scale * (dx * math.cos(angle) - dy * math.sin(angle))
    biased_dy = scale * (dx * math.sin(angle) + dy * math.cos(angle))
    return biased_dy * fps, biased_dx * fps


def _noisy_motion(
    rng: np.random.Generator,
    truth_wind: tuple[float, float],
    *,
    fps: float,
) -> tuple[float, float]:
    speed = max(float(np.hypot(*truth_wind)), 0.05)
    noise = rng.normal(0.0, 0.12 * speed, size=2)
    measured_dy = float(truth_wind[0] + noise[0])
    measured_dx = float(truth_wind[1] + noise[1])
    return measured_dy * fps, measured_dx * fps
