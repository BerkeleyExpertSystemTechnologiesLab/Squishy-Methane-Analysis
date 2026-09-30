"""Named, reportable model profiles for causal plume experiments.

Profiles make legacy demonstration behavior explicit.  They are configuration
contracts, not evidence that the values are calibrated for real sensors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Literal

from plume_focus.plume_dynamics import PlumeDynamicsConfig


ProfileScenario = Literal["object", "plume", "methane"]
ProfileUse = Literal["synthetic_demo", "uncalibrated_video"]


def _require_finite(
    name: str,
    value: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None:
        invalid = value < minimum if minimum_inclusive else value <= minimum
        if invalid:
            qualifier = "at least" if minimum_inclusive else "greater than"
            raise ValueError(f"{name} must be {qualifier} {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}.")


@dataclass(frozen=True)
class MeasurementModelConfig:
    """Sparse-scan likelihood and offline support-extraction parameters."""

    foreground_threshold_ratio: float
    likelihood_blur_sigma_px: float = 2.0
    neutral_likelihood: float = 0.35
    observed_likelihood_floor: float = 0.05
    observed_likelihood_scale: float = 0.95
    intensity_lower_percentile: float = 5.0
    intensity_upper_percentile: float = 99.0
    minimum_likelihood: float = 1e-6

    def __post_init__(self) -> None:
        for name, value in (
            ("foreground_threshold_ratio", self.foreground_threshold_ratio),
            ("neutral_likelihood", self.neutral_likelihood),
            ("observed_likelihood_floor", self.observed_likelihood_floor),
            ("observed_likelihood_scale", self.observed_likelihood_scale),
        ):
            _require_finite(name, value, minimum=0.0, maximum=1.0)
        _require_finite(
            "likelihood_blur_sigma_px",
            self.likelihood_blur_sigma_px,
            minimum=0.0,
        )
        _require_finite(
            "intensity_lower_percentile",
            self.intensity_lower_percentile,
            minimum=0.0,
            maximum=100.0,
        )
        _require_finite(
            "intensity_upper_percentile",
            self.intensity_upper_percentile,
            minimum=0.0,
            maximum=100.0,
        )
        if self.intensity_lower_percentile >= self.intensity_upper_percentile:
            raise ValueError(
                "intensity_lower_percentile must be below "
                "intensity_upper_percentile."
            )
        _require_finite(
            "minimum_likelihood",
            self.minimum_likelihood,
            minimum=0.0,
            minimum_inclusive=False,
        )
        if (
            self.observed_likelihood_floor
            + self.observed_likelihood_scale
            > 1.0
        ):
            raise ValueError(
                "observed likelihood floor and scale must sum to at most 1."
            )


@dataclass(frozen=True)
class ResolvedUncertaintyConfig:
    """Uncertainty parameters resolved into physical seconds."""

    wind_error_speed_px_per_second: float
    model_error_radius_px: float
    plume_spread_speed_px_per_second: float
    quantization_radius_px: float
    reacquisition_radius_fraction_of_frame_diagonal: float
    diffusion_sigma_floor_px: float
    diffusion_radius_divisor: float

    @property
    def bound_kwargs(self) -> dict[str, float]:
        return {
            "wind_error_speed": self.wind_error_speed_px_per_second,
            "model_error_radius": self.model_error_radius_px,
            "plume_spread_speed": self.plume_spread_speed_px_per_second,
            "quantization_radius": self.quantization_radius_px,
        }


@dataclass(frozen=True)
class UncertaintyModelConfig:
    """Uncalibrated uncertainty-growth parameters with explicit units."""

    wind_error_speed_px_per_frame: float
    model_error_radius_px: float
    plume_spread_speed_px_per_frame: float
    quantization_radius_px: float = 0.5
    reacquisition_radius_fraction_of_frame_diagonal: float = 0.5
    diffusion_sigma_floor_px: float = 0.5
    diffusion_radius_divisor: float = 3.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _require_finite(
                name,
                float(value),
                minimum=0.0,
                minimum_inclusive=(
                    name
                    not in {
                        "reacquisition_radius_fraction_of_frame_diagonal",
                        "diffusion_radius_divisor",
                    }
                ),
            )

    def resolve(self, frame_rate_hz: float) -> ResolvedUncertaintyConfig:
        """Convert legacy per-frame speeds to pixels per physical second."""

        _require_finite(
            "frame_rate_hz",
            frame_rate_hz,
            minimum=0.0,
            minimum_inclusive=False,
        )
        return ResolvedUncertaintyConfig(
            wind_error_speed_px_per_second=(
                self.wind_error_speed_px_per_frame * frame_rate_hz
            ),
            model_error_radius_px=self.model_error_radius_px,
            plume_spread_speed_px_per_second=(
                self.plume_spread_speed_px_per_frame * frame_rate_hz
            ),
            quantization_radius_px=self.quantization_radius_px,
            reacquisition_radius_fraction_of_frame_diagonal=(
                self.reacquisition_radius_fraction_of_frame_diagonal
            ),
            diffusion_sigma_floor_px=self.diffusion_sigma_floor_px,
            diffusion_radius_divisor=self.diffusion_radius_divisor,
        )

    @classmethod
    def from_legacy_bound_kwargs(
        cls,
        values: dict[str, float],
    ) -> ResolvedUncertaintyConfig:
        """Resolve the former ``BoundConfig`` mapping compatibility surface."""

        return cls(
            wind_error_speed_px_per_frame=values.get(
                "wind_error_speed",
                1.0,
            ),
            model_error_radius_px=values.get(
                "model_error_radius",
                1.0,
            ),
            plume_spread_speed_px_per_frame=values.get(
                "plume_spread_speed",
                0.25,
            ),
            quantization_radius_px=values.get(
                "quantization_radius",
                0.5,
            ),
        ).resolve(1.0)


@dataclass(frozen=True)
class StaticCorridorConfig:
    """Declared geometry of the non-oracle fixed source-corridor baseline."""

    downwind_center_px: float = 34.0
    downwind_sigma_px: float = 34.0
    crosswind_sigma_px: float = 24.0
    upwind_decay_length_px: float = 8.0

    def __post_init__(self) -> None:
        _require_finite(
            "downwind_center_px",
            self.downwind_center_px,
            minimum=0.0,
        )
        for name, value in (
            ("downwind_sigma_px", self.downwind_sigma_px),
            ("crosswind_sigma_px", self.crosswind_sigma_px),
            ("upwind_decay_length_px", self.upwind_decay_length_px),
        ):
            _require_finite(
                name,
                value,
                minimum=0.0,
                minimum_inclusive=False,
            )


@dataclass(frozen=True)
class ScanPolicyConfig:
    """Credible-region, source-inclusion, and mechanical-budget policy."""

    credible_mass: float
    credible_blur_sigma_px: float
    minimum_area_fraction: float
    column_mass_fraction: float
    row_mass_fraction: float
    minimum_sweep_columns: int
    minimum_sweep_rows: int
    maximum_sweep_columns: int | None
    force_known_source_in_scan: bool = True
    source_include_radius_min_px: int = 4
    source_include_radius_uncertainty_divisor: float = 3.0
    initial_source_corridor_dy_px_per_frame: float = 0.0
    initial_source_corridor_dx_px_per_frame: float = 1.0
    planning_initial_horizon_frames: float = 1.0
    planning_max_iterations: int = 3
    convergence_absolute_seconds: float = 1e-6
    convergence_relative_fraction: float = 0.05
    static_corridor: StaticCorridorConfig = field(
        default_factory=StaticCorridorConfig
    )

    def __post_init__(self) -> None:
        for name, value in (
            ("credible_mass", self.credible_mass),
            ("column_mass_fraction", self.column_mass_fraction),
            ("row_mass_fraction", self.row_mass_fraction),
        ):
            _require_finite(
                name,
                value,
                minimum=0.0,
                maximum=1.0,
                minimum_inclusive=False,
            )
        _require_finite(
            "credible_blur_sigma_px",
            self.credible_blur_sigma_px,
            minimum=0.0,
        )
        _require_finite(
            "minimum_area_fraction",
            self.minimum_area_fraction,
            minimum=0.0,
            maximum=1.0,
        )
        if self.minimum_sweep_columns <= 0:
            raise ValueError("minimum_sweep_columns must be positive.")
        if self.minimum_sweep_rows <= 0:
            raise ValueError("minimum_sweep_rows must be positive.")
        if (
            self.maximum_sweep_columns is not None
            and self.maximum_sweep_columns <= 0
        ):
            raise ValueError("maximum_sweep_columns must be positive when set.")
        if (
            self.maximum_sweep_columns is not None
            and self.maximum_sweep_columns < self.minimum_sweep_columns
        ):
            raise ValueError(
                "maximum_sweep_columns must be at least "
                "minimum_sweep_columns."
            )
        if self.source_include_radius_min_px < 0:
            raise ValueError("source_include_radius_min_px must be non-negative.")
        _require_finite(
            "source_include_radius_uncertainty_divisor",
            self.source_include_radius_uncertainty_divisor,
            minimum=0.0,
            minimum_inclusive=False,
        )
        for name, value in (
            (
                "initial_source_corridor_dy_px_per_frame",
                self.initial_source_corridor_dy_px_per_frame,
            ),
            (
                "initial_source_corridor_dx_px_per_frame",
                self.initial_source_corridor_dx_px_per_frame,
            ),
        ):
            _require_finite(name, value)
        _require_finite(
            "planning_initial_horizon_frames",
            self.planning_initial_horizon_frames,
            minimum=0.0,
            minimum_inclusive=False,
        )
        if self.planning_max_iterations <= 0:
            raise ValueError("planning_max_iterations must be positive.")
        _require_finite(
            "convergence_absolute_seconds",
            self.convergence_absolute_seconds,
            minimum=0.0,
        )
        _require_finite(
            "convergence_relative_fraction",
            self.convergence_relative_fraction,
            minimum=0.0,
        )


@dataclass(frozen=True)
class ModelProfile:
    """Complete behavior profile consumed by a causal experiment."""

    profile_id: str
    scenario: ProfileScenario
    intended_use: ProfileUse
    provenance: str
    dynamics: PlumeDynamicsConfig
    measurement: MeasurementModelConfig
    uncertainty: UncertaintyModelConfig
    scan_policy: ScanPolicyConfig
    profile_schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id must not be empty.")
        if self.scenario not in {"object", "plume", "methane"}:
            raise ValueError("scenario must be object, plume, or methane.")
        if not self.provenance:
            raise ValueError("provenance must not be empty.")
        if self.profile_schema_version <= 0:
            raise ValueError("profile_schema_version must be positive.")

    def to_payload(self, *, frame_rate_hz: float) -> dict[str, object]:
        """Serialize declared and physically resolved values for evidence."""

        resolved = self.uncertainty.resolve(frame_rate_hz)
        return {
            "profile_schema_version": self.profile_schema_version,
            "profile_id": self.profile_id,
            "scenario": self.scenario,
            "intended_use": self.intended_use,
            "provenance": self.provenance,
            "calibration_status": "uncalibrated",
            "dynamics": asdict(self.dynamics),
            "measurement": asdict(self.measurement),
            "uncertainty": {
                **asdict(self.uncertainty),
                "resolved_for_frame_rate_hz": frame_rate_hz,
                "resolved": asdict(resolved),
            },
            "scan_policy": asdict(self.scan_policy),
        }


def synthetic_demo_v1(scenario: ProfileScenario) -> ModelProfile:
    """Return the explicit compatibility profile for generated fixtures."""

    return _legacy_profile(
        scenario,
        profile_id="synthetic_demo_v1",
        intended_use="synthetic_demo",
        synthetic=True,
    )


def uncalibrated_video_v1(scenario: ProfileScenario) -> ModelProfile:
    """Return explicit legacy settings for uncalibrated file-backed video."""

    return _legacy_profile(
        scenario,
        profile_id="uncalibrated_video_v1",
        intended_use="uncalibrated_video",
        synthetic=False,
    )


def _legacy_profile(
    scenario: ProfileScenario,
    *,
    profile_id: str,
    intended_use: ProfileUse,
    synthetic: bool,
) -> ModelProfile:
    if scenario == "methane":
        foreground_threshold = 0.22 if synthetic else 0.30
        uncertainty = UncertaintyModelConfig(0.25, 2.0, 0.45)
        policy = ScanPolicyConfig(
            credible_mass=0.93,
            credible_blur_sigma_px=3.0,
            minimum_area_fraction=0.08,
            column_mass_fraction=0.8,
            row_mass_fraction=0.85,
            minimum_sweep_columns=30,
            minimum_sweep_rows=56,
            maximum_sweep_columns=120,
        )
    elif scenario == "plume":
        foreground_threshold = 0.24 if synthetic else 0.30
        uncertainty = UncertaintyModelConfig(
            0.7 if synthetic else 0.8,
            3.0,
            0.65,
        )
        policy = ScanPolicyConfig(
            credible_mass=0.9,
            credible_blur_sigma_px=3.0,
            minimum_area_fraction=0.03,
            column_mass_fraction=1.0,
            row_mass_fraction=1.0,
            minimum_sweep_columns=1,
            minimum_sweep_rows=1,
            maximum_sweep_columns=None,
        )
    else:
        foreground_threshold = 0.35
        uncertainty = UncertaintyModelConfig(0.4, 1.5, 0.25)
        policy = ScanPolicyConfig(
            credible_mass=0.9,
            credible_blur_sigma_px=3.0,
            minimum_area_fraction=0.03,
            column_mass_fraction=1.0,
            row_mass_fraction=1.0,
            minimum_sweep_columns=1,
            minimum_sweep_rows=1,
            maximum_sweep_columns=None,
        )
    return ModelProfile(
        profile_id=profile_id,
        scenario=scenario,
        intended_use=intended_use,
        provenance=(
            "Legacy project behavior made explicit for reproducibility; "
            "not calibrated against measured sensor data."
        ),
        dynamics=PlumeDynamicsConfig(),
        measurement=MeasurementModelConfig(
            foreground_threshold_ratio=foreground_threshold,
        ),
        uncertainty=uncertainty,
        scan_policy=policy,
    )
