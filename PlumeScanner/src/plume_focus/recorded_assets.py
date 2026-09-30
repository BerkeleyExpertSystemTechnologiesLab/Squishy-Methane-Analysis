"""Hashed asset, clock, normalization, and wind adapters."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from plume_focus.predictor import BoundingBox
from plume_focus.recorded_contracts import (
    CausalMotionProvider,
    FrameClockSpec,
    NormalizationSpec,
    SourceAssumptionSpec,
    WindSpec,
)
from plume_focus.recorded_manifest_io import file_sha256
from plume_focus.recorded_providers import (
    ConstantMotionProvider,
    MissingMotionProvider,
    TimeSeriesMotionProvider,
)


def resolve_frame_clock(
    spec: FrameClockSpec,
    *,
    data_root: Path,
    frame_count: int,
    container_fps: float,
    verify_hashes: bool,
) -> tuple[tuple[float, ...], float, str | None]:
    """Resolve the authoritative frame timestamps and nominal rate."""

    if spec.mode == "constant_fps":
        assert spec.fps is not None
        tolerance = max(0.05, 0.01 * spec.fps)
        if abs(container_fps - spec.fps) > tolerance:
            raise ValueError(
                "container FPS differs from the manifest by more than "
                "the allowed 1%/0.05 Hz tolerance."
            )
        timestamps = tuple(
            index / spec.fps
            for index in range(frame_count)
        )
        return timestamps, spec.fps, None

    assert spec.path is not None and spec.sha256 is not None
    path = resolve_data_asset(data_root, spec.path)
    if verify_hashes:
        verify_file_hash(path, spec.sha256)
    timestamps = _load_frame_timestamps(path, frame_count)
    nominal_fps = (
        (len(timestamps) - 1) / timestamps[-1]
        if len(timestamps) > 1 and timestamps[-1] > 0.0
        else 1.0
    )
    return timestamps, nominal_fps, spec.sha256


def load_motion_provider(
    spec: WindSpec,
    *,
    data_root: Path,
    verify_hashes: bool,
) -> tuple[CausalMotionProvider, str | None]:
    """Resolve the manifest wind declaration into a causal provider."""

    if spec.mode == "none":
        return MissingMotionProvider(), None
    if spec.mode == "constant":
        assert (
            spec.dy_px_per_second is not None
            and spec.dx_px_per_second is not None
        )
        return (
            ConstantMotionProvider(
                spec.dy_px_per_second,
                spec.dx_px_per_second,
            ),
            None,
        )
    assert (
        spec.path is not None
        and spec.sha256 is not None
        and spec.max_age_seconds is not None
    )
    path = resolve_data_asset(data_root, spec.path)
    if verify_hashes:
        verify_file_hash(path, spec.sha256)
    timestamps, dy, dx = _load_wind_csv(path)
    return (
        TimeSeriesMotionProvider(
            timestamps,
            dy,
            dx,
            spec.max_age_seconds,
        ),
        spec.sha256,
    )


def resolve_source_assumption(
    spec: SourceAssumptionSpec,
    shape: tuple[int, int],
) -> tuple[tuple[float, float] | None, BoundingBox]:
    """Resolve and bounds-check the declared source assumption."""

    if spec.mode == "source_free":
        return None, BoundingBox(0, 0, shape[0] - 1, shape[1] - 1)
    assert spec.initial_roi is not None
    roi = BoundingBox(*spec.initial_roi)
    if roi.clipped(shape) != roi:
        raise ValueError("manifest initial_roi lies outside the frame.")
    if spec.source_point is not None:
        row, col = spec.source_point
        if not (0.0 <= row < shape[0] and 0.0 <= col < shape[1]):
            raise ValueError(
                "manifest source_point lies outside the frame."
            )
    return spec.source_point, roi


def normalize_frame(
    frame: NDArray[np.uint8],
    spec: NormalizationSpec,
) -> NDArray[np.uint8]:
    """Apply the sequence-fixed normalization declared by the manifest."""

    values = np.asarray(frame)
    if spec.mode == "uint8_identity":
        return np.asarray(values, dtype=np.uint8).copy()
    assert spec.raw_min is not None and spec.raw_max is not None
    normalized = (
        (values.astype(float) - spec.raw_min)
        / (spec.raw_max - spec.raw_min)
    )
    return np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)


def data_root_for_manifest(manifest_path: Path) -> Path:
    """Return the recorded-data root for a manifest in `manifests/`."""

    if manifest_path.parent.name != "manifests":
        raise ValueError(
            "sequence manifests must live in a manifests directory whose "
            "parent is the recorded data root."
        )
    return manifest_path.parent.parent.resolve()


def resolve_data_asset(data_root: Path, relative_path: str) -> Path:
    """Resolve one confined asset path below the recorded-data root."""

    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(
            "recorded asset paths must be relative and contain no '..'."
        )
    candidate = (data_root / relative).resolve()
    if candidate != data_root and data_root not in candidate.parents:
        raise ValueError("recorded asset path escapes the data root.")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def resolve_data_asset_without_symlinks(
    data_root: Path,
    relative_path: str,
    *,
    label: str,
) -> Path:
    """Resolve one confined asset while rejecting declared symlink paths.

    ``Path.resolve()`` follows symlinks, so checking ``is_symlink()`` on its
    result cannot establish that the declared lineage path was symlink-free.
    Inspect each lexical component before canonicalizing instead.
    """

    root = Path(data_root).resolve()
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(
            "recorded asset paths must be relative and contain no '..'."
        )
    lexical = root / relative
    current = root
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{label} must not use symlink paths.")
    candidate = lexical.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("recorded asset path escapes the data root.")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def resolve_manifest_reference(
    manifest_directory: Path,
    relative_path: str,
) -> Path:
    """Resolve one confined sequence reference in a suite manifest."""

    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(
            "suite sequence paths must be relative and contain no '..'."
        )
    candidate = (manifest_directory / relative).resolve()
    if (
        candidate != manifest_directory
        and manifest_directory not in candidate.parents
    ):
        raise ValueError("suite sequence path escapes its manifest directory.")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def verify_file_hash(path: Path, expected: str) -> None:
    """Reject an asset whose SHA-256 differs from its declaration."""

    actual = file_sha256(path)
    if actual != expected.lower():
        raise ValueError(f"SHA-256 mismatch for asset: {path}")


def _load_frame_timestamps(
    path: Path,
    frame_count: int,
) -> tuple[float, ...]:
    rows = _load_csv_rows(
        path,
        expected_columns={"frame_index", "timestamp_seconds"},
    )
    if len(rows) != frame_count:
        raise ValueError(
            "timestamp row count must match the decoded frame count."
        )
    timestamps: list[float] = []
    for expected_index, row in enumerate(rows):
        actual_index = int(row["frame_index"])
        if actual_index != expected_index:
            raise ValueError(
                "timestamp frame_index values must be contiguous from zero."
            )
        timestamps.append(float(row["timestamp_seconds"]))
    values = np.asarray(timestamps, dtype=float)
    if (
        np.any(~np.isfinite(values))
        or values[0] != 0.0
        or np.any(np.diff(values) <= 0.0)
    ):
        raise ValueError(
            "frame timestamps must start at zero and increase strictly."
        )
    return tuple(float(value) for value in values)


def _load_wind_csv(
    path: Path,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    rows = _load_csv_rows(
        path,
        expected_columns={
            "timestamp_seconds",
            "dy_px_per_second",
            "dx_px_per_second",
        },
    )
    timestamps = tuple(float(row["timestamp_seconds"]) for row in rows)
    dy = tuple(float(row["dy_px_per_second"]) for row in rows)
    dx = tuple(float(row["dx_px_per_second"]) for row in rows)
    TimeSeriesMotionProvider(timestamps, dy, dx, max_age_seconds=0.0)
    return timestamps, dy, dx


def _load_csv_rows(
    path: Path,
    *,
    expected_columns: set[str],
) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or ()) != expected_columns:
            raise ValueError(
                f"{path.name} columns must be "
                f"{sorted(expected_columns)}."
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path.name} must contain at least one data row.")
    return rows
