"""Canonical fixed-calibration artifacts for recorded sensor emulation.

The artifact contains only resolved calibration values: shape, leak-off sample
count, per-pixel background, and per-pixel noise.  It deliberately contains no
video, paths, provenance, or dataset lineage; those belong in evidence
manifests that hash this artifact.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path
import struct
import tempfile

import numpy as np

from plume_focus.sensor_emulation import FixedLeakOffCalibration


__all__ = [
    "deserialize_fixed_calibration",
    "fixed_calibration_sha256",
    "load_fixed_calibration",
    "serialize_fixed_calibration",
    "write_fixed_calibration",
]


_MAGIC = b"PLMCAL01"
_FORMAT_VERSION = 1
_FLAGS = 0
_HEADER = struct.Struct("<8sHHIIQ")
_CHECKSUM_SIZE = hashlib.sha256().digest_size
_FLOAT_BYTES = np.dtype("<f8").itemsize
_MAX_UINT32 = (1 << 32) - 1
_MAX_UINT64 = (1 << 64) - 1


def serialize_fixed_calibration(
    calibration: FixedLeakOffCalibration,
) -> bytes:
    """Serialize one calibration into canonical, platform-neutral bytes."""

    rows, cols = calibration.shape
    if rows <= 0 or cols <= 0:
        raise ValueError("calibration shape must be non-empty.")
    if rows > _MAX_UINT32 or cols > _MAX_UINT32:
        raise ValueError("calibration dimensions exceed artifact limits.")
    if calibration.sample_count > _MAX_UINT64:
        raise ValueError("sample_count exceeds artifact limits.")

    background = _canonical_array(
        calibration.background_raw_units,
        name="calibration background",
        positive=False,
    )
    noise = _canonical_array(
        calibration.noise_sigma_raw_units,
        name="calibration noise",
        positive=True,
    )
    if background.shape != (rows, cols) or noise.shape != (rows, cols):
        raise ValueError("calibration arrays must match calibration shape.")

    header = _HEADER.pack(
        _MAGIC,
        _FORMAT_VERSION,
        _FLAGS,
        rows,
        cols,
        calibration.sample_count,
    )
    content = (
        header
        + background.tobytes(order="C")
        + noise.tobytes(order="C")
    )
    return content + hashlib.sha256(content).digest()


def deserialize_fixed_calibration(
    artifact: bytes | bytearray | memoryview,
) -> FixedLeakOffCalibration:
    """Strictly decode and validate one canonical calibration artifact."""

    encoded = bytes(artifact)
    minimum_size = _HEADER.size + _CHECKSUM_SIZE
    if len(encoded) < minimum_size:
        raise ValueError("fixed-calibration artifact is truncated.")

    try:
        magic, version, flags, rows, cols, sample_count = _HEADER.unpack_from(
            encoded
        )
    except struct.error as error:
        raise ValueError("fixed-calibration header is malformed.") from error
    if magic != _MAGIC:
        raise ValueError("fixed-calibration artifact has invalid magic.")
    if version != _FORMAT_VERSION:
        raise ValueError(
            "unsupported fixed-calibration artifact version."
        )
    if flags != _FLAGS:
        raise ValueError("fixed-calibration artifact has unknown flags.")
    if rows <= 0 or cols <= 0:
        raise ValueError("fixed-calibration shape must be non-empty.")
    if sample_count < 2:
        raise ValueError(
            "fixed-calibration sample_count must be at least 2."
        )

    value_count = rows * cols
    expected_size = (
        _HEADER.size
        + 2 * value_count * _FLOAT_BYTES
        + _CHECKSUM_SIZE
    )
    if len(encoded) != expected_size:
        raise ValueError(
            "fixed-calibration artifact has an invalid byte length."
        )

    content = encoded[:-_CHECKSUM_SIZE]
    stored_checksum = encoded[-_CHECKSUM_SIZE:]
    actual_checksum = hashlib.sha256(content).digest()
    if not hmac.compare_digest(stored_checksum, actual_checksum):
        raise ValueError("fixed-calibration artifact checksum mismatch.")

    background_start = _HEADER.size
    noise_start = background_start + value_count * _FLOAT_BYTES
    checksum_start = len(content)
    background = np.frombuffer(
        content,
        dtype="<f8",
        count=value_count,
        offset=background_start,
    ).reshape((rows, cols), order="C")
    noise = np.frombuffer(
        content,
        dtype="<f8",
        count=value_count,
        offset=noise_start,
    ).reshape((rows, cols), order="C")
    if noise_start + noise.nbytes != checksum_start:
        raise ValueError(
            "fixed-calibration artifact contains unexpected payload bytes."
        )
    if np.any(~np.isfinite(background)):
        raise ValueError("fixed-calibration background must be finite.")
    if np.any(~np.isfinite(noise)) or np.any(noise <= 0.0):
        raise ValueError(
            "fixed-calibration noise must be finite and positive."
        )

    return FixedLeakOffCalibration(
        background_raw_units=background,
        noise_sigma_raw_units=noise,
        sample_count=int(sample_count),
    )


def fixed_calibration_sha256(
    calibration: FixedLeakOffCalibration,
) -> str:
    """Return the SHA-256 of the canonical serialized calibration."""

    return hashlib.sha256(
        serialize_fixed_calibration(calibration)
    ).hexdigest()


def load_fixed_calibration(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> FixedLeakOffCalibration:
    """Load an artifact, optionally requiring its canonical file SHA-256."""

    artifact_path = Path(path)
    if artifact_path.is_symlink():
        raise ValueError("fixed-calibration artifact must not be a symlink.")
    encoded = artifact_path.read_bytes()
    actual_sha256 = hashlib.sha256(encoded).hexdigest()
    if expected_sha256 is not None:
        expected = _normalize_sha256(expected_sha256)
        if not hmac.compare_digest(actual_sha256, expected):
            raise ValueError(
                "fixed-calibration artifact SHA-256 mismatch."
            )
    return deserialize_fixed_calibration(encoded)


def write_fixed_calibration(
    path: Path,
    calibration: FixedLeakOffCalibration,
) -> str:
    """Atomically create an artifact or accept identical existing bytes.

    Existing different content is never replaced.  The returned digest is the
    SHA-256 of the exact canonical bytes at ``path``.
    """

    artifact_path = Path(path)
    encoded = serialize_fixed_calibration(calibration)
    digest = hashlib.sha256(encoded).hexdigest()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    if artifact_path.exists() or artifact_path.is_symlink():
        _require_identical_existing(artifact_path, encoded)
        return digest

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{artifact_path.name}.",
        suffix=".tmp",
        dir=artifact_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # A same-directory hard link publishes complete bytes atomically
            # and, unlike replace/rename, cannot overwrite existing content.
            os.link(temporary_path, artifact_path)
        except FileExistsError:
            _require_identical_existing(artifact_path, encoded)
    finally:
        temporary_path.unlink(missing_ok=True)
    return digest


def _canonical_array(
    values: np.ndarray,
    *,
    name: str,
    positive: bool,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array.")
    canonical = np.ascontiguousarray(array, dtype="<f8")
    if np.any(~np.isfinite(canonical)):
        raise ValueError(f"{name} must be finite.")
    if positive and np.any(canonical <= 0.0):
        raise ValueError(f"{name} must be positive.")
    return canonical


def _require_identical_existing(path: Path, expected: bytes) -> None:
    if path.is_symlink():
        raise ValueError("fixed-calibration artifact must not be a symlink.")
    if not path.is_file():
        raise FileExistsError(
            f"fixed-calibration output is not a regular file: {path}"
        )
    if path.read_bytes() != expected:
        raise FileExistsError(
            "refusing to overwrite a different fixed-calibration artifact: "
            f"{path}"
        )


def _normalize_sha256(value: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64:
        raise ValueError("expected_sha256 must contain 64 hexadecimal characters.")
    try:
        int(normalized, 16)
    except ValueError as error:
        raise ValueError(
            "expected_sha256 must contain 64 hexadecimal characters."
        ) from error
    return normalized
