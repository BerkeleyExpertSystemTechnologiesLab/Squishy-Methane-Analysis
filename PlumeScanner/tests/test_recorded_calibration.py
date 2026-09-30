from __future__ import annotations

import hashlib
from pathlib import Path
import struct

import numpy as np
import pytest

from plume_focus.recorded_calibration import (
    deserialize_fixed_calibration,
    fixed_calibration_sha256,
    load_fixed_calibration,
    serialize_fixed_calibration,
    write_fixed_calibration,
)
from plume_focus.sensor_emulation import FixedLeakOffCalibration


_HEADER_SIZE = struct.calcsize("<8sHHIIQ")
_CHECKSUM_SIZE = hashlib.sha256().digest_size


def _calibration(
    *,
    background: np.ndarray | None = None,
    noise: np.ndarray | None = None,
) -> FixedLeakOffCalibration:
    return FixedLeakOffCalibration(
        background_raw_units=(
            np.asarray([[10.5, 20.25], [30.0, 40.75]], dtype=np.float64)
            if background is None
            else background
        ),
        noise_sigma_raw_units=(
            np.asarray([[1.0, 2.0], [3.5, 4.25]], dtype=np.float64)
            if noise is None
            else noise
        ),
        sample_count=7,
    )


def _replace_float_and_rechecksum(
    encoded: bytes,
    *,
    offset: int,
    value: float,
) -> bytes:
    changed = bytearray(encoded)
    struct.pack_into("<d", changed, offset, value)
    content = bytes(changed[:-_CHECKSUM_SIZE])
    changed[-_CHECKSUM_SIZE:] = hashlib.sha256(content).digest()
    return bytes(changed)


def test_serialization_is_identical_across_paths_and_memory_layouts(
    tmp_path: Path,
) -> None:
    background = np.asarray(
        [[10.5, 20.25], [30.0, 40.75]],
        dtype=">f8",
        order="F",
    )
    noise = np.asfortranarray(
        np.asarray([[1.0, 2.0], [3.5, 4.25]], dtype=np.float64)
    )
    non_native = _calibration(background=background, noise=noise)
    ordinary = _calibration()

    first_path = tmp_path / "first" / "calibration.bin"
    second_path = tmp_path / "second" / "different-name.bin"
    first_hash = write_fixed_calibration(first_path, non_native)
    second_hash = write_fixed_calibration(second_path, ordinary)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_hash == second_hash
    assert first_hash == fixed_calibration_sha256(ordinary)


def test_fixed_calibration_round_trip_preserves_values_and_read_only_state(
    tmp_path: Path,
) -> None:
    original = _calibration()
    path = tmp_path / "calibration.bin"
    digest = write_fixed_calibration(path, original)

    loaded = load_fixed_calibration(path, expected_sha256=digest.upper())

    np.testing.assert_array_equal(
        loaded.background_raw_units,
        original.background_raw_units,
    )
    np.testing.assert_array_equal(
        loaded.noise_sigma_raw_units,
        original.noise_sigma_raw_units,
    )
    assert loaded.sample_count == original.sample_count
    assert not loaded.background_raw_units.flags.writeable
    assert not loaded.noise_sigma_raw_units.flags.writeable
    assert serialize_fixed_calibration(loaded) == path.read_bytes()


def test_one_value_mutation_changes_canonical_hash() -> None:
    original = _calibration()
    changed_background = original.background_raw_units.copy()
    changed_background[1, 1] += 1.0
    changed = _calibration(background=changed_background)

    assert fixed_calibration_sha256(original) != (
        fixed_calibration_sha256(changed)
    )


@pytest.mark.parametrize("mutation", ["truncated", "extra", "checksum", "magic"])
def test_decoder_rejects_malformed_artifacts(mutation: str) -> None:
    encoded = serialize_fixed_calibration(_calibration())
    if mutation == "truncated":
        malformed = encoded[:-1]
    elif mutation == "extra":
        malformed = encoded + b"\x00"
    elif mutation == "checksum":
        malformed_bytes = bytearray(encoded)
        malformed_bytes[_HEADER_SIZE] ^= 1
        malformed = bytes(malformed_bytes)
    else:
        malformed = b"NOTCAL01" + encoded[8:]

    with pytest.raises(ValueError):
        deserialize_fixed_calibration(malformed)


@pytest.mark.parametrize(
    ("payload_offset", "value", "message"),
    [
        (_HEADER_SIZE, float("nan"), "background must be finite"),
        (_HEADER_SIZE + 4 * 8, 0.0, "noise must be finite and positive"),
        (_HEADER_SIZE + 4 * 8, -1.0, "noise must be finite and positive"),
    ],
)
def test_decoder_rejects_invalid_values_even_with_valid_checksum(
    payload_offset: int,
    value: float,
    message: str,
) -> None:
    encoded = serialize_fixed_calibration(_calibration())
    malformed = _replace_float_and_rechecksum(
        encoded,
        offset=payload_offset,
        value=value,
    )

    with pytest.raises(ValueError, match=message):
        deserialize_fixed_calibration(malformed)


def test_writer_is_idempotent_and_refuses_different_overwrite(
    tmp_path: Path,
) -> None:
    path = tmp_path / "calibration.bin"
    original = _calibration()
    digest = write_fixed_calibration(path, original)
    original_bytes = path.read_bytes()
    original_stat = path.stat()

    repeated_digest = write_fixed_calibration(path, original)

    assert repeated_digest == digest
    assert path.read_bytes() == original_bytes
    assert path.stat().st_mtime_ns == original_stat.st_mtime_ns

    changed_noise = original.noise_sigma_raw_units.copy()
    changed_noise[0, 0] += 0.5
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_fixed_calibration(
            path,
            _calibration(noise=changed_noise),
        )
    assert path.read_bytes() == original_bytes


def test_load_rejects_wrong_or_malformed_expected_hash(
    tmp_path: Path,
) -> None:
    path = tmp_path / "calibration.bin"
    write_fixed_calibration(path, _calibration())

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_fixed_calibration(path, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="64 hexadecimal"):
        load_fixed_calibration(path, expected_sha256="not-a-hash")
