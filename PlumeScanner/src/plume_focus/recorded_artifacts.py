"""Deterministic, no-overwrite writers for recorded evidence metadata."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping


def canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    """Serialize one JSON object into stable, human-readable UTF-8 bytes."""

    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_json_artifact(
    path: Path,
    payload: Mapping[str, object],
) -> str:
    """Publish canonical JSON without replacing different existing bytes."""

    return write_bytes_artifact(path, canonical_json_bytes(payload))


def write_bytes_artifact(path: Path, content: bytes) -> str:
    """Atomically create a file or accept an identical existing artifact."""

    artifact_path = Path(path)
    digest = hashlib.sha256(content).hexdigest()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    if artifact_path.exists() or artifact_path.is_symlink():
        _require_identical_existing(artifact_path, content)
        return digest

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{artifact_path.name}.",
        suffix=".tmp",
        dir=artifact_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, artifact_path)
        except FileExistsError:
            _require_identical_existing(artifact_path, content)
    finally:
        temporary_path.unlink(missing_ok=True)
    return digest


def _require_identical_existing(path: Path, expected: bytes) -> None:
    if path.is_symlink():
        raise ValueError("recorded artifact output must not be a symlink.")
    if not path.is_file():
        raise FileExistsError(
            f"recorded artifact output is not a regular file: {path}"
        )
    if path.read_bytes() != expected:
        raise FileExistsError(
            "refusing to overwrite a different recorded artifact: "
            f"{path}"
        )
