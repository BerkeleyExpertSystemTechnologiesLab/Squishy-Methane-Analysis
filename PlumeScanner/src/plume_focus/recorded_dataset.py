"""Dataset-neutral accession auditing for recorded plume videos.

This module inventories external video assets before they are converted into
sequence manifests.  It deliberately does not prepare labels, choose splits, or
make an evidence claim: those are separate, reviewable stages.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Literal

import cv2
import numpy as np

from plume_focus.recorded_manifest_io import file_sha256


RightsStatus = Literal["verified", "unverified", "restricted"]
AssetAuditStatus = Literal[
    "valid",
    "unreadable",
    "empty",
    "insufficient_frames",
    "inconsistent_shape",
    "frame_count_mismatch",
]

DATASET_AUDIT_SCHEMA_VERSION = 2
SUPPORTED_DATASET_AUDIT_SCHEMA_VERSIONS = frozenset({1, 2})
DATASET_ACCESSION_SCHEMA_VERSION = 1
VIDEO_SUFFIXES = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4"})


def opencv_decoder_environment_sha256() -> str:
    """Hash the OpenCV version and build configuration used for decoding."""

    identity = (
        f"opencv_version={cv2.__version__}\n"
        f"{cv2.getBuildInformation()}"
    ).encode("utf-8")
    return hashlib.sha256(identity).hexdigest()


def decoded_frame_content_sha256(
    frames: Iterable[np.ndarray],
) -> str:
    """Hash decoded array dtype, shape, order, and exact pixel bytes."""

    digest = hashlib.sha256()
    digest.update(b"plume-focus-decoded-frame-stream-v1\0")
    frame_count = 0
    for frame_count, frame in enumerate(frames, start=1):
        _update_decoded_frame_digest(digest, frame)
    digest.update(frame_count.to_bytes(8, byteorder="big"))
    return digest.hexdigest()


def _update_decoded_frame_digest(
    digest: Any,
    frame: np.ndarray,
) -> None:
    array = np.ascontiguousarray(frame)
    dtype = array.dtype.str.encode("ascii")
    digest.update(len(dtype).to_bytes(2, byteorder="big"))
    digest.update(dtype)
    digest.update(array.ndim.to_bytes(2, byteorder="big"))
    for dimension in array.shape:
        digest.update(int(dimension).to_bytes(8, byteorder="big"))
    digest.update(array.nbytes.to_bytes(8, byteorder="big"))
    digest.update(array.tobytes(order="C"))


def _require_text(name: str, value: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{name} must not be empty.")


def _require_sha256(name: str, value: str) -> None:
    _require_text(name, value)
    if len(value) != 64:
        raise ValueError(f"{name} must contain 64 hexadecimal characters.")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(
            f"{name} must contain 64 hexadecimal characters."
        ) from error


def _require_utc_timestamp(name: str, value: str) -> None:
    _require_text(name, value)
    if not value.endswith("Z"):
        raise ValueError(f"{name} must use an explicit UTC Z suffix.")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO-8601 timestamp.") from error
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"{name} must use UTC.")


@dataclass(frozen=True)
class DatasetAccession:
    """Declared identity, retrieval, and rights state for external data."""

    dataset_id: str
    accession_id: str
    source_uri: str
    accessed_at_utc: str
    rights_status: RightsStatus
    rights_provenance: str
    rights_uri: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("dataset_id", self.dataset_id),
            ("accession_id", self.accession_id),
            ("source_uri", self.source_uri),
            ("rights_provenance", self.rights_provenance),
        ):
            _require_text(name, value)
        _require_utc_timestamp("accessed_at_utc", self.accessed_at_utc)
        if self.rights_status not in {
            "verified",
            "unverified",
            "restricted",
        }:
            raise ValueError("unknown rights_status.")
        if self.rights_status == "verified":
            if self.rights_uri is None:
                raise ValueError(
                    "verified rights require a rights_uri."
                )
            _require_text("rights_uri", self.rights_uri)
        elif self.rights_uri is not None:
            _require_text("rights_uri", self.rights_uri)


@dataclass(frozen=True)
class VideoAssetAudit:
    """Deterministic hash and decode result for one candidate video."""

    relative_path: str
    sha256: str
    byte_count: int
    status: AssetAuditStatus
    decoded_frame_count: int
    container_frame_count: int | None
    frame_shape: tuple[int, int] | None
    nominal_fps: float | None
    nominal_duration_seconds: float | None
    container_fourcc: str | None
    issue: str | None = None
    decoded_content_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_text("video relative_path", self.relative_path)
        relative = PurePosixPath(self.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                "video relative_path must be confined and relative."
            )
        _require_sha256("video sha256", self.sha256)
        for name, value in (
            ("video byte_count", self.byte_count),
            ("decoded_frame_count", self.decoded_frame_count),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer.")
        if self.container_frame_count is not None and (
            isinstance(self.container_frame_count, bool)
            or not isinstance(self.container_frame_count, int)
            or self.container_frame_count <= 0
        ):
            raise ValueError(
                "container_frame_count must be a positive integer or null."
            )
        if self.frame_shape is not None and (
            len(self.frame_shape) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in self.frame_shape
            )
        ):
            raise ValueError(
                "frame_shape must contain two positive integers or be null."
            )
        for name, value in (
            ("nominal_fps", self.nominal_fps),
            ("nominal_duration_seconds", self.nominal_duration_seconds),
        ):
            if value is not None and (
                not math.isfinite(value) or value <= 0.0
            ):
                raise ValueError(f"{name} must be positive and finite.")
        if self.container_fourcc is not None:
            _require_text("container_fourcc", self.container_fourcc)
        if self.decoded_content_sha256 is not None:
            _require_sha256(
                "decoded_content_sha256",
                self.decoded_content_sha256,
            )
        if self.status not in {
            "valid",
            "unreadable",
            "empty",
            "insufficient_frames",
            "inconsistent_shape",
            "frame_count_mismatch",
        }:
            raise ValueError("unknown video audit status.")
        if self.status == "valid":
            if (
                self.decoded_frame_count <= 0
                or self.frame_shape is None
                or self.issue is not None
            ):
                raise ValueError(
                    "valid video audit requires decoded frames and shape "
                    "with no issue."
                )
        elif not isinstance(self.issue, str) or not self.issue.strip():
            raise ValueError(
                "invalid video audit status requires an issue."
            )


@dataclass(frozen=True)
class DatasetAuditReport:
    """Inventory report that is intentionally not sequence evidence."""

    accession: DatasetAccession
    accession_manifest_sha256: str
    assets: tuple[VideoAssetAudit, ...]
    dataset_audit_schema_version: int = DATASET_AUDIT_SCHEMA_VERSION
    decode_scope: str = "full_sequential_decode"
    opencv_version: str = cv2.__version__
    opencv_build_sha256: str | None = field(
        default_factory=lambda: opencv_decoder_environment_sha256()
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.dataset_audit_schema_version, bool)
            or not isinstance(self.dataset_audit_schema_version, int)
            or self.dataset_audit_schema_version
            not in SUPPORTED_DATASET_AUDIT_SCHEMA_VERSIONS
        ):
            raise ValueError("unsupported dataset audit schema version.")
        _require_sha256(
            "accession_manifest_sha256",
            self.accession_manifest_sha256,
        )
        if self.decode_scope != "full_sequential_decode":
            raise ValueError(
                "dataset audit decode_scope must be full_sequential_decode."
            )
        _require_text("opencv_version", self.opencv_version)
        if self.dataset_audit_schema_version == 1:
            if self.opencv_build_sha256 is not None:
                raise ValueError(
                    "dataset audit schema v1 has no OpenCV build hash."
                )
            if any(
                asset.decoded_content_sha256 is not None
                for asset in self.assets
            ):
                raise ValueError(
                    "dataset audit schema v1 has no decoded-content hashes."
                )
        elif self.opencv_build_sha256 is None:
            raise ValueError(
                "dataset audit schema v2 requires opencv_build_sha256."
            )
        else:
            _require_sha256(
                "opencv_build_sha256",
                self.opencv_build_sha256,
            )
            if any(
                asset.decoded_content_sha256 is None
                for asset in self.assets
            ):
                raise ValueError(
                    "dataset audit schema v2 requires a decoded-content "
                    "hash for every video."
                )
        relative_paths = [asset.relative_path for asset in self.assets]
        if len(relative_paths) != len(set(relative_paths)):
            raise ValueError(
                "dataset audit asset paths must be unique."
            )

    @property
    def valid(self) -> bool:
        return bool(self.assets) and all(
            asset.status == "valid" for asset in self.assets
        )

    @property
    def rights_blocked(self) -> bool:
        return self.accession.rights_status != "verified"

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "dataset_audit_schema_version": (
                self.dataset_audit_schema_version
            ),
            "artifact_kind": "dataset_accession_audit",
            "claim_eligibility": "not_sequence_evidence",
            "valid": self.valid,
            "rights_blocked": self.rights_blocked,
            "asset_count": len(self.assets),
            "decode_scope": self.decode_scope,
            "opencv_version": self.opencv_version,
            "accession_manifest_sha256": (
                self.accession_manifest_sha256
            ),
            "accession": asdict(self.accession),
            "assets": [
                _video_audit_payload(
                    asset,
                    include_decoded_content=(
                        self.dataset_audit_schema_version >= 2
                    ),
                )
                for asset in self.assets
            ],
        }
        if self.dataset_audit_schema_version >= 2:
            payload["opencv_build_sha256"] = (
                self.opencv_build_sha256
            )
        return payload

    def to_json(self) -> str:
        return json.dumps(
            self.to_payload(),
            indent=2,
            sort_keys=True,
        )


def audit_dataset_directory(
    dataset_root: Path,
    accession: DatasetAccession,
    *,
    accession_manifest_sha256: str,
) -> DatasetAuditReport:
    """Hash and fully decode every supported video below ``dataset_root``."""

    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise ValueError(f"dataset root is not a directory: {root}")
    candidates: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        if path.is_symlink():
            raise ValueError(
                f"dataset video assets must not be symlinks: {path}"
            )
        try:
            path.resolve().relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"dataset video asset escapes the declared root: {path}"
            ) from error
        candidates.append(path)
    candidates = sorted(
        candidates,
        key=lambda path: path.relative_to(root).as_posix(),
    )
    return DatasetAuditReport(
        accession=accession,
        accession_manifest_sha256=accession_manifest_sha256,
        assets=tuple(
            _audit_video(path, root=root) for path in candidates
        ),
    )


def load_dataset_accession(path: Path) -> DatasetAccession:
    """Load one strict, reusable dataset-accession declaration."""

    from plume_focus.recorded_manifest_io import load_json_object

    payload = load_json_object(Path(path))
    required = {
        "dataset_accession_schema_version",
        "dataset_id",
        "accession_id",
        "source_uri",
        "accessed_at_utc",
        "rights_status",
        "rights_provenance",
        "rights_uri",
    }
    missing = required - set(payload)
    extra = set(payload) - required
    if missing:
        raise ValueError(
            f"dataset accession is missing keys: {sorted(missing)}"
        )
    if extra:
        raise ValueError(
            f"dataset accession has unknown keys: {sorted(extra)}"
        )
    version = payload["dataset_accession_schema_version"]
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != DATASET_ACCESSION_SCHEMA_VERSION
    ):
        raise ValueError("unsupported dataset accession schema version.")
    return _parse_accession_payload(
        {
            key: value
            for key, value in payload.items()
            if key != "dataset_accession_schema_version"
        }
    )


def parse_dataset_audit_report(
    payload: dict[str, Any],
) -> DatasetAuditReport:
    """Strictly parse a supported dataset-audit artifact."""

    audit_version = _require_json_int(
        "dataset_audit_schema_version",
        payload.get("dataset_audit_schema_version"),
        minimum=1,
    )
    if audit_version not in SUPPORTED_DATASET_AUDIT_SCHEMA_VERSIONS:
        raise ValueError(
            "dataset audit must use dataset_audit_schema_version 1 or 2."
        )
    required = {
        "dataset_audit_schema_version",
        "artifact_kind",
        "claim_eligibility",
        "valid",
        "rights_blocked",
        "asset_count",
        "decode_scope",
        "opencv_version",
        "accession_manifest_sha256",
        "accession",
        "assets",
    }
    if audit_version >= 2:
        required.add("opencv_build_sha256")
    _check_exact_keys(payload, required, context="dataset audit")
    if payload["artifact_kind"] != "dataset_accession_audit":
        raise ValueError(
            "dataset audit artifact_kind must be dataset_accession_audit."
        )
    if payload["claim_eligibility"] != "not_sequence_evidence":
        raise ValueError(
            "dataset audit claim_eligibility must be not_sequence_evidence."
        )
    accession_payload = payload["accession"]
    if not isinstance(accession_payload, dict):
        raise ValueError("dataset audit accession must be an object.")
    accession = _parse_accession_payload(accession_payload)
    asset_payloads = payload["assets"]
    if not isinstance(asset_payloads, list):
        raise ValueError("dataset audit assets must be an array.")
    assets = tuple(
        _parse_video_audit(item, audit_version=audit_version)
        for item in asset_payloads
    )
    asset_count = _require_json_int(
        "dataset audit asset_count",
        payload["asset_count"],
        minimum=0,
    )
    if asset_count != len(assets):
        raise ValueError(
            "dataset audit asset_count does not match its assets array."
        )
    report = DatasetAuditReport(
        accession=accession,
        accession_manifest_sha256=_require_json_string(
            "accession_manifest_sha256",
            payload["accession_manifest_sha256"],
        ),
        assets=assets,
        dataset_audit_schema_version=audit_version,
        decode_scope=_require_json_string(
            "dataset audit decode_scope",
            payload["decode_scope"],
        ),
        opencv_version=_require_json_string(
            "dataset audit opencv_version",
            payload["opencv_version"],
        ),
        opencv_build_sha256=(
            _require_json_string(
                "dataset audit opencv_build_sha256",
                payload["opencv_build_sha256"],
            )
            if audit_version >= 2
            else None
        ),
    )
    if not isinstance(payload["valid"], bool):
        raise ValueError("dataset audit valid must be boolean.")
    if payload["valid"] != report.valid:
        raise ValueError(
            "dataset audit valid does not match its asset results."
        )
    if not isinstance(payload["rights_blocked"], bool):
        raise ValueError("dataset audit rights_blocked must be boolean.")
    if payload["rights_blocked"] != report.rights_blocked:
        raise ValueError(
            "dataset audit rights status and rights_blocked disagree."
        )
    return report


def _parse_accession_payload(
    payload: dict[str, Any],
) -> DatasetAccession:
    required = {
        "dataset_id",
        "accession_id",
        "source_uri",
        "accessed_at_utc",
        "rights_status",
        "rights_provenance",
        "rights_uri",
    }
    _check_exact_keys(payload, required, context="dataset accession")
    rights_uri = payload["rights_uri"]
    if rights_uri is not None and not isinstance(rights_uri, str):
        raise ValueError("rights_uri must be a string or null.")
    return DatasetAccession(
        dataset_id=_require_json_string(
            "dataset_id",
            payload["dataset_id"],
        ),
        accession_id=_require_json_string(
            "accession_id",
            payload["accession_id"],
        ),
        source_uri=_require_json_string(
            "source_uri",
            payload["source_uri"],
        ),
        accessed_at_utc=_require_json_string(
            "accessed_at_utc",
            payload["accessed_at_utc"],
        ),
        rights_status=_require_json_string(  # type: ignore[arg-type]
            "rights_status",
            payload["rights_status"],
        ),
        rights_provenance=_require_json_string(
            "rights_provenance",
            payload["rights_provenance"],
        ),
        rights_uri=rights_uri,
    )


def _parse_video_audit(
    payload: object,
    *,
    audit_version: int,
) -> VideoAssetAudit:
    if not isinstance(payload, dict):
        raise ValueError("dataset audit asset must be an object.")
    required = {
        "relative_path",
        "sha256",
        "byte_count",
        "status",
        "decoded_frame_count",
        "container_frame_count",
        "frame_shape",
        "nominal_fps",
        "nominal_duration_seconds",
        "container_fourcc",
        "issue",
    }
    if audit_version >= 2:
        required.add("decoded_content_sha256")
    _check_exact_keys(payload, required, context="dataset audit asset")
    shape_payload = payload["frame_shape"]
    frame_shape: tuple[int, int] | None
    if shape_payload is None:
        frame_shape = None
    elif (
        isinstance(shape_payload, (list, tuple))
        and len(shape_payload) == 2
    ):
        frame_shape = (
            _require_json_int("frame_shape[0]", shape_payload[0], minimum=1),
            _require_json_int("frame_shape[1]", shape_payload[1], minimum=1),
        )
    else:
        raise ValueError(
            "dataset audit frame_shape must have two integers or be null."
        )
    container_count = payload["container_frame_count"]
    if container_count is not None:
        container_count = _require_json_int(
            "container_frame_count",
            container_count,
            minimum=1,
        )
    fourcc = payload["container_fourcc"]
    if fourcc is not None and not isinstance(fourcc, str):
        raise ValueError("container_fourcc must be a string or null.")
    issue = payload["issue"]
    if issue is not None and not isinstance(issue, str):
        raise ValueError("video audit issue must be a string or null.")
    return VideoAssetAudit(
        relative_path=_require_json_string(
            "video relative_path",
            payload["relative_path"],
        ),
        sha256=_require_json_string("video sha256", payload["sha256"]),
        byte_count=_require_json_int(
            "video byte_count",
            payload["byte_count"],
            minimum=0,
        ),
        status=_require_json_string(  # type: ignore[arg-type]
            "video status",
            payload["status"],
        ),
        decoded_frame_count=_require_json_int(
            "decoded_frame_count",
            payload["decoded_frame_count"],
            minimum=0,
        ),
        container_frame_count=container_count,
        frame_shape=frame_shape,
        nominal_fps=_require_optional_json_number(
            "nominal_fps",
            payload["nominal_fps"],
        ),
        nominal_duration_seconds=_require_optional_json_number(
            "nominal_duration_seconds",
            payload["nominal_duration_seconds"],
        ),
        container_fourcc=fourcc,
        issue=issue,
        decoded_content_sha256=(
            _require_json_string(
                "decoded_content_sha256",
                payload["decoded_content_sha256"],
            )
            if audit_version >= 2
            else None
        ),
    )


def _video_audit_payload(
    asset: VideoAssetAudit,
    *,
    include_decoded_content: bool,
) -> dict[str, object]:
    payload = asdict(asset)
    if not include_decoded_content:
        payload.pop("decoded_content_sha256")
    return payload


def _check_exact_keys(
    payload: dict[str, Any],
    required: set[str],
    *,
    context: str,
) -> None:
    missing = required - set(payload)
    extra = set(payload) - required
    if missing:
        raise ValueError(f"{context} is missing keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"{context} has unknown keys: {sorted(extra)}")


def _require_json_string(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string.")
    return value


def _require_json_int(
    name: str,
    value: object,
    *,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def _require_optional_json_number(
    name: str,
    value: object,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number or null.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return result


def _audit_video(path: Path, *, root: Path) -> VideoAssetAudit:
    relative_path = path.relative_to(root).as_posix()
    digest = file_sha256(path)
    byte_count = path.stat().st_size
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return VideoAssetAudit(
            relative_path=relative_path,
            sha256=digest,
            byte_count=byte_count,
            status="unreadable",
            decoded_frame_count=0,
            container_frame_count=None,
            frame_shape=None,
            nominal_fps=None,
            nominal_duration_seconds=None,
            container_fourcc=None,
            issue="OpenCV could not open the video.",
            decoded_content_sha256=decoded_frame_content_sha256(()),
        )

    reported_count_value = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    container_frame_count = (
        int(reported_count_value)
        if math.isfinite(reported_count_value)
        and reported_count_value > 0.0
        else None
    )
    reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
    nominal_fps = (
        reported_fps
        if math.isfinite(reported_fps) and reported_fps > 0.0
        else None
    )
    fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
    container_fourcc = (
        "".join(
            chr((fourcc_value >> (8 * index)) & 0xFF)
            for index in range(4)
        ).rstrip("\x00")
        or None
    )
    decoded_count = 0
    frame_shape: tuple[int, int] | None = None
    inconsistent_shape = False
    decoded_digest = hashlib.sha256()
    decoded_digest.update(b"plume-focus-decoded-frame-stream-v1\0")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            shape = (
                int(decoded_frame.shape[0]),
                int(decoded_frame.shape[1]),
            )
            if frame_shape is None:
                frame_shape = shape
            elif shape != frame_shape:
                inconsistent_shape = True
            _update_decoded_frame_digest(decoded_digest, decoded_frame)
            decoded_count += 1
    finally:
        capture.release()

    if decoded_count == 0:
        status: AssetAuditStatus = "empty"
        issue = "Video opened but yielded no decoded frames."
    elif decoded_count < 3:
        status = "insufficient_frames"
        issue = (
            "Video yielded fewer than the three frames required by the "
            "recorded runtime."
        )
    elif inconsistent_shape:
        status = "inconsistent_shape"
        issue = "Decoded frame dimensions changed within the video."
    elif (
        container_frame_count is not None
        and decoded_count != container_frame_count
    ):
        status = "frame_count_mismatch"
        issue = (
            "Decoded frame count does not match the container-reported "
            "frame count."
        )
    else:
        status = "valid"
        issue = None
    duration = (
        decoded_count / nominal_fps
        if nominal_fps is not None
        else None
    )
    decoded_digest.update(decoded_count.to_bytes(8, byteorder="big"))
    return VideoAssetAudit(
        relative_path=relative_path,
        sha256=digest,
        byte_count=byte_count,
        status=status,
        decoded_frame_count=decoded_count,
        container_frame_count=container_frame_count,
        frame_shape=frame_shape,
        nominal_fps=nominal_fps,
        nominal_duration_seconds=duration,
        container_fourcc=container_fourcc,
        issue=issue,
        decoded_content_sha256=decoded_digest.hexdigest(),
    )
