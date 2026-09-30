"""Cross-suite asset-role isolation for recorded evidence.

The checks operate on validated sequence evidence rather than raw manifests so
all referenced hashes and calibration records have already been resolved.
Legacy schema-v2 calibration captures lack a capture-group identifier; their
byte-role isolation is still enforced, while group-level completeness remains
explicitly false.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationUse:
    source_sha256: str
    calibration_scope: str
    capture_group_id: str | None

    def __post_init__(self) -> None:
        _require_sha256(
            "calibration source sha256",
            self.source_sha256,
        )
        if self.calibration_scope not in {
            "development_only",
            "independent_pre_run",
        }:
            raise ValueError("unknown calibration scope in suite evidence.")
        if (
            self.capture_group_id is not None
            and not self.capture_group_id.strip()
        ):
            raise ValueError(
                "calibration capture_group_id must not be empty."
            )


@dataclass(frozen=True)
class SuiteAssetGraph:
    media_sha256s: frozenset[str]
    media_capture_group_ids: frozenset[str]
    calibration_uses: frozenset[CalibrationUse]
    truth_sha256s: frozenset[str]

    @property
    def calibration_source_sha256s(self) -> frozenset[str]:
        return frozenset(
            item.source_sha256.lower()
            for item in self.calibration_uses
        )

    @property
    def calibration_capture_group_ids(self) -> frozenset[str]:
        return frozenset(
            item.capture_group_id
            for item in self.calibration_uses
            if item.capture_group_id is not None
        )

    @property
    def calibration_groups_complete(self) -> bool:
        return all(
            item.capture_group_id is not None
            for item in self.calibration_uses
        )


def extract_suite_asset_graph(
    validation: dict[str, object],
) -> SuiteAssetGraph:
    """Extract exact evidence roles from one validated suite payload."""

    sequences = validation.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise ValueError("suite validation has no sequence records.")

    media_hashes: set[str] = set()
    media_groups: set[str] = set()
    calibration_uses: set[CalibrationUse] = set()
    truth_hashes: set[str] = set()
    for sequence in sequences:
        if not isinstance(sequence, dict):
            raise ValueError(
                "suite sequence validation must be an object."
            )
        evidence = sequence.get("evidence_contract")
        if not isinstance(evidence, dict):
            raise ValueError(
                "paired suite validation requires classified evidence."
            )
        lineage = evidence.get("dataset_lineage")
        capture = evidence.get("capture_group")
        if not isinstance(lineage, dict) or not isinstance(capture, dict):
            raise ValueError(
                "paired suite validation requires lineage and capture "
                "groups."
            )
        media_hash = _mapping_sha256(
            lineage,
            "source_asset_sha256",
            context="source media",
        )
        group_id = capture.get("capture_group_id")
        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError(
                "paired suite validation requires capture_group_id."
            )
        media_hashes.add(media_hash)
        media_groups.add(group_id)

        truth_hash = sequence.get("truth_sha256")
        if truth_hash is not None:
            if not isinstance(truth_hash, str):
                raise ValueError("truth_sha256 must be a string or null.")
            _require_sha256("truth sha256", truth_hash)
            truth_hashes.add(truth_hash.lower())

        sensor = evidence.get("sensor_input")
        if not isinstance(sensor, dict):
            raise ValueError(
                "paired suite validation requires sensor evidence."
            )
        calibration = sensor.get("frozen_calibration")
        frozen = True
        if calibration is None:
            calibration = sensor.get("leak_off_calibration")
            frozen = False
        if calibration is None:
            continue
        if not isinstance(calibration, dict):
            raise ValueError(
                "calibration evidence must be an object."
            )
        calibration_uses.add(
            CalibrationUse(
                source_sha256=_mapping_sha256(
                    calibration,
                    "source_sha256" if frozen else "sha256",
                    context="calibration source",
                ),
                calibration_scope=_mapping_text(
                    calibration,
                    "calibration_scope",
                    context="calibration source",
                ),
                capture_group_id=(
                    _optional_mapping_text(
                        calibration,
                        "calibration_capture_group_id",
                    )
                    if frozen
                    else None
                ),
            )
        )

    return SuiteAssetGraph(
        media_sha256s=frozenset(media_hashes),
        media_capture_group_ids=frozenset(media_groups),
        calibration_uses=frozenset(calibration_uses),
        truth_sha256s=frozenset(truth_hashes),
    )


def validate_cross_split_asset_roles(
    development: dict[str, object],
    heldout: dict[str, object],
) -> dict[str, object]:
    """Reject byte or capture identities assigned incompatible roles."""

    dev = extract_suite_asset_graph(development)
    held = extract_suite_asset_graph(heldout)

    _reject_overlap(
        dev.media_capture_group_ids,
        held.media_capture_group_ids,
        "capture groups overlap across development and held-out suites",
    )
    _reject_overlap(
        dev.media_sha256s,
        held.media_sha256s,
        "source media assets overlap across development and held-out suites",
    )

    all_media = dev.media_sha256s | held.media_sha256s
    all_calibration = (
        dev.calibration_source_sha256s
        | held.calibration_source_sha256s
    )
    _reject_overlap(
        all_media,
        all_calibration,
        "calibration sources overlap recorded sequence media",
    )
    _reject_overlap(
        all_media,
        dev.truth_sha256s | held.truth_sha256s,
        "evaluation truth overlaps recorded sequence media",
    )
    _reject_overlap(
        all_calibration,
        dev.truth_sha256s | held.truth_sha256s,
        "evaluation truth overlaps calibration source media",
    )
    all_media_groups = (
        dev.media_capture_group_ids
        | held.media_capture_group_ids
    )
    all_calibration_groups = (
        dev.calibration_capture_group_ids
        | held.calibration_capture_group_ids
    )
    _reject_overlap(
        all_media_groups,
        all_calibration_groups,
        "calibration capture groups overlap recorded sequence groups",
    )
    _validate_consistent_calibration_identity(
        dev.calibration_uses | held.calibration_uses
    )

    return {
        "capture_group_isolation_verified": True,
        "source_asset_isolation_verified": True,
        "asset_role_isolation_verified": True,
        "calibration_capture_group_isolation_verified": (
            dev.calibration_groups_complete
            and held.calibration_groups_complete
        ),
        "development_capture_groups": sorted(
            dev.media_capture_group_ids
        ),
        "development_source_asset_sha256s": sorted(
            dev.media_sha256s
        ),
        "development_calibration_source_sha256s": sorted(
            dev.calibration_source_sha256s
        ),
        "development_calibration_capture_groups": sorted(
            dev.calibration_capture_group_ids
        ),
        "heldout_capture_groups": sorted(
            held.media_capture_group_ids
        ),
        "heldout_source_asset_sha256s": sorted(
            held.media_sha256s
        ),
        "heldout_calibration_source_sha256s": sorted(
            held.calibration_source_sha256s
        ),
        "heldout_calibration_capture_groups": sorted(
            held.calibration_capture_group_ids
        ),
    }


def _validate_consistent_calibration_identity(
    uses: frozenset[CalibrationUse],
) -> None:
    by_hash: dict[str, list[CalibrationUse]] = {}
    for item in uses:
        by_hash.setdefault(item.source_sha256.lower(), []).append(item)
    for source_hash, items in by_hash.items():
        scopes = {item.calibration_scope for item in items}
        groups = {
            item.capture_group_id
            for item in items
            if item.capture_group_id is not None
        }
        if len(scopes) > 1 or len(groups) > 1:
            raise ValueError(
                "calibration source has conflicting scope or capture-group "
                f"roles: {source_hash}"
            )


def _reject_overlap(
    left: frozenset[str],
    right: frozenset[str],
    message: str,
) -> None:
    overlap = sorted(left & right)
    if overlap:
        raise ValueError(f"{message}: {overlap}")


def _mapping_sha256(
    payload: dict[str, object],
    key: str,
    *,
    context: str,
) -> str:
    value = _mapping_text(payload, key, context=context)
    _require_sha256(f"{context} {key}", value)
    return value.lower()


def _mapping_text(
    payload: dict[str, object],
    key: str,
    *,
    context: str,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} requires {key}.")
    return value


def _optional_mapping_text(
    payload: dict[str, object],
    key: str,
) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string or null.")
    return value


def _require_sha256(name: str, value: str) -> None:
    if len(value) != 64:
        raise ValueError(
            f"{name} must contain 64 hexadecimal characters."
        )
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(
            f"{name} must contain 64 hexadecimal characters."
        ) from error
