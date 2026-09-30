from __future__ import annotations

import pytest

from plume_focus.recorded_role_isolation import (
    validate_cross_split_asset_roles,
)


def _suite_validation(
    *,
    media_sha256: str,
    media_group: str,
    calibration_sha256: str | None = None,
    calibration_group: str | None = None,
    calibration_scope: str = "development_only",
    truth_sha256: str | None = None,
    legacy_calibration: bool = False,
) -> dict[str, object]:
    sensor: dict[str, object] = {
        "kind": (
            "structure_suppressed_emulation"
            if calibration_sha256 is not None
            else "raw_recorded"
        )
    }
    if calibration_sha256 is not None:
        if legacy_calibration:
            sensor["leak_off_calibration"] = {
                "sha256": calibration_sha256,
                "calibration_scope": calibration_scope,
            }
        else:
            sensor["frozen_calibration"] = {
                "source_sha256": calibration_sha256,
                "calibration_scope": calibration_scope,
                "calibration_capture_group_id": calibration_group,
            }
    return {
        "sequences": [
            {
                "truth_sha256": truth_sha256,
                "evidence_contract": {
                    "dataset_lineage": {
                        "source_asset_sha256": media_sha256,
                    },
                    "capture_group": {
                        "capture_group_id": media_group,
                    },
                    "sensor_input": sensor,
                },
            }
        ]
    }


def test_role_isolation_allows_one_frozen_development_calibration_across_splits(
) -> None:
    development = _suite_validation(
        media_sha256="1" * 64,
        media_group="development-media",
        calibration_sha256="3" * 64,
        calibration_group="development-calibration",
    )
    heldout = _suite_validation(
        media_sha256="2" * 64,
        media_group="heldout-media",
        calibration_sha256="3" * 64,
        calibration_group="development-calibration",
    )

    result = validate_cross_split_asset_roles(
        development,
        heldout,
    )

    assert result["asset_role_isolation_verified"] is True
    assert (
        result["calibration_capture_group_isolation_verified"]
        is True
    )
    assert result["development_calibration_source_sha256s"] == [
        "3" * 64
    ]
    assert result["heldout_calibration_capture_groups"] == [
        "development-calibration"
    ]


def test_role_isolation_rejects_sequence_media_reused_for_calibration(
) -> None:
    development = _suite_validation(
        media_sha256="1" * 64,
        media_group="development-media",
        calibration_sha256="2" * 64,
        calibration_group="calibration-group",
    )
    heldout = _suite_validation(
        media_sha256="2" * 64,
        media_group="heldout-media",
    )

    with pytest.raises(
        ValueError,
        match="calibration sources overlap recorded sequence media",
    ):
        validate_cross_split_asset_roles(development, heldout)


def test_role_isolation_rejects_calibration_capture_reused_as_sequence_group(
) -> None:
    development = _suite_validation(
        media_sha256="1" * 64,
        media_group="development-media",
        calibration_sha256="3" * 64,
        calibration_group="heldout-media",
    )
    heldout = _suite_validation(
        media_sha256="2" * 64,
        media_group="heldout-media",
    )

    with pytest.raises(
        ValueError,
        match="calibration capture groups overlap",
    ):
        validate_cross_split_asset_roles(development, heldout)


def test_role_isolation_rejects_conflicting_calibration_declarations(
) -> None:
    development = _suite_validation(
        media_sha256="1" * 64,
        media_group="development-media",
        calibration_sha256="3" * 64,
        calibration_group="calibration-group",
        calibration_scope="development_only",
    )
    heldout = _suite_validation(
        media_sha256="2" * 64,
        media_group="heldout-media",
        calibration_sha256="3" * 64,
        calibration_group="calibration-group",
        calibration_scope="independent_pre_run",
    )

    with pytest.raises(
        ValueError,
        match="conflicting scope or capture-group roles",
    ):
        validate_cross_split_asset_roles(development, heldout)


def test_role_isolation_marks_legacy_calibration_group_as_unverifiable(
) -> None:
    development = _suite_validation(
        media_sha256="1" * 64,
        media_group="development-media",
        calibration_sha256="3" * 64,
        legacy_calibration=True,
    )
    heldout = _suite_validation(
        media_sha256="2" * 64,
        media_group="heldout-media",
    )

    result = validate_cross_split_asset_roles(
        development,
        heldout,
    )

    assert (
        result["calibration_capture_group_isolation_verified"]
        is False
    )
    assert result["asset_role_isolation_verified"] is True


def test_role_isolation_does_not_treat_equal_label_bytes_as_capture_identity(
) -> None:
    development = _suite_validation(
        media_sha256="1" * 64,
        media_group="development-media",
        truth_sha256="4" * 64,
    )
    heldout = _suite_validation(
        media_sha256="2" * 64,
        media_group="heldout-media",
        truth_sha256="4" * 64,
    )

    result = validate_cross_split_asset_roles(development, heldout)

    assert result["asset_role_isolation_verified"] is True
