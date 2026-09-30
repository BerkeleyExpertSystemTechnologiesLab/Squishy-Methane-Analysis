"""Strict JSON parsing and hashing for recorded-data manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from plume_focus.recorded_contracts import (
    CaptureGroupSpec,
    DatasetLineageSpec,
    DatasetQualificationLineageSpec,
    FrameClockSpec,
    FrozenCalibrationAssetSpec,
    GasIdentitySpec,
    LeakOffCalibrationAssetSpec,
    MediaSpec,
    MediaWindowSpec,
    NormalizationSpec,
    RecordedSequenceManifest,
    RecordedSuiteManifest,
    ScannerSpec,
    SensorInputSpec,
    SensorCalibrationSpec,
    SequenceReference,
    SourceAssumptionSpec,
    TruthSpec,
    WindSpec,
    SUPPORTED_SEQUENCE_MANIFEST_SCHEMA_VERSIONS,
)
from plume_focus.sensor_emulation import SensorEmulationProfile


def load_json_object(path: Path) -> dict[str, Any]:
    """Read one JSON object with a path-specific decode error."""

    try:
        payload = Path(path).read_text(encoding="utf-8")
        decoded = json.loads(payload)
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON: {path}") from error
    return _require_object(Path(path).name, decoded)


def load_recorded_suite_manifest(
    suite_path: Path,
) -> RecordedSuiteManifest:
    """Parse one frozen development or held-out suite manifest."""

    payload = load_json_object(Path(suite_path).resolve())
    _check_keys(
        payload,
        required={
            "suite_schema_version",
            "suite_id",
            "split",
            "frozen_at_utc",
            "frozen_before_model_commit",
            "freeze_provenance",
            "sequences",
        },
        context="recorded suite",
    )
    sequence_payloads = _require_list(
        "sequences",
        payload["sequences"],
    )
    references: list[SequenceReference] = []
    for item in sequence_payloads:
        values = _require_object("sequence reference", item)
        _check_keys(
            values,
            required={"path", "manifest_sha256"},
            context="sequence reference",
        )
        references.append(
            SequenceReference(
                path=_require_string("path", values["path"]),
                manifest_sha256=_require_string(
                    "manifest_sha256",
                    values["manifest_sha256"],
                ),
            )
        )
    return RecordedSuiteManifest(
        suite_schema_version=_require_int(
            "suite_schema_version",
            payload["suite_schema_version"],
        ),
        suite_id=_require_string("suite_id", payload["suite_id"]),
        split=_require_string(  # type: ignore[arg-type]
            "split",
            payload["split"],
        ),
        frozen_at_utc=_require_string(
            "frozen_at_utc",
            payload["frozen_at_utc"],
        ),
        frozen_before_model_commit=_require_string(
            "frozen_before_model_commit",
            payload["frozen_before_model_commit"],
        ),
        freeze_provenance=_require_string(
            "freeze_provenance",
            payload["freeze_provenance"],
        ),
        sequences=tuple(references),
    )


def parse_recorded_sequence_manifest(
    payload: dict[str, Any],
) -> RecordedSequenceManifest:
    """Parse and strictly validate a supported sequence payload."""

    version = _require_int(
        "manifest_schema_version",
        payload.get("manifest_schema_version"),
    )
    if version not in SUPPORTED_SEQUENCE_MANIFEST_SCHEMA_VERSIONS:
        raise ValueError("unsupported sequence manifest schema version.")
    common_keys = {
        "manifest_schema_version",
        "sequence_id",
        "split",
        "split_provenance",
        "scenario",
        "media",
        "frame_clock",
        "normalization",
        "calibration",
        "wind",
        "source_assumption",
        "scanner",
        "evaluation_truth",
        "model_profile_id",
    }
    classified_keys = {
        "dataset_lineage",
        "capture_group",
        "sensor_input",
    }
    version_keys: set[str] = set()
    if version == 4 or (version == 5 and "media_window" in payload):
        version_keys.add("media_window")
    if version == 5:
        version_keys.update({"gas_identity", "dataset_qualification"})
    _check_keys(
        payload,
        required=(
            common_keys
            if version == 1
            else common_keys | classified_keys | version_keys
        ),
        context="recorded sequence",
    )
    return RecordedSequenceManifest(
        manifest_schema_version=version,
        sequence_id=_require_string(
            "sequence_id",
            payload["sequence_id"],
        ),
        split=_require_string(  # type: ignore[arg-type]
            "split",
            payload["split"],
        ),
        split_provenance=_require_string(
            "split_provenance",
            payload["split_provenance"],
        ),
        scenario=_require_string(  # type: ignore[arg-type]
            "scenario",
            payload["scenario"],
        ),
        media=_parse_media(
            _require_object("media", payload["media"])
        ),
        frame_clock=_parse_frame_clock(
            _require_object("frame_clock", payload["frame_clock"])
        ),
        normalization=_parse_normalization(
            _require_object(
                "normalization",
                payload["normalization"],
            )
        ),
        calibration=_parse_calibration(
            _require_object("calibration", payload["calibration"])
        ),
        wind=_parse_wind(
            _require_object("wind", payload["wind"])
        ),
        source_assumption=_parse_source_assumption(
            _require_object(
                "source_assumption",
                payload["source_assumption"],
            )
        ),
        scanner=_parse_scanner(
            _require_object("scanner", payload["scanner"]),
            schema_version=version,
        ),
        evaluation_truth=_parse_truth(
            _require_object(
                "evaluation_truth",
                payload["evaluation_truth"],
            ),
            schema_version=version,
        ),
        model_profile_id=_require_string(
            "model_profile_id",
            payload["model_profile_id"],
        ),
        gas_identity=(
            _parse_gas_identity(
                _require_object(
                    "gas_identity",
                    payload["gas_identity"],
                )
            )
            if version == 5
            else None
        ),
        dataset_qualification=(
            _parse_dataset_qualification_lineage(
                _require_object(
                    "dataset_qualification",
                    payload["dataset_qualification"],
                )
            )
            if version == 5
            else None
        ),
        dataset_lineage=(
            _parse_dataset_lineage(
                _require_object(
                    "dataset_lineage",
                    payload["dataset_lineage"],
                )
            )
            if version >= 2
            else None
        ),
        capture_group=(
            _parse_capture_group(
                _require_object(
                    "capture_group",
                    payload["capture_group"],
                )
            )
            if version >= 2
            else None
        ),
        sensor_input=(
            _parse_sensor_input(
                _require_object(
                    "sensor_input",
                    payload["sensor_input"],
                ),
                schema_version=version,
            )
            if version >= 2
            else None
        ),
        media_window=(
            _parse_media_window(
                _require_object(
                    "media_window",
                    payload["media_window"],
                )
            )
            if version == 4 or (version == 5 and "media_window" in payload)
            else None
        ),
    )


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    """Hash canonical JSON independently of whitespace and key ordering."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash one asset without loading it fully into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _parse_media(payload: dict[str, Any]) -> MediaSpec:
    _check_keys(
        payload,
        required={"path", "sha256", "capture_provenance"},
        context="media",
    )
    return MediaSpec(
        path=_require_string("media path", payload["path"]),
        sha256=_require_string("media sha256", payload["sha256"]),
        capture_provenance=_require_string(
            "media capture_provenance",
            payload["capture_provenance"],
        ),
    )


def _parse_gas_identity(payload: dict[str, Any]) -> GasIdentitySpec:
    _check_keys(
        payload,
        required={"identity", "provenance"},
        context="gas identity",
    )
    return GasIdentitySpec(
        identity=_require_string(  # type: ignore[arg-type]
            "gas identity",
            payload["identity"],
        ),
        provenance=_require_string(
            "gas identity provenance",
            payload["provenance"],
        ),
    )


def _parse_dataset_qualification_lineage(
    payload: dict[str, Any],
) -> DatasetQualificationLineageSpec:
    _check_keys(
        payload,
        required={"path", "sha256", "purpose"},
        context="dataset qualification lineage",
    )
    return DatasetQualificationLineageSpec(
        path=_require_string(
            "dataset qualification path", payload["path"]
        ),
        sha256=_require_string(
            "dataset qualification sha256", payload["sha256"]
        ),
        purpose=_require_string(  # type: ignore[arg-type]
            "dataset qualification purpose", payload["purpose"]
        ),
    )


def _parse_media_window(payload: dict[str, Any]) -> MediaWindowSpec:
    _check_keys(
        payload,
        required={
            "requested_start_time_seconds",
            "requested_end_time_seconds",
            "source_start_frame_index",
            "source_end_frame_index_exclusive",
            "first_included_timestamp_seconds",
            "last_included_timestamp_seconds",
            "parent_media_sha256",
            "decoded_window_sha256",
            "selection_metadata_path",
            "selection_metadata_sha256",
            "selection_provenance",
        },
        context="media window",
    )
    return MediaWindowSpec(
        requested_start_time_seconds=_require_number(
            "requested_start_time_seconds",
            payload["requested_start_time_seconds"],
        ),
        requested_end_time_seconds=_require_number(
            "requested_end_time_seconds",
            payload["requested_end_time_seconds"],
        ),
        source_start_frame_index=_require_int(
            "source_start_frame_index",
            payload["source_start_frame_index"],
        ),
        source_end_frame_index_exclusive=_require_int(
            "source_end_frame_index_exclusive",
            payload["source_end_frame_index_exclusive"],
        ),
        first_included_timestamp_seconds=_require_number(
            "first_included_timestamp_seconds",
            payload["first_included_timestamp_seconds"],
        ),
        last_included_timestamp_seconds=_require_number(
            "last_included_timestamp_seconds",
            payload["last_included_timestamp_seconds"],
        ),
        parent_media_sha256=_require_string(
            "parent_media_sha256",
            payload["parent_media_sha256"],
        ),
        decoded_window_sha256=_require_string(
            "decoded_window_sha256",
            payload["decoded_window_sha256"],
        ),
        selection_metadata_path=_require_string(
            "selection_metadata_path",
            payload["selection_metadata_path"],
        ),
        selection_metadata_sha256=_require_string(
            "selection_metadata_sha256",
            payload["selection_metadata_sha256"],
        ),
        selection_provenance=_require_string(
            "selection_provenance",
            payload["selection_provenance"],
        ),
    )
def _parse_dataset_lineage(
    payload: dict[str, Any],
) -> DatasetLineageSpec:
    _check_keys(
        payload,
        required={
            "dataset_id",
            "accession_id",
            "audit_path",
            "audit_sha256",
            "source_asset_sha256",
        },
        context="dataset lineage",
    )
    return DatasetLineageSpec(
        dataset_id=_require_string(
            "dataset_id",
            payload["dataset_id"],
        ),
        accession_id=_require_string(
            "accession_id",
            payload["accession_id"],
        ),
        audit_path=_require_string(
            "dataset audit path",
            payload["audit_path"],
        ),
        audit_sha256=_require_string(
            "dataset audit sha256",
            payload["audit_sha256"],
        ),
        source_asset_sha256=_require_string(
            "dataset source asset sha256",
            payload["source_asset_sha256"],
        ),
    )


def _parse_capture_group(
    payload: dict[str, Any],
) -> CaptureGroupSpec:
    _check_keys(
        payload,
        required={"capture_group_id", "grouping_provenance"},
        context="capture group",
    )
    return CaptureGroupSpec(
        capture_group_id=_require_string(
            "capture_group_id",
            payload["capture_group_id"],
        ),
        grouping_provenance=_require_string(
            "capture group provenance",
            payload["grouping_provenance"],
        ),
    )


def _parse_sensor_input(
    payload: dict[str, Any],
    *,
    schema_version: int,
) -> SensorInputSpec:
    kind = _require_string("sensor input kind", payload.get("kind"))
    provenance = _require_string(
        "sensor input provenance",
        payload.get("provenance"),
    )
    if kind == "raw_recorded":
        _check_keys(
            payload,
            required={"kind", "provenance"},
            context="raw recorded sensor input",
        )
        return SensorInputSpec(
            kind="raw_recorded",
            provenance=provenance,
        )
    if kind != "structure_suppressed_emulation":
        raise ValueError("unknown sensor input kind.")
    calibration_key = (
        "leak_off_calibration"
        if schema_version == 2
        else "frozen_calibration"
    )
    _check_keys(
        payload,
        required={
            "kind",
            "provenance",
            "emulation_profile",
            calibration_key,
        },
        context="structure-suppressed sensor input",
    )
    profile_payload = _require_object(
        "emulation_profile",
        payload["emulation_profile"],
    )
    _check_keys(
        profile_payload,
        required={
            "profile_id",
            "residual_mode",
            "provenance",
            "noise_floor_raw_units",
            "output_units_per_noise_sigma",
            "smoothing_time_constant_seconds",
            "profile_schema_version",
        },
        context="sensor emulation profile",
    )
    calibration_payload = _require_object(
        calibration_key,
        payload[calibration_key],
    )
    smoothing: float | None = None
    smoothing_value = profile_payload["smoothing_time_constant_seconds"]
    if smoothing_value is not None:
        smoothing = _require_number(
            "smoothing_time_constant_seconds",
            smoothing_value,
        )
    leak_off_calibration: LeakOffCalibrationAssetSpec | None = None
    frozen_calibration: FrozenCalibrationAssetSpec | None = None
    if schema_version == 2:
        _check_keys(
            calibration_payload,
            required={
                "path",
                "sha256",
                "provenance",
                "calibration_scope",
            },
            context="leak-off calibration",
        )
        leak_off_calibration = LeakOffCalibrationAssetSpec(
            path=_require_string(
                "leak-off calibration path",
                calibration_payload["path"],
            ),
            sha256=_require_string(
                "leak-off calibration sha256",
                calibration_payload["sha256"],
            ),
            provenance=_require_string(
                "leak-off calibration provenance",
                calibration_payload["provenance"],
            ),
            calibration_scope=_require_string(  # type: ignore[arg-type]
                "leak-off calibration_scope",
                calibration_payload["calibration_scope"],
            ),
        )
    else:
        _check_keys(
            calibration_payload,
            required={
                "path",
                "sha256",
                "source_path",
                "source_sha256",
                "source_provenance",
                "calibration_scope",
                "producing_commit",
                "preparation_spec_path",
                "preparation_spec_sha256",
                "preparation_profile_path",
                "preparation_profile_sha256",
                "calibration_record_path",
                "calibration_record_sha256",
            },
            context="frozen calibration",
        )
        frozen_calibration = FrozenCalibrationAssetSpec(
            path=_require_string(
                "frozen calibration path",
                calibration_payload["path"],
            ),
            sha256=_require_string(
                "frozen calibration sha256",
                calibration_payload["sha256"],
            ),
            source_path=_require_string(
                "calibration source path",
                calibration_payload["source_path"],
            ),
            source_sha256=_require_string(
                "calibration source sha256",
                calibration_payload["source_sha256"],
            ),
            source_provenance=_require_string(
                "calibration source provenance",
                calibration_payload["source_provenance"],
            ),
            calibration_scope=_require_string(  # type: ignore[arg-type]
                "frozen calibration_scope",
                calibration_payload["calibration_scope"],
            ),
            producing_commit=_require_string(
                "calibration producing_commit",
                calibration_payload["producing_commit"],
            ),
            preparation_spec_path=_require_string(
                "preparation spec path",
                calibration_payload["preparation_spec_path"],
            ),
            preparation_spec_sha256=_require_string(
                "preparation spec sha256",
                calibration_payload["preparation_spec_sha256"],
            ),
            preparation_profile_path=_require_string(
                "preparation profile path",
                calibration_payload["preparation_profile_path"],
            ),
            preparation_profile_sha256=_require_string(
                "preparation profile sha256",
                calibration_payload["preparation_profile_sha256"],
            ),
            calibration_record_path=_require_string(
                "calibration record path",
                calibration_payload["calibration_record_path"],
            ),
            calibration_record_sha256=_require_string(
                "calibration record sha256",
                calibration_payload["calibration_record_sha256"],
            ),
        )
    return SensorInputSpec(
        kind="structure_suppressed_emulation",
        provenance=provenance,
        emulation_profile=SensorEmulationProfile(
            profile_id=_require_string(
                "emulation profile_id",
                profile_payload["profile_id"],
            ),
            residual_mode=_require_string(  # type: ignore[arg-type]
                "emulation residual_mode",
                profile_payload["residual_mode"],
            ),
            provenance=_require_string(
                "emulation profile provenance",
                profile_payload["provenance"],
            ),
            noise_floor_raw_units=_require_number(
                "noise_floor_raw_units",
                profile_payload["noise_floor_raw_units"],
            ),
            output_units_per_noise_sigma=_require_number(
                "output_units_per_noise_sigma",
                profile_payload["output_units_per_noise_sigma"],
            ),
            smoothing_time_constant_seconds=smoothing,
            profile_schema_version=_require_int(
                "profile_schema_version",
                profile_payload["profile_schema_version"],
            ),
        ),
        leak_off_calibration=leak_off_calibration,
        frozen_calibration=frozen_calibration,
    )


def _parse_frame_clock(payload: dict[str, Any]) -> FrameClockSpec:
    mode = _require_string("frame clock mode", payload.get("mode"))
    if mode == "constant_fps":
        _check_keys(
            payload,
            required={"mode", "fps", "provenance"},
            context="constant frame clock",
        )
        return FrameClockSpec(
            mode="constant_fps",
            provenance=_require_string(
                "frame clock provenance",
                payload["provenance"],
            ),
            fps=_require_number("frame clock fps", payload["fps"]),
        )
    if mode == "timestamp_csv":
        _check_keys(
            payload,
            required={"mode", "path", "sha256", "provenance"},
            context="timestamp frame clock",
        )
        return FrameClockSpec(
            mode="timestamp_csv",
            provenance=_require_string(
                "frame clock provenance",
                payload["provenance"],
            ),
            path=_require_string("timestamp path", payload["path"]),
            sha256=_require_string(
                "timestamp sha256",
                payload["sha256"],
            ),
        )
    raise ValueError("unknown frame clock mode.")


def _parse_normalization(
    payload: dict[str, Any],
) -> NormalizationSpec:
    mode = _require_string("normalization mode", payload.get("mode"))
    if mode == "uint8_identity":
        _check_keys(
            payload,
            required={"mode", "provenance"},
            context="identity normalization",
        )
        return NormalizationSpec(
            mode="uint8_identity",
            provenance=_require_string(
                "normalization provenance",
                payload["provenance"],
            ),
        )
    if mode == "fixed_linear":
        _check_keys(
            payload,
            required={
                "mode",
                "provenance",
                "raw_min",
                "raw_max",
            },
            context="fixed normalization",
        )
        return NormalizationSpec(
            mode="fixed_linear",
            provenance=_require_string(
                "normalization provenance",
                payload["provenance"],
            ),
            raw_min=_require_number("raw_min", payload["raw_min"]),
            raw_max=_require_number("raw_max", payload["raw_max"]),
        )
    raise ValueError("unknown normalization mode.")


def _parse_calibration(
    payload: dict[str, Any],
) -> SensorCalibrationSpec:
    _check_keys(
        payload,
        required={"sensor_id", "signal_units", "provenance"},
        context="sensor calibration",
    )
    return SensorCalibrationSpec(
        sensor_id=_require_string("sensor_id", payload["sensor_id"]),
        signal_units=_require_string(
            "signal_units",
            payload["signal_units"],
        ),
        provenance=_require_string(
            "calibration provenance",
            payload["provenance"],
        ),
    )


def _parse_wind(payload: dict[str, Any]) -> WindSpec:
    mode = _require_string("wind mode", payload.get("mode"))
    common = {"mode", "units", "provenance", "conversion_provenance"}
    kwargs = {
        "units": _require_string("wind units", payload.get("units")),
        "provenance": _require_string(
            "wind provenance",
            payload.get("provenance"),
        ),
        "conversion_provenance": _require_string(
            "wind conversion provenance",
            payload.get("conversion_provenance"),
        ),
    }
    if mode == "none":
        _check_keys(payload, required=common, context="missing wind")
        return WindSpec(mode="none", **kwargs)
    if mode == "constant":
        _check_keys(
            payload,
            required=common
            | {"dy_px_per_second", "dx_px_per_second"},
            context="constant wind",
        )
        return WindSpec(
            mode="constant",
            dy_px_per_second=_require_number(
                "wind dy_px_per_second",
                payload["dy_px_per_second"],
            ),
            dx_px_per_second=_require_number(
                "wind dx_px_per_second",
                payload["dx_px_per_second"],
            ),
            **kwargs,
        )
    if mode == "time_series_csv":
        _check_keys(
            payload,
            required=common | {"path", "sha256", "max_age_seconds"},
            context="time-series wind",
        )
        return WindSpec(
            mode="time_series_csv",
            path=_require_string("wind path", payload["path"]),
            sha256=_require_string("wind sha256", payload["sha256"]),
            max_age_seconds=_require_number(
                "wind max_age_seconds",
                payload["max_age_seconds"],
            ),
            **kwargs,
        )
    raise ValueError("unknown wind mode.")


def _parse_source_assumption(
    payload: dict[str, Any],
) -> SourceAssumptionSpec:
    mode = _require_string(
        "source assumption mode",
        payload.get("mode"),
    )
    provenance = _require_string(
        "source provenance",
        payload.get("provenance"),
    )
    if mode == "known_source":
        _check_keys(
            payload,
            required={
                "mode",
                "provenance",
                "source_point",
                "initial_roi",
            },
            context="known source",
        )
        return SourceAssumptionSpec(
            mode="known_source",
            provenance=provenance,
            source_point=_parse_point(payload["source_point"]),
            initial_roi=_parse_roi(payload["initial_roi"]),
        )
    if mode == "known_corridor":
        _check_keys(
            payload,
            required={"mode", "provenance", "initial_roi"},
            context="known corridor",
        )
        return SourceAssumptionSpec(
            mode="known_corridor",
            provenance=provenance,
            initial_roi=_parse_roi(payload["initial_roi"]),
        )
    if mode == "source_free":
        _check_keys(
            payload,
            required={"mode", "provenance"},
            context="source-free assumption",
        )
        return SourceAssumptionSpec(
            mode="source_free",
            provenance=provenance,
        )
    raise ValueError("unknown source assumption mode.")


def _parse_scanner(
    payload: dict[str, Any],
    *,
    schema_version: int,
) -> ScannerSpec:
    required = {
        "provenance",
        "scan_mode",
        "scan_rate_hz",
        "sample_stride_px",
        "vertical_pass_seconds",
        "horizontal_step_seconds",
        "column_stride_px",
        "column_mass_fraction",
        "row_mass_fraction",
        "minimum_sweep_columns",
        "minimum_sweep_rows",
        "maximum_sweep_columns",
    }
    if schema_version >= 2:
        required.add("evidence_kind")
    _check_keys(payload, required=required, context="scanner")
    maximum = payload["maximum_sweep_columns"]
    if maximum is not None:
        maximum = _require_int("maximum_sweep_columns", maximum)
    return ScannerSpec(
        provenance=_require_string(
            "scanner provenance",
            payload["provenance"],
        ),
        scan_mode=_require_string(  # type: ignore[arg-type]
            "scan_mode",
            payload["scan_mode"],
        ),
        scan_rate_hz=_require_number(
            "scan_rate_hz",
            payload["scan_rate_hz"],
        ),
        sample_stride_px=_require_int(
            "sample_stride_px",
            payload["sample_stride_px"],
        ),
        vertical_pass_seconds=_require_number(
            "vertical_pass_seconds",
            payload["vertical_pass_seconds"],
        ),
        horizontal_step_seconds=_require_number(
            "horizontal_step_seconds",
            payload["horizontal_step_seconds"],
        ),
        column_stride_px=_require_int(
            "column_stride_px",
            payload["column_stride_px"],
        ),
        column_mass_fraction=_require_number(
            "column_mass_fraction",
            payload["column_mass_fraction"],
        ),
        row_mass_fraction=_require_number(
            "row_mass_fraction",
            payload["row_mass_fraction"],
        ),
        minimum_sweep_columns=_require_int(
            "minimum_sweep_columns",
            payload["minimum_sweep_columns"],
        ),
        minimum_sweep_rows=_require_int(
            "minimum_sweep_rows",
            payload["minimum_sweep_rows"],
        ),
        maximum_sweep_columns=maximum,
        evidence_kind=(
            _require_string(  # type: ignore[arg-type]
                "scanner evidence_kind",
                payload["evidence_kind"],
            )
            if schema_version >= 2
            else "legacy_unclassified"
        ),
    )


def _parse_truth(
    payload: dict[str, Any],
    *,
    schema_version: int,
) -> TruthSpec:
    if (
        schema_version >= 3
        and payload.get("evidence_grade") == "none"
    ):
        _check_keys(
            payload,
            required={"evidence_grade", "annotation_provenance"},
            context="truth-free evaluation",
        )
        return TruthSpec(
            annotation_provenance=_require_string(
                "annotation_provenance",
                payload["annotation_provenance"],
            ),
            storage_format="none",
            evidence_grade="none",
        )
    required = {
        "path",
        "sha256",
        "truth_kind",
        "annotation_provenance",
    }
    if schema_version >= 2:
        required |= {
            "storage_format",
            "evidence_grade",
            "method_id",
            "protocol_path",
            "protocol_sha256",
        }
    _check_keys(
        payload,
        required=required,
        context="evaluation truth",
    )
    return TruthSpec(
        path=_require_string("truth path", payload["path"]),
        sha256=_require_string("truth sha256", payload["sha256"]),
        truth_kind=_require_string(  # type: ignore[arg-type]
            "truth_kind",
            payload["truth_kind"],
        ),
        annotation_provenance=_require_string(
            "annotation_provenance",
            payload["annotation_provenance"],
        ),
        storage_format=(
            _require_string(  # type: ignore[arg-type]
                "truth storage_format",
                payload["storage_format"],
            )
            if schema_version >= 2
            else "legacy_video"
        ),
        evidence_grade=(
            _require_string(  # type: ignore[arg-type]
                "truth evidence_grade",
                payload["evidence_grade"],
            )
            if schema_version >= 2
            else "legacy_unclassified"
        ),
        method_id=(
            _require_string("truth method_id", payload["method_id"])
            if schema_version >= 2
            else None
        ),
        protocol_path=(
            _require_string(
                "truth protocol_path",
                payload["protocol_path"],
            )
            if schema_version >= 2
            else None
        ),
        protocol_sha256=(
            _require_string(
                "truth protocol_sha256",
                payload["protocol_sha256"],
            )
            if schema_version >= 2
            else None
        ),
    )


def _check_keys(
    payload: dict[str, Any],
    *,
    required: set[str],
    context: str,
) -> None:
    keys = set(payload)
    missing = required - keys
    extra = keys - required
    if missing:
        raise ValueError(f"{context} is missing keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"{context} has unknown keys: {sorted(extra)}")


def _require_object(name: str, value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return value


def _require_list(name: str, value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array.")
    return value


def _require_string(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string.")
    return value


def _require_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number.")
    return float(value)


def _require_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    return value


def _parse_point(value: object) -> tuple[float, float]:
    values = _require_list("source_point", value)
    if len(values) != 2:
        raise ValueError("source_point must have two values.")
    return (
        _require_number("source row", values[0]),
        _require_number("source col", values[1]),
    )


def _parse_roi(value: object) -> tuple[int, int, int, int]:
    values = _require_list("initial_roi", value)
    if len(values) != 4:
        raise ValueError("initial_roi must have four values.")
    return tuple(
        _require_int(f"initial_roi[{index}]", item)
        for index, item in enumerate(values)
    )  # type: ignore[return-value]
