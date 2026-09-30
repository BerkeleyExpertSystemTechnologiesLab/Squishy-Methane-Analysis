"""Filesystem workflow for validated recorded-sequence experiments."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

from plume_focus.experiment_contracts import VideoProofReport
from plume_focus.experiment_runner import ExperimentRequest, run_experiment
from plume_focus.recorded_sources import (
    load_recorded_sequence,
    require_recorded_operation,
)
from plume_focus.video_rendering import VideoRenderer


def run_recorded_sequence(
    manifest_path: Path,
    *,
    max_scans: int,
    report_output: Path,
    output_video: Path | None = None,
    render_scale: int = 1,
) -> VideoProofReport:
    """Validate assets, run causally, and write one evidence report."""

    sequence = load_recorded_sequence(
        Path(manifest_path),
        verify_hashes=True,
    )
    if output_video is not None:
        require_recorded_operation(sequence, "render_local_media")
    observer = (
        VideoRenderer(
            Path(output_video),
            render_scale=render_scale,
        )
        if output_video is not None
        else None
    )
    result = run_experiment(
        ExperimentRequest(
            recorded_sequence=sequence,
            scenario=sequence.manifest.scenario,
            max_scans=max_scans,
        ),
        observer=observer,
    )
    data_provenance = copy.deepcopy(result.report.data_provenance)
    evidence_contract = data_provenance.get("evidence_contract")
    if isinstance(evidence_contract, dict):
        attribution = evidence_contract.get("media_attribution")
        if isinstance(attribution, dict):
            attribution["rendered_derivative"] = output_video is not None
            attribution["rendering_changes"] = (
                [
                    "created a time-compressed annotated three-panel proof "
                    "video from causal experiment steps"
                ]
                if output_video is not None
                else []
            )
    report = replace(
        result.report,
        output_video=(
            str(output_video)
            if output_video is not None
            else ""
        ),
        data_provenance=data_provenance,
    )
    report_path = Path(report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        report.to_json() + "\n",
        encoding="utf-8",
    )
    return report
