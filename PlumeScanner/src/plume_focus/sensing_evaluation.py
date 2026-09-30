"""Evaluator-only spatial truth and development reports for sensing traces."""

from __future__ import annotations

from dataclasses import asdict
import numpy as np

from plume_focus.physical_scan import coverage_mask
from plume_focus.signal_belief import grid_positions
from plume_focus.sensing_simulation import SensorSource, SensingRun


def summarize_run(run: SensingRun, truth: SensorSource | None = None, *, evaluation_shape=(120, 160), background_threshold=0.) -> dict:
    positions = grid_positions(evaluation_shape)
    scans = []
    for trace in run.scans:
        executed, receipt = trace.execution.executed_plan, trace.execution.receipt
        swept = coverage_mask(executed, evaluation_shape, run.config.capabilities)
        sampled = coverage_mask(executed, evaluation_shape, run.config.capabilities, sampled_only=True)
        mass_recall = support_recall = iou = interval_recall = None
        if truth is not None and (truth.end_time_seconds is None or receipt.completed_at_seconds <= truth.end_time_seconds + 1e-10):
            values = np.maximum(truth.signal_at(np.full(len(positions), receipt.completed_at_seconds), positions) - background_threshold, 0).reshape(evaluation_shape)
            total = values.sum()
            if total > 1e-15:
                mass_recall = float(values[swept].sum() / total)
                support = values >= .08 * values.max()
                support_recall = float(np.count_nonzero(support & swept) / np.count_nonzero(support))
                iou = float(np.count_nonzero(support & swept) / np.count_nonzero(support | swept))
            intervals = []
            for t in np.linspace(executed.started_at_seconds, receipt.completed_at_seconds, 5):
                v = np.maximum(truth.signal_at(np.full(len(positions), t), positions) - background_threshold, 0).reshape(evaluation_shape)
                if v.sum() > 1e-15:
                    intervals.append(float(v[swept].sum() / v.sum()))
            interval_recall = float(np.mean(intervals)) if intervals else None
        scans.append({"scan_id": trace.plan.scan_id, "start_seconds": executed.started_at_seconds,
                      "planned_completion_seconds": trace.plan.completed_at_seconds,
                      "actual_completion_seconds": receipt.completed_at_seconds,
                      "receipt_available_seconds": receipt.received_at_seconds,
                      "duration_seconds": executed.duration_seconds,
                      "requested_fraction": len(trace.plan.requested_cells) / np.prod(trace.plan.lattice_shape),
                      "swept_fraction": float(swept.mean()), "sampled_fraction": float(sampled.mean()),
                      "reading_count": len(trace.execution.readings), "planning_status": trace.plan.status,
                      "planning_converged": trace.plan.converged, "planning_reason": trace.plan.reason,
                      "predicted_mass_recall": trace.plan.predicted_mass_recall,
                      "actual_sweep_mass_recall": mass_recall, "actual_sweep_support_recall": support_recall,
                      "actual_sweep_iou": iou, "interval_mass_recall": interval_recall,
                      "uncertainty_radius_fov": trace.wind.error_speed * trace.plan.duration_seconds,
                      "mean_signal_std": float(trace.posterior.std.mean())})

    def mean(key):
        values = [s[key] for s in scans if s[key] is not None]
        return float(np.mean(values)) if values else None

    tracking = scans[1:] if run.config.start_full_scene else scans
    tracking_values = [s["actual_sweep_mass_recall"] for s in tracking if s["actual_sweep_mass_recall"] is not None]

    return {"schema_version": 1, "report_type": "sensing_recovery", "profile": "recovery_scalar_v1",
            "claim_status": "not_demonstrated", "evidence_role": "development",
            "source_assumption": run.source_assumption, "wind_source": "provided_synthetic_proxy",
            "declared_source_position": run.declared_source_position,
            "truth_available": truth is not None, "evaluated_predictions": sum(s["actual_sweep_mass_recall"] is not None for s in scans),
            "operational_hash": run.operational_hash, "configuration": asdict(run.config),
            "termination_reason": run.termination_reason, "scans_completed": len(scans),
            "total_scan_seconds": float(sum(s["duration_seconds"] for s in scans)),
            "total_elapsed_seconds": scans[-1]["receipt_available_seconds"] if scans else 0,
            "initial_scan_seconds": scans[0]["duration_seconds"] if scans else None,
            "mean_scan_seconds": mean("duration_seconds"),
            "maximum_scan_seconds": max((s["duration_seconds"] for s in scans), default=None),
            "final_scan_seconds": scans[-1]["duration_seconds"] if scans else None,
            "mean_requested_fraction": mean("requested_fraction"),
            "mean_swept_fraction": mean("swept_fraction"), "mean_sampled_fraction": mean("sampled_fraction"),
            "final_swept_fraction": scans[-1]["swept_fraction"] if scans else None,
            "mean_mass_recall": mean("actual_sweep_mass_recall"), "mean_interval_mass_recall": mean("interval_mass_recall"),
            "mean_tracking_mass_recall": float(np.mean(tracking_values)) if tracking_values else None,
            "mean_support_recall": mean("actual_sweep_support_recall"), "mean_iou": mean("actual_sweep_iou"),
            "scan_schedule": scans,
            "limitations": ["Uncalibrated synthetic scalar signal and mechanical capabilities.",
                            "Common time-budget ceilings do not imply equal actual expenditure.",
                            "Coarse Gaussian signal model; midpoint likelihood approximates the integrated exposure.",
                            "90% predicted signal mass is not whole-plume containment probability."]}
