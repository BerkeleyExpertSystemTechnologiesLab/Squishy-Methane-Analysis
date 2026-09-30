"""Render representative v2 runs only after reproducing their recorded execution."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from plume_focus.sensing_cli_v1 import artifact_manifest, load_config, write_json
from plume_focus.sensing_execution_v1 import run_sensing_v1
from plume_focus.sensing_rendering import RenderSettings, render_sensing_video


def plot_focusing_sequence(root, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report = json.loads((root / name / "report.json").read_text())
    rows = [r for r in report["scan_schedule"] if r["planning_status"] != "startup_full_scene"]
    times = [r["start_seconds"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 3.6), layout="constrained")
    ax.plot(times, [100*r["requested_fraction"] for r in rows], "o-", label="Requested cells")
    ax.plot(times, [100*r["swept_fraction"] for r in rows], "s-", label="Swept footprint")
    for row in rows:
        if row["planning_status"] == "reacquisition":
            ax.axvspan(row["start_seconds"], row["actual_completion_seconds"], color="gray", alpha=.15,
                       label="Charged surveillance")
    shape = report["configuration"]["session"]["instrument"]["lattice_shape"]
    minimum = report["configuration"]["session"]["policy"]["minimum_samples"]
    ax.axhline(100*minimum/(shape[0]*shape[1]), color="gray", ls=":", label="Four-reading selection floor")
    ax.set(xlabel="Physical time at decision (s)", ylabel="Fraction of field (%)", ylim=(0, 32),
           title="The focused region expands and contracts\nCoarse known-wind seed 0; full startup ends at 33.575 s")
    ax.legend(fontsize=8, ncols=2)
    ax.grid(alpha=.2)
    for suffix in ("png", "pdf"):
        fig.savefig(root / f"focusing-sequence.{suffix}", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    original = json.loads((root / "artifact-manifest.json").read_text())
    for item in original["artifacts"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Existing evidence hash mismatch: {item['path']}")
    names = ("coarse-full-startup-known-wind-threshold90_segment_v2-seed-0",
             "double-prior-startup-known-wind-threshold90_segment_v2-seed-0",
             "diagnostic-blank-source-free")
    if any((root / name / "video-manifest.json").exists() for name in names):
        raise ValueError("Inspection videos already exist; preserve the original evidence.")
    archive = root / "original-data-artifact-manifest.json"
    if archive.exists():
        raise ValueError("Original data manifest already archived; inspect partial rendering before retrying.")
    archive.write_text((root / "artifact-manifest.json").read_text(), encoding="utf-8")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    checked = []
    for name in names:
        directory = root / name
        config = load_config(directory / "configuration.json")
        report = json.loads((directory / "report.json").read_text())
        run = run_sensing_v1(config)
        if run.operational_hash != report["operational_hash"]:
            raise ValueError(f"Execution changed while preparing inspection: {name}")
        records = []
        for mode in ("physical", "summary"):
            start = time.perf_counter()
            record = render_sensing_video(run, directory / f"{mode}.mp4", source=config.source.build(),
                                           settings=RenderSettings(mode=mode, fps=12., playback_speed=1.))
            record.update(renderer_revision=revision, encoding_compute_seconds=time.perf_counter() - start)
            records.append(record)
            print(f"{name} / {mode}: unchanged trace, {record['encoded_frames']} frames", flush=True)
        write_json(directory / "video-manifest.json", records)
        checked.append({"run": name, "operational_hash": run.operational_hash})
    plot_focusing_sequence(root, names[0])
    artifact_manifest(root)
    final = json.loads((root / "artifact-manifest.json").read_text())
    final.update(data_producing_revision=original["producing_revision"], renderer_revision=revision,
                 original_data_manifest=archive.name, trace_equivalence_verified=checked)
    write_json(root / "artifact-manifest.json", final)


if __name__ == "__main__":
    main()
