"""Add labelled inspection views, verifying unchanged execution against saved traces."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from plume_focus.sensing_cli_v1 import artifact_manifest, load_config, write_json
from plume_focus.sensing_execution_v1 import run_sensing_v1
from plume_focus.sensing_rendering import RenderSettings, render_sensing_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    original = json.loads((root / "artifact-manifest.json").read_text())
    for artifact in original["artifacts"]:
        path = (root / artifact["path"]).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
            raise ValueError(f"Existing evidence hash mismatch: {artifact['path']}")
    archive = root / "original-data-artifact-manifest.json"
    if not archive.exists():
        archive.write_text((root / "artifact-manifest.json").read_text(), encoding="utf-8")
    data_revision = original.get("data_producing_revision", original["producing_revision"])
    renderer_revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    checked = []
    for name in ("coarse-known-wind-mass90_column_v1-seed-0", "narrow-known-wind-mass90_column_v1-seed-0", "delayed-inspection"):
        directory = root / name
        config = load_config(directory / "configuration.json")
        report = json.loads((directory / "report.json").read_text())
        run = run_sensing_v1(config)
        if run.operational_hash != report["operational_hash"]:
            raise ValueError(f"Execution changed while preparing labelled views: {name}")
        old_records = [r for r in json.loads((directory / "video-manifest.json").read_text()) if "-profile.mp4" not in r["output_path"]]
        for record in old_records:
            record.setdefault("renderer_revision", data_revision)
            record.setdefault("variant", "before_profile_labels")
        new_records = []
        for mode in ("physical", "summary"):
            start = time.perf_counter()
            record = render_sensing_video(run, directory / f"{mode}-profile.mp4", source=config.source.build(),
                                           settings=RenderSettings(mode=mode, fps=12., playback_speed=1.))
            record.update(renderer_revision=renderer_revision, variant="profile_units_and_idle_labels",
                          encoding_compute_seconds=time.perf_counter() - start)
            new_records.append(record)
            print(f"{name} / {mode}: trace unchanged; encoded {record['encoded_frames']} frames in {record['encoding_compute_seconds']:.1f}s", flush=True)
        write_json(directory / "video-manifest.json", old_records + new_records)
        checked.append({"run": name, "operational_hash": run.operational_hash})
    artifact_manifest(root)
    final = json.loads((root / "artifact-manifest.json").read_text())
    final.update(data_producing_revision=data_revision, renderer_revision=renderer_revision,
                 original_data_manifest="original-data-artifact-manifest.json", trace_equivalence_verified=checked)
    write_json(root / "artifact-manifest.json", final)


if __name__ == "__main__":
    main()
