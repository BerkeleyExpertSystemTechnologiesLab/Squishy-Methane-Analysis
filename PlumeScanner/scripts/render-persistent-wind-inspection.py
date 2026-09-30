"""Render and decode paired saved executions without changing the panel layout."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
from plume_focus.sensing_cli_v1 import artifact_manifest, load_config, write_json
from plume_focus.sensing_execution_v1 import run_sensing_v1
from plume_focus.sensing_rendering import RenderSettings, render_sensing_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--case", action="append", help="Inspect completed named runs while the remaining study executes.")
    args = parser.parse_args()
    root = args.directory.resolve()
    if not args.case:
        if not (root / "study-summary.json").exists():
            raise ValueError("Complete the study or select a completed run with --case.")
        original = json.loads((root / "artifact-manifest.json").read_text())
        for record in original["artifacts"]:
            path = (root / record["path"]).resolve()
            if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"Evidence hash changed: {record['path']}")
        archive = root / "original-data-artifact-manifest.json"
        if not archive.exists():
            archive.write_bytes((root / "artifact-manifest.json").read_bytes())
    cases = [f"{profile}-wind-error-threshold90_segment_v2-seed-0-{estimator}"
             for profile in ("coarse-full-startup", "double-prior-startup") for estimator in ("baseline", "motion")]
    cases.append("diagnostic-source-free-biased-scale-0-seed-0-motion")
    if args.case:
        if any(name not in cases for name in args.case):
            raise ValueError("Case is outside the declared video inspection set.")
        cases = args.case
    manifest_path = root / "inspection-manifest.json"
    records = json.loads(manifest_path.read_text())["videos"] if manifest_path.exists() else []
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    frame_root = root / "inspection-frames"; frame_root.mkdir(exist_ok=True)
    for name in cases:
        directory = root / "development" / name
        inputs = {filename: hashlib.sha256((directory / filename).read_bytes()).hexdigest()
                  for filename in ("configuration.json", "report.json", "trace.json")}
        config = load_config(directory / "configuration.json")
        report = json.loads((directory / "report.json").read_text())
        existing = {r["mode"]:r for r in records if r["name"] == name}
        for mode, record in existing.items():
            if (record["input_sha256"] != inputs or record["operational_hash"] != report["operational_hash"]
                    or hashlib.sha256((directory / f"{mode}.mp4").read_bytes()).hexdigest() != record["sha256"]):
                raise ValueError(f"Previously inspected evidence changed: {name}/{mode}")
        if set(existing) == {"physical", "summary"}:
            print("Verified existing inspection", name, flush=True)
            continue
        print("Reproducing", name, flush=True)
        run = run_sensing_v1(config)
        if run.operational_hash != report["operational_hash"]:
            raise ValueError(f"Inspection cannot reproduce saved execution: {name}")
        for mode in ("physical", "summary"):
            if mode in existing:
                continue
            path = directory / f"{mode}.mp4"
            if path.exists():
                raise ValueError("Preserve existing media; use an empty inspection destination.")
            rendered = render_sensing_video(run, path, source=config.source.build(),
                                           settings=RenderSettings(mode=mode, fps=12., playback_speed=1.))
            first_completion = run.scans[0].execution.receipt.received_at_seconds
            if mode == "physical":
                chosen = sorted({0, int(round(first_completion*12)), int(round(first_completion*6)),
                                 int(45*12), rendered["encoded_frames"]-1})
            else:
                chosen = sorted({0, rendered["encoded_frames"]//2, rendered["encoded_frames"]-1})
            capture=cv2.VideoCapture(str(path)); decoded=0; frames=[]
            while True:
                ok, frame=capture.read()
                if not ok:
                    break
                if decoded in chosen:
                    target=frame_root/f"{name}-{mode}-{decoded:04d}.png"
                    cv2.imwrite(str(target),frame)
                    frames.append({"frame_index":decoded,"path":str(target.relative_to(root)).replace("\\","/")})
                decoded+=1
            capture.release()
            if decoded != rendered["encoded_frames"]:
                raise ValueError(f"Incomplete video decode: {path}")
            records.append({"name":name,**rendered,"decoded_frames":decoded,"inspection_frames":frames,
                            "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"input_sha256":inputs,
                            "renderer_revision":revision,"data_revision":report.get("producing_revision")})
            if inputs != {filename: hashlib.sha256((directory / filename).read_bytes()).hexdigest() for filename in inputs}:
                raise ValueError(f"Input changed during inspection: {name}")
            write_json(manifest_path,{"layout":"Existing six panels; mixture uncertainty label only", "videos":records})
            print(name, mode, "decoded", decoded, flush=True)
    if not args.case:
        artifact_manifest(root)


if __name__ == "__main__":
    main()
