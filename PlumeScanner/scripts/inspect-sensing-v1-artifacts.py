"""Verify frozen-v1 artifacts, fully decode videos, and extract event-time frames."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    manifest_path = root / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for item in manifest["artifacts"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root) or digest(path) != item["sha256"]:
            raise ValueError(f"Invalid artifact path or hash: {item['path']}")
    videos = []
    for video_manifest in sorted(root.rglob("video-manifest.json")):
        directory = video_manifest.parent
        trace = json.loads((directory / "trace.json").read_text())
        events = trace["events"]
        decisions = [e for e in events if e["operation"] == "plan_next" and e["plan"]["samples"]]
        deliveries = [e for e in events if e["operation"] == "ingest_readings"]
        receipts = [e for e in events if e["operation"] == "finish_scan"]
        physical_times = {0.}
        if decisions:
            sweep = next(s for s in decisions[0]["plan"]["segments"] if s["kind"] == "sweep")
            physical_times.add((sweep["started_at_seconds"] + sweep["completed_at_seconds"]) / 2)
            physical_times.add(decisions[0]["plan"]["completed_at_seconds"])
        if deliveries:
            physical_times.update((max(0., deliveries[0]["at_seconds"] - .1), deliveries[0]["at_seconds"] + .1))
        if receipts:
            physical_times.update((max(0., receipts[0]["at_seconds"] - .1), receipts[0]["at_seconds"] + .1))
        if len(decisions) > 1:
            physical_times.add(decisions[1]["plan"]["started_at_seconds"] + decisions[1]["plan"]["samples"][0]["integration_seconds"])
        output = root / "inspection-frames" / directory.relative_to(root)
        output.mkdir(parents=True, exist_ok=True)
        for record in json.loads(video_manifest.read_text()):
            path = directory / Path(record["output_path"]).name
            count_expected = record["encoded_frames"]
            if record["mode"] == "summary":
                targets = set(range(count_expected))
            else:
                targets = {max(0, min(count_expected - 1, round((t - record["physical_start_seconds"]) * record["fps"] / record["playback_speed"])))
                           for t in physical_times}
                targets.update((count_expected // 2, count_expected - 1))
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise ValueError(f"Cannot decode {path}")
            count, frames = 0, []
            fps = capture.get(cv2.CAP_PROP_FPS)
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if count in targets:
                        target = output / f"{path.stem}-{count:04d}.png"
                        if not cv2.imwrite(str(target), frame):
                            raise ValueError(f"Cannot save inspection frame {target}")
                        if record["mode"] == "summary":
                            physical_time = receipts[count]["at_seconds"]
                        else:
                            physical_time = min(record["physical_end_seconds"], record["physical_start_seconds"] + count * record["playback_speed"] / record["fps"])
                        frames.append({"frame": count, "physical_time_seconds": physical_time,
                                       "path": str(target.relative_to(root)).replace("\\", "/"), "sha256": digest(target)})
                    count += 1
            finally:
                capture.release()
            if count != count_expected or abs(fps - record["fps"]) > 1e-6:
                raise ValueError(f"Video count/FPS mismatch: {path}")
            videos.append({"path": str(path.relative_to(root)).replace("\\", "/"), "decoded_frames": count,
                           "fps": fps, "mode": record["mode"], "operational_hash": record["operational_hash"],
                           "sha256": digest(path), "inspection_frames": frames})
            print(f"Decoded {count} frames: {path.relative_to(root)}", flush=True)
    result = {"data_producing_revision": manifest.get("data_producing_revision", manifest["producing_revision"]),
              "renderer_revision": manifest.get("renderer_revision", manifest["producing_revision"]),
              "artifact_manifest_sha256": digest(manifest_path),
              "validated_artifact_count": len(manifest["artifacts"]), "videos": videos,
              "review_scope": "Full decode and event-time frame extraction; human/model visual review recorded separately."}
    (root / "inspection-manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
