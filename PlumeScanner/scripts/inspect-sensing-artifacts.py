"""Verify artifact hashes, fully decode videos, and extract inspection frames."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    artifacts = json.loads((root / "artifact-manifest.json").read_text())
    for artifact in artifacts["artifacts"]:
        if sha256(root / artifact["path"]) != artifact["sha256"]:
            raise ValueError(f"Artifact hash mismatch: {artifact['path']}")
    video_manifest = root / "video-manifest.json"
    video_results = []
    if video_manifest.exists():
        frame_dir = root / "inspection-frames"
        frame_dir.mkdir(exist_ok=True)
        for record in json.loads(video_manifest.read_text()):
            path = root / Path(record["output_path"]).name
            expected = record["encoded_frames"]
            targets = {0, expected // 4, expected // 2, 3 * expected // 4, expected - 1}
            capture = cv2.VideoCapture(str(path))
            count, frames = 0, []
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if count in targets:
                        output = frame_dir / f"{path.stem}-{count:04d}.jpg"
                        if not cv2.imwrite(str(output), frame):
                            raise ValueError(f"Could not save {output}")
                        frames.append({"frame": count, "path": str(output.relative_to(root)), "sha256": sha256(output)})
                    count += 1
            finally:
                capture.release()
            if count != expected:
                raise ValueError(f"{path.name}: decoded {count}, expected {expected} frames")
            video_results.append({"path": path.name, "decoded_frames": count, "fps": record["fps"],
                                  "sha256": sha256(path), "inspection_frames": frames})
    plots = [{"path": p.name, "sha256": sha256(p)} for p in root.glob("coverage-time.*")]
    result = {"producing_revision": artifacts["producing_revision"],
              "source_working_tree_dirty_at_generation": artifacts["working_tree_dirty"],
              "artifact_manifest_sha256": sha256(root / "artifact-manifest.json"),
              "validated_artifact_count": len(artifacts["artifacts"]), "videos": video_results, "plots": plots}
    (root / "inspection-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
