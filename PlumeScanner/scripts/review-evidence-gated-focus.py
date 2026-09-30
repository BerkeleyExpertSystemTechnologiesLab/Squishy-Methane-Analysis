"""Re-evaluate frozen study records without rewriting study evidence or its freeze.

The output records both producing-code and reviewing-code provenance. This is an
engineering compatibility check and derived-report correction, not a new study.
"""
import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


def driver():
    path = Path(__file__).with_name("run-evidence-gated-focus.py")
    spec = importlib.util.spec_from_file_location("evidence_gate_review_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def review(root, output, *, workers=1, shard=None, shards=1):
    d = driver()
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("Review output must be separate from the frozen evidence directory.")
    if output.exists() and shard is None:
        raise ValueError("Review output already exists; choose a fresh directory.")
    declaration = d.load_declaration(root)
    inventory = d.read(root / "hash-inventory.json")
    frozen = d.read(root / "freeze.json")
    hashes = {row["name"]: row["sha256"] for row in inventory["runs"]}
    source_hashes = {name: d.sha(root / name) for name in
                     ("declaration.json", "freeze.json", "report.json", "hash-inventory.json", "heldout-manifest.json")}
    output.mkdir(parents=True, exist_ok=shard is not None)
    if workers > 1:
        children = [subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
            "--root", str(root), "--output", str(output), "--shard", str(index), "--shards", str(workers)])
            for index in range(workers)]
        codes = [child.wait() for child in children]
        if any(codes):
            raise RuntimeError(f"Review workers failed: {codes}")
    items, records = [], []
    for index, job in enumerate(declaration["development"] + declaration["heldout"]):
        if shard is not None and index % shards != shard:
            continue
        path = root / "runs" / f"{job['name']}.json"
        digest = d.sha(path)
        if digest != hashes[job["name"]]:
            raise ValueError(f"Frozen record hash changed: {job['name']}")
        if workers > 1:
            summary = d.read(output / "records" / f"{job['name']}.json")
            if summary["source_sha256"] != digest:
                raise ValueError("Worker summary differs from the frozen execution.")
            records.append(summary)
            items.append({"job": job, "record": summary["paired_input"]})
            continue
        record = d.read(path)
        config = d.configured(root, job)
        if record["configuration_hash"] != config.digest or d.decode_config(record["configuration"]) != config:
            raise ValueError(f"Configuration lineage differs: {job['name']}")
        if record["code_sha256"] != frozen["code_sha256"]:
            raise ValueError(f"Producing code differs from freeze: {job['name']}")
        if record["study_sha256"] != d.sha(d.STUDY) or record["declaration_sha256"] != source_hashes["declaration.json"]:
            raise ValueError(f"Study lineage differs: {job['name']}")
        if job["stage"] == "heldout" and record["freeze_file_sha256"] != source_hashes["freeze.json"]:
            raise ValueError(f"Held-out freeze lineage differs: {job['name']}")
        if not (root / "truth-cache" / f"{d.truth_key(config)}.npz").exists():
            raise ValueError("A saved truth cache is required; review never writes to the frozen root.")
        q = d.truth(root, config)
        replay = d.verify_replay(record, truth=q)
        if d.metrics(record, q, config, replay) != record["metrics"]:
            raise ValueError(f"Historical metrics changed: {job['name']}")
        changes = d.change_metrics(record, q, config)
        summary = {"name": job["name"], "source_sha256": digest, "configuration_hash": config.digest,
                   "producing_code_sha256": record["code_sha256"], "public_replay": "passed",
                   "causal_loss_replay": "passed", "metrics_unchanged": True,
                   "old_change_metrics": record["change_metrics"], "change_metrics": changes,
                   "paired_input": {"metrics": record["metrics"], "J_over_T": record["J_over_T"],
                                    "change_metrics": changes}}
        d.write(output / "records" / f"{job['name']}.json", summary)
        # Keep only the fields needed to pair and report; discard large event arrays.
        items.append({"job": job, "record": {"metrics": record["metrics"], "J_over_T": record["J_over_T"],
                                               "change_metrics": changes}})
        records.append(summary)
        if d.sha(path) != digest:
            raise ValueError("A frozen execution changed during review.")
        print(job["name"], "replay and loss passed", flush=True)
    for name, digest in source_hashes.items():
        if d.sha(root / name) != digest:
            raise ValueError(f"Frozen metadata changed during review: {name}")
    if shard is not None:
        return
    rows = d.paired_rows(items)
    result = {"artifact_type": "evidence_gated_focus_cleanup_review_v1", "review_revision": d.revision(),
              "review_code_sha256": d.code_hash(), "review_script_sha256": d.sha(Path(__file__)),
              "source_root": str(root), "source_metadata_sha256": source_hashes,
              "producing_code_sha256": frozen["code_sha256"], "records": records,
              "paired": rows, "criteria": d.criteria(d.study(), rows, items),
              "note": "Original metrics and causal loss replay unchanged; derived change diagnostics recomputed. "
                      "The predeclared full-restart eligibility is retained, with straddling opportunities explicit."}
    d.write(output / "report.json", result)
    print({"records": len(records), "criteria": {k: v["status"] for k, v in result["criteria"].items()}}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/evidence-gated-focus"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1 or args.shards < 1 or (args.shard is not None and not 0 <= args.shard < args.shards):
        parser.error("Worker counts must be positive and shard indices must fit the shard count.")
    review(args.root, args.output, workers=args.workers, shard=args.shard, shards=args.shards)
