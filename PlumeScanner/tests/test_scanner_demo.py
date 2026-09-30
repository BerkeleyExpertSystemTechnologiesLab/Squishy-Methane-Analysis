"""Exercise the documented single-run CLI, including no-overwrite behavior."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import pytest

from plume_focus.evidence_gated_execution import EvidenceGatedRunConfig
from plume_focus.full_fov_execution import verify_replay
from test_evidence_gate import gate_session
from test_reconstruction_risk import config


def test_demo_cli_writes_a_replayable_candidate_trace_and_preserves_existing_output(tmp_path, capsys):
    base = config()
    run = EvidenceGatedRunConfig(gate_session(), replace(base.source, signal_scale=0.), base.wind,
        duration_seconds=40., evaluation_shape=(8, 8), evaluation_step_seconds=.2)
    path = Path(__file__).resolve().parents[1] / "scripts/run-scanner-demo.py"
    spec = importlib.util.spec_from_file_location("scanner_demo", path)
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(run.to_dict()), encoding="utf-8")
    output = tmp_path / "result"
    args = ["--config", str(config_path), "--output", str(output)]
    demo.main(args)
    summary = json.loads(capsys.readouterr().out)
    target = output / "runs/demo.json"
    content = target.read_bytes()
    record = json.loads(content)
    assert summary["policy"] == run.policy and summary["public_replay"] == "passed"
    assert summary["scans"]["FULL"] > 0
    assert verify_replay(record).startup_complete
    assert record["configuration_hash"] == run.digest
    with pytest.raises(SystemExit) as error:
        demo.main(args)
    assert error.value.code == 2
    assert target.read_bytes() == content
