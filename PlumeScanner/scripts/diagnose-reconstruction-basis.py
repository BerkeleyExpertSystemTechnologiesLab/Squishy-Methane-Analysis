"""Evaluator-only spatial representation check at declared physical snapshots.

An oracle least-squares projection is a lower bound on snapshot map error, not
an executable controller, online score, or tuning target.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from plume_focus.sensing_cli_v1 import config_from_dict, write_json
from plume_focus.sensing_execution_v1 import CausalReplayV1, ControllerCallV1
from plume_focus.signal_belief import grid_positions, interpolation_stencil


def projection_operator(shape, points):
    indices, weights = interpolation_stencil(points, shape)
    size = int(np.prod(shape))
    gram = np.zeros((size, size))
    for i in range(4):
        for j in range(4):
            np.add.at(gram, (indices[:, i], indices[:, j]), weights[:, i] * weights[:, j] / len(points))
    inverse = np.linalg.pinv(gram, hermitian=True)

    def project(values):
        right = np.bincount(indices.ravel(), (weights * values[:, None]).ravel(), minlength=size) / len(points)
        coefficients = inverse @ right
        estimate = np.sum(coefficients[indices] * weights, axis=1)
        # Verify the normal equations independently of the inverse expression.
        residual = np.bincount(indices.ravel(), (weights * (estimate-values)[:, None]).ravel(), minlength=size) / len(points)
        if np.max(np.abs(residual)) > 1e-10:
            raise ValueError("Oracle projection did not satisfy the normal equations.")
        return float(np.mean((estimate-values)**2))
    return project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve the previous diagnostic; use a new output path.")
    payload = json.loads(args.trace.read_text())
    config = config_from_dict(payload["configuration"])
    times = (0., 5., 15., 30., 45., 60., 75.)
    if config.duration_seconds < max(times):
        raise ValueError("This snapshot diagnostic requires the declared 75-second fixture.")
    events = tuple(ControllerCallV1.from_dict(e) for e in payload["events"])
    replay = CausalReplayV1(config.session, events)
    shape = config.evaluation.shape
    points = grid_positions(shape)
    basis_shapes = (config.model.grid_shape, tuple(n*2 for n in config.model.grid_shape))
    operators = [projection_operator(basis, points) for basis in basis_shapes]
    source = config.source.build()
    rows = []
    for t in times:
        truth = source.signal_at(np.full(len(points), t), points)
        belief = replay.advance_to(t)
        mean, _ = belief.predictive_moments(t, shape)
        row = {"time_seconds": t, "online_map_mse": float(np.mean((mean.ravel()-truth)**2)),
               "zero_field_mse": float(np.mean(truth**2)),
               "oracle_projection_mse": {f"{s[0]}x{s[1]}": project(truth) for s, project in zip(basis_shapes, operators)}}
        rows.append(row)
        print(row, flush=True)
    write_json(args.output, {"artifact_type": "oracle_spatial_representation_diagnostic",
        "producing_revision": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
        "diagnostic_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(),
        "evaluation_shape": shape, "snapshots": rows,
        "limitations": ["Truth-fed, unconstrained spatial projection is evaluator-only and cannot be used by a causal scanner.",
                        "Seven snapshot errors are not a converged time-integrated loss comparison.",
                        "The doubled basis is a diagnostic; the study's inference grid and dynamics remain unchanged."]})


if __name__ == "__main__":
    main()
