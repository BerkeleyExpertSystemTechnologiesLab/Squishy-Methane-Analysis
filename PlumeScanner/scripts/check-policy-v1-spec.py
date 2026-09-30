"""Independent arithmetic examples for the frozen specification, not a planner.

Uses NumPy and the standard library; does not import or certify production code.
Run from any directory. Output is deterministic; optional JSON stays local.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NORMAL = NormalDist()


def condition(mu, cov, h, value, variance):
    cross = cov @ h
    scale = h @ cross + variance
    return mu + cross * (value - h @ mu) / scale, cov - np.outer(cross, cross) / scale


def positive(mean, std):
    if std == 0:
        return max(mean, 0.0)
    return mean * NORMAL.cdf(mean / std) + std * NORMAL.pdf(mean / std)


def select(mass, minimum=1):
    mass = np.asarray(mass, dtype=float)
    order = np.argsort(-mass, kind="stable")
    count = max(minimum, int(np.searchsorted(np.cumsum(mass[order] / mass.sum()), .90 - 1e-10)) + 1)
    return tuple(sorted(int(i) for i in order[:count]))


def profile_semantics(profile):
    """Only cross-field arithmetic. JSON Schema validation is a separate command."""
    a, m = profile["acquisition"], profile["mechanics"]
    required = m["sweep_vertical_fov_per_second"] * profile["lattice_shape"][0]
    return (a["cadence_hz"] <= a["detector_capacity_hz"]
            and 0 < a["integration_seconds"] <= 1 / a["cadence_hz"]
            and (profile["scan_mode"] == "point"
                 or math.isclose(a["cadence_hz"], required, rel_tol=1e-10, abs_tol=0)))


def column_arithmetic(cells, profile, pose=(0.0, .125)):
    """Small deterministic closure/travel calculation; no forecast or live API."""
    rows, cols = profile["lattice_shape"]
    m, a = profile["mechanics"], profile["acquisition"]
    active = sorted({i % cols for i in cells})
    if active and abs(pose[1] - (active[-1] + .5) / cols) < abs(pose[1] - (active[0] + .5) / cols):
        active.reverse()
    travel = acquisition = settling = 0.0
    swept, samples = [], []
    for col in active:
        selected_rows = [i // cols for i in cells if i % cols == col]
        lo, hi = min(selected_rows), max(selected_rows)
        ylo, yhi, x = lo / rows, (hi + 1) / rows, (col + .5) / cols
        down = abs(pose[0] - ylo) <= abs(pose[0] - yhi)
        start, end = (ylo, yhi) if down else (yhi, ylo)
        travel += abs(pose[1] - x) / m["move_horizontal_fov_per_second"]
        travel += abs(pose[0] - start) / m["move_vertical_fov_per_second"]
        settling += m["settling_seconds_per_visit"]
        pass_start = travel + acquisition + settling
        ordered_rows = range(lo, hi + 1) if down else range(hi, lo - 1, -1)
        for slot, row in enumerate(ordered_rows):
            samples.append((row * cols + col, pass_start + (slot + .5) / a["cadence_hz"]))
        acquisition += (hi - lo + 1) / a["cadence_hz"]
        swept.extend(row * cols + col for row in range(lo, hi + 1))
        pose = (end, x)
    return {"duration": travel + acquisition + settling, "travel": travel,
            "acquisition": acquisition, "settling": settling,
            "swept": sorted(swept), "samples": samples, "final_pose": pose}


def mass_field(entries):
    field = np.zeros(16)
    for index, value in entries.items():
        field[index] = value
    return field


def run():
    report = {"artifact_type": "policy_v1_specification_arithmetic",
              "implementation_conformance": "not_assessed_by_this_script", "checks": {}}
    checks = report["checks"]
    profile = json.loads((ROOT / "docs/specs/instrument-profile-v1.example.json").read_text())
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    report["profile_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    assert profile_semantics(profile)
    bad = deepcopy(profile)
    bad["mechanics"]["sweep_vertical_fov_per_second"] = .5
    assert not profile_semantics(bad)  # Reject speed/cadence mismatch; do not slow.
    bad = deepcopy(profile)
    bad["acquisition"]["integration_seconds"] = 1.1
    assert not profile_semantics(bad)
    checks["instrument_consistency"] = {"valid_fixture": True, "speed_and_integration_mismatches_rejected": True}

    prior = np.array([[1., .6], [.6, 1.]])
    h = np.array([1., 0.])
    mu, cov = condition(np.zeros(2), prior, h, 2., .25)
    np.testing.assert_allclose(mu, [1.6, .96])
    np.testing.assert_allclose(cov, [[.2, .12], [.12, .712]])
    zero_mu, _ = condition(np.ones(2), prior, h, 0., .25)
    np.testing.assert_allclose(zero_mu, [.2, .52])
    independent_mu, independent_cov = condition(np.zeros(2), np.eye(2), h, 2., .25)
    assert independent_mu[1] == 0 and independent_cov[1, 1] == 1
    repeated_mu, repeated_cov = np.zeros(1), np.eye(1)
    for _ in range(2):
        repeated_mu, repeated_cov = condition(repeated_mu, repeated_cov, np.ones(1), 2., .25)
    np.testing.assert_allclose(repeated_mu, [16 / 9])
    np.testing.assert_allclose(repeated_cov, [[1 / 9]])
    checks["scalar_neighbors_zero_and_repetition"] = {
        "positive_mean": mu.tolist(), "positive_covariance": cov.tolist(),
        "zero_reading_mean": zero_mu.tolist(), "repeated_independent_mean": repeated_mu.tolist(),
        "repeated_independent_variance": repeated_cov.item()}

    footprint_mu, footprint_cov = condition(np.zeros(3), np.eye(3), np.array([.5, .5, 0]), 2., .5)
    np.testing.assert_allclose(footprint_mu, [1, 1, 0])
    np.testing.assert_allclose(footprint_cov, [[.75, -.25, 0], [-.25, .75, 0], [0, 0, 1]])
    checks["one_sparse_footprint"] = {"mean": footprint_mu.tolist(), "covariance": footprint_cov.tolist()}

    # Independent batch calculation from an augmented joint prior over X1,X2,X3.
    F, Q = np.array([[.9, .1], [0., .8]]), np.eye(2) * .02
    latent_cov = np.zeros((8, 8))  # X0, eta1, eta2, eta3
    latent_cov[:2, :2] = prior
    for i in range(1, 4):
        latent_cov[2*i:2*i+2, 2*i:2*i+2] = Q
    maps = []
    for t in range(1, 4):
        A = np.zeros((2, 8))
        A[:, :2] = np.linalg.matrix_power(F, t)
        for j in range(1, t + 1):
            A[:, 2*j:2*j+2] = np.linalg.matrix_power(F, t - j)
        maps.append(A)
    A = np.vstack(maps)
    joint = A @ latent_cov @ A.T
    H = np.zeros((2, 6))
    H[0, :2], H[1, 2:4] = [1, 0], [.25, .75]
    y, R = np.array([1.5, .2]), np.diag([.25, .4])
    cross = joint[4:6] @ H.T
    innovation = H @ joint @ H.T + R
    batch_mu = cross @ np.linalg.solve(innovation, y)
    batch_cov = joint[4:6, 4:6] - cross @ np.linalg.solve(innovation, cross.T)
    sequential_mu, sequential_cov = np.zeros(2), prior.copy()
    for t in range(3):
        sequential_mu = F @ sequential_mu
        sequential_cov = F @ sequential_cov @ F.T + Q
        if t < 2:
            sequential_mu, sequential_cov = condition(sequential_mu, sequential_cov, H[t, 2*t:2*t+2], y[t], R[t, t])
    np.testing.assert_allclose(sequential_mu, batch_mu, atol=1e-12)
    np.testing.assert_allclose(sequential_cov, batch_cov, atol=1e-12)
    assert np.linalg.eigvalsh(batch_cov).min() > 0
    checks["dynamic_sequential_vs_joint_batch"] = {
        "mean_at_3_seconds": batch_mu.tolist(), "covariance_at_3_seconds": batch_cov.tolist(),
        "maximum_error": float(max(np.max(abs(sequential_mu-batch_mu)), np.max(abs(sequential_cov-batch_cov))))}

    still = mass_field({0: .55, 4: .35, 15: .10})
    expanded = mass_field({0: .35, 4: .30, 3: .25, 15: .10})
    moved = mass_field({3: .55, 7: .35, 0: .10})
    assert select(still) == select(still.copy()) == (0, 4)
    assert select(expanded) == (0, 3, 4)
    assert select(moved) == (3, 7)
    checks["stationary_expanding_shrinking_and_displaced"] = {
        "stationary_selection": select(still), "repeated_selection": select(still.copy()),
        "expanded_selection": select(expanded), "shrunk_back_selection": select(still),
        "displaced_selection": select(moved)}

    separated = mass_field({0: .23, 12: .23, 3: .23, 15: .23, 6: .08})
    selected = select(separated)
    path = column_arithmetic(selected, profile)
    assert set(selected).issubset(path["swept"])
    assert path["duration"] == 9.5 and path["travel"] == 1.5
    retained = float(separated[path["swept"]].sum())
    np.testing.assert_allclose(retained, .92)
    assert path["duration"] > 6.0
    assert min(np.diff([t for _, t in path["samples"]])) >= 1
    for display in ((120, 160), (480, 640)):
        assert column_arithmetic(selected, profile) == path  # Display is no input.
    checks["closure_and_budget_shortfall"] = {"requested": selected, "path": path,
        "budget": 6.0, "retained_mass": retained, "tracking_executable": False,
        "single_column_mass_if_trimmed": float(separated[[0, 4, 8, 12]].sum())}

    # Initially never-visited columns: full-column additions, then age/rotation.
    last, rotation, pose, now = np.full(4, -np.inf), 0, (0., .125), 0.
    reacquisitions = []
    for _ in range(4):
        order = sorted(range(4), key=lambda c: (last[c], (c - rotation) % 4))
        chosen = []
        for c in order:
            trial = chosen + [r * 4 + c for r in range(4)]
            if column_arithmetic(trial, profile, pose)["duration"] <= 4.6:
                chosen = trial
        path = column_arithmetic(chosen, profile, pose)
        cols = sorted({i % 4 for i in chosen})
        assert len(cols) == 1
        # The oldest row midpoint determines this fully sampled column's priority.
        last[cols[0]] = now + min(t for _, t in path["samples"])
        reacquisitions.append({"column": cols[0], "duration": path["duration"]})
        now += path["duration"]
        pose, rotation = path["final_pose"], (rotation + 1) % 4
    assert [v["column"] for v in reacquisitions] == [0, 1, 2, 3]
    assert now == 17.5
    checks["charged_age_reacquisition"] = {"scans": reacquisitions, "total_seconds": now}

    b_positive = positive(1., 1.)
    expected_ratio = 10 / (10 + b_positive)
    actual_probability = NORMAL.cdf(10 / 9 - 1)
    kappa = NORMAL.inv_cdf(1 - .05 / (2 * 2))
    bound = 10 / (10 + max(1 + kappa, 0))
    assert expected_ratio > .9 and actual_probability < .55 and bound < .9
    checks["mass_is_not_coverage_probability"] = {
        "expected_positive_B": b_positive, "expected_mass_fraction": expected_ratio,
        "probability_actual_fraction_at_least_90_percent": actual_probability,
        "simultaneous_95_percent_bound": bound, "kappa": kappa,
        "uniform_mean_zero_std_point2_expected_positive": positive(0., .2)}

    # Same final estimate, different availability; left rectangles on [0,4].
    def causal_loss(available):
        return sum((1 - (.8 if t >= available else 0))**2 for t in range(4))
    early, late = causal_loss(1), causal_loss(3)
    np.testing.assert_allclose([early, late], [1.12, 3.04])
    checks["delivery_latency_changes_reconstruction_loss"] = {"stream_loss": early, "late_batch_loss": late}
    report["result"] = "passed"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
