"""
Checks spc.py against the hand-calculated exercise in Section 9 of the
SPC guide. If these fail, the code is wrong, not the textbook.

Run from the repo root:
    python tests/test_spc.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spc import capability, control_limits, subgroup_stats, western_electric  # noqa: E402

TOL = 1e-4

# Section 9 data: four subgroups of n = 4, spec 1.95-2.05 mm
EXERCISE = pd.DataFrame({
    "x1": [2.01, 2.00, 2.02, 1.99],
    "x2": [1.99, 2.01, 2.00, 2.00],
    "x3": [2.00, 1.98, 2.01, 2.02],
    "x4": [2.02, 2.00, 2.01, 2.00],
})

# Answer key, worked by hand
EXPECTED = {
    "xbar": [2.005, 1.9975, 2.010, 2.0025],
    "R": [0.03, 0.03, 0.02, 0.03],
    "xbarbar": 2.00375,
    "rbar": 0.0275,
    "x_ucl": 2.02380,
    "x_lcl": 1.98370,
    "r_ucl": 0.06276,
    "r_lcl": 0.0,
    "sigma_hat": 0.01336,
    "cp": 1.2479,
    "cpk": 1.1543,
}


def close(a, b, tol=TOL):
    return abs(a - b) < tol


def test_exercise():
    stats = subgroup_stats(EXERCISE)
    assert np.allclose(stats["xbar"], EXPECTED["xbar"], atol=TOL)
    assert np.allclose(stats["R"], EXPECTED["R"], atol=TOL)

    lim = control_limits(stats["xbar"], stats["R"], n=4)
    for key in ["xbarbar", "rbar", "x_ucl", "x_lcl", "r_ucl", "r_lcl", "sigma_hat"]:
        assert close(lim[key], EXPECTED[key]), f"{key}: got {lim[key]:.5f}, expected {EXPECTED[key]}"

    cap = capability(lim["xbarbar"], lim["sigma_hat"], lsl=1.95, usl=2.05)
    assert close(cap["cp"], EXPECTED["cp"], 1e-3), f"Cp: got {cap['cp']:.4f}"
    assert close(cap["cpk"], EXPECTED["cpk"], 1e-3), f"Cpk: got {cap['cpk']:.4f}"


def test_rules():
    # centre 0, sigma 1, so the X-bar values are already z-scores
    limits = {"xbarbar": 0.0, "sigma_xbar": 1.0}
    cases = {
        "rule1": ([0, 0, 3.5, 0], 2),                        # one point beyond 3
        "rule2": ([0, 2.5, 0.5, 2.5, 0], 3),                 # 2 of 3 beyond 2
        "rule3": ([0, 1.5, 1.5, 0.2, 1.5, 1.5, 0], 5),       # 4 of 5 beyond 1
        "rule4": ([-1] + [0.3] * 8 + [-1], 8),               # 8 in a row above
    }
    for rule, (values, expected_first) in cases.items():
        flags = western_electric(pd.Series(values, dtype=float), limits)[rule]
        first = int(np.argmax(flags.to_numpy())) if flags.any() else None
        assert first == expected_first, f"{rule}: first alarm at {first}, expected {expected_first}"

    # pure noise inside 1 sigma should never fire rules 1-3
    quiet = western_electric(pd.Series([0.5, -0.5] * 10), limits)
    assert not quiet[["rule1", "rule2", "rule3", "rule4"]].any().any()


if __name__ == "__main__":
    test_exercise()
    print("Section 9 exercise: all values match the hand calculation")
    test_rules()
    print("Western Electric rules: all test patterns fire at the right point")
    