"""
spc.py
SPC engine for the catheter OD data: X-bar and R charts, Western Electric
rules, and process capability (Cp, Cpk). Written from the formulas, with no
SPC library, so every number can be traced back to the textbook.

Formulas follow the NIST/SEMATECH e-Handbook (section 6.3.2.1) and
Montgomery, Introduction to Statistical Quality Control, ch. 6 and 8.

Typical use:
    from spc import run_spc
    result = run_spc(readings_df, lsl=1.95, usl=2.05, phase1_subgroups=100)

Quick check from the repo root:
    python src/spc.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Control chart constants, indexed by subgroup size n
#    A2 = 3 / (d2 * sqrt(n)). Values from NIST and Montgomery, Appendix VI.
# ---------------------------------------------------------------------------
CONSTANTS = {
    #  n:  (A2,    D3,    D4,    d2)
    2:  (1.880, 0.000, 3.267, 1.128),
    3:  (1.023, 0.000, 2.574, 1.693),
    4:  (0.729, 0.000, 2.282, 2.059),
    5:  (0.577, 0.000, 2.114, 2.326),
    6:  (0.483, 0.000, 2.004, 2.534),
    7:  (0.419, 0.076, 1.924, 2.704),
    8:  (0.373, 0.136, 1.864, 2.847),
    9:  (0.337, 0.184, 1.816, 2.970),
    10: (0.308, 0.223, 1.777, 3.078),
}

RULE_NAMES = {
    "rule1": "1 point beyond 3 sigma",
    "rule2": "2 of 3 points beyond 2 sigma (same side)",
    "rule3": "4 of 5 points beyond 1 sigma (same side)",
    "rule4": "8 points in a row on one side of centre",
    "r_rule1": "Range above R-chart UCL",
}


# ---------------------------------------------------------------------------
# 2. Subgroup statistics
# ---------------------------------------------------------------------------
def reading_columns(df: pd.DataFrame) -> list[str]:
    """The measurement columns: x1, x2, ..., xn."""
    cols = [c for c in df.columns if c.startswith("x") and c[1:].isdigit()]
    if not cols:
        raise ValueError("No reading columns found (expected x1, x2, ...).")
    return cols


def subgroup_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add X-bar (subgroup mean) and R (max - min) to each row."""
    x = df[reading_columns(df)]
    out = df.copy()
    out["xbar"] = x.mean(axis=1)
    out["R"] = x.max(axis=1) - x.min(axis=1)
    return out


# ---------------------------------------------------------------------------
# 3. Control limits
# ---------------------------------------------------------------------------
def control_limits(xbar: pd.Series, r: pd.Series, n: int) -> dict:
    """Shewhart X-bar and R limits from a set of in-control subgroups."""
    if n not in CONSTANTS:
        raise ValueError(f"Subgroup size n={n} not supported (use 2-10).")
    a2, d3, d4, d2 = CONSTANTS[n]

    xbarbar = xbar.mean()        # grand mean: centre line of X-bar chart
    rbar = r.mean()              # average range: centre line of R chart

    return {
        "n": n,
        "xbarbar": xbarbar,
        "rbar": rbar,
        "x_ucl": xbarbar + a2 * rbar,
        "x_lcl": xbarbar - a2 * rbar,
        "r_ucl": d4 * rbar,
        "r_lcl": d3 * rbar,
        "sigma_xbar": a2 * rbar / 3,   # 1 sigma of the subgroup means
        "sigma_hat": rbar / d2,         # 1 sigma of individual tubes
    }


# ---------------------------------------------------------------------------
# 4. Western Electric rules (X-bar chart)
#    Each rule returns a boolean array: True at the point that completes
#    the pattern, i.e. the moment the chart would raise the alarm.
# ---------------------------------------------------------------------------
def _k_of_m(z: np.ndarray, k: int, m: int, threshold: float) -> np.ndarray:
    """True where at least k of the last m points are beyond `threshold`
    sigma on the same side, and the current point is one of them."""
    above = (z > threshold).astype(int)
    below = (z < -threshold).astype(int)
    window = np.ones(m, dtype=int)
    count_above = np.convolve(above, window)[: len(z)]   # rolling sum of last m
    count_below = np.convolve(below, window)[: len(z)]
    return ((count_above >= k) & (above == 1)) | ((count_below >= k) & (below == 1))


def _run_same_side(z: np.ndarray, length: int) -> np.ndarray:
    """True where the last `length` points are all on the same side of centre."""
    side = np.sign(z)
    flags = np.zeros(len(z), dtype=bool)
    run = 0
    for i in range(len(z)):
        if i > 0 and side[i] != 0 and side[i] == side[i - 1]:
            run += 1
        else:
            run = 1 if side[i] != 0 else 0
        flags[i] = run >= length
    return flags


def western_electric(xbar: pd.Series, limits: dict) -> pd.DataFrame:
    """Apply the four Western Electric rules to the X-bar series."""
    z = ((xbar - limits["xbarbar"]) / limits["sigma_xbar"]).to_numpy()
    return pd.DataFrame({
        "rule1": np.abs(z) > 3,
        "rule2": _k_of_m(z, k=2, m=3, threshold=2),
        "rule3": _k_of_m(z, k=4, m=5, threshold=1),
        "rule4": _run_same_side(z, length=8),
    }, index=xbar.index)


def r_chart_rules(r: pd.Series, limits: dict) -> pd.DataFrame:
    """R chart: flag ranges outside the control limits."""
    return pd.DataFrame({
        "r_rule1": (r > limits["r_ucl"]) | (r < limits["r_lcl"]),
    }, index=r.index)


# ---------------------------------------------------------------------------
# 5. Process capability
# ---------------------------------------------------------------------------
def capability(xbarbar: float, sigma_hat: float, lsl: float, usl: float) -> dict:
    """Cp and Cpk from the within-subgroup sigma (R-bar / d2)."""
    cp = (usl - lsl) / (6 * sigma_hat)
    cpu = (usl - xbarbar) / (3 * sigma_hat)
    cpl = (xbarbar - lsl) / (3 * sigma_hat)
    return {"cp": cp, "cpu": cpu, "cpl": cpl, "cpk": min(cpu, cpl)}


def capability_status(cpk: float) -> str:
    """Traffic light for the dashboard (thresholds from the SPC guide, 7.4)."""
    if cpk >= 1.33:
        return "green"
    if cpk >= 1.00:
        return "yellow"
    return "red"


# ---------------------------------------------------------------------------
# 6. Full pipeline
# ---------------------------------------------------------------------------
def run_spc(df: pd.DataFrame, lsl: float, usl: float,
            phase1_subgroups: int = 100) -> dict:
    """
    Phase I: compute limits from the first `phase1_subgroups` subgroups.
    Phase II: freeze those limits and check every subgroup against them.
    """
    data = subgroup_stats(df)
    n = len(reading_columns(df))

    base = data.iloc[:phase1_subgroups]
    limits = control_limits(base["xbar"], base["R"], n)
    cap = capability(limits["xbarbar"], limits["sigma_hat"], lsl, usl)

    flags = pd.concat([western_electric(data["xbar"], limits),
                       r_chart_rules(data["R"], limits)], axis=1)
    data = pd.concat([data, flags], axis=1)
    data["alarm"] = flags.any(axis=1)

    # long table: one row per (subgroup, rule) that fired
    alarm_rows = flags.stack()
    alarm_rows = alarm_rows[alarm_rows].reset_index()
    alarm_rows.columns = ["row", "rule", "_"]
    alarms = data.loc[alarm_rows["row"]].reset_index(drop=True)
    alarms = alarms[[c for c in ["subgroup", "timestamp", "xbar", "R"] if c in alarms]]
    alarms["rule"] = alarm_rows["rule"].to_numpy()
    alarms["description"] = alarms["rule"].map(RULE_NAMES)

    return {"data": data, "limits": limits, "capability": cap,
            "status": capability_status(cap["cpk"]), "alarms": alarms}


# ---------------------------------------------------------------------------
# Quick check on the generated data
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[1] / "data"
    readings = pd.read_csv(data_dir / "od_readings.csv", parse_dates=["timestamp"])
    faults = pd.read_csv(data_dir / "fault_log.csv")

    res = run_spc(readings, lsl=1.95, usl=2.05, phase1_subgroups=100)
    L, C = res["limits"], res["capability"]

    print("Phase I limits (first 100 subgroups)")
    print(f"  X-bar chart: LCL {L['x_lcl']:.4f}  CL {L['xbarbar']:.4f}  UCL {L['x_ucl']:.4f}")
    print(f"  R chart:     LCL {L['r_lcl']:.4f}  CL {L['rbar']:.4f}  UCL {L['r_ucl']:.4f}")
    print(f"  sigma-hat {L['sigma_hat']:.4f}  Cp {C['cp']:.2f}  Cpk {C['cpk']:.2f}  ({res['status']})")

    print("\nFirst alarm inside each fault window")
    d = res["data"]
    for _, f in faults.iterrows():
        win = d[(d["subgroup"] >= f["start_subgroup"]) & (d["subgroup"] <= f["end_subgroup"])]
        hit = win[win["alarm"]]
        if hit.empty:
            print(f"  {f['type']:8s} not detected")
            continue
        first = hit.iloc[0]
        rules = [r for r in RULE_NAMES if first[r]]
        delay_h = (first["subgroup"] - f["start_subgroup"]) * 0.5
        print(f"  {f['type']:8s} detected after {delay_h:5.1f} h  by {', '.join(rules)}")

    in_fault = np.zeros(len(d), dtype=bool)
    for _, f in faults.iterrows():
        in_fault |= d["subgroup"].between(f["start_subgroup"], f["end_subgroup"]).to_numpy()
    false_alarms = d.loc[~in_fault & d["alarm"]]
    print(f"\nAlarms outside fault windows (false alarms): {len(false_alarms)} "
          f"of {(~in_fault).sum()} clean subgroups")
    