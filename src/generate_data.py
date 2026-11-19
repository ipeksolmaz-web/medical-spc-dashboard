"""
generate_data.py
Synthetic outer-diameter (OD) data for a catheter tube extrusion line.

Simulates a laser OD gauge (stage 8 of the line) that records one subgroup of
n consecutive readings every INTERVAL_MIN minutes, and injects three known
faults. Every fault is written to a ground-truth log so the SPC engine can
later be scored on detection delay and false alarms.

All values are hypothetical. Noise level is based on the +/-20 um OD stability
reported by Cho et al. (2020), Polymers 12(8), 1628, p. 7, Fig. 6b.

Outputs (in data/):
    od_readings.csv  one row per subgroup: subgroup, timestamp, day, x1..xn
    fault_log.csv    one row per fault: type, start/end subgroup and time, cause

Run from the repo root:
    python src/generate_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Process settings  (change these, then re-run)
# ---------------------------------------------------------------------------
SEED = 42                       # fixed seed: everyone gets the same data
TARGET_OD = 2.00                # mm, nominal outer diameter
SIGMA = 0.008                   # mm, in-control short-term std dev
N = 4                           # readings per subgroup
INTERVAL_MIN = 30               # minutes between subgroups
DAYS = 30                       # length of the run
START = "2026-01-01 00:00"      # timestamp of the first subgroup
GAUGE_DECIMALS = 4              # gauge resolution: 0.0001 mm

# ---------------------------------------------------------------------------
# 2. Faults to inject  (days are 1-based: day 1 = first day of the run)
#    start_day: fault begins at 00:00 of this day
#    end_day:   fault ends at 24:00 of this day (process back to normal after)
# ---------------------------------------------------------------------------
FAULTS = [
    {
        "type": "drift",
        "start_day": 10, "end_day": 14,
        "magnitude": 0.03,      # mm: mean rises linearly from 0 to +0.03 mm
        "cause": "Melt temperature drift (barrel heater zone)",
    },
    {
        "type": "shift",
        "start_day": 18, "end_day": 20,
        "magnitude": 0.02,      # mm: mean jumps by +0.02 mm and stays there
        "cause": "New resin lot with different melt flow",
    },
    {
        "type": "variance",
        "start_day": 24, "end_day": 27,
        "magnitude": 2.5,       # sigma multiplier: sigma ramps from 1x to 2.5x
        "cause": "Unstable puller drive (speed fluctuation)",
    },
]

# ---------------------------------------------------------------------------
# 3. Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


def subgroups_per_day() -> int:
    return 24 * 60 // INTERVAL_MIN


def fault_window(fault: dict) -> tuple[int, int]:
    """Return (first, last) subgroup index (0-based, inclusive) of a fault."""
    per_day = subgroups_per_day()
    first = (fault["start_day"] - 1) * per_day
    last = fault["end_day"] * per_day - 1
    return first, last


def build_profiles(k: int) -> tuple[np.ndarray, np.ndarray]:
    """True process mean and sigma for each of the k subgroups."""
    mean = np.full(k, TARGET_OD)
    sigma = np.full(k, SIGMA)

    for fault in FAULTS:
        first, last = fault_window(fault)
        length = last - first + 1
        ramp = np.linspace(0, 1, length)   # 0 at fault start, 1 at fault end

        if fault["type"] == "drift":
            mean[first:last + 1] += fault["magnitude"] * ramp
        elif fault["type"] == "shift":
            mean[first:last + 1] += fault["magnitude"]
        elif fault["type"] == "variance":
            sigma[first:last + 1] *= 1 + (fault["magnitude"] - 1) * ramp
        else:
            raise ValueError(f"Unknown fault type: {fault['type']}")

    return mean, sigma


def generate_readings(rng: np.random.Generator) -> pd.DataFrame:
    k = DAYS * subgroups_per_day()
    mean, sigma = build_profiles(k)

    # one row per subgroup, N independent normal readings around the true mean
    readings = rng.normal(loc=mean[:, None], scale=sigma[:, None], size=(k, N))
    readings = readings.round(GAUGE_DECIMALS)

    timestamps = pd.date_range(START, periods=k, freq=f"{INTERVAL_MIN}min")
    df = pd.DataFrame(readings, columns=[f"x{i + 1}" for i in range(N)])
    df.insert(0, "subgroup", np.arange(1, k + 1))
    df.insert(1, "timestamp", timestamps)
    df.insert(2, "day", np.arange(k) // subgroups_per_day() + 1)
    return df


def build_fault_log(readings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, fault in enumerate(FAULTS, start=1):
        first, last = fault_window(fault)
        rows.append({
            "fault_id": i,
            "type": fault["type"],
            "start_subgroup": first + 1,          # 1-based, matches od_readings
            "end_subgroup": last + 1,
            "start_time": readings["timestamp"].iloc[first],
            "end_time": readings["timestamp"].iloc[last],
            "magnitude": fault["magnitude"],
            "magnitude_unit": "sigma multiplier" if fault["type"] == "variance" else "mm",
            "cause": fault["cause"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(SEED)
    readings = generate_readings(rng)
    fault_log = build_fault_log(readings)

    DATA_DIR.mkdir(exist_ok=True)
    readings.to_csv(DATA_DIR / "od_readings.csv", index=False)
    fault_log.to_csv(DATA_DIR / "fault_log.csv", index=False)

    print(f"Wrote {len(readings)} subgroups of n={N} to data/od_readings.csv")
    print(f"Wrote {len(fault_log)} faults to data/fault_log.csv\n")
    print(fault_log[["type", "start_subgroup", "end_subgroup", "start_time", "cause"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
