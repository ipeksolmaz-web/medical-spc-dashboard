"""
app.py
Streamlit dashboard for catheter tube OD statistical process control.

Shows X-bar and R charts with alarm points in red, an alarm table with likely
causes on the extrusion line, a Cp/Cpk traffic light, and (for the bundled
synthetic data) how fast each injected fault was detected.

Run from the repo root:
    streamlit run app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from spc import RULE_NAMES, reading_columns, run_spc  # noqa: E402

DATA_DIR = ROOT / "data"

# Likely causes on the extrusion line, by chart (literature doc, Phase B table)
LIKELY_CAUSE = {
    "rule1": "Sudden level change: new resin lot, step in screw or puller speed, air setting changed",
    "rule2": "Level moving: melt temperature drift, die deposits, screw wear",
    "rule3": "Small sustained shift: melt temperature drift, gradual flow change",
    "rule4": "Process off-centre: drift or small shift in melt flow or puller speed",
    "r_rule1": "Spread increased: unstable puller drive, melt pressure swings, quench gap",
}

STATUS_STYLE = {
    "green": ("#1a7f37", "Capable (Cpk ≥ 1.33)"),
    "yellow": ("#bf8700", "Marginal (1.00 ≤ Cpk < 1.33)"),
    "red": ("#cf222e", "Not capable (Cpk < 1.00)"),
}

st.set_page_config(page_title="Catheter OD SPC", layout="wide")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_bundled():
    readings = pd.read_csv(DATA_DIR / "od_readings.csv", parse_dates=["timestamp"])
    faults = pd.read_csv(DATA_DIR / "fault_log.csv")
    return readings, faults


def load_upload(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    reading_columns(df)                      # raises if there are no x1..xn columns
    if "subgroup" not in df:
        df.insert(0, "subgroup", np.arange(1, len(df) + 1))
    if "timestamp" in df:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Data")
source = st.sidebar.radio("Source", ["Synthetic line data (30 days)", "Upload CSV"])

faults = None
if source.startswith("Synthetic"):
    readings, faults = load_bundled()
else:
    st.sidebar.caption("One row per subgroup, readings in columns x1, x2, …, xn. "
                       "Optional columns: subgroup, timestamp.")
    upload = st.sidebar.file_uploader("CSV file", type="csv")
    if upload is None:
        st.info("Upload a CSV in the sidebar to analyse your own data.")
        st.stop()
    try:
        readings = load_upload(upload)
    except ValueError as err:
        st.error(str(err))
        st.stop()

st.sidebar.header("Specification")
c1, c2 = st.sidebar.columns(2)
lsl = c1.number_input("LSL (mm)", value=1.95, step=0.01, format="%.3f")
usl = c2.number_input("USL (mm)", value=2.05, step=0.01, format="%.3f")

st.sidebar.header("Control limits")
phase1 = st.sidebar.slider("Phase I baseline (subgroups)", 20, min(400, len(readings)),
                           min(100, len(readings)), step=5,
                           help="Limits are computed from these first subgroups, then frozen.")

st.sidebar.header("Rules")
active = [r for r in RULE_NAMES
          if st.sidebar.checkbox(RULE_NAMES[r], value=True, key=r)]

st.sidebar.header("View")
x_axis = "timestamp" if "timestamp" in readings else "subgroup"
first, last = int(readings["subgroup"].min()), int(readings["subgroup"].max())
view = st.sidebar.slider("Subgroups shown", first, last, (first, last))

# ---------------------------------------------------------------------------
# SPC
# ---------------------------------------------------------------------------
res = run_spc(readings, lsl=lsl, usl=usl, phase1_subgroups=phase1)
data, L, cap = res["data"], res["limits"], res["capability"]
data["alarm"] = data[active].any(axis=1) if active else False
alarms = res["alarms"][res["alarms"]["rule"].isin(active)].copy()
alarms["likely cause"] = alarms["rule"].map(LIKELY_CAUSE)

# ---------------------------------------------------------------------------
# Header and capability panel
# ---------------------------------------------------------------------------
st.title("Catheter tube OD: statistical process control")
st.caption(f"Outer diameter, n = {L['n']} readings per subgroup. Spec {lsl:.3f}–{usl:.3f} mm. "
           "Synthetic data, hypothetical spec.")

colour, label = STATUS_STYLE[res["status"]]
k1, k2, k3, k4, k5 = st.columns(5)
k1.markdown(
    f"<div style='background:{colour};color:white;border-radius:8px;padding:10px 14px'>"
    f"<div style='font-size:0.85rem;opacity:0.9'>Cpk</div>"
    f"<div style='font-size:2rem;font-weight:700;line-height:1.1'>{cap['cpk']:.2f}</div>"
    f"<div style='font-size:0.8rem'>{label}</div></div>", unsafe_allow_html=True)
k2.metric("Cp", f"{cap['cp']:.2f}")
k3.metric("σ̂ = R̄/d₂ (mm)", f"{L['sigma_hat']:.4f}")
k4.metric("Grand mean X̿ (mm)", f"{L['xbarbar']:.4f}")
k5.metric("Alarm subgroups", f"{data['alarm'].mean():.0%}",
          help=f"{int(data['alarm'].sum())} of {len(data)} subgroups raised at least one alarm")
st.caption(f"Capability is computed from the Phase I baseline (first {phase1} subgroups), "
           "and is only valid if that baseline is in control.")

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
shown = data[data["subgroup"].between(*view)]
x = shown[x_axis]

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                    subplot_titles=("X̄ chart (subgroup mean)", "R chart (subgroup range)"))

# shaded fault windows (ground truth, synthetic data only)
if faults is not None:
    fault_colours = {"drift": "rgba(255,165,0,0.15)", "shift": "rgba(128,0,128,0.10)",
                     "variance": "rgba(0,128,255,0.10)"}
    for _, f in faults.iterrows():
        lo, hi = max(f["start_subgroup"], view[0]), min(f["end_subgroup"], view[1])
        if lo > hi:
            continue
        x0 = data.loc[data["subgroup"] == lo, x_axis].iloc[0]
        x1 = data.loc[data["subgroup"] == hi, x_axis].iloc[0]
        if isinstance(x0, pd.Timestamp):          # plotly shapes need plain strings for dates
            x0, x1 = x0.isoformat(), x1.isoformat()
        # a filled rectangle as a trace, spanning the full height of each chart
        for row, col_name in ((1, "xbar"), (2, "R")):
            y_lo = min(data[col_name].min(), L["x_lcl"] if row == 1 else 0)
            y_hi = max(data[col_name].max(), L["x_ucl"] if row == 1 else L["r_ucl"])
            fig.add_trace(go.Scatter(
                x=[x0, x1, x1, x0, x0], y=[y_lo, y_lo, y_hi, y_hi, y_lo],
                fill="toself", fillcolor=fault_colours.get(f["type"], "rgba(0,0,0,0.08)"),
                line=dict(width=0), mode="lines", hoverinfo="skip", showlegend=False,
            ), row=row, col=1)
        fig.add_annotation(x=x0, y=1, yref="y domain", xanchor="left", yanchor="top", xshift=4,
                           text=f["type"], showarrow=False, font=dict(size=11), row=1, col=1)


def add_chart(row, y, flag_cols, cl, ucl, lcl, name):
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=name,
                             line=dict(color="#4c6ef5", width=1), marker=dict(size=3),
                             showlegend=False), row=row, col=1)
    flagged = shown[shown[flag_cols].any(axis=1)] if flag_cols else shown.iloc[0:0]
    fig.add_trace(go.Scatter(x=flagged[x_axis], y=y.loc[flagged.index], mode="markers",
                             marker=dict(color="#cf222e", size=7), name="Alarm",
                             showlegend=False), row=row, col=1)
    for val, dash, colour in [(cl, "solid", "#1a7f37"), (ucl, "dash", "#cf222e"),
                              (lcl, "dash", "#cf222e")]:
        fig.add_hline(y=val, line_dash=dash, line_color=colour, line_width=1, row=row, col=1)


x_rules = [r for r in active if r.startswith("rule")]
r_rules = [r for r in active if r.startswith("r_")]
add_chart(1, shown["xbar"], x_rules, L["xbarbar"], L["x_ucl"], L["x_lcl"], "X̄")
add_chart(2, shown["R"], r_rules, L["rbar"], L["r_ucl"], L["r_lcl"], "R")
fig.update_yaxes(title_text="mm", row=1, col=1)
fig.update_yaxes(title_text="mm", row=2, col=1)
fig.update_layout(height=620, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified")
st.plotly_chart(fig, width="stretch")

st.caption(f"X̄ limits {L['x_lcl']:.4f} / {L['xbarbar']:.4f} / {L['x_ucl']:.4f} mm.  "
           f"R limits {L['r_lcl']:.4f} / {L['rbar']:.4f} / {L['r_ucl']:.4f} mm.  "
           "Red dots: alarms. Shaded bands: injected faults (ground truth).")

# ---------------------------------------------------------------------------
# Fault detection (synthetic data only)
# ---------------------------------------------------------------------------
if faults is not None:
    st.subheader("Fault detection")
    interval_h = (data["timestamp"].iloc[1] - data["timestamp"].iloc[0]).total_seconds() / 3600
    rows = []
    in_fault = np.zeros(len(data), dtype=bool)
    for _, f in faults.iterrows():
        mask = data["subgroup"].between(f["start_subgroup"], f["end_subgroup"]).to_numpy()
        in_fault |= mask
        hits = data[mask & data["alarm"].to_numpy()]
        if hits.empty:
            rows.append({"fault": f["type"], "cause (injected)": f["cause"],
                         "detected": "No", "delay (h)": None, "first rule": "", "alarms in window": 0})
            continue
        first_hit = hits.iloc[0]
        rows.append({
            "fault": f["type"], "cause (injected)": f["cause"], "detected": "Yes",
            "delay (h)": (first_hit["subgroup"] - f["start_subgroup"]) * interval_h,
            "first rule": ", ".join(r for r in active if first_hit[r]),
            "alarms in window": len(hits),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    clean = ~in_fault
    clean[:phase1] = False          # baseline is excluded from the false-alarm count
    n_false = int((data["alarm"].to_numpy() & clean).sum())
    st.caption(f"False alarms: {n_false} of {int(clean.sum())} fault-free subgroups after the "
               f"baseline ({n_false / max(clean.sum(), 1):.1%}). The first alarm in a window can "
               "be a coincidence; Step 5 scores each rule against the fault it is meant to catch.")

# ---------------------------------------------------------------------------
# Alarm table
# ---------------------------------------------------------------------------
st.subheader(f"Alarms ({len(alarms)})")
show_cols = [c for c in ["subgroup", "timestamp", "xbar", "R", "description", "likely cause"]
             if c in alarms]
st.dataframe(alarms[show_cols].round({"xbar": 4, "R": 4}), hide_index=True,
             width="stretch", height=320)
st.download_button("Download alarms as CSV", alarms[show_cols].to_csv(index=False),
                   file_name="spc_alarms.csv", mime="text/csv")
