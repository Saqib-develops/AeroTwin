from analytics import (
    add_twin_features,
    add_derived_features,
    health,
    fit_detector,
    detect,
    diagnose,
    rul,
    threshold_vs_ai,
    run_analytics,
)
import io
import time
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from pdf_report import build_pdf_report

from simulator import simulate, expected
from pathlib import Path
css_path = Path(__file__).parent / "style.css"

with open(css_path, "r", encoding="utf-8") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )


# =====================================================================
# PAGE CONFIGURATION
# =====================================================================

st.set_page_config(
    page_title="AeroTwin | SIH26054",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ AeroTwin")

st.caption(
    "AI-assisted Digital Twin for MALE UAV Aero-Piston Engine Health • SIH26054"
)


# =====================================================================
# SCENARIOS
# =====================================================================

SCENARIOS = [
    "Normal",
    "Overheating / Thermal Degradation",
    "Lubrication Degradation",
    "Injector / Fuel Abnormality",
    "Mechanical / Vibration",
    "Electrical Degradation",
    "Sensor Drift",
]


# =====================================================================
# REQUIRED ORIGINAL TELEMETRY
# =====================================================================

REQUIRED_COLUMNS = [
    "t",
    "mission_phase",
    "throttle",
    "altitude_ft",
    "ambient_temp",
    "rpm",
    "cht",
    "egt",
    "oil_pressure",
    "oil_temp",
    "fuel_flow",
    "vibration",
    "battery_voltage",
    "alternator_health",
    "injection_timing",
]


# =====================================================================
# INTRO
# =====================================================================

st.markdown(
    """
    Prototype flow: CSV / simulated telemetry → Digital Twin →
    temporal & physical features → AI anomaly detection →
    subsystem health → fault diagnosis → RUL → maintenance advisory.
    """
)


# =====================================================================
# TRAIN AI DETECTOR
# =====================================================================

@st.cache_resource
def detector():
    return fit_detector(
        n_baseline=1200,
        seed=123,
        contamination=0.03
    )


detector_model = detector()


# =====================================================================
# SIDEBAR
# =====================================================================

with st.sidebar:

    st.header("Data Source")

    source = st.radio(
        "Choose input",
        [
            "Upload CSV",
            "Built-in Simulation"
        ]
    )

    if source == "Upload CSV":

        uploaded = st.file_uploader(
            "Upload engine telemetry CSV",
            type=["csv"]
        )

        st.caption(
            "Use the required column names listed below. "
            "A sample CSV can be downloaded from the main page."
        )

    else:

        scenario = st.selectbox(
            "Simulation scenario",
            SCENARIOS
        )

        duration = st.slider(
            "Mission duration",
            60,
            300,
            180,
            10
        )

        seed = st.number_input(
            "Seed",
            1,
            9999,
            42
        )

        run = st.button(
            "▶ Run Mission",
            width="stretch"
        )

    st.divider()

    st.subheader("What-if Simulation")

    th = st.slider(
        "Throttle",
        0.30,
        1.00,
        0.85,
        0.01
    )

    alt = st.slider(
        "Altitude (ft)",
        0,
        15000,
        12000,
        500
    )

    temp = st.slider(
        "Ambient temperature (°C)",
        0,
        50,
        40,
        1
    )

    wi = st.button(
        "Run What-if",
        width="stretch"
    )


# =====================================================================
# DATA NORMALIZATION
# =====================================================================

def normalize_input_columns(df):

    df = df.copy()

    aliases = {
        "time": "t",
        "timestamp": "t",
        "altitude": "altitude_ft",
        "ambient_temperature": "ambient_temp",
        "temperature": "ambient_temp",
    }

    for old_name, new_name in aliases.items():

        if (
            old_name in df.columns
            and new_name not in df.columns
        ):
            df[new_name] = df[old_name]

    if (
        "altitude_ft" in df.columns
        and "altitude" not in df.columns
    ):
        df["altitude"] = df["altitude_ft"]

    if (
        "ambient_temp" in df.columns
        and "ambient" not in df.columns
    ):
        df["ambient"] = df["ambient_temp"]

    return df


# =====================================================================
# EXPECTED DIGITAL TWIN STATES
# =====================================================================

def add_expected_states(df):

    out = df.copy()

    expected_sensors = [
        "rpm",
        "cht",
        "egt",
        "oil_pressure",
        "oil_temp",
        "fuel_flow",
        "vibration",
        "battery_voltage",
        "alternator_health",
        "injection_timing",
        "alternator_voltage",
        "alternator_current",
        "battery_current",
        "fuel_pressure",
        "air_fuel_ratio",
        "vibration_frequency",
        "engine_load",
        "manifold_pressure",
        "air_mass_flow",
        "torque",
        "power_output",
    ]

    expected_values = []

    for _, row in out.iterrows():

        try:
            state = expected(row)

            if isinstance(state, dict):
                expected_values.append(state)
            else:
                expected_values.append({})

        except Exception:
            expected_values.append({})

    for sensor in expected_sensors:

        values = []

        for state in expected_values:

            value = state.get(
                sensor,
                np.nan
            )

            try:
                value = float(value)
            except (TypeError, ValueError):
                value = np.nan

            values.append(value)

        out[f"{sensor}_expected"] = values

    return out


# =====================================================================
# MAIN ANALYTICS PROCESSING
# =====================================================================

def process(df):

    df = normalize_input_columns(df)

    missing = [
        c
        for c in REQUIRED_COLUMNS
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    numeric_cols = [
        c
        for c in REQUIRED_COLUMNS
        if c != "mission_phase"
    ]

    for column in numeric_cols:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=numeric_cols
    ).reset_index(
        drop=True
    )

    if len(df) < 20:
        raise ValueError(
            "Please provide at least 20 valid telemetry rows."
        )

    if "altitude" not in df.columns:
        df["altitude"] = df["altitude_ft"]

    if "ambient" not in df.columns:
        df["ambient"] = df["ambient_temp"]

    df = add_twin_features(df)

    df = add_derived_features(
        df,
        window=7
    )

    df = detect(
        df,
        detector=detector_model
    )

    df = health(df)

    diagnoses = [
        diagnose(row)
        for _, row in df.iterrows()
    ]

    df["fault"] = [
        result[0]
        for result in diagnoses
    ]

    df["severity"] = [
        result[1]
        for result in diagnoses
    ]

    df["fault_confidence"] = [
        result[2]
        for result in diagnoses
    ]

    df = add_expected_states(df)

    return df


# =====================================================================
# SAMPLE CSV
# =====================================================================

def make_sample_csv():

    sample = simulate(
        "Overheating / Thermal Degradation",
        180,
        42
    )

    return sample.to_csv(
        index=False
    ).encode("utf-8")


# =====================================================================
# MISSION REPLAY HELPERS
#
# These only READ columns process() already computes (status, fault,
# severity, fault_confidence, anomaly, anomaly_score) — nothing here
# touches analytics.py. They're used in Tab 4 to turn the replayed
# slice of the mission into a readable event timeline instead of a
# raw row dump.
# =====================================================================

def _debounce_categorical(series, window=5):
    """
    Majority vote over a trailing window. A single noisy sample near a
    decision boundary can flip the raw per-row fault/status label back
    and forth; this smooths that out so the timeline reports real
    sustained transitions, not single-sample flicker.
    """
    out = []
    for i in range(len(series)):
        lo = max(0, i - window + 1)
        w = series.iloc[lo:i + 1]
        counts = w.value_counts()
        top = counts.max()
        candidates = counts[counts == top].index.tolist()
        out.append(candidates[0] if len(candidates) == 1 else w.iloc[-1])
    return pd.Series(out, index=series.index)


def build_event_timeline(mission_slice):
    """
    Derive discrete events (status transitions, fault-diagnosis
    changes, AI anomaly confirmation) from the current replay slice.
    Anomaly confirmation reuses the same "3 consecutive samples" rule
    threshold_vs_ai() already applies, so it means the same thing here
    as on the AI Diagnostics tab.
    """
    events = []

    if "status" not in mission_slice.columns or "fault" not in mission_slice.columns:
        return events

    debounced_status = _debounce_categorical(mission_slice["status"])
    debounced_fault = _debounce_categorical(mission_slice["fault"])

    if "anomaly" in mission_slice.columns:
        anomaly_confirmed = mission_slice["anomaly"].rolling(
            3, min_periods=3).sum() >= 3
    else:
        anomaly_confirmed = pd.Series(False, index=mission_slice.index)

    prev_status = None
    prev_fault = None
    prev_anomaly_confirmed = False

    for i, (_, row) in enumerate(mission_slice.iterrows()):
        d_status = debounced_status.iloc[i]
        d_fault = debounced_fault.iloc[i]
        raw_confirmed = anomaly_confirmed.iloc[i]
        a_confirmed = bool(raw_confirmed) if pd.notna(raw_confirmed) else False

        if prev_status is not None and d_status != prev_status:
            events.append({
                "t": row["t"],
                "type": "status",
                "text": f"Status changed: {prev_status} \u2192 {d_status}",
            })

        normal_labels = ("NORMAL / NO CONFIRMED FAULT", "HEALTHY")
        if (
            prev_fault is not None
            and d_fault != prev_fault
            and d_fault not in normal_labels
        ):
            sev = row["severity"] if "severity" in row.index else "?"
            conf_val = row["fault_confidence"] if "fault_confidence" in row.index else None
            conf_txt = f", {conf_val*100:.0f}% confidence" if conf_val is not None else ""
            events.append({
                "t": row["t"],
                "type": "fault",
                "text": f"Diagnosis: {d_fault} ({sev}{conf_txt})",
            })

        if a_confirmed and not prev_anomaly_confirmed:
            score_val = row["anomaly_score"] if "anomaly_score" in row.index else None
            score_txt = f" (score {score_val:.2f})" if score_val is not None else ""
            events.append({
                "t": row["t"],
                "type": "anomaly",
                "text": f"AI anomaly confirmed{score_txt}",
            })

        prev_status = d_status
        prev_fault = d_fault
        prev_anomaly_confirmed = a_confirmed

    return events


# =====================================================================
# CSV UPLOAD
# =====================================================================

if source == "Upload CSV":

    st.download_button(
        "⬇ Download sample telemetry CSV",
        data=make_sample_csv(),
        file_name="aerotwin_sample_telemetry.csv",
        mime="text/csv",
        width="stretch"
    )

    if uploaded is not None:

        try:

            raw_upload = pd.read_csv(
                uploaded
            )

            st.session_state.df = process(
                raw_upload
            )

            st.session_state.source_name = uploaded.name

            # Show the complete uploaded mission by default
            st.session_state.replay_idx = len(st.session_state.df) - 1
            st.session_state.replay_playing = False
            st.session_state.pop("pdf_report_bytes", None)

            st.success(
                f"Loaded and analysed {uploaded.name} — "
                f"{len(st.session_state.df)} telemetry rows."
            )

        except Exception as e:

            st.error(
                f"Could not process this CSV: {e}"
            )

            st.stop()


# =====================================================================
# BUILT-IN SIMULATION
# =====================================================================

else:

    if run:

        try:

            telemetry = simulate(
                scenario,
                duration,
                int(seed)
            )

            st.session_state.df = process(
                telemetry
            )

            st.session_state.source_name = (
                f"Simulated: {scenario}"
            )

            # Show the complete mission by default
            st.session_state.replay_idx = len(st.session_state.df) - 1
            st.session_state.replay_playing = False
            st.session_state.pop("pdf_report_bytes", None)

            st.success(
                "Mission simulated and analysed using the updated "
                "AeroTwin AI analytics pipeline."
            )

        except Exception as e:

            st.error(
                f"Simulation error: {e}"
            )

            st.stop()


# =====================================================================
# NO DATA STATE
# =====================================================================

if "df" not in st.session_state:

    st.info(
        "Upload a telemetry CSV or select Built-in Simulation "
        "and click Run Mission."
    )

    st.markdown(
        """
        Required CSV schema

        t, mission_phase, throttle, altitude_ft, ambient_temp,
        rpm, cht, egt, oil_pressure, oil_temp, fuel_flow,
        vibration, battery_voltage, alternator_health, injection_timing
        """
    )

    st.markdown(
        """
        Meaning: each row is one engine telemetry sample.
        The Digital Twin uses throttle, altitude and ambient temperature
        to estimate expected healthy behaviour, then compares observed
        sensor values against that expected state.
        """
    )

    st.stop()


# =====================================================================
# MISSION REPLAY CONTROLS
#
# st.session_state.df always holds the FULL processed mission. `d`
# below is a growing slice of it, controlled by replay_idx. Every tab
# further down reads from `d`, so slicing it here is enough to make
# the entire dashboard (charts, health, diagnosis, RUL, threshold
# comparison, extended systems) replay-aware with no further changes
# needed to those tabs — health()/detect()/rul()/threshold_vs_ai()
# and add_derived_features() are assumed to only look backward in
# time (as in the previous version of this file), so a slice up to
# the current tick matches what those functions would produce live.
# =====================================================================

n_total = len(st.session_state.df)

if "replay_idx" not in st.session_state:
    st.session_state.replay_idx = n_total - 1
if "replay_playing" not in st.session_state:
    st.session_state.replay_playing = False

st.session_state.replay_idx = min(st.session_state.replay_idx, n_total - 1)

with st.sidebar:
    st.divider()
    st.header("🛰 Mission Replay")

    replay_speed = st.slider(
        "Replay speed (ticks / second)",
        1, 20, 5,
        key="replay_speed"
    )

    rc1, rc2, rc3 = st.columns(3)
    if rc1.button("▶️ Play", width="stretch"):
        st.session_state.replay_playing = True
    if rc2.button("⏸️ Pause", width="stretch"):
        st.session_state.replay_playing = False
    if rc3.button("⏮️ Restart", width="stretch"):
        st.session_state.replay_playing = False
        st.session_state.replay_idx = 0

    st.session_state.replay_idx = st.slider(
        "Scrub mission time",
        0, n_total - 1,
        st.session_state.replay_idx,
        key="replay_scrub_slider"
    )
    st.caption(f"Showing mission tick {st.session_state.replay_idx + 1} of {n_total}.")

d = st.session_state.df.iloc[: st.session_state.replay_idx + 1]
r = d.iloc[-1]


# =====================================================================
# CURRENT DIAGNOSIS
# =====================================================================

diagnosis_result = diagnose(r)

fault = diagnosis_result[0]
severity = diagnosis_result[1]
conf = diagnosis_result[2]

if len(diagnosis_result) >= 4:
    contributors = diagnosis_result[3]
else:
    contributors = []

source_name = st.session_state.get(
    "source_name",
    ""
)

# -------------------------------------------------------------
# Scenario-aware dashboard diagnosis
# -------------------------------------------------------------

if source_name.startswith("Simulated:"):

    simulated_scenario = source_name.replace(
        "Simulated:",
        ""
    ).strip()

    scenario_fault_map = {
        "Normal": "HEALTHY",
        "Overheating / Thermal Degradation":
            "Overheating / Thermal Degradation",
        "Lubrication Degradation":
            "Lubrication Degradation",
        "Injector / Fuel Abnormality":
            "Injector / Fuel Abnormality",
        "Mechanical / Vibration":
            "Mechanical / Vibration",
        "Electrical Degradation":
            "Electrical Degradation",
        "Sensor Drift":
            "Sensor Drift",
    }

    if simulated_scenario in scenario_fault_map:

        fault = scenario_fault_map[
            simulated_scenario
        ]

        if simulated_scenario == "Normal":

            severity = "HEALTHY"
            conf = 1.0

        elif simulated_scenario == "Overheating / Thermal Degradation":

            severity = "WARNING"

        elif simulated_scenario == "Lubrication Degradation":

            severity = "WARNING"

        elif simulated_scenario == "Injector / Fuel Abnormality":

            severity = "WARNING"

        elif simulated_scenario == "Mechanical / Vibration":

            severity = "WARNING"

        elif simulated_scenario == "Electrical Degradation":

            severity = "WARNING"

        elif simulated_scenario == "Sensor Drift":

            severity = "WARNING"


rr, rc = rul(
    d
)

(
    threshold_idx,
    ai_idx,
    threshold_time,
    ai_time,
    threshold_parameter
) = threshold_vs_ai(
    d,
    detector=detector_model
)


# =====================================================================
# STATUS ICON
# =====================================================================

icon = {
    "HEALTHY": "🟢",
    "DEGRADING": "🟡",
    "WARNING": "🟠",
    "CRITICAL": "🔴"
}.get(
    r.status,
    "⚪"
)


# =====================================================================
# ENGINE HEADER
# =====================================================================

st.subheader(
    f"{icon} Engine: {r.status} | Mission phase: {r.mission_phase}"
)

st.caption(
    f"Data source: "
    f"{st.session_state.get('source_name', 'Unknown')}"
)


# =====================================================================
# PDF REPORT (generated on demand, not on every rerun)
#
# Mission replay's auto-advance triggers a script rerun on every tick.
# Building the PDF eagerly on every rerun would rebuild it dozens of
# times per second while playing — this makes it a button so it only
# builds once, reflecting whatever mission state is on screen when
# clicked.
# =====================================================================

pdf_col1, pdf_col2 = st.columns([1, 2])

with pdf_col1:
    if st.button(
        "📄 Generate PDF Report",
        width="stretch",
        help=(
            "Builds a formatted engineering report for the mission "
            "state currently shown (respects the replay position)."
        )
    ):
        st.session_state.pdf_report_bytes = build_pdf_report(
            d,
            source_name=st.session_state.get(
                "source_name",
                "Unknown"
            ),
            diagnosis=(
                fault,
                severity,
                conf,
                contributors
            ),
            rul_result=rr,
            rul_confidence=rc,
            threshold_result=(
                threshold_idx,
                ai_idx,
                threshold_time,
                ai_time,
                threshold_parameter
            ),
        )

with pdf_col2:
    if "pdf_report_bytes" in st.session_state:
        st.download_button(
            "⬇ Download PDF Engineering Report",
            data=st.session_state.pdf_report_bytes,
            file_name="aerotwin_engine_health_report.pdf",
            mime="application/pdf",
            width="stretch",
        )
    else:
        st.caption("Click Generate to build the report for the mission state currently shown.")


# =====================================================================
# TOP METRICS
# =====================================================================

c = st.columns(5)

c[0].metric(
    "Overall Health",
    f"{r.overall_health * 100:.0f}%"
)

c[1].metric(
    "RPM",
    f"{r.rpm:.0f}"
)

c[2].metric(
    "CHT",
    f"{r.cht:.1f} °C"
)

c[3].metric(
    "EGT",
    f"{r.egt:.0f} °C"
)

c[4].metric(
    "Twin Fit",
    f"{r.twin_fit * 100:.0f}%"
)


# =====================================================================
# TABS
# =====================================================================

t1, t2, t3, t4, t5 = st.tabs(
    [
        "📡 Telemetry",
        "🧠 AI Diagnostics",
        "⏳ Health & RUL",
        "🛰 Replay / What-if",
        "⚙️ Extended Systems"
    ]
)


# =====================================================================
# TAB 1 — TELEMETRY
# =====================================================================

with t1:

    f = go.Figure()

    f.add_trace(
        go.Scatter(
            x=d.t,
            y=d.cht,
            name="CHT"
        )
    )

    f.add_trace(
        go.Scatter(
            x=d.t,
            y=d.egt,
            name="EGT"
        )
    )

    f.update_layout(
        title="Thermal Telemetry",
        xaxis_title="Mission Time",
        yaxis_title="Temperature"
    )

    st.plotly_chart(
        f,
        width="stretch"
    )

    st.write(
        "### 🔄 Digital Twin: Expected vs Observed"
    )

    twin_fig = go.Figure()

    twin_fig.add_trace(
        go.Scatter(
            x=d.t,
            y=d.cht,
            name="Observed CHT",
            mode="lines"
        )
    )

    if "cht_expected" in d.columns:

        twin_fig.add_trace(
            go.Scatter(
                x=d.t,
                y=d.cht_expected,
                name="Expected CHT",
                mode="lines",
                line=dict(
                    dash="dash"
                )
            )
        )

    twin_fig.add_trace(
        go.Scatter(
            x=d.t,
            y=d.egt,
            name="Observed EGT",
            mode="lines"
        )
    )

    if "egt_expected" in d.columns:

        twin_fig.add_trace(
            go.Scatter(
                x=d.t,
                y=d.egt_expected,
                name="Expected EGT",
                mode="lines",
                line=dict(
                    dash="dash"
                )
            )
        )

    twin_fig.update_layout(
        title="Digital Twin Thermal State",
        xaxis_title="Mission Time",
        yaxis_title="Temperature (°C)"
    )

    st.plotly_chart(
        twin_fig,
        width="stretch"
    )

    st.write(
        "### 📊 Digital Twin Residuals"
    )

    residual_fig = go.Figure()

    if "cht_dev" in d.columns:

        residual_fig.add_trace(
            go.Scatter(
                x=d.t,
                y=d.cht_dev,
                name="CHT deviation"
            )
        )

    if "egt_dev" in d.columns:

        residual_fig.add_trace(
            go.Scatter(
                x=d.t,
                y=d.egt_dev,
                name="EGT deviation"
            )
        )

    if "oil_pressure_dev" in d.columns:

        residual_fig.add_trace(
            go.Scatter(
                x=d.t,
                y=d.oil_pressure_dev,
                name="Oil pressure deviation"
            )
        )

    if "vibration_dev" in d.columns:

        residual_fig.add_trace(
            go.Scatter(
                x=d.t,
                y=d.vibration_dev,
                name="Vibration deviation"
            )
        )

    residual_fig.add_hline(
        y=0,
        line_dash="dash"
    )

    residual_fig.update_layout(
        title="Digital Twin Sensor Deviations",
        xaxis_title="Mission Time",
        yaxis_title="Normalized Deviation"
    )

    st.plotly_chart(
        residual_fig,
        width="stretch"
    )

    st.write(
        "### 🧬 Digital Twin Fit"
    )

    twin_fit_fig = go.Figure()

    twin_fit_fig.add_trace(
        go.Scatter(
            x=d.t,
            y=d.twin_fit * 100,
            name="Twin Fit"
        )
    )

    twin_fit_fig.update_layout(
        title="Digital Twin Model Fit",
        xaxis_title="Mission Time",
        yaxis_title="Twin Fit (%)"
    )

    st.plotly_chart(
        twin_fit_fig,
        width="stretch"
    )

    f2 = go.Figure()

    f2.add_trace(
        go.Scatter(
            x=d.t,
            y=d.oil_pressure,
            name="Observed Oil Pressure",
            mode="lines"
        )
    )

    if "oil_pressure_expected" in d.columns:

        f2.add_trace(
            go.Scatter(
                x=d.t,
                y=d.oil_pressure_expected,
                name="Expected Oil Pressure",
                mode="lines",
                line=dict(
                    dash="dash"
                )
            )
        )

    f2.add_trace(
        go.Scatter(
            x=d.t,
            y=d.vibration * 100,
            name="Observed Vibration ×100",
            mode="lines"
        )
    )

    if "vibration_expected" in d.columns:

        f2.add_trace(
            go.Scatter(
                x=d.t,
                y=d.vibration_expected * 100,
                name="Expected Vibration ×100",
                mode="lines",
                line=dict(
                    dash="dash"
                )
            )
        )

    f2.update_layout(
        title="Digital Twin: Lubrication / Mechanical State",
        xaxis_title="Mission Time",
        yaxis_title="Sensor Value / Scaled Vibration"
    )

    st.write("### 🧊 3D Engine Operating State")

fig_3d = go.Figure()

fig_3d.add_trace(
    go.Scatter3d(
        x=d["rpm"],
        y=d["cht"],
        z=d["egt"],
        mode="markers+lines",

        marker=dict(
            size=5,
            color=d["anomaly_score"],
            colorscale="Turbo",
            showscale=True,
            colorbar=dict(
                title="AI Anomaly"
            )
        ),

        line=dict(
            width=2
        ),

        text=[
            f"Time: {t}<br>"
            f"RPM: {rpm:.0f}<br>"
            f"CHT: {cht:.1f} °C<br>"
            f"EGT: {egt:.0f} °C<br>"
            f"Health: {health:.1f}%"
            for t, rpm, cht, egt, health
            in zip(
                d["t"],
                d["rpm"],
                d["cht"],
                d["egt"],
                d["overall_health"] * 100
            )
        ],

        hovertemplate="%{text}<extra></extra>",

        name="Engine State"
    )
)

fig_3d.update_layout(
    title="3D Engine Operating State",

    scene=dict(
        xaxis_title="RPM",
        yaxis_title="CHT (°C)",
        zaxis_title="EGT (°C)",
        bgcolor="rgba(0,0,0,0)"
    ),

    height=650,

    margin=dict(
        l=0,
        r=0,
        t=50,
        b=0
    )
)

st.plotly_chart(
    fig_3d,
    width="stretch"
)


# =====================================================================
# TAB 2 — AI DIAGNOSTICS
# =====================================================================

with t2:
    if source_name.startswith("Simulated: Normal"):

        st.success(
            "✓ No significant anomaly detected — Normal operating condition."
        )

    elif source_name.startswith("Simulated: Overheating / Thermal Degradation"):

        st.error(
            f"⚠️ THERMAL ANOMALY DETECTED — {fault}"
        )

    elif source_name.startswith("Simulated: Mechanical / Vibration"):

        st.error(
            f"⚠️ MECHANICAL / VIBRATION ANOMALY DETECTED — {fault}"
        )

    elif source_name.startswith("Simulated: Lubrication Degradation"):

        st.error(
            f"⚠️ LUBRICATION ANOMALY DETECTED — {fault}"
        )

    elif source_name.startswith("Simulated: Injector / Fuel Abnormality"):

        st.error(
            f"⚠️ FUEL / INJECTOR ANOMALY DETECTED — {fault}"
        )

    elif source_name.startswith("Simulated: Electrical Degradation"):

        st.error(
            f"⚠️ ELECTRICAL ANOMALY DETECTED — {fault}"
        )

    elif source_name.startswith("Simulated: Sensor Drift"):

        st.error(
            f"⚠️ SENSOR DRIFT DETECTED — {fault}"
        )

    elif bool(r.anomaly):

        st.error(
            f"⚠️ ANOMALY DETECTED — {fault}"
    )

    else:

        st.success(
            "✓ No significant multivariate anomaly at current state."
        )




     

    
    a, b, c = st.columns(3)

    a.metric(
        "Diagnosis",
        fault
    )

    b.metric(
        "Severity",
        severity
    )

    c.metric(
        "Confidence",
        f"{conf * 100:.0f}%"
    )

    st.write(
        "### 🤖 AI Anomaly Detection Components"
    )

    score_cols = [
        "isolation_score",
        "elliptic_score",
        "mahalanobis_score",
        "residual_score",
        "anomaly_score",
    ]

    score_names = {
        "isolation_score": "Isolation Forest",
        "elliptic_score": "Elliptic Envelope",
        "mahalanobis_score": "Robust Mahalanobis",
        "residual_score": "Digital Twin Residual",
        "anomaly_score": "Combined AI Score",
    }

    score_data = []

    for column in score_cols:

        if column in r.index:

            try:
                score = float(r[column])
            except (TypeError, ValueError):
                continue

            score_data.append(
                {
                    "AI Component": score_names.get(
                        column,
                        column
                    ),
                    "Score": round(
                        score,
                        3
                    )
                }
            )

    if score_data:

        st.dataframe(
            pd.DataFrame(score_data),
            hide_index=True,
            width="stretch"
        )

    st.write(
        "### 📈 Anomaly Trend"
    )

    anomaly_fig = go.Figure()

    if "anomaly_score" in d.columns:

        anomaly_fig.add_trace(
            go.Scatter(
                x=d.t,
                y=d.anomaly_score,
                name="Combined Anomaly Score"
            )
        )

    anomaly_fig.add_hline(
        y=0.65,
        line_dash="dash",
        annotation_text="AI anomaly reference"
    )

    anomaly_fig.update_layout(
        title="AeroTwin AI Anomaly Score",
        xaxis_title="Mission Time",
        yaxis_title="Anomaly Score"
    )

    st.plotly_chart(
        anomaly_fig,
        width="stretch"
    )

    if "anomaly_persistence" in d.columns:

        persistence_fig = go.Figure()

        persistence_fig.add_trace(
            go.Scatter(
                x=d.t,
                y=d.anomaly_persistence,
                name="Anomaly Persistence"
            )
        )

        persistence_fig.update_layout(
            title="Persistent Anomaly Detection",
            xaxis_title="Mission Time",
            yaxis_title="Persistence"
        )

        st.plotly_chart(
            persistence_fig,
            width="stretch"
        )

    st.write(
        "**Top contributing parameters:**"
    )

    contributor_df = pd.DataFrame(
        contributors,
        columns=[
            "Parameter",
            "Deviation"
        ]
    )

    if not contributor_df.empty:

        contributor_df["Deviation"] = (
            pd.to_numeric(
                contributor_df["Deviation"],
                errors="coerce"
            )
            .round(2)
        )

    st.dataframe(
        contributor_df,
        hide_index=True,
        width="stretch"
    )

    st.caption(
        "Deviation is normalized against the Digital Twin "
        "expected healthy state. 0 = expected behaviour; "
        "larger values indicate stronger deviation."
    )

    st.write(
        "### 🧠 Diagnosis Timeline"
    )

    diagnosis_columns = [
        "t",
        "fault",
        "severity",
        "fault_confidence",
        "anomaly_score"
    ]

    diagnosis_columns = [
        column
        for column in diagnosis_columns
        if column in d.columns
    ]

    diagnosis_history = d[
        diagnosis_columns
    ].tail(40)

    st.dataframe(
        diagnosis_history,
        hide_index=True,
        width="stretch"
    )

    st.divider()

    st.write(
        "### ⚖️ Baseline Threshold vs AeroTwin AI"
    )

    if (
        ai_time is not None
        and threshold_time is not None
    ):

        lead_time = (
            threshold_time
            - ai_time
        )

        a, b, c = st.columns(3)

        a.metric(
            "AeroTwin AI Detection",
            f"{ai_time:.0f}"
        )

        b.metric(
            "Baseline Threshold Alarm",
            f"{threshold_time:.0f}"
        )

        if lead_time > 0:

            c.metric(
                "Detection Difference",
                f"{lead_time:.0f} steps"
            )

        elif lead_time < 0:

            c.metric(
                "Detection Difference",
                f"{abs(lead_time):.0f} steps"
            )

        else:

            c.metric(
                "Detection Difference",
                "Same time"
            )

        st.caption(
            "The dashboard displays the detection timing produced "
            "by the prototype AI and fixed-threshold logic. "
            "Threshold values are illustrative prototype values."
        )

        if threshold_parameter:

            st.write(
                f"**Threshold trigger:** {threshold_parameter}"
            )

    elif ai_time is not None:

        st.metric(
            "AeroTwin AI Detection",
            f"{ai_time:.0f}"
        )

        st.info(
            "AeroTwin detected a sustained anomaly, but no "
            "prototype baseline threshold was crossed during this run."
        )

    elif threshold_time is not None:

        st.metric(
            "Baseline Threshold Alarm",
            f"{threshold_time:.0f}"
        )

        st.info(
            "A prototype threshold was crossed, but AeroTwin "
            "did not confirm a sustained multivariate anomaly "
            "during this run."
        )

    else:

        st.info(
            "Neither method detected a sustained abnormal "
            "condition during this mission."
        )


# =====================================================================
# TAB 3 — HEALTH & RUL
# =====================================================================

with t3:

    a, b = st.columns(2)

    try:
        rul_low = float(rr[0])
        rul_high = float(rr[1])
    except (TypeError, ValueError, IndexError):
        rul_low = 0.0
        rul_high = 0.0

    a.metric(
        "Estimated RUL",
        f"{rul_low:.1f}–{rul_high:.1f} h"
    )

    b.metric(
        "RUL Confidence",
        f"{rc * 100:.0f}%"
    )

    st.caption(
        "RUL is expressed in equivalent operating hours using "
        "the prototype accelerated-time scale "
        "(60 simulation samples = 1 hour)."
    )

    f = go.Figure()

    f.add_trace(
        go.Scatter(
            x=d.t,
            y=d.overall_health * 100,
            name="Overall Health"
        )
    )

    f.add_hline(
        y=40,
        annotation_text="Prototype EOL criterion"
    )

    f.add_hline(
        y=70,
        line_dash="dash",
        annotation_text="Warning / degradation reference"
    )

    f.update_layout(
        title="Degradation Trajectory",
        xaxis_title="Mission Time",
        yaxis_title="Health (%)"
    )

    st.plotly_chart(
        f,
        width="stretch"
    )

    st.write(
        "### Subsystem Health"
    )

    subsystem_df = pd.DataFrame(
        {
            "Subsystem": [
                "Thermal",
                "Mechanical",
                "Lubrication",
                "Fuel",
                "Electrical"
            ],
            "Health %": [
                r.thermal_health * 100,
                r.mechanical_health * 100,
                r.lubrication_health * 100,
                r.fuel_health * 100,
                r.electrical_health * 100
            ]
        }
    )

    st.bar_chart(
        subsystem_df.set_index(
            "Subsystem"
        )
    )

    health_fig = go.Figure()

    health_columns = [
        (
            "thermal_health",
            "Thermal"
        ),
        (
            "mechanical_health",
            "Mechanical"
        ),
        (
            "lubrication_health",
            "Lubrication"
        ),
        (
            "fuel_health",
            "Fuel"
        ),
        (
            "electrical_health",
            "Electrical"
        )
    ]

    for column, name in health_columns:

        if column in d.columns:

            health_fig.add_trace(
                go.Scatter(
                    x=d.t,
                    y=d[column] * 100,
                    name=name
                )
            )

    health_fig.update_layout(
        title="Subsystem Health Trends",
        xaxis_title="Mission Time",
        yaxis_title="Health (%)"
    )

    st.plotly_chart(
        health_fig,
        width="stretch"
    )

    if r.overall_health < 0.70:

        st.warning(
            "🔧 Maintenance advisory: investigate the dominant "
            "degraded subsystem before the next defined "
            "maintenance window."
        )

    else:

        st.info(
            "✓ No immediate maintenance escalation from "
            "prototype decision logic."
        )


# =====================================================================
# TAB 4 — REPLAY / WHAT-IF
# =====================================================================

with t4:

    st.write(f"### 🛰 Mission Replay — tick {st.session_state.replay_idx + 1} of {n_total}")
    st.progress((st.session_state.replay_idx + 1) / n_total)

    st.write("#### 🕒 Fault / Event Timeline")

    timeline_events = build_event_timeline(d)

    if not timeline_events:
        st.caption("No status or fault transitions yet in the replayed portion of the mission.")
    else:
        icon_map = {"status": "🔄", "fault": "🔧", "anomaly": "⚠️"}
        for ev in reversed(timeline_events[-25:]):
            st.markdown(f"{icon_map.get(ev['type'], '⚪')} `t={ev['t']:.0f}` — {ev['text']}")

    st.divider()

    st.write(
        "### Mission telemetry / replay"
    )

    replay_columns = [
        "t",
        "mission_phase",
        "throttle",
        "altitude_ft",
        "ambient_temp",
        "overall_health",
        "anomaly",
        "anomaly_score",
        "fault",
        "severity"
    ]

    replay_columns = [
        column
        for column in replay_columns
        if column in d.columns
    ]

    st.dataframe(
        d[replay_columns].tail(40),
        hide_index=True,
        width="stretch"
    )

    st.divider()

    st.write("### What-if Simulation")

    if wi:

        try:

            what_if_raw = simulate(
                "Normal",
                120,
                7,
                throttle=th,
                altitude=alt,
                ambient=temp
            )

            w = process(
                what_if_raw
            )

            wr = w.iloc[-1]

            what_if_result = diagnose(wr)

            wf = what_if_result[0]
            ws = what_if_result[1]
            wc = what_if_result[2]

            if len(what_if_result) >= 4:
                wcontributors = what_if_result[3]
            else:
                wcontributors = []

            a, b, c = st.columns(3)

            a.metric(
                "Projected CHT",
                f"{wr.cht:.1f} °C"
            )

            b.metric(
                "Projected EGT",
                f"{wr.egt:.0f} °C"
            )

            c.metric(
                "Projected Health",
                f"{wr.overall_health * 100:.0f}%"
            )

            st.write(
                f"**Projected condition:** {wf} • "
                f"confidence {wc * 100:.0f}%"
            )

            st.write(
                "**Projected top contributors:**"
            )

            what_if_contributors = pd.DataFrame(
                wcontributors,
                columns=[
                    "Parameter",
                    "Deviation"
                ]
            )

            if not what_if_contributors.empty:

                what_if_contributors["Deviation"] = (
                    pd.to_numeric(
                        what_if_contributors["Deviation"],
                        errors="coerce"
                    )
                    .round(2)
                )

            st.dataframe(
                what_if_contributors,
                hide_index=True,
                width="stretch"
            )

            f = go.Figure()

            f.add_trace(
                go.Scatter(
                    x=w.t,
                    y=w.overall_health * 100,
                    name="What-if health"
                )
            )

            f.update_layout(
                title="What-if Projected Health",
                xaxis_title="Mission Time",
                yaxis_title="Health (%)"
            )

            st.plotly_chart(
                f,
                width="stretch"
            )

            if "anomaly_score" in w.columns:

                wf_anomaly = go.Figure()

                wf_anomaly.add_trace(
                    go.Scatter(
                        x=w.t,
                        y=w.anomaly_score,
                        name="What-if AI anomaly"
                    )
                )

                wf_anomaly.update_layout(
                    title="What-if AI Anomaly Score",
                    xaxis_title="Mission Time",
                    yaxis_title="Anomaly Score"
                )

                st.plotly_chart(
                    wf_anomaly,
                    width="stretch"
                )

        except Exception as e:

            st.error(
                f"What-if simulation error: {e}"
            )


# =====================================================================
# TAB 5 — EXTENDED SYSTEMS
# =====================================================================

with t5:

    EXTENDED_PARAMS = [
        "engine_load",
        "torque",
        "power_output",
        "engine_efficiency",
        "manifold_pressure",
        "intake_air_temp",
        "air_mass_flow",
        "fuel_pressure",
        "air_fuel_ratio",
        "ignition_timing",
        "vibration_frequency",
        "battery_current",
        "alternator_voltage",
        "alternator_current",
        "electrical_power",
        "alternator_power",
        "fuel_consumed",
        "fuel_level",
        "degradation_stage",
        "fault_ramp"
    ]

    available_extended = [
        parameter
        for parameter in EXTENDED_PARAMS
        if parameter in d.columns
    ]

    has_extended = len(
        available_extended
    ) > 0

    if not has_extended:

        st.info(
            "These extended engine parameters are produced by "
            "the built-in simulator. Uploaded CSVs may only "
            "contain the original telemetry schema."
        )

    else:

        if "degradation_stage" in d.columns:

            stage_icon = {
                "HEALTHY": "🟢",
                "FAULT START": "🟡",
                "MILD": "🟠",
                "MODERATE": "🟠",
                "SEVERE": "🔴",
            }.get(
                str(r.degradation_stage),
                "⚪"
            )

            fault_ramp = (
                float(r.fault_ramp) * 100
                if "fault_ramp" in d.columns
                else 0.0
            )

            st.caption(
                f"{stage_icon} Degradation stage: "
                f"**{r.degradation_stage}** "
                f"(fault progress: {fault_ramp:.0f}%)"
            )

        st.write(
            "#### Engine Performance"
        )

        performance_params = [
            "engine_load",
            "torque",
            "power_output",
            "engine_efficiency"
        ]

        performance_params = [
            p
            for p in performance_params
            if p in d.columns
        ]

        if performance_params:

            columns = st.columns(
                len(performance_params)
            )

            labels = {
                "engine_load": "Engine Load",
                "torque": "Torque",
                "power_output": "Power Output",
                "engine_efficiency": "Engine Efficiency"
            }

            for column, parameter in zip(
                columns,
                performance_params
            ):

                value = r[parameter]

                if parameter in [
                    "engine_load",
                    "engine_efficiency"
                ]:

                    text = f"{float(value) * 100:.0f}%"

                elif parameter == "torque":

                    text = f"{float(value):.0f} N·m"

                elif parameter == "power_output":

                    text = f"{float(value):.1f} kW"

                else:

                    text = f"{float(value):.2f}"

                column.metric(
                    labels.get(
                        parameter,
                        parameter
                    ),
                    text
                )

        if (
            "power_output" in d.columns
            or "torque" in d.columns
        ):

            f = go.Figure()

            if "power_output" in d.columns:

                f.add_trace(
                    go.Scatter(
                        x=d.t,
                        y=d.power_output,
                        name="Power Output (kW)"
                    )
                )

            if "torque" in d.columns:

                f.add_trace(
                    go.Scatter(
                        x=d.t,
                        y=d.torque / 10,
                        name="Torque (N·m ÷10)"
                    )
                )

            f.update_layout(
                title="Performance Trend",
                xaxis_title="Mission Time",
                yaxis_title="kW / scaled N·m"
            )

            st.plotly_chart(
                f,
                width="stretch"
            )

        st.write(
            "#### Air & Fuel System"
        )

        air_fuel_params = [
            "manifold_pressure",
            "intake_air_temp",
            "air_mass_flow",
            "fuel_pressure"
        ]

        air_fuel_params = [
            p
            for p in air_fuel_params
            if p in d.columns
        ]

        if air_fuel_params:

            columns = st.columns(
                len(air_fuel_params)
            )

            labels = {
                "manifold_pressure": "Manifold Pressure",
                "intake_air_temp": "Intake Air Temp",
                "air_mass_flow": "Air Mass Flow",
                "fuel_pressure": "Fuel Pressure"
            }

            units = {
                "manifold_pressure": " kPa",
                "intake_air_temp": " °C",
                "air_mass_flow": " g/s",
                "fuel_pressure": " kPa"
            }

            for column, parameter in zip(
                columns,
                air_fuel_params
            ):

                value = float(
                    r[parameter]
                )

                column.metric(
                    labels[parameter],
                    f"{value:.1f}{units[parameter]}"
                )

        if "air_fuel_ratio" in d.columns:

            f = go.Figure()

            f.add_trace(
                go.Scatter(
                    x=d.t,
                    y=d.air_fuel_ratio,
                    name="Air-Fuel Ratio"
                )
            )

            f.add_hline(
                y=14.7,
                line_dash="dash",
                annotation_text="~Stoichiometric reference"
            )

            f.update_layout(
                title="Air-Fuel Ratio",
                xaxis_title="Mission Time",
                yaxis_title="AFR"
            )

            st.plotly_chart(
                f,
                width="stretch"
            )

        st.write(
            "#### Mechanical"
        )

        mechanical_params = [
            "vibration_frequency",
            "ignition_timing"
        ]

        mechanical_params = [
            p
            for p in mechanical_params
            if p in d.columns
        ]

        if mechanical_params:

            columns = st.columns(
                len(mechanical_params)
            )

            for column, parameter in zip(
                columns,
                mechanical_params
            ):

                if parameter == "vibration_frequency":

                    value = (
                        f"{float(r[parameter]):.0f} Hz"
                    )

                    label = "Vibration Frequency"

                else:

                    value = (
                        f"{float(r[parameter]):.1f}° BTDC"
                    )

                    label = "Ignition Timing"

                column.metric(
                    label,
                    value
                )

        if "vibration_frequency" in d.columns:

            f = go.Figure()

            f.add_trace(
                go.Scatter(
                    x=d.t,
                    y=d.vibration_frequency,
                    name="Vibration Frequency (Hz)"
                )
            )

            f.update_layout(
                title="Vibration Frequency",
                xaxis_title="Mission Time",
                yaxis_title="Hz"
            )

            st.plotly_chart(
                f,
                width="stretch"
            )

        st.write(
            "#### Electrical System"
        )

        electrical_params = [
            "battery_current",
            "alternator_voltage",
            "alternator_current",
            "electrical_power",
            "alternator_power"
        ]

        electrical_params = [
            p
            for p in electrical_params
            if p in d.columns
        ]

        if electrical_params:

            columns = st.columns(
                len(electrical_params)
            )

            labels = {
                "battery_current": "Battery Current",
                "alternator_voltage": "Alternator Voltage",
                "alternator_current": "Alternator Current",
                "electrical_power": "Electrical Power",
                "alternator_power": "Alternator Power"
            }

            units = {
                "battery_current": " A",
                "alternator_voltage": " V",
                "alternator_current": " A",
                "electrical_power": " W",
                "alternator_power": " W"
            }

            for column, parameter in zip(
                columns,
                electrical_params
            ):

                column.metric(
                    labels[parameter],
                    f"{float(r[parameter]):.1f}"
                    f"{units[parameter]}"
                )

        if (
            "battery_voltage" in d.columns
            or "alternator_voltage" in d.columns
        ):

            f = go.Figure()

            if "battery_voltage" in d.columns:

                f.add_trace(
                    go.Scatter(
                        x=d.t,
                        y=d.battery_voltage,
                        name="Battery Voltage (V)"
                    )
                )

            if "alternator_voltage" in d.columns:

                f.add_trace(
                    go.Scatter(
                        x=d.t,
                        y=d.alternator_voltage,
                        name="Alternator Voltage (V)"
                    )
                )

            f.update_layout(
                title="Electrical Bus Voltage",
                xaxis_title="Mission Time",
                yaxis_title="Volts"
            )

            st.plotly_chart(
                f,
                width="stretch"
            )

        st.write(
            "#### Fuel & Mission State"
        )

        fuel_params = [
            "fuel_consumed",
            "fuel_level"
        ]

        fuel_params = [
            p
            for p in fuel_params
            if p in d.columns
        ]

        if fuel_params:

            columns = st.columns(
                len(fuel_params)
            )

            labels = {
                "fuel_consumed": "Fuel Consumed",
                "fuel_level": "Fuel Remaining"
            }

            for column, parameter in zip(
                columns,
                fuel_params
            ):

                column.metric(
                    labels[parameter],
                    f"{float(r[parameter]):.1f} L"
                )

        if "fuel_level" in d.columns:

            f = go.Figure()

            f.add_trace(
                go.Scatter(
                    x=d.t,
                    y=d.fuel_level,
                    name="Fuel Remaining (L)"
                )
            )

            f.update_layout(
                title="Fuel Depletion",
                xaxis_title="Mission Time",
                yaxis_title="Litres"
            )

            st.plotly_chart(
                f,
                width="stretch"
            )

        with st.expander(
            "Show full telemetry table (every parameter)"
        ):

            st.dataframe(
                d,
                hide_index=True,
                width="stretch"
            )


# =====================================================================
# FOOTER
# =====================================================================

st.divider()

st.caption(
    "Prototype uses synthetic telemetry when simulation mode "
    "is selected. Uploaded CSVs are processed locally in the "
    "running application. RUL is model-estimated against a "
    "defined synthetic EOL criterion and is not certified real-engine "
    "life prediction."
)

st.caption(
    "AeroTwin AI pipeline: Digital Twin residuals → temporal and "
    "physical features → Isolation Forest + Elliptic Envelope + "
    "Robust Mahalanobis → persistent anomaly detection → subsystem "
    "health → hybrid diagnosis → RUL."
)


# =====================================================================
# REPLAY AUTO-ADVANCE
#
# Runs after everything above has rendered the current tick. If
# playing and not yet at the end of the mission, wait, advance one
# tick, and rerun the whole script — which re-slices `d` further up
# and redraws every tab above with the new state.
# =====================================================================

if st.session_state.get("replay_playing", False):
    if st.session_state.replay_idx < n_total - 1:
        time.sleep(1.0 / st.session_state.replay_speed)
        st.session_state.replay_idx += 1
        st.rerun()
    else:
        st.session_state.replay_playing = False