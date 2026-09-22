import io
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from simulator import simulate, expected
from analytics import (
    add_twin_features,
    health,
    fit_detector,
    detect,
    diagnose,
    rul,
    threshold_vs_ai
)

st.set_page_config(page_title='AeroTwin | SIH26054', page_icon='✈️', layout='wide')
st.title('✈️ AeroTwin')
st.caption('AI-assisted Digital Twin for MALE UAV Aero-Piston Engine Health • SIH26054')

SCENARIOS = [
    'Normal',
    'Overheating / Thermal Degradation',
    'Lubrication Degradation',
    'Injector / Fuel Abnormality',
    'Mechanical / Vibration',
    'Sensor Drift',
]

REQUIRED_COLUMNS = [
    't', 'mission_phase', 'throttle', 'altitude_ft', 'ambient_temp',
    'rpm', 'cht', 'egt', 'oil_pressure', 'oil_temp', 'fuel_flow',
    'vibration', 'battery_voltage', 'alternator_health', 'injection_timing'
]

st.markdown('''
**Prototype flow:** CSV / simulated telemetry → Digital Twin → health monitoring → AI anomaly detection → fault diagnosis → RUL → maintenance advisory.
''')

@st.cache_resource
def detector():
    return fit_detector()

sc, model = detector()

with st.sidebar:
    st.header('Data Source')
    source = st.radio('Choose input', ['Upload CSV', 'Built-in Simulation'])

    if source == 'Upload CSV':
        uploaded = st.file_uploader('Upload engine telemetry CSV', type=['csv'])
        st.caption('Use the required column names listed below. A sample CSV can be downloaded from the main page.')
    else:
        scenario = st.selectbox('Simulation scenario', SCENARIOS)
        duration = st.slider('Mission duration', 60, 300, 180, 10)
        seed = st.number_input('Seed', 1, 9999, 42)
        run = st.button('▶ Run Mission', width='stretch')

    st.divider()
    st.subheader('What-if Simulation')
    th = st.slider('Throttle', .30, 1.00, .85, .01)
    alt = st.slider('Altitude (ft)', 0, 15000, 12000, 500)
    temp = st.slider('Ambient temperature (°C)', 0, 50, 40, 1)
    wi = st.button('Run What-if', width='stretch')


def process(df):
    df = df.copy()
    # Normalize common alternative timestamp/time column names if present.
    if 'time' in df.columns and 't' not in df.columns:
        df['t'] = df['time']
    if 'altitude' in df.columns and 'altitude_ft' not in df.columns:
        df['altitude_ft'] = df['altitude']
    if 'ambient_temperature' in df.columns and 'ambient_temp' not in df.columns:
        df['ambient_temp'] = df['ambient_temperature']

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(missing))

    # Keep only required + useful metadata, and coerce numerical telemetry.
    numeric_cols = [c for c in REQUIRED_COLUMNS if c not in ['mission_phase']]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=numeric_cols).reset_index(drop=True)
    if len(df) < 20:
        raise ValueError('Please provide at least 20 valid telemetry rows.')

    d = add_twin_features(df)

    # Calculate the healthy Digital Twin expected state
    # for every telemetry sample.
    expected_states = d.apply(
        lambda row: expected(row),
        axis=1,
        result_type='expand'
    )

    expected_states.index = d.index

    for sensor in [
        'rpm',
        'cht',
        'egt',
        'oil_pressure',
        'oil_temp',
        'fuel_flow',
        'vibration',
        'battery_voltage',
        'alternator_health',
        'injection_timing'
    ]:
        d[sensor + '_expected'] = expected_states[sensor]

    d = health(d)
    d = detect(d, sc, model)
    diagnoses = [diagnose(r) for _, r in d.iterrows()]
    d['fault'] = [x[0] for x in diagnoses]
    d['severity'] = [x[1] for x in diagnoses]
    d['fault_confidence'] = [x[2] for x in diagnoses]
    return d


def make_sample_csv():
    sample = simulate('Overheating / Thermal Degradation', 180, 42)
    return sample.to_csv(index=False).encode('utf-8')

if source == 'Upload CSV':
    st.download_button(
        '⬇ Download sample telemetry CSV',
        data=make_sample_csv(),
        file_name='aerotwin_sample_telemetry.csv',
        mime='text/csv',
        width='stretch'
    )

    if uploaded is not None:
        try:
            raw_upload = pd.read_csv(uploaded)
            st.session_state.df = process(raw_upload)
            st.session_state.source_name = uploaded.name
            st.success(f'Loaded and analysed {uploaded.name} — {len(st.session_state.df)} telemetry rows.')
        except Exception as e:
            st.error(f'Could not process this CSV: {e}')
            st.stop()
else:
    if run:
        try:
            st.session_state.df = process(simulate(scenario, duration, int(seed)))
            st.session_state.source_name = f'Simulated: {scenario}'
            st.success('Mission simulated and analysed.')
        except Exception as e:
            st.error(str(e))
            st.stop()

if 'df' not in st.session_state:
    st.info('Upload a telemetry CSV or select Built-in Simulation and click Run Mission.')
    st.markdown('''
### Required CSV schema

```text
t, mission_phase, throttle, altitude_ft, ambient_temp,
rpm, cht, egt, oil_pressure, oil_temp, fuel_flow,
vibration, battery_voltage, alternator_health, injection_timing
```

**Meaning:** each row is one engine telemetry sample. The Digital Twin uses throttle, altitude and ambient temperature to estimate expected healthy behaviour, then compares observed sensor values against that expected state.
''')
    st.stop()

d = st.session_state.df
r = d.iloc[-1]

fault, severity, conf, contributors = diagnose(r)
rr, rc = rul(d)

threshold_idx, ai_idx, threshold_time, ai_time, threshold_parameter = threshold_vs_ai(d)
icon = {'HEALTHY': '🟢', 'DEGRADING': '🟡', 'WARNING': '🟠', 'CRITICAL': '🔴'}.get(r.status, '⚪')

st.subheader(f'{icon} Engine: {r.status}  |  Mission phase: {r.mission_phase}')
st.caption(f'Data source: {st.session_state.get("source_name", "Unknown")}')

c = st.columns(5)
c[0].metric('Overall Health', f'{r.overall_health*100:.0f}%')
c[1].metric('RPM', f'{r.rpm:.0f}')
c[2].metric('CHT', f'{r.cht:.1f} °C')
c[3].metric('EGT', f'{r.egt:.0f} °C')
c[4].metric('Twin Fit', f'{r.twin_fit*100:.0f}%')

t1, t2, t3, t4 = st.tabs(['📡 Telemetry', '🧠 AI Diagnostics', '⏳ Health & RUL', '🛰 Replay / What-if'])

with t1:
    f = go.Figure()
    f.add_trace(go.Scatter(x=d.t, y=d.cht, name='CHT'))
    f.add_trace(go.Scatter(x=d.t, y=d.egt, name='EGT'))
    f.update_layout(title='Thermal Telemetry', xaxis_title='Mission Time', yaxis_title='Temperature')
    st.plotly_chart(f, width='stretch')
    st.write('### 🔄 Digital Twin: Expected vs Observed')

    twin_fig = go.Figure()

    twin_fig.add_trace(
        go.Scatter(
            x=d.t,
            y=d.cht,
            name='Observed CHT',
            mode='lines'
        )
    )

    twin_fig.add_trace(
        go.Scatter(
            x=d.t,
            y=d.cht_expected,
            name='Expected CHT',
            mode='lines',
            line=dict(dash='dash')
        )
    )

    twin_fig.add_trace(
        go.Scatter(
            x=d.t,
            y=d.egt,
            name='Observed EGT',
            mode='lines'
        )
    )

    twin_fig.add_trace(
        go.Scatter(
            x=d.t,
            y=d.egt_expected,
            name='Expected EGT',
            mode='lines',
            line=dict(dash='dash')
        )
    )

    twin_fig.update_layout(
        title='Digital Twin Thermal State',
        xaxis_title='Mission Time',
        yaxis_title='Temperature (°C)'
    )

    st.plotly_chart(twin_fig, width='stretch')

    f2 = go.Figure()

    # Oil pressure
    f2.add_trace(
        go.Scatter(
            x=d.t,
            y=d.oil_pressure,
            name='Observed Oil Pressure',
            mode='lines'
        )
    )

    f2.add_trace(
        go.Scatter(
            x=d.t,
            y=d.oil_pressure_expected,
            name='Expected Oil Pressure',
            mode='lines',
            line=dict(dash='dash')
        )
    )

    # Vibration is scaled ×100 only for visualization.
    f2.add_trace(
        go.Scatter(
            x=d.t,
            y=d.vibration * 100,
            name='Observed Vibration ×100',
            mode='lines'
        )
    )

    f2.add_trace(
        go.Scatter(
            x=d.t,
            y=d.vibration_expected * 100,
            name='Expected Vibration ×100',
            mode='lines',
            line=dict(dash='dash')
        )
    )

    f2.update_layout(
        title='Digital Twin: Lubrication / Mechanical State',
        xaxis_title='Mission Time',
        yaxis_title='Sensor Value / Scaled Vibration'
    )

    st.plotly_chart(f2, width='stretch')

with t2:
    if r.anomaly:
        st.error(f'⚠️ ANOMALY DETECTED — {fault}')
    else:
        st.success('✓ No significant multivariate anomaly at current state.')

    a, b, c = st.columns(3)
    a.metric('Diagnosis', fault)
    b.metric('Severity', severity)
    c.metric('Confidence', f'{conf*100:.0f}%')

    st.write('**Top contributing parameters:**')

    contributor_df = pd.DataFrame(
        contributors,
        columns=['Parameter', 'Deviation']
    )

    contributor_df['Deviation'] = contributor_df['Deviation'].round(2)

    st.dataframe(
        contributor_df,
        hide_index=True,
        width='stretch'
    )

    st.caption(
        'Deviation is normalized against the Digital Twin expected healthy state. '
        '0 = expected behaviour; larger values indicate stronger deviation.'
    )

    st.divider()

    st.write('### ⚖️ Baseline Threshold vs AeroTwin AI')

    if ai_time is not None and threshold_time is not None:

        lead_time = threshold_time - ai_time

        a, b, c = st.columns(3)

        a.metric(
            'AeroTwin AI Detection',
            f'{ai_time:.0f}'
        )

        b.metric(
            'Baseline Threshold Alarm',
            f'{threshold_time:.0f}'
        )

        if lead_time > 0:
            c.metric(
                'Earlier Detection',
                f'{lead_time:.0f} steps'
            )
        elif lead_time < 0:
            c.metric(
                'Detection Difference',
                f'{abs(lead_time):.0f} steps later'
            )
        else:
            c.metric(
                'Detection Difference',
                'Same time'
            )

        st.caption(
            'Both methods require 3 consecutive abnormal samples. '
            'Threshold limits are illustrative prototype warning values; '
            'they are not certified engine operating limits.'
        )
        if threshold_parameter:
            st.write(
                f'**Threshold trigger:** {threshold_parameter}'
            )

    elif ai_time is not None:

        st.metric(
            'AeroTwin AI Detection',
            f'{ai_time:.0f}'
        )

        st.info(
            'AeroTwin detected a sustained anomaly, but no prototype '
            'baseline threshold was crossed during this run.'
        )

    elif threshold_time is not None:

        st.metric(
            'Baseline Threshold Alarm',
            f'{threshold_time:.0f}'
        )

        st.info(
            'A prototype threshold was crossed, but AeroTwin did not '
            'confirm a sustained multivariate anomaly during this run.'
        )

    else:

        st.info(
            'Neither method detected a sustained abnormal condition '
            'during this mission.'
        )

with t3:
    a, b = st.columns(2)
    a.metric('Estimated RUL', f'{rr[0]:.1f}–{rr[1]:.1f} h')
    b.metric('RUL confidence', f'{rc*100:.0f}%')
    st.caption(
        'RUL is expressed in equivalent operating hours using the prototype '
        'accelerated-time scale (60 simulation samples = 1 hour).'
    )

    f = go.Figure()
    f.add_trace(go.Scatter(x=d.t, y=d.overall_health * 100, name='Health'))
    f.add_hline(y=40, annotation_text='Prototype EOL criterion')
    f.update_layout(title='Degradation Trajectory', yaxis_title='Health (%)')
    st.plotly_chart(f, width='stretch')

    st.bar_chart(
    pd.DataFrame({
        'Subsystem': ['Thermal', 'Mechanical', 'Lubrication', 'Fuel', 'Electrical'],
        'Health %': [
            r.thermal_health * 100,
            r.mechanical_health * 100,
            r.lubrication_health * 100,
            r.fuel_health * 100,
            r.electrical_health * 100
            ]
        })
    )

    if r.overall_health < .7:
        st.warning('🔧 Maintenance advisory: investigate the dominant degraded subsystem before the next defined maintenance window.')
    else:
        st.info('✓ No immediate maintenance escalation from prototype decision logic.')

with t4:
    st.write('### Mission telemetry / replay')
    st.dataframe(
        d[['t', 'mission_phase', 'throttle', 'altitude_ft', 'ambient_temp', 'overall_health', 'anomaly', 'fault']].tail(40),
        hide_index=True,
        width='stretch'
    )

    if wi:
        w = process(simulate('Normal', 120, 7, throttle=th, altitude=alt, ambient=temp))
        wr = w.iloc[-1]
        wf, ws, wc, _ = diagnose(wr)
        a, b, c = st.columns(3)
        a.metric('Projected CHT', f'{wr.cht:.1f} °C')
        b.metric('Projected EGT', f'{wr.egt:.0f} °C')
        c.metric('Projected Health', f'{wr.overall_health*100:.0f}%')
        st.write(f'**Projected condition:** {wf} • confidence {wc*100:.0f}%')
        f = go.Figure()
        f.add_trace(go.Scatter(x=w.t, y=w.overall_health*100, name='What-if health'))
        f.update_layout(title='What-if Projected Health', yaxis_title='Health (%)')
        st.plotly_chart(f, width='stretch')

st.divider()
st.caption('Prototype uses synthetic telemetry when simulation mode is selected. Uploaded CSVs are processed locally in the running application. RUL is model-estimated against a defined synthetic EOL criterion; not certified real-engine life prediction.')
