import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from simulator import expected

SENSORS=['rpm','cht','egt','oil_pressure','oil_temp','fuel_flow','vibration','battery_voltage','alternator_health','injection_timing']
FEATURES=[x+'_dev' for x in SENSORS]

def add_twin_features(df):
    out=df.copy()
    for s in SENSORS:
        vals=[]
        for _,r in out.iterrows():
            e=expected(r)[s]; vals.append((float(r[s])-e)/max(abs(e)*.08,.1))
        out[s+'_dev']=vals
    out['twin_fit']=(1-out[FEATURES].abs().clip(upper=4).mean(axis=1)/4).clip(0,1)
    return out

def health(df):
    """
    Calculate subsystem and overall engine health from
    Digital Twin observed-vs-expected deviations.

    The Digital Twin provides *_dev features where:
        0    = observed value matches expected healthy behaviour
        +ve  = observed above expected
        -ve  = observed below expected

    Health is converted from deviation magnitude:
        small deviation  -> high health
        large deviation  -> low health
    """

    o = df.copy()

    # Smooth noisy telemetry before health calculation.
    for c in [
        'cht_dev',
        'egt_dev',
        'oil_pressure_dev',
        'oil_temp_dev',
        'fuel_flow_dev',
        'vibration_dev',
        'battery_voltage_dev',
        'alternator_health_dev',
        'injection_timing_dev'
    ]:
        o[c + '_sm'] = o[c].rolling(10, min_periods=1).mean()

    # Convert Twin deviation into a health score.
    # Deviation of 0 -> 100% health.
    # Deviation of 4 or more -> 0% health.
    def deviation_health(series):
        return (1 - series.abs() / 4).clip(0, 1)

    # ---------------------------------------------------------
    # SUBSYSTEM HEALTH
    # ---------------------------------------------------------

    # Thermal subsystem
    thermal_cht = deviation_health(o['cht_dev_sm'])
    thermal_egt = deviation_health(o['egt_dev_sm'])

    o['thermal_health'] = (
        0.55 * thermal_cht +
        0.45 * thermal_egt
    ).clip(0, 1)

    # Mechanical subsystem
    o['mechanical_health'] = deviation_health(
        o['vibration_dev_sm']
    )

    # Lubrication subsystem
    lubrication_pressure = deviation_health(
        o['oil_pressure_dev_sm']
    )

    lubrication_temperature = deviation_health(
        o['oil_temp_dev_sm']
    )

    o['lubrication_health'] = (
        0.65 * lubrication_pressure +
        0.35 * lubrication_temperature
    ).clip(0, 1)

    # Fuel / combustion subsystem
    fuel_flow_health = deviation_health(
        o['fuel_flow_dev_sm']
    )

    injection_health = deviation_health(
        o['injection_timing_dev_sm']
    )

    o['fuel_health'] = (
        0.65 * fuel_flow_health +
        0.35 * injection_health
    ).clip(0, 1)

    # Electrical subsystem
    battery_health = deviation_health(
        o['battery_voltage_dev_sm']
    )

    alternator_health = deviation_health(
        o['alternator_health_dev_sm']
    )

    o['electrical_health'] = (
        0.55 * battery_health +
        0.45 * alternator_health
    ).clip(0, 1)

    # ---------------------------------------------------------
    # OVERALL ENGINE HEALTH
    # ---------------------------------------------------------

    o['overall_health'] = (
        0.30 * o['thermal_health'] +
        0.25 * o['mechanical_health'] +
        0.20 * o['lubrication_health'] +
        0.15 * o['fuel_health'] +
        0.10 * o['electrical_health']
    ).clip(0, 1)

    # ---------------------------------------------------------
    # HEALTH STATUS
    # ---------------------------------------------------------

    o['status'] = pd.cut(
        o['overall_health'],
        [-0.01, 0.40, 0.70, 0.90, 1.01],
        labels=[
            'CRITICAL',
            'WARNING',
            'DEGRADING',
            'HEALTHY'
        ]
    ).astype(str)

    return o

def fit_detector():
    base=health(add_twin_features(__import__('simulator').simulate('Normal',900,123)))
    sc=StandardScaler().fit(base[FEATURES].fillna(0)); model=IsolationForest(n_estimators=160,contamination=.03,random_state=42).fit(sc.transform(base[FEATURES].fillna(0)))
    return sc,model

def detect(df,sc,model):
    o=df.copy(); x=sc.transform(o[FEATURES].fillna(0)); raw=-model.decision_function(x); o['anomaly_score']=((raw-raw.min())/(raw.max()-raw.min()+1e-9)).clip(0,1); o['anomaly']=model.predict(x)==-1; return o

# Prototype baseline thresholds.
# These are illustrative warning limits for the demo,
# NOT certified real-engine operating limits.

THRESHOLDS = {
    'cht': 200.0,
    'egt': 700.0,
    'oil_pressure': 43.0,
    'oil_temp': 100.0,
    'vibration': 0.28,
    'fuel_flow': 27.0,
    'injection_timing_deviation': 1.0
}


def threshold_vs_ai(df):
    """
    Compare conventional fixed-threshold monitoring
    with AeroTwin's AI anomaly detection.

    Thresholds are illustrative prototype warning values,
    not certified engine operating limits.
    """

    o = df.copy()

    # ---------------------------------------------------------
    # Fixed prototype thresholds
    # ---------------------------------------------------------

    THRESHOLDS = {
        'cht': 205.0,
        'egt': 700.0,
        'oil_pressure': 45.0,
        'oil_temp': 105.0,
        'vibration': 0.28,
        'fuel_flow': 29.0,
        'injection_timing_deviation': 1.5
    }

    # ---------------------------------------------------------
    # Individual threshold conditions
    # ---------------------------------------------------------

    conditions = {
        'CHT': o['cht'] > THRESHOLDS['cht'],
        'EGT': o['egt'] > THRESHOLDS['egt'],
        'Oil pressure': o['oil_pressure'] < THRESHOLDS['oil_pressure'],
        'Oil temperature': o['oil_temp'] > THRESHOLDS['oil_temp'],
        'Vibration': o['vibration'] > THRESHOLDS['vibration'],
        'Fuel flow': o['fuel_flow'] > THRESHOLDS['fuel_flow'],
        'Injection timing': (
            abs(o['injection_timing'] - 18.0)
            > THRESHOLDS['injection_timing_deviation']
        )
    }

    threshold_alarm = pd.DataFrame(
        conditions,
        index=o.index
    ).any(axis=1)

    # Require three consecutive abnormal samples.
    threshold_confirmed = (
        threshold_alarm
        .rolling(3, min_periods=3)
        .sum()
        >= 3
    )

    ai_confirmed = (
        o['anomaly']
        .rolling(3, min_periods=3)
        .sum()
        >= 3
    )

    # ---------------------------------------------------------
    # Find first confirmed threshold alarm
    # ---------------------------------------------------------

    threshold_matches = o.index[threshold_confirmed]

    threshold_idx = (
        int(threshold_matches[0])
        if len(threshold_matches) > 0
        else None
    )

    threshold_time = (
        float(o.loc[threshold_idx, 't'])
        if threshold_idx is not None
        else None
    )

    # Identify which sensor triggered the threshold.
    threshold_parameter = None

    if threshold_idx is not None:
        triggered = []

        for name, condition in conditions.items():
            if bool(condition.loc[threshold_idx]):
                triggered.append(name)

        if triggered:
            threshold_parameter = ', '.join(triggered)

    # ---------------------------------------------------------
    # Find first confirmed AI anomaly
    # ---------------------------------------------------------

    ai_matches = o.index[ai_confirmed]

    ai_idx = (
        int(ai_matches[0])
        if len(ai_matches) > 0
        else None
    )

    ai_time = (
        float(o.loc[ai_idx, 't'])
        if ai_idx is not None
        else None
    )

    return (
        threshold_idx,
        ai_idx,
        threshold_time,
        ai_time,
        threshold_parameter
    )
def diagnose(r):
    """
    Hybrid fault diagnosis.

    Uses:
    1. Digital Twin deviations
    2. Multivariate AI anomaly score
    3. Engineering relationships between sensors

    The result is a probable fault, not a certified diagnosis.
    """

    # ---------------------------------------------------------
    # Calculate evidence scores for each fault category
    # ---------------------------------------------------------

    thermal_evidence = np.mean([
        abs(r.cht_dev),
        abs(r.egt_dev),
        abs(r.oil_temp_dev)
    ])

    lubrication_evidence = np.mean([
        abs(r.oil_pressure_dev),
        abs(r.oil_temp_dev)
    ])

    mechanical_evidence = abs(r.vibration_dev)

    injector_evidence = np.mean([
        abs(r.injection_timing_dev),
        abs(r.fuel_flow_dev),
        abs(r.egt_dev)
    ])

    # ---------------------------------------------------------
    # Add engineering relationships
    # ---------------------------------------------------------

    # Thermal degradation normally produces simultaneous
    # elevation in CHT and EGT.
    if r.cht_dev > 1.0 and r.egt_dev > 1.0:
        thermal_evidence += 1.0

    # Lubrication problems commonly combine low oil pressure
    # with increased oil temperature.
    if r.oil_pressure_dev < -1.0 and r.oil_temp_dev > 1.0:
        lubrication_evidence += 1.0

    # Mechanical problems are strongly associated with vibration.
    if r.vibration_dev > 1.5:
        mechanical_evidence += 1.0

    # Injector/fuel abnormalities can appear as timing and
    # fuel-flow deviations accompanied by combustion changes.
    if abs(r.injection_timing_dev) > 1.5 and abs(r.fuel_flow_dev) > 1.0:
        injector_evidence += 1.0

    evidence = {
        'THERMAL DEGRADATION / OVERHEATING': thermal_evidence,
        'LUBRICATION DEGRADATION': lubrication_evidence,
        'MECHANICAL / VIBRATION ABNORMALITY': mechanical_evidence,
        'INJECTOR / FUEL ABNORMALITY': injector_evidence
    }

    # ---------------------------------------------------------
    # Select the most likely fault
    # ---------------------------------------------------------

    fault, best_score = max(
        evidence.items(),
        key=lambda x: x[1]
    )

    # Second-best score helps determine whether the diagnosis
    # is clearly separated from competing fault hypotheses.
    sorted_scores = sorted(
        evidence.values(),
        reverse=True
    )

    second_score = sorted_scores[1]

    # ---------------------------------------------------------
    # Decide whether there is enough evidence for a diagnosis
    # ---------------------------------------------------------

    if r.anomaly_score < 0.35 and best_score < 1.0:
        fault = 'NORMAL / NO CONFIRMED FAULT'
        severity = 'LOW'
        confidence = 0.80

    elif best_score < 1.0:
        fault = 'UNCLASSIFIED ABNORMAL OPERATING CONDITION'
        severity = 'MEDIUM'
        confidence = 0.50 + 0.10 * r.anomaly_score

    else:
        # Stronger evidence produces higher severity.
        if best_score >= 3.0:
            severity = 'HIGH'
        else:
            severity = 'MEDIUM'

        # Diagnosis confidence is based on:
        # - strength of engineering evidence
        # - AI anomaly score
        # - separation from the second-best fault
        evidence_strength = min(best_score / 4.0, 1.0)

        separation = min(
            max(best_score - second_score, 0) / 2.0,
            1.0
        )

        confidence = (
            0.45
            + 0.25 * evidence_strength
            + 0.20 * r.anomaly_score
            + 0.10 * separation
        )

        confidence = float(
            np.clip(confidence, 0.50, 0.95)
        )

    # ---------------------------------------------------------
    # Top contributing sensor deviations
    # ---------------------------------------------------------

    contributions = {
        'CHT': abs(r.cht_dev),
        'EGT': abs(r.egt_dev),
        'Oil pressure': abs(r.oil_pressure_dev),
        'Oil temperature': abs(r.oil_temp_dev),
        'Fuel flow': abs(r.fuel_flow_dev),
        'Vibration': abs(r.vibration_dev),
        'Injection timing': abs(r.injection_timing_dev)
    }

    top = sorted(
        contributions.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return fault, severity, confidence, top[:3]

def rul(df):
    """
    Estimate Remaining Useful Life using an accelerated prototype time scale.

    Prototype assumption:
        60 simulation samples = 1 equivalent operating hour.

    This is a demonstrative RUL estimate. Real deployment requires
    calibration using historical engine degradation/run-to-failure data.
    """

    y = df.overall_health.to_numpy()
    n = len(y)

    if n < 12:
        return (180, 240), 0.40

    # Use the most recent degradation window.
    w = min(90, n)
    yy = y[-w:]

    # Simulation samples.
    x = np.arange(w).reshape(-1, 1)

    model = LinearRegression().fit(x, yy)
    slope_per_sample = float(model.coef_[0])

    current_health = float(y[-1])
    eol_health = 0.40

    # Accelerated simulation time:
    # 60 simulation samples = 1 equivalent operating hour.
    SAMPLES_PER_OPERATING_HOUR = 60.0

    # If health is not degrading, give a bounded prototype estimate.
    if slope_per_sample >= -1e-6:
        if current_health <= eol_health:
            return (0, 1), 0.62
        return (180, 240), 0.45

    # Number of simulation samples until the EOL health criterion.
    samples_to_eol = max(
        (current_health - eol_health) / abs(slope_per_sample),
        0
    )

    # Convert accelerated simulation samples to equivalent operating hours.
    operating_hours = samples_to_eol / SAMPLES_PER_OPERATING_HOUR

    # Prototype uncertainty band.
    spread = max(operating_hours * 0.12, 1.0)

    lower = max(0, operating_hours - spread)
    upper = operating_hours + spread

    # Confidence is heuristic and intentionally bounded.
    confidence = float(
        np.clip(
            0.55 + min(abs(slope_per_sample) * 30, 0.30),
            0,
            0.90
        )
    )

    return (lower, upper), confidence
