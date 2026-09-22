"""
AeroTwin virtual engine behavior layer.

Implements the flow described in the "AeroTwin — Digital Twin & Engine
Simulator Pre-Coding Design Guide" (Team Member 2 scope):

    INPUTS -> ENGINE MODEL -> SIMULATED SENSORS -> DERIVED PARAMETERS -> ANALYTICS

Design decisions carried over from the guide:

  * The ORIGINAL ten telemetry columns (rpm, cht, egt, oil_pressure,
    oil_temp, fuel_flow, vibration, battery_voltage, alternator_health,
    injection_timing) keep their exact original formulas and column
    names (Design Guide, Section 3: "Keep them for compatibility").
    ``analytics.py`` calibrates its IsolationForest and its illustrative
    threshold table against these exact value ranges, so their math is
    NOT touched here.
  * Everything else in this module is new: fourteen additional simulated
    parameters (Section 4) plus seven derived parameters (Section 5),
    wired together using the dependency map in Section 6 and the
    step-by-step model in Section 7, in the calculation order from
    Section 9.
  * Faults still change the *physical* subsystem for engine faults, and
    only the *reported* measurement for sensor faults (Rule 7). New
    parameters that are functions of already-faulted upstream values
    (e.g. Air-Fuel Ratio, Power Output, Vibration Frequency) therefore
    inherit fault behaviour automatically instead of needing their own
    hard-coded deltas for every scenario.
  * health / anomaly / diagnosis / RUL logic stays out of this file
    (Rule 8) and continues to live in analytics.py.

This is a prototype design aid, not a certified aero-engine model
(Section 7 / Section 14 "Source basis").
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration constants (Rule 4 — keep formulas explainable, Rule 5 — keep
# faults configurable instead of hard-coded magic numbers scattered in code).
# ---------------------------------------------------------------------------

SCENARIOS = [
    'Normal',
    'Overheating / Thermal Degradation',
    'Lubrication Degradation',
    'Injector / Fuel Abnormality',
    'Mechanical / Vibration',
    'Electrical Degradation',
    'Sensor Drift',
]

# Accelerated prototype time scale, shared with analytics.rul(): each
# telemetry sample represents 1 simulated minute, i.e. 60 samples = 1
# equivalent operating hour.
SAMPLES_PER_HOUR = 60.0

# Air / fuel / electrical prototype constants. These are illustrative
# round numbers for a small aero-piston (Rotax-9xx class) engine, chosen
# to keep every new signal in a physically plausible band — they are not
# calibrated against a certified engine data sheet.
SEA_LEVEL_PRESSURE_KPA = 101.3
FUEL_DENSITY_G_PER_L = 750.0        # aviation gasoline, approx.
FUEL_ENERGY_KWH_PER_L = 9.7         # approx. usable energy density
FUEL_TANK_CAPACITY_L = 220.0
K_AIR_MASS_FLOW = 0.035             # AirMassFlow ~= K * pressure * RPM / temp(K)
TORQUE_MAX_NM = 140.0
CYLINDERS_FIRING_ORDER = 4          # 4-stroke, 4-cylinder -> RPM/30 Hz firing freq
ALTERNATOR_BUS_VOLTAGE = 28.0       # nominal DC bus voltage

# Fault ramp shape: fraction of the mission at which a fault begins, and
# how long (as a fraction of the remaining mission) it takes to reach
# full severity. Kept as named parameters (not magic numbers) so a
# scenario run can override them.
DEFAULT_FAULT_START_FRACTION = 0.58

# Degradation-stage labels used for the HEALTHY -> FAULT START -> MILD ->
# MODERATE -> SEVERE progression in Section 11 of the design guide.
DEGRADATION_STAGE_BOUNDS = [0.0, 0.02, 0.35, 0.70, 1.0]
DEGRADATION_STAGE_LABELS = ['HEALTHY', 'FAULT START', 'MILD', 'MODERATE', 'SEVERE']

# Plausible operating ranges used by validate_telemetry(). Generous on
# purpose: this catches broken formulas (NaN, sign errors, runaway
# values), not tight engineering tolerances.
RANGES = {
    'rpm': (2000, 5200),
    'cht': (40, 300),
    'egt': (300, 950),
    'oil_pressure': (0, 95),
    'oil_temp': (30, 160),
    'fuel_flow': (0, 40),
    'vibration': (0, 0.8),
    'battery_voltage': (15, 30),
    'alternator_health': (0, 1),
    'injection_timing': (0, 35),
    'engine_load': (0, 1),
    'torque': (0, 220),
    'power_output': (0, 100),
    'manifold_pressure': (5, 130),
    'intake_air_temp': (-30, 100),
    'fuel_pressure': (0, 340),
    'fuel_level': (0, FUEL_TANK_CAPACITY_L),
    'fuel_consumed': (0, FUEL_TANK_CAPACITY_L * 2),
    'air_mass_flow': (0, 130),
    'air_fuel_ratio': (2, 45),
    'ignition_timing': (0, 45),
    'vibration_frequency': (0, 260),
    'battery_current': (-45, 45),
    'alternator_voltage': (0, 33),
    'alternator_current': (0, 65),
    'electrical_power': (-1200, 1200),
    'alternator_power': (0, 2000),
    'engine_efficiency': (0, 1),
}


def mission_profile(n=180):
    """Unchanged mission generator: produces throttle/altitude/ambient
    inputs for a canned TAKEOFF -> CLIMB -> CRUISE -> HIGH LOAD -> DESCENT
    profile."""
    t = np.arange(n)
    phase = []
    throttle = []
    altitude = []
    ambient = []
    for i in t:
        if i < 25:
            ph, th, alt = 'TAKEOFF', .85, 150 + i * 90
        elif i < 65:
            ph, th, alt = 'CLIMB', .78, 2400 + (i - 25) * 180
        elif i < 145:
            ph, th, alt = 'CRUISE', .62, 9600
        elif i < 165:
            ph, th, alt = 'HIGH LOAD', .88, 10500
        else:
            ph, th, alt = 'DESCENT', .48, max(800, 10500 - (i - 165) * 450)
        phase.append(ph)
        throttle.append(th)
        altitude.append(alt)
        ambient.append(22 + .0015 * alt)
    return pd.DataFrame({
        't': t, 'mission_phase': phase, 'throttle': throttle,
        'altitude_ft': altitude, 'ambient_temp': ambient,
    })


def _degradation_stage(ramp):
    """Map a 0..1 fault ramp onto the HEALTHY/FAULT START/MILD/MODERATE/
    SEVERE labels from Section 11 of the design guide."""
    idx = np.digitize(ramp, DEGRADATION_STAGE_BOUNDS, right=True)
    idx = np.clip(idx, 0, len(DEGRADATION_STAGE_LABELS) - 1)
    # ramp == 0 should read HEALTHY even though digitize with right=True
    # puts an exact 0 at bin 0 already, so this is just defensive.
    labels = np.array(DEGRADATION_STAGE_LABELS)[idx]
    labels = np.where(ramp <= 0, 'HEALTHY', labels)
    return labels


def simulate(scenario='Normal', n=180, seed=42, throttle=None, altitude=None,
             ambient=None, fault_start_fraction=DEFAULT_FAULT_START_FRACTION):
    """Produce one mission's worth of correlated, deterministic engine
    telemetry.

    Same mission + same seed + same scenario reproduces the same
    baseline and the same fault progression (Design Guide "Success
    condition").
    """
    rng = np.random.default_rng(seed)
    m = mission_profile(n)
    if throttle is not None:
        m['throttle'] = throttle
    if altitude is not None:
        m['altitude_ft'] = altitude
    if ambient is not None:
        m['ambient_temp'] = ambient

    t = m.t.to_numpy()
    th = m.throttle.to_numpy()
    alt = m.altitude_ft.to_numpy()
    amb = m.ambient_temp.to_numpy()

    # --- Step 1 — Normalize inputs -----------------------------------
    af = np.clip(alt / 10000, 0, 1.5)

    # --- Step 2 — Engine load (Section 4, #11) -----------------------
    load = .35 + .65 * th

    # --- Step 3 — Original ten telemetry columns (UNCHANGED formulas) -
    # Kept byte-for-byte identical to the pre-existing simulator so that
    # analytics.py's expected(), THRESHOLDS and calibrated IsolationForest
    # stay valid (Design Guide, Rule 1 / Rule 2 / Section 3).
    rpm = 3300 + 1150 * th - 45 * af + rng.normal(0, 22, n)
    fuel_flow = 7.5 + 15.5 * th + .7 * af + rng.normal(0, .35, n)
    egt = 500 + 190 * th + 18 * af + .45 * (amb - 25) + rng.normal(0, 9, n)
    cht = 135 + 52 * th + 10 * af + .35 * (amb - 25) + rng.normal(0, 2.8, n)
    oil_t = 65 + 26 * load + .25 * (amb - 25) + rng.normal(0, 1.5, n)
    oil_p = 66 - 10 * load - 2.5 * af + rng.normal(0, 1.3, n)
    vib = .10 + .045 * load + rng.normal(0, .007, n)
    batt = 24.4 - .35 * th + rng.normal(0, .06, n)
    alt_h = np.clip(.98 - .02 * th + rng.normal(0, .004, n), 0, 1)
    inj = 18 + rng.normal(0, .12, n)

    # --- Step 4 — Build the air system (Section 4, #14/#15/#18) -------
    intake_air_temp = amb + 8 * load + rng.normal(0, .5, n)
    ambient_pressure_kpa = SEA_LEVEL_PRESSURE_KPA * np.clip(1 - .10 * af, .55, 1.0)
    manifold_pressure = ambient_pressure_kpa * (.25 + .75 * th) + rng.normal(0, 1.0, n)
    intake_air_temp_k = intake_air_temp + 273.15
    air_mass_flow = np.clip(
        K_AIR_MASS_FLOW * manifold_pressure * rpm / intake_air_temp_k
        + rng.normal(0, .6, n), 0, None,
    )

    # --- Step 5 — Build the fuel system (Section 4, #16/#17) -----------
    # Fuel pressure is stable under normal operation and only responds to
    # fuel-system degradation applied later in the fault-injection step.
    fuel_pressure = 300.0 + rng.normal(0, 3.0, n)

    # --- Step 6 — Build combustion (Section 4, #19/#20) -----------------
    fuel_mass_flow_g_s = fuel_flow * FUEL_DENSITY_G_PER_L / 3600.0
    ignition_timing = (
        22 + .003 * (rpm - 3300) - 3 * (load - .5) + rng.normal(0, .3, n)
    )

    # --- Step 7/8 — Thermal behaviour: EGT/CHT already computed above --

    # --- Step 9 — Mechanical behaviour: Torque & Vibration Frequency ---
    # (Section 4, #12/#21). Torque is driven by load and RPM (with a mild
    # high-RPM roll-off) and is refined once AFR is known below.
    torque = (
        TORQUE_MAX_NM * load - .004 * np.clip(rpm - 4200, 0, None)
        + rng.normal(0, 2.0, n)
    )
    vibration_frequency = np.clip(
        rpm / (CYLINDERS_FIRING_ORDER * 7.5) + rng.normal(0, 1.5, n), 0, None,
    )

    # --- Step 10/11 — Electrical behaviour (Section 4, #22/#23/#24) ----
    electrical_load_a = 12 + 6 * load + rng.normal(0, .3, n)
    alternator_voltage = (
        ALTERNATOR_BUS_VOLTAGE * alt_h * np.clip(rpm / 3300, .6, 1.15)
        + rng.normal(0, .15, n)
    )
    alternator_current = np.clip(
        electrical_load_a * (2 - alt_h) + rng.normal(0, .5, n), 0, None,
    )
    battery_current = alternator_current - electrical_load_a + rng.normal(0, .2, n)

    # ------------------------------------------------------------------
    # Step 13 — Apply scenario-specific degradation (Section 8). Faults
    # are added as deltas on top of the already-computed baseline arrays
    # so a fault changes physical state, not just a reading (Rule 7),
    # and every derived value computed afterwards inherits the fault
    # automatically instead of needing its own hard-coded delta.
    # ------------------------------------------------------------------
    fs = int(n * fault_start_fraction)
    ramp = np.clip((t - fs) / max(1, n - fs), 0, 1)

    if scenario == 'Overheating / Thermal Degradation':
        cht = cht + 42 * ramp
        egt = egt + 82 * ramp
        oil_t = oil_t + 15 * ramp
        vib = vib + .025 * ramp
        intake_air_temp = intake_air_temp + 10 * ramp       # heat soak
        torque = torque - 5 * ramp                          # power loss

    elif scenario == 'Lubrication Degradation':
        oil_p = oil_p - 22 * ramp
        oil_t = oil_t + 20 * ramp
        vib = vib + .075 * ramp
        cht = cht + 7 * ramp
        torque = torque - 8 * ramp                          # friction loss
        vibration_frequency = vibration_frequency + 4 * ramp

    elif scenario == 'Injector / Fuel Abnormality':
        inj = inj + 2.2 * ramp
        fuel_flow = fuel_flow + 4.5 * ramp
        egt = egt + 45 * ramp
        rpm = rpm - 180 * ramp
        vib = vib + .055 * ramp
        fuel_pressure = fuel_pressure - 35 * ramp
        ignition_timing = ignition_timing - 1.5 * ramp

    elif scenario == 'Mechanical / Vibration':
        vib = vib + .20 * ramp
        rpm = rpm + 75 * np.sin(t / 2.5) * ramp
        oil_p = oil_p - 6 * ramp
        torque = torque - 6 * ramp
        vibration_frequency = vibration_frequency + 10 * ramp * np.sin(t / 1.7)

    elif scenario == 'Electrical Degradation':
        alt_h = np.clip(alt_h - .25 * ramp, 0, 1)
        # alternator/battery readings below are recomputed after this
        # block so they inherit the degraded alternator health directly.

    elif scenario == 'Sensor Drift':
        # Sensor drift changes only the *reported* measurement — the
        # physical engine state (and every new parameter derived from
        # it) must remain unaffected (Rule 7).
        cht = cht + 24 * ramp
        egt = egt - 18 * ramp

    # Recompute the electrical chain that depends on alternator health so
    # an Electrical Degradation fault propagates into voltage/current
    # without a separate hard-coded delta for each signal.
    if scenario == 'Electrical Degradation':
        alternator_voltage = (
            ALTERNATOR_BUS_VOLTAGE * alt_h * np.clip(rpm / 3300, .6, 1.15)
            + rng.normal(0, .15, n)
        )
        alternator_current = np.clip(
            electrical_load_a * (2 - alt_h) + rng.normal(0, .5, n), 0, None,
        )
        battery_current = alternator_current - electrical_load_a + rng.normal(0, .2, n)

    # ------------------------------------------------------------------
    # Step 12 — Derived parameters (Section 5). Calculated from the
    # (possibly fault-adjusted) telemetry above instead of being
    # independently randomized (Rule 3).
    # ------------------------------------------------------------------
    air_fuel_ratio = air_mass_flow / np.clip(fuel_mass_flow_g_s, .01, None)
    power_output = np.clip(torque * rpm / 9550.0, 0, None)   # kW

    dt_hours = 1.0 / SAMPLES_PER_HOUR
    fuel_consumed = np.cumsum(np.clip(fuel_flow, 0, None) * dt_hours)
    fuel_level = np.clip(FUEL_TANK_CAPACITY_L - fuel_consumed, 0, FUEL_TANK_CAPACITY_L)

    electrical_power = batt * battery_current
    alternator_power = alternator_voltage * alternator_current
    fuel_power_kw = np.clip(fuel_flow, .01, None) * FUEL_ENERGY_KWH_PER_L
    engine_efficiency = np.clip(power_output / fuel_power_kw, 0, 1)

    degradation_stage = _degradation_stage(ramp)

    # ------------------------------------------------------------------
    # Step 15 — Assemble the DataFrame: preserve existing column names,
    # add the new ones.
    # ------------------------------------------------------------------
    m = m.copy()
    columns = {
        # Original ten (unchanged names/values).
        'rpm': rpm, 'cht': cht, 'egt': egt, 'oil_pressure': oil_p,
        'oil_temp': oil_t, 'fuel_flow': fuel_flow, 'vibration': vib,
        'battery_voltage': batt, 'alternator_health': alt_h,
        'injection_timing': inj,
        # New simulated sensors / model outputs (Section 4).
        'engine_load': load,
        'torque': torque,
        'manifold_pressure': manifold_pressure,
        'intake_air_temp': intake_air_temp,
        'fuel_pressure': fuel_pressure,
        'air_mass_flow': air_mass_flow,
        'ignition_timing': ignition_timing,
        'vibration_frequency': vibration_frequency,
        'battery_current': battery_current,
        'alternator_voltage': alternator_voltage,
        'alternator_current': alternator_current,
        # Derived parameters (Section 5).
        'power_output': power_output,
        'air_fuel_ratio': air_fuel_ratio,
        'fuel_consumed': fuel_consumed,
        'fuel_level': fuel_level,
        'electrical_power': electrical_power,
        'alternator_power': alternator_power,
        'engine_efficiency': engine_efficiency,
    }
    for k, v in columns.items():
        m[k] = v

    m['scenario'] = scenario
    m['fault_start'] = fs
    m['fault_ramp'] = ramp
    m['degradation_stage'] = degradation_stage
    return m


def expected(row):
    """Deterministic (noise-free, fault-free) healthy expected state for
    a single telemetry row. Used by analytics.py's Digital Twin
    deviation features and by app.py's expected-vs-observed charts.

    The ORIGINAL ten keys/formulas are unchanged (analytics.py depends
    on these exact values). Additional keys are provided for the new
    parameters for forward compatibility; analytics.py currently ignores
    unknown keys, so this is a safe, additive change.
    """
    th = float(row.throttle)
    alt = float(row.altitude_ft)
    amb = float(row.ambient_temp)
    af = np.clip(alt / 10000, 0, 1.5)
    load = .35 + .65 * th
    rpm = 3300 + 1150 * th - 45 * af
    fuel_flow = 7.5 + 15.5 * th + .7 * af

    intake_air_temp = amb + 8 * load
    ambient_pressure_kpa = SEA_LEVEL_PRESSURE_KPA * np.clip(1 - .10 * af, .55, 1.0)
    manifold_pressure = ambient_pressure_kpa * (.25 + .75 * th)
    air_mass_flow = max(
        K_AIR_MASS_FLOW * manifold_pressure * rpm / (intake_air_temp + 273.15), 0.0,
    )
    fuel_mass_flow_g_s = fuel_flow * FUEL_DENSITY_G_PER_L / 3600.0
    air_fuel_ratio = air_mass_flow / max(fuel_mass_flow_g_s, .01)

    torque = TORQUE_MAX_NM * load - .004 * max(rpm - 4200, 0)
    power_output = max(torque * rpm / 9550.0, 0.0)

    electrical_load_a = 12 + 6 * load
    alt_h = np.clip(.98 - .02 * th, 0, 1)
    alternator_voltage = ALTERNATOR_BUS_VOLTAGE * alt_h * np.clip(rpm / 3300, .6, 1.15)
    alternator_current = max(electrical_load_a * (2 - alt_h), 0.0)
    battery_current = alternator_current - electrical_load_a
    battery_voltage = 24.4 - .35 * th

    return {
        # Original ten — unchanged.
        'rpm': rpm,
        'cht': 135 + 52 * th + 10 * af + .35 * (amb - 25),
        'egt': 500 + 190 * th + 18 * af + .45 * (amb - 25),
        'oil_pressure': 66 - 10 * load - 2.5 * af,
        'oil_temp': 65 + 26 * load + .25 * (amb - 25),
        'fuel_flow': fuel_flow,
        'vibration': .10 + .045 * load,
        'battery_voltage': battery_voltage,
        'alternator_health': alt_h,
        'injection_timing': 18.0,
        # New parameters.
        'engine_load': load,
        'torque': torque,
        'manifold_pressure': manifold_pressure,
        'intake_air_temp': intake_air_temp,
        'fuel_pressure': 300.0,
        'air_mass_flow': air_mass_flow,
        'ignition_timing': 22 + .003 * (rpm - 3300) - 3 * (load - .5),
        'vibration_frequency': max(rpm / (CYLINDERS_FIRING_ORDER * 7.5), 0.0),
        'battery_current': battery_current,
        'alternator_voltage': alternator_voltage,
        'alternator_current': alternator_current,
        'power_output': power_output,
        'air_fuel_ratio': air_fuel_ratio,
        'electrical_power': battery_voltage * battery_current,
        'alternator_power': alternator_voltage * alternator_current,
    }


def validate_telemetry(df):
    """Rule 9 — validate every column: ranges, missing values, and a few
    cross-parameter sanity checks. Returns a list of human-readable issue
    strings; an empty list means the telemetry looks sane.

    This does not raise so it stays safe to call from the interactive
    app; callers/tests that want a hard failure should assert on the
    returned list being empty.
    """
    issues = []

    for col in RANGES:
        if col not in df.columns:
            issues.append(f'missing column: {col}')
            continue
        s = df[col]
        if s.isna().any():
            issues.append(f'{col}: contains NaN')
        if not np.isfinite(s.to_numpy()).all():
            issues.append(f'{col}: contains non-finite values')
        lo, hi = RANGES[col]
        below = s < lo
        above = s > hi
        if below.any() or above.any():
            issues.append(
                f'{col}: {int(below.sum() + above.sum())} sample(s) outside '
                f'expected range [{lo}, {hi}] (min={s.min():.3g}, max={s.max():.3g})'
            )

    if 'fuel_consumed' in df.columns and (df['fuel_consumed'].diff().dropna() < -1e-6).any():
        issues.append('fuel_consumed: not monotonically non-decreasing')

    if 'fuel_level' in df.columns and (df['fuel_level'].diff().dropna() > 1e-6).any():
        issues.append('fuel_level: not monotonically non-increasing')

    return issues
