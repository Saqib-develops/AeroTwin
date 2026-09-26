"""
Tests for the expanded simulator.py, mapped directly onto the "What to
Test After Coding" checklist (Design Guide, Section 12).

Runs standalone (no extra dependency):
    python test_simulator.py

Also discoverable by pytest if it happens to be installed, since every
check is a ``test_*`` function using plain asserts.
"""

import numpy as np
import pandas as pd

from simulator import simulate, expected, validate_telemetry, SCENARIOS


def _steady(scenario, n=200, seed=1, throttle=.70, altitude=6000.0, ambient=25.0):
    """Run a scenario under CONSTANT mission conditions so that any
    change between the pre-fault and post-fault windows can only come
    from the fault itself, not from the canned mission profile's own
    throttle/altitude swings (takeoff/climb/cruise/descent)."""
    return simulate(
        scenario, n, seed,
        throttle=np.full(n, throttle),
        altitude=np.full(n, altitude),
        ambient=np.full(n, ambient),
    )


def test_same_seed_is_deterministic():
    a = simulate('Normal', 180, 42)
    b = simulate('Normal', 180, 42)
    pd.testing.assert_frame_equal(a, b)


def test_throttle_sweep_raises_load_rpm_fuel_thermal():
    low = simulate('Normal', 120, 1, throttle=np.full(120, .35)).iloc[-1]
    high = simulate('Normal', 120, 1, throttle=np.full(120, .95)).iloc[-1]
    assert high.engine_load > low.engine_load
    assert high.rpm > low.rpm
    assert high.fuel_flow > low.fuel_flow
    assert high.egt > low.egt
    assert high.cht > low.cht
    assert high.power_output > low.power_output


def test_altitude_sweep_reduces_available_pressure():
    low_alt = simulate('Normal', 120, 2, altitude=np.full(120, 500)).iloc[-1]
    high_alt = simulate('Normal', 120, 2, altitude=np.full(120, 14000)).iloc[-1]
    assert high_alt.manifold_pressure < low_alt.manifold_pressure
    assert high_alt.air_mass_flow < low_alt.air_mass_flow


def test_normal_run_within_plausible_ranges():
    df = simulate('Normal', 300, 7)
    issues = validate_telemetry(df)
    assert issues == [], f'unexpected out-of-range telemetry: {issues}'


def test_overheating_chain():
    df = _steady('Overheating / Thermal Degradation', seed=3)
    before = df[df.t < df.fault_start.iloc[0]]
    after = df.tail(10)
    assert after.cht.mean() > before.cht.mean()
    assert after.egt.mean() > before.egt.mean()
    assert after.oil_temp.mean() > before.oil_temp.mean()


def test_lubrication_chain():
    df = _steady('Lubrication Degradation', seed=4)
    before = df[df.t < df.fault_start.iloc[0]]
    after = df.tail(10)
    assert after.oil_pressure.mean() < before.oil_pressure.mean()
    assert after.oil_temp.mean() > before.oil_temp.mean()
    assert after.vibration.mean() > before.vibration.mean()


def test_injector_fuel_chain():
    df = _steady('Injector / Fuel Abnormality', seed=5)
    before = df[df.t < df.fault_start.iloc[0]]
    after = df.tail(10)
    assert after.fuel_pressure.mean() < before.fuel_pressure.mean()
    assert after.injection_timing.mean() > before.injection_timing.mean()
    assert after.fuel_flow.mean() > before.fuel_flow.mean()
    assert after.air_fuel_ratio.mean() != before.air_fuel_ratio.mean()
    assert after.egt.mean() > before.egt.mean()
    assert after.rpm.mean() < before.rpm.mean()


def test_mechanical_chain():
    df = _steady('Mechanical / Vibration', seed=6)
    before = df[df.t < df.fault_start.iloc[0]]
    after = df.tail(10)
    assert after.vibration.mean() > before.vibration.mean()
    assert after.vibration_frequency.std() > before.vibration_frequency.std()
    assert after.power_output.mean() != before.power_output.mean()


def test_electrical_chain():
    df = _steady('Electrical Degradation', seed=8)
    before = df[df.t < df.fault_start.iloc[0]]
    after = df.tail(10)
    assert after.alternator_health.mean() < before.alternator_health.mean()
    assert after.alternator_voltage.mean() < before.alternator_voltage.mean()
    assert after.alternator_current.mean() > before.alternator_current.mean()
    assert after.battery_current.mean() != before.battery_current.mean()


def test_sensor_drift_is_isolated():
    df = _steady('Sensor Drift', seed=9)
    before = df[df.t < df.fault_start.iloc[0]]
    after = df.tail(10)
    # The drifted readings change...
    assert after.cht.mean() > before.cht.mean()
    assert after.egt.mean() < before.egt.mean()
    # ...but physical state elsewhere does not (Rule 7). Under constant
    # mission conditions, non-drifted signals should barely move.
    for col in ['rpm', 'oil_pressure', 'oil_temp', 'vibration', 'torque',
                'power_output', 'fuel_flow', 'alternator_health']:
        tol = max(0.03 * (abs(before[col].mean()) + 1e-6), 3 * before[col].std() + 1e-6)
        assert abs(after[col].mean() - before[col].mean()) < tol, (
            f'{col} unexpectedly drifted with a sensor-only fault'
        )


def test_expected_matches_noise_free_normal_baseline():
    """expected(row) should track the deterministic mean of a Normal run
    (allowing for the small amount of sensor noise)."""
    df = simulate('Normal', 400, 11)
    for _, row in df.sample(5, random_state=1).iterrows():
        e = expected(row)
        assert abs(row.rpm - e['rpm']) < 150
        assert abs(row.cht - e['cht']) < 15
        assert abs(row.egt - e['egt']) < 40
        assert abs(row.power_output - e['power_output']) < 8


def test_all_scenarios_run_and_validate():
    for scenario in SCENARIOS:
        df = simulate(scenario, 150, 99)
        issues = validate_telemetry(df)
        assert issues == [], f'{scenario}: {issues}'

def test_normal_has_no_degradation():
    df = simulate("Normal", 180, 42)

    assert (df["fault_ramp"] == 0).all()
    assert (df["degradation_stage"] == "HEALTHY").all()

def _run_all():
    tests = [(name, fn) for name, fn in globals().items()
             if name.startswith('test_') and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f'PASS  {name}')
        except AssertionError as e:
            failed += 1
            print(f'FAIL  {name}: {e}')
    print(f'\n{len(tests) - failed}/{len(tests)} passed')
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    _run_all()
