# AeroTwin — SIH26054 Prototype

A demoable, defensible prototype of an AI-assisted Digital Twin for MALE UAV aero-piston engine health monitoring.

## Core demo

**Telemetry CSV → Digital Twin → Health → AI anomaly → Fault diagnosis → RUL → Maintenance advisory → Streamlit dashboard**

The app also includes a built-in physics-informed simulator so the team can demonstrate the system without an external dataset.

## Input CSV

The live/demo input is a CSV file. Required columns:

```text
t, mission_phase, throttle, altitude_ft, ambient_temp,
rpm, cht, egt, oil_pressure, oil_temp, fuel_flow,
vibration, battery_voltage, alternator_health, injection_timing
```

A sample is included in `data/sample_telemetry_overheating.csv`.

## Why CSV instead of SQLite

For this prototype, CSV is the clearest input boundary: a test-rig export, recorded UAV telemetry file, or synthetic dataset can be uploaded and immediately analysed. There is no database setup and no hidden persistence layer.

A production system would normally use a telemetry/time-series data service for continuous fleet data. That is outside the three-day prototype scope.

## Run

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```



