import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope, MinCovDet
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import HuberRegressor


from simulator import simulate, expected


# =====================================================================
# AeroTwin Analytics
# =====================================================================
#
# Pipeline:
#
# Telemetry
#     ↓
# Digital Twin expected values
#     ↓
# Sensor residuals / deviations
#     ↓
# Temporal + physical relationship features
#     ↓
# Robust scaling
#     ↓
# Isolation Forest
#     +
# Robust Mahalanobis distance
#     +
# Elliptic Envelope
#     +
# Digital Twin residual severity
#     ↓
# Persistent anomaly detection
#     ↓
# Subsystem health
#     ↓
# Hybrid fault diagnosis
#     ↓
# RUL estimation
#
# IMPORTANT:
# This is a prototype analytics system for the AeroTwin demo.
# It is NOT a certified aircraft/engine diagnostic or maintenance system.
# =====================================================================


# =====================================================================
# SENSOR CONFIGURATION
# =====================================================================

BASE_SENSORS = [
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


EXTRA_SENSORS = [
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


ALL_CANDIDATE_SENSORS = BASE_SENSORS + EXTRA_SENSORS


# =====================================================================
# SENSOR DISCOVERY
# =====================================================================
#
# We do not blindly assume every simulator version exposes every sensor.
# This makes analytics.py more resistant to changes in simulator.py.
# =====================================================================


def _discover_sensors():
    try:
        probe = simulate(
            "Normal",
            5,
            0,
            throttle=np.full(5, 0.70),
            altitude=np.full(5, 6000.0),
            ambient=np.full(5, 25.0),
        )

        if len(probe) == 0:
            return list(BASE_SENSORS)

        expected_row = expected(probe.iloc[0])

        discovered = [
            sensor
            for sensor in ALL_CANDIDATE_SENSORS
            if sensor in probe.columns
            and sensor in expected_row
        ]

        if discovered:
            return discovered

    except Exception:
        pass

    return list(BASE_SENSORS)


SENSORS = _discover_sensors()

FEATURES = [
    f"{sensor}_dev"
    for sensor in SENSORS
]


# =====================================================================
# FAULT LABELS
# =====================================================================

FAULTS = [
    "THERMAL DEGRADATION / OVERHEATING",
    "LUBRICATION DEGRADATION",
    "INJECTOR / FUEL ABNORMALITY",
    "MECHANICAL / VIBRATION ABNORMALITY",
    "ELECTRICAL DEGRADATION",
    "SENSOR DRIFT",
]


# =====================================================================
# PROTOTYPE THRESHOLDS
# =====================================================================

THRESHOLDS = {
    "cht": 205.0,
    "egt": 700.0,
    "oil_pressure": 45.0,
    "oil_temp": 105.0,
    "vibration": 0.28,
    "fuel_flow": 29.0,
    "injection_timing_deviation": 1.5,
}


# =====================================================================
# EXPECTED-VALUE CACHE
# =====================================================================
#
# Calling expected(row) once for every sensor is wasteful.
#
# We calculate expected(row) once per operating point and reuse it.
# =====================================================================


def _operating_point_key(row):
    return (
        round(float(row["throttle"]), 6)
        if "throttle" in row
        else None,

        round(float(row["altitude"]), 6)
        if "altitude" in row
        else None,

        round(float(row["ambient"]), 6)
        if "ambient" in row
        else None,
    )


# =====================================================================
# DIGITAL TWIN FEATURES
# =====================================================================


def add_twin_features(df):
    """
    Calculate Digital Twin residuals.

    For every available sensor:

        dev = (actual - expected) / scale

    Interpretation:

        dev ≈ 0
            Sensor is behaving close to healthy expected behaviour.

        dev > 0
            Sensor is above expected.

        dev < 0
            Sensor is below expected.

    The residual is normalized so different units can be compared.
    """

    out = df.copy()

    n = len(out)

    # Preallocate arrays.
    values = {
        sensor: np.full(n, np.nan, dtype=float)
        for sensor in SENSORS
    }

    expected_cache = {}

    for i, (_, row) in enumerate(out.iterrows()):

        # -------------------------------------------------------------
        # Get expected values once per operating point
        # -------------------------------------------------------------

        key = _operating_point_key(row)

        if key not in expected_cache:

            try:
                expected_cache[key] = expected(row)
            except Exception:
                expected_cache[key] = {}

        expected_values = expected_cache[key]

        # -------------------------------------------------------------
        # Calculate sensor residuals
        # -------------------------------------------------------------

        for sensor in SENSORS:

            if sensor not in row.index:
                continue

            if sensor not in expected_values:
                continue

            try:
                actual = float(row[sensor])
                baseline = float(expected_values[sensor])
            except (TypeError, ValueError):
                continue

            if not np.isfinite(actual) or not np.isfinite(baseline):
                continue

            scale = max(abs(baseline) * 0.08, 0.1)

            values[sensor][i] = (
                (actual - baseline) / scale
            )

    # Add residual columns.
    for sensor in SENSORS:
        out[f"{sensor}_dev"] = values[sensor]

    # -------------------------------------------------------------
    # Digital Twin overall fit
    # -------------------------------------------------------------

    available_features = [
        feature
        for feature in FEATURES
        if feature in out.columns
    ]

    if available_features:

        abs_dev = (
            out[available_features]
            .abs()
            .clip(upper=4.0)
        )

        out["twin_fit"] = (
            1.0
            - abs_dev.mean(axis=1).fillna(0.0) / 4.0
        ).clip(0.0, 1.0)

    else:

        out["twin_fit"] = 1.0

    return out


# =====================================================================
# DERIVED / TEMPORAL FEATURES
# =====================================================================


def add_derived_features(df, window=7):
    """
    Add temporal and relationship features.

    These features are useful for gradual degradation and intermittent
    faults.

    Examples:

        vibration_dev_roll
        vibration_dev_std
        vibration_dev_delta
        vibration_dev_trend

    Also creates physical coupling features.
    """

    out = df.copy()

    dev_columns = [
        feature
        for feature in FEATURES
        if feature in out.columns
    ]

    # -------------------------------------------------------------
    # Individual sensor temporal features
    # -------------------------------------------------------------

    for column in dev_columns:

        series = pd.to_numeric(
            out[column],
            errors="coerce"
        )

        # Absolute deviation
        out[f"{column}_abs"] = series.abs()

        # Rolling average
        out[f"{column}_roll"] = (
            series
            .rolling(window, min_periods=1)
            .mean()
        )

        # Rolling variation
        out[f"{column}_std"] = (
            series
            .rolling(window, min_periods=2)
            .std()
            .fillna(0.0)
        )

        # Point-to-point change
        out[f"{column}_delta"] = (
            series
            .diff()
            .fillna(0.0)
        )

        # Trend
        short_mean = (
            series
            .rolling(window, min_periods=2)
            .mean()
        )

        long_mean = (
            series
            .rolling(window * 2, min_periods=2)
            .mean()
        )

        out[f"{column}_trend"] = (
            short_mean - long_mean
        ).fillna(0.0)

    # -------------------------------------------------------------
    # Thermal relationship
    # -------------------------------------------------------------

    if {
        "cht_dev",
        "egt_dev",
    }.issubset(out.columns):

        out["thermal_coupling"] = (
            0.55 * out["cht_dev"]
            + 0.45 * out["egt_dev"]
        )

    # -------------------------------------------------------------
    # Lubrication relationship
    # -------------------------------------------------------------

    if {
        "oil_pressure_dev",
        "oil_temp_dev",
    }.issubset(out.columns):

        out["lubrication_coupling"] = (
            -0.60 * out["oil_pressure_dev"]
            + 0.40 * out["oil_temp_dev"]
        )

    # -------------------------------------------------------------
    # Fuel relationship
    # -------------------------------------------------------------

    fuel_columns = {
        "fuel_pressure_dev",
        "fuel_flow_dev",
        "injection_timing_dev",
    }

    if fuel_columns.issubset(out.columns):

        out["fuel_coupling"] = (
            -0.30 * out["fuel_pressure_dev"]
            + 0.35 * out["fuel_flow_dev"]
            + 0.35 * out["injection_timing_dev"]
        )

    # -------------------------------------------------------------
    # Mechanical relationship
    # -------------------------------------------------------------

    if {
        "vibration_dev",
        "vibration_frequency_dev",
    }.issubset(out.columns):

        out["mechanical_coupling"] = (
            0.65 * out["vibration_dev"]
            + 0.35 * out["vibration_frequency_dev"]
        )

    # -------------------------------------------------------------
    # Electrical relationship
    # -------------------------------------------------------------

    if {
        "alternator_health_dev",
        "alternator_voltage_dev",
    }.issubset(out.columns):

        out["electrical_coupling"] = (
            -0.55 * out["alternator_health_dev"]
            -0.45 * out["alternator_voltage_dev"]
        )

    # -------------------------------------------------------------
    # Global residual statistics
    # -------------------------------------------------------------

    absolute_columns = [
        f"{column}_abs"
        for column in dev_columns
    ]

    if absolute_columns:

        out["deviation_mean"] = (
            out[absolute_columns]
            .mean(axis=1)
            .fillna(0.0)
        )

        out["deviation_max"] = (
            out[absolute_columns]
            .max(axis=1)
            .fillna(0.0)
        )

    else:

        out["deviation_mean"] = 0.0
        out["deviation_max"] = 0.0

    return out


# =====================================================================
# HELPER
# =====================================================================


def _safe_numeric(value, default=0.0):

    try:

        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except (TypeError, ValueError):

        return default


# =====================================================================
# ROBUST MAHALANOBIS
# =====================================================================


def _fit_robust_covariance(X):

    try:

        return MinCovDet(
            random_state=42,
            support_fraction=0.80
        ).fit(X)

    except Exception:

        return None


def _mahalanobis_score(model, X):

    if model is None:
        return np.zeros(len(X))

    try:

        distance = model.mahalanobis(X)

        # Smooth saturation into 0-1.
        score = (
            1.0
            - np.exp(
                -np.maximum(distance, 0.0) / 12.0
            )
        )

        return np.clip(score, 0.0, 1.0)

    except Exception:

        return np.zeros(len(X))


# =====================================================================
# DETECTOR CLASS
# =====================================================================


class AeroTwinDetector:
    """
    Combined anomaly detection model.

    Models:

        1. Isolation Forest
        2. Elliptic Envelope
        3. Robust Mahalanobis distance
        4. Digital Twin residual severity

    The class also supports:

        sc, model = fit_detector()

    for compatibility with the earlier analytics.py.
    """

    def __init__(
        self,
        scaler,
        feature_columns,
        isolation_forest,
        elliptic_envelope,
        robust_covariance,
        iso_q95,
        iso_q995,
        elliptic_q95,
        elliptic_q995,
        maha_q95,
        maha_q995,
    ):

        self.scaler = scaler

        self.feature_columns = feature_columns

        self.isolation_forest = isolation_forest

        self.elliptic_envelope = elliptic_envelope

        self.robust_covariance = robust_covariance

        self.iso_q95 = iso_q95
        self.iso_q995 = iso_q995

        self.elliptic_q95 = elliptic_q95
        self.elliptic_q995 = elliptic_q995

        self.maha_q95 = maha_q95
        self.maha_q995 = maha_q995

    def __iter__(self):
        """
        Compatibility with:

            sc, model = fit_detector()

        """

        yield self.scaler
        yield self

    def decision_function(self, X):

        iso = -self.isolation_forest.decision_function(X)

        if self.elliptic_envelope is not None:
            elliptic = (
                -self.elliptic_envelope
                .decision_function(X)
            )
        else:
            elliptic = np.zeros(len(X))

        maha = _mahalanobis_score(
            self.robust_covariance,
            X
        )

        return (
            0.50 * iso
            + 0.25 * elliptic
            + 0.25 * maha
        )

    def predict(self, X):

        score = self.decision_function(X)

        threshold = np.quantile(
            score,
            0.97
        )

        return np.where(
            score >= threshold,
            -1,
            1
        )


# =====================================================================
# CALIBRATION
# =====================================================================


def _calibrated_score(
    raw,
    q95,
    q995
):
    """
    Convert raw detector values into a stable 0-1 anomaly severity.

    Approximate interpretation:

        0.0 - 0.3 = mostly normal
        0.3 - 0.65 = unusual
        0.65 - 0.85 = abnormal
        0.85 - 1.0 = severe abnormality
    """

    denominator = max(
        q995 - q95,
        1e-9
    )

    score = (
        0.50
        + 0.50 * (
            (raw - q95)
            / denominator
        )
    )

    return np.clip(
        score,
        0.0,
        1.0
    )


# =====================================================================
# FIT ANOMALY DETECTOR
# =====================================================================


def fit_detector(
    n_baseline=1200,
    seed=123,
    contamination=0.03,
):
    """
    Train the anomaly detection system using healthy Normal operation.

    Returns:
        AeroTwinDetector

    Compatible with the previous:

        sc, model = fit_detector()

    because AeroTwinDetector implements __iter__.
    """

    # -------------------------------------------------------------
    # Generate healthy baseline
    # -------------------------------------------------------------

    base = simulate(
        "Normal",
        n_baseline,
        seed
    )

    # -------------------------------------------------------------
    # Digital Twin
    # -------------------------------------------------------------

    base = add_twin_features(base)

    base = add_derived_features(base)

    # -------------------------------------------------------------
    # Training features
    #
    # We intentionally prioritize stable residual features for the
    # ML models rather than throwing every rolling feature into the
    # detector. This prevents a huge high-dimensional feature space.
    # -------------------------------------------------------------

    train_features = [
        feature
        for feature in FEATURES
        if feature in base.columns
    ]

    # Add important relationship features when available.
    relationship_features = [
        "thermal_coupling",
        "lubrication_coupling",
        "fuel_coupling",
        "mechanical_coupling",
        "electrical_coupling",
    ]

    for feature in relationship_features:

        if feature in base.columns:
            train_features.append(feature)

    if not train_features:

        raise ValueError(
            "No Digital Twin features available for anomaly detection."
        )

    X = (
        base[train_features]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0.0)
    )

    # -------------------------------------------------------------
    # Robust scaling
    # -------------------------------------------------------------

    scaler = RobustScaler()

    X_scaled = scaler.fit_transform(X)

    # -------------------------------------------------------------
    # Isolation Forest
    # -------------------------------------------------------------

    isolation = IsolationForest(
        n_estimators=250,
        max_samples="auto",
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )

    isolation.fit(X_scaled)

    # -------------------------------------------------------------
    # Elliptic Envelope
    # -------------------------------------------------------------

    elliptic = None

    try:

        elliptic = EllipticEnvelope(
            contamination=contamination,
            support_fraction=0.90,
            random_state=42
        )

        elliptic.fit(X_scaled)

    except Exception:

        elliptic = None

    # -------------------------------------------------------------
    # Robust covariance
    # -------------------------------------------------------------

    robust_covariance = _fit_robust_covariance(
        X_scaled
    )

    # -------------------------------------------------------------
    # Training distributions
    # -------------------------------------------------------------

    iso_raw = (
        -isolation
        .decision_function(X_scaled)
    )

    if elliptic is not None:

        elliptic_raw = (
            -elliptic
            .decision_function(X_scaled)
        )

    else:

        elliptic_raw = np.zeros(
            len(X_scaled)
        )

    maha_raw = _mahalanobis_score(
        robust_covariance,
        X_scaled
    )

    # -------------------------------------------------------------
    # Quantile calibration
    # -------------------------------------------------------------

    iso_q95 = float(
        np.quantile(
            iso_raw,
            0.95
        )
    )

    iso_q995 = float(
        np.quantile(
            iso_raw,
            0.995
        )
    )

    elliptic_q95 = float(
        np.quantile(
            elliptic_raw,
            0.95
        )
    )

    elliptic_q995 = float(
        np.quantile(
            elliptic_raw,
            0.995
        )
    )

    maha_q95 = float(
        np.quantile(
            maha_raw,
            0.95
        )
    )

    maha_q995 = float(
        np.quantile(
            maha_raw,
            0.995
        )
    )

    # -------------------------------------------------------------
    # Return combined detector
    # -------------------------------------------------------------

    return AeroTwinDetector(
        scaler=scaler,
        feature_columns=train_features,
        isolation_forest=isolation,
        elliptic_envelope=elliptic,
        robust_covariance=robust_covariance,
        iso_q95=iso_q95,
        iso_q995=iso_q995,
        elliptic_q95=elliptic_q95,
        elliptic_q995=elliptic_q995,
        maha_q95=maha_q95,
        maha_q995=maha_q995,
    )


# =====================================================================
# ANOMALY DETECTION
# =====================================================================


def detect(
    df,
    sc=None,
    model=None,
    detector=None,
):
    """
    Run anomaly detection.

    Supported styles:

        detector = fit_detector()
        result = detect(df, detector=detector)

    OR:

        sc, model = fit_detector()
        result = detect(df, sc, model)

    """

    o = df.copy()

    # -------------------------------------------------------------
    # Resolve detector
    # -------------------------------------------------------------

    if detector is None:

        if isinstance(model, AeroTwinDetector):

            detector = model

        elif isinstance(sc, AeroTwinDetector):

            detector = sc

        else:

            detector = fit_detector()

    # -------------------------------------------------------------
    # Digital Twin features
    # -------------------------------------------------------------

    required_twin_features = [
        feature
        for feature in detector.feature_columns
        if feature.endswith("_dev")
    ]

    missing_twin = [
        feature
        for feature in required_twin_features
        if feature not in o.columns
    ]

    if missing_twin:

        o = add_twin_features(o)

    # -------------------------------------------------------------
    # Derived features
    # -------------------------------------------------------------

    if "deviation_mean" not in o.columns:

        o = add_derived_features(o)

    # -------------------------------------------------------------
    # Ensure detector columns exist
    # -------------------------------------------------------------

    for feature in detector.feature_columns:

        if feature not in o.columns:

            o[feature] = 0.0

    X = (
        o[detector.feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0.0)
    )

    X_scaled = detector.scaler.transform(X)

    # -------------------------------------------------------------
    # Isolation Forest
    # -------------------------------------------------------------

    iso_raw = (
        -detector
        .isolation_forest
        .decision_function(X_scaled)
    )

    iso_score = _calibrated_score(
        iso_raw,
        detector.iso_q95,
        detector.iso_q995,
    )

    # -------------------------------------------------------------
    # Elliptic Envelope
    # -------------------------------------------------------------

    if detector.elliptic_envelope is not None:

        elliptic_raw = (
            -detector
            .elliptic_envelope
            .decision_function(X_scaled)
        )

        elliptic_score = _calibrated_score(
            elliptic_raw,
            detector.elliptic_q95,
            detector.elliptic_q995,
        )

    else:

        elliptic_score = np.zeros(
            len(o)
        )

    # -------------------------------------------------------------
    # Mahalanobis
    # -------------------------------------------------------------

    maha_raw = _mahalanobis_score(
        detector.robust_covariance,
        X_scaled
    )

    maha_score = _calibrated_score(
        maha_raw,
        detector.maha_q95,
        detector.maha_q995,
    )

    # -------------------------------------------------------------
    # Digital Twin residual severity
    # -------------------------------------------------------------

    residual_score = np.clip(
        o["deviation_mean"].fillna(0.0)
        / 3.0,
        0.0,
        1.0
    )

    # -------------------------------------------------------------
    # Combined anomaly score
    #
    # 40% Isolation Forest
    # 20% Elliptic Envelope
    # 20% Robust Mahalanobis
    # 20% Digital Twin residual
    # -------------------------------------------------------------

    o["isolation_score"] = iso_score

    o["elliptic_score"] = elliptic_score

    o["mahalanobis_score"] = maha_score

    o["residual_score"] = residual_score

    o["anomaly_score"] = (
        0.40 * iso_score
        + 0.20 * elliptic_score
        + 0.20 * maha_score
        + 0.20 * residual_score
    ).clip(0.0, 1.0)

    # -------------------------------------------------------------
    # Raw anomaly condition
    # -------------------------------------------------------------

    raw_alarm = (
        (o["anomaly_score"] >= 0.65)
        |
        (o["deviation_max"] >= 2.5)
    )

    # -------------------------------------------------------------
    # Persistence
    #
    # 2 anomalous samples out of the last 3 = confirmed anomaly.
    # -------------------------------------------------------------

    o["anomaly"] = (
        raw_alarm
        .astype(int)
        .rolling(
            3,
            min_periods=1
        )
        .sum()
        >= 2
    )

    o["anomaly"] = o["anomaly"].astype(bool)

    # -------------------------------------------------------------
    # Persistence score
    # -------------------------------------------------------------

    o["anomaly_persistence"] = (
        o["anomaly"]
        .astype(int)
        .rolling(
            5,
            min_periods=1
        )
        .mean()
    )

    return o


# =====================================================================
# HEALTH CALCULATION
# =====================================================================


def health(df):
    """
    Calculate subsystem and overall engine health.
    """

    o = df.copy()

    required = [
        "cht_dev",
        "egt_dev",
        "oil_pressure_dev",
        "oil_temp_dev",
        "fuel_flow_dev",
        "vibration_dev",
        "battery_voltage_dev",
        "alternator_health_dev",
        "injection_timing_dev",
        "fuel_pressure_dev",
        "air_fuel_ratio_dev",
        "vibration_frequency_dev",
        "alternator_voltage_dev",
        "alternator_current_dev",
        "battery_current_dev",
    ]

    # -------------------------------------------------------------
    # Missing optional sensors are neutral.
    # -------------------------------------------------------------

    for column in required:

        if column not in o.columns:

            o[column] = 0.0

    # -------------------------------------------------------------
    # Exponential smoothing
    # -------------------------------------------------------------

    for column in required:

        series = (
            pd.to_numeric(
                o[column],
                errors="coerce"
            )
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(0.0)
        )

        o[f"{column}_sm"] = (
            series
            .ewm(
                span=8,
                adjust=False,
                min_periods=1
            )
            .mean()
        )

    # -------------------------------------------------------------
    # Health conversion
    # -------------------------------------------------------------

    def deviation_health(series):

        return (
            1.0
            - series.abs() / 4.0
        ).clip(
            0.0,
            1.0
        )

    # =================================================================
    # THERMAL
    # =================================================================

    cht_health = deviation_health(
        o["cht_dev_sm"]
    )

    egt_health = deviation_health(
        o["egt_dev_sm"]
    )

    oil_temp_health = deviation_health(
        o["oil_temp_dev_sm"]
    )

    o["thermal_health"] = (
        0.45 * cht_health
        + 0.40 * egt_health
        + 0.15 * oil_temp_health
    ).clip(0.0, 1.0)

    # =================================================================
    # MECHANICAL
    # =================================================================

    vibration_health = deviation_health(
        o["vibration_dev_sm"]
    )

    frequency_health = deviation_health(
        o["vibration_frequency_dev_sm"]
    )

    o["mechanical_health"] = (
        0.70 * vibration_health
        + 0.30 * frequency_health
    ).clip(0.0, 1.0)

    # =================================================================
    # LUBRICATION
    # =================================================================

    pressure_health = deviation_health(
        o["oil_pressure_dev_sm"]
    )

    lubrication_temp_health = deviation_health(
        o["oil_temp_dev_sm"]
    )

    o["lubrication_health"] = (
        0.65 * pressure_health
        + 0.35 * lubrication_temp_health
    ).clip(0.0, 1.0)

    # =================================================================
    # FUEL
    # =================================================================

    fuel_flow_health = deviation_health(
        o["fuel_flow_dev_sm"]
    )

    fuel_pressure_health = deviation_health(
        o["fuel_pressure_dev_sm"]
    )

    injection_health = deviation_health(
        o["injection_timing_dev_sm"]
    )

    afr_health = deviation_health(
        o["air_fuel_ratio_dev_sm"]
    )

    o["fuel_health"] = (
        0.30 * fuel_flow_health
        + 0.30 * fuel_pressure_health
        + 0.25 * injection_health
        + 0.15 * afr_health
    ).clip(0.0, 1.0)

    # =================================================================
    # ELECTRICAL
    # =================================================================

    battery_health = deviation_health(
        o["battery_voltage_dev_sm"]
    )

    alternator_health = deviation_health(
        o["alternator_health_dev_sm"]
    )

    alternator_voltage_health = deviation_health(
        o["alternator_voltage_dev_sm"]
    )

    alternator_current_health = deviation_health(
        o["alternator_current_dev_sm"]
    )

    battery_current_health = deviation_health(
        o["battery_current_dev_sm"]
    )

    o["electrical_health"] = (
        0.30 * battery_health
        + 0.30 * alternator_health
        + 0.20 * alternator_voltage_health
        + 0.10 * alternator_current_health
        + 0.10 * battery_current_health
    ).clip(0.0, 1.0)

    # =================================================================
    # OVERALL HEALTH
    # =================================================================

    if "anomaly_score" in o.columns:

        anomaly_penalty = (
            0.08
            * o["anomaly_score"]
            .clip(0.0, 1.0)
        )

    else:

        anomaly_penalty = 0.0

    o["overall_health"] = (
        0.27 * o["thermal_health"]
        + 0.23 * o["mechanical_health"]
        + 0.20 * o["lubrication_health"]
        + 0.17 * o["fuel_health"]
        + 0.13 * o["electrical_health"]
        - anomaly_penalty
    ).clip(0.0, 1.0)

    # -------------------------------------------------------------
    # Percentage
    # -------------------------------------------------------------

    o["health_percent"] = (
        o["overall_health"]
        * 100.0
    ).round(1)

    # -------------------------------------------------------------
    # Status
    # -------------------------------------------------------------

    o["status"] = pd.cut(
        o["overall_health"],
        [
            -0.01,
            0.40,
            0.70,
            0.90,
            1.01,
        ],
        labels=[
            "CRITICAL",
            "WARNING",
            "DEGRADING",
            "HEALTHY",
        ],
    ).astype(str)

    return o


# =====================================================================
# THRESHOLD VS AI
# =====================================================================


def threshold_vs_ai(
    df,
    detector=None,
):
    """
    Compare traditional fixed thresholds against AI anomaly detection.

    Returns:

        threshold_idx
        ai_idx
        threshold_time
        ai_time
        threshold_parameter
    """

    o = detect(
        df,
        detector=detector
    )

    # -------------------------------------------------------------
    # Fixed threshold conditions
    # -------------------------------------------------------------

    conditions = {

        "CHT":
            o["cht"]
            > THRESHOLDS["cht"],

        "EGT":
            o["egt"]
            > THRESHOLDS["egt"],

        "Oil pressure":
            o["oil_pressure"]
            < THRESHOLDS["oil_pressure"],

        "Oil temperature":
            o["oil_temp"]
            > THRESHOLDS["oil_temp"],

        "Vibration":
            o["vibration"]
            > THRESHOLDS["vibration"],

        "Fuel flow":
            o["fuel_flow"]
            > THRESHOLDS["fuel_flow"],

        "Injection timing":
            (
                o["injection_timing"]
                - 18.0
            ).abs()
            > THRESHOLDS[
                "injection_timing_deviation"
            ],
    }

    threshold_alarm = (
        pd.DataFrame(
            conditions,
            index=o.index
        )
        .any(axis=1)
    )

    # Three consecutive threshold violations.
    threshold_confirmed = (
        threshold_alarm
        .rolling(
            3,
            min_periods=3
        )
        .sum()
        >= 3
    )

    # Two of three AI anomalies.
    ai_confirmed = (
        o["anomaly"]
        .astype(int)
        .rolling(
            3,
            min_periods=3
        )
        .sum()
        >= 2
    )

    threshold_matches = (
        o.index[
            threshold_confirmed
        ]
    )

    ai_matches = (
        o.index[
            ai_confirmed
        ]
    )

    threshold_idx = (
        int(threshold_matches[0])
        if len(threshold_matches)
        else None
    )

    ai_idx = (
        int(ai_matches[0])
        if len(ai_matches)
        else None
    )

    # -------------------------------------------------------------
    # Times
    # -------------------------------------------------------------

    threshold_time = None

    if (
        threshold_idx is not None
        and "t" in o.columns
    ):

        threshold_time = float(
            o.loc[
                threshold_idx,
                "t"
            ]
        )

    ai_time = None

    if (
        ai_idx is not None
        and "t" in o.columns
    ):

        ai_time = float(
            o.loc[
                ai_idx,
                "t"
            ]
        )

    # -------------------------------------------------------------
    # Triggering parameter
    # -------------------------------------------------------------

    threshold_parameter = None

    if threshold_idx is not None:

        triggered = [
            name
            for name, condition
            in conditions.items()
            if bool(
                condition.loc[
                    threshold_idx
                ]
            )
        ]

        if triggered:

            threshold_parameter = (
                ", ".join(triggered)
            )

    return (
        threshold_idx,
        ai_idx,
        threshold_time,
        ai_time,
        threshold_parameter,
    )


# =====================================================================
# DIAGNOSIS HELPERS
# =====================================================================


def _get(row, name, default=0.0):

    try:

        value = row[name]

        if pd.isna(value):
            return default

        return float(value)

    except (
        KeyError,
        AttributeError,
        TypeError,
        ValueError,
    ):

        return default


def _evidence(
    row,
    condition_values,
):
    """
    Weighted average absolute residual.
    """

    values = []

    for column, weight in condition_values:

        if column not in row.index:
            continue

        value = abs(
            _get(
                row,
                column
            )
        )

        values.append(
            value * weight
        )

    if not values:
        return 0.0

    return float(
        np.mean(values)
    )


# =====================================================================
# TOP CONTRIBUTORS
# =====================================================================


def _top_contributors(
    row,
    n=3,
):

    contribution_map = {

        "CHT":
            "cht_dev",

        "EGT":
            "egt_dev",

        "Oil pressure":
            "oil_pressure_dev",

        "Oil temperature":
            "oil_temp_dev",

        "Fuel pressure":
            "fuel_pressure_dev",

        "Fuel flow":
            "fuel_flow_dev",

        "Air-fuel ratio":
            "air_fuel_ratio_dev",

        "Vibration":
            "vibration_dev",

        "Vibration frequency":
            "vibration_frequency_dev",

        "Injection timing":
            "injection_timing_dev",

        "Alternator health":
            "alternator_health_dev",

        "Alternator voltage":
            "alternator_voltage_dev",

        "Alternator current":
            "alternator_current_dev",

        "Battery voltage":
            "battery_voltage_dev",

        "Battery current":
            "battery_current_dev",

        "Power output":
            "power_output_dev",

        "Torque":
            "torque_dev",

        "RPM":
            "rpm_dev",

        "Engine load":
            "engine_load_dev",

        "Manifold pressure":
            "manifold_pressure_dev",

        "Air mass flow":
            "air_mass_flow_dev",
    }

    contributions = []

    for name, column in contribution_map.items():

        if column not in row.index:
            continue

        value = abs(
            _get(
                row,
                column
            )
        )

        contributions.append(
            (
                name,
                value
            )
        )

    contributions.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return contributions[:n]


# =====================================================================
# FAULT DIAGNOSIS
# =====================================================================


def diagnose(row):
    """
    Hybrid fault diagnosis.

    Uses:

        Digital Twin residuals
        +
        physical relationships
        +
        anomaly score
        +
        temporal persistence
        +
        sensor-isolation logic

    Returns:

        fault
        severity
        confidence
        top contributors
    """

    # =================================================================
    # BASE EVIDENCE
    # =================================================================

    thermal = _evidence(
        row,
        [
            ("cht_dev", 1.0),
            ("egt_dev", 1.0),
            ("oil_temp_dev", 0.5),
        ],
    )

    lubrication = _evidence(
        row,
        [
            ("oil_pressure_dev", 1.2),
            ("oil_temp_dev", 0.8),
            ("vibration_dev", 0.3),
        ],
    )

    injector = _evidence(
        row,
        [
            ("fuel_pressure_dev", 1.0),
            ("injection_timing_dev", 1.0),
            ("fuel_flow_dev", 0.8),
            ("air_fuel_ratio_dev", 0.8),
            ("egt_dev", 0.6),
            ("rpm_dev", 0.4),
        ],
    )

    mechanical = _evidence(
        row,
        [
            ("vibration_dev", 1.3),
            ("vibration_frequency_dev", 1.0),
            ("power_output_dev", 0.6),
            ("torque_dev", 0.5),
        ],
    )

    electrical = _evidence(
        row,
        [
            ("alternator_health_dev", 1.0),
            ("alternator_voltage_dev", 1.0),
            ("alternator_current_dev", 0.7),
            ("battery_voltage_dev", 0.8),
            ("battery_current_dev", 0.5),
        ],
    )

    # =================================================================
    # SENSOR DRIFT
    # =================================================================
    #
    # Your simulator specifically tests that sensor drift changes CHT
    # and EGT while physical engine state remains approximately stable.
    #
    # Therefore this gets explicit isolation logic.
    # =================================================================

    drift = 0.0

    cht_dev = _get(
        row,
        "cht_dev"
    )

    egt_dev = _get(
        row,
        "egt_dev"
    )

    drift_magnitude = (
        abs(cht_dev)
        + abs(egt_dev)
    )

    stable_physical_columns = [

        "rpm_dev",

        "oil_pressure_dev",

        "oil_temp_dev",

        "vibration_dev",

        "torque_dev",

        "power_output_dev",

        "fuel_flow_dev",

        "alternator_health_dev",
    ]

    physical_values = []

    for column in stable_physical_columns:

        if column in row.index:

            physical_values.append(
                abs(
                    _get(
                        row,
                        column
                    )
                )
            )

    if physical_values:

        physical_stability = float(
            np.mean(
                physical_values
            )
        )

        if physical_stability < 0.75:

            drift = (
                0.50
                * drift_magnitude
                + 1.0
            )

    # =================================================================
    # ENGINEERING RELATIONSHIPS
    # =================================================================

    # Thermal chain
    if (
        cht_dev > 1.0
        and egt_dev > 1.0
    ):

        thermal += 1.25

    # Lubrication chain
    if (
        _get(row, "oil_pressure_dev") < -1.0
        and _get(row, "oil_temp_dev") > 1.0
    ):

        lubrication += 1.25

    # Mechanical chain
    if (
        _get(row, "vibration_dev") > 1.5
        and _get(
            row,
            "vibration_frequency_dev"
        ) > 0.8
    ):

        mechanical += 1.25

    # Injector chain
    if (
        _get(
            row,
            "fuel_pressure_dev"
        ) < -1.0
        and abs(
            _get(
                row,
                "injection_timing_dev"
            )
        ) > 1.0
    ):

        injector += 1.25

    # Electrical chain
    if (
        _get(
            row,
            "alternator_health_dev"
        ) < -1.0
        and _get(
            row,
            "alternator_voltage_dev"
        ) < -1.0
    ):

        electrical += 1.25

    # =================================================================
    # EVIDENCE TABLE
    # =================================================================

    evidence = {

        "THERMAL DEGRADATION / OVERHEATING":
            thermal,

        "LUBRICATION DEGRADATION":
            lubrication,

        "INJECTOR / FUEL ABNORMALITY":
            injector,

        "MECHANICAL / VIBRATION ABNORMALITY":
            mechanical,

        "ELECTRICAL DEGRADATION":
            electrical,

        "SENSOR DRIFT":
            drift,
    }

    ranked = sorted(
        evidence.items(),
        key=lambda x: x[1],
        reverse=True
    )

    fault = ranked[0][0]

    best_score = ranked[0][1]

    second_score = ranked[1][1]

    # =================================================================
    # ANOMALY INFORMATION
    # =================================================================

    anomaly_score = _get(
        row,
        "anomaly_score"
    )

    persistence = _get(
        row,
        "anomaly_persistence"
    )

    # =================================================================
    # NORMAL
    # =================================================================

    if (
        anomaly_score < 0.35
        and best_score < 1.0
    ):

        return (
            "NORMAL / NO CONFIRMED FAULT",
            "LOW",
            0.80,
            _top_contributors(row),
        )

    # =================================================================
    # UNKNOWN ABNORMAL CONDITION
    # =================================================================

    if best_score < 1.0:

        confidence = (
            0.45
            + 0.20
            * anomaly_score
        )

        confidence = float(
            np.clip(
                confidence,
                0.50,
                0.75
            )
        )

        return (
            "UNCLASSIFIED ABNORMAL OPERATING CONDITION",
            "MEDIUM",
            confidence,
            _top_contributors(row),
        )

    # =================================================================
    # CONFIDENCE
    # =================================================================

    evidence_strength = min(
        best_score / 4.0,
        1.0
    )

    separation = np.clip(
        max(
            best_score
            - second_score,
            0.0
        )
        / 2.0,
        0.0,
        1.0
    )

    confidence = (
        0.40
        + 0.25
        * evidence_strength
        + 0.20
        * anomaly_score
        + 0.10
        * separation
        + 0.05
        * persistence
    )

    confidence = float(
        np.clip(
            confidence,
            0.50,
            0.97
        )
    )

    # =================================================================
    # SEVERITY
    # =================================================================

    if (
        best_score >= 3.0
        or anomaly_score >= 0.85
    ):

        severity = "HIGH"

    elif (
        best_score >= 1.75
        or anomaly_score >= 0.65
    ):

        severity = "MEDIUM"

    else:

        severity = "LOW"

    # Sensor drift is not treated like a direct engine mechanical
    # failure.
    if fault == "SENSOR DRIFT":

        severity = "LOW"

    return (
        fault,
        severity,
        confidence,
        _top_contributors(row),
    )


# =====================================================================
# RUL ESTIMATION
# =====================================================================


def rul(df):
    """
    Prototype Remaining Useful Life estimation.

    Uses:

        smoothed overall health
        +
        robust Huber regression

    Assumption:

        60 simulation samples = 1 equivalent operating hour.

    This is a prototype only. Real RUL requires historical run-to-failure
    data.
    """

    if len(df) < 12:

        return (
            (180, 240),
            0.40
        )

    # -------------------------------------------------------------
    # Make sure health exists
    # -------------------------------------------------------------

    if "overall_health" not in df.columns:

        work = health(df)

    else:

        work = df.copy()

    # -------------------------------------------------------------
    # Clean health signal
    # -------------------------------------------------------------

    y = (
        pd.to_numeric(
            work["overall_health"],
            errors="coerce"
        )
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .interpolate()
        .bfill()
        .ffill()
        .to_numpy()
    )

    n = len(y)

    # Last 90 samples.
    window = min(
        90,
        n
    )

    recent_health = y[-window:]

    x = np.arange(
        window,
        dtype=float
    ).reshape(
        -1,
        1
    )

    # -------------------------------------------------------------
    # Smooth health before robust regression
    # -------------------------------------------------------------

    smoothed = (
        pd.Series(
            recent_health
        )
        .ewm(
            span=min(
                15,
                max(
                    5,
                    window // 5
                )
            ),
            adjust=False
        )
        .mean()
        .to_numpy()
    )

    # -------------------------------------------------------------
    # Huber regression
    #
    # Better than ordinary LinearRegression when telemetry has
    # occasional spikes.
    # -------------------------------------------------------------

    try:

        model = HuberRegressor()

        model.fit(
            x,
            smoothed
        )

        slope = float(
            model.coef_[0]
        )

    except Exception:

        slope = 0.0

    current_health = float(
        y[-1]
    )

    # Prototype end-of-life health.
    eol_health = 0.40

    samples_per_hour = 60.0

    # -------------------------------------------------------------
    # Already below EOL threshold
    # -------------------------------------------------------------

    if current_health <= eol_health:

        return (
            (0.0, 1.0),
            0.75
        )

    # -------------------------------------------------------------
    # No degradation trend
    # -------------------------------------------------------------

    if slope >= -1e-5:

        return (
            (180.0, 240.0),
            0.45
        )

    # -------------------------------------------------------------
    # Estimate samples to EOL
    # -------------------------------------------------------------

    samples_to_eol = max(
        (
            current_health
            - eol_health
        )
        / abs(slope),
        0.0
    )

    hours = (
        samples_to_eol
        / samples_per_hour
    )

    # -------------------------------------------------------------
    # Uncertainty
    # -------------------------------------------------------------

    trend_strength = min(
        abs(slope) * 40.0,
        1.0
    )

    spread_fraction = (
        0.25
        - 0.12
        * trend_strength
    )

    spread = max(
        hours
        * spread_fraction,
        1.0
    )

    lower = max(
        0.0,
        hours - spread
    )

    upper = (
        hours
        + spread
    )

    # -------------------------------------------------------------
    # Confidence
    # -------------------------------------------------------------

    confidence = float(
        np.clip(
            0.45
            + 0.25
            * trend_strength
            + 0.10
            * min(
                max(
                    current_health,
                    0.0
                ),
                1.0
            ),
            0.45,
            0.85
        )
    )

    return (
        (lower, upper),
        confidence
    )


# =====================================================================
# COMPLETE ANALYTICS PIPELINE
# =====================================================================


def run_analytics(
    df,
    detector=None,
):
    """
    Complete AeroTwin analytics pipeline.

        telemetry
            ↓
        twin features
            ↓
        temporal features
            ↓
        anomaly detection
            ↓
        health
    """

    o = df.copy()

    # -------------------------------------------------------------
    # Digital Twin
    # -------------------------------------------------------------

    o = add_twin_features(o)

    # -------------------------------------------------------------
    # Temporal features
    # -------------------------------------------------------------

    o = add_derived_features(o)

    # -------------------------------------------------------------
    # AI anomaly detection
    # -------------------------------------------------------------

    o = detect(
        o,
        detector=detector
    )

    # -------------------------------------------------------------
    # Health
    # -------------------------------------------------------------

    o = health(o)

    return o


# =====================================================================
# COMPLETE SCENARIO TEST
# =====================================================================


def run_scenario_test():

    from simulator import SCENARIOS

    print()
    print("=" * 90)
    print("AeroTwin Analytics Scenario Test")
    print("=" * 90)

    print()
    print("Training healthy baseline...")

    detector = fit_detector(
        n_baseline=1200,
        seed=123
    )

    print("Detector trained successfully.")

    print()

    for scenario in SCENARIOS:

        print("-" * 90)

        print(
            f"Scenario: {scenario}"
        )

        try:

            telemetry = simulate(
                scenario,
                300,
                7
            )

            result = run_analytics(
                telemetry,
                detector
            )

            latest = result.iloc[-1]

            fault, severity, confidence, contributors = diagnose(
                latest
            )

            anomaly_rate = float(
                result.tail(20)[
                    "anomaly"
                ].mean()
            )

            print(
                f"Anomaly rate: "
                f"{anomaly_rate:.2f}"
            )

            print(
                f"Health: "
                f"{latest['health_percent']:.1f}%"
            )

            print(
                f"Status: "
                f"{latest['status']}"
            )

            print(
                f"Anomaly score: "
                f"{latest['anomaly_score']:.3f}"
            )

            print(
                f"Diagnosis: "
                f"{fault}"
            )

            print(
                f"Severity: "
                f"{severity}"
            )

            print(
                f"Confidence: "
                f"{confidence:.2f}"
            )

            print(
                f"Top contributors: "
                f"{contributors}"
            )

        except Exception as exc:

            print(
                f"ERROR: {exc}"
            )

    print()
    print("=" * 90)
    print("Scenario testing complete.")
    print("=" * 90)


# =====================================================================
# COMPATIBILITY HELPERS
# =====================================================================


def get_detector_features(detector):
    """
    Return the features used by the trained detector.
    """

    if detector is None:
        return []

    return list(
        detector.feature_columns
    )


# =====================================================================
# MAIN
# =====================================================================


if __name__ == "__main__":

    run_scenario_test()