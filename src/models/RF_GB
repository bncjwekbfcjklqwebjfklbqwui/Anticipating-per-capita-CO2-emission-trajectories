# ============================================================
# Supervised models: Random Forest & Gradient Boosting
# Static (no temporal dynamics)
# ============================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# Random Forest (static)
# ============================================================

def run_random_forest(df_final):
    """
    Static Random Forest model:
    f(GDP per capita, energy mix) -> CO2 per capita

    Parameters
    ----------
    df_final : pd.DataFrame
        Clean country-year panel.

    Returns
    -------
    df_rf_country : pd.DataFrame
        Country-level mean table with columns:
        - Country
        - ISO3
        - y_true
        - co2_pred_rf

    metrics : dict
        RMSE, MAE, R2 computed on the full sample
        (diagnostic purpose only).
    """

    # --------------------
    # Required columns
    # --------------------
    required_cols = [
        "Country",
        "ISO3",
        "co2_per_capita",
        "gdp_per_capita",
    ]

    missing = [c for c in required_cols if c not in df_final.columns]
    if missing:
        raise ValueError(f"Missing required columns for RF: {missing}")

    # --------------------
    # Optional energy variables
    # --------------------
    energy_vars = [
        "Coal", "Oil", "Gas",
        "Nuclear", "Hydro", "Wind", "Solar", "Other",
    ]

    available_energy = [c for c in energy_vars if c in df_final.columns]

    if len(available_energy) == 0:
        raise ValueError(
            "No energy variables available for Random Forest model."
        )

    # --------------------
    # Features and target
    # --------------------
    features = ["gdp_per_capita"] + available_energy
    X = df_final[features]
    y = df_final["co2_per_capita"]

    # --------------------
    # Model estimation
    # --------------------
    rf = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y)

    # --------------------
    # Predictions
    # --------------------
    df_pred = df_final[["Country", "ISO3"]].copy()
    df_pred["y_true"] = y.values
    df_pred["co2_pred_rf"] = rf.predict(X)

    # --------------------
    # Country-level aggregation
    # --------------------
    df_rf_country = (
        df_pred
        .groupby(["Country", "ISO3"], as_index=False)
        .agg(
            y_true=("y_true", "mean"),
            co2_pred_rf=("co2_pred_rf", "mean")
        )
    )

    # --------------------
    # Diagnostics (full sample)
    # --------------------
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y, rf.predict(X)))),
        "mae": float(mean_absolute_error(y, rf.predict(X))),
        "r2": float(r2_score(y, rf.predict(X))),
    }

    return df_rf_country, metrics


# ============================================================
# Gradient Boosting (static)
# ============================================================

def run_gradient_boosting(df_final):
    """
    Static Gradient Boosting model:
    f(GDP per capita, energy mix) -> CO2 per capita

    Parameters
    ----------
    df_final : pd.DataFrame
        Clean country-year panel.

    Returns
    -------
    df_gb_country : pd.DataFrame
        Country-level mean table with columns:
        - Country
        - ISO3
        - co2_per_capita
        - co2_predicted

    metrics : dict
        RMSE, MAE, R2 computed on the full sample
        (diagnostic purpose only).
    """

    # --------------------
    # Required columns
    # --------------------
    required_cols = [
        "Country",
        "ISO3",
        "co2_per_capita",
        "gdp_per_capita",
    ]

    missing = [c for c in required_cols if c not in df_final.columns]
    if missing:
        raise ValueError(f"Missing required columns for GB: {missing}")

    # --------------------
    # Optional energy variables
    # --------------------
    energy_vars = [
        "Coal", "Oil", "Gas",
        "Nuclear", "Hydro", "Wind", "Solar", "Other",
    ]

    available_energy = [c for c in energy_vars if c in df_final.columns]

    if len(available_energy) == 0:
        raise ValueError(
            "No energy variables available for Gradient Boosting model."
        )

    # --------------------
    # Features and target
    # --------------------
    features = ["gdp_per_capita"] + available_energy
    X = df_final[features]
    y = df_final["co2_per_capita"]

    # --------------------
    # Model estimation
    # --------------------
    gb = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    gb.fit(X, y)

    # --------------------
    # Predictions
    # --------------------
    df_pred = df_final[["Country", "ISO3"]].copy()
    df_pred["co2_per_capita"] = y.values
    df_pred["co2_predicted"] = gb.predict(X)

    # --------------------
    # Country-level aggregation
    # --------------------
    df_gb_country = (
        df_pred
        .groupby(["Country", "ISO3"], as_index=False)
        .agg(
            co2_per_capita=("co2_per_capita", "mean"),
            co2_predicted=("co2_predicted", "mean")
        )
    )

    # --------------------
    # Diagnostics (full sample)
    # --------------------
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y, gb.predict(X)))),
        "mae": float(mean_absolute_error(y, gb.predict(X))),
        "r2": float(r2_score(y, gb.predict(X))),
    }

    return df_gb_country, metrics
