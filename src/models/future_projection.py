# =====================================================
# Random Forest – CO2 per capita prediction
# =====================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# =====================================================
# 1. Load clean data
# =====================================================

df_final = pd.read_csv("data/clean/df_final.csv")

# =====================================================
# 2. Lagged features construction
# =====================================================

df_ml = df_final.sort_values(["ISO3", "Year"]).copy()

lag_vars = [
    "co2_per_capita", "gdp_per_capita",
    "Coal", "Oil", "Gas", "Nuclear",
    "Hydro", "Wind", "Solar", "Other"
]

for var in lag_vars:
    df_ml[f"{var}_lag1"] = (
        df_ml
        .groupby("ISO3")[var]
        .shift(1)
    )

df_ml = df_ml.dropna(subset=[f"{v}_lag1" for v in lag_vars])


# =====================================================
# 3. Features and target
# =====================================================

features_lag = [
    "co2_per_capita_lag1", "gdp_per_capita_lag1",
    "Coal_lag1", "Oil_lag1", "Gas_lag1", "Nuclear_lag1",
    "Hydro_lag1", "Wind_lag1", "Solar_lag1", "Other_lag1"
]

X = df_ml[features_lag]
y = df_ml["co2_per_capita"]


# =====================================================
# 4. Temporal split (STRICT)
# =====================================================

train = df_ml[df_ml["Year"] <= 2018]
val   = df_ml[(df_ml["Year"] >= 2019) & (df_ml["Year"] <= 2021)]
test  = df_ml[df_ml["Year"] >= 2022]

X_train, y_train = train[features_lag], train["co2_per_capita"]
X_val,   y_val   = val[features_lag],   val["co2_per_capita"]
X_test,  y_test  = test[features_lag],  test["co2_per_capita"]


# =====================================================
# 5. Model training
# =====================================================

rf = RandomForestRegressor(
    n_estimators=500,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)


# =====================================================
# 6. Evaluation function
# =====================================================

def eval_model(y_true, y_pred, label):
    print(label)
    print("RMSE :", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("MAE  :", mean_absolute_error(y_true, y_pred))
    print("R2   :", r2_score(y_true, y_pred))
    print("-" * 40)


# =====================================================
# 7. Validation & test
# =====================================================

y_val_pred = rf.predict(X_val)
eval_model(y_val, y_val_pred, "VALIDATION 2019–2021")

y_test_pred = rf.predict(X_test)
eval_model(y_test, y_test_pred, "TEST FUTUR 2022–2023")



# ============================================================
# Post-processing of future CO2 projections (2024–2030)
# ============================================================
# Objective:
# - Merge future CO2 forecasts with last observed (2023) covariates
# - Produce clean tables for analysis and LaTeX export
#
# Inputs:
# - df_forecast : ISO3, Country, Year, co2_pred
# - last_obs    : index = ISO3, values observed in 2023
#
# Outputs:
# - df_future   : long-format table with projections
# - table_co2   : wide-format table (Country × Year)
# ============================================================

import pandas as pd

# ------------------------------------------------------------------
# Variables assumed constant under the status quo hypothesis
# ------------------------------------------------------------------
vars_statu_quo = [
    "gdp_per_capita",
    "Coal",
    "Oil",
    "Gas",
    "Nuclear",
    "Hydro",
    "Wind",
    "Solar",
    "Other",
]

# ------------------------------------------------------------------
# Merge future projections with last observed covariates (2023)
# ------------------------------------------------------------------
df_future = df_forecast.merge(
    last_obs[vars_statu_quo].reset_index(),  # ISO3 becomes a column
    on="ISO3",
    how="left",
)

# ------------------------------------------------------------------
# Rename for consistency with forecasting outputs
# ------------------------------------------------------------------
df_future = df_future.rename(
    columns={"co2_pred": "co2_per_capita_pred"}
)

# ------------------------------------------------------------------
# Pivot table: Country × Year (for LaTeX / tables)
# ------------------------------------------------------------------
table_co2 = (
    df_future
    .pivot_table(
        index="Country",
        columns="Year",
        values="co2_per_capita_pred"
    )
    .reset_index()
)

# ------------------------------------------------------------------
# Export (example: first 6 countries)
# ------------------------------------------------------------------
table_co2_fin = table_co2.head(6)

table_co2_fin.to_csv(
    "/files/Project-DS/Projection/table_co2_fin.csv",
    index=False
)


# =====================================================
# Recursive future projection 2024–2030
# =====================================================

import pandas as pd


# =====================================================
# 1. Base state (last observed year)
# =====================================================

df_final = pd.read_csv("data/clean/df_final.csv")
df_proj_base = df_final.sort_values(["ISO3", "Year"]).copy()

last_obs = (
    df_proj_base[df_proj_base["Year"] == 2023]
    .set_index("ISO3")
)[[
    "Country", "co2_per_capita", "gdp_per_capita",
    "Coal", "Oil", "Gas", "Nuclear",
    "Hydro", "Wind", "Solar", "Other"
]]

state = last_obs.copy()


# =====================================================
# 2. Recursive projection
# =====================================================

years_future = list(range(2024, 2031))
results = []

for year in years_future:

    X_y = pd.DataFrame({
        "co2_per_capita_lag1": state["co2_per_capita"],
        "gdp_per_capita_lag1": state["gdp_per_capita"],
        "Coal_lag1": state["Coal"],
        "Oil_lag1": state["Oil"],
        "Gas_lag1": state["Gas"],
        "Nuclear_lag1": state["Nuclear"],
        "Hydro_lag1": state["Hydro"],
        "Wind_lag1": state["Wind"],
        "Solar_lag1": state["Solar"],
        "Other_lag1": state["Other"],
    }, index=state.index)

    X_y = X_y[features_lag]

    co2_pred = rf.predict(X_y)

    tmp = pd.DataFrame({
        "ISO3": state.index,
        "Country": state["Country"].values,
        "Year": year,
        "co2_pred": co2_pred
    })

    results.append(tmp)

    state = state.copy()
    state["co2_per_capita"] = co2_pred


df_future = pd.concat(results, ignore_index=True)
df_future.to_csv("results/future_projection_rf_2024_2030.csv", index=False)

# ============================================================
# Trajectory analysis: CO2 per capita (2024 → 2030)
# ============================================================

import pandas as pd


def classify_trajectory(delta, eps):
    """
    Classify CO2 trajectory between two horizons.

    Parameters
    ----------
    delta : float
        Relative change between 2024 and 2030.
    eps : float
        Tolerance threshold.

    Returns
    -------
    str : {"increase", "stable", "decrease"}
    """
    if delta > eps:
        return "increase"
    elif delta < -eps:
        return "decrease"
    else:
        return "stable"


def build_trajectory_table(df_forecast, epsilon, output_path):
    """
    Build country-level CO2 trajectory table between 2024 and 2030.

    Parameters
    ----------
    df_forecast : pd.DataFrame
        Must contain columns ["Country", "Year", "co2_pred"].
    epsilon : float
        Stability threshold.
    output_path : str
        Path to save CSV output.

    Returns
    -------
    pd.DataFrame
        Trajectory table.
    """

    # --- Extract CO2 predictions for 2024 and 2030
    co2_2024 = (
        df_forecast[df_forecast["Year"] == 2024][["Country", "co2_pred"]]
        .rename(columns={"co2_pred": "co2_2024"})
    )

    co2_2030 = (
        df_forecast[df_forecast["Year"] == 2030][["Country", "co2_pred"]]
        .rename(columns={"co2_pred": "co2_2030"})
    )

    # --- Merge horizons
    df_trajectory = co2_2024.merge(
        co2_2030,
        on="Country",
        how="inner"
    )

    # --- Relative change
    df_trajectory["delta_pct"] = (
        (df_trajectory["co2_2030"] - df_trajectory["co2_2024"])
        / df_trajectory["co2_2024"]
    )

    # --- Trajectory classification
    df_trajectory["trajectory"] = df_trajectory["delta_pct"].apply(
        lambda x: classify_trajectory(x, epsilon)
    )

    # --- Sort for readability
    df_trajectory = df_trajectory.sort_values("delta_pct")

    # --- Export
    df_trajectory[
        ["Country", "co2_2024", "co2_2030", "delta_pct", "trajectory"]
    ].to_csv(output_path, index=False)

    return df_trajectory
