#     Anticipating per capita CO2 emission trajectories

## Project Overview

This project analyses and predicts **country-level CO₂ emissions per capita** using a combination of **unsupervised** and **supervised** machine learning methods.  
The objective is twofold:

1. **Describe structural differences between countries** in terms of emissions and energy mix.
2. **Predict future CO₂ per capita trajectories (2024–2030)** using historical economic and energy data.

The pipeline is fully reproducible and implemented in **Python**, following a modular and transparent architecture.

---

## Research Question

> Can historical economic and energy data be used to anticipate future trajectories
  of per capita CO₂ emissions at the country level ?

---

## Data

### Unit of observation
- Country–year panel

### Time span
- 2000–2023 (historical)
- 2024–2030 (projections)

### Sources
- **Our World in Data (OWID)**  
  - CO₂ emissions per capita  
  - Energy consumption by source (coal, oil, gas, nuclear, renewables)
- **World Bank / Maddison Project**  
  - GDP per capita

### Key variables
- `co2_per_capita`
- `gdp_per_capita`
- Energy mix variables: `Coal`, `Oil`, `Gas`, `Nuclear`, `Hydro`, `Wind`, `Solar`, `Other`

All datasets are cleaned, harmonised (ISO3 codes), and merged into a single panel (`df_final`).

---

## Methodology

### 1. Principal Component Analysis (PCA)
- **Standardised PCA** applied to energy and economic variables.
- Used as a **descriptive tool** to visualise structural differences.
- Variance explained is reported and saved.

### 2. K-Means Clustering
- Applied on PCA country means.
- Countries are grouped into **Low / Medium / High CO₂ profiles**.
- This step is **purely descriptive**, not predictive.

### 3. Supervised Learning (Static Models)
Two ensemble models are estimated:

- **Random Forest Regressor**
- **Gradient Boosting Regressor**

Target variable:
- CO₂ emissions per capita

Features:
- GDP per capita
- Available energy mix variables

Models are evaluated using:
- RMSE
- MAE
- R²

Evaluation is diagnostic only (no causal interpretation).

### 4. Dynamic Forecasting (Recursive Projection)
- A **lagged Random Forest** model is used.
- One-year lags of CO₂, GDP, and energy variables are constructed.
- Predictions are generated recursively from **2024 to 2030**.
- Countries are classified into:
  - Increase
  - Stable
  - Decrease  
  based on percentage change over the horizon.

---

## Project Structure

Anticipating-per-capita-CO2-emission-trajectories/
├── main.py                  # Entry point (pipeline orchestration)
├── environment.yml          # Reproducible Conda environment
│
├── data/
│   ├── raw/                 # Raw input data
│   └── clean/               # Cleaned datasets
│
├── src/
│   ├── data_loader.py       # Data loading & preprocessing
│   ├── evaluation.py        # Quantitative evaluation only
│   ├── results.py           # Tables & figures (saved to /results)
│   │
│   ├── models/
│   │   ├── pca_standardized.py
│   │   ├── kmean_pca.py
│   │   ├── RF_GB.py
│   │   ├── future_projection.py
│   │   └── __init__.py      # REQUIRED (exports models)
│   │
│   └── __init__.py          # REQUIRED (marks src as a package)
│
├── results/
│   ├── tables/              # CSV outputs
│   └── figures/             # PNG figures
│
└── notebooks/               # (optional) exploratory notebooks



---

## Results

When running `main.py`, the following outputs are automatically generated:

### Tables (`results/tables/`)
- PCA variance explained
- Country-level RF predictions
- Country-level GB predictions
- RF vs GB comparison
- K-Means country clusters
- CO₂ forecasts (2024–2030)
- Trajectory classification

### Figures (`results/figures/`)
- PCA scatter (all observations)
- PCA scatter (country means)
- K-Means clusters on PCA space
- CO₂ trajectory plots for:
  - Low emitters
  - Medium emitters
  - High emitters

---

## Reproducibility

## How to run the project

### Step 1 — Download the project

1. Go to the GitHub repository
2. Click on **Code**
3. Click on **Download ZIP**
4. Unzip the folder on your computer

---

### Step 2 — Run the pipeline

Open a terminal (bash), move to the project folder, and run:

```bash
python main.py
