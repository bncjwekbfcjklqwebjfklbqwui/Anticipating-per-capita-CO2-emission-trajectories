# ============================================================
# PCA standardisé (from scratch)
# ============================================================

import numpy as np
import pandas as pd


def run_pca_standardized(df_final):
    # 1. Selection of features (robust to missing columns)
    features_all = [
        "co2_per_capita",
        "gdp_per_capita",
        "Coal",
        "Oil",
        "Gas",
        "Nuclear",
        "Hydro",
        "Wind",
        "Solar",
        "Other"
    ]

    # Keep only features that actually exist in the data
    features = [f for f in features_all if f in df_final.columns]

    if len(features) < 2:
        raise ValueError("Not enough features available for PCA.")

    df_pca = df_final.dropna(subset=features).copy()
    X = df_pca[features].values

    # 2. Standardization
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    X_standardized = (X - mu) / sigma

    # 3. Covariance matrix
    m = X_standardized.shape[0]
    C = (1 / m) * np.dot(X_standardized.T, X_standardized)

    # 4. Eigen decomposition
    eigvals, eigvecs = np.linalg.eig(C)

    # Sort eigenvalues decreasing
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # 5. Reduce to 2 principal components
    U_reduce = eigvecs[:, :2]

    # 6. Projection
    Z = np.dot(X_standardized, U_reduce)
    df_pca["PC1"] = Z[:, 0]
    df_pca["PC2"] = Z[:, 1]

    df_pca["Country"] = df_final.loc[df_pca.index, "Country"].values

    # 7. Mean per country
    df_pca_mean = (
        df_pca
        .groupby("Country")[["PC1", "PC2"]]
        .mean()
        .reset_index()
    )

    # 8. Variance explained
    explained = eigvals[:2] / eigvals.sum()

    return df_pca, df_pca_mean, eigvals, explained
