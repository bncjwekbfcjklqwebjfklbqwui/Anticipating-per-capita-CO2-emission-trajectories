# =====================================================
# K-Means clustering on PCA space
# =====================================================

import pandas as pd
from sklearn.cluster import KMeans


def run_kmeans_pca(
    df_pca,
    df_pca_mean,
    n_clusters=3,
    random_state=42
):
    """
    Perform KMeans clustering on PCA space and return
    country-level CO2 clusters.

    Parameters
    ----------
    df_pca : pd.DataFrame
        Observation-level PCA results (with PC1, PC2, Country, co2_per_capita).
    df_pca_mean : pd.DataFrame
        Country-level PCA means (Country, PC1, PC2).
    n_clusters : int
        Number of clusters.
    random_state : int
        Random seed.

    Returns
    -------
    df_country : pd.DataFrame
        Country-level cluster assignment with CO2 group labels.
    """

    # -------------------------------------------------
    # 1. KMeans on country-mean PCA
    # -------------------------------------------------
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    df_pca_mean = df_pca_mean.copy()

    df_pca_mean["cluster"] = kmeans.fit_predict(
        df_pca_mean[["PC1", "PC2"]]
    )

    # -------------------------------------------------
    # 2. Assign clusters to full PCA trajectories
    # -------------------------------------------------
    df_pca = df_pca.copy()
    df_pca["cluster"] = kmeans.predict(df_pca[["PC1", "PC2"]])

    # -------------------------------------------------
    # 3. Country-level aggregation
    # -------------------------------------------------
    df_country = (
        df_pca
        .groupby("Country")
        .agg(
            co2_mean=("co2_per_capita", "mean"),
            cluster_mode=("cluster", lambda x: x.mode()[0])
        )
        .reset_index()
    )

    # -------------------------------------------------
    # 4. Non-arbitrary cluster labeling
    # -------------------------------------------------
    cluster_order = (
        df_country
        .groupby("cluster_mode")["co2_mean"]
        .mean()
        .sort_values()
    )

    cluster_mapping = {
        cluster_order.index[0]: "Low CO2",
        cluster_order.index[1]: "Medium CO2",
        cluster_order.index[2]: "High CO2",
    }

    df_country["co2_group"] = df_country["cluster_mode"].map(cluster_mapping)

    return df_country
