# ============================================================
# Main script
# Orchestration of models, evaluation and results
# ============================================================

from pathlib import Path

# --------------------
# Data
# --------------------
from src.data_loader import load_data

# --------------------
# Models
# --------------------
from src.models import (
    run_pca_standardized,
    run_kmeans_pca,
    run_random_forest,
    run_gradient_boosting,
    run_future_projection,
)

# --------------------
# Evaluation (metrics only)
# --------------------
from src.evaluation import (
    regression_metrics,
    pca_variance_table,
)

# --------------------
# Results (tables & figures)
# --------------------
from src.results import (
    save_table,
    country_prediction_table,
    rf_gb_comparison_table,
    kmeans_cluster_table,
    plot_pca_scatter,
    plot_pca_country_means,
    plot_kmeans_pca,
    plot_emission_trajectories_by_group,
)


def main():

    # =====================================================
    # Project root
    # =====================================================
    project_root = Path(__file__).resolve().parent
    print("Project root:", project_root)

    # =====================================================
    # 1. Load data
    # =====================================================
    df_final = load_data()
    print("Data loaded:", df_final.shape)

    # =====================================================
    # 2. PCA (standardized)
    # =====================================================
    print("Running standardized PCA...")
    df_pca, df_pca_mean, eigvals, explained = run_pca_standardized(df_final)
    print("PCA completed.")

    # --- Evaluation
    df_pca_variance = pca_variance_table(eigvals)
    save_table(df_pca_variance, "pca_variance_explained.csv")

    # --- Figures
    plot_pca_scatter(df_pca)
    plot_pca_country_means(df_pca_mean)

    # =====================================================
    # 3. KMeans on PCA
    # =====================================================
    print("Running KMeans on PCA space...")
    df_country_clusters = run_kmeans_pca(df_pca, df_pca_mean)
    print("KMeans completed.")

    df_cluster_table = kmeans_cluster_table(df_country_clusters)
    save_table(df_cluster_table, "kmeans_country_clusters.csv")

    # Merge PCA means with cluster labels for plotting
    df_pca_kmeans = df_pca_mean.merge(
    df_country_clusters[["Country", "cluster_mode"]],
    on="Country",
    how="left")

    plot_kmeans_pca(df_pca_kmeans)

    # =====================================================
    # 4. Random Forest (static)
    # =====================================================
    print("Running Random Forest (static)...")
    df_rf_country, rf_metrics = run_random_forest(df_final)
    print("RF model diagnostics:", rf_metrics)

    rf_eval = regression_metrics(
        y_true=df_rf_country["y_true"],
        y_pred=df_rf_country["co2_pred_rf"]
    )
    save_table(df_rf_country, "rf_country_predictions.csv")

    # =====================================================
    # 5. Gradient Boosting (static)
    # =====================================================
    print("Running Gradient Boosting (static)...")
    df_gb_country, gb_metrics = run_gradient_boosting(df_final)
    print("GB model diagnostics:", gb_metrics)

    gb_eval = regression_metrics(
        y_true=df_gb_country["co2_per_capita"],
        y_pred=df_gb_country["co2_predicted"]
    )
    save_table(df_gb_country, "gb_country_predictions.csv")

    # =====================================================
    # 6. RF vs GB comparison
    # =====================================================
    df_rf_gb_table = rf_gb_comparison_table(
        df_rf_country,
        df_gb_country
    )
    save_table(df_rf_gb_table, "rf_vs_gb_comparison.csv")

    # =====================================================
    # 7. Dynamic future projections
    # =====================================================
    print("Running dynamic future projections...")
    df_forecast, df_trajectory = run_future_projection(df_final)
    print("Future projection completed.")

    save_table(df_forecast, "future_co2_projections_2024_2030.csv")
    save_table(df_trajectory, "co2_trajectory_classification.csv")
    # =====================================================
    # 9. Emission trajectories (LOW / MEDIUM / HIGH)
    # =====================================================
    plot_emission_trajectories_by_group(df_forecast)


    # =====================================================
    # End
    # =====================================================
    print("Pipeline executed successfully.")


if __name__ == "__main__":
    main()
