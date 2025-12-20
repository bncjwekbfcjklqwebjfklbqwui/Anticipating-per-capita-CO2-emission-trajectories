# ============================================================
# Main script
# ============================================================

from data_loader.py import load_data()

from models import (
    run_pca_standardized,
    run_kmeans_pca,
    run_random_forest,
    run_gradient_boosting,
    run_future_projection
)

from evaluation import (
    eval_model,
    plot_rf_gb_comparison,
    plot_residuals,
    plot_pca_variance,
    plot_kmeans_pca
)

# ----------------------
# 1. Load data
# ----------------------

df_final = load_data()

# ----------------------
# 2. PCA standardized
# ----------------------

df_pca, df_pca_mean, eigvals, explained = run_pca_standardized(df_final)
plot_pca_variance(eigvals)

# ----------------------
# 3. KMeans on PCA
# ----------------------

df_country_clusters = run_kmeans_pca(df_pca_mean)
plot_kmeans_pca(df_pca_mean)

# ----------------------
# 4. RF & GB (static)
# ----------------------

df_rf_gb = run_random_forest(df_final)
df_gb = run_gradient_boosting(df_final)

plot_rf_gb_comparison(df_rf_gb)
plot_residuals(df_rf_gb)

# ----------------------
# 5. Future projection
# ----------------------

df_forecast, df_trajectory = run_future_projection(df_final)

# ----------------------
# 6. Save results
# ----------------------

df_country_clusters.to_csv("results/country_clusters.csv", index=False)
df_rf_gb.to_csv("results/rf_gb_comparison.csv", index=False)
df_forecast.to_csv("results/future_projection.csv", index=False)
df_trajectory.to_csv("results/trajectory_classification.csv", index=False)

print("Pipeline completed successfully.")
