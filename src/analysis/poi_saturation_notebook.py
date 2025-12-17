# %% [markdown];
"""
# Grid-Based POI Quality Assessment Workflow

Assesses POI data quality using multi-scale neighborhood analysis of 1km census grid cells.

## Steps
1. Aggregate grid-level POI counts with multi-scale population neighborhoods
2. Fit Random Forest regression: POI_count ~ pop_local + pop_intermediate + pop_large
3. Aggregate to city level and classify by saturation quadrant
4-7. EDA, regression diagnostics, feature importance, and report generation

## Key Outputs
- **grid_counts_regress.gpkg**: Grid cells with z-scores (deviation from expected POI counts)
- **city_analysis_results.gpkg**: City-level statistics with quadrant classification
- **city_assessment_report.md**: Comprehensive markdown report

## Z-Score Interpretation
- z < 0: Fewer POIs than expected (undersaturated)
- z > 0: More POIs than expected (saturated)
- Quadrant analysis: mean z-score (level) × std z-score (variability)
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

from src import tools
from src.analysis.aggregate_grid_stats import aggregate_grid_stats

logger = tools.get_logger(__name__)

# Configuration - modify these paths as needed
BOUNDS_PATH = "temp/datasets/boundaries.gpkg"
OVERTURE_DATA_DIR = "temp/cities_data/overture"
CENSUS_PATH = "temp/Eurostat_Census-GRID_2021_V2/ESTAT_Census_2021_V2.gpkg"
OUTPUT_DIR = "src/analysis/outputs"

# %%
"""
## Step 1: Aggregate Grid-Level Statistics

Counts POIs within census grid cells and computes multi-scale population neighborhoods.
"""

output_path = Path(OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)
grid_stats_file = output_path / "grid_stats.gpkg"

if grid_stats_file.exists():
    logger.info("Skipping grid aggregation (grid_stats.gpkg exists)")
else:
    logger.info("STEP 1: Aggregating grid-level statistics")
    bounds_gdf = gpd.read_file(BOUNDS_PATH)
    census_gdf = gpd.read_file(CENSUS_PATH)

    grid_stats = aggregate_grid_stats(
        bounds_gdf=bounds_gdf,
        census_gdf=census_gdf,
        overture_data_dir=OVERTURE_DATA_DIR,
    )

    grid_stats.to_file(grid_stats_file, driver="GPKG")
    logger.info(f"Grid aggregation complete: {grid_stats_file}")

# %%
"""
## Step 2: Multiple Regression Analysis

Fit Random Forest regression per category: POI_count ~ pop_local + pop_intermediate + pop_large
Z-scores on residuals identify grids with more/fewer POIs than expected.

**Log Transform Rationale**: Both features and target are log-transformed because:
1. POI counts follow a power-law relationship with population (POI ∝ pop^β), which becomes linear in log-log space
2. Log-scale residuals are more normally distributed for count data, making z-scores statistically valid
3. Interpretation becomes multiplicative (% deviation) rather than additive (count deviation), which is more meaningful across density ranges
"""

logger.info("STEP 2: Multiple regression analysis")

grid_gdf = gpd.read_file(grid_stats_file)
poi_categories = sorted([col.replace("_count", "") for col in grid_gdf.columns if col.endswith("_count")])


def format_category_name(cat):
    """Format category names: remove underscores, title case."""
    return cat.replace("_", " ").title()


category_names = {cat: format_category_name(cat) for cat in poi_categories}

logger.info(f"Found {len(poi_categories)} POI categories, {len(grid_gdf)} grid cells")

# Initialize all output columns for all categories (even if some will be skipped)
for cat in poi_categories:
    grid_gdf[f"{cat}_residual"] = np.nan
    grid_gdf[f"{cat}_zscore"] = np.nan
    grid_gdf[f"{cat}_predicted"] = np.nan

# Store regression results
regression_results = {}

# Fit multiple regression for each category using all scales as features
for cat in poi_categories:
    poi_col = f"{cat}_count"

    # Prepare data for regression
    poi = grid_gdf[poi_col].values.astype(float)
    pop_local = grid_gdf["pop_local"].values.astype(float)
    pop_intermediate = grid_gdf["pop_intermediate"].values.astype(float)
    pop_large = grid_gdf["pop_large"].values.astype(float)

    # Use grids with valid population data AND poi > 0 for log transform
    # (grids with poi=0 will get z-scores based on prediction alone after)
    valid_mask = (poi > 0) & (pop_local > 0) & (pop_intermediate > 0) & (pop_large > 0)

    # Log-transform both features and target to linearize power-law relationships
    # This helps RF capture the full range of POI counts without underpredicting high values
    log_poi = np.log1p(poi[valid_mask])  # log(1+x) handles edge cases
    log_pop_local = np.log1p(pop_local[valid_mask])
    log_pop_intermediate = np.log1p(pop_intermediate[valid_mask])
    log_pop_large = np.log1p(pop_large[valid_mask])

    # Prepare feature matrix in log space
    X = np.column_stack([log_pop_local, log_pop_intermediate, log_pop_large])
    y = log_poi

    # Fit Random Forest in log space
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    # Get predictions in log space, then transform back for interpretable residuals
    log_predictions = model.predict(X)
    predictions = np.expm1(log_predictions)  # Back to original scale
    observed = poi[valid_mask]

    # Compute residuals in log space (more normally distributed for z-scores)
    log_residuals = log_poi - log_predictions

    # Calculate R² in log space (where model was fit)
    ss_res = np.sum(log_residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # Calculate z-scores of log-scale residuals (standardized deviation from predicted)
    residual_mean = np.mean(log_residuals)
    residual_std = np.std(log_residuals)
    z_scores = (log_residuals - residual_mean) / residual_std

    # Update grid_gdf with stats for valid grids
    grid_indices = np.where(valid_mask)[0]
    for idx, grid_idx in enumerate(grid_indices):
        grid_gdf.loc[grid_gdf.index[grid_idx], f"{cat}_residual"] = log_residuals[idx]
        grid_gdf.loc[grid_gdf.index[grid_idx], f"{cat}_zscore"] = z_scores[idx]
        grid_gdf.loc[grid_gdf.index[grid_idx], f"{cat}_predicted"] = predictions[idx]

    # Check residual normality (Shapiro-Wilk on sample if large) - important for z-score validity
    sample_size = min(5000, len(log_residuals))  # Shapiro-Wilk limited to 5000
    residual_sample = np.random.default_rng(42).choice(log_residuals, size=sample_size, replace=False)
    _, normality_p = stats.shapiro(residual_sample)

    regression_results[cat] = {
        "n_samples": valid_mask.sum(),
        "r2": r2,
        "importance_local": model.feature_importances_[0],
        "importance_intermediate": model.feature_importances_[1],
        "importance_large": model.feature_importances_[2],
        "residual_std": residual_std,
        "normality_p": normality_p,
        "model": model,
        "valid_indices": grid_indices.tolist(),
    }

    logger.info(f"{cat}: R²={r2:.4f}, n={valid_mask.sum()}")

# Save grid dataset with z-scores
output_path.mkdir(parents=True, exist_ok=True)
grid_gdf.to_file(output_path / "grid_counts_regress.gpkg", driver="GPKG")
logger.info(f"Saved grid dataset to {output_path / 'grid_counts_regress.gpkg'}")

# %%
"""
## Step 3: City-Level Aggregation

Aggregate z-score statistics per city and classify into quadrants.
"""

logger.info("\nSTEP 3: City-level aggregation and quadrant analysis")

bounds_gdf = gpd.read_file(BOUNDS_PATH)
city_gdf = bounds_gdf.copy()

for idx, row in bounds_gdf.iterrows():
    bounds_fid = row.get("bounds_fid", row.get("fid", idx))
    city_grids = grid_gdf[grid_gdf["bounds_fid"] == bounds_fid]

    if len(city_grids) == 0:
        continue

    city_gdf.loc[idx, "total_grids"] = len(city_grids)

    for cat in poi_categories:
        zscores = city_grids[f"{cat}_zscore"].dropna()

        # Z-score statistics
        if len(zscores) > 0:
            city_gdf.loc[idx, f"{cat}_z_mean"] = zscores.mean()
            city_gdf.loc[idx, f"{cat}_z_std"] = zscores.std()
        else:
            city_gdf.loc[idx, f"{cat}_z_mean"] = np.nan
            city_gdf.loc[idx, f"{cat}_z_std"] = np.nan

# Calculate average z-score across all categories for each city
for idx in city_gdf.index:
    z_mean_cols = [f"{cat}_z_mean" for cat in poi_categories if f"{cat}_z_mean" in city_gdf.columns]
    z_values = city_gdf.loc[idx, z_mean_cols].dropna()

# Quadrant Analysis: Classify cities by saturation level (mean) × variability (std)
# Compute both std approaches for comparison
z_mean_cols = [f"{cat}_z_mean" for cat in poi_categories]
z_std_cols = [f"{cat}_z_std" for cat in poi_categories]
city_gdf["overall_z_mean_category"] = city_gdf[z_mean_cols].mean(axis=1)
# Category variability: std across category means (how consistent across POI types?)
city_gdf["overall_z_std_category"] = city_gdf[z_mean_cols].std(axis=1)


# Assign quadrant for a given mean/std pair relative to a std threshold
def assign_quadrant(mean_z, std_z, std_threshold):
    if pd.isna(mean_z) or pd.isna(std_z):
        return np.nan
    if mean_z < 0:
        return "Consistently Undersaturated" if std_z < std_threshold else "Variable Undersaturated"
    return "Consistently Saturated" if std_z < std_threshold else "Variable Saturated"


# Compute global std threshold: median of all per-category stds pooled together
all_category_stds = []
for cat in poi_categories:
    std_col = f"{cat}_z_std"
    if std_col in city_gdf.columns:
        all_category_stds.extend(city_gdf[std_col].dropna().tolist())
global_std_threshold = np.median(all_category_stds)
logger.info(f"\nGlobal std threshold for quadrant classification: {global_std_threshold:.4f}")

# Per-category quadrants (using global threshold for consistency)
for cat in poi_categories:
    mean_col, std_col = f"{cat}_z_mean", f"{cat}_z_std"
    city_gdf[f"{cat}_quadrant"] = city_gdf.apply(
        lambda row, mc=mean_col, sc=std_col: assign_quadrant(row[mc], row[sc], global_std_threshold), axis=1
    )

# Between-category quadrant (using global threshold)
city_gdf["between_category_quadrant"] = city_gdf.apply(
    lambda row: assign_quadrant(row["overall_z_mean_category"], row["overall_z_std_category"], global_std_threshold),
    axis=1,
)

# Log quadrant summary
logger.info("\nQuadrant Summary (between-category variability):")
for quadrant, count in city_gdf["between_category_quadrant"].value_counts().items():
    logger.info(f"  {quadrant}: {count} cities")

# Save results
results_gpkg_path = output_path / "city_analysis_results.gpkg"
city_gdf.to_file(results_gpkg_path, driver="GPKG")
logger.info(f"\nSaved: {results_gpkg_path}")

# Quadrant visualization - 4x3 grid: 11 categories + 1 between-category summary
fig, axes = plt.subplots(4, 3, figsize=(12, 14))
axes = axes.flatten()
city_quadrant = city_gdf[city_gdf["overall_z_mean_category"].notna()].copy()

# Compute global std threshold: median of all per-category stds pooled together
all_stds = []
for cat in poi_categories:
    std_col = f"{cat}_z_std"
    if std_col in city_quadrant.columns:
        all_stds.extend(city_quadrant[std_col].dropna().tolist())
global_std_median = np.median(all_stds)

# Compute consistent axis limits across all plots
all_means = []
for cat in poi_categories:
    mean_col = f"{cat}_z_mean"
    if mean_col in city_quadrant.columns:
        all_means.extend(city_quadrant[mean_col].dropna().tolist())
x_abs_max = max(abs(np.percentile(all_means, 1)), abs(np.percentile(all_means, 99))) * 1.1
y_max = np.percentile(all_stds, 99) * 1.1


def plot_quadrant_scatter(ax, x_data, y_data, std_med, title):
    """Plot quadrant scatter with colors based on position relative to dividers."""
    for i in range(len(x_data)):
        x, y = x_data.iloc[i], y_data.iloc[i]
        if pd.isna(x) or pd.isna(y):
            continue
        if x < 0:
            color = "#d62728" if y < std_med else "#ff7f0e"
        else:
            color = "#2ca02c" if y < std_med else "#1f77b4"
        ax.scatter(x, y, c=color, alpha=0.6, edgecolors="black", linewidths=0.3, s=25)

    ax.axhline(y=std_med, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlim(-x_abs_max, x_abs_max)  # Center zero
    ax.set_ylim(0, y_max)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)


# Plot each category individually (mean vs std for that category)
for i, cat in enumerate(poi_categories):
    ax = axes[i]
    x_col, y_col = f"{cat}_z_mean", f"{cat}_z_std"
    if x_col in city_quadrant.columns and y_col in city_quadrant.columns:
        x_data = city_quadrant[x_col].dropna()
        y_data = city_quadrant.loc[x_data.index, y_col]
        plot_quadrant_scatter(ax, x_data, y_data, global_std_median, category_names[cat])
        ax.set_xlabel("Mean Z", fontsize=8)
        ax.set_ylabel("Std Z", fontsize=8)

# Final panel: Between-category summary (mean across cats vs std across cat means)
ax_summary = axes[len(poi_categories)]
between_cat_std_data = city_quadrant["overall_z_std_category"].dropna()
between_cat_std_median = between_cat_std_data.median()
between_cat_y_max = np.percentile(between_cat_std_data, 99) * 1.1

# Plot with its own y-axis range
for i in range(len(city_quadrant)):
    x = city_quadrant["overall_z_mean_category"].iloc[i]
    y = city_quadrant["overall_z_std_category"].iloc[i]
    if pd.isna(x) or pd.isna(y):
        continue
    if x < 0:
        color = "#d62728" if y < between_cat_std_median else "#ff7f0e"
    else:
        color = "#2ca02c" if y < between_cat_std_median else "#1f77b4"
    ax_summary.scatter(x, y, c=color, alpha=0.6, edgecolors="black", linewidths=0.3, s=25)

ax_summary.axhline(y=between_cat_std_median, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax_summary.axvline(x=0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax_summary.set_xlim(-x_abs_max, x_abs_max)
ax_summary.set_ylim(0, between_cat_y_max)
ax_summary.set_title(
    f"Between Categories\n(std threshold = {between_cat_std_median:.2f})", fontsize=9, fontweight="bold"
)
ax_summary.tick_params(labelsize=7)
ax_summary.grid(True, alpha=0.3)
ax_summary.set_xlabel("Mean Z (across cats)", fontsize=8)
ax_summary.set_ylabel("Std Z (between cats)", fontsize=8)

# Hide any unused subplots
for i in range(len(poi_categories) + 1, len(axes)):
    axes[i].set_visible(False)

plt.suptitle(
    f"City Quadrant Analysis by POI Category\n(Global std threshold = {global_std_median:.2f})",
    fontsize=12,
    fontweight="bold",
)
plt.tight_layout()

viz_path = output_path / "city_quadrant_analysis.png"
plt.savefig(viz_path, dpi=150, bbox_inches="tight")
plt.show()

logger.info(f"\nSaved quadrant analysis to {viz_path}")

# %%
"""
## Step 4: Exploratory Data Analysis (EDA)

Model fit and z-score distribution visualization.
"""

logger.info("\nSTEP 4: Exploratory data analysis")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: R² values by category
ax = axes[0]
r2_values = [regression_results[cat]["r2"] for cat in poi_categories if not np.isnan(regression_results[cat]["r2"])]
ax.bar(range(len(r2_values)), r2_values, color="green", alpha=0.7, edgecolor="black")
ax.set_xticks(range(len(r2_values)))
ax.set_xticklabels(
    [category_names[cat] for cat in poi_categories if not np.isnan(regression_results[cat]["r2"])],
    rotation=45,
    ha="right",
)
ax.set_ylabel("R² Value", fontsize=11)
ax.set_title("Model Fit by Category", fontsize=12, fontweight="bold")
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3, axis="y")

# 2. Z-Score distribution by category (box plot)
ax = axes[1]
zscore_data = []
valid_categories = []
for cat in poi_categories:
    zscore_col = f"{cat}_zscore"
    if zscore_col in grid_gdf.columns:
        zscores = grid_gdf[zscore_col].dropna()
        if len(zscores) > 0:
            zscore_data.append(zscores.values)
            valid_categories.append(cat)

if len(zscore_data) > 0:
    # Create a dataframe for seaborn boxplot
    z_df_list = []
    for i, cat in enumerate(valid_categories):
        for z_val in zscore_data[i]:
            z_df_list.append({"Category": category_names[cat], "Z-Score": z_val})
    z_df = pd.DataFrame(z_df_list)

    sns.boxplot(data=z_df, x="Category", y="Z-Score", ax=ax, color="steelblue", fliersize=0)
    # Set y-axis limits to focus on main distribution (exclude extreme outliers)
    q1 = z_df["Z-Score"].quantile(0.05)
    q3 = z_df["Z-Score"].quantile(0.95)
    iqr = q3 - q1
    ax.set_ylim(q1 - 0.5 * iqr, q3 + 0.5 * iqr)

    ax.set_ylabel("Z-Score", fontsize=11)
    ax.set_title("Z-Score Distribution by Category", fontsize=12, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
else:
    ax.text(0.5, 0.5, "No z-score data available", ha="center", va="center", transform=ax.transAxes)

plt.tight_layout()
plt.savefig(output_path / "eda_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

logger.info(f"\nSaved EDA plots to {output_path / 'eda_analysis.png'}")

# %%
"""
## Step 5: Regression Diagnostic Plots

Predicted vs observed POI counts for each category.
"""

logger.info("\nSTEP 5: Creating regression diagnostic plots")

n_cols = 3
n_rows = int(np.ceil(len(poi_categories) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

for i, cat in enumerate(poi_categories):
    ax = axes[i]

    # Get data
    poi_col = f"{cat}_count"
    poi = grid_gdf[poi_col].values.astype(float)
    pop_local = grid_gdf["pop_local"].values.astype(float)
    pop_intermediate = grid_gdf["pop_intermediate"].values.astype(float)
    pop_large = grid_gdf["pop_large"].values.astype(float)

    # Filter for valid data
    valid_mask = (poi > 0) & (pop_local > 0) & (pop_intermediate > 0) & (pop_large > 0)

    if valid_mask.sum() < 10:
        ax.text(0.5, 0.5, f"{cat}\n(insufficient data)", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        continue

    # Prepare data in log space (matching training)
    poi_counts = poi[valid_mask]
    log_poi = np.log1p(poi_counts)  # Keep in log space for plotting
    log_pop_local = np.log1p(pop_local[valid_mask])
    log_pop_intermediate = np.log1p(pop_intermediate[valid_mask])
    log_pop_large = np.log1p(pop_large[valid_mask])

    # Reuse trained model from Step 2 (trained in log space)
    X = np.column_stack([log_pop_local, log_pop_intermediate, log_pop_large])
    model_entry = regression_results.get(cat, {})
    model = model_entry["model"]
    log_predictions = model.predict(X)

    # Set equal limits in log space
    max_pred = np.percentile(log_predictions, 99.9)
    max_obs = np.percentile(log_poi, 99.9)
    log_lim = max(max_pred, max_obs) * 1.05

    # Plot hexbin in log-log space (where model was trained)
    hb = ax.hexbin(
        log_predictions,
        log_poi,
        gridsize=25,  # 25 bins across extent
        extent=(0, log_lim, 0, log_lim),  # force grid to cover full axis range (log space)
        cmap="Reds",
        norm=LogNorm(),
        mincnt=1,
        linewidths=0.2,
        edgecolors="gray",
    )
    # Add colorbar
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label("Point count")
    cb.ax.tick_params(labelsize=8)

    # Perfect prediction line (diagonal in log space)
    ax.plot([0, log_lim], [0, log_lim], "k--", alpha=0.5, linewidth=1, label="Perfect prediction")

    # Get R² (use stored value from log-space training)
    r2 = regression_results[cat]["r2"]

    ax.set_xlabel("Predicted log(POI+1)", fontsize=10)
    ax.set_ylabel("Observed log(POI+1)", fontsize=10)
    ax.set_title(f"{category_names[cat]}\nR²={r2:.3f}", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Set equal limits based on log-space range
    ax.set_xlim(left=0, right=log_lim)
    ax.set_ylim(bottom=0, top=log_lim)

    if i == 0:
        ax.legend(fontsize=9, loc="upper left")

# Hide unused subplots
for i in range(len(poi_categories), len(axes)):
    axes[i].set_visible(False)

plt.suptitle("Multiple Regression Diagnostics: Predicted vs Observed", fontsize=14, fontweight="bold", y=1.00)
plt.tight_layout()
plt.savefig(output_path / "regression_diagnostics.png", dpi=150, bbox_inches="tight")
plt.show()

logger.info(f"Saved regression diagnostics to {output_path / 'regression_diagnostics.png'}")

# %%
"""
## Step 6: Feature Importance Analysis

Compare population scale contributions to POI prediction.
"""

logger.info("\nSTEP 6: Feature Importance Analysis")

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(poi_categories))
width = 0.25

imp_local = [regression_results[cat]["importance_local"] for cat in poi_categories]
imp_int = [regression_results[cat]["importance_intermediate"] for cat in poi_categories]
imp_large = [regression_results[cat]["importance_large"] for cat in poi_categories]

ax.bar(x - width, imp_local, width, label="Local", alpha=0.7, edgecolor="black")
ax.bar(x, imp_int, width, label="Intermediate", alpha=0.7, edgecolor="black")
ax.bar(x + width, imp_large, width, label="Large", alpha=0.7, edgecolor="black")

ax.set_xlabel("POI Category", fontsize=11)
ax.set_ylabel("Feature Importance", fontsize=11)
ax.set_title("Scale Importance for POI Prediction (Random Forest)", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([category_names[cat] for cat in poi_categories], rotation=45, ha="right")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(output_path / "feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()

logger.info(f"Saved feature importance analysis to {output_path / 'feature_importance.png'}")

# %%
"""
## Step 7: Markdown Report Generation

Generate final assessment report with statistics, visualizations, and analysis.
"""

logger.info("\nSTEP 7: Markdown Report Generation")

# Generate comprehensive markdown report
report_lines = [
    "# POI Quality Assessment Report",
    "",
    "## Executive Summary",
    "",
    f"- **Total cities analyzed**: {len(city_gdf)}",
    f"- **Total grid cells**: {len(grid_gdf)}",
    f"- **POI categories**: {len(poi_categories)}",
    "",
    "**Note**: Z-scores represent continuous deviations from expected POI counts.",
    "No arbitrary thresholds are applied. The quadrant analysis identifies cities",
    "based on their mean z-score (saturation level) and variability (consistency).",
    "",
    "---",
    "",
    "## City Quadrant Classification",
    "",
    "### Consistently Undersaturated",
    "Low POI coverage with uniform spatial distribution across categories.",
    "",
    "| City | Country | Mean Z | Between-Cat Std |",
    "|------|---------|--------|-----------------|",
]

# Between-category quadrants
quadrants = [
    ("Consistently Undersaturated", "Low & Uniform"),
    ("Variable Undersaturated", "Low & Variable"),
    ("Consistently Saturated", "High & Uniform"),
    ("Variable Saturated", "High & Variable"),
]

for quad, desc in quadrants:
    quad_cities = city_gdf[city_gdf["between_category_quadrant"] == quad].sort_values("overall_z_mean_category")

    if len(quad_cities) > 0:
        if quad != quadrants[0][0]:  # Add header for non-first quadrants
            report_lines.extend(
                [
                    "",
                    f"### {quad}",
                    desc,
                    "",
                    "| City | Country | Mean Z | Between-Cat Std |",
                    "|------|---------|--------|-----------------|",
                ]
            )

        # Add up to 30 cities per quadrant to provide more geographic examples
        for _, row in quad_cities.head(30).iterrows():
            city_label = str(row.get("label", row.get("bounds_fid", "Unknown")))
            country = str(row.get("country", "Unknown"))
            mean_z = row.get("overall_z_mean_category", 0)
            std_z = row.get("overall_z_std_category", 0)
            if pd.isna(mean_z):
                mean_z = 0
            if pd.isna(std_z):
                std_z = 0
            report_lines.append(f"| {city_label} | {country} | {mean_z:.4f} | {std_z:.4f} |")

report_lines.extend(
    [
        "",
        "---",
        "",
        "## Model Performance by Category",
        "",
        "| Category | R² Score | Local Importance | Intermediate Importance | Large Importance |",
        "|----------|----------|------------------|-------------------------|------------------|",
    ]
)

for cat in poi_categories:
    if not np.isnan(regression_results[cat]["r2"]):
        r2 = regression_results[cat]["r2"]
        imp_local = regression_results[cat]["importance_local"]
        imp_int = regression_results[cat]["importance_intermediate"]
        imp_large = regression_results[cat]["importance_large"]
        report_lines.append(f"| {category_names[cat]} | {r2:.4f} | {imp_local:.4f} | {imp_int:.4f} | {imp_large:.4f} |")

report_lines.extend(
    [
        "",
        "---",
        "",
        "## Visualizations",
        "",
        "The following visualizations have been generated to support this analysis:",
        "",
        "### Exploratory Data Analysis",
        "![EDA Analysis](eda_analysis.png)",
        "",
        "Key insights:",
        "- **Z-Score Distribution**: Distribution of z-scores across grid cells per category",
        "- **Population Distribution**: Distribution of local population across census grid cells",
        "- **Model Fit (R²)**: Model fit quality for each POI category",
        "- **City Z-Score Distribution**: Distribution of mean z-scores across cities",
        "",
        "### Feature Importance Analysis",
        "![Feature Importance](feature_importance.png)",
        "",
        "Shows which population scale (local, intermediate, large) is most predictive for each POI category.",
        "Higher values indicate the scale is more important for predicting POI distribution.",
        "",
        "### Regression Diagnostics",
        "![Regression Diagnostics](regression_diagnostics.png)",
        "",
        "Predicted vs observed POI counts for each category. Shows model fit quality and outliers.",
        "Points closer to the diagonal line indicate better predictions.",
        "",
        "### City Quadrant Analysis",
        "![City Quadrant Analysis](city_quadrant_analysis.png)",
        "",
        "12-panel visualization (4×3 grid) showing city quadrant classification by POI category:",
        "- **First 11 panels**: Per-category analysis (mean z-score vs spatial std within category)",
        "- **12th panel**: Between-category summary (mean across categories vs std between categories)",
        "",
        "Each panel uses consistent color coding for quadrants:",
        "- **Red** (bottom-left): Consistently Undersaturated",
        "- **Green** (bottom-right): Consistently Saturated",
        "- **Orange** (top-left): Variable Undersaturated",
        "- **Blue** (top-right): Variable Saturated",
        "",
        "---",
        "",
        "## Output Files",
        "",
        "### Data Files",
        "- **grid_counts_regress.gpkg**: Vector grid dataset with z-scores and predictions",
        "- **city_analysis_results.gpkg**: City-level z-score statistics and per-category + between-category quadrant classifications",
        "",
        "### Visualization Files",
        "- **eda_analysis.png**: Exploratory data analysis",
        "- **feature_importance.png**: Random Forest feature importance comparison",
        "- **regression_diagnostics.png**: Predicted vs observed plots for all categories",
        "- **city_quadrant_analysis.png**: 12-panel per-category and between-category quadrant analysis",
        "",
    ]
)

# Write report to file
report_path = output_path / "city_assessment_report.md"
with open(report_path, "w") as f:
    f.write("\n".join(report_lines))

logger.info(f"Saved report to {report_path}")

"""
## Analysis Complete
"""

logger.info("\nANALYSIS COMPLETE")
logger.info(f"Output directory: {output_path}")
logger.info("  - grid_counts_regress.gpkg: Grid z-scores")
logger.info("  - city_analysis_results.gpkg: City statistics and quadrant classification")
logger.info("  - city_assessment_report.md: Full markdown report")
logger.info("  - Visualizations: eda_analysis.png, regression_diagnostics.png,")
logger.info("    feature_importance.png, city_quadrant_analysis.png")

# %%
