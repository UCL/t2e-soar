# %% [markdown]
"""
# Grid-Based POI Quality Assessment Workflow

This notebook implements a grid-based approach to assessing POI data quality
using multi-scale neighborhood analysis of 1km census grid cells.

## Overview
1. Aggregate grid-level statistics (direct POI counts, no buffering)
2. Fit multiple regression using all 3 population scales as simultaneous features
3. Identify underserved areas based on residuals
4. Rank cities by coverage deficits

## Methodology
- Use census 1km grid cells completely contained within city boundaries
- Count POIs directly within grid cells (no buffering)
- Compute 3 population scales: 1x1 (local), 3x3 (intermediate), 5x5 (large) neighborhoods
- Fit single multiple regression per category: POI ~ log(pop_local) + log(pop_intermediate) + log(pop_large)
- Calculate z-scores on residuals to identify underserved areas (users can set their own threshold)
- Neighborhoods computed on full grid BEFORE filtering by boundaries (complete context)
- All grid cells included (no minimum population threshold)

## Key Advantages
- **Single unified model**: All scales used simultaneously to predict POI distribution
- **Reveals dominant scale**: Coefficient magnitudes show which scale(s) drive POI location
- **Accounts for multicollinearity**: Joint fitting extracts true independent effects
- **Residual-based flagging**: Identifies grids with lower POIs than predicted by all-scales model
- **Direct grid counting**: Avoids geometric complexity of buffering
- **Complete spatial context**: Neighborhoods computed before boundary filtering

## Usage
Run all cells sequentially. Modify the configuration paths in the next cell as needed.
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src import tools
from src.analysis.modules.aggregate_grid_stats import aggregate_grid_stats

logger = tools.get_logger(__name__)

# Configuration - modify these paths as needed
BOUNDS_PATH = "temp/datasets/boundaries.gpkg"
OVERTURE_DATA_DIR = "temp/cities_data/overture"
CENSUS_PATH = "temp/Eurostat_Census-GRID_2021_V2/ESTAT_Census_2021_V2.gpkg"
OUTPUT_DIR = "src/analysis/output_grid"

# %%
"""
## Step 1: Aggregate Grid-Level Statistics

Counts POIs directly within census grid cells (no buffering).
Computes multi-scale population neighborhoods on full grid before filtering by boundaries.
Keeps all grid cells (no minimum population threshold).

**Multi-Scale Neighborhoods:**
- `pop_local`: Population in single grid cell (1x1)
- `pop_intermediate`: Population in 3x3 neighborhood (intermediate scale)
- `pop_large`: Population in 5x5 neighborhood (large scale)

These scales capture how POI distribution varies with neighborhood density,
enabling analysis of multi-scale agglomeration effects.
"""

output_path = Path(OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)
grid_stats_file = output_path / "grid_stats.gpkg"

if grid_stats_file.exists():
    logger.info("Skipping grid aggregation step (grid_stats.gpkg exists)")
else:
    logger.info("STEP 1: Aggregating grid-level statistics")
    aggregate_grid_stats(
        BOUNDS_PATH,
        OVERTURE_DATA_DIR,
        CENSUS_PATH,
        OUTPUT_DIR,
    )
    logger.info("Grid aggregation complete")
    logger.info(f"  Output: {grid_stats_file}")

# %%
"""
## Step 2: Multiple Regression Analysis

Fit a single multiple regression model using all three population scales simultaneously as features.
This reveals which scale(s) best predict POI distribution and enables residual-based flagging.

**Model Specification:**
- Response: POI_count (raw POI counts)
- Features: pop_local, pop_intermediate, pop_large (raw population counts)
- Model: Random Forest Regressor (handles non-linearity and scale differences automatically)
- **Advantages**:
  - Captures complex non-linear relationships without transformation
  - Naturally handles nested population scales
  - No manual feature engineering needed
  - Interpretable feature importance values

**Interpretation of Coefficients:**
- Coefficients show **independent effects** after controlling for other scales
- Nested structure: large ⊃ intermediate ⊃ local → high multicollinearity expected
- Negative coefficient = "After controlling for other scales, this scale shows inverse relationship"
  - This is mathematically valid; means the effect is captured elsewhere
  - E.g., if local is positive and large is negative: local effect is the primary driver
- **Focus on R² for model fit, and z-scores (residuals) for practical utility**

**Z-Score Interpretation:**
- Z-score = (residual - mean) / std_dev = standardized deviation from predicted value
- Negative z-score = undersaturated (fewer POIs than predicted)
- Positive z-score = fully saturated (more POIs than predicted)
- |z| < 1: Normal (within 1 std dev)
- |z| < 2: Expected (within 2 std devs, covers ~95% of normal distribution)
- |z| > 2: Unusual (beyond 2 std devs)
- Users can filter based on their own threshold (e.g., z < -1.5, z < -2.0, etc.)

**Why All Scales Together:**
- Eliminates scale-specific comparisons; uses holistic prediction
- Accounts for correlation between scales
- Single regression model per category
- Residuals and z-scores indicate observed POI count deviation from all-scales prediction
"""

logger.info("STEP 2: Multiple regression analysis (all scales as features)")

# Load grid statistics
grid_gdf = gpd.read_file(grid_stats_file)

# Get list of POI categories (columns ending with "_count")
poi_categories = sorted([col.replace("_count", "") for col in grid_gdf.columns if col.endswith("_count")])

logger.info(f"Found {len(poi_categories)} POI categories")
logger.info(f"Total grid cells: {len(grid_gdf)}")

# Initialize all output columns for all categories (even if some will be skipped)
for cat in poi_categories:
    grid_gdf[f"{cat}_flagged"] = False
    grid_gdf[f"{cat}_residual"] = np.nan
    grid_gdf[f"{cat}_zscore"] = np.nan
    grid_gdf[f"{cat}_predicted"] = np.nan

# Store regression results
regression_results = {}
epsilon = 1e-6

# Fit multiple regression for each category using all scales as features
logger.info("\nFitting multiple regression models (all 3 scales as features):")
logger.info("=" * 110)

for cat in poi_categories:
    poi_col = f"{cat}_count"

    # Prepare data for regression
    poi = grid_gdf[poi_col].values.astype(float)
    pop_local = grid_gdf["pop_local"].values.astype(float)
    pop_intermediate = grid_gdf["pop_intermediate"].values.astype(float)
    pop_large = grid_gdf["pop_large"].values.astype(float)

    # Only use grids with valid POI data AND all populations > 0
    valid_mask = (poi > 0) & (pop_local > 0) & (pop_intermediate > 0) & (pop_large > 0)
    n_valid = valid_mask.sum()

    if n_valid < 10:
        logger.info(f"{cat:30s}: SKIPPED (only {n_valid} grids with complete data)")
        regression_results[cat] = {
            "n_samples": n_valid,
            "r2": np.nan,
            "importance_local": np.nan,
            "importance_intermediate": np.nan,
            "importance_large": np.nan,
        }
        continue

    # Use raw POI counts and populations for regression
    poi_counts = poi[valid_mask]
    pop_local_counts = pop_local[valid_mask]
    pop_intermediate_counts = pop_intermediate[valid_mask]
    pop_large_counts = pop_large[valid_mask]

    # Prepare feature matrix: [pop_local, pop_intermediate, pop_large]
    X = np.column_stack([pop_local_counts, pop_intermediate_counts, pop_large_counts])
    y = poi_counts

    # Fit Random Forest (handles non-linearity and interactions automatically)
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    # Get predictions and residuals (on log scale)
    predictions = model.predict(X)
    residuals = y - predictions

    # Calculate R²
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # Calculate z-scores of log-scale residuals (standardized deviation from predicted)
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)
    z_scores = (residuals - residual_mean) / residual_std if residual_std > 0 else np.zeros(len(residuals))

    # Update grid_gdf with stats for valid grids
    grid_indices = np.where(valid_mask)[0]
    for idx, grid_idx in enumerate(grid_indices):
        grid_gdf.loc[grid_gdf.index[grid_idx], f"{cat}_residual"] = residuals[idx]
        grid_gdf.loc[grid_gdf.index[grid_idx], f"{cat}_zscore"] = z_scores[idx]
        grid_gdf.loc[grid_gdf.index[grid_idx], f"{cat}_predicted"] = predictions[idx]

    regression_results[cat] = {
        "n_samples": n_valid,
        "r2": r2,
        "importance_local": model.feature_importances_[0],
        "importance_intermediate": model.feature_importances_[1],
        "importance_large": model.feature_importances_[2],
    }

    logger.info(f"{cat}: R²={r2:.4f}, n={n_valid}")

# Save grid results with flagged columns
grid_gdf.to_file(output_path / "grid_multiscale.gpkg", driver="GPKG")
logger.info(f"\nSaved grid results to {output_path / 'grid_multiscale.gpkg'}")

# Log summary of feature importances
logger.info("=" * 110)
logger.info("RANDOM FOREST FEATURE IMPORTANCES")
logger.info("=" * 110)
logger.info("Higher importance = feature contributes more to predictions")
logger.info("=" * 110)
logger.info(f"{'Category':<30s} {'Local':<12s} {'Intermediate':<12s} {'Large':<12s} {'R²':<8s}")
logger.info("-" * 85)

for cat in poi_categories:
    try:
        imp_local = regression_results[cat]["importance_local"]
        imp_int = regression_results[cat]["importance_intermediate"]
        imp_large = regression_results[cat]["importance_large"]
        r2 = regression_results[cat]["r2"]

        local_str = f"{imp_local:.4f}" if not np.isnan(imp_local) else "N/A"
        int_str = f"{imp_int:.4f}" if not np.isnan(imp_int) else "N/A"
        large_str = f"{imp_large:.4f}" if not np.isnan(imp_large) else "N/A"
        r2_str = f"{r2:.3f}" if not np.isnan(r2) else "N/A"

        logger.info(f"{cat:<30s} {local_str:<12s} {int_str:<12s} {large_str:<12s} {r2_str:<8s}")
    except (KeyError, TypeError):
        pass

# %%
"""
## Step 3: Results Summary

Aggregate grid-level results to city level and generate summary statistics.
"""

logger.info("\n" + "=" * 80)
logger.info("RESULTS SUMMARY")
logger.info("=" * 80)

# Summary by category - z-score statistics
logger.info("\nZ-SCORE STATISTICS BY CATEGORY:")
for cat in poi_categories:
    zscore_col = f"{cat}_zscore"
    valid_zscores = grid_gdf[zscore_col].dropna()

    if len(valid_zscores) > 0:
        logger.info(
            f"{cat}: mean={valid_zscores.mean():.4f}, median={valid_zscores.median():.4f}, "
            f"std={valid_zscores.std():.4f}, min={valid_zscores.min():.4f}, max={valid_zscores.max():.4f}"
        )

# City-level aggregation
city_results = []
bounds_gdf = gpd.read_file(BOUNDS_PATH)

for idx, row in bounds_gdf.iterrows():
    bounds_fid = row.get("bounds_fid", row.get("fid", idx))
    city_grids = grid_gdf[grid_gdf["bounds_fid"] == bounds_fid]

    if len(city_grids) == 0:
        continue

    city_result = {"bounds_fid": bounds_fid, "total_grids": len(city_grids)}

    # Per-category z-score based metrics
    for cat in poi_categories:
        zscore_col = f"{cat}_zscore"
        zscores = city_grids[zscore_col].dropna()

        if len(zscores) > 0:
            city_result[f"{cat}_z_mean"] = zscores.mean()
            city_result[f"{cat}_z_median"] = zscores.median()
            city_result[f"{cat}_z_std"] = zscores.std()
            city_result[f"{cat}_z_min"] = zscores.min()
            city_result[f"{cat}_z_max"] = zscores.max()
        else:
            # No data for this category in this city
            city_result[f"{cat}_z_mean"] = np.nan
            city_result[f"{cat}_z_median"] = np.nan
            city_result[f"{cat}_z_std"] = np.nan
            city_result[f"{cat}_z_min"] = np.nan
            city_result[f"{cat}_z_max"] = np.nan

    city_results.append(city_result)

city_df = pd.DataFrame(city_results)
city_df.to_csv(output_path / "city_results.csv", index=False)

logger.info(f"\nCities analyzed: {len(city_df)}")
logger.info(f"Saved city results to {output_path / 'city_results.csv'}")

# %%
"""
## Step 4: City-Level Data Quality Assessment

Summarize z-score statistics by city for understanding coverage patterns.
"""

logger.info("\nSTEP 4: City-level data quality assessment")
logger.info("\n" + "=" * 100)
logger.info("CITY DATA QUALITY BY Z-SCORE THRESHOLDS")
logger.info("=" * 100)

# Create city summary for geopackage
city_quality = []

for bounds_fid in sorted(city_df["bounds_fid"].unique()):
    city_row = city_df[city_df["bounds_fid"] == bounds_fid].iloc[0]

    quality_row = {
        "bounds_fid": bounds_fid,
        "total_grids": int(city_row["total_grids"]),
    }

    # Add all z-score based metrics from city_row
    for col in city_row.index:
        if col not in ["bounds_fid", "total_grids"]:
            quality_row[col] = city_row[col]

    city_quality.append(quality_row)

city_quality_df = pd.DataFrame(city_quality)

# For ranking, use average z-score mean across all categories
z_means = []
for cat in poi_categories:
    col_name = f"{cat}_z_mean"
    if col_name in city_quality_df.columns:
        z_means.append(city_quality_df[col_name].values)

if z_means:
    city_quality_df["avg_z_mean"] = np.mean(np.array(z_means), axis=0)
    city_quality_df = city_quality_df.sort_values("avg_z_mean", ascending=True)
else:
    city_quality_df["avg_z_mean"] = 0.0

# Load boundaries to create geopackage and get city names/countries
bounds_gdf = gpd.read_file(BOUNDS_PATH)
city_quality_geo = bounds_gdf.merge(city_quality_df, left_on="bounds_fid", right_on="bounds_fid", how="inner")

# Extract label and country columns for reporting
city_names_df = bounds_gdf[["bounds_fid", "label", "country"]].copy()
city_quality_df = city_quality_df.merge(city_names_df, left_on="bounds_fid", right_on="bounds_fid", how="left")

# Save as geopackage
quality_gpkg_path = output_path / "city_quality_ranking.gpkg"
city_quality_geo.to_file(quality_gpkg_path, driver="GPKG")
logger.info(f"\nSaved city quality ranking to {quality_gpkg_path}")

# Also save CSV for reference
city_quality_df.to_csv(output_path / "city_quality_ranking.csv", index=False)
logger.info(f"Saved quality ranking CSV to {output_path / 'city_quality_ranking.csv'}")

# Log quality ranking summary
logger.info("\n" + "-" * 100)
logger.info(f"{'City':<30s} {'Country':<12s} {'Grids':<8s} {'Avg Z-Mean':<12s}")
logger.info("-" * 100)

for _, row in city_quality_df.iterrows():
    avg_z = row.get("avg_z_mean", 0)
    city_label = str(row.get("label", row.get("bounds_fid", "Unknown")))
    country = str(row.get("country", "Unknown"))
    logger.info(f"{city_label:<30s} {country:<12s} {int(row['total_grids']):<8d} {avg_z:>10.4f}")

logger.info("=" * 100)
logger.info("Z-SCORE INTERPRETATION GUIDE")
logger.info("=" * 100)
logger.info("Z-scores measure standardized deviation from regression predictions:")
logger.info("  z > 0: More POIs than predicted (fully saturated areas)")
logger.info("  z < 0: Fewer POIs than predicted (undersaturated areas)")
logger.info("  |z| < 1: Within 1 std dev (typical variation)")
logger.info("  |z| < 2: Within 2 std devs (~95% of normal distribution)")
logger.info("  z < -2: Extreme undersaturation (beyond normal range)")
logger.info("\nYou control the threshold - use city_results.csv and grid_multiscale.gpkg")
logger.info("to filter grids by your preferred z-score threshold (e.g., z < -1.0, z < -1.5, z < -2.0).")
logger.info("=" * 100)

# Generate markdown report
logger.info("\nGenerating markdown report...")

# Filter cities with sufficient data (at least 10 grids and valid avg_z_mean)
min_grids = 10
city_quality_filtered = city_quality_df[
    (city_quality_df["total_grids"] >= min_grids) & (city_quality_df["avg_z_mean"].notna())
].copy()

report_lines = [
    "# POI Quality Assessment Report",
    "",
    "## Executive Summary",
    "",
    f"- **Total cities analyzed**: {len(city_quality_df)}",
    f"- **Cities with sufficient data** (≥{min_grids} grids): {len(city_quality_filtered)}",
    f"- **Total grid cells**: {len(grid_gdf)}",
    f"- **POI categories**: {len(poi_categories)}",
    "",
    "---",
    "",
    "## City Rankings",
    "",
    "### Most Undersaturated Cities (Lowest Avg Z-Score)",
    "",
    "| City | Country | Grids | Avg Z-Score |",
    "|------|---------|-------|-------------|",
]

# Add top 10 most underserved
for _, row in city_quality_filtered.head(10).iterrows():
    city_label = str(row.get("label", row.get("bounds_fid", "Unknown")))
    country = str(row.get("country", "Unknown"))
    report_lines.append(f"| {city_label} | {country} | {int(row['total_grids'])} | {row.get('avg_z_mean', 0):.4f} |")

report_lines.extend(
    [
        "",
        "### Most Fully Saturated Cities (Highest Avg Z-Score)",
        "",
        "| City | Country | Grids | Avg Z-Score |",
        "|------|---------|-------|-------------|",
    ]
)

# Add top 10 best served
for _, row in city_quality_filtered.tail(10).iterrows():
    city_label = str(row.get("label", row.get("bounds_fid", "Unknown")))
    country = str(row.get("country", "Unknown"))
    report_lines.append(f"| {city_label} | {country} | {int(row['total_grids'])} | {row.get('avg_z_mean', 0):.4f} |")

report_lines.extend(
    [
        "",
        "---",
        "",
        "## Performance by POI Category",
        "",
    ]
)

# Add category statistics (using filtered dataset)
for cat in poi_categories:
    z_col = f"{cat}_z_mean"
    if z_col in city_quality_filtered.columns:
        z_values = city_quality_filtered[z_col].dropna()
        if len(z_values) > 0:
            report_lines.append(f"### {cat}")
            report_lines.append("")
            report_lines.append(f"- **Avg Z-Score**: {z_values.mean():.4f}")
            report_lines.append(f"- **Min Z-Score**: {z_values.min():.4f} (Most undersaturated)")
            report_lines.append(f"- **Max Z-Score**: {z_values.max():.4f} (Most fully saturated)")
            report_lines.append(f"- **Std Dev**: {z_values.std():.4f}")
            report_lines.append(f"- **Cities with data**: {len(z_values)}")
            report_lines.append("")

            # Most/least underserved cities for this category
            most_underserved_idx = z_values.idxmin()
            most_overserved_idx = z_values.idxmax()
            most_underserved_z = z_values.min()
            most_overserved_z = z_values.max()
            underserved_city = str(city_quality_filtered.loc[most_underserved_idx, "label"])
            underserved_country = str(city_quality_filtered.loc[most_underserved_idx, "country"])
            overserved_city = str(city_quality_filtered.loc[most_overserved_idx, "label"])
            overserved_country = str(city_quality_filtered.loc[most_overserved_idx, "country"])

            report_lines.append(
                f"**Most undersaturated**: {underserved_city}, {underserved_country} (z={most_underserved_z:.4f})"
            )
            report_lines.append(
                f"**Most fully saturated**: {overserved_city}, {overserved_country} (z={most_overserved_z:.4f})"
            )
            report_lines.append("")

# Add country-level summary section
report_lines.extend(
    [
        "---",
        "",
        "## Country Summary",
        "",
    ]
)

# Calculate country-level statistics
country_stats = (
    city_quality_filtered.groupby("country")
    .agg(
        {
            "avg_z_mean": ["mean", "min", "max", "count"],
            "total_grids": "sum",
        }
    )
    .reset_index()
)
country_stats.columns = ["country", "avg_z_mean", "min_z_mean", "max_z_mean", "num_cities", "total_grids"]
country_stats = country_stats.sort_values("avg_z_mean")

report_lines.append("| Country | Cities | Total Grids | Avg Z-Score | Min | Max |")
report_lines.append("|---------|--------|------------|-------------|-----|-----|")

for _, row in country_stats.iterrows():
    country = str(row["country"])
    num_cities = int(row["num_cities"])
    total_grids = int(row["total_grids"])
    avg_z = row["avg_z_mean"]
    min_z = row["min_z_mean"]
    max_z = row["max_z_mean"]
    report_lines.append(f"| {country} | {num_cities} | {total_grids} | {avg_z:.4f} | {min_z:.4f} | {max_z:.4f} |")

report_lines.extend(["", ""])

# Add model performance section
report_lines.extend(
    [
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
        report_lines.append(f"| {cat} | {r2:.4f} | {imp_local:.4f} | {imp_int:.4f} | {imp_large:.4f} |")

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
        "- Percentage of undersaturated grids by POI category",
        "- Distribution of local population across grids",
        "- Model fit (R²) by category",
        "- Distribution of mean z-scores across cities",
        "",
        "### Feature Importance Analysis",
        "![Feature Importance](feature_importance.png)",
        "",
        "Shows which population scale (local, intermediate, large) is most predictive for each POI category.",
        "",
        "### Regression Diagnostics",
        "![Regression Diagnostics](regression_diagnostics.png)",
        "",
        "Predicted vs observed POI counts for each category. Shows model fit quality and outliers.",
        "",
        "---",
        "",
        "## Output Files",
        "",
        "### Data Files",
        "- **[grid_multiscale.gpkg](grid_multiscale.gpkg)**:",
        "  Vector grid dataset with z-scores and predictions for all POI categories.",
        "  Contains residuals, z-scores, and model predictions at the grid cell level.",
        "  Can be filtered by z-score thresholds to identify undersaturated/fully saturated areas.",
        "",
        "- **[city_results.csv](city_results.csv)**:",
        "  City-level summary statistics with per-category z-score metrics",
        "  (mean, median, std, min, max) for each city.",
        "",
        "- **[city_quality_ranking.gpkg](city_quality_ranking.gpkg)**:",
        "  City ranking dataset with geographic boundaries, ranked by average z-score.",
        "  Includes city names, countries, grid counts, and all performance metrics.",
        "",
        "### Visualization Files",
        "- **[eda_analysis.png](eda_analysis.png)**:",
        "  Exploratory data analysis showing undersaturation %, population distribution,",
        "  model fit (R²), and z-score distribution.",
        "",
        "- **[feature_importance.png](feature_importance.png)**:",
        "  Random Forest feature importance comparing local, intermediate, and large",
        "  population scales across POI categories.",
        "",
        "- **[regression_diagnostics.png](regression_diagnostics.png)**:",
        "  Predicted vs observed scatter plots for model diagnostics and fit quality assessment.",
        "",
        "---",
        "",
        "## Z-Score Interpretation",
        "",
        "- **z < 0**: Fewer POIs than predicted (undersaturated)",
        "- **z > 0**: More POIs than predicted (fully saturated)",
        "- **z < -1**: Moderately undersaturated (1 std dev below predicted)",
        "- **z < -2**: Severely undersaturated (2 std devs below predicted)",
        "- **z > 2**: Significantly fully saturated (2 std devs above predicted)",
        "",
    ]
)

# Write report to file
report_path = output_path / "city_assessment_report.md"
with open(report_path, "w") as f:
    f.write("\n".join(report_lines))

logger.info(f"Saved report to {report_path}")
logger.info("=" * 100)

# %%
"""
## Step 5: Exploratory Data Analysis (EDA)

Analyze distribution of POI counts and population scales.
"""

logger.info("\nSTEP 5: Exploratory data analysis")

logger.info("\n" + "=" * 80)
logger.info("POI COUNT STATISTICS BY CATEGORY")
logger.info("=" * 80)

for cat in poi_categories:
    poi_counts = grid_gdf[f"{cat}_count"].values.astype(float)
    nonzero_counts = poi_counts[poi_counts > 0]

    logger.info(
        f"\n{cat}:"
        f"\n  Grids with data: {len(nonzero_counts)}/{len(grid_gdf)} ({100 * len(nonzero_counts) / len(grid_gdf):.1f}%)"
        f"\n  Mean count: {nonzero_counts.mean():.1f}"
        f"\n  Median count: {np.median(nonzero_counts):.1f}"
        f"\n  Max count: {nonzero_counts.max():.0f}"
        f"\n  Std dev: {nonzero_counts.std():.1f}"
    )

# Create EDA visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Percentage of underserved grids by category (z < -1)
ax = axes[0, 0]
underserved_pcts = []
valid_categories = []
for cat in poi_categories:
    zscore_col = f"{cat}_zscore"
    if zscore_col in grid_gdf.columns:
        zscores = grid_gdf[zscore_col].dropna()
        if len(zscores) > 0:
            pct_underserved = 100 * (zscores < -1).sum() / len(zscores)
            underserved_pcts.append(pct_underserved)
            valid_categories.append(cat)

if len(underserved_pcts) > 0:
    colors = ["coral" if pct > 10 else "steelblue" for pct in underserved_pcts]
    ax.barh(valid_categories, underserved_pcts, color=colors, alpha=0.7, edgecolor="black")
    ax.axvline(x=5, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="5% threshold")
    ax.set_xlabel("Percentage of Grids (%)", fontsize=11)
    ax.set_title("Underserved Grids by Category (z < -1)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    ax.legend()
else:
    ax.text(0.5, 0.5, "No z-score data available", ha="center", va="center", transform=ax.transAxes)

# 2. Population distribution
ax = axes[0, 1]
pop_local = grid_gdf["pop_local"].values.astype(float)
pop_local_nonzero = pop_local[pop_local > 0]
if len(pop_local_nonzero) > 0:
    ax.hist(np.log10(pop_local_nonzero + 1), bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    ax.set_xlabel("log10(Local Population)", fontsize=11)
    ax.set_ylabel("Number of Grids", fontsize=11)
    ax.set_title("Distribution of Local Population", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

# 3. R² values by category
ax = axes[1, 0]
r2_values = [regression_results[cat]["r2"] for cat in poi_categories if not np.isnan(regression_results[cat]["r2"])]
ax.bar(range(len(r2_values)), r2_values, color="green", alpha=0.7, edgecolor="black")
ax.set_xticks(range(len(r2_values)))
ax.set_xticklabels(
    [cat for cat in poi_categories if not np.isnan(regression_results[cat]["r2"])], rotation=45, ha="right"
)
ax.set_ylabel("R² Value", fontsize=11)
ax.set_title("Model Fit by Category", fontsize=12, fontweight="bold")
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3, axis="y")

# 4. City-level mean z-score distribution
ax = axes[1, 1]
z_means_all = []
for cat in poi_categories:
    col_name = f"{cat}_z_mean"
    if col_name in city_df.columns:
        z_means_all.extend(city_df[col_name].dropna().values)

if len(z_means_all) > 0:
    ax.hist(z_means_all, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(
        np.mean(z_means_all), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(z_means_all):.3f}"
    )
    ax.set_xlabel("Mean Z-Score per Category per City", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Distribution of Mean Z-Scores", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(output_path / "eda_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

logger.info(f"\nSaved EDA plots to {output_path / 'eda_analysis.png'}")

# %%
"""
## Step 6: Regression Diagnostic Plots

Visualize regression relationships for each POI category.
"""

logger.info("STEP 6: Creating regression diagnostic plots")

# For each category, create a scatter plot showing residuals
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
    flagged = grid_gdf[f"{cat}_flagged"].values.astype(bool)

    # Filter for valid data
    valid_mask = (poi > 0) & (pop_local > 0) & (pop_intermediate > 0) & (pop_large > 0)

    if valid_mask.sum() < 10:
        ax.text(0.5, 0.5, f"{cat}\n(insufficient data)", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        continue

    # Prepare data (raw counts to match training)
    poi_counts = poi[valid_mask]
    pop_local_counts = pop_local[valid_mask]
    pop_intermediate_counts = pop_intermediate[valid_mask]
    pop_large_counts = pop_large[valid_mask]

    # Fit Random Forest (same as main regression)
    X = np.column_stack([pop_local_counts, pop_intermediate_counts, pop_large_counts])
    model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
    model.fit(X, poi_counts)
    predictions = model.predict(X)
    residuals = poi_counts - predictions

    is_flagged = flagged[valid_mask]

    # Plot: predicted vs observed
    ax.scatter(predictions[~is_flagged], poi_counts[~is_flagged], alpha=0.3, s=15, c="steelblue", label="Normal")
    if is_flagged.sum() > 0:
        ax.scatter(predictions[is_flagged], poi_counts[is_flagged], alpha=0.6, s=30, c="red", label="Flagged")

    # Perfect prediction line
    pred_range = np.linspace(predictions.min(), predictions.max(), 100)
    ax.plot(pred_range, pred_range, "k--", alpha=0.5, linewidth=1, label="Perfect")

    # Get R²
    r2 = model.score(X, poi_counts)

    ax.set_xlabel("Predicted POI Count", fontsize=10)
    ax.set_ylabel("Observed POI Count", fontsize=10)
    ax.set_title(f"{cat}\nR²={r2:.3f}, Flagged={is_flagged.sum()}", fontsize=11)
    ax.grid(True, alpha=0.3)
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
## Step 7: Feature Importance Analysis

Compare how different population scales contribute to POI prediction across categories.
"""

logger.info("\n" + "=" * 80)
logger.info("STEP 7: Feature Importance Analysis")
logger.info("=" * 80)

# Create feature importance comparison plot
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
ax.set_xticklabels(poi_categories, rotation=45, ha="right")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(output_path / "feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()

logger.info(f"Saved feature importance analysis to {output_path / 'feature_importance.png'}")

logger.info("\n" + "=" * 80)
logger.info("ANALYSIS COMPLETE")
logger.info("=" * 80)
logger.info(f"\nOutput directory: {output_path}")
logger.info("  - grid_multiscale.gpkg: Grid-level results with z-scores and residuals")
logger.info("  - city_results.csv: City-level z-score statistics")
logger.info("  - city_quality_ranking.gpkg: City quality ranking by avg z-score")
logger.info("  - city_assessment_report.md: Markdown report with city rankings and statistics")
logger.info("  - regression_diagnostics.png: Predicted vs observed plots")
logger.info("  - feature_importance.png: Scale importance comparison (Random Forest)")
logger.info("  - eda_analysis.png: Exploratory data analysis visualizations")

# %%
