# %% [markdown];
"""
# Green Block Accessibility and Density Correlation Workflow

Analyzes the relationship between population density and access to green blocks
and tree canopy across European cities. Computes per-city correlations to reveal
heterogeneity in urban planning approaches.

## Steps
1. Load city boundaries
2. Aggregate green/tree access and density metrics from city files
3. Compute per-city correlations between density and green access
4. Generate scatter plots of city correlation AND distance vs. mean density (2x2 grid)
5. Generate diverging bar chart visualization showing correlation distribution
6. Generate summary report

## Key Outputs
- **correlation_vs_density.png**: 2x2 scatter plots (correlation and distance vs. density)
- **city_density_correlations.png**: Four-panel diverging bar chart
- **city_density_correlations.csv**: Per-city correlation data
- **README.md**: Summary report with key findings

## Metrics Used (from SOAR pre-computed)
### Green/Tree Access
- `cc_green_nearest_max_1600`: Network distance to nearest green block (m)
- `cc_trees_nearest_max_1600`: Network distance to nearest tree canopy (m)

### Density
- `density`: Population density (persons/km²) from Eurostat 1km grid

## Analysis Interpretation
- **Negative correlation (ρ)**: Denser areas have shorter distances to green block (better access)
- **Positive correlation (ρ)**: Denser areas have longer distances to green block (worse access)
- Uses Spearman's rank correlation (robust to skewed distributions and outliers common in urban data)
- City-level heterogeneity demonstrates why aggregate statistics can be misleading
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Green/tree access columns from SOAR metrics
GREEN_ACCESS_COLS = [
    "cc_green_nearest_max_1600",  # Distance to nearest green block
    "cc_trees_nearest_max_1600",  # Distance to nearest tree canopy
]

# Density column from SOAR metrics (interpolated from Eurostat grid)
DENSITY_COLS = [
    "density",  # Population density (persons/km²)
]

# All columns to load from metrics files
METRICS_COLS = GREEN_ACCESS_COLS + DENSITY_COLS


# %%
"""
## Helper Functions for Green Block Analysis
"""


def load_city_metrics(
    metrics_dir: Path,
    bounds_fid: int,
    columns: list[str] | None = None,
) -> gpd.GeoDataFrame | None:
    """Load metrics file for a specific city.

    Parameters
    ----------
    metrics_dir
        Path to directory containing metrics_*.gpkg files
    bounds_fid
        City boundary ID (matches file naming: metrics_{bounds_fid}.gpkg)
    columns
        Optional list of columns to load (plus geometry). If None, loads all.

    Returns
    -------
    GeoDataFrame with city metrics, or None if file doesn't exist
    """
    metrics_file = metrics_dir / f"metrics_{bounds_fid}.gpkg"
    if not metrics_file.exists():
        return None

    try:
        if columns:
            # Always include geometry
            gdf = gpd.read_file(metrics_file, columns=columns, layer="streets")
        else:
            gdf = gpd.read_file(metrics_file, layer="streets")
        return gdf
    except Exception as e:
        print(f"WARNING: Error loading metrics for city {bounds_fid}: {e}")
        return None


def aggregate_city_green_metrics(
    metrics_dir: Path,
    bounds_gdf: gpd.GeoDataFrame,
    metrics_cols: list[str],
) -> gpd.GeoDataFrame:
    """Aggregate green access and demographic metrics across all cities.

    Parameters
    ----------
    metrics_dir
        Path to directory containing metrics_*.gpkg files
    bounds_gdf
        GeoDataFrame with city boundaries (must have 'bounds_fid' column)
    metrics_cols
        List of column names to extract (green access + demographics)

    Returns
    -------
    GeoDataFrame with all nodes from all cities, including green access
    and demographic metrics
    """
    print(f"Aggregating metrics from {len(bounds_gdf)} cities...")

    all_nodes = []
    for idx, row in bounds_gdf.iterrows():
        bounds_fid = row.get("bounds_fid", row.get("fid", idx))

        gdf = load_city_metrics(metrics_dir, bounds_fid, columns=metrics_cols)
        if gdf is None:
            continue

        # Add city identifier
        gdf["bounds_fid"] = bounds_fid
        if "label" in row.index:
            gdf["city_label"] = row["label"]
        if "country" in row.index:
            gdf["country"] = row["country"]

        all_nodes.append(gdf)

        if len(all_nodes) % 50 == 0:
            print(f"  Processed {len(all_nodes)} cities...")

    if not all_nodes:
        raise ValueError("No city metrics files found")

    print(f"  Concatenating {len(all_nodes)} city datasets...")
    combined_gdf = pd.concat(all_nodes, ignore_index=True)
    combined_gdf = gpd.GeoDataFrame(combined_gdf, geometry="geometry")

    print(f"  Total nodes: {len(combined_gdf)}")
    return combined_gdf


# %%
"""
## Configuration
"""

# Configuration - modify these paths as needed
BOUNDS_PATH = "temp/datasets/boundaries.gpkg"
METRICS_DIR = "temp/cities_data/processed"
OUTPUT_DIR = "paper_research/code/eg2_green_space/outputs"
TEMP_DIR = "temp/egs/eg2_green_space"

# %%
"""
## Step 1: Load Data
"""

output_path = Path(OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)
metrics_dir = Path(METRICS_DIR)
temp_path = Path(TEMP_DIR)
temp_path.mkdir(parents=True, exist_ok=True)

print("STEP 1: Loading city boundaries")

# Load city boundaries
bounds_gdf = gpd.read_file(BOUNDS_PATH)
print(f"  Loaded {len(bounds_gdf)} city boundaries")

# %%
"""
## Step 2: Aggregate Green/Tree Access and Density Metrics from City Files
"""

print("STEP 2: Aggregating green/tree access and density metrics")

green_nodes_file = temp_path / "green_nodes.parquet"

if green_nodes_file.exists():
    print("  Loading cached metrics...")
    green_nodes_gdf = pd.read_parquet(green_nodes_file)
else:
    green_nodes_gdf = aggregate_city_green_metrics(
        metrics_dir=metrics_dir,
        bounds_gdf=bounds_gdf,
        metrics_cols=METRICS_COLS,
    )
    # Save as parquet without geometry (much smaller, geometry not needed for aggregation)
    green_nodes_gdf.drop(columns=["geometry"]).to_parquet(green_nodes_file)
    print(f"  Saved aggregated metrics to {green_nodes_file}")

# Diagnostic: Check raw density values in the node data
print("\n  Raw density statistics (all nodes):")
print(f"    Min: {green_nodes_gdf['density'].min():.1f}")
print(f"    Max: {green_nodes_gdf['density'].max():.1f}")
print(f"    Mean: {green_nodes_gdf['density'].mean():.1f}")
print(f"    Median: {green_nodes_gdf['density'].median():.1f}")
print(f"    95th percentile: {green_nodes_gdf['density'].quantile(0.95):.1f}")
print(f"    Non-null count: {green_nodes_gdf['density'].notna().sum()}/{len(green_nodes_gdf)}")

# Clean data: remove infinite and NaN values
print("\n  Cleaning data: removing infinite/NaN values...")
n_before = len(green_nodes_gdf)
green_nodes_gdf = green_nodes_gdf[
    (green_nodes_gdf["density"].notna())
    & (np.isfinite(green_nodes_gdf["density"]))
    & (green_nodes_gdf["cc_green_nearest_max_1600"].notna())
    & (np.isfinite(green_nodes_gdf["cc_green_nearest_max_1600"]))
    & (green_nodes_gdf["cc_trees_nearest_max_1600"].notna())
    & (np.isfinite(green_nodes_gdf["cc_trees_nearest_max_1600"]))
]
n_after = len(green_nodes_gdf)
print(f"  Removed {n_before - n_after} rows with NaN/Inf values ({100 * (n_before - n_after) / n_before:.1f}%)")


# %%
"""
## Step 3: Generate Per-City Correlation Analysis
"""

print("STEP 3: Computing per-city density-green correlations")


# Calculate correlation for each city
MIN_NODES_PER_CITY = 100  # Minimum nodes for reliable correlation

city_correlations = []
for bounds_fid in green_nodes_gdf["bounds_fid"].unique():
    city_data = green_nodes_gdf[green_nodes_gdf["bounds_fid"] == bounds_fid]

    if len(city_data) < MIN_NODES_PER_CITY:
        continue

    # Get city label and country
    city_label = city_data["city_label"].iloc[0] if "city_label" in city_data.columns else str(bounds_fid)
    country = city_data["country"].iloc[0] if "country" in city_data.columns else "Unknown"

    # Calculate correlations (density vs distance)
    valid_green = city_data[["density", "cc_green_nearest_max_1600"]].dropna()
    valid_trees = city_data[["density", "cc_trees_nearest_max_1600"]].dropna()

    if len(valid_green) >= MIN_NODES_PER_CITY:
        green_corr = valid_green.corr(method="spearman").iloc[0, 1]
    else:
        green_corr = np.nan

    if len(valid_trees) >= MIN_NODES_PER_CITY:
        trees_corr = valid_trees.corr(method="spearman").iloc[0, 1]
    else:
        trees_corr = np.nan

    # Calculate mean city density and green distances
    mean_density = city_data["density"].mean()
    mean_green_dist = city_data["cc_green_nearest_max_1600"].mean()
    mean_trees_dist = city_data["cc_trees_nearest_max_1600"].mean()

    city_correlations.append(
        {
            "bounds_fid": bounds_fid,
            "city_label": city_label,
            "country": country,
            "green_corr": green_corr,
            "trees_corr": trees_corr,
            "n_nodes": len(city_data),
            "mean_density": mean_density,
            "mean_green_dist": mean_green_dist,
            "mean_trees_dist": mean_trees_dist,
        }
    )

city_corr_df = pd.DataFrame(city_correlations)
print(f"  Computed correlations for {len(city_corr_df)} cities")

# Summary statistics
green_positive = (city_corr_df["green_corr"] > 0).sum()
green_negative = (city_corr_df["green_corr"] < 0).sum()
trees_positive = (city_corr_df["trees_corr"] > 0).sum()
trees_negative = (city_corr_df["trees_corr"] < 0).sum()

print(f"\n  Green block correlations: {green_positive} positive, {green_negative} negative")
print(f"  Tree canopy correlations: {trees_positive} positive, {trees_negative} negative")

# %%
"""
## Step 4: Correlation vs. Density Scatter Plots
"""

print("STEP 4: Generating density vs. correlation and distance scatter plots")


# Helper function for simple regression
def add_regression_line(ax, x, y, color="red"):
    """Add a simple regression line to scatter plot"""
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), color=color, linewidth=2, linestyle="--", alpha=0.7, label="OLS fit")


# Apply subtle seaborn styling
sns.set_style("whitegrid", {"grid.color": ".9", "axes.edgecolor": ".6"})
sns.set_context("paper")

# Create four-panel figure: green block and tree canopy (2 rows: distance, correlation)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Filter out cities with all NaN values (keep cities with at least some valid data)
check_cols = ["mean_density", "green_corr", "trees_corr", "mean_green_dist", "mean_trees_dist"]
plot_df = city_corr_df[~city_corr_df[check_cols].isna().all(axis=1)]
# Also drop rows where any required column is NaN for complete cases
plot_df = plot_df.dropna(subset=check_cols)

# Diagnostic: Check density distribution
print("\n  Density statistics:")
print(f"    Min: {plot_df['mean_density'].min():.1f}")
print(f"    Max: {plot_df['mean_density'].max():.1f}")
print(f"    Mean: {plot_df['mean_density'].mean():.1f}")
print(f"    Median: {plot_df['mean_density'].median():.1f}")
print(f"    95th percentile: {plot_df['mean_density'].quantile(0.95):.1f}")

max_density = plot_df["mean_density"].quantile(0.99)
xlim_max = (
    max_density * 1.1 if pd.notna(max_density) and np.isfinite(max_density) else plot_df["mean_density"].max() * 1.1
)

# Row 1: Distance vs. Density
# Panel 1a: Green block distance vs. density
ax = axes[0, 0]
colors_green = sns.color_palette("icefire", as_cmap=True)((plot_df["green_corr"] + 1) / 2)
colors_green_dist = colors_green
ax.scatter(
    plot_df["mean_density"],
    plot_df["mean_green_dist"],
    c=colors_green_dist,
    s=30,
    alpha=0.6,
    edgecolors="white",
    linewidths=0.5,
)
add_regression_line(ax, plot_df["mean_density"].values, plot_df["mean_green_dist"].values)
ax.set_xlabel("Mean City Density (persons per km²)", fontsize=10, color="dimgrey")
ax.set_ylabel("Mean Distance to Green Block (m)", fontsize=10, color="dimgrey")
ax.set_title("Green Block: Distance vs Density", fontsize=11, fontweight="bold")
ax.set_xlim(0, xlim_max)
ax.tick_params(colors="dimgrey", labelsize=9)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

# Panel 1b: Tree canopy distance vs. density
ax = axes[0, 1]
colors_trees = sns.color_palette("icefire", as_cmap=True)((plot_df["trees_corr"] + 1) / 2)
colors_trees_dist = colors_trees
ax.scatter(
    plot_df["mean_density"],
    plot_df["mean_trees_dist"],
    c=colors_trees_dist,
    s=30,
    alpha=0.6,
    edgecolors="white",
    linewidths=0.5,
)
add_regression_line(ax, plot_df["mean_density"].values, plot_df["mean_trees_dist"].values)
ax.set_xlabel("Mean City Density (persons per km²)", fontsize=10, color="dimgrey")
ax.set_ylabel("Mean Distance to Tree Canopy (m)", fontsize=10, color="dimgrey")
ax.set_title("Tree Canopy: Distance vs Density", fontsize=11, fontweight="bold")
ax.set_xlim(0, xlim_max)
ax.tick_params(colors="dimgrey", labelsize=9)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

# Row 2: Correlation vs. Density
# Panel 2a: Green block correlation vs. density
ax = axes[1, 0]
ax.scatter(
    plot_df["mean_density"],
    plot_df["green_corr"],
    c=colors_green,
    s=30,
    alpha=0.6,
    edgecolors="white",
    linewidths=0.5,
)
ax.axhline(0, color="dimgrey", linewidth=0.8, linestyle="--", alpha=0.5)
add_regression_line(ax, plot_df["mean_density"].values, plot_df["green_corr"].values)
ax.set_xlabel("Mean City Density (persons per km²)", fontsize=10, color="dimgrey")
ax.set_ylabel("Spearman ρ (density vs green distance)", fontsize=10, color="dimgrey")
ax.set_title("Green Block: Correlation vs Density", fontsize=11, fontweight="bold")
ax.set_xlim(0, xlim_max)
ax.tick_params(colors="dimgrey", labelsize=9)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

# Panel 2b: Tree canopy correlation vs. density
ax = axes[1, 1]
ax.scatter(
    plot_df["mean_density"],
    plot_df["trees_corr"],
    c=colors_trees,
    s=30,
    alpha=0.6,
    edgecolors="white",
    linewidths=0.5,
)
ax.axhline(0, color="dimgrey", linewidth=0.8, linestyle="--", alpha=0.5)
add_regression_line(ax, plot_df["mean_density"].values, plot_df["trees_corr"].values)
ax.set_xlabel("Mean City Density (persons per km²)", fontsize=10, color="dimgrey")
ax.set_ylabel("Spearman ρ (density vs tree distance)", fontsize=10, color="dimgrey")
ax.set_title("Tree Canopy: Correlation vs Density", fontsize=11, fontweight="bold")
ax.set_xlim(0, xlim_max)
ax.tick_params(colors="dimgrey", labelsize=9)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

# Clean up all axes
for ax in axes.flat:
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("lightgrey")

plt.suptitle("Density vs. Green Block Access (Correlation and Distance)", fontsize=12, fontweight="bold", y=0.995)
plt.tight_layout()
scatter_path = output_path / "correlation_vs_density.png"
plt.savefig(scatter_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.show()
print(f"  Saved scatter plot to {scatter_path}")

# Identify outlier cities (dense but equitable)
high_density_threshold = plot_df["mean_density"].quantile(0.75)
dense_cities = plot_df[plot_df["mean_density"] >= high_density_threshold]

# Dense cities with negative green correlation (equitable)
dense_equitable_green = dense_cities[dense_cities["green_corr"] < -0.2].nsmallest(5, "green_corr")
# Dense cities with positive green correlation (inequitable)
dense_inequitable_green = dense_cities[dense_cities["green_corr"] > 0.2].nlargest(5, "green_corr")

print(f"\n  High-density cities (>{high_density_threshold:.0f} persons/km²):")
print(f"    Total: {len(dense_cities)}")
if len(dense_equitable_green) > 0:
    print("\n    Dense cities with EQUITABLE green access (ρ < -0.2):")
    for _, row in dense_equitable_green.iterrows():
        print(
            f"      • {row['city_label']} ({row['country']}): ρ = {row['green_corr']:.3f}, density = {row['mean_density']:.0f}"
        )
if len(dense_inequitable_green) > 0:
    print("\n    Dense cities with INEQUITABLE green access (ρ > 0.2):")
    for _, row in dense_inequitable_green.iterrows():
        print(
            f"      • {row['city_label']} ({row['country']}): ρ = {row['green_corr']:.3f}, density = {row['mean_density']:.0f}"
        )

# %%
"""
## Step 5: Diverging Bar Chart Visualization
"""

print("STEP 5: Generating diverging bar chart")

# Apply subtle seaborn styling
sns.set_style("whitegrid", {"grid.color": ".9", "axes.edgecolor": ".6"})
sns.set_context("paper")

# Create 4-panel figure: all green, labeled green, all trees, labeled trees
fig, axes = plt.subplots(1, 4, figsize=(14, 12), width_ratios=[0.75, 1.5, 0.75, 1.5])

panel_configs = [
    (axes[0], axes[1], "green_corr", "Green Block", green_negative, green_positive),
    (axes[2], axes[3], "trees_corr", "Tree Canopy", trees_negative, trees_positive),
]

for ax_full, ax_labeled, corr_col, title, n_neg, n_pos in panel_configs:
    # Get data sorted by correlation
    plot_df_full = city_corr_df[["city_label", "country", corr_col, "n_nodes"]].dropna()
    plot_df_full = plot_df_full.sort_values(corr_col)

    # --- Full distribution panel (thin bars, no labels) ---
    colors_full = sns.color_palette("icefire", as_cmap=True)((plot_df_full[corr_col] + 1) / 2)
    y_pos_full = range(len(plot_df_full))

    ax_full.barh(y_pos_full, plot_df_full[corr_col], color=colors_full, edgecolor="none", height=1.0)
    ax_full.axvline(0, color="dimgrey", linewidth=0.8)
    ax_full.set_yticks([])
    ax_full.set_xlabel("ρ", fontsize=9, color="dimgrey")
    ax_full.set_title(f"{title}\n(all {len(plot_df_full)} cities)", fontsize=11, fontweight="bold")
    ax_full.set_xlim(-0.8, 0.8)
    ax_full.tick_params(axis="x", colors="dimgrey", labelsize=8)
    ax_full.xaxis.set_major_locator(plt.MultipleLocator(0.4))

    # Add count labels for positive (top right, red) and negative (bottom left, blue)
    ax_full.text(
        0.90,
        0.98,
        f"{n_pos}",
        transform=ax_full.transAxes,
        fontsize=14,
        fontweight="bold",
        color="firebrick",
        ha="right",
        va="top",
    )
    ax_full.text(
        0.10,
        0.02,
        f"{n_neg}",
        transform=ax_full.transAxes,
        fontsize=14,
        fontweight="bold",
        color="steelblue",
        ha="left",
        va="bottom",
    )

    # --- Labeled extremes panel ---
    n_show = min(40, len(plot_df_full))
    n_each = n_show // 2
    plot_df_bottom = plot_df_full.head(n_each).copy()  # Most negative
    plot_df_top = plot_df_full.tail(n_each).copy()  # Most positive

    # Create y positions with a gap in the middle
    gap_size = 2  # Number of bar heights for the gap
    y_pos_bottom = list(range(n_each))
    y_pos_top = list(range(n_each + gap_size, 2 * n_each + gap_size))

    # Plot bottom 20 (negative correlations)
    colors_bottom = sns.color_palette("icefire", as_cmap=True)((plot_df_bottom[corr_col] + 1) / 2)
    ax_labeled.barh(y_pos_bottom, plot_df_bottom[corr_col], color=colors_bottom, edgecolor="none", height=0.8)

    # Plot top 20 (positive correlations)
    colors_top = sns.color_palette("icefire", as_cmap=True)((plot_df_top[corr_col] + 1) / 2)
    ax_labeled.barh(y_pos_top, plot_df_top[corr_col], color=colors_top, edgecolor="none", height=0.8)

    # Set y-ticks and labels
    all_y_pos = y_pos_bottom + y_pos_top
    all_labels = list(plot_df_bottom["city_label"]) + list(plot_df_top["city_label"])
    ax_labeled.set_yticks(all_y_pos)
    ax_labeled.set_yticklabels(all_labels, fontsize=8, color="dimgrey")
    ax_labeled.axvline(0, color="dimgrey", linewidth=0.8)
    ax_labeled.set_xlabel("ρ", fontsize=9, color="dimgrey")
    ax_labeled.set_title(f"{title}\n(top/bottom 20)", fontsize=11, fontweight="bold")
    ax_labeled.set_xlim(-0.8, 0.8)
    ax_labeled.tick_params(axis="x", colors="dimgrey", labelsize=8)
    ax_labeled.tick_params(axis="y", length=0)
    ax_labeled.xaxis.set_major_locator(plt.MultipleLocator(0.4))

    # Add interpretation labels (vertical orientation along Y-axis)
    max_y = max(y_pos_top)
    min_y = min(y_pos_bottom)

    # Top label (positive correlations - red)
    ax_labeled.text(
        0.85,
        max_y,
        "Denser = Farther ↑",
        fontsize=8,
        color="firebrick",
        fontweight="bold",
        ha="left",
        va="center",
        rotation=0,
    )
    # Bottom label (negative correlations - blue)
    ax_labeled.text(
        0.85,
        min_y,
        "↓ Denser = Closer",
        fontsize=8,
        color="steelblue",
        fontweight="bold",
        ha="left",
        va="center",
        rotation=0,
    )

# Remove spines for cleaner look
for ax in axes:
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("lightgrey")

plt.tight_layout(w_pad=0)
viz_path = output_path / "city_density_correlations.png"
plt.savefig(viz_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.show()
print(f"  Saved visualization to {viz_path}")

# Overall correlation (for comparison)
green_density_corr = (
    green_nodes_gdf[["density", "cc_green_nearest_max_1600"]].dropna().corr(method="spearman").iloc[0, 1]
)
trees_density_corr = (
    green_nodes_gdf[["density", "cc_trees_nearest_max_1600"]].dropna().corr(method="spearman").iloc[0, 1]
)
print("\n  Overall correlation (all nodes pooled):")
print(f"    • Green Block: r = {green_density_corr:.3f}")
print(f"    • Tree canopy: r = {trees_density_corr:.3f}")

# %%
"""
## Step 6: Compute Summary Statistics and Generate Report
"""

print("STEP 6: Computing summary statistics and generating report")

# Calculate overall statistics
summary_stats = {
    "total_nodes": len(green_nodes_gdf),
    "cities_analyzed": green_nodes_gdf["bounds_fid"].nunique(),
    "cities_with_correlations": len(city_corr_df),
    "green_mean_dist": green_nodes_gdf["cc_green_nearest_max_1600"].mean(),
    "green_median_dist": green_nodes_gdf["cc_green_nearest_max_1600"].median(),
    "green_pct_within_400m": ((green_nodes_gdf["cc_green_nearest_max_1600"] <= 400).sum() / len(green_nodes_gdf) * 100),
    "trees_mean_dist": green_nodes_gdf["cc_trees_nearest_max_1600"].mean(),
    "trees_median_dist": green_nodes_gdf["cc_trees_nearest_max_1600"].median(),
    "trees_pct_within_400m": ((green_nodes_gdf["cc_trees_nearest_max_1600"] <= 400).sum() / len(green_nodes_gdf) * 100),
}

# Cities with strongest correlations
strongest_green_neg = city_corr_df.nsmallest(5, "green_corr")
strongest_green_pos = city_corr_df.nlargest(5, "green_corr")
strongest_trees_neg = city_corr_df.nsmallest(5, "trees_corr")
strongest_trees_pos = city_corr_df.nlargest(5, "trees_corr")

report_lines = [
    "# Green Block Accessibility Analysis Report",
    "",
    f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}",
    "",
    "## Summary Statistics",
    "",
    f"- **Total Street Network Nodes Analyzed:** {summary_stats['total_nodes']:,}",
    f"- **Cities Analyzed:** {summary_stats['cities_analyzed']}",
    f"- **Cities with Correlation Data:** {summary_stats['cities_with_correlations']} (min {MIN_NODES_PER_CITY} nodes)",
    "",
    "### Green Blocks (Parks & Green Spaces)",
    f"- **Mean Distance:** {summary_stats['green_mean_dist']:.1f}m",
    f"- **Median Distance:** {summary_stats['green_median_dist']:.1f}m",
    f"- **% Within 400m (5-min walk):** {summary_stats['green_pct_within_400m']:.1f}%",
    "",
    "### Tree Canopy",
    f"- **Mean Distance:** {summary_stats['trees_mean_dist']:.1f}m",
    f"- **Median Distance:** {summary_stats['trees_median_dist']:.1f}m",
    f"- **% Within 400m (5-min walk):** {summary_stats['trees_pct_within_400m']:.1f}%",
    "",
    "## Per-City Density-Green Access Correlations",
    "",
    "This analysis examines whether denser urban areas have better or worse access to green block.",
    "- **Negative correlation**: Denser areas are *closer* to green block (better access)",
    "- **Positive correlation**: Denser areas are *farther* from green block (worse access)",
    "",
    f"### Green Block: {green_negative} cities negative, {green_positive} cities positive",
    "",
    "**Strongest negative (denser = closer):**",
    "",
]
for _, row in strongest_green_neg.iterrows():
    report_lines.append(f"- {row['city_label']} ({row['country']}): r = {row['green_corr']:.3f}")

report_lines.extend(["", "**Strongest positive (denser = farther):**", ""])
for _, row in strongest_green_pos.iterrows():
    report_lines.append(f"- {row['city_label']} ({row['country']}): r = {row['green_corr']:.3f}")

report_lines.extend(
    [
        "",
        f"### Tree Canopy: {trees_negative} cities negative, {trees_positive} cities positive",
        "",
        "**Strongest negative (denser = closer):**",
        "",
    ]
)
for _, row in strongest_trees_neg.iterrows():
    report_lines.append(f"- {row['city_label']} ({row['country']}): r = {row['trees_corr']:.3f}")

report_lines.extend(["", "**Strongest positive (denser = farther):**", ""])
for _, row in strongest_trees_pos.iterrows():
    report_lines.append(f"- {row['city_label']} ({row['country']}): r = {row['trees_corr']:.3f}")

report_lines.extend(
    [
        "",
        "## Key Finding",
        "",
        "The relationship between population density and green block access varies substantially",
        "by city, reflecting different urban planning approaches and historical development patterns.",
        "This heterogeneity demonstrates why city-specific analysis is essential for understanding",
        "environmental equity rather than relying on aggregate statistics.",
        "",
        "## Visualization",
        "",
        "![City Density Correlations](outputs/city_density_correlations.png)",
        "",
    ]
)

# Write report to file in the example folder root (parent of outputs)
report_path = output_path.parent / "README.md"
with open(report_path, "w") as f:
    f.write("\n".join(report_lines))

print(f"  Saved report to {report_path}")

# Save city correlations as CSV for further analysis
city_corr_df.to_csv(output_path / "city_density_correlations.csv", index=False)
print(f"  Saved city correlations to {output_path / 'city_density_correlations.csv'}")

print("\nANALYSIS COMPLETE")
print(f"Output directory: {output_path}")
print("  - correlation_vs_density.png: Scatter plot of city correlation vs mean density")
print("  - simpsons_paradox_multiscale.png: Multi-scale analysis (continental/within-city/between-city)")
print("  - city_density_correlations.png: Diverging bar chart of per-city correlations")
print("  - city_density_correlations.csv: Raw correlation data")
print(f"README.md saved to: {report_path}")

# %%
