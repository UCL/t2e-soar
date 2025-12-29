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
    "cc_green_area_sum_800_nw",  # Green area within 800m weighted by distance
    "cc_trees_area_sum_800_nw",  # Tree canopy area within 800m weighted by distance
]

# Density column from SOAR metrics (interpolated from Eurostat grid)
DENSITY_COLS = [
    "density",  # Population density (persons/km²)
]

# All columns to load from metrics files
METRICS_COLS = GREEN_ACCESS_COLS + DENSITY_COLS

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
    print(f"Aggregating metrics from {len(bounds_gdf)} cities...")
    all_nodes = []
    for idx, row in bounds_gdf.iterrows():
        bounds_fid = row.get("bounds_fid", row.get("fid", idx))
        metrics_file = metrics_dir / f"metrics_{bounds_fid}.gpkg"
        if not metrics_file.exists():
            continue
        gdf = gpd.read_file(metrics_file, columns=METRICS_COLS, layer="streets")
        if gdf is None:
            continue
        # Doublecheck geoms are dropped if outside boundary
        gdf = gdf[gdf.geometry.within(row.geometry)]
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
    green_nodes_gdf = combined_gdf
    # Save as parquet without geometry (much smaller, geometry not needed for aggregation)
    green_nodes_gdf.drop(columns=["geometry"]).to_parquet(green_nodes_file)
    print(f"  Saved aggregated metrics to {green_nodes_file}")

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

# Diagnostic: Check raw density values in the node data
print("\n  Raw density statistics (all nodes):")
print(f"    Min: {green_nodes_gdf['density'].min():.1f}")
print(f"    Max: {green_nodes_gdf['density'].max():.1f}")
print(f"    Mean: {green_nodes_gdf['density'].mean():.1f}")
print(f"    Median: {green_nodes_gdf['density'].median():.1f}")
print(f"    95th percentile: {green_nodes_gdf['density'].quantile(0.95):.1f}")
print(f"    Non-null count: {green_nodes_gdf['density'].notna().sum()}/{len(green_nodes_gdf)}")


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

    # Calculate correlations (density vs distance and density vs area)
    valid_green = city_data[["density", "cc_green_nearest_max_1600"]].dropna()
    valid_trees = city_data[["density", "cc_trees_nearest_max_1600"]].dropna()
    valid_green_area = city_data[["density", "cc_green_area_sum_800_nw"]].dropna()
    valid_trees_area = city_data[["density", "cc_trees_area_sum_800_nw"]].dropna()

    green_corr = valid_green.corr(method="spearman").iloc[0, 1] if len(valid_green) >= MIN_NODES_PER_CITY else np.nan
    trees_corr = valid_trees.corr(method="spearman").iloc[0, 1] if len(valid_trees) >= MIN_NODES_PER_CITY else np.nan
    green_area_corr = (
        valid_green_area.corr(method="spearman").iloc[0, 1] if len(valid_green_area) >= MIN_NODES_PER_CITY else np.nan
    )
    trees_area_corr = (
        valid_trees_area.corr(method="spearman").iloc[0, 1] if len(valid_trees_area) >= MIN_NODES_PER_CITY else np.nan
    )

    # Calculate mean city density, green distances, and area-weighted metrics
    mean_density = city_data["density"].mean()
    mean_green_dist = city_data["cc_green_nearest_max_1600"].mean()
    mean_trees_dist = city_data["cc_trees_nearest_max_1600"].mean()
    mean_green_area = (
        city_data["cc_green_area_sum_800_nw"].mean() if "cc_green_area_sum_800_nw" in city_data.columns else np.nan
    )
    mean_trees_area = (
        city_data["cc_trees_area_sum_800_nw"].mean() if "cc_trees_area_sum_800_nw" in city_data.columns else np.nan
    )

    city_correlations.append(
        {
            "bounds_fid": bounds_fid,
            "city_label": city_label,
            "country": country,
            "green_corr": green_corr,
            "trees_corr": trees_corr,
            "green_area_corr": green_area_corr,
            "trees_area_corr": trees_area_corr,
            "n_nodes": len(city_data),
            "mean_density": mean_density,
            "mean_green_dist": mean_green_dist,
            "mean_trees_dist": mean_trees_dist,
            "cc_green_area_sum_800_nw": mean_green_area,
            "cc_trees_area_sum_800_nw": mean_trees_area,
        }
    )

city_corr_df = pd.DataFrame(city_correlations)
print(f"  Computed correlations for {len(city_corr_df)} cities")

# Summary statistics
green_positive = (city_corr_df["green_corr"] > 0).sum()
green_negative = (city_corr_df["green_corr"] < 0).sum()
trees_positive = (city_corr_df["trees_corr"] > 0).sum()
trees_negative = (city_corr_df["trees_corr"] < 0).sum()
green_area_positive = (city_corr_df["green_area_corr"] > 0).sum()
green_area_negative = (city_corr_df["green_area_corr"] < 0).sum()
trees_area_positive = (city_corr_df["trees_area_corr"] > 0).sum()
trees_area_negative = (city_corr_df["trees_area_corr"] < 0).sum()

print(f"\n  Green block correlations: {green_positive} positive, {green_negative} negative")
print(f"  Tree canopy correlations: {trees_positive} positive, {trees_negative} negative")
print(f"  Green area correlations: {green_area_positive} positive, {green_area_negative} negative")
print(f"  Tree canopy area correlations: {trees_area_positive} positive, {trees_area_negative} negative")

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


# Row 1: Distance vs. log(1 + Density)


# Panel 1a: Green area correlation (log area vs log density) vs log density
ax = axes[0, 0]
log_density = np.log1p(plot_df["mean_density"])
green_area_corrs = []
for bounds_fid in plot_df["bounds_fid"]:
    city = green_nodes_gdf[green_nodes_gdf["bounds_fid"] == bounds_fid]
    if len(city) > 1:
        log_dens = np.log1p(city["density"])
        log_area = np.log1p(city["cc_green_area_sum_800_nw"])
        mask = log_dens.notna() & log_area.notna()
        if mask.sum() > 1:
            corr = pd.DataFrame({"x": log_dens[mask], "y": log_area[mask]}).corr(method="spearman").iloc[0, 1]
        else:
            corr = np.nan
    else:
        corr = np.nan
    green_area_corrs.append(corr)
green_area_corrs = np.array(green_area_corrs)
# Use 'icefire' for green area correlation
colors_green = sns.color_palette("icefire", as_cmap=True)((green_area_corrs + 1) / 2)
ax.scatter(
    log_density,
    green_area_corrs,
    c=colors_green,
    s=30,
    alpha=0.6,
    edgecolors="white",
    linewidths=0.5,
)
# Make y=0 line bolder
ax.axhline(0, color="dimgrey", linewidth=2, linestyle="--", alpha=0.7)
ax.set_xlabel("Intra-city average log(1 + density)", fontsize=10, color="dimgrey")
ax.set_ylabel("Within-city Spearman ρ (log density vs log green area)", fontsize=10, color="dimgrey")
ax.set_title("Per City Green Area ~ Density Correlations", fontsize=11, fontweight="bold")
ax.set_xlim(log_density.min(), log_density.max() * 1.05)
ax.set_ylim(-1, 1)
ax.tick_params(colors="dimgrey", labelsize=9)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

# Panel 1b: Tree canopy area correlation (log area vs log density) vs log density
ax = axes[0, 1]
tree_area_corrs = []
for bounds_fid in plot_df["bounds_fid"]:
    city = green_nodes_gdf[green_nodes_gdf["bounds_fid"] == bounds_fid]
    if len(city) > 1:
        log_dens = np.log1p(city["density"])
        log_area = np.log1p(city["cc_trees_area_sum_800_nw"])
        mask = log_dens.notna() & log_area.notna()
        if mask.sum() > 1:
            corr = pd.DataFrame({"x": log_dens[mask], "y": log_area[mask]}).corr(method="spearman").iloc[0, 1]
        else:
            corr = np.nan
    else:
        corr = np.nan
    tree_area_corrs.append(corr)
tree_area_corrs = np.array(tree_area_corrs)
# Use 'crest' for tree area correlation
# Normalize spearman correlations to span [-1, 1] -> [0, 1] for the colormap
colors_trees = sns.color_palette("crest", as_cmap=True)((tree_area_corrs + 1) / 2)
ax.scatter(
    log_density,
    tree_area_corrs,
    c=colors_trees,
    s=30,
    alpha=0.6,
    edgecolors="white",
    linewidths=0.5,
)
# Make y=0 line bolder
ax.axhline(0, color="dimgrey", linewidth=2, linestyle="--", alpha=0.7)
ax.set_xlabel("Intra-city average log(1 + density)", fontsize=10, color="dimgrey")
ax.set_ylabel("Within-city Spearman ρ (log density vs log tree area)", fontsize=10, color="dimgrey")
ax.set_title("Per City Tree Canopy Area ~ Density Correlations", fontsize=11, fontweight="bold")
ax.set_xlim(log_density.min(), log_density.max() * 1.05)
ax.set_ylim(-1, 1)
ax.tick_params(colors="dimgrey", labelsize=9)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

# Row 2: Correlation vs. log-density
# Panel 2a: Green block correlation vs. log-density
ax = axes[1, 0]
# Use 'icefire' for green distance correlation
colors_green_corr = sns.color_palette("icefire", as_cmap=True)((plot_df["green_corr"] + 1) / 2)
ax.scatter(
    log_density,
    plot_df["green_corr"],
    c=colors_green_corr,
    s=30,
    alpha=0.6,
    edgecolors="white",
    linewidths=0.5,
)
# Make y=0 line bolder
ax.axhline(0, color="dimgrey", linewidth=2, linestyle="--", alpha=0.7)
ax.set_xlabel("Intra-city average log(1 + density)", fontsize=10, color="dimgrey")
ax.set_ylabel("Within-city Spearman ρ (log density vs log green distance)", fontsize=10, color="dimgrey")
ax.set_title("Per City Green Block Distance ~ Density Correlations", fontsize=11, fontweight="bold")
ax.set_xlim(log_density.min(), log_density.max() * 1.05)
ax.set_ylim(-1, 1)
ax.tick_params(colors="dimgrey", labelsize=9)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

# Panel 2b: Tree canopy correlation vs. log-density
ax = axes[1, 1]
# Use 'crest' for tree distance correlation and normalize to [-1,1]
colors_trees_corr = sns.color_palette("crest", as_cmap=True)((plot_df["trees_corr"] + 1) / 2)
ax.scatter(
    log_density,
    plot_df["trees_corr"],
    c=colors_trees_corr,
    s=30,
    alpha=0.6,
    edgecolors="white",
    linewidths=0.5,
)
# Make y=0 line bolder
ax.axhline(0, color="dimgrey", linewidth=2, linestyle="--", alpha=0.7)
ax.set_xlabel("Intra-city average log(1 + density)", fontsize=10, color="dimgrey")
ax.set_ylabel("Within-city Spearman ρ (log density vs log tree distance)", fontsize=10, color="dimgrey")
ax.set_title("Per City Tree Canopy Distance ~ Density Correlations", fontsize=11, fontweight="bold")
ax.set_xlim(log_density.min(), log_density.max() * 1.05)
ax.set_ylim(-1, 1)
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
            f"      • {row['city_label']} ({row['country']}): "
            f"ρ = {row['green_corr']:.3f}, "
            f"density = {row['mean_density']:.0f}"
        )
if len(dense_inequitable_green) > 0:
    print("\n    Dense cities with INEQUITABLE green access (ρ > 0.2):")
    for _, row in dense_inequitable_green.iterrows():
        print(
            f"      • {row['city_label']} ({row['country']}): "
            f"ρ = {row['green_corr']:.3f}, "
            f"density = {row['mean_density']:.0f}"
        )

# %%
"""
## Step 5: Diverging Bar Chart Visualization
"""

print("STEP 5: Generating diverging bar chart")

# Apply subtle seaborn styling
sns.set_style("whitegrid", {"grid.color": ".9", "axes.edgecolor": ".6"})
sns.set_context("paper")


# Create 8-panel figure
fig, axes = plt.subplots(
    1,
    8,
    figsize=(20, 12),
    width_ratios=[1.0, 1.5, 1.0, 1.5, 1.0, 1.5, 1.0, 1.5],
)

# Compute positive/negative counts for area correlations
green_area_positive = (city_corr_df["green_area_corr"].notna() & (city_corr_df["green_area_corr"] > 0)).sum()
green_area_negative = (city_corr_df["green_area_corr"].notna() & (city_corr_df["green_area_corr"] < 0)).sum()
trees_area_positive = (city_corr_df["trees_area_corr"].notna() & (city_corr_df["trees_area_corr"] > 0)).sum()
trees_area_negative = (city_corr_df["trees_area_corr"].notna() & (city_corr_df["trees_area_corr"] < 0)).sum()

panel_configs = [
    (axes[0], axes[1], "green_area_corr", "Green Area", green_area_negative, green_area_positive),
    (axes[2], axes[3], "trees_area_corr", "Tree Canopy Area", trees_area_negative, trees_area_positive),
    (axes[4], axes[5], "green_corr", "Green Block Distance", green_negative, green_positive),
    (axes[6], axes[7], "trees_corr", "Tree Canopy Distance", trees_negative, trees_positive),
]

for ax_full, ax_labeled, corr_col, title, n_neg, n_pos in panel_configs:
    # All panels now show correlations
    if corr_col == "green_corr" or corr_col == "green_area_corr":
        panel_palette = "icefire"
        panel_label = "Within-city Spearman ρ"
        panel_title = title
    elif corr_col == "trees_corr" or corr_col == "trees_area_corr":
        panel_palette = "crest"
        panel_label = "Within-city Spearman ρ"
        panel_title = title
    else:
        panel_palette = None
        panel_label = None
        panel_title = None

    # All panels use the same correlation plotting logic
    if corr_col in ["green_corr", "trees_corr", "green_area_corr", "trees_area_corr"]:
        plot_df_full = city_corr_df[["city_label", "country", corr_col, "n_nodes"]].dropna()
        plot_df_full = plot_df_full.sort_values(corr_col)
        # Use correct palette for each correlation panel and normalize correlations to [-1,1]
        if panel_palette == "icefire":
            colors_full = sns.color_palette("icefire", as_cmap=True)((plot_df_full[corr_col] + 1) / 2)
        else:
            colors_full = sns.color_palette("crest", as_cmap=True)((plot_df_full[corr_col] + 1) / 2)
        y_pos_full = range(len(plot_df_full))
        ax_full.barh(y_pos_full, plot_df_full[corr_col], color=colors_full, edgecolor="none", height=1.0)
        ax_full.axvline(0, color="dimgrey", linewidth=2)
        ax_full.set_yticks([])
        ax_full.set_xlabel(panel_label, fontsize=9, color="dimgrey")
        ax_full.set_title(f"{panel_title}\n(all {len(plot_df_full)} cities)", fontsize=11, fontweight="bold")
        ax_full.set_xlim(-1, 1)
        ax_full.tick_params(axis="x", colors="dimgrey", labelsize=8)
        ax_full.xaxis.set_major_locator(plt.MultipleLocator(0.4))
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
        if panel_palette == "icefire":
            colors_bottom = sns.color_palette("icefire", as_cmap=True)((plot_df_bottom[corr_col] + 1) / 2)
            colors_top = sns.color_palette("icefire", as_cmap=True)((plot_df_top[corr_col] + 1) / 2)
        else:
            colors_bottom = sns.color_palette("crest", as_cmap=True)((plot_df_bottom[corr_col] + 1) / 2)
            colors_top = sns.color_palette("crest", as_cmap=True)((plot_df_top[corr_col] + 1) / 2)
        ax_labeled.barh(y_pos_bottom, plot_df_bottom[corr_col], color=colors_bottom, edgecolor="none", height=0.8)
        # Plot top 20 (positive correlations)
        ax_labeled.barh(y_pos_top, plot_df_top[corr_col], color=colors_top, edgecolor="none", height=0.8)
        # Set y-ticks and labels
        all_y_pos = y_pos_bottom + y_pos_top
        all_labels = list(plot_df_bottom["city_label"]) + list(plot_df_top["city_label"])
        ax_labeled.set_yticks(all_y_pos)
        ax_labeled.set_yticklabels(all_labels, fontsize=8, color="dimgrey")
        ax_labeled.axvline(0, color="dimgrey", linewidth=2)
        ax_labeled.set_xlabel(panel_label, fontsize=9, color="dimgrey")
        ax_labeled.set_title(f"{panel_title}\n(Top/Bottom 20)", fontsize=11, fontweight="bold")
        ax_labeled.set_xlim(-1, 1)
        ax_labeled.tick_params(axis="x", colors="dimgrey", labelsize=8)
        ax_labeled.tick_params(axis="y", length=0)
        ax_labeled.xaxis.set_major_locator(plt.MultipleLocator(0.4))
    else:
        # For area columns, use log1p transform before plotting
        plot_df_full = city_corr_df[["city_label", "country", corr_col, "n_nodes"]].dropna()
        plot_df_full = plot_df_full.copy()
        plot_df_full["log_area"] = np.log1p(plot_df_full[corr_col])
        plot_df_full = plot_df_full.sort_values("log_area")
        colors_full = sns.color_palette(panel_palette, as_cmap=True)(
            (plot_df_full["log_area"] - plot_df_full["log_area"].min())
            / (plot_df_full["log_area"].max() - plot_df_full["log_area"].min() + 1e-9)
        )
        y_pos_full = range(len(plot_df_full))
        ax_full.barh(y_pos_full, plot_df_full["log_area"], color=colors_full, edgecolor="none", height=1.0)
        ax_full.set_yticks([])
        ax_full.set_xlabel(panel_label, fontsize=9, color="dimgrey")
        ax_full.set_title(f"{panel_title}\n(all {len(plot_df_full)} cities)", fontsize=11, fontweight="bold")
        ax_full.set_xlim(-1, 1)
        ax_full.tick_params(axis="x", colors="dimgrey", labelsize=8)
        # --- Labeled extremes panel ---
        n_show = min(40, len(plot_df_full))
        n_each = n_show // 2
        plot_df_bottom = plot_df_full.head(n_each).copy()  # Smallest area
        plot_df_top = plot_df_full.tail(n_each).copy()  # Largest area
        gap_size = 2
        y_pos_bottom = list(range(n_each))
        y_pos_top = list(range(n_each + gap_size, 2 * n_each + gap_size))
        colors_bottom = sns.color_palette(panel_palette, as_cmap=True)(
            (plot_df_bottom["log_area"] - plot_df_full["log_area"].min())
            / (plot_df_full["log_area"].max() - plot_df_full["log_area"].min() + 1e-9)
        )
        ax_labeled.barh(y_pos_bottom, plot_df_bottom["log_area"], color=colors_bottom, edgecolor="none", height=0.8)
        colors_top = sns.color_palette(panel_palette, as_cmap=True)(
            (plot_df_top["log_area"] - plot_df_full["log_area"].min())
            / (plot_df_full["log_area"].max() - plot_df_full["log_area"].min() + 1e-9)
        )
        ax_labeled.barh(y_pos_top, plot_df_top["log_area"], color=colors_top, edgecolor="none", height=0.8)
        all_y_pos = y_pos_bottom + y_pos_top
        all_labels = list(plot_df_bottom["city_label"]) + list(plot_df_top["city_label"])
        ax_labeled.set_yticks(all_y_pos)
        ax_labeled.set_yticklabels(all_labels, fontsize=8, color="dimgrey")
        ax_labeled.set_xlabel(panel_label, fontsize=9, color="dimgrey")
        ax_labeled.set_title(f"{panel_title}\n(Top/Bottom 20)", fontsize=11, fontweight="bold")
        ax_labeled.set_xlim(-1, 1)
        ax_labeled.tick_params(axis="x", colors="dimgrey", labelsize=8)
        ax_labeled.tick_params(axis="y", length=0)

# Remove spines for cleaner look
for ax in axes:
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("lightgrey")
plt.suptitle(
    "Per-city correlations for green space and tree canopy variables against population density",
    fontsize=12,
    fontweight="bold",
    y=0.995,
)
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
strongest_green_area_neg = city_corr_df.nsmallest(5, "green_area_corr")
strongest_green_area_pos = city_corr_df.nlargest(5, "green_area_corr")
strongest_trees_area_neg = city_corr_df.nsmallest(5, "trees_area_corr")
strongest_trees_area_pos = city_corr_df.nlargest(5, "trees_area_corr")

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
        f"### Green Area (800m buffer): {green_area_negative} cities negative, {green_area_positive} cities positive",
        "",
        "**Strongest negative (denser = more green area):**",
        "",
    ]
)
for _, row in strongest_green_area_neg.iterrows():
    report_lines.append(f"- {row['city_label']} ({row['country']}): r = {row['green_area_corr']:.3f}")

report_lines.extend(["", "**Strongest positive (denser = less green area):**", ""])
for _, row in strongest_green_area_pos.iterrows():
    report_lines.append(f"- {row['city_label']} ({row['country']}): r = {row['green_area_corr']:.3f}")

report_lines.extend(
    [
        "",
        f"### Tree Canopy Area (800m buffer): {trees_area_negative} cities negative, {trees_area_positive} cities positive",
        "",
        "**Strongest negative (denser = more tree canopy):**",
        "",
    ]
)
for _, row in strongest_trees_area_neg.iterrows():
    report_lines.append(f"- {row['city_label']} ({row['country']}): r = {row['trees_area_corr']:.3f}")

report_lines.extend(["", "**Strongest positive (denser = less tree canopy):**", ""])
for _, row in strongest_trees_area_pos.iterrows():
    report_lines.append(f"- {row['city_label']} ({row['country']}): r = {row['trees_area_corr']:.3f}")

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
