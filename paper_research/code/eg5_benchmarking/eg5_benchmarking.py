# %% [markdown];
"""
# 15-Minute City Benchmarking

Evaluates how well European cities approximate the "15-minute city" ideal by
measuring the proportion of street network nodes with access to all essential
POI categories within 15-minute (1200m) and 20-minute (1600m) walking distances.

## Steps
1. Load city saturation results from EG1 and filter to well-covered cities
2. Load POI distance metrics for all categories from city metrics files
3. Compute per-node completeness scores (how many categories accessible)
4. Aggregate to city-level statistics (% nodes with full access)
5. Generate rankings, visualizations, and summary report

## Key Outputs
- **city_15min_scores.csv**: Per-city 15-minute completeness scores
- **city_20min_scores.csv**: Per-city 20-minute completeness scores
- **15min_city_ranking.png**: Bar chart of top/bottom cities
- **completeness_distribution.png**: Histogram of completeness across cities
- **README.md**: Summary report with key findings

## Metrics Used (from SOAR pre-computed)
- `cc_{category}_nearest_max_1600`: Network distance to nearest POI of each category (m)

## 15-Minute City Concept
The "15-minute city" (ville du quart d'heure) proposes that residents should access
essential daily services within a 15-minute walk. We operationalize this as:
- 15-minute threshold: 1200m (assuming ~80m/min walking speed)
- 20-minute threshold: 1600m (more relaxed standard)

## POI Categories Assessed (11 categories)
1. accommodation <- skipped
2. active_life
3. arts_and_entertainment
4. attractions_and_activities
5. business_and_services
6. eat_and_drink
7. education
8. health_and_medical
9. public_services
10. religious
11. retail
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# %%
"""
## Configuration
"""

# POI categories for 15-minute city assessment
# These represent essential daily services
POI_CATEGORIES = [
    # "accommodation",  # Excluded as not essential for daily access
    "active_life",
    "arts_and_entertainment",
    "attractions_and_activities",
    "business_and_services",
    "eat_and_drink",
    "education",
    "health_and_medical",
    "public_services",
    "religious",
    "retail",
]

# Generate column names for nearest distance metrics
# Pattern: cc_{category}_nearest_max_1600
POI_DISTANCE_COLS = [f"cc_{cat}_nearest_max_1600" for cat in POI_CATEGORIES]

# Walking distance threshold (meters)
THRESHOLD_15MIN = 1200  # ~15 min at 80m/min

# Minimum nodes per city for reliable statistics
MIN_NODES = 100
MIN_CITIES_PER_COUNTRY = 3  # For country-level aggregation

# Configuration paths - modify these as needed
BOUNDS_PATH = "temp/datasets/boundaries.gpkg"
METRICS_DIR = "temp/cities_data/processed"
SATURATION_RESULTS_PATH = "paper_research/code/eg1_data_quality/outputs/city_analysis_results.gpkg"
OUTPUT_DIR = "paper_research/code/eg5_benchmarking/outputs"
TEMP_DIR = "temp/egs/eg5_benchmarking"

# Saturation quadrants to include (reliable POI data)
# Use Consistently Saturated and Variable Saturated for broader coverage
SATURATED_QUADRANTS = ["Consistently Saturated", "Variable Saturated"]

# %%
"""
## Setup Paths
"""

output_path = Path(OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)
metrics_dir = Path(METRICS_DIR)
temp_path = Path(TEMP_DIR)
temp_path.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Exploratory Question 5: 15-Minute City Benchmarking")
print("=" * 80)
print(f"\nOutput directory: {output_path}")
print(f"Temp directory: {temp_path}")

# %%
"""
## Step 1: Load Saturation Results and Filter to Saturated Cities
"""

print("\nSTEP 1: Loading saturation results and filtering cities")

# Load saturation results from EG1
saturation_gdf = gpd.read_file(SATURATION_RESULTS_PATH)
print(f"  Loaded saturation results for {len(saturation_gdf)} cities")

# For 15-minute city analysis, filter to cities with good combined POI coverage
# Use the between-category quadrant metric from EG1 to ensure reliable data quality
if "between_category_quadrant" not in saturation_gdf.columns:
    raise ValueError("between_category_quadrant column not found in saturation results. Run EG1 first.")

# Filter to cities with combined saturation in reliable quadrants
saturated_cities = saturation_gdf[saturation_gdf["between_category_quadrant"].isin(SATURATED_QUADRANTS)].copy()

print(f"\n  Cities with combined saturation in {SATURATED_QUADRANTS}: {len(saturated_cities)}")
print("  Distribution of combined saturation quadrants:")
for quadrant in saturation_gdf["between_category_quadrant"].unique():
    count = (saturation_gdf["between_category_quadrant"] == quadrant).sum()
    print(f"    {quadrant}: {count} cities")

# Get list of saturated bounds_fids
saturated_fids = set(saturated_cities["bounds_fid"].tolist())

# %%
"""
## Step 2: Load POI Distance Metrics for Cities
"""

print("\nSTEP 2: Loading POI distance metrics for cities")

# Check for cached data
cache_file = temp_path / "poi_minutes_city_data.parquet"

if cache_file.exists():
    print("  Loading cached POI distance data...")
    city_data = pd.read_parquet(cache_file).to_dict("records")
    print(f"  Loaded cached data for {len(city_data)} cities")
else:
    print("  Loading individual city metrics files...")
    city_data = []

    for idx, row in tqdm(
        saturated_cities.iterrows(),
        total=len(saturated_cities),
        desc="Loading city metrics",
    ):
        bounds_fid = row["bounds_fid"]
        city_label = row.get("label", str(bounds_fid))
        country = row.get("country", "Unknown")

        # Load metrics file for this city
        metrics_file = metrics_dir / f"metrics_{bounds_fid}.gpkg"
        if not metrics_file.exists():
            continue

        try:
            # Load only the distance columns we need
            gdf = gpd.read_file(metrics_file, columns=POI_DISTANCE_COLS, layer="streets")
            # Doublecheck geoms are dropped if outside boundary
            gdf = gdf[gdf.geometry.within(row.geometry)]
            # Filter out rows with any NaN/inf values in distance columns
            valid_mask = pd.Series(True, index=gdf.index)
            for col in POI_DISTANCE_COLS:
                if col in gdf.columns:
                    valid_mask &= gdf[col].notna() & np.isfinite(gdf[col]) & (gdf[col] >= 0)

            gdf = gdf[valid_mask]

            if len(gdf) < MIN_NODES:
                continue

            n_nodes = len(gdf)

            # Compute completeness scores
            # For each node, count how many categories are within the threshold
            completeness_15min = pd.Series(0, index=gdf.index)

            categories_present = []
            for cat, col in zip(POI_CATEGORIES, POI_DISTANCE_COLS):
                if col in gdf.columns:
                    categories_present.append(cat)
                    completeness_15min += (gdf[col] <= THRESHOLD_15MIN).astype(int)

            n_categories = len(categories_present)

            # Compute statistics
            # Full access = all categories within threshold
            pct_full_15min = (completeness_15min == n_categories).sum() / n_nodes * 100

            # Mean completeness (0-1 scale)
            mean_completeness_15min = completeness_15min.mean() / n_categories

            # Median number of categories accessible
            median_categories_15min = completeness_15min.median()

            # Percentage of nodes with at least N categories accessible
            pct_ge_6_15min = (completeness_15min >= 6).sum() / n_nodes * 100
            pct_ge_9_15min = (completeness_15min >= 9).sum() / n_nodes * 100

            # Per-category access rates
            category_access_15min = {}
            for cat, col in zip(POI_CATEGORIES, POI_DISTANCE_COLS):
                if col in gdf.columns:
                    category_access_15min[f"pct_{cat}_15min"] = (gdf[col] <= THRESHOLD_15MIN).sum() / n_nodes * 100

            city_record = {
                "bounds_fid": bounds_fid,
                "city_label": city_label,
                "country": country,
                "n_nodes": n_nodes,
                "n_categories": n_categories,
                # 15-minute metrics
                "pct_full_15min": pct_full_15min,
                "mean_completeness_15min": mean_completeness_15min,
                "median_categories_15min": median_categories_15min,
                "pct_ge_6_15min": pct_ge_6_15min,
                "pct_ge_9_15min": pct_ge_9_15min,
            }
            # Add per-category access rates
            city_record.update(category_access_15min)

            city_data.append(city_record)

        except Exception:
            continue

    print(f"  Loaded data for {len(city_data)} cities")

    # Save to cache
    if city_data:
        cache_df = pd.DataFrame(city_data)
        cache_df.to_parquet(cache_file)
        print(f"  Saved cache to {cache_file}")

# %%
"""
## Step 3: Create City DataFrame and Compute Rankings
"""

print("\nSTEP 3: Computing city rankings")

city_df = pd.DataFrame(city_data)

# Sort by 15-minute full access percentage
city_df_15min = city_df.sort_values("pct_full_15min", ascending=False)

# Summary statistics
print("\n  15-Minute City Summary:")
print(f"    Cities analyzed: {len(city_df)}")
print(f"    Mean full access: {city_df['pct_full_15min'].mean():.1f}%")
print(f"    Median full access: {city_df['pct_full_15min'].median():.1f}%")
print(f"    Best city: {city_df_15min.iloc[0]['city_label']} ({city_df_15min.iloc[0]['pct_full_15min']:.1f}%)")
print(f"    Worst city: {city_df_15min.iloc[-1]['city_label']} ({city_df_15min.iloc[-1]['pct_full_15min']:.1f}%)")

# %%
"""
## Step 4: Generate Visualizations
"""

print("\nSTEP 4: Generating visualizations")

# Apply subtle seaborn styling
sns.set_style("whitegrid", {"grid.color": ".9", "axes.edgecolor": ".6"})
sns.set_context("paper")

# Figure 1: Top and bottom cities bar chart (15-minute)
fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# Get min/max values across both top and bottom plots for consistent scaling
top_20 = city_df_15min.head(20)
bottom_20 = city_df_15min.tail(20)  # Don't reverse - lowest will be at bottom with invert_yaxis
val_min = min(top_20["pct_full_15min"].min(), bottom_20["pct_full_15min"].min())
val_max = max(top_20["pct_full_15min"].max(), bottom_20["pct_full_15min"].max())

# Top 20 cities - use bright end of colormap (high values)
ax = axes[0]
colors_top = plt.cm.viridis(np.linspace(0.5, 1.0, len(top_20)))[::-1]  # Brightest at top
bars = ax.barh(range(len(top_20)), top_20["pct_full_15min"], color=colors_top)
ax.set_yticks(range(len(top_20)))
ax.set_yticklabels([f"{row['city_label']} ({row['country']})" for _, row in top_20.iterrows()])
ax.set_xlabel("% Nodes with Full 15-min Access", fontsize=10)
ax.set_title("Top 20 Cities: 15-Minute City Completeness", fontsize=11, fontweight="bold")
ax.invert_yaxis()
ax.set_xlim(0, 100)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# Bottom 20 cities - use dark end of colormap (low values)
ax = axes[1]
colors_bottom = plt.cm.viridis(np.linspace(0, 0.5, len(bottom_20)))[::-1]  # Darkest at bottom
bars = ax.barh(range(len(bottom_20)), bottom_20["pct_full_15min"], color=colors_bottom)
ax.set_yticks(range(len(bottom_20)))
ax.set_yticklabels([f"{row['city_label']} ({row['country']})" for _, row in bottom_20.iterrows()])
ax.set_xlabel("% Nodes with Full 15-min Access", fontsize=10)
ax.set_title("Bottom 20 Cities: 15-Minute City Completeness", fontsize=11, fontweight="bold")
ax.invert_yaxis()
ax.set_xlim(0, 100)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
ranking_path = output_path / "15min_city_ranking.png"
plt.savefig(ranking_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  Saved ranking plot to {ranking_path}")

# Figure 2: Distribution of completeness scores
fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(city_df["pct_full_15min"], bins=30, edgecolor="black", alpha=0.7, color=plt.cm.viridis(0.6))
ax.axvline(
    city_df["pct_full_15min"].median(),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Median: {city_df['pct_full_15min'].median():.1f}%",
)
ax.set_xlabel("% Nodes with Full 15-min Access", fontsize=10)
ax.set_ylabel("Number of Cities", fontsize=10)
ax.set_title("Distribution of 15-Minute City Scores", fontsize=11, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
dist_path = output_path / "completeness_distribution.png"
plt.savefig(dist_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  Saved distribution plot to {dist_path}")

# Figure 3: Per-category access rates (heatmap for top/bottom cities)
fig, axes = plt.subplots(1, 2, figsize=(14, 10))

# Prepare data for heatmap
category_cols_15min = [f"pct_{cat}_15min" for cat in POI_CATEGORIES]
category_labels = [cat.replace("_", " ").title() for cat in POI_CATEGORIES]

# Compute min/max across both heatmaps for consistent scaling
top_15 = city_df_15min.head(15)
bottom_15 = city_df_15min.tail(15).iloc[::-1]
heatmap_data_top = top_15[category_cols_15min].values
heatmap_data_bottom = bottom_15[category_cols_15min].values
hmap_min = min(heatmap_data_top.min(), heatmap_data_bottom.min())
hmap_max = max(heatmap_data_top.max(), heatmap_data_bottom.max())

# Top 15 cities heatmap
ax = axes[0]
im = ax.imshow(heatmap_data_top, aspect="auto", cmap="viridis", vmin=hmap_min, vmax=hmap_max)
ax.set_xticks(range(len(POI_CATEGORIES)))
ax.set_xticklabels(category_labels, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(top_15)))
ax.set_yticklabels([f"{row['city_label']}" for _, row in top_15.iterrows()], fontsize=9)
ax.set_title("Top 15 Cities: Category Access (15-min)", fontsize=11, fontweight="bold")
plt.colorbar(im, ax=ax, label="% Nodes with Access", shrink=0.8)

# Bottom 15 cities heatmap
ax = axes[1]
im = ax.imshow(heatmap_data_bottom, aspect="auto", cmap="viridis", vmin=hmap_min, vmax=hmap_max)
ax.set_xticks(range(len(POI_CATEGORIES)))
ax.set_xticklabels(category_labels, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(bottom_15)))
ax.set_yticklabels([f"{row['city_label']}" for _, row in bottom_15.iterrows()], fontsize=9)
ax.set_title("Bottom 15 Cities: Category Access (15-min)", fontsize=11, fontweight="bold")
plt.colorbar(im, ax=ax, label="% Nodes with Access", shrink=0.8)

plt.tight_layout()
heatmap_path = output_path / "category_access_heatmap.png"
plt.savefig(heatmap_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  Saved heatmap to {heatmap_path}")

# %%
"""
## Step 5: Export Results
"""

print("\nSTEP 5: Exporting results")

# Export 15-minute scores
cols_15min = [
    "city_label",
    "country",
    "n_nodes",
    "pct_full_15min",
    "mean_completeness_15min",
    "median_categories_15min",
    "pct_ge_6_15min",
    "pct_ge_9_15min",
]
export_15min = city_df_15min[cols_15min].copy()
export_15min.columns = [
    "City",
    "Country",
    "Nodes",
    "% Full Access",
    "Mean Completeness",
    "Median Categories",
    "% >= 6 Categories",
    "% >= 9 Categories",
]
out_file = output_path / "city_15min_scores.csv"
export_15min.to_csv(out_file, index=False, float_format="%.1f")
print(f"  Saved 15-minute scores to {out_file}")

# Export per-category access rates
category_cols_all = ["city_label", "country"] + [f"pct_{cat}_15min" for cat in POI_CATEGORIES]
export_categories = city_df_15min[category_cols_all].copy()
out_file = output_path / "city_category_access_15min.csv"
export_categories.to_csv(out_file, index=False, float_format="%.1f")
print(f"  Saved per-category access rates to {out_file}")

# %%
"""
## Step 6: Generate Summary Report
"""

print("\nSTEP 6: Generating summary report")

# Identify bottleneck categories (lowest access rates on average)
category_means = {}
for cat in POI_CATEGORIES:
    col = f"pct_{cat}_15min"
    if col in city_df.columns:
        category_means[cat] = city_df[col].mean()

category_means_sorted = sorted(category_means.items(), key=lambda x: x[1])

# Top/bottom cities
top_10_15min = city_df_15min.head(10)
bottom_10_15min = city_df_15min.tail(10).iloc[::-1]

report_lines = [
    "# 15-Minute City Benchmarking Report",
    "",
    f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}",
    "",
    "## Overview",
    "",
    'This analysis evaluates how well European cities approximate the "15-minute city" ideal,',
    "where residents can access all essential services within a 15-minute walk (1200m).",
    "",
    "## Summary Statistics",
    "",
    f"- **Cities Analyzed:** {len(city_df)}",
    f"- **Total Street Network Nodes:** {city_df['n_nodes'].sum():,}",
    f"- **POI Categories Assessed:** {len(POI_CATEGORIES)}",
    "",
    "### 15-Minute City (1200m threshold)",
    "",
    f"- **Mean Full Access:** {city_df['pct_full_15min'].mean():.1f}% of nodes",
    f"- **Median Full Access:** {city_df['pct_full_15min'].median():.1f}% of nodes",
    f"- **Range:** {city_df['pct_full_15min'].min():.1f}% to {city_df['pct_full_15min'].max():.1f}%",
    f"- **Cities with >50% Full Access:** {(city_df['pct_full_15min'] > 50).sum()} ({(city_df['pct_full_15min'] > 50).mean() * 100:.1f}%)",
    "",
    "## Top 10 Cities (15-Minute Access)",
    "",
    "| Rank | City | Country | % Full Access | Mean Completeness |",
    "|------|------|---------|---------------|-------------------|",
]

for rank, (_, row) in enumerate(top_10_15min.iterrows(), 1):
    report_lines.append(
        f"| {rank} | {row['city_label']} | {row['country']} | {row['pct_full_15min']:.1f}% | {row['mean_completeness_15min']:.2f} |"
    )

report_lines.extend(
    [
        "",
        "## Bottom 10 Cities (15-Minute Access)",
        "",
        "| Rank | City | Country | % Full Access | Mean Completeness |",
        "|------|------|---------|---------------|-------------------|",
    ]
)

for rank, (_, row) in enumerate(bottom_10_15min.iterrows(), 1):
    report_lines.append(
        f"| {rank} | {row['city_label']} | {row['country']} | {row['pct_full_15min']:.1f}% | {row['mean_completeness_15min']:.2f} |"
    )

report_lines.extend(
    [
        "",
        "## Bottleneck Categories",
        "",
        "Categories with lowest average access rates (limiting factors for 15-minute completeness):",
        "",
        "| Rank | Category | Mean Access Rate |",
        "|------|----------|------------------|",
    ]
)

for rank, (cat, mean_access) in enumerate(category_means_sorted[:5], 1):
    report_lines.append(f"| {rank} | {cat.replace('_', ' ').title()} | {mean_access:.1f}% |")

report_lines.extend(
    [
        "",
        "## Best-Covered Categories",
        "",
        "| Rank | Category | Mean Access Rate |",
        "|------|----------|------------------|",
    ]
)

for rank, (cat, mean_access) in enumerate(category_means_sorted[-5:][::-1], 1):
    report_lines.append(f"| {rank} | {cat.replace('_', ' ').title()} | {mean_access:.1f}% |")

report_lines.extend(
    [
        "",
        "## Visualizations",
        "",
        "### City Rankings (15-Minute)",
        "",
        "![15-Minute City Ranking](outputs/15min_city_ranking.png)",
        "",
        "### Completeness Distribution",
        "",
        "![Completeness Distribution](outputs/completeness_distribution.png)",
        "",
        "### Category Access Heatmap",
        "",
        "![Category Access Heatmap](outputs/category_access_heatmap.png)",
        "",
        "## Key Findings",
        "",
        "1. **Few cities achieve true 15-minute completeness**: The median city has only",
        f"   {city_df['pct_full_15min'].median():.1f}% of nodes with access to all {len(POI_CATEGORIES)} POI categories within 1200m.",
        "",
        f"2. **Bottleneck categories**: {category_means_sorted[0][0].replace('_', ' ').title()} and",
        f"   {category_means_sorted[1][0].replace('_', ' ').title()} are the most limiting categories,",
        "   suggesting targeted infrastructure investment priorities.",
        "",
        "3. **Geographic variation**: Top-performing cities cluster in [countries/regions],",
        "   while lower-performing cities tend to be [characteristics].",
        "",
        "## Methodology Notes",
        "",
        "- Walking distance threshold: 1200m (~15 min at 80m/min)",
        "- Network distances (not Euclidean) from street network nodes to nearest POI",
        "- Restricted to cities with sufficient POI data quality (from EG1 saturation analysis)",
        "- Required combined saturation in reliable quadrants",
        "",
        "## Output Files",
        "",
        "- `city_15min_scores.csv`: Per-city 15-minute completeness metrics",
        "- `city_category_access_15min.csv`: Per-category access rates by city",
        "- `15min_city_ranking.png`: Bar chart of top/bottom cities",
        "- `completeness_distribution.png`: Histogram of completeness scores",
        "- `category_access_heatmap.png`: Heatmap of per-category access",
        "",
    ]
)

# Write report
report_path = output_path.parent / "README.md"
with open(report_path, "w") as f:
    f.write("\n".join(report_lines))
print(f"  Saved report to {report_path}")

# %%
"""
## Step 6: LaTeX Table Generation
"""

print("\nSTEP 6: LaTeX Table Generation")

# Table 1: Top 10 Cities
latex_top = [
    r"\begin{tabular}{@{}l l r r@{}}",
    r"  \toprule",
    r"  City & Country & \% Full Access & Mean Completeness \\",
    r"  \midrule",
]
for _, row in top_10_15min.iterrows():
    city = str(row["city_label"]).replace("&", r"\&")
    latex_top.append(
        f"  {city} & {row['country']} & {row['pct_full_15min']:.1f} & {row['mean_completeness_15min']:.3f} \\\\"
    )
latex_top.extend([r"  \bottomrule", r"\end{tabular}"])

with open(output_path / "table_top_cities.tex", "w") as f:
    f.write("\n".join(latex_top))
print("  Saved: table_top_cities.tex")

# Table 2: Bottom 10 Cities
latex_bottom = [
    r"\begin{tabular}{@{}l l r r@{}}",
    r"  \toprule",
    r"  City & Country & \% Full Access & Mean Completeness \\",
    r"  \midrule",
]
for _, row in bottom_10_15min.iterrows():
    city = str(row["city_label"]).replace("&", r"\&")
    latex_bottom.append(
        f"  {city} & {row['country']} & {row['pct_full_15min']:.1f} & {row['mean_completeness_15min']:.3f} \\\\"
    )
latex_bottom.extend([r"  \bottomrule", r"\end{tabular}"])

with open(output_path / "table_bottom_cities.tex", "w") as f:
    f.write("\n".join(latex_bottom))
print("  Saved: table_bottom_cities.tex")

# Table 3: Bottleneck Categories
latex_bottlenecks = [
    r"\begin{tabular}{@{}l r@{}}",
    r"  \toprule",
    r"  Category & Mean Access Rate (\%) \\",
    r"  \midrule",
]
for cat, mean_access in category_means_sorted:
    cat_name = cat.replace("_", " ").title().replace("And", "and")
    latex_bottlenecks.append(f"  {cat_name} & {mean_access:.1f} \\\\")
latex_bottlenecks.extend([r"  \bottomrule", r"\end{tabular}"])

with open(output_path / "table_bottlenecks.tex", "w") as f:
    f.write("\n".join(latex_bottlenecks))
print("  Saved: table_bottlenecks.tex")

# %%
"""
## Step 7: Country-Level Aggregation
"""

print("\nSTEP 7: Country-level aggregation")

# Add country info to city_df if not present
if "country" in city_df.columns:
    # Compute median distances per category for each city (for country aggregation)
    city_df["essential_median"] = city_df[
        [f"pct_{cat}_15min" for cat in POI_CATEGORIES if f"pct_{cat}_15min" in city_df.columns]
    ].mean(axis=1)

    # Filter to countries with enough cities
    country_counts = city_df.groupby("country").size()
    valid_countries = country_counts[country_counts >= MIN_CITIES_PER_COUNTRY].index.tolist()
    print(f"  Countries with >= {MIN_CITIES_PER_COUNTRY} cities: {len(valid_countries)}")

    country_data = city_df[city_df["country"].isin(valid_countries)].copy()

    # Aggregate to country level
    country_stats = []
    for country in valid_countries:
        cc = country_data[country_data["country"] == country]
        stats = {
            "country": country,
            "n_cities": len(cc),
            "mean_pct_full_15min": cc["pct_full_15min"].mean(),
            "median_pct_full_15min": cc["pct_full_15min"].median(),
        }
        # Add per-category means
        for cat in POI_CATEGORIES:
            col = f"pct_{cat}_15min"
            if col in cc.columns:
                stats[f"mean_{col}"] = cc[col].mean()
        country_stats.append(stats)

    country_df = pd.DataFrame(country_stats).sort_values("mean_pct_full_15min", ascending=False)

    print("\n  Top 5 Countries by 15-min Access:")
    for _, row in country_df.head(5).iterrows():
        print(f"    {row['country']}: {row['mean_pct_full_15min']:.1f}% (n={int(row['n_cities'])})")

    # Save country results
    country_df.to_csv(output_path / "country_15min_scores.csv", index=False, float_format="%.1f")
    print("\n  Saved country_15min_scores.csv")

    # Country ranking visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_data = country_df.sort_values("mean_pct_full_15min")
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(plot_data)))

    ax.barh(range(len(plot_data)), plot_data["mean_pct_full_15min"], color=colors)
    ax.set_yticks(range(len(plot_data)))
    ax.set_yticklabels(plot_data["country"])
    ax.set_xlabel("Mean % Full 15-min Access")
    ax.set_title("Country Rankings: 15-Minute City Performance", fontweight="bold")

    for i, (_, row) in enumerate(plot_data.iterrows()):
        ax.annotate(f"n={int(row['n_cities'])}", xy=(row["mean_pct_full_15min"] + 1, i), va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path / "country_ranking_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved country_ranking_chart.png")

    # LaTeX table for countries
    latex_country = [
        r"\begin{tabular}{@{}l r r r@{}}",
        r"  \toprule",
        r"  Country & Cities & Mean \% Full Access & Median \% Full Access \\",
        r"  \midrule",
    ]
    for _, row in country_df.head(5).iterrows():
        latex_country.append(
            f"  {row['country']} & {int(row['n_cities'])} & {row['mean_pct_full_15min']:.1f} & {row['median_pct_full_15min']:.1f} \\\\"
        )
    latex_country.append(r"  \ldots & & & \\")
    for _, row in country_df.tail(5).iterrows():
        latex_country.append(
            f"  {row['country']} & {int(row['n_cities'])} & {row['mean_pct_full_15min']:.1f} & {row['median_pct_full_15min']:.1f} \\\\"
        )
    latex_country.extend([r"  \bottomrule", r"\end{tabular}"])

    with open(output_path / "table_country_rankings.tex", "w") as f:
        f.write("\n".join(latex_country))
    print("  Saved table_country_rankings.tex")

# %%
"""
## Analysis Complete
"""

print("\n" + "=" * 80)
print("DEMONSTRATOR 5 COMPLETE")
print("=" * 80)
print(f"\nOutputs saved to: {output_path}")
print("\nCity-level files:")
print("  - city_15min_scores.csv")
print("  - city_category_access_15min.csv")
print("  - 15min_city_ranking.png")
print("  - completeness_distribution.png")
print("  - category_access_heatmap.png")
print("\nCountry-level files:")
print("  - country_15min_scores.csv")
print("  - country_ranking_chart.png")
print("  - table_country_rankings.tex")
print(f"\nREADME.md saved to: {report_path}")
