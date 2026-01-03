# %% [markdown];
"""
# European Urban Metrics Benchmarking Reference

Computes reference statistics (5th, 25th, 50th, 75th, 95th percentiles) across European cities
for key urban metrics including land-use accessibility, diversity (Hill numbers), network
centrality, green space access, building morphology, and census demographics.

## Metric Categories
1. **Land-use accessibility** - Distance to nearest POI by category
2. **Land-use diversity** - Hill numbers q0, q1, q2 at 400m
3. **Network centrality** - Betweenness, density, cycles at 400m
4. **Green space** - Distance to green/trees, area coverage
5. **Building morphology** - Height, area, compactness, volume
6. **Census demographics** - Population density, employment, age structure

## Key Outputs
- **benchmark_reference.csv**: Full benchmark statistics per city
- **benchmark_summary.csv**: Aggregated percentile statistics across all cities
- **country_rankings.csv**: Country-level aggregations
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

# Distance threshold for metrics (meters) - ~5 min walk
DISTANCE = 400

# Land-use categories
LANDUSE_CATEGORIES = [
    "accommodation",
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

# Infrastructure categories
INFRASTRUCTURE_CATEGORIES = [
    "street_furn",
    "parking",
    "transport",
]

# Metrics to extract - using 400m distance where applicable
METRICS_CONFIG = {
    # Land-use nearest distance (meters)
    "landuse_nearest": {
        "columns": [f"cc_{cat}_nearest_max_1600" for cat in LANDUSE_CATEGORIES],
        "labels": [f"{cat}_nearest_m" for cat in LANDUSE_CATEGORIES],
    },
    # Land-use counts at distance
    "landuse_count": {
        "columns": [f"cc_{cat}_{DISTANCE}_nw" for cat in LANDUSE_CATEGORIES],
        "labels": [f"{cat}_count_{DISTANCE}m" for cat in LANDUSE_CATEGORIES],
    },
    # Infrastructure nearest
    "infra_nearest": {
        "columns": [f"cc_{cat}_nearest_max_1600" for cat in INFRASTRUCTURE_CATEGORIES],
        "labels": [f"{cat}_nearest_m" for cat in INFRASTRUCTURE_CATEGORIES],
    },
    # Hill diversity indices at distance
    "hill_diversity": {
        "columns": [f"cc_hill_q0_{DISTANCE}_nw", f"cc_hill_q1_{DISTANCE}_nw", f"cc_hill_q2_{DISTANCE}_nw"],
        "labels": ["hill_q0_richness", "hill_q1_shannon", "hill_q2_simpson"],
    },
    # Network centrality at distance
    "centrality": {
        "columns": [
            f"cc_betweenness_{DISTANCE}",
            f"cc_density_{DISTANCE}",
            f"cc_cycles_{DISTANCE}",
            f"cc_harmonic_{DISTANCE}",
        ],
        "labels": ["betweenness", "network_density", "cycles", "harmonic_closeness"],
    },
    # Green space
    "green": {
        "columns": [
            "cc_green_nearest_max_1600",
            "cc_trees_nearest_max_1600",
            f"cc_green_area_sum_{DISTANCE}_nw",
            f"cc_trees_area_sum_{DISTANCE}_nw",
        ],
        "labels": ["green_nearest_m", "trees_nearest_m", f"green_area_{DISTANCE}m", f"trees_area_{DISTANCE}m"],
    },
    # Building morphology (200m aggregation - local scale)
    "morphology": {
        "columns": [
            "cc_mean_height_median_200_nw",
            "cc_area_median_200_nw",
            "cc_compactness_median_200_nw",
            "cc_volume_median_200_nw",
            "cc_floor_area_ratio_median_200_nw",
            "cc_block_covered_ratio_median_200_nw",
        ],
        "labels": [
            "building_height_m",
            "building_area_m2",
            "building_compactness",
            "building_volume_m3",
            "floor_area_ratio",
            "block_coverage_ratio",
        ],
    },
    # Census demographics
    "census": {
        "columns": [
            "density",
            "emp_%",
            "y_lt15_%",
            "y_1564_%",
            "y_ge65_%",
            "m_%",
            "f_%",
        ],
        "labels": [
            "pop_density_per_km2",
            "employment_rate",
            "age_under15_pct",
            "age_15to64_pct",
            "age_over65_pct",
            "male_pct",
            "female_pct",
        ],
    },
}

# Minimum nodes per city for reliable statistics
MIN_NODES = 100
MIN_CITIES_PER_COUNTRY = 3

# Configuration paths
BOUNDS_PATH = "temp/datasets/boundaries.gpkg"
METRICS_DIR = "temp/cities_data/processed"
SATURATION_RESULTS_PATH = "paper_research/code/eg1_data_quality/outputs/city_analysis_results.gpkg"
OUTPUT_DIR = "paper_research/code/eg5_benchmarking/outputs"
TEMP_DIR = "temp/egs/eg5_benchmarking"

# Saturation quadrants to include (reliable data)
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
print("Exploratory Question 5: European Urban Metrics Benchmarking")
print("=" * 80)
print(f"\nOutput directory: {output_path}")
print(f"Distance threshold: {DISTANCE}m")

# %%
"""
## Step 1: Load City List from Saturation Results
"""

print("\nSTEP 1: Loading saturation results and filtering cities")

saturation_gdf = gpd.read_file(SATURATION_RESULTS_PATH)
print(f"  Loaded saturation results for {len(saturation_gdf)} cities")

if "between_category_quadrant" not in saturation_gdf.columns:
    raise ValueError("between_category_quadrant column not found. Run EG1 first.")

saturated_cities = saturation_gdf[saturation_gdf["between_category_quadrant"].isin(SATURATED_QUADRANTS)].copy()
print(f"  Cities with reliable POI coverage: {len(saturated_cities)}")

saturated_fids = set(saturated_cities["bounds_fid"].tolist())

# %%
"""
## Step 2: Extract Raw Metrics for All Cities (Pooled)
"""

print("\nSTEP 2: Extracting raw metrics from all cities")

# Build flat list of all columns needed
all_columns = []
all_labels = []
for category, config in METRICS_CONFIG.items():
    all_columns.extend(config["columns"])
    all_labels.extend(config["labels"])

# Check for cached pooled data
cache_file = temp_path / "benchmark_pooled_data.parquet"
city_cache_file = temp_path / "benchmark_city_data.parquet"

if cache_file.exists() and city_cache_file.exists():
    print("  Loading cached pooled benchmark data...")
    pooled_df = pd.read_parquet(cache_file)
    city_df = pd.read_parquet(city_cache_file)
    print(f"  Loaded cached data: {len(pooled_df):,} nodes from {len(city_df)} cities")
else:
    print("  Loading individual city metrics files...")
    pooled_records = []
    city_records = []

    for idx, row in tqdm(
        saturated_cities.iterrows(),
        total=len(saturated_cities),
        desc="Processing cities",
    ):
        bounds_fid = row["bounds_fid"]
        city_label = row.get("label", str(bounds_fid))
        country = row.get("country", "Unknown")

        metrics_file = metrics_dir / f"metrics_{bounds_fid}.gpkg"
        if not metrics_file.exists():
            continue

        try:
            # Load streets layer
            gdf = gpd.read_file(metrics_file, layer="streets")
            gdf = gdf[gdf.geometry.within(row.geometry)]

            if len(gdf) < MIN_NODES:
                continue

            n_nodes = len(gdf)

            # Extract raw values for pooled statistics
            for i, (col, label) in enumerate(zip(all_columns, all_labels)):
                if col in gdf.columns:
                    values = gdf[col].dropna()
                    values = values[np.isfinite(values)]

                    for val in values:
                        pooled_records.append(
                            {
                                "bounds_fid": bounds_fid,
                                "country": country,
                                "metric": label,
                                "value": val,
                            }
                        )

            # Also keep city-level summary for maps and country aggregations
            city_record = {
                "bounds_fid": bounds_fid,
                "city_label": city_label,
                "country": country,
                "n_nodes": n_nodes,
            }

            for col, label in zip(all_columns, all_labels):
                if col in gdf.columns:
                    values = gdf[col].dropna()
                    values = values[np.isfinite(values)]

                    if len(values) > 0:
                        city_record[f"{label}_median"] = values.median()
                    else:
                        city_record[f"{label}_median"] = np.nan

            city_records.append(city_record)

        except Exception as e:
            print(f"  Error processing {bounds_fid}: {e}")
            continue

    pooled_df = pd.DataFrame(pooled_records)
    city_df = pd.DataFrame(city_records)
    print(f"  Processed {len(pooled_df):,} raw values from {len(city_df)} cities")

    # Save cache
    pooled_df.to_parquet(cache_file)
    city_df.to_parquet(city_cache_file)
    print(f"  Saved cache to {cache_file}")

# %%
"""
## Step 3: Compute Aggregate Benchmark Statistics (Pooled Data)
"""

print("\nSTEP 3: Computing aggregate benchmark statistics from pooled data")

# Compute statistics across ALL raw values (not city medians)
benchmark_summary = []

for metric in pooled_df["metric"].unique():
    values = pooled_df[pooled_df["metric"] == metric]["value"].dropna()

    if len(values) > 0:
        benchmark_summary.append(
            {
                "metric": metric,
                "n_values": len(values),
                "min": values.min(),
                "p5": values.quantile(0.05),
                "p25": values.quantile(0.25),
                "p50": values.quantile(0.50),
                "p75": values.quantile(0.75),
                "p95": values.quantile(0.95),
                "max": values.max(),
                "mean": values.mean(),
                "std": values.std(),
            }
        )

benchmark_summary_df = pd.DataFrame(benchmark_summary)

print(f"\n  Computed benchmarks for {len(benchmark_summary_df)} metrics")
print(f"  Total data points: {pooled_df['value'].notna().sum():,}")

# %%
"""
## Step 4: Country-Level Aggregations
"""

print("\nSTEP 4: Computing country-level statistics")

# Get all median columns for country aggregation
median_cols = [col for col in city_df.columns if col.endswith("_median")]

# Filter to countries with enough cities
country_counts = city_df["country"].value_counts()
valid_countries = country_counts[country_counts >= MIN_CITIES_PER_COUNTRY].index.tolist()

country_df = city_df[city_df["country"].isin(valid_countries)].copy()
print(f"  Countries with >= {MIN_CITIES_PER_COUNTRY} cities: {len(valid_countries)}")

# Aggregate by country (using city medians for country-level comparisons)
country_stats = []

for country in valid_countries:
    country_data = country_df[country_df["country"] == country]
    n_cities = len(country_data)

    country_record = {
        "country": country,
        "n_cities": n_cities,
    }

    for col in median_cols:
        metric_name = col.replace("_median", "")
        values = country_data[col].dropna()

        if len(values) > 0:
            country_record[f"{metric_name}_median"] = values.median()
            country_record[f"{metric_name}_q1"] = values.quantile(0.25)
            country_record[f"{metric_name}_q3"] = values.quantile(0.75)

    country_stats.append(country_record)

country_stats_df = pd.DataFrame(country_stats)

# %%
"""
## Step 5: Save Outputs
"""

print("\nSTEP 5: Saving outputs")

# Full city benchmark data
city_output = output_path / "benchmark_reference.csv"
city_df.to_csv(city_output, index=False)
print(f"  Saved city benchmarks to {city_output}")

# Summary statistics
summary_output = output_path / "benchmark_summary.csv"
benchmark_summary_df.to_csv(summary_output, index=False)
print(f"  Saved summary statistics to {summary_output}")

# Country statistics
country_output = output_path / "country_rankings.csv"
country_stats_df.to_csv(country_output, index=False)
print(f"  Saved country statistics to {country_output}")

# %%
"""
## Step 6: Generate LaTeX Tables
"""

print("\nSTEP 6: Generating LaTeX tables")


def format_number(val, decimals=1):
    """Format number for LaTeX table."""
    if pd.isna(val):
        return "--"
    if abs(val) >= 1000:
        return f"{val:,.0f}"
    return f"{val:.{decimals}f}"


# Table 1: Full Benchmark Summary (all metrics)
latex_summary = "\\begin{table}[htbp]\n\\centering\n"
latex_summary += "\\caption{European Urban Metrics Benchmark Reference}\n"
latex_summary += "\\label{tab:benchmark_summary}\n"
latex_summary += "\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
latex_summary += "Metric & Min & P5 & P25 & P50 & P75 & P95 & Max \\\\\n\\midrule\n"

for _, row in benchmark_summary_df.iterrows():
    metric = row["metric"].replace("_", " ").title()
    latex_summary += (
        f"{metric} & {format_number(row['min'])} & {format_number(row['p5'])} & {format_number(row['p25'])} & "
    )
    latex_summary += f"{format_number(row['p50'])} & {format_number(row['p75'])} & {format_number(row['p95'])} & {format_number(row['max'])} \\\\\n"

latex_summary += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

with open(output_path / "table_benchmark_summary.tex", "w") as f:
    f.write(latex_summary)
print("  Saved benchmark summary table")

# Table 2: Land-use accessibility summary
landuse_metrics = [f"{cat}_nearest_m" for cat in LANDUSE_CATEGORIES]
landuse_subset = benchmark_summary_df[benchmark_summary_df["metric"].isin(landuse_metrics)].copy()

latex_landuse = "\\begin{table}[htbp]\n\\centering\n"
latex_landuse += "\\caption{Land-use Accessibility Benchmarks (Distance to Nearest POI)}\n"
latex_landuse += "\\label{tab:landuse_benchmarks}\n"
latex_landuse += "\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
latex_landuse += "Category & Min & P5 & P25 & P50 & P75 & P95 & Max \\\\\n\\midrule\n"

for _, row in landuse_subset.sort_values("p50").iterrows():
    cat_name = row["metric"].replace("_nearest_m", "").replace("_", " ").title()
    latex_landuse += f"{cat_name} & {format_number(row['min'], 0)} & {format_number(row['p5'], 0)} & {format_number(row['p25'], 0)} & "
    latex_landuse += f"{format_number(row['p50'], 0)} & {format_number(row['p75'], 0)} & {format_number(row['p95'], 0)} & {format_number(row['max'], 0)} \\\\\n"

latex_landuse += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

with open(output_path / "table_landuse_benchmarks.tex", "w") as f:
    f.write(latex_landuse)
print("  Saved land-use benchmarks table")

# %%
"""
## Step 7: Print Summary Report
"""

print("\n" + "=" * 80)
print("BENCHMARK SUMMARY REPORT")
print("=" * 80)

print(f"\nCities analyzed: {len(city_df)}")
print(f"Countries represented: {city_df['country'].nunique()}")
print(f"Total network nodes: {city_df['n_nodes'].sum():,}")

print("\n--- Benchmark Statistics (P5 / P25 / P50 / P75 / P95) ---")
for _, row in benchmark_summary_df.iterrows():
    metric = row["metric"].replace("_", " ").title()
    print(
        f"  {metric}: {format_number(row['p5'])} / {format_number(row['p25'])} / {format_number(row['p50'])} / {format_number(row['p75'])} / {format_number(row['p95'])}"
    )
"""
## Step 8: Generate README.md
"""

print("\nSTEP 8: Generating README.md")

readme_lines = [
    "# European Urban Metrics Benchmarking Reference",
    "",
    f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}",
    "",
    "## Vignette Purpose",
    "",
    "Provides standardised reference statistics (min, 5th, 25th, 50th, 75th, 95th percentiles, max) across",
    "all street network nodes in European cities, enabling comparative assessment against peer data.",
    "",
    "## Summary Statistics",
    "",
    f"- **Cities analyzed:** {len(city_df)}",
    f"- **Countries represented:** {city_df['country'].nunique()}",
    f"- **Total network nodes:** {city_df['n_nodes'].sum():,}",
    f"- **Total data points:** {len(pooled_df):,}",
    f"- **Distance threshold:** {DISTANCE}m",
    "",
    "## Metrics Categories",
    "",
    "| Category | Metrics | Distance |",
    "|----------|---------|----------|",
    "| **Land-use accessibility** | Distance to nearest POI per category | nearest |",
    f"| **Land-use diversity** | Hill q0 (richness), q1 (Shannon), q2 (Simpson) | {DISTANCE}m |",
    f"| **Network centrality** | Betweenness, density, cycles, harmonic closeness | {DISTANCE}m |",
    f"| **Green space** | Distance to green/trees, area coverage | {DISTANCE}m |",
    "| **Building morphology** | Height, area, compactness, volume, FAR, coverage | 200m |",
    "| **Census demographics** | Population density, employment, age structure | interpolated |",
    "",
    "## Benchmark Statistics",
    "",
    "| Metric | Min | P5 | P25 | P50 | P75 | P95 | Max |",
    "|--------|-----|-----|-----|-----|-----|-----|-----|",
]

for _, row in benchmark_summary_df.iterrows():
    metric = row["metric"].replace("_", " ").title()
    readme_lines.append(
        f"| {metric} | {format_number(row['min'])} | {format_number(row['p5'])} | {format_number(row['p25'])} | {format_number(row['p50'])} | {format_number(row['p75'])} | {format_number(row['p95'])} | {format_number(row['max'])} |"
    )

readme_lines.extend([])

readme_lines.extend(
    [
        "",
        "## Output Files",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `benchmark_reference.csv` | Full per-city statistics for all metrics |",
        "| `benchmark_summary.csv` | Cross-city aggregated statistics (P5, P25, P50, P75, P95) |",
        "| `country_rankings.csv` | Country-level metric aggregations |",
        "",
        "## Visualizations",
        "",
        "### Correlation Matrix",
        "",
        "![Correlation Matrix](outputs/correlation_matrix.png)",
        "",
        "### City Archetype Profiles",
        "",
        "![Radar Archetypes](outputs/radar_archetypes.png)",
        "",
        "### Geographic Distribution of Eat & Drink Access",
        "",
        "![Eat & Drink Access Map](outputs/map_eat_drink_access.png)",
        "",
        "## LaTeX Tables",
        "",
        "- `table_benchmark_summary.tex` - Full metrics summary (all percentiles)",
        "- `table_landuse_benchmarks.tex` - Land-use accessibility by category",
        "",
        "## Usage",
        "",
        "The benchmark reference enables:",
        "",
        "1. **City comparison** - Compare individual city metrics against European percentiles",
        "2. **Policy targets** - Set evidence-based targets using percentile thresholds",
        "3. **Gap analysis** - Identify metrics where a city falls below P25 or above P75",
        "4. **Country patterns** - Understand national-level urban form characteristics",
        "",
    ]
)

readme_path = output_path.parent / "README.md"
with open(readme_path, "w") as f:
    f.write("\n".join(readme_lines))
print(f"  Saved README to {readme_path}")

# %%
"""
## Step 9: Correlation Matrix
"""

print("\nSTEP 9: Generating correlation matrix")

# Select key metrics for correlation analysis
corr_metrics = [
    "green_nearest_m_median",
    "trees_nearest_m_median",
    "hill_q0_richness_median",
    "hill_q1_shannon_median",
    "betweenness_median",
    "network_density_median",
    "building_height_m_median",
    "building_area_m2_median",
    "block_coverage_ratio_median",
    f"building_count_{DISTANCE}m_median",
    f"block_count_{DISTANCE}m_median",
    "pop_density_per_km2_median",
    "employment_rate_median",
]

# Filter to available columns
corr_cols = [c for c in corr_metrics if c in city_df.columns]
corr_data = city_df[corr_cols].dropna()

# Compute correlation
corr_matrix = corr_data.corr()

# Clean labels for display
label_map = {
    "green_nearest_m_median": "Green Distance",
    "trees_nearest_m_median": "Trees Distance",
    "hill_q0_richness_median": "Diversity (q0)",
    "hill_q1_shannon_median": "Diversity (q1)",
    "betweenness_median": "Betweenness",
    "network_density_median": "Network Density",
    "building_height_m_median": "Building Height",
    "building_area_m2_median": "Building Area",
    "block_coverage_ratio_median": "Block Coverage",
    f"building_count_{DISTANCE}m_median": "Building Count",
    f"block_count_{DISTANCE}m_median": "Block Count",
    "pop_density_per_km2_median": "Pop. Density",
    "employment_rate_median": "Employment",
}
corr_labels = [label_map.get(c, c) for c in corr_matrix.columns]

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    square=True,
    linewidths=0.5,
    xticklabels=corr_labels,
    yticklabels=corr_labels,
    ax=ax,
    vmin=-1,
    vmax=1,
    cbar_kws={"shrink": 0.8, "label": "Correlation"},
)
ax.set_title("Urban Metrics Correlation Matrix", fontsize=12, fontweight="bold")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(output_path / "correlation_matrix.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved correlation_matrix.png")

# %%
"""
## Step 10: Radar Charts - City Archetypes
"""

print("\nSTEP 10: Generating radar charts for city archetypes")

# Select metrics for radar (normalize to 0-1 scale using percentile ranks)
radar_metrics = [
    "green_nearest_m_median",
    "hill_q1_shannon_median",
    "betweenness_median",
    "building_height_m_median",
    "block_coverage_ratio_median",
    "pop_density_per_km2_median",
]

radar_cols = [c for c in radar_metrics if c in city_df.columns]

# Compute percentile ranks (0-100) for each metric
radar_data = city_df[["city_label", "country"] + radar_cols].copy()
for col in radar_cols:
    # For distance metrics, invert so higher = better (closer)
    if "nearest" in col:
        radar_data[f"{col}_pct"] = 100 - radar_data[col].rank(pct=True) * 100
    else:
        radar_data[f"{col}_pct"] = radar_data[col].rank(pct=True) * 100

pct_cols = [f"{c}_pct" for c in radar_cols]

# Select archetype cities (check if they exist in data)
archetype_candidates = [
    ("Venezia", "IT", "Compact Historic"),
    ("Rotterdam", "NL", "Planned Modern"),
    ("Firenze", "IT", "Dense Mixed"),
    ("Birmingham", "UK", "Industrial"),
]

archetypes = []
for city, country, label in archetype_candidates:
    match = radar_data[
        (radar_data["city_label"].str.contains(city, case=False, na=False)) | (radar_data["city_label"] == city)
    ]
    if len(match) > 0:
        row = match.iloc[0]
        archetypes.append((row["city_label"], row["country"], label, row[pct_cols].values))

if len(archetypes) >= 2:
    # Radar chart setup
    radar_labels = [
        "Green Access",
        "Diversity",
        "Connectivity",
        "Building Height",
        "Block Coverage",
        "Pop. Density",
    ]
    num_vars = len(radar_labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, len(archetypes)))

    for i, (city_name, country, label, values) in enumerate(archetypes):
        values_plot = values.tolist() + [values[0]]  # Complete the loop
        ax.plot(angles, values_plot, "o-", linewidth=2, label=f"{city_name} ({label})", color=colors[i])
        ax.fill(angles, values_plot, alpha=0.15, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, size=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25th", "50th", "75th", "100th"], size=8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=9)
    ax.set_title("City Archetype Profiles\n(Percentile Ranks)", fontsize=12, fontweight="bold", y=1.08)

    plt.tight_layout()
    plt.savefig(output_path / "radar_archetypes.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved radar_archetypes.png with {len(archetypes)} cities")
else:
    print("  Insufficient archetype cities found, skipping radar chart")

# %%
"""
## Step 11: Geographic Map - Eat & Drink Access
"""

print("\nSTEP 11: Generating geographic map")

# Merge city benchmarks with geometry
city_geo = saturated_cities[["bounds_fid", "geometry"]].merge(
    city_df[["bounds_fid", "eat_and_drink_nearest_m_median", "city_label", "country"]],
    on="bounds_fid",
    how="inner",
)
city_geo = gpd.GeoDataFrame(city_geo, geometry="geometry", crs=saturated_cities.crs)

# Get centroids for plotting
city_geo["centroid"] = city_geo.geometry.centroid
city_points = city_geo.set_geometry("centroid")

if len(city_points) > 0:
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot points colored by eat_and_drink access (inverted - lower distance = better)
    vmin = city_points["eat_and_drink_nearest_m_median"].quantile(0.05)
    vmax = city_points["eat_and_drink_nearest_m_median"].quantile(0.95)

    scatter = ax.scatter(
        city_points.centroid.x,
        city_points.centroid.y,
        c=city_points["eat_and_drink_nearest_m_median"],
        cmap="RdYlGn_r",  # Red = far (bad), Green = close (good)
        s=40,
        alpha=0.7,
        edgecolors="white",
        linewidths=0.5,
        vmin=vmin,
        vmax=vmax,
    )

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Median Distance to Eat & Drink (m)", fontsize=10)

    ax.set_xlabel("Easting (EPSG:3035)", fontsize=10)
    ax.set_ylabel("Northing (EPSG:3035)", fontsize=10)
    ax.set_title("Eat & Drink Accessibility Across European Cities", fontsize=12, fontweight="bold")
    ax.set_aspect("equal")

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path / "map_eat_drink_access.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved map_eat_drink_access.png")
else:
    print("  No city geometries available for mapping")

print("\n" + "=" * 80)
print("Benchmark analysis complete!")
print("=" * 80)
