# %% [markdown];
"""
# Educational Infrastructure Gap Analysis

Analyzes educational facility accessibility across European cities. This
implementation deliberately restricts the sample to cities with
`Consistently Saturated` POI coverage (low spatial variability) from
Demonstrator 1 to ensure within-city equity metrics reflect real infrastructure
patterns rather than POI completeness artefacts.

## Steps
1. Load city saturation results from EG1 and filter to `Consistently Saturated`
2. Load education accessibility metrics from city metrics files
3. Compute cross-city summary statistics (mean access distances)
4. Compute within-city percentages below average and equity ratios
5. Generate two output tables and a short markdown report

## Key Outputs
- `cross_city_education_access.csv`: Mean education access by city
- `within_city_below_average.csv`: % of nodes below city average per city
- `README.md`: Summary report with key findings (notes the `Consistently Saturated` filter)

## Metrics Used (from SOAR pre-computed)
- `cc_education_nearest_max_1600`: Network distance to nearest education POI (m)
- `cc_education_1600_nw`: Count of education POIs within 1600m

## Notes
- Within-city metrics (pct below mean, equity ratio) are pre-computed during
    aggregation and cached to `temp/egs/eg3_education/education_city_data.parquet`.
    This avoids storing large per-node arrays in the cache while preserving the
    required summary statistics for reporting.
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

# %%
"""
## Configuration
"""

# Education access columns from SOAR metrics
EDUCATION_DIST_COL = "cc_education_nearest_max_1600"  # Distance to nearest education POI
EDUCATION_COUNT_COL = "cc_education_1600_nw"  # Count within 1600m

# Configuration - modify these paths as needed
BOUNDS_PATH = "temp/datasets/boundaries.gpkg"
METRICS_DIR = "temp/cities_data/processed"
SATURATION_RESULTS_PATH = "paper_research/code/eg1_poi_compare/outputs/city_analysis_results.gpkg"
OUTPUT_DIR = "paper_research/code/eg3_education/outputs"
TEMP_DIR = "temp/egs/eg3_education"

# Saturation quadrants to include (reliable POI data)
# Use only Consistently Saturated for within-city equity analysis
# to ensure spatial variability reflects real infrastructure gaps, not data gaps
SATURATED_QUADRANTS = ["Consistently Saturated"]

# %%
"""
## Step 1: Load Saturation Results and Filter to Saturated Cities
"""

output_path = Path(OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)
metrics_dir = Path(METRICS_DIR)
temp_path = Path(TEMP_DIR)
temp_path.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("EDUCATIONAL INFRASTRUCTURE GAP ANALYSIS")
print("=" * 80)

print("\nSTEP 1: Loading saturation results and filtering cities")

# Load saturation results from EG1
saturation_gdf = gpd.read_file(SATURATION_RESULTS_PATH)
print(f"  Loaded saturation results for {len(saturation_gdf)} cities")

# Filter to cities with saturated education POI coverage
saturated_cities = saturation_gdf[saturation_gdf["education_quadrant"].isin(SATURATED_QUADRANTS)].copy()
print(f"  Cities with saturated education data: {len(saturated_cities)}")
print(f"    - Consistently Saturated: {(saturated_cities['education_quadrant'] == 'Consistently Saturated').sum()}")
print(f"    - Variable Saturated: {(saturated_cities['education_quadrant'] == 'Variable Saturated').sum()}")

# Get list of saturated bounds_fids
saturated_fids = set(saturated_cities["bounds_fid"].tolist())

# %%
"""
## Step 2: Load Education Metrics for Saturated Cities
"""

print("\nSTEP 2: Loading education accessibility metrics")

# Check for cached data
education_cache_file = temp_path / "education_city_data.parquet"

if education_cache_file.exists():
    print("  Loading cached education metrics...")
    cached_df = pd.read_parquet(education_cache_file)
    # Reconstruct city_data from cached DataFrame
    city_data = cached_df.to_dict("records")
    print(f"  Loaded cached data for {len(city_data)} cities")
else:
    city_data = []
    for idx, row in tqdm(saturated_cities.iterrows(), total=len(saturated_cities), desc="Loading cities"):
        bounds_fid = row["bounds_fid"]
        city_label = row.get("label", str(bounds_fid))
        country = row.get("country", "Unknown")
        # Load metrics file for this city
        metrics_file = metrics_dir / f"metrics_{bounds_fid}.gpkg"
        if not metrics_file.exists():
            print(f"  WARNING: Metrics file not found for bounds_fid {bounds_fid} at {metrics_file}")
            continue
        gdf = gpd.read_file(metrics_file, columns=[EDUCATION_DIST_COL, EDUCATION_COUNT_COL], layer="streets")
        # Doublecheck geoms are dropped if outside boundary
        gdf = gdf[gdf.geometry.within(row.geometry)]
        # Filter out invalid values
        valid_mask = (
            gdf[EDUCATION_DIST_COL].notna() & (gdf[EDUCATION_DIST_COL] < float("inf")) & (gdf[EDUCATION_DIST_COL] >= 0)
        )
        gdf = gdf[valid_mask]

        if len(gdf) < 100:  # Minimum nodes for reliable stats
            continue

        distances = gdf[EDUCATION_DIST_COL].values
        mean_dist = gdf[EDUCATION_DIST_COL].mean()
        p25_dist = gdf[EDUCATION_DIST_COL].quantile(0.25)
        p75_dist = gdf[EDUCATION_DIST_COL].quantile(0.75)

        city_data.append(
            {
                "bounds_fid": bounds_fid,
                "city_label": city_label,
                "country": country,
                "n_nodes": len(gdf),
                "mean_dist": mean_dist,
                "median_dist": gdf[EDUCATION_DIST_COL].median(),
                "std_dist": gdf[EDUCATION_DIST_COL].std(),
                "p25_dist": p25_dist,
                "p75_dist": p75_dist,
                "pct_within_400m": (distances <= 400).sum() / len(distances) * 100,
                "pct_within_800m": (distances <= 800).sum() / len(distances) * 100,
                "pct_within_1600m": (distances <= 1600).sum() / len(distances) * 100,
                # Pre-compute within-city metrics (so we don't need raw distances in cache)
                "pct_below_city_mean": (distances < mean_dist).sum() / len(distances) * 100,
                "pct_above_2x_city_mean": (distances > 2 * mean_dist).sum() / len(distances) * 100,
                "equity_ratio": p75_dist / p25_dist if p25_dist > 0 else float("nan"),
            }
        )

    print(f"  Loaded data for {len(city_data)} cities with valid education metrics")

    # Save to cache
    cache_df = pd.DataFrame(city_data)
    cache_df.to_parquet(education_cache_file)
    print(f"  Saved cache to {education_cache_file}")

# %%
"""
## Step 3: Compute Cross-City Summary Statistics
"""

print("\nSTEP 3: Computing cross-city education access statistics")

# Create DataFrame from city_data
city_df = pd.DataFrame(city_data)

# Sort by mean distance (best access first)
city_df = city_df.sort_values("mean_dist")

# Compute overall statistics
overall_mean = city_df["mean_dist"].mean()
overall_median = city_df["median_dist"].mean()

print("\n  Cross-city summary:")
print(f"    Mean distance to education: {overall_mean:.1f}m")
print(f"    Median distance to education: {overall_median:.1f}m")
print(f"    Best city: {city_df.iloc[0]['city_label']} ({city_df.iloc[0]['mean_dist']:.1f}m)")
print(f"    Worst city: {city_df.iloc[-1]['city_label']} ({city_df.iloc[-1]['mean_dist']:.1f}m)")

# %%
"""
## Step 4: Compute Within-City Percentages Below Average
"""

print("\nSTEP 4: Computing within-city below-average percentages")

# Within-city metrics are pre-computed in Step 2
# Create within_city_df from city_df with relevant columns
within_city_df = city_df[
    [
        "bounds_fid",
        "city_label",
        "country",
        "n_nodes",
        "mean_dist",
        "pct_below_city_mean",
        "pct_above_2x_city_mean",
        "equity_ratio",
    ]
].copy()
within_city_df = within_city_df.rename(columns={"mean_dist": "city_mean_dist"})

# Compute % below EU average for each city
# This requires knowing overall_mean from Step 3, so we calculate it here
# For cached data, we approximate using city_mean_dist relative to overall_mean
within_city_df["pct_below_eu_avg"] = within_city_df.apply(
    lambda row: row["pct_below_city_mean"] * (overall_mean / row["city_mean_dist"]) if row["city_mean_dist"] > 0 else 0,
    axis=1,
)
# Cap at 100%
within_city_df["pct_below_eu_avg"] = within_city_df["pct_below_eu_avg"].clip(upper=100)

# Sort by equity ratio (higher = more inequitable)
within_city_df = within_city_df.sort_values("equity_ratio", ascending=False)

print("\n  Within-city summary:")
print(f"    Mean pct below city average: {within_city_df['pct_below_city_mean'].mean():.1f}%")
print(f"    Mean pct below EU average: {within_city_df['pct_below_eu_avg'].mean():.1f}%")
print(f"    Mean pct severely underserved: {within_city_df['pct_above_2x_city_mean'].mean():.1f}%")

# %%
"""
## Step 5: Generate Output Tables
"""

print("\nSTEP 5: Generating output tables")

# Table 1: Cross-city education access
cross_city_table = city_df[
    [
        "city_label",
        "country",
        "n_nodes",
        "mean_dist",
        "median_dist",
        "std_dist",
        "pct_within_400m",
        "pct_within_800m",
    ]
].copy()
cross_city_table.columns = [
    "City",
    "Country",
    "Nodes",
    "Mean Dist (m)",
    "Median Dist (m)",
    "Std Dev (m)",
    "% within 400m",
    "% within 800m",
]

# Save cross-city table
cross_city_path = output_path / "cross_city_education_access.csv"
cross_city_table.to_csv(cross_city_path, index=False, float_format="%.1f")
print(f"  Saved cross-city table to {cross_city_path}")

# Table 2: Within-city below-average percentages
within_city_table = within_city_df[
    [
        "city_label",
        "country",
        "n_nodes",
        "city_mean_dist",
        "pct_below_city_mean",
        "pct_below_eu_avg",
        "pct_above_2x_city_mean",
        "equity_ratio",
    ]
].copy()
within_city_table.columns = [
    "City",
    "Country",
    "Nodes",
    "City Mean (m)",
    "% Below City Mean",
    "% Below EU Avg",
    "% Severely Underserved",
    "Equity Ratio (P75/P25)",
]

# Save within-city table
within_city_path = output_path / "within_city_below_average.csv"
within_city_table.to_csv(within_city_path, index=False, float_format="%.1f")
print(f"  Saved within-city table to {within_city_path}")

# %%
"""
## Step 6: Generate Summary Report
"""

print("\nSTEP 6: Generating summary report")

# Top/bottom cities by mean access
top_10 = city_df.head(10)
bottom_10 = city_df.tail(10)

# Most/least equitable cities
most_equitable = within_city_df.nsmallest(10, "equity_ratio")
least_equitable = within_city_df.nlargest(10, "equity_ratio")

report_lines = [
    "# Educational Infrastructure Gap Analysis Report",
    "",
    f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}",
    "",
    "## Overview",
    "",
    "This analysis examines educational facility accessibility across European cities,",
    "restricted to cities with reliable (saturated) POI coverage from Demonstrator 1.",
    "",
    "## Summary Statistics",
    "",
    f"- **Cities Analyzed:** {len(city_df)} (saturated education POI coverage only)",
    f"- **Total Street Network Nodes:** {city_df['n_nodes'].sum():,}",
    f"- **Mean Distance to Education (cross-city):** {overall_mean:.1f}m",
    f"- **Median Distance to Education (cross-city):** {overall_median:.1f}m",
    "",
    "## Cross-City Comparison",
    "",
    "### Best Access (Top 10 Cities)",
    "",
    "| City | Country | Mean Dist (m) | % within 400m |",
    "|------|---------|---------------|---------------|",
]
for _, row in top_10.iterrows():
    report_lines.append(
        f"| {row['city_label']} | {row['country']} | {row['mean_dist']:.1f} | {row['pct_within_400m']:.1f}% |"
    )

report_lines.extend(
    [
        "",
        "### Worst Access (Bottom 10 Cities)",
        "",
        "| City | Country | Mean Dist (m) | % within 400m |",
        "|------|---------|---------------|---------------|",
    ]
)
for _, row in bottom_10.iterrows():
    report_lines.append(
        f"| {row['city_label']} | {row['country']} | {row['mean_dist']:.1f} | {row['pct_within_400m']:.1f}% |"
    )

report_lines.extend(
    [
        "",
        "## Within-City Equity Analysis",
        "",
        "The equity ratio (P75/P25) measures inequality of access within each city.",
        "Higher values indicate greater disparity between well-served and underserved areas.",
        "",
        "### Most Equitable Cities (Lowest P75/P25 Ratio)",
        "",
        "| City | Country | Equity Ratio | % Severely Underserved |",
        "|------|---------|--------------|------------------------|",
    ]
)
for _, row in most_equitable.iterrows():
    report_lines.append(
        f"| {row['city_label']} | {row['country']} | {row['equity_ratio']:.2f} | {row['pct_above_2x_city_mean']:.1f}% |"
    )

report_lines.extend(
    [
        "",
        "### Least Equitable Cities (Highest P75/P25 Ratio)",
        "",
        "| City | Country | Equity Ratio | % Severely Underserved |",
        "|------|---------|--------------|------------------------|",
    ]
)
for _, row in least_equitable.iterrows():
    report_lines.append(
        f"| {row['city_label']} | {row['country']} | {row['equity_ratio']:.2f} | {row['pct_above_2x_city_mean']:.1f}% |"
    )

report_lines.extend(
    [
        "",
        "## Key Findings",
        "",
        "1. **Cross-city variation**: Mean distance to education varies substantially across",
        f"   European cities, from {city_df['mean_dist'].min():.1f}m to {city_df['mean_dist'].max():.1f}m.",
        "",
        "2. **Within-city inequality**: Even in cities with good average access, significant",
        "   portions of the population may be underserved. The equity ratio captures this disparity.",
        "",
        "3. **Data quality matters**: By filtering to saturated cities only, these results",
        "   reflect actual infrastructure gaps rather than POI data incompleteness.",
        "",
        "## Output Files",
        "",
        "- `cross_city_education_access.csv`: Full cross-city comparison table",
        "- `within_city_below_average.csv`: Within-city equity metrics",
        "",
    ]
)

# Write report
report_path = output_path.parent / "README.md"
with open(report_path, "w") as f:
    f.write("\n".join(report_lines))
print(f"  Saved report to {report_path}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"Output directory: {output_path}")
print("  - cross_city_education_access.csv")
print("  - within_city_below_average.csv")
print(f"README.md saved to: {report_path}")

# %%
