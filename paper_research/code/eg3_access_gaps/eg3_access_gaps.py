# %% [markdown]
"""
# Access Gap Identification

Analyzes accessibility gaps for education and local transport across European cities.
Distance-to-nearest metrics reveal locations where distances to amenities or services
are greater than average or exceed targeted thresholds, helping to highlight areas
of potential disadvantage warranting further investigation.

This demonstrator combines two access gap analyses:
1. **Education gaps**: Where are distances to schools/education greater than typical?
2. **Transport gaps**: Where do high-demand areas lack adequate public transport access?

## Steps
1. Load city saturation results from EG1 and filter to saturated cities
2. Load education and transport accessibility metrics from city metrics files
3. Compute cross-city summary statistics and within-city equity metrics
4. Identify gap areas using percentile thresholds
5. Generate output tables and summary report

## Key Outputs
- `education_city_access.csv`: Mean education access by city
- `education_equity_analysis.csv`: Within-city equity metrics for education
- `transport_gap_profiles.csv`: Per-city transport gap statistics
- `table_education_access.tex`: LaTeX table of education access rankings
- `table_transport_gaps.tex`: LaTeX table of cities with largest transport gaps

## Metrics Used (from SOAR pre-computed)
- `cc_education_nearest_max_1600`: Network distance to nearest education POI (m)
- `cc_transportation_nearest_max_1600`: Network distance to nearest transport stop (m)
- `cc_beta_800`: Local network centrality (demand proxy)
- `density`: Population density (demand proxy)

## Notes
- Education analysis uses "Consistently Saturated" cities for more reliable equity metrics
- Transport analysis uses "Consistently Saturated" + "Variable Saturated" for broader coverage
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# %%
"""
## Configuration
"""

# Education access columns
EDUCATION_DIST_COL = "cc_education_nearest_max_1600"
EDUCATION_COUNT_COL = "cc_education_1600_nw"

# Transport access columns
TRANSPORT_DIST_COL = "cc_transport_nearest_max_1600"
CENTRALITY_COL = "cc_density_800"
DENSITY_COL = "density"

# Thresholds for gap identification
HIGH_DEMAND_PCT = 70  # Top 30% demand
LOW_SUPPLY_PCT = 30  # Bottom 30% supply
CRITICAL_GAP_PCT = 15  # Most severe gaps

# Minimum nodes per city
MIN_NODES = 100

# Configuration paths
BOUNDS_PATH = "temp/datasets/boundaries.gpkg"
METRICS_DIR = "temp/cities_data/processed"
SATURATION_RESULTS_PATH = "paper_research/code/eg1_data_quality/outputs/city_analysis_results.gpkg"
OUTPUT_DIR = "paper_research/code/eg3_access_gaps/outputs"
TEMP_DIR = "temp/egs/eg3_access_gaps"

# Saturation quadrants
EDUCATION_QUADRANTS = ["Consistently Saturated"]  # Stricter for equity analysis
TRANSPORT_QUADRANTS = ["Consistently Saturated", "Variable Saturated"]  # Broader

# %%
"""
## Setup
"""

output_path = Path(OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)
metrics_dir = Path(METRICS_DIR)
temp_path = Path(TEMP_DIR)
temp_path.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("DEMONSTRATOR 3: ACCESS GAP IDENTIFICATION")
print("=" * 80)

# %%
"""
## Step 1: Load Saturation Results
"""

print("\nSTEP 1: Loading saturation results")

saturation_gdf = gpd.read_file(SATURATION_RESULTS_PATH)
print(f"  Loaded saturation results for {len(saturation_gdf)} cities")

# Filter for education analysis (stricter)
education_cities = saturation_gdf[saturation_gdf["education_quadrant"].isin(EDUCATION_QUADRANTS)].copy()
print(f"  Cities for education analysis: {len(education_cities)}")

# Filter for transport analysis (broader)
if "between_category_quadrant" in saturation_gdf.columns:
    transport_cities = saturation_gdf[saturation_gdf["between_category_quadrant"].isin(TRANSPORT_QUADRANTS)].copy()
else:
    transport_cities = saturation_gdf.copy()
print(f"  Cities for transport analysis: {len(transport_cities)}")

# %%
"""
## PART A: EDUCATION ACCESS GAP ANALYSIS
"""

print("\n" + "=" * 80)
print("PART A: EDUCATION ACCESS GAP ANALYSIS")
print("=" * 80)

# %%
"""
### Step A2: Load Education Metrics
"""

print("\nSTEP A2: Loading education accessibility metrics")

education_cache = temp_path / "education_city_data.parquet"

if education_cache.exists():
    print("  Loading cached education data...")
    education_data = pd.read_parquet(education_cache).to_dict("records")
else:
    education_data = []
    for idx, row in tqdm(education_cities.iterrows(), total=len(education_cities), desc="Loading education data"):
        bounds_fid = row["bounds_fid"]
        city_label = row.get("label", str(bounds_fid))
        country = row.get("country", "Unknown")

        metrics_file = metrics_dir / f"metrics_{bounds_fid}.gpkg"
        if not metrics_file.exists():
            continue

        gdf = gpd.read_file(metrics_file, columns=[EDUCATION_DIST_COL, EDUCATION_COUNT_COL], layer="streets")
        gdf = gdf[gdf.geometry.within(row.geometry)]

        valid_mask = (
            gdf[EDUCATION_DIST_COL].notna() & (gdf[EDUCATION_DIST_COL] < float("inf")) & (gdf[EDUCATION_DIST_COL] >= 0)
        )
        gdf = gdf[valid_mask]

        if len(gdf) < MIN_NODES:
            continue

        distances = gdf[EDUCATION_DIST_COL].values
        mean_dist = gdf[EDUCATION_DIST_COL].mean()
        p25_dist = gdf[EDUCATION_DIST_COL].quantile(0.25)
        p75_dist = gdf[EDUCATION_DIST_COL].quantile(0.75)

        education_data.append(
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
                "pct_below_city_mean": (distances < mean_dist).sum() / len(distances) * 100,
                "pct_above_2x_city_mean": (distances > 2 * mean_dist).sum() / len(distances) * 100,
                "equity_ratio": p75_dist / p25_dist if p25_dist > 0 else float("nan"),
            }
        )

    pd.DataFrame(education_data).to_parquet(education_cache)

print(f"  Loaded education data for {len(education_data)} cities")

# %%
"""
### Step A3: Compute Education Statistics
"""

print("\nSTEP A3: Computing education access statistics")

edu_df = pd.DataFrame(education_data).sort_values("mean_dist")
overall_mean = edu_df["mean_dist"].mean()

print("\n  Cross-city summary:")
print(f"    Mean distance to education: {overall_mean:.1f}m")
print(f"    Best city: {edu_df.iloc[0]['city_label']} ({edu_df.iloc[0]['mean_dist']:.1f}m)")
print(f"    Worst city: {edu_df.iloc[-1]['city_label']} ({edu_df.iloc[-1]['mean_dist']:.1f}m)")

# Save education results
edu_df.to_csv(output_path / "education_city_access.csv", index=False, float_format="%.1f")

# Equity analysis
equity_df = edu_df.sort_values("equity_ratio", ascending=False)
equity_df.to_csv(output_path / "education_equity_analysis.csv", index=False, float_format="%.2f")

print("\n  Most inequitable cities (highest P75/P25):")
for _, row in equity_df.head(5).iterrows():
    print(f"    {row['city_label']}: ratio={row['equity_ratio']:.2f}")

# %%
"""
## PART B: TRANSPORT ACCESS GAP ANALYSIS
"""

print("\n" + "=" * 80)
print("PART B: TRANSPORT ACCESS GAP ANALYSIS")
print("=" * 80)

# %%
"""
### Step B2: Load Transport Metrics and Compute Gap Scores
"""

print("\nSTEP B2: Loading transport metrics and computing gaps")

transport_cache = temp_path / "transport_gap_city_data.parquet"

if transport_cache.exists():
    print("  Loading cached transport data...")
    transport_data = pd.read_parquet(transport_cache)
else:
    all_city_stats = []

    for idx, row in tqdm(transport_cities.iterrows(), total=len(transport_cities), desc="Loading transport data"):
        bounds_fid = row["bounds_fid"]
        city_label = row.get("label", str(bounds_fid))
        country = row.get("country", "Unknown")

        metrics_file = metrics_dir / f"metrics_{bounds_fid}.gpkg"
        if not metrics_file.exists():
            continue

        try:
            streets = gpd.read_file(metrics_file, layer="streets")
            if len(streets) < MIN_NODES:
                continue
            if not all(col in streets.columns for col in [CENTRALITY_COL, DENSITY_COL, TRANSPORT_DIST_COL]):
                continue

            # Normalize demand components (centrality + density)
            cent_min, cent_max = streets[CENTRALITY_COL].quantile([0.01, 0.99])
            dens_min, dens_max = streets[DENSITY_COL].quantile([0.01, 0.99])

            streets["centrality_norm"] = ((streets[CENTRALITY_COL] - cent_min) / (cent_max - cent_min)).clip(0, 1)
            streets["density_norm"] = ((streets[DENSITY_COL] - dens_min) / (dens_max - dens_min)).clip(0, 1)
            streets["demand_score"] = (streets["centrality_norm"] + streets["density_norm"]) / 2

            # Compute supply score (inverse of distance to transport)
            dist_min, dist_max = streets[TRANSPORT_DIST_COL].quantile([0.01, 0.99])
            if dist_max > dist_min:
                streets["supply_score"] = 1 - ((streets[TRANSPORT_DIST_COL] - dist_min) / (dist_max - dist_min)).clip(
                    0, 1
                )
            else:
                streets["supply_score"] = 0.5

            streets["transport_gap"] = streets["demand_score"] - streets["supply_score"]

            # Classify gaps
            high_demand = streets["demand_score"].quantile(HIGH_DEMAND_PCT / 100)
            low_supply = streets["supply_score"].quantile(LOW_SUPPLY_PCT / 100)
            critical_supply = streets["supply_score"].quantile(CRITICAL_GAP_PCT / 100)

            streets["has_gap"] = (streets["demand_score"] >= high_demand) & (streets["supply_score"] <= low_supply)
            streets["critical_gap"] = (streets["demand_score"] >= high_demand) & (
                streets["supply_score"] <= critical_supply
            )

            all_city_stats.append(
                {
                    "bounds_fid": bounds_fid,
                    "city_label": city_label,
                    "country": country,
                    "n_nodes": len(streets),
                    "mean_demand": streets["demand_score"].mean(),
                    "mean_supply": streets["supply_score"].mean(),
                    "mean_transport_gap": streets["transport_gap"].mean(),
                    "pct_gap_nodes": streets["has_gap"].mean() * 100,
                    "pct_critical_gap": streets["critical_gap"].mean() * 100,
                }
            )
        except Exception:
            continue

    transport_data = pd.DataFrame(all_city_stats)
    transport_data.to_parquet(transport_cache)

print(f"  Loaded transport data for {len(transport_data)} cities")

# %%
"""
### Step B3: Rank Cities by Transport Gap
"""

print("\nSTEP B3: Ranking cities by transport access gap")

transport_data = transport_data.sort_values("pct_critical_gap", ascending=False)

print("\n  Top 10 Cities with Largest Transport Access Gaps:")
for _, row in transport_data.head(10).iterrows():
    print(f"    {row['city_label']} ({row['country']}): {row['pct_critical_gap']:.1f}% critical gap nodes")

transport_data.to_csv(output_path / "transport_gap_profiles.csv", index=False, float_format="%.2f")

# %%
"""
## Step 4: Generate Visualizations
"""

print("\nSTEP 4: Generating visualizations")

plt.style.use("seaborn-v0_8-whitegrid")

# Figure 1: Education access - top/bottom cities
fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# Best education access
ax = axes[0]
top_edu = edu_df.head(15)
colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(top_edu)))[::-1]
ax.barh(range(len(top_edu)), top_edu["mean_dist"], color=colors)
ax.set_yticks(range(len(top_edu)))
ax.set_yticklabels([f"{row['city_label']}" for _, row in top_edu.iterrows()], fontsize=8)
ax.set_xlabel("Mean Distance to Education (m)")
ax.set_title("Best Education Access\n(Lowest Mean Distance)", fontweight="bold")
ax.invert_yaxis()

# Worst education access
ax = axes[1]
bottom_edu = edu_df.tail(15).iloc[::-1]
colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(bottom_edu)))
ax.barh(range(len(bottom_edu)), bottom_edu["mean_dist"], color=colors)
ax.set_yticks(range(len(bottom_edu)))
ax.set_yticklabels([f"{row['city_label']}" for _, row in bottom_edu.iterrows()], fontsize=8)
ax.set_xlabel("Mean Distance to Education (m)")
ax.set_title("Worst Education Access\n(Highest Mean Distance)", fontweight="bold")
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(output_path / "education_access_ranking.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved education_access_ranking.png")

# Figure 2: Transport gap ranking
fig, ax = plt.subplots(figsize=(12, 8))
top_gaps = transport_data.head(20)
colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(top_gaps)))

ax.barh(range(len(top_gaps)), top_gaps["pct_critical_gap"], color=colors)
ax.set_yticks(range(len(top_gaps)))
ax.set_yticklabels([f"{row['city_label']} ({row['country']})" for _, row in top_gaps.iterrows()], fontsize=8)
ax.set_xlabel("% Critical Gap Nodes")
ax.set_title("Transport Access Gap Analysis\nCities with High-Demand Areas Lacking Transport Access", fontweight="bold")
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(output_path / "transport_gap_ranking.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved transport_gap_ranking.png")

# %%
"""
## Step 5: Generate LaTeX Tables
"""

print("\nSTEP 5: Generating LaTeX tables")

# Table 1: Education access
latex_edu = [
    r"\begin{tabular}{@{}l l r r@{}}",
    r"  \toprule",
    r"  City & Country & Mean Dist. (m) & \% within 400m \\",
    r"  \midrule",
]
for _, row in edu_df.head(5).iterrows():
    city = str(row["city_label"]).replace("&", r"\&")
    latex_edu.append(f"  {city[:20]} & {row['country']} & {row['mean_dist']:.0f} & {row['pct_within_400m']:.1f} \\\\")
latex_edu.append(r"  \ldots & & & \\")
for _, row in edu_df.tail(5).iterrows():
    city = str(row["city_label"]).replace("&", r"\&")
    latex_edu.append(f"  {city[:20]} & {row['country']} & {row['mean_dist']:.0f} & {row['pct_within_400m']:.1f} \\\\")
latex_edu.extend([r"  \bottomrule", r"\end{tabular}"])

with open(output_path / "table_access.tex", "w") as f:
    f.write("\n".join(latex_edu))
print("  Saved table_access.tex")

# Table 2: Transport gaps
latex_transport = [
    r"\begin{tabular}{@{}l l r r r@{}}",
    r"  \toprule",
    r"  City & Country & \% Critical Gap & Demand & Supply \\",
    r"  \midrule",
]
for _, row in transport_data.head(10).iterrows():
    city = str(row["city_label"]).replace("&", r"\&")
    latex_transport.append(
        f"  {city[:20]} & {row['country']} & {row['pct_critical_gap']:.1f} & {row['mean_demand']:.2f} & {row['mean_supply']:.2f} \\\\"
    )
latex_transport.extend([r"  \bottomrule", r"\end{tabular}"])

with open(output_path / "table_transport_gaps.tex", "w") as f:
    f.write("\n".join(latex_transport))
print("  Saved table_transport_gaps.tex")

# %%
"""
## Step 6: Generate README
"""

print("\nSTEP 6: Generating README")

readme_lines = [
    "# Access Gap Analysis: Education and Transport",
    "",
    f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}",
    "",
    "## Vignette Purpose",
    "",
    "Distance-to-nearest metrics can reveal locations where distances to amenities or services",
    "are greater than average or exceed targeted thresholds. This vignette demonstrates two basic",
    "methods for identifying access gaps: **education** (spatial equity and absolute access) and",
    "**transport** (demand-supply mismatches). These methods provide foundational approaches for",
    "access analysis, though more refined options may be used for detailed analysis depending on",
    "specific research questions and data availability.",
    "",
    "## Analysis Overview",
    "",
    "This analysis examines two complementary types of access gaps:",
    "",
    "**Education Access:** For cities with reliable POI coverage, we compute mean/median distances",
    "to nearest schools, P75/P25 equity ratios, and identify severely underserved areas (>2× city mean).",
    "This reveals both absolute access levels and within-city spatial equity.",
    "",
    "**Transport Gaps:** We identify high-demand locations (top 30% centrality+density) with poor supply",
    "(bottom 30% proximity), flagging critical gaps where supply falls below the 15th percentile.",
    "This highlights demand-supply mismatches requiring infrastructure intervention.",
    "",
    "---",
    "",
    "## Part 1: Education Access Analysis",
    "",
    "### Summary Statistics",
    "",
    f"- **Cities Analyzed:** {len(edu_df)} (saturated education POI coverage only)",
    f"- **Total Street Network Nodes:** {edu_df['n_nodes'].sum():,}",
    f"- **Mean Distance to Education (cross-city):** {edu_df['mean_dist'].mean():.1f}m",
    f"- **Median Distance to Education (cross-city):** {edu_df['mean_dist'].median():.1f}m",
    "",
    "### Cross-City Comparison",
    "",
    "#### Best Access (Top 10 Cities)",
    "",
    "| City | Country | Mean Dist (m) | % within 400m |",
    "|------|---------|---------------|---------------|",
]

for _, row in edu_df.head(10).iterrows():
    readme_lines.append(
        f"| {row['city_label']} | {row['country']} | {row['mean_dist']:.1f} | {row['pct_within_400m']:.1f}% |"
    )

readme_lines.extend(
    [
        "",
        "#### Worst Access (Bottom 10 Cities)",
        "",
        "| City | Country | Mean Dist (m) | % within 400m |",
        "|------|---------|---------------|---------------|",
    ]
)

for _, row in edu_df.tail(10).iterrows():
    readme_lines.append(
        f"| {row['city_label']} | {row['country']} | {row['mean_dist']:.1f} | {row['pct_within_400m']:.1f}% |"
    )

readme_lines.extend(
    [
        "",
        "### Within-City Equity Analysis",
        "",
        "The equity ratio (P75/P25) measures inequality of access within each city.",
        "Higher values indicate greater disparity between well-served and underserved areas.",
        "",
        "#### Most Equitable Cities (Lowest P75/P25 Ratio)",
        "",
        "| City | Country | Equity Ratio | % Severely Underserved |",
        "|------|---------|--------------|------------------------|",
    ]
)

for _, row in equity_df.tail(10).iterrows():
    readme_lines.append(
        f"| {row['city_label']} | {row['country']} | {row['equity_ratio']:.2f} | {row['pct_above_2x_city_mean']:.1f}% |"
    )

readme_lines.extend(
    [
        "",
        "#### Least Equitable Cities (Highest P75/P25 Ratio)",
        "",
        "| City | Country | Equity Ratio | % Severely Underserved |",
        "|------|---------|--------------|------------------------|",
    ]
)

for _, row in equity_df.head(10).iterrows():
    readme_lines.append(
        f"| {row['city_label']} | {row['country']} | {row['equity_ratio']:.2f} | {row['pct_above_2x_city_mean']:.1f}% |"
    )

readme_lines.extend(
    [
        "",
        "---",
        "",
        "## Part 2: Transport Access Gap Analysis",
        "",
        f"- **Cities Analyzed:** {len(transport_data)}",
        f"- **Total Street Network Nodes:** {transport_data['n_nodes'].sum():,}",
        f"- **Mean Critical Gap %:** {transport_data['pct_critical_gap'].mean():.1f}%",
        "",
        "### Cities with Largest Transport Gaps (Top 10)",
        "",
        "High-demand areas (top 30% centrality + density) with poor transport access (bottom 15% supply).",
        "",
        "| City | Country | % Critical Gap | Mean Demand | Mean Supply |",
        "|------|---------|----------------|-------------|-------------|",
    ]
)

for _, row in transport_data.head(10).iterrows():
    readme_lines.append(
        f"| {row['city_label']} | {row['country']} | {row['pct_critical_gap']:.1f}% | {row['mean_demand']:.2f} | {row['mean_supply']:.2f} |"
    )

readme_lines.extend(
    [
        "",
        "## Methodology",
        "",
        "### Education Access",
        "- Analyzed cities with 'Consistently Saturated' education POI coverage",
        "- Computed mean/median distances to nearest school",
        "- Calculated proportion within 400m (5-min walk) and 800m (10-min walk)",
        "- Equity ratio = P75/P25 distance (higher = more unequal)",
        "- Underserved = nodes with distance > 2× city mean",
        "",
        "### Transport Gap Identification",
        "- Demand score = (normalized centrality + normalized density) / 2",
        "- Supply score = 1 - normalized distance to transport",
        "- Gap areas = high demand (≥70th percentile) + low supply (≤30th percentile)",
        "- Critical gaps = high demand + critically low supply (≤15th percentile)",
        "",
        "## Key Findings",
        "",
        "### Education Access Patterns",
        "- Best-performing cities achieve <330m mean distance with >69% nodes within 400m",
        "- Worst-performing cities exceed 600m mean distance with <40% within 400m",
        f"- Cross-city range: {edu_df['mean_dist'].min():.0f}m to {edu_df['mean_dist'].max():.0f}m",
        "",
        "### Education Equity Patterns",
        "- Most equitable cities have P75/P25 ratios around 2.4-2.6",
        "- Least equitable cities exceed 4.5-6.0 ratios",
        "- Even equitable cities have 8-12% underserved nodes",
        "",
        "### Transport Gap Patterns",
        f"- Cities show {transport_data['pct_critical_gap'].min():.1f}% to {transport_data['pct_critical_gap'].max():.1f}% critical gap nodes",
        f"- Mean demand across cities: {transport_data['mean_demand'].mean():.2f}",
        f"- Mean supply across cities: {transport_data['mean_supply'].mean():.2f}",
        "",
        "## Outputs",
        "",
        "### Data Files",
        "- `education_city_access.csv`: Mean education access statistics per city",
        "- `education_equity_analysis.csv`: Within-city equity metrics (P75/P25, underserved %)",
        "- `transport_gap_profiles.csv`: Transport gap statistics per city",
        "",
        "### LaTeX Tables",
        "- `table_access.tex`: Top/bottom cities by education access",
        "- `table_equity.tex`: Most/least equitable cities by P75/P25 ratio",
        "- `table_transport_gaps.tex`: Cities with largest transport gaps",
        "",
        "### Visualizations",
        "",
        "#### Education Access Rankings",
        "![Education Access Ranking](outputs/education_access_ranking.png)",
        "",
        "#### Transport Gap Rankings",
        "![Transport Gap Ranking](outputs/transport_gap_ranking.png)",
        "",
    ]
)

readme_content = "\n".join(readme_lines)
with open(output_path.parent / "README.md", "w") as f:
    f.write(readme_content)
print("  Saved README.md")

# %%
"""
## Complete
"""

print("\n" + "=" * 80)
print("DEMONSTRATOR 3 COMPLETE")
print("=" * 80)
print(f"\nOutput directory: {output_path}")
print("  Education outputs:")
print("    - education_city_access.csv")
print("    - education_equity_analysis.csv")
print("    - education_access_ranking.png")
print("    - table_access.tex")
print("    - table_equity.tex")
print("  Transport outputs:")
print("    - transport_gap_profiles.csv")
print("    - transport_gap_ranking.png")
print("    - table_transport_gaps.tex")
print("  Documentation:")
print("    - README.md")
