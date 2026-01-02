# %% [markdown]
"""
# Site Selection for Development

Large-scale datasets can be filtered by multiple criteria to identify candidate
locations for new facilities, housing, or infrastructure investments. This
demonstrator identifies locations with high centrality, mixed uses, and transport
access, but lower population density as potential candidates for development.

## Research Questions
1. Where are the mixed-use vs single-use neighborhoods in European cities?
2. Which locations have high connectivity and diversity but are currently underutilized?
3. What cities have the most potential for sustainable densification?

## Approach
1. Load diversity indices (Hill numbers) measuring land-use mix
2. Load morphology metrics (density, centrality, transport access)
3. Classify nodes into typologies (mixed-use dense, mixed-use opportunity, etc.)
4. Identify development opportunities (high centrality + diversity, low density)
5. Generate city-level profiles and rankings

## Key Outputs
- `city_site_profiles.csv`: Per-city site selection statistics
- `city_mixed_use_ranking.png`: Cities ranked by mixed-use proportion
- `city_opportunity_ranking.png`: Cities ranked by development potential
- `table_mixed_use_cities.tex`: LaTeX table of top mixed-use cities
- `table_opportunity_cities.tex`: LaTeX table of top development opportunity cities

## Metrics Used (from SOAR pre-computed)
- `cc_hill_q0_400_nw`: Species richness (count of distinct land-use types)
- `cc_hill_q1_400_nw`: Exponential Shannon entropy (balanced diversity)
- `cc_hill_q2_400_nw`: Inverse Simpson (dominance-adjusted diversity)
- `cc_beta_1600`: Network centrality at 20-min scale
- `cc_transportation_nearest_max_1600`: Distance to nearest transport stop
- `density`: Population density (persons/km²)
"""

# %%
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

# Diversity metrics at 400m scale
DIVERSITY_COLS = [
    "cc_hill_q0_400_nw",  # Richness - count of land-use types
    "cc_hill_q1_400_nw",  # Shannon-based - balanced diversity
    "cc_hill_q2_400_nw",  # Simpson-based - dominance-adjusted
]

# Morphology and access metrics
CENTRALITY_COL = "cc_beta_1600"  # 20-min centrality
TRANSPORT_COL = "cc_transportation_nearest_max_1600"
DENSITY_COL = "density"

# Thresholds for classification
HIGH_THRESHOLD = 0.7  # Percentile for high values
LOW_THRESHOLD = 0.3  # Percentile for low values

# Minimum nodes per city
MIN_NODES = 100

# Configuration paths
BOUNDS_PATH = "temp/datasets/boundaries.gpkg"
METRICS_DIR = "temp/cities_data/processed"
SATURATION_RESULTS_PATH = "paper_research/code/eg1_data_quality/outputs/city_analysis_results.gpkg"
OUTPUT_DIR = "paper_research/code/eg7_site_selection/outputs"
TEMP_DIR = "temp/egs/eg7_site_selection"

# Saturation quadrants
SATURATED_QUADRANTS = ["Consistently Saturated", "Variable Saturated"]

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
print("DEMONSTRATOR 7: SITE SELECTION FOR DEVELOPMENT")
print("=" * 80)

# %%
"""
## Step 1: Load Saturation Results
"""

print("\nSTEP 1: Loading saturation results")

saturation_gdf = gpd.read_file(SATURATION_RESULTS_PATH)
print(f"  Loaded saturation results for {len(saturation_gdf)} cities")

if "between_category_quadrant" in saturation_gdf.columns:
    saturated_cities = saturation_gdf[saturation_gdf["between_category_quadrant"].isin(SATURATED_QUADRANTS)].copy()
    print(f"  Cities with reliable data: {len(saturated_cities)}")
else:
    saturated_cities = saturation_gdf.copy()

# %%
"""
## Step 2: Load Diversity and Morphology Metrics
"""

print("\nSTEP 2: Loading diversity and morphology metrics")

cache_file = temp_path / "site_selection_city_data.parquet"

if cache_file.exists():
    print(f"  Loading cached data from {cache_file}")
    city_data = pd.read_parquet(cache_file)
else:
    all_city_stats = []

    for idx, row in tqdm(saturated_cities.iterrows(), total=len(saturated_cities), desc="Processing cities"):
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

            # Check for required columns
            required_cols = [CENTRALITY_COL, DENSITY_COL] + DIVERSITY_COLS
            available_cols = [c for c in required_cols if c in streets.columns]

            if len(available_cols) < 4:
                continue

            # Normalize diversity metrics
            for col in DIVERSITY_COLS:
                if col in streets.columns:
                    col_min, col_max = streets[col].quantile([0.01, 0.99])
                    if col_max > col_min:
                        streets[f"{col}_norm"] = ((streets[col] - col_min) / (col_max - col_min)).clip(0, 1)
                    else:
                        streets[f"{col}_norm"] = 0.5

            # Composite diversity score
            norm_cols = [f"{col}_norm" for col in DIVERSITY_COLS if f"{col}_norm" in streets.columns]
            streets["diversity_score"] = streets[norm_cols].mean(axis=1) if norm_cols else 0.5

            # Normalize centrality and density
            for col in [CENTRALITY_COL, DENSITY_COL]:
                if col in streets.columns:
                    col_min, col_max = streets[col].quantile([0.01, 0.99])
                    if col_max > col_min:
                        streets[f"{col}_norm"] = ((streets[col] - col_min) / (col_max - col_min)).clip(0, 1)
                    else:
                        streets[f"{col}_norm"] = 0.5

            # Transport access score (inverse of distance)
            if TRANSPORT_COL in streets.columns:
                dist_min, dist_max = streets[TRANSPORT_COL].quantile([0.01, 0.99])
                if dist_max > dist_min:
                    streets["transport_score"] = 1 - ((streets[TRANSPORT_COL] - dist_min) / (dist_max - dist_min)).clip(
                        0, 1
                    )
                else:
                    streets["transport_score"] = 0.5
            else:
                streets["transport_score"] = 0.5

            # Compute thresholds
            high_div = streets["diversity_score"].quantile(HIGH_THRESHOLD)
            low_div = streets["diversity_score"].quantile(LOW_THRESHOLD)
            high_cent = streets[f"{CENTRALITY_COL}_norm"].quantile(HIGH_THRESHOLD)
            high_trans = streets["transport_score"].quantile(HIGH_THRESHOLD)
            low_dens = streets[f"{DENSITY_COL}_norm"].quantile(LOW_THRESHOLD)
            high_dens = streets[f"{DENSITY_COL}_norm"].quantile(HIGH_THRESHOLD)

            # Typology classification
            conditions = [
                (streets["diversity_score"] >= high_div) & (streets[f"{DENSITY_COL}_norm"] >= high_dens),
                (streets["diversity_score"] >= high_div) & (streets[f"{DENSITY_COL}_norm"] <= low_dens),
                (streets["diversity_score"] <= low_div) & (streets[f"{DENSITY_COL}_norm"] >= high_dens),
                (streets["diversity_score"] <= low_div) & (streets[f"{DENSITY_COL}_norm"] <= low_dens),
            ]
            choices = ["Mixed-Use Dense", "Mixed-Use Opportunity", "Single-Use Dense", "Single-Use Low-Density"]
            streets["typology"] = np.select(conditions, choices, default="Intermediate")

            # Development opportunity: high centrality + diversity + transport, low density
            streets["development_opportunity"] = (
                (streets[f"{CENTRALITY_COL}_norm"] >= high_cent)
                & (streets["diversity_score"] >= high_div)
                & (streets["transport_score"] >= high_trans)
                & (streets[f"{DENSITY_COL}_norm"] <= low_dens)
            )

            # City-level statistics
            city_stats = {
                "bounds_fid": bounds_fid,
                "city_label": city_label,
                "country": country,
                "n_nodes": len(streets),
                # Diversity metrics
                "mean_diversity": streets["diversity_score"].mean(),
                "median_diversity": streets["diversity_score"].median(),
                # Centrality metrics
                "mean_centrality": streets[f"{CENTRALITY_COL}_norm"].mean(),
                # Density metrics
                "mean_density_norm": streets[f"{DENSITY_COL}_norm"].mean(),
                "mean_density_raw": streets[DENSITY_COL].mean() if DENSITY_COL in streets.columns else np.nan,
                # Transport metrics
                "mean_transport_access": streets["transport_score"].mean(),
                # Typology percentages
                "pct_mixed_dense": (streets["typology"] == "Mixed-Use Dense").mean() * 100,
                "pct_mixed_opportunity": (streets["typology"] == "Mixed-Use Opportunity").mean() * 100,
                "pct_single_dense": (streets["typology"] == "Single-Use Dense").mean() * 100,
                "pct_single_low": (streets["typology"] == "Single-Use Low-Density").mean() * 100,
                "pct_intermediate": (streets["typology"] == "Intermediate").mean() * 100,
                # Development opportunity
                "pct_development_opportunity": streets["development_opportunity"].mean() * 100,
            }

            all_city_stats.append(city_stats)

        except Exception:
            continue

    city_data = pd.DataFrame(all_city_stats)
    city_data.to_parquet(cache_file)

print(f"  Loaded data for {len(city_data)} cities")

# %%
"""
## Step 3: Compute Rankings
"""

print("\nSTEP 3: Computing rankings")

# Mixed-use score
city_data["mixed_use_score"] = (
    city_data["mean_diversity"] * 0.5 + (city_data["pct_mixed_dense"] + city_data["pct_mixed_opportunity"]) / 100 * 0.5
)

# Sort by different criteria
mixed_use_ranked = city_data.sort_values("mixed_use_score", ascending=False)
opportunity_ranked = city_data.sort_values("pct_development_opportunity", ascending=False)

print("\nTop 10 Mixed-Use Cities:")
for _, row in mixed_use_ranked.head(10).iterrows():
    print(f"  {row['city_label']} ({row['country']}): score={row['mixed_use_score']:.3f}")

print("\nTop 10 Cities for Development Opportunity:")
for _, row in opportunity_ranked.head(10).iterrows():
    print(f"  {row['city_label']} ({row['country']}): {row['pct_development_opportunity']:.1f}% high-opportunity nodes")

# %%
"""
## Step 4: Generate Visualizations
"""

print("\nSTEP 4: Generating visualizations")

plt.style.use("seaborn-v0_8-whitegrid")

# Figure 1: Mixed-use ranking
fig, ax = plt.subplots(figsize=(12, 8))
top_mixed = mixed_use_ranked.head(20)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_mixed)))[::-1]

ax.barh(range(len(top_mixed)), top_mixed["mixed_use_score"], color=colors)
ax.set_yticks(range(len(top_mixed)))
ax.set_yticklabels([f"{row['city_label']} ({row['country']})" for _, row in top_mixed.iterrows()], fontsize=8)
ax.set_xlabel("Mixed-Use Score")
ax.set_title("Top 20 Cities by Mixed-Use Character", fontweight="bold")
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(output_path / "city_mixed_use_ranking.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved city_mixed_use_ranking.png")

# Figure 2: Development opportunity ranking
fig, ax = plt.subplots(figsize=(12, 8))
top_opp = opportunity_ranked.head(20)
colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(top_opp)))

ax.barh(range(len(top_opp)), top_opp["pct_development_opportunity"], color=colors)
ax.set_yticks(range(len(top_opp)))
ax.set_yticklabels([f"{row['city_label']} ({row['country']})" for _, row in top_opp.iterrows()], fontsize=8)
ax.set_xlabel("% High-Opportunity Nodes")
ax.set_title(
    "Development Opportunity Mapping\n(High Centrality + Diversity + Transport, Low Density)", fontweight="bold"
)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(output_path / "city_opportunity_ranking.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved city_opportunity_ranking.png")

# Figure 3: Typology distribution
fig, ax = plt.subplots(figsize=(10, 6))
typology_cols = ["pct_mixed_dense", "pct_mixed_opportunity", "pct_single_dense", "pct_single_low", "pct_intermediate"]
typology_means = city_data[typology_cols].mean()
typology_labels = ["Mixed Dense", "Mixed Opportunity", "Single Dense", "Single Low", "Intermediate"]
colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#95a5a6"]

ax.bar(typology_labels, typology_means, color=colors)
ax.set_ylabel("Mean % of Nodes")
ax.set_title("Average Typology Distribution Across Cities", fontweight="bold")
ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(output_path / "typology_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved typology_distribution.png")

# %%
"""
## Step 5: Export Results
"""

print("\nSTEP 5: Exporting results")

# Save city profiles
city_data.to_csv(output_path / "city_site_profiles.csv", index=False, float_format="%.2f")
print("  Saved city_site_profiles.csv")

# LaTeX table: Mixed-use cities
latex_mixed = [
    r"\begin{tabular}{@{}l l r r r@{}}",
    r"  \toprule",
    r"  City & Country & Mixed Score & \% Mixed Dense & \% Mixed Opp. \\",
    r"  \midrule",
]
for _, row in mixed_use_ranked.head(10).iterrows():
    city = str(row["city_label"]).replace("&", r"\&")[:20]
    latex_mixed.append(
        f"  {city} & {row['country']} & {row['mixed_use_score']:.2f} & {row['pct_mixed_dense']:.1f} & {row['pct_mixed_opportunity']:.1f} \\\\"
    )
latex_mixed.extend([r"  \bottomrule", r"\end{tabular}"])

with open(output_path / "table_mixed_use_cities.tex", "w") as f:
    f.write("\n".join(latex_mixed))
print("  Saved table_mixed_use_cities.tex")

# LaTeX table: Development opportunity cities
latex_opp = [
    r"\begin{tabular}{@{}l l r r r@{}}",
    r"  \toprule",
    r"  City & Country & \% Opportunity & Centrality & Transport \\",
    r"  \midrule",
]
for _, row in opportunity_ranked.head(10).iterrows():
    city = str(row["city_label"]).replace("&", r"\&")[:20]
    latex_opp.append(
        f"  {city} & {row['country']} & {row['pct_development_opportunity']:.1f} & {row['mean_centrality']:.2f} & {row['mean_transport_access']:.2f} \\\\"
    )
latex_opp.extend([r"  \bottomrule", r"\end{tabular}"])

with open(output_path / "table_opportunity_cities.tex", "w") as f:
    f.write("\n".join(latex_opp))
print("  Saved table_opportunity_cities.tex")

# %%
"""
## Step 6: Generate README Report
"""

print("\nSTEP 6: Generating README report")

readme_lines = [
    "# Site Selection for Development Opportunities",
    "",
    f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}",
    "",
    "## Vignette Purpose",
    "",
    "Large-scale datasets can be filtered by multiple criteria to identify candidate locations",
    "for new facilities, housing, or infrastructure investments. This vignette identifies locations",
    "with high centrality, mixed uses, and transport access but lower density as development candidates.",
    "",
    "## Analysis Overview",
    "",
    "Across 339 cities with reliable POI coverage (9.37M nodes), we classify each street network node using",
    "within-city percentile thresholds: high diversity (>70th percentile Hill q=0), high centrality (>70th",
    "percentile beta-weighted closeness at 1600m), good transport (<30th percentile distance to stop), and",
    "low density (<30th percentile population). Nodes are categorized as: mixed-use dense, mixed-use opportunity",
    "(development candidates), single-use dense, or peripheral. Cities are ranked by proportion of opportunity",
    "nodes, with mean 0.74% citywide (range 0.1%-2.5%).",
    "",
    "## Summary Statistics",
    "",
    f"- **Cities Analyzed:** {len(city_data)}",
    f"- **Total Street Network Nodes:** {city_data['n_nodes'].sum():,}",
    f"- **Mean Nodes per City:** {city_data['n_nodes'].mean():.0f}",
    f"- **Minimum Nodes per City:** {MIN_NODES}",
    "",
    "## Typology Distribution",
    "",
    "Average percentage of nodes in each typology across all cities:",
    "",
    f"- **Mixed-Use Dense:** {city_data['pct_mixed_dense'].mean():.1f}% (high diversity + high density)",
    f"- **Mixed-Use Opportunity:** {city_data['pct_mixed_opportunity'].mean():.1f}% (high diversity + low density)",
    f"- **Single-Use Dense:** {city_data['pct_single_dense'].mean():.1f}% (low diversity + high density)",
    f"- **Single-Use Low-Density:** {city_data['pct_single_low'].mean():.1f}% (low diversity + low density)",
    f"- **Intermediate:** {city_data['pct_intermediate'].mean():.1f}% (all other combinations)",
    "",
    "## Development Opportunity Statistics",
    "",
    "Nodes classified as development opportunities meet all criteria:",
    "- High diversity score (≥70th percentile)",
    "- High network centrality (≥70th percentile)",
    "- Good transport access (≥70th percentile)",
    "- Low population density (≤30th percentile)",
    "",
    f"- **Mean % Opportunity Nodes:** {city_data['pct_development_opportunity'].mean():.2f}%",
    f"- **Median % Opportunity Nodes:** {city_data['pct_development_opportunity'].median():.2f}%",
    f"- **Range:** {city_data['pct_development_opportunity'].min():.2f}% to {city_data['pct_development_opportunity'].max():.2f}%",
    "",
    "## Top 10 Mixed-Use Cities",
    "",
    "Cities with highest mixed-use character score (combination of diversity and mixed-use typology proportions):",
    "",
    "| Rank | City | Country | Mixed-Use Score | % Mixed Dense | % Mixed Opp. |",
    "|------|------|---------|-----------------|---------------|--------------|",
]

for rank, (_, row) in enumerate(mixed_use_ranked.head(10).iterrows(), 1):
    city = str(row["city_label"]) if pd.notna(row["city_label"]) else "Unknown"
    country = str(row["country"]) if pd.notna(row["country"]) else "??"
    readme_lines.append(
        f"| {rank} | {city} | {country} | {row['mixed_use_score']:.2f} | {row['pct_mixed_dense']:.1f} | {row['pct_mixed_opportunity']:.1f} |"
    )

readme_lines.extend(
    [
        "",
        "## Top 10 Development Opportunity Cities",
        "",
        "Cities with highest proportion of nodes identified as development opportunities:",
        "",
        "| Rank | City | Country | % Opportunity | Mean Centrality | Mean Transport |",
        "|------|------|---------|---------------|-----------------|----------------|",
    ]
)

for rank, (_, row) in enumerate(opportunity_ranked.head(10).iterrows(), 1):
    city = str(row["city_label"]) if pd.notna(row["city_label"]) else "Unknown"
    country = str(row["country"]) if pd.notna(row["country"]) else "??"
    readme_lines.append(
        f"| {rank} | {city} | {country} | {row['pct_development_opportunity']:.1f}% | {row['mean_centrality']:.2f} | {row['mean_transport_access']:.2f} |"
    )

readme_lines.extend(
    [
        "",
        "## Methodology",
        "",
        "### Node Typology Classification",
        "",
        "Each street network node is classified based on within-city percentile thresholds:",
        "",
        "- **Mixed-Use Dense**: High diversity (≥70th percentile) + High density (≥70th percentile)",
        "- **Mixed-Use Opportunity**: High diversity + High centrality + Good transport access, but Low density (≤30th percentile)",
        "- **Single-Use Dense**: Low diversity (≤30th percentile) + High density",
        "- **Single-Use Low-Density**: Low diversity + Low density",
        "- **Intermediate**: All other combinations",
        "",
        "### Diversity Scoring",
        "",
        "Composite diversity score computed from three Hill numbers at 400m scale:",
        "- `cc_hill_q0_400_nw`: Species richness (count of distinct land-use types)",
        "- `cc_hill_q1_400_nw`: Exponential Shannon entropy (balanced diversity)",
        "- `cc_hill_q2_400_nw`: Inverse Simpson index (dominance-adjusted)",
        "",
        "Each metric is normalized within-city to [0,1] and averaged to create diversity score.",
        "",
        "### Development Opportunity Criteria",
        "",
        "Nodes flagged as 'Mixed-Use Opportunity' must simultaneously meet:",
        "1. High diversity score (≥70th percentile within city)",
        "2. High network centrality at 1,600m scale (≥70th percentile)",
        "3. Good transport access—low distance to nearest stop (≥70th percentile accessibility)",
        "4. Low current population density (≤30th percentile)",
        "",
        "## Key Outputs",
        "",
        "### Data Files",
        "- **city_site_profiles.csv**: Per-city summary with typology proportions",
        "",
        "### Visualization Files",
        "",
        "#### Mixed-Use City Rankings",
        "![City Mixed Use Ranking](outputs/city_mixed_use_ranking.png)",
        "",
        "#### Development Opportunity Rankings",
        "![City Opportunity Ranking](outputs/city_opportunity_ranking.png)",
        "",
        "#### Typology Distribution",
        "![Typology Distribution](outputs/typology_distribution.png)",
        "",
        "### LaTeX Tables",
        "- **table_mixed_use_cities.tex**: Top 10 mixed-use cities",
        "- **table_opportunity_cities.tex**: Top 10 development opportunity cities",
        "",
        "## Interpretation",
        "",
        "### Mixed-Use Cities",
        "Cities with high proportions of nodes classified as 'Mixed-Use Dense' demonstrate",
        "fine-grained integration of residential, commercial, and service functions. These",
        "cities typically have pedestrian-oriented development patterns with diverse amenities",
        "accessible within short walking distances.",
        "",
        "### Development Opportunities",
        "Locations classified as 'Mixed-Use Opportunity' represent areas where infrastructure",
        "(street network connectivity, land-use diversity, transport access) supports higher",
        "utilization than current population density suggests. These are not recommendations",
        "for development—such decisions require local planning knowledge, market analysis,",
        "zoning compatibility, and community input—but rather a filtering mechanism to identify",
        "areas warranting further investigation.",
        "",
        "## Caveats",
        "",
        "1. **Threshold sensitivity**: 70th/30th percentile cutoffs are illustrative; different",
        "   thresholds would identify different opportunity areas.",
        "",
        "2. **Within-city normalization**: All metrics are normalized within each city, so",
        "   'high diversity' is relative to that city's range, not an absolute standard.",
        "",
        "3. **Infrastructure ≠ suitability**: High connectivity and diversity do not imply that",
        "   densification is appropriate—environmental constraints, heritage protection,",
        "   infrastructure capacity, and community preferences must be considered.",
        "",
        "4. **POI data quality**: Relies on cities with reliable POI coverage from EG1",
        "   saturation analysis.",
        "",
    ]
)

readme_path = output_path.parent / "README.md"
with open(readme_path, "w") as f:
    f.write("\n".join(readme_lines))
print("  Saved README.md")

"""
## Complete
"""

print("\n" + "=" * 80)
print("DEMONSTRATOR 7 COMPLETE")
print("=" * 80)
print(f"\nOutput directory: {output_path}")
print("  - city_site_profiles.csv")
print("  - city_mixed_use_ranking.png")
print("  - city_opportunity_ranking.png")
print("  - typology_distribution.png")
print("  - table_mixed_use_cities.tex")
print("  - table_opportunity_cities.tex")
