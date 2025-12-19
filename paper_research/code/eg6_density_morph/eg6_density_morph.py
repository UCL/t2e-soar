# %% [markdown]
"""
# Urban Density and Building Morphology Patterns

Analyzes relationships between population density and building morphology across
European cities to reveal typological patterns in urban form.

## Research Question
How do density patterns (population vs. building volume/coverage) vary across cities?
Do cities follow consistent density-form relationships, or exhibit distinct typologies?

## Approach
Rather than aggregating to city means (which loses within-city heterogeneity), we:
1. Cluster NODES by their morphology profile
2. Characterize each CITY by proportions of node types

This reveals whether cities are homogeneous or mixed - e.g., a city might be
"40% dense-vertical, 30% suburban-sprawl, 30% historic-compact".

## Steps
1. Load ALL city boundaries (no pre-filtering)
2. Load and sample morphology metrics at 400m scale
3. Cluster NODES into morphology types (K-means)
4. Extract satellite imagery for cluster representatives
5. Compute city profiles (% nodes in each cluster)
6. Visualize city typology distributions
7. Export results

## SOAR Metrics Used (400m scale, weighted versions)
- **Building metrics**: area, perimeter, compactness, orientation, volume, form factor, corners, shape index, fractal dimension
- **Block metrics**: area, perimeter, compactness, orientation, coverage ratio
- **Population**: density (persons/km²)

All metrics use weighted neighborhood averages (_wt suffix) to account for spatial autocorrelation.

## Key Outputs
- **city_morphology_profiles.csv**: Per-city node type proportions
- **node_cluster_summary.csv**: Node cluster characteristics
- **city_profiles_heatmap.png**: Heatmap of city typology proportions
- **city_profiles_pca.png**: PCA of city morphology profiles
- **README.md**: Summary report with key findings
"""

# %%
import math
import urllib.request
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# %%
"""
## Configuration
"""

# Morphology metrics at 400m scale (using weighted versions)
MORPH_COLUMNS = [
    "cc_area_mean_400_wt",
    "cc_perimeter_mean_400_wt",
    "cc_compactness_mean_400_wt",
    "cc_orientation_mean_400_wt",
    "cc_volume_mean_400_wt",
    "cc_form_factor_mean_400_wt",
    "cc_corners_mean_400_wt",
    "cc_shape_index_mean_400_wt",
    "cc_fractal_dimension_mean_400_wt",
    "cc_block_area_mean_400_wt",
    "cc_block_perimeter_mean_400_wt",
    "cc_block_compactness_mean_400_wt",
    "cc_block_orientation_mean_400_wt",
    "cc_block_covered_ratio_mean_400_wt",
]

# Human-readable names for metrics
MORPH_LABELS = {
    "cc_area_mean_400_wt": "Building Area",
    "cc_perimeter_mean_400_wt": "Building Perimeter",
    "cc_compactness_mean_400_wt": "Compactness",
    "cc_orientation_mean_400_wt": "Orientation",
    "cc_volume_mean_400_wt": "Building Volume",
    "cc_form_factor_mean_400_wt": "Form Factor",
    "cc_corners_mean_400_wt": "Corners",
    "cc_shape_index_mean_400_wt": "Shape Index",
    "cc_fractal_dimension_mean_400_wt": "Fractal Dimension",
    "cc_block_area_mean_400_wt": "Block Area",
    "cc_block_perimeter_mean_400_wt": "Block Perimeter",
    "cc_block_compactness_mean_400_wt": "Block Compactness",
    "cc_block_orientation_mean_400_wt": "Block Orientation",
    "cc_block_covered_ratio_mean_400_wt": "Coverage Ratio",
    "density": "Population Density",
}

# Population density column
DENSITY_COL = "density"

# Minimum nodes per city for reliable statistics
MIN_NODES = 100

# Number of node clusters for typology (reduced to ensure clusters are at least 5% of total)
N_NODE_CLUSTERS = 4

# Sample size per city for node clustering: max(50000, 50% of total nodes)
MIN_SAMPLE_NODES = 50000
SAMPLE_FRACTION = 0.5

# Configuration paths
BOUNDS_PATH = "temp/datasets/boundaries.gpkg"
METRICS_DIR = "temp/cities_data/processed"
OUTPUT_DIR = "paper_research/code/eg6_density_morph/outputs"
TEMP_DIR = "temp/egs/eg6_density_morph"

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
print("Vignette 6: Urban Density and Building Morphology Patterns")
print("=" * 80)
print(f"\nOutput directory: {output_path}")
print(f"Temp directory: {temp_path}")

# %%
"""
## Step 1: Load City Boundaries
"""

print("\nSTEP 1: Loading city boundaries")

bounds_gdf = gpd.read_file(BOUNDS_PATH, layer="bounds")
print(f"  Loaded {len(bounds_gdf)} city boundaries")

# %%
"""
## Step 2: Load and Sample Node-Level Morphology Metrics
"""

print("\nSTEP 2: Loading node-level morphology metrics")

# Columns to load for clustering
load_cols = MORPH_COLUMNS + [DENSITY_COL]

# Check for cached data
node_cache_file = temp_path / "morphology_nodes_sampled.parquet"
city_cache_file = temp_path / "morphology_city_summary.parquet"

if node_cache_file.exists() and city_cache_file.exists():
    print("  Loading cached node and city data...")
    nodes_df = pd.read_parquet(node_cache_file)
    city_df = pd.read_parquet(city_cache_file)
    print(f"  Loaded {len(nodes_df)} sampled nodes from {len(city_df)} cities")
else:
    print("  Loading and sampling from individual city metrics files...")
    all_nodes = []
    city_summaries = []

    for idx, row in tqdm(bounds_gdf.iterrows(), total=len(bounds_gdf), desc="Loading cities"):
        bounds_fid = row.name if isinstance(row.name, int) else idx
        city_label = row.get("label", str(bounds_fid))
        country = row.get("country", "Unknown")

        # Load metrics file for this city
        metrics_file = metrics_dir / f"metrics_{bounds_fid}.gpkg"
        if not metrics_file.exists():
            continue

        try:
            # Load with geopandas to get geometry, then extract coordinates
            gdf = gpd.read_file(metrics_file, columns=MORPH_COLUMNS + [DENSITY_COL], layer="streets")

            if len(gdf) == 0:
                continue

            # Check we have all required columns
            if not all(col in gdf.columns for col in load_cols):
                continue

            # Extract x/y coordinates from geometry centroid
            gdf["x"] = gdf.geometry.centroid.x
            gdf["y"] = gdf.geometry.centroid.y

            # Drop geometry column to save memory
            gdf = gdf.drop(columns=["geometry"])

            # Filter to valid rows
            valid_mask = gdf[load_cols].notna().all(axis=1)
            gdf = gdf[valid_mask]

            if len(gdf) < MIN_NODES:
                continue

            n_total_nodes = len(gdf)

            # Sample nodes for clustering: max(1000, 10% of nodes)
            n_sample = max(MIN_SAMPLE_NODES, int(n_total_nodes * SAMPLE_FRACTION))
            if n_sample < n_total_nodes:
                sample_gdf = gdf.sample(n=n_sample, random_state=42)
            else:
                sample_gdf = gdf

            # Add city identifiers to sampled nodes (keep x, y for satellite imagery)
            sample_df = sample_gdf[load_cols + ["x", "y"]].copy()
            sample_df["bounds_fid"] = bounds_fid
            sample_df["city_label"] = city_label
            sample_df["country"] = country
            all_nodes.append(sample_df)

            # Compute city-level summary for quality assessment
            city_summaries.append(
                {
                    "bounds_fid": bounds_fid,
                    "city_label": city_label,
                    "country": country,
                    "n_nodes": n_total_nodes,
                    "density_mean": gdf[DENSITY_COL].mean(),
                    "coverage_mean": gdf["cc_block_covered_ratio_mean_400_wt"].mean(),
                    "volume_mean": gdf["cc_volume_mean_400_wt"].mean()
                    if "cc_volume_mean_400_wt" in gdf.columns
                    else np.nan,
                }
            )

        except Exception as e:
            # Print first few errors to diagnose
            if len(all_nodes) < 3:
                print(f"    Error loading {city_label}: {e}")
            continue

    # Combine all nodes
    if len(all_nodes) == 0:
        raise ValueError(f"No cities loaded successfully. Check that metrics files exist in {metrics_dir}")

    nodes_df = pd.concat(all_nodes, ignore_index=True)
    city_df = pd.DataFrame(city_summaries)

    print(f"  Loaded {len(nodes_df)} sampled nodes from {len(city_df)} cities")

    # Save to cache
    nodes_df.to_parquet(node_cache_file)
    city_df.to_parquet(city_cache_file)
    print("  Saved cache files")

# %%
"""
## Step 3: Cluster Nodes by Morphology Profile
"""

print("\nSTEP 4: Clustering nodes by morphology profile")

# Prepare node features for clustering
node_features = nodes_df[MORPH_COLUMNS].copy()

# Handle missing values and infinities
node_features = node_features.replace([np.inf, -np.inf], np.nan)
valid_nodes = node_features.notna().all(axis=1)
node_features_clean = node_features[valid_nodes].copy()
nodes_df_clean = nodes_df[valid_nodes].copy()

print(f"  Valid nodes: {len(node_features_clean)}")

# Standardize features
scaler = StandardScaler()
X_nodes = scaler.fit_transform(node_features_clean)

# Fit K-means on original features
print(f"  Fitting K-means with {N_NODE_CLUSTERS} clusters...")
kmeans = KMeans(n_clusters=N_NODE_CLUSTERS, random_state=42, n_init=10)
nodes_df_clean["node_cluster"] = kmeans.fit_predict(X_nodes)

print("\n  Node cluster characteristics:")
cluster_profiles = []
for c in range(N_NODE_CLUSTERS):
    cluster_nodes = node_features_clean[nodes_df_clean["node_cluster"] == c]
    n_nodes = len(cluster_nodes)

    profile = {"cluster": c, "n_nodes": n_nodes, "pct_nodes": n_nodes / len(nodes_df_clean) * 100}

    for col in MORPH_COLUMNS:
        col_short = col.replace("cc_", "").replace("_mean_400_wt", "").replace("_", " ").title()
        profile[f"{col_short} Mean"] = cluster_nodes[col].mean()

    cluster_profiles.append(profile)

    # Characterize cluster using key features
    volume = cluster_nodes["cc_volume_mean_400_wt"].mean()
    coverage = cluster_nodes["cc_block_covered_ratio_mean_400_wt"].mean()
    area = cluster_nodes["cc_area_mean_400_wt"].mean()
    compactness = cluster_nodes["cc_compactness_mean_400_wt"].mean()
    shape_idx = cluster_nodes["cc_shape_index_mean_400_wt"].mean()

    volume_cat = "vertical" if volume > node_features_clean["cc_volume_mean_400_wt"].median() else "low-rise"
    coverage_cat = (
        "compact" if coverage > node_features_clean["cc_block_covered_ratio_mean_400_wt"].median() else "sparse"
    )
    area_cat = "large" if area > node_features_clean["cc_area_mean_400_wt"].median() else "small"
    shape_cat = "complex" if shape_idx > node_features_clean["cc_shape_index_mean_400_wt"].median() else "simple"

    print(
        f"    Cluster {c}: {n_nodes:,} nodes ({profile['pct_nodes']:.1f}%) - "
        f"{volume_cat}, {coverage_cat}, {area_cat} footprint, {shape_cat} shape"
    )

cluster_profiles_df = pd.DataFrame(cluster_profiles)

# %%
"""
## Step 4: Extract Representative Satellite Images for Each Cluster
Set DOWNLOAD_SATELLITE = False to skip if network is slow/unavailable.
"""

DOWNLOAD_SATELLITE = False  # Set to True to download satellite imagery

print("\nSTEP 4: Finding representative locations for each cluster")

# Find representative node for each cluster (closest to centroid in feature space)
cluster_reps = []

for c in range(N_NODE_CLUSTERS):
    # Get nodes in this cluster
    cluster_mask = nodes_df_clean["node_cluster"] == c
    cluster_features = X_nodes[cluster_mask]

    # Find centroid in standardized space
    centroid = cluster_features.mean(axis=0)

    # Find closest actual node to centroid
    distances = np.linalg.norm(cluster_features - centroid, axis=1)
    closest_idx = np.argmin(distances)

    # Get the original node data
    cluster_nodes_df = nodes_df_clean[cluster_mask].reset_index(drop=True)
    rep_node = cluster_nodes_df.iloc[closest_idx]

    cluster_reps.append(
        {
            "cluster": c,
            "bounds_fid": rep_node["bounds_fid"],
            "city_label": rep_node["city_label"],
            "country": rep_node["country"],
            "x": rep_node["x"],
            "y": rep_node["y"],
        }
    )
    print(f"  Cluster {c} representative: {rep_node['city_label']}, {rep_node['country']}")

# Save representative locations for manual lookup or later satellite download
reps_df = pd.DataFrame(cluster_reps)
reps_df.to_csv(output_path / "cluster_representatives.csv", index=False)
print("  Saved cluster representatives to cluster_representatives.csv")

if DOWNLOAD_SATELLITE:
    print("\n  Downloading satellite tiles (this may take a moment)...")

    def latlon_to_tile(lat, lon, zoom):
        """Convert lat/lon to tile coordinates."""
        lat_rad = math.radians(lat)
        n = 2.0**zoom
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return x, y

    zoom = 18  # Street-level detail

    for rep in cluster_reps:
        c = rep["cluster"]
        x, y = rep["x"], rep["y"]

        try:
            # Assume coordinates are in projected CRS (EPSG:3035) - convert to WGS84
            from pyproj import Transformer

            # Transform from EPSG:3035 (LAEA Europe) to EPSG:4326 (WGS84)
            transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(x, y)

            # Convert to tile coordinates
            x_tile, y_tile = latlon_to_tile(lat, lon, zoom)

            # ESRI World Imagery tile URL (free, no authentication)
            tile_url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y_tile}/{x_tile}"

            # Download tile with short timeout
            response = urllib.request.urlopen(tile_url, timeout=5)
            img_data = response.read()
            img = Image.open(BytesIO(img_data))

            # Save
            output_file = output_path / f"cluster_{c}_satellite.png"
            img.save(output_file)
            print(f"    Cluster {c}: Saved {output_file.name} (lat={lat:.4f}, lon={lon:.4f})")

        except Exception as e:
            print(f"    Cluster {c}: Error - {e}")
else:
    print("  Satellite download skipped (set DOWNLOAD_SATELLITE=True to enable)")

# %%
"""
## Step 5: Compute City Profiles (Proportion of Node Types)
"""

print("\nSTEP 5: Computing city profiles from node cluster proportions")

# Compute proportion of nodes in each cluster for each city
city_profiles = []

for bounds_fid in nodes_df_clean["bounds_fid"].unique():
    city_nodes = nodes_df_clean[nodes_df_clean["bounds_fid"] == bounds_fid]
    city_label = city_nodes["city_label"].iloc[0]
    country = city_nodes["country"].iloc[0]
    n_nodes = len(city_nodes)

    profile = {
        "bounds_fid": bounds_fid,
        "city_label": city_label,
        "country": country,
        "n_nodes_sampled": n_nodes,
    }

    # Compute proportion in each cluster
    for c in range(N_NODE_CLUSTERS):
        pct = (city_nodes["node_cluster"] == c).sum() / n_nodes * 100
        profile[f"pct_cluster_{c}"] = pct

    city_profiles.append(profile)

city_profiles_df = pd.DataFrame(city_profiles)

# Merge with city data
city_profiles_df = city_profiles_df.merge(
    city_df[["bounds_fid", "n_nodes", "density_mean", "coverage_mean"]],
    on="bounds_fid",
    how="left",
)

print(f"  Computed profiles for {len(city_profiles_df)} cities")

print("\n  Cluster distribution across all nodes:")
for c in range(N_NODE_CLUSTERS):
    total_in_cluster = (nodes_df_clean["node_cluster"] == c).sum()
    pct = total_in_cluster / len(nodes_df_clean) * 100
    print(f"    Cluster {c}: {total_in_cluster:,} nodes ({pct:.1f}%)")

# %%
"""
## Step 6: Node Cluster Profile Visualization
"""

print("\nSTEP 6: Generating node cluster profile visualization")

# Compute cluster means from nodes
node_cluster_means = node_features_clean.copy()
node_cluster_means["node_cluster"] = nodes_df_clean["node_cluster"]
cluster_means = node_cluster_means.groupby("node_cluster")[MORPH_COLUMNS].mean()

# Create bar chart of cluster profiles for all morphology features
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.flatten()

cluster_colors = plt.cm.viridis(np.linspace(0, 1, N_NODE_CLUSTERS))

for i, col in enumerate(MORPH_COLUMNS):
    ax = axes[i]
    label = MORPH_LABELS.get(col, col)
    values = cluster_means[col].values
    bars = ax.bar(range(N_NODE_CLUSTERS), values, color=cluster_colors)
    ax.set_xlabel("Cluster", fontsize=9)
    ax.set_ylabel(label, fontsize=9)
    ax.set_xticks(range(N_NODE_CLUSTERS))
    ax.set_title(label, fontsize=10)
    ax.tick_params(labelsize=8)

# Hide unused subplots (we have 14 features, so 2 unused plots in 4x4 grid)
for i in range(len(MORPH_COLUMNS), len(axes)):
    axes[i].axis("off")

plt.suptitle("Node Morphology Cluster Profiles", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(output_path / "node_cluster_profiles.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: node_cluster_profiles.png")

# %%
"""
## Step 7: City Profile Heatmap (Proportions by Node Cluster)
"""

print("\nSTEP 7: Generating city profile heatmap")

# Sort cities by cluster proportions (hierarchical: cluster 0 %, then cluster 1 %, etc.)
sort_cols = [f"pct_cluster_{c}" for c in range(N_NODE_CLUSTERS)]
city_profiles_sorted = city_profiles_df.sort_values(sort_cols, ascending=False)

# Create proportion matrix for heatmap
pct_cols = [f"pct_cluster_{c}" for c in range(N_NODE_CLUSTERS)]
prop_matrix = city_profiles_sorted[pct_cols].values

# Select a sample of cities for visualization if too many
max_cities_display = 100
if len(city_profiles_sorted) > max_cities_display:
    # Take top cities by diversity (using std of cluster percentages as proxy)
    city_profiles_sorted["cluster_std"] = city_profiles_sorted[pct_cols].std(axis=1)
    # Mix of high and low diversity
    top_diverse = city_profiles_sorted.nlargest(max_cities_display // 2, "cluster_std")
    top_uniform = city_profiles_sorted.nsmallest(max_cities_display // 2, "cluster_std")
    city_sample = pd.concat([top_diverse, top_uniform]).sort_values(sort_cols, ascending=False)
else:
    city_sample = city_profiles_sorted

fig, ax = plt.subplots(figsize=(10, max(8, len(city_sample) * 0.08)))
im = ax.imshow(city_sample[pct_cols].values, aspect="auto", cmap="YlOrRd")

ax.set_yticks(range(len(city_sample)))
ax.set_yticklabels(city_sample["city_label"], fontsize=6)
ax.set_xticks(range(N_NODE_CLUSTERS))
ax.set_xticklabels([f"Cluster {c}" for c in range(N_NODE_CLUSTERS)])
ax.set_xlabel("Node Morphology Cluster")
ax.set_ylabel("City")
ax.set_title("City Profiles: Proportion of Nodes in Each Morphology Cluster (%)")

plt.colorbar(im, ax=ax, label="% of nodes")
plt.tight_layout()
plt.savefig(output_path / "city_profile_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: city_profile_heatmap.png")

# %%
"""
## Step 8: PCA of City Profiles
"""

print("\nSTEP 8: Generating PCA visualization of city profiles")

# PCA on city proportion profiles
X_city = city_profiles_df[pct_cols].values
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_city)

city_profiles_df["pca1"] = X_pca[:, 0]
city_profiles_df["pca2"] = X_pca[:, 1]

fig, ax = plt.subplots(figsize=(12, 8))

# Color by country (top 5 countries by city count, rest as "Other")
top_countries = city_profiles_df["country"].value_counts().head(5).index
city_profiles_df["country_group"] = city_profiles_df["country"].apply(lambda x: x if x in top_countries else "Other")

colors = plt.cm.tab10(np.linspace(0, 1, len(city_profiles_df["country_group"].unique())))
for i, country in enumerate(city_profiles_df["country_group"].unique()):
    subset = city_profiles_df[city_profiles_df["country_group"] == country]
    ax.scatter(
        subset["pca1"],
        subset["pca2"],
        c=[colors[i]],
        label=f"{country} ({len(subset)})",
        alpha=0.7,
        s=40,
    )

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)", fontsize=12)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)", fontsize=12)
ax.set_title("City Morphology Profiles (PCA)", fontsize=14)
ax.legend(loc="best")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_path / "city_profiles_pca.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: city_profiles_pca.png")

# %%
"""
## Step 9: Export Results
"""

print("\nSTEP 9: Exporting results")

# Prepare export dataframe with city profiles
export_cols = ["bounds_fid", "city_label", "country", "n_nodes_sampled", "n_nodes", "density_mean", "coverage_mean"]
export_cols += [f"pct_cluster_{c}" for c in range(N_NODE_CLUSTERS)]

export_df = city_profiles_df[export_cols].copy()

# Rename columns for clarity
rename_map = {
    "city_label": "City",
    "country": "Country",
    "n_nodes_sampled": "Nodes Sampled",
    "n_nodes": "Total Nodes",
    "density_mean": "Pop Density",
    "coverage_mean": "Coverage Ratio",
}
for c in range(N_NODE_CLUSTERS):
    rename_map[f"pct_cluster_{c}"] = f"% Cluster {c}"

export_df = export_df.rename(columns=rename_map)

# Sort by cluster proportions
sort_cols = [f"% Cluster {c}" for c in range(N_NODE_CLUSTERS)]
export_df = export_df.sort_values(sort_cols, ascending=False)

# Save full results
export_df.to_csv(output_path / "city_morphology_profiles.csv", index=False)
print(f"  Saved: city_morphology_profiles.csv ({len(export_df)} cities)")

# Save node cluster summary
cluster_profiles_df.to_csv(output_path / "node_cluster_summary.csv", index=False)
print("  Saved: node_cluster_summary.csv")

# Export top cities by each cluster type
for c in range(N_NODE_CLUSTERS):
    top_cities = export_df.nlargest(10, f"% Cluster {c}")[
        ["City", "Country"] + [f"% Cluster {i}" for i in range(N_NODE_CLUSTERS)]
    ]
    top_cities.to_csv(output_path / f"top_cluster_{c}_cities.csv", index=False)
print(f"  Saved: top_cluster_{{0-{N_NODE_CLUSTERS - 1}}}_cities.csv")

# %%
"""
## Step 10: Generate README Report
"""

print("\nSTEP 10: Generating README report")

# Compute summary statistics
n_total = len(city_df)
n_profiled = len(city_profiles_df)

# Node cluster descriptions
node_cluster_descriptions = []
for c in range(N_NODE_CLUSTERS):
    profile = cluster_profiles_df[cluster_profiles_df["cluster"] == c].iloc[0]
    n_nodes = profile["n_nodes"]
    pct = profile["pct_nodes"]

    # Get mean values from cluster_means (morphology features only)
    volume = cluster_means.loc[c, "cc_volume_mean_400_wt"]
    coverage = cluster_means.loc[c, "cc_block_covered_ratio_mean_400_wt"]
    area = cluster_means.loc[c, "cc_area_mean_400_wt"]

    # Simple characterization
    volume_cat = "High" if volume > cluster_means["cc_volume_mean_400_wt"].median() else "Low"
    coverage_cat = "High" if coverage > cluster_means["cc_block_covered_ratio_mean_400_wt"].median() else "Low"
    area_cat = "Large" if area > cluster_means["cc_area_mean_400_wt"].median() else "Small"

    node_cluster_descriptions.append(
        f"| {c} | {n_nodes:,} | {pct:.1f}% | {area:.0f} | {volume:.0f} | {coverage:.3f} | {area_cat} footprint, {volume_cat} volume, {coverage_cat} coverage |"
    )

readme_content = f"""# EG6: Urban Density and Building Morphology Patterns

## Summary

Analysis of population density vs building morphology across {n_total} European cities.
We cluster **nodes** by morphology profile, then characterize each **city** by the proportion
of its nodes in each cluster type. This preserves within-city heterogeneity.

## Methodology

1. Sample max({MIN_SAMPLE_NODES:,}, {SAMPLE_FRACTION * 100:.0f}%) nodes per city
2. Cluster nodes by {len(MORPH_COLUMNS)} morphology features (K-means, k={N_NODE_CLUSTERS}):
   - **Building**: area, perimeter, compactness, orientation, volume, form factor, corners, shape index, fractal dimension
   - **Block**: area, perimeter, compactness, orientation, coverage ratio
3. For each city, compute proportion of nodes in each morphology cluster
4. Cities are characterized by their full cluster proportion profile


## Node Morphology Clusters

K-means clustering (k={N_NODE_CLUSTERS}) on 14 morphology features identified these node types:

| Cluster | Nodes | % Total | Mean Area (m²) | Mean Volume (m³) | Mean Coverage | Characterization |
|---------|-------|---------|----------------|------------------|---------------|------------------|
{chr(10).join(node_cluster_descriptions)}

![Node Cluster Profiles](outputs/node_cluster_profiles.png)

## City Morphology Profiles

Each city is characterized by its distribution across node clusters:

![City Profile Heatmap](outputs/city_profile_heatmap.png)

![City Profiles PCA](outputs/city_profiles_pca.png)

## Cluster Proportion Analysis

Cities are characterized by their distribution across morphology clusters.

### Example City Profiles (Top 5 by Cluster 0)

| City | Country | % Cluster 0 | % Cluster 1 | % Cluster 2 | % Cluster 3 |
|------|---------|-------------|-------------|-------------|-------------|
"""

for _, row in city_profiles_df.nlargest(5, "pct_cluster_0").iterrows():
    pcts = " | ".join([f"{row[f'pct_cluster_{c}']:.1f}%" for c in range(N_NODE_CLUSTERS)])
    readme_content += f"| {row['city_label']} | {row['country']} | {pcts} |\n"

readme_content += """
## Outputs

- `city_morphology_profiles.csv`: Full city profiles with cluster proportions
- `node_cluster_summary.csv`: Node cluster characteristics
- `cluster_representatives.csv`: Representative city for each cluster
- `top_cluster_{{0-3}}_cities.csv`: Top 10 cities by each cluster type
- `node_cluster_profiles.png`: Node cluster characteristic profiles
- `city_profile_heatmap.png`: City proportions across node clusters
- `city_profiles_pca.png`: PCA of city morphology profiles

## Reproducibility

```bash
cd paper_research/code/eg6_density_morph
python eg6_density_morph.py
```

All outputs generated in `outputs/` subfolder.
"""

with open(output_path / ".." / "README.md", "w") as f:
    f.write(readme_content)
print("  Saved: README.md")

# %%
"""
## Complete
"""

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)
print(f"\nTotal cities analyzed: {n_total}")
print(f"Cities profiled: {n_profiled}")
print(f"Node clusters: {N_NODE_CLUSTERS}")
print(f"Morphology features: {len(MORPH_COLUMNS)}")
print(f"Nodes sampled per city: max({MIN_SAMPLE_NODES:,}, {SAMPLE_FRACTION * 100:.0f}% of total)")
print(f"\nOutputs saved to: {output_path}")
