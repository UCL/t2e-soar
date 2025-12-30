# %% [markdown]
"""
# Urban Density and Building Morphology Patterns

Analyzes relationships between population density and building morphology across
European cities to reveal typological patterns in urban form.

## Research Question
How do morphology patterns vary across cities? Do cities follow consistent form
relationships, or exhibit distinct typologies? Can we characterize cities by their
mix of morphological neighborhood types?

## Approach
Rather than aggregating to city means (which loses within-city heterogeneity), we:
1. Cluster NODES by 8 morphology features (log-transformed, standardized)
2. Correlate clusters back to original features for interpretability
3. Characterize clusters by external variables (density, network density, mixed uses)
4. Characterize each CITY by proportions of node types
5. Hierarchically cluster countries by their composition vectors

This reveals whether cities are homogeneous or mixed in their morphological makeup.

## Clustering Method
We use BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies):
- Memory-efficient O(n) complexity scales to 1M+ nodes
- Builds CF-Tree structure for efficient hierarchical clustering
- Final clustering uses AgglomerativeClustering (Ward linkage) on subclusters
- Parameters: threshold=0.5, branching_factor=50

## Feature Selection (8 features covering 6 urban form dimensions)
1. **Density**: Building Count, Block Count
2. **Verticality**: Mean Height (central tendency), Height MAD (variation)
3. **Scale**: Building Area
4. **Form Complexity**: Fractal Dimension
5. **Aggregation**: Block Coverage Ratio, Shared Walls

All features at 200m scale with weighted neighborhood averages (_wt suffix).

## Steps
1. Load city boundaries
2. Load and sample morphology metrics at 200m scale
3a. (Optional) Evaluate optimal number of clusters using BIRCH
3b. Cluster nodes using BIRCH with log-transformed features
3c. Characterize clusters by external variables
4. Correlate clusters to original features
5. (Optional) Extract satellite imagery for cluster representatives
6. Compute city/country profiles (% nodes in each cluster)
7. Visualize cluster profiles (radar plots, bar charts, correlations)
8. Visualize country profiles (hierarchical heatmap, stacked bars)
9. Visualize city profiles using contrasting cluster pairs
10. Export results
11. Generate README report

## Key Outputs
- **cluster_radar_profiles.png**: Individual radar plots per cluster
- **cluster_external_rankings.png**: Clusters ranked by density, network density, mixed uses
- **cluster_feature_correlations.png**: Cluster-feature correlation heatmap
- **country_profile_heatmap.png**: Hierarchically ordered country compositions
- **country_composition_stacked.png**: Countries as stacked bars
- **city_profiles_clusters.png**: Cities plotted by contrasting cluster proportions
- **country_morphology_profiles.csv**: Per-country cluster proportions
- **node_cluster_summary.csv**: Node cluster characteristics
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
from pyproj import Transformer
from scipy.cluster.hierarchy import leaves_list, linkage
from sklearn.cluster import Birch
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

"""
## Configuration
"""
# Morphology metrics at 400m scale (using weighted versions)
AVAILABLE_MORPH_DATA = {
    "cc_area_median_{d}_nw": "Building Area Median",
    "cc_area_mad_{d}_nw": "Building Area MAD",
    "cc_mean_height_median_{d}_nw": "Mean Height Median",
    "cc_mean_height_mad_{d}_nw": "Mean Height MAD",
    "cc_perimeter_median_{d}_nw": "Perimeter Median",
    "cc_perimeter_mad_{d}_nw": "Perimeter MAD",
    "cc_compactness_median_{d}_nw": "Compactness Median",
    "cc_compactness_mad_{d}_nw": "Compactness MAD",
    "cc_orientation_median_{d}_nw": "Orientation Median",
    "cc_orientation_mad_{d}_nw": "Orientation MAD",
    "cc_volume_median_{d}_nw": "Volume Median",
    "cc_volume_mad_{d}_nw": "Volume MAD",
    "cc_floor_area_ratio_median_{d}_nw": "Floor Area Ratio Median",
    "cc_floor_area_ratio_mad_{d}_nw": "Floor Area Ratio MAD",
    "cc_form_factor_median_{d}_nw": "Form Factor Median",
    "cc_form_factor_mad_{d}_nw": "Form Factor MAD",
    "cc_corners_median_{d}_nw": "Corners Median",
    "cc_corners_mad_{d}_nw": "Corners MAD",
    "cc_shape_index_median_{d}_nw": "Shape Index Median",
    "cc_shape_index_mad_{d}_nw": "Shape Index MAD",
    "cc_shared_walls_median_{d}_nw": "Shared Walls Median",
    "cc_shared_walls_mad_{d}_nw": "Shared Walls MAD",
    "cc_fractal_dimension_median_{d}_nw": "Fractal Dimension Median",
    "cc_fractal_dimension_mad_{d}_nw": "Fractal Dimension MAD",
    "cc_building_{d}_nw": "Building Count",
    "cc_block_area_median_{d}_nw": "Block Area Median",
    "cc_block_area_mad_{d}_nw": "Block Area MAD",
    "cc_block_perimeter_median_{d}_nw": "Block Perimeter Median",
    "cc_block_perimeter_mad_{d}_nw": "Block Perimeter MAD",
    "cc_block_compactness_median_{d}_nw": "Block Compactness Median",
    "cc_block_compactness_mad_{d}_nw": "Block Compactness MAD",
    "cc_block_orientation_median_{d}_nw": "Block Orientation Median",
    "cc_block_orientation_mad_{d}_nw": "Block Orientation MAD",
    "cc_block_covered_ratio_median_{d}_nw": "Block Covered Ratio Median",
    "cc_block_covered_ratio_mad_{d}_nw": "Block Covered Ratio MAD",
    "cc_block_{d}_nw": "Block Count",
}


# --- FEATURE SELECTION FOR CLUSTERING ---
# Comprehensive set covering 6 urban form dimensions:
# - Density: Building Count, Block Count
# - Verticality: Mean Height (central tendency), Height MAD (variation)
# - Scale: Building Area
# - Form Complexity: Fractal Dimension
# - Aggregation: Block Coverage, Shared Walls
MORPH_COLS = [
    "cc_building_200_wt",  # Density (count)
    "cc_block_200_wt",  # Density (network grain)
    "cc_mean_height_median_200_wt",  # Verticality (central tendency)
    "cc_mean_height_mad_200_wt",  # Verticality (variation) - uniform vs mixed skyline
    "cc_area_median_200_wt",  # Scale (building footprint)
    "cc_fractal_dimension_median_200_wt",  # Form complexity - simple vs irregular
    "cc_block_covered_ratio_median_200_wt",  # Aggregation (coverage)
    "cc_shared_walls_median_200_wt",  # Aggregation (attachment)
]
MORPH_COL_NAMES = [
    "Building Count",
    "Block Count",
    "Mean Height",
    "Height Variation",
    "Building Area",
    "Fractal Dimension",
    "Block Coverage",
    "Shared Walls",
]
MORPH_LABELS = dict(zip(MORPH_COLS, MORPH_COL_NAMES, strict=True))

CHARACTERISATION_COLS = [
    "density",  # Population density
    "cc_beta_1200",  # Density of street network (network density)
    "cc_hill_q0_200_wt",  # Hill Number q=0 (landuse richness)
]

# Minimum nodes per city for reliable statistics
MIN_NODES = 100

# Number of node clusters for typology
# The script will evaluate k=2 to k=10 and show metrics to help choose optimal k
# Adjust this value based on cluster_evaluation.png and cluster_dendrogram.png
# Consider: silhouette score, elbow point, dendrogram structure, and interpretability
N_NODE_CLUSTERS = 8
COMPUTE_CLUSTER_SCORES = True  # Set to True to compute scores for all k (slower)
CLUSTER_EVAL_N = 15  # Number of clusters to evaluate for optimal k
CLUSTER_EVAL_SAMPLES = 20000  # Sample size for evaluation (balance speed vs accuracy)

# Sample size per city for node clustering: max(5000, 25% of total nodes)
MIN_SAMPLE_NODES = 5000
SAMPLE_FRACTION = 0.25
MAX_TOTAL_NODES = 200000  # Cap total nodes to prevent memory issues

# BIRCH clustering parameters (shared between evaluation and actual clustering)
BIRCH_THRESHOLD = 0.5  # Higher = fewer subclusters = less memory (range: 0.1-1.0)
BIRCH_BRANCHING_FACTOR = 50  # Number of subclusters in each node

# Set to True to download satellite imagery for cluster representative areas
DOWNLOAD_SATELLITE = True  # Set to True to download satellite imagery

# Configuration paths
BOUNDS_PATH = "temp/datasets/boundaries.gpkg"
METRICS_DIR = "temp/cities_data/processed"
OUTPUT_DIR = "paper_research/code/eg6_density_morph/outputs"
TEMP_DIR = "temp/egs/eg6_density_morph"

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
load_cols = MORPH_COLS + CHARACTERISATION_COLS

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

    # Debug: check if metrics directory exists
    if not metrics_dir.exists():
        print(f"  WARNING: Metrics directory does not exist: {metrics_dir}")
    else:
        print(f"  Metrics directory exists: {metrics_dir}")
        # Count how many metrics files exist
        metrics_files = list(metrics_dir.glob("metrics_*.gpkg"))
        print(f"  Found {len(metrics_files)} metrics files")

    for idx, row in tqdm(bounds_gdf.iterrows(), total=len(bounds_gdf), desc="Loading cities"):
        bounds_fid = row.name if isinstance(row.name, int) else idx
        city_label = row.get("label", str(bounds_fid))
        country = row.get("country", "Unknown")

        # Load metrics file for this city
        metrics_file = metrics_dir / f"metrics_{bounds_fid}.gpkg"
        if not metrics_file.exists():
            if idx < 3:  # Print first few missing files
                print(f"    Skipping {city_label}: metrics file not found at {metrics_file}")
            continue

        try:
            # Load with geopandas to get geometry, then extract coordinates
            gdf = gpd.read_file(metrics_file, columns=MORPH_COLS + CHARACTERISATION_COLS, layer="streets")
            # Doublecheck geoms are dropped if outside boundary
            gdf = gdf[gdf.geometry.within(row.geometry)]
            if len(gdf) == 0:
                logger.info(f"    No valid nodes in {city_label}, skipping")
                continue

            # Check we have all required columns
            if not all(col in gdf.columns for col in load_cols):
                logger.info(f"    Missing required columns in {city_label}, skipping")
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
                logger.info(f"    Not enough valid nodes in {city_label} ({len(gdf)}), skipping")
                continue

            n_total_nodes = len(gdf)

            # Sample nodes for clustering: max(MIN_SAMPLE_NODES, SAMPLE_FRACTION of nodes)
            n_sample = max(MIN_SAMPLE_NODES, int(n_total_nodes * SAMPLE_FRACTION))
            sample_gdf = gdf.sample(n=n_sample, random_state=42) if n_sample < n_total_nodes else gdf

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
                    "density_mean": gdf["density"].mean(),
                    "coverage_mean": gdf["cc_block_covered_ratio_median_200_wt"].mean()
                    if "cc_block_covered_ratio_median_200_wt" in gdf.columns
                    else np.nan,
                    "height_mean": gdf["cc_mean_height_median_200_wt"].mean()
                    if "cc_mean_height_median_200_wt" in gdf.columns
                    else np.nan,
                }
            )

        except Exception as e:
            # Print first few errors to diagnose
            if len(all_nodes) < 3:
                print(f"    Error loading {city_label} ({metrics_file.name}): {e}")
            continue

    print(f"  Successfully loaded {len(all_nodes)} cities")

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
print("\nSTEP 3: Clustering nodes by morphology profile")

# Prepare node features for clustering
node_features = nodes_df[MORPH_COLS].copy()

# Handle missing values and infinities
node_features = node_features.replace([np.inf, -np.inf], np.nan)
valid_nodes = node_features.notna().all(axis=1)
node_features_clean = node_features[valid_nodes].copy()
# Always align nodes_df_clean to node_features_clean index
nodes_df_clean = nodes_df.loc[node_features_clean.index].copy()

print(f"  Valid nodes: {len(node_features_clean)}")

# Apply log transformation to handle skewed distributions
# (common in urban morphology data: area, volume, density)
print("  Applying log transformation to features...")
node_features_log = node_features_clean.apply(np.log1p)  # Keep as DataFrame for consistent indexing

# Standardize features (zero mean, unit variance)
scaler = StandardScaler()
X_nodes_scaled = scaler.fit_transform(node_features_log)
print(f"  Using all {len(MORPH_COLS)} morphology features (log-transformed) for clustering")

# %%
"""
## Step 3a: Determine Optimal Number of Clusters
"""
if COMPUTE_CLUSTER_SCORES:
    print("\n  Determining optimal number of clusters using BIRCH...")

    # For computational efficiency, use a smaller sample for cluster evaluation
    eval_sample_size = min(CLUSTER_EVAL_SAMPLES, len(X_nodes_scaled))
    np.random.seed(42)
    eval_idx = np.random.choice(len(X_nodes_scaled), eval_sample_size, replace=False)
    X_eval = X_nodes_scaled[eval_idx]

    # Test different numbers of clusters
    k_range = range(2, CLUSTER_EVAL_N + 1)
    silhouettes = []
    calinski_scores = []
    davies_bouldin_scores = []

    print(f"  Evaluating k={k_range.start} to k={k_range.stop - 1} on {eval_sample_size:,} samples...")

    for k in k_range:
        # Use BIRCH with same parameters as actual clustering
        birch_eval = Birch(
            n_clusters=k,
            threshold=BIRCH_THRESHOLD,
            branching_factor=BIRCH_BRANCHING_FACTOR,
        )
        labels = birch_eval.fit_predict(X_eval)

        # Compute metrics (no inertia for BIRCH since it's not based on centroids)
        silhouette = silhouette_score(X_eval, labels, sample_size=min(10000, len(X_eval)))
        calinski = calinski_harabasz_score(X_eval, labels)
        davies_bouldin = davies_bouldin_score(X_eval, labels)

        silhouettes.append(silhouette)
        calinski_scores.append(calinski)
        davies_bouldin_scores.append(davies_bouldin)

        print(
            f"    k={k}: Silhouette={silhouette:.3f}, "
            f"Calinski-Harabasz={calinski:.1f}, Davies-Bouldin={davies_bouldin:.3f}"
        )

    # Plot cluster evaluation metrics (3 plots instead of 4, no elbow/inertia)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Silhouette score (higher is better, range [-1, 1])
    ax = axes[0]
    ax.plot(k_range, silhouettes, "go-", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax.set_ylabel("Silhouette Score", fontsize=11)
    ax.set_title("Silhouette Score\n(Higher is better, measures cluster separation)", fontsize=12)
    ax.axvline(x=N_NODE_CLUSTERS, color="r", linestyle="--", label=f"Selected k={N_NODE_CLUSTERS}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Calinski-Harabasz score (higher is better)
    ax = axes[1]
    ax.plot(k_range, calinski_scores, "mo-", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax.set_ylabel("Calinski-Harabasz Score", fontsize=11)
    ax.set_title("Calinski-Harabasz Score\n(Higher is better, ratio of between/within cluster variance)", fontsize=12)
    ax.axvline(x=N_NODE_CLUSTERS, color="r", linestyle="--", label=f"Selected k={N_NODE_CLUSTERS}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Davies-Bouldin score (lower is better)
    ax = axes[2]
    ax.plot(k_range, davies_bouldin_scores, "co-", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax.set_ylabel("Davies-Bouldin Score", fontsize=11)
    ax.set_title("Davies-Bouldin Score\n(Lower is better, measures cluster similarity)", fontsize=12)
    ax.axvline(x=N_NODE_CLUSTERS, color="r", linestyle="--", label=f"Selected k={N_NODE_CLUSTERS}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"BIRCH Cluster Evaluation Metrics\n(Red line shows selected k={N_NODE_CLUSTERS})", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path / "cluster_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: cluster_evaluation.png")

    # Provide recommendation based on metrics
    best_silhouette_k = list(k_range)[np.argmax(silhouettes)]
    best_calinski_k = list(k_range)[np.argmax(calinski_scores)]
    best_davies_bouldin_k = list(k_range)[np.argmin(davies_bouldin_scores)]

    print("\n  Metric-based recommendations:")
    print(f"    Best Silhouette score: k={best_silhouette_k} ({max(silhouettes):.3f})")
    print(f"    Best Calinski-Harabasz score: k={best_calinski_k} ({max(calinski_scores):.1f})")
    print(f"    Best Davies-Bouldin score: k={best_davies_bouldin_k} ({min(davies_bouldin_scores):.3f})")
    print(f"    Using: k={N_NODE_CLUSTERS} (adjust N_NODE_CLUSTERS in config if needed)")

# %%
"""
## Step 3b: Fit Hierarchical Clustering
"""
# BIRCH is used as the default - it's hierarchical, memory-efficient, and scales well
print(f"  Fitting BIRCH clustering with {N_NODE_CLUSTERS} clusters...")
print(f"  Dataset size: {len(X_nodes_scaled):,} nodes, {len(MORPH_COLS)} features")

# Remove any remaining NaN or inf values before clustering
nan_mask = np.isnan(X_nodes_scaled).any(axis=1) | np.isinf(X_nodes_scaled).any(axis=1)
if nan_mask.any():
    print(f"  Removing {nan_mask.sum()} rows with NaN/inf values...")
    valid_indices = np.where(~nan_mask)[0]
    X_nodes_scaled = X_nodes_scaled[valid_indices]
    nodes_df_clean = nodes_df_clean.iloc[valid_indices].reset_index(drop=True)
    node_features_clean = node_features_clean.iloc[valid_indices].reset_index(drop=True)
    node_features_log = node_features_log.iloc[valid_indices].reset_index(drop=True)
    print(f"  Remaining nodes: {len(X_nodes_scaled):,}")

# Apply overall cap on total nodes if needed
if len(X_nodes_scaled) > MAX_TOTAL_NODES:
    print(f"  Sampling {MAX_TOTAL_NODES:,} nodes from {len(X_nodes_scaled):,} for memory efficiency...")
    np.random.seed(42)
    sample_idx = np.random.choice(len(X_nodes_scaled), MAX_TOTAL_NODES, replace=False)

    # Verify indices are valid before sampling
    assert len(nodes_df_clean) == len(X_nodes_scaled), (
        f"Length mismatch: df={len(nodes_df_clean)}, array={len(X_nodes_scaled)}"
    )
    assert sample_idx.max() < len(nodes_df_clean), (
        f"Index out of bounds: max={sample_idx.max()}, len={len(nodes_df_clean)}"
    )

    X_nodes_final = X_nodes_scaled[sample_idx]
    nodes_df_final = nodes_df_clean.iloc[sample_idx].reset_index(drop=True)
    node_features_final = node_features_clean.iloc[sample_idx].reset_index(drop=True)
    node_features_log_final = node_features_log.iloc[sample_idx].reset_index(drop=True)
else:
    X_nodes_final = X_nodes_scaled
    nodes_df_final = nodes_df_clean.copy()
    node_features_final = node_features_clean.copy()
    node_features_log_final = node_features_log.copy()

print(f"  Clustering {len(X_nodes_final):,} nodes...")

# BIRCH: Balanced Iterative Reducing and Clustering using Hierarchies
# - Builds a CF-tree (hierarchical structure)
# - Memory efficient O(n)
# - Final step uses AgglomerativeClustering on subclusters
birch = Birch(
    n_clusters=N_NODE_CLUSTERS,
    threshold=BIRCH_THRESHOLD,
    branching_factor=BIRCH_BRANCHING_FACTOR,
)
nodes_df_final["node_cluster"] = birch.fit_predict(X_nodes_final)
print(f"  BIRCH complete: {len(birch.subcluster_centers_)} subclusters → {N_NODE_CLUSTERS} clusters")

# Update references to use final sampled data
X_nodes_scaled = X_nodes_final
nodes_df_clean = nodes_df_final
node_features_clean = node_features_final

# Verify alignment
assert len(nodes_df_clean) == len(node_features_clean), "Length mismatch after clustering"
print(f"  Verified: {len(nodes_df_clean)} nodes clustered")

print("\n  Node cluster characteristics:")
cluster_profiles = []
for c in range(N_NODE_CLUSTERS):
    cluster_mask = (nodes_df_clean["node_cluster"] == c).values
    cluster_nodes = node_features_clean[cluster_mask]
    n_nodes = len(cluster_nodes)

    profile = {"cluster": c, "n_nodes": n_nodes, "pct_nodes": n_nodes / len(nodes_df_clean) * 100}

    # Compute means for selected features
    for col in MORPH_COLS:
        col_short = col.replace("cc_", "").replace("_200_wt", "").replace("_", " ").title()
        profile[f"{col_short} Mean"] = cluster_nodes[col].mean()

    cluster_profiles.append(profile)

    # Simple characterization based on key features
    bldg_count = cluster_nodes["cc_building_200_wt"].mean()
    height = cluster_nodes["cc_mean_height_median_200_wt"].mean()
    shared = cluster_nodes["cc_shared_walls_median_200_wt"].mean()

    bldg_cat = "dense" if bldg_count > node_features_clean["cc_building_200_wt"].median() else "sparse"
    height_cat = "tall" if height > node_features_clean["cc_mean_height_median_200_wt"].median() else "low"
    attach_cat = "attached" if shared > node_features_clean["cc_shared_walls_median_200_wt"].median() else "detached"

    print(
        f"    Cluster {c + 1}: {n_nodes:,} nodes ({profile['pct_nodes']:.1f}%) - {bldg_cat}, {height_cat}, {attach_cat}"
    )

cluster_profiles_df = pd.DataFrame(cluster_profiles)

# %%
"""
## Step 3c: Characterize Clusters by External Variables
Computes mean values for population density, network density, and mixed uses for each cluster.
Rankings are visualized in a separate plot.
"""
print("\n  Computing external characteristics for each cluster (density, network density, mixed uses)...")

# Compute global percentile caps to handle extreme outliers
density_cap = np.nanpercentile(nodes_df_clean["density"].replace([np.inf, -np.inf], np.nan), 99.9)
print(f"  Capping density at 99.9th percentile: {density_cap:.0f}")

# Compute external characteristics for each cluster
CLUSTER_EXTERNAL_CHARS = {}
cluster_char_data = []

for c in range(N_NODE_CLUSTERS):
    cluster_mask = nodes_df_clean["node_cluster"] == c
    cluster_data = nodes_df_clean[cluster_mask]

    # Compute means for characterization variables, filtering out inf values and capping extremes
    density_values = cluster_data["density"].replace([np.inf, -np.inf], np.nan).clip(upper=density_cap)
    walkability_values = cluster_data["cc_beta_1200"].replace([np.inf, -np.inf], np.nan)
    mixed_use_values = cluster_data["cc_hill_q0_200_wt"].replace([np.inf, -np.inf], np.nan)

    density_mean = density_values.mean()
    walkability_mean = walkability_values.mean()
    mixed_use_mean = mixed_use_values.mean()

    CLUSTER_EXTERNAL_CHARS[c] = {
        "density_mean": density_mean,
        "walkability_mean": walkability_mean,
        "mixed_use_mean": mixed_use_mean,
    }

    cluster_char_data.append(
        {
            "cluster": c + 1,  # Store as 1-based in CSV
            "density_mean": density_mean,
            "walkability_mean": walkability_mean,
            "mixed_use_mean": mixed_use_mean,
        }
    )

print(f"  Computed external metrics for {N_NODE_CLUSTERS} clusters")


# Save external characterization
cluster_char_df = pd.DataFrame(cluster_char_data)
cluster_char_df.to_csv(output_path / "cluster_external_characterization.csv", index=False)
print("  Saved: cluster_external_characterization.csv")

# --- Visualization: Cluster ranking by external variables ---
print("\n  Creating cluster ranking visualization by external variables...")

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Sort clusters by each metric for ranking
density_ranking = sorted(range(N_NODE_CLUSTERS), key=lambda c: CLUSTER_EXTERNAL_CHARS[c]["density_mean"])
walkability_ranking = sorted(range(N_NODE_CLUSTERS), key=lambda c: CLUSTER_EXTERNAL_CHARS[c]["walkability_mean"])
mixed_use_ranking = sorted(range(N_NODE_CLUSTERS), key=lambda c: CLUSTER_EXTERNAL_CHARS[c]["mixed_use_mean"])

# Plot 1: Population Density
ax = axes[0]
colors_density = plt.cm.Reds(np.linspace(0.3, 0.9, N_NODE_CLUSTERS))
sorted_density = [CLUSTER_EXTERNAL_CHARS[c]["density_mean"] for c in density_ranking]
sorted_labels_density = [c + 1 for c in density_ranking]
bars = ax.barh(range(N_NODE_CLUSTERS), sorted_density, color=colors_density)
ax.set_yticks(range(N_NODE_CLUSTERS))
ax.set_yticklabels([f"Cluster {label}" for label in sorted_labels_density], fontsize=9)
ax.set_xlabel("Population Density (people/km²)", fontsize=10, weight="bold")
ax.set_title("Clusters Ranked by\nPopulation Density", fontsize=11, weight="bold")
ax.grid(True, alpha=0.3, axis="x")
# Put labels inside bars on the right side
for i, (c, val) in enumerate(zip(density_ranking, sorted_density)):
    if not np.isnan(val) and val > 0:
        ax.text(val * 0.95, i, f"{val:.0f}", va="center", ha="right", fontsize=8, color="white", weight="bold")

# Plot 2: Network Density (Street Network Density)
ax = axes[1]
colors_walk = plt.cm.Greens(np.linspace(0.3, 0.9, N_NODE_CLUSTERS))
sorted_walkability = [CLUSTER_EXTERNAL_CHARS[c]["walkability_mean"] for c in walkability_ranking]
sorted_labels_walk = [c + 1 for c in walkability_ranking]
bars = ax.barh(range(N_NODE_CLUSTERS), sorted_walkability, color=colors_walk)
ax.set_yticks(range(N_NODE_CLUSTERS))
ax.set_yticklabels([f"Cluster {label}" for label in sorted_labels_walk], fontsize=9)
ax.set_xlabel("Street Network Density (beta)", fontsize=10, weight="bold")
ax.set_title("Clusters Ranked by\nNetwork Density", fontsize=11, weight="bold")
ax.grid(True, alpha=0.3, axis="x")
# Put labels inside bars on the right side
for i, (c, val) in enumerate(zip(walkability_ranking, sorted_walkability)):
    if not np.isnan(val) and val > 0:
        ax.text(val * 0.95, i, f"{val:.2f}", va="center", ha="right", fontsize=8, color="white", weight="bold")

# Plot 3: Mixed Uses (Landuse Richness)
ax = axes[2]
colors_mixed = plt.cm.Blues(np.linspace(0.3, 0.9, N_NODE_CLUSTERS))
sorted_mixed_use = [CLUSTER_EXTERNAL_CHARS[c]["mixed_use_mean"] for c in mixed_use_ranking]
sorted_labels_mixed = [c + 1 for c in mixed_use_ranking]
bars = ax.barh(range(N_NODE_CLUSTERS), sorted_mixed_use, color=colors_mixed)
ax.set_yticks(range(N_NODE_CLUSTERS))
ax.set_yticklabels([f"Cluster {label}" for label in sorted_labels_mixed], fontsize=9)
ax.set_xlabel("Landuse Richness (Hill q0)", fontsize=10, weight="bold")
ax.set_title("Clusters Ranked by\nMixed Uses", fontsize=11, weight="bold")
ax.grid(True, alpha=0.3, axis="x")
# Put labels inside bars on the right side
for i, (c, val) in enumerate(zip(mixed_use_ranking, sorted_mixed_use)):
    if not np.isnan(val) and val > 0:
        ax.text(val * 0.95, i, f"{val:.2f}", va="center", ha="right", fontsize=8, color="white", weight="bold")

plt.suptitle("Cluster Rankings by External Characteristics", fontsize=14, weight="bold", y=1.02)
plt.tight_layout()
plt.savefig(output_path / "cluster_external_rankings.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: cluster_external_rankings.png")

# %%
"""
## Step 4: Correlate Clusters to Original Features
"""
print("\n  Computing cluster-feature correlations...")

# Simple correlation summary for each cluster
cluster_feature_corr = []
for c in range(N_NODE_CLUSTERS):
    cluster_binary = (nodes_df_clean["node_cluster"] == c).astype(int).values
    corr_row = {"cluster": c}
    for col in MORPH_COLS:
        corr = np.corrcoef(node_features_clean[col].values, cluster_binary)[0, 1]
        col_short = col.replace("cc_", "").replace("_200_wt", "").replace("_", " ").title()
        corr_row[col_short] = corr
    cluster_feature_corr.append(corr_row)

cluster_feature_corr_df = pd.DataFrame(cluster_feature_corr).set_index("cluster")
cluster_feature_corr_df.to_csv(output_path / "cluster_feature_correlations.csv")
print("  Saved: cluster_feature_correlations.csv")

# %%
"""
## Step 5: Extract Representative Satellite Images for Each Cluster
Set DOWNLOAD_SATELLITE = False to skip
"""
print("\nSTEP 5: Finding representative locations for each cluster")

# Find representative node for each cluster using multiple criteria:
# 1. Medoid: actual node closest to cluster centroid in feature space (most "typical")
# 2. Prefer nodes that are geographically clustered (surrounded by same cluster type)
# 3. This ensures satellite imagery shows coherent urban form, not isolated nodes

# Spatial neighborhood radius for computing local cluster density (in meters, EPSG:3035)
SPATIAL_RADIUS = 600  # 600m radius to check for neighboring same-cluster nodes

# Store ranked candidates for each cluster (for fallback if imagery unavailable)
cluster_candidates = {}

for c in range(N_NODE_CLUSTERS):
    # Get nodes in this cluster
    cluster_mask = nodes_df_clean["node_cluster"] == c
    cluster_features = X_nodes_scaled[cluster_mask]
    cluster_nodes_df = nodes_df_clean[cluster_mask].reset_index(drop=True)

    # Get coordinates for spatial analysis
    cluster_coords = cluster_nodes_df[["x", "y"]].values

    # Find centroid in feature space
    centroid = cluster_features.mean(axis=0)

    # Compute distance to centroid for all nodes
    feature_distances = np.linalg.norm(cluster_features - centroid, axis=1)

    # Get top 200 candidates (closest to centroid in feature space)
    n_candidates = min(200, len(feature_distances))
    candidate_indices = np.argsort(feature_distances)[:n_candidates]

    # Score ALL candidates by local geographic cluster density
    # Count how many same-cluster neighbors within SPATIAL_RADIUS
    candidate_scores = []

    for idx in candidate_indices:
        node_x, node_y = cluster_coords[idx]

        # Count same-cluster neighbors within spatial radius
        spatial_dists = np.sqrt((cluster_coords[:, 0] - node_x) ** 2 + (cluster_coords[:, 1] - node_y) ** 2)
        neighbor_count = np.sum(spatial_dists <= SPATIAL_RADIUS) - 1  # Exclude self

        candidate_scores.append((idx, neighbor_count, feature_distances[idx]))

    # Sort by neighbor count (descending), then by feature distance (ascending) as tiebreaker
    candidate_scores.sort(key=lambda x: (-x[1], x[2]))

    # Store all ranked candidates for this cluster
    cluster_candidates[c] = [
        (cluster_nodes_df.iloc[idx], neighbor_count) for idx, neighbor_count, _ in candidate_scores
    ]

    # Best candidate
    best_node, best_neighbor_count = cluster_candidates[c][0]

    print(
        f"  Cluster {c + 1} best candidate: {best_node['city_label']}, {best_node['country']} ({best_neighbor_count} same-cluster neighbors within 600m)"
    )

# Build cluster_reps from best candidates (will be updated if imagery fails)
cluster_reps = []
for c in range(N_NODE_CLUSTERS):
    rep_node, neighbor_count = cluster_candidates[c][0]
    cluster_reps.append(
        {
            "cluster": c,
            "bounds_fid": rep_node["bounds_fid"],
            "city_label": rep_node["city_label"],
            "country": rep_node["country"],
            "x": rep_node["x"],
            "y": rep_node["y"],
            "neighbors_within_600m": neighbor_count,
        }
    )

# Save representative locations for manual lookup or later satellite download
reps_df = pd.DataFrame(cluster_reps)
reps_df.to_csv(output_path / "cluster_representatives.csv", index=False)
print("  Saved cluster representatives to cluster_representatives.csv")

if DOWNLOAD_SATELLITE:
    print("\n  Downloading satellite tile grids (5x5 per location)...")

    def latlon_to_tile(lat, lon, zoom):
        """Convert lat/lon to tile coordinates."""
        lat_rad = math.radians(lat)
        n = 2.0**zoom
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return x, y

    def check_image_has_data(img):
        """Check if image has actual satellite data (not empty/uniform)."""
        # Convert to grayscale and check variance
        gray = img.convert("L")
        pixels = list(gray.getdata())
        if len(set(pixels)) < 10:  # Almost uniform color = no data
            return False
        # Check if it's mostly a single color (empty ocean/no coverage)
        from collections import Counter

        pixel_counts = Counter(pixels)
        most_common_pct = pixel_counts.most_common(1)[0][1] / len(pixels)
        if most_common_pct > 0.9:  # >90% same color = likely no data
            return False
        return True

    def download_satellite_grid(x, y, zoom, grid_size, half_grid):
        """Download satellite grid and return (composite, has_data)."""
        transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(x, y)

        x_tile, y_tile = latlon_to_tile(lat, lon, zoom)

        tiles = []
        failed_tiles = 0
        for dy in range(-half_grid, half_grid + 1):
            row = []
            for dx in range(-half_grid, half_grid + 1):
                tile_x = x_tile + dx
                tile_y = y_tile + dy

                tile_url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{tile_y}/{tile_x}"

                try:
                    response = urllib.request.urlopen(tile_url, timeout=5)
                    img_data = response.read()
                    img = Image.open(BytesIO(img_data))
                    row.append(img)
                except Exception:
                    failed_tiles += 1
                    row.append(Image.new("RGB", (256, 256), color="gray"))

            tiles.append(row)

        # Stitch tiles together
        tile_width, tile_height = 256, 256
        grid_width = grid_size * tile_width
        grid_height = grid_size * tile_height
        composite = Image.new("RGB", (grid_width, grid_height))

        for row_idx, row in enumerate(tiles):
            for col_idx, tile_img in enumerate(row):
                x_offset = col_idx * tile_width
                y_offset = row_idx * tile_height
                composite.paste(tile_img, (x_offset, y_offset))

        # Check if image has real data
        has_data = check_image_has_data(composite) and failed_tiles < grid_size * grid_size // 2

        return composite, has_data, lat, lon

    zoom = 19  # High-resolution street detail
    grid_size = 5  # 5x5 grid of tiles
    half_grid = grid_size // 2  # 2 tiles in each direction from center
    max_candidates_to_try = 10  # Try up to 10 candidates if imagery unavailable

    for c in range(N_NODE_CLUSTERS):
        candidates = cluster_candidates[c]
        success = False

        for attempt, (candidate_node, neighbor_count) in enumerate(candidates[:max_candidates_to_try]):
            x, y = candidate_node["x"], candidate_node["y"]

            try:
                composite, has_data, lat, lon = download_satellite_grid(x, y, zoom, grid_size, half_grid)

                if has_data:
                    # Save successful image
                    output_file = output_path / f"cluster_{c + 1}_satellite_5x5.jpg"
                    composite.save(output_file, format="JPEG", quality=85)
                    file_size_mb = output_file.stat().st_size / (1024 * 1024)

                    location_info = f"{candidate_node['city_label']}, {candidate_node['country']}"
                    if attempt > 0:
                        print(f"    Cluster {c + 1}: Found valid imagery on attempt {attempt + 1}")
                    print(
                        f"    Cluster {c + 1}: Saved {output_file.name} ({location_info}, {neighbor_count} neighbors, lat={lat:.4f}, lon={lon:.4f})"
                    )

                    # Update cluster_reps with the successful candidate
                    cluster_reps[c] = {
                        "cluster": c,
                        "bounds_fid": candidate_node["bounds_fid"],
                        "city_label": candidate_node["city_label"],
                        "country": candidate_node["country"],
                        "x": x,
                        "y": y,
                        "neighbors_within_600m": neighbor_count,
                    }
                    success = True
                    break
                else:
                    if attempt == 0:
                        print(
                            f"    Cluster {c + 1}: No satellite data at {candidate_node['city_label']}, trying alternatives..."
                        )

            except Exception as e:
                print(f"    Cluster {c + 1}: Error on attempt {attempt + 1} - {e}")

        if not success:
            print(
                f"    Cluster {c + 1}: WARNING - Could not find valid satellite imagery after {max_candidates_to_try} attempts"
            )

    # Update saved CSV with final representatives (may have changed due to imagery fallback)
    reps_df = pd.DataFrame(cluster_reps)
    reps_df.to_csv(output_path / "cluster_representatives.csv", index=False)
    print("  Updated cluster_representatives.csv with final selections")
else:
    print("  Satellite download skipped (set DOWNLOAD_SATELLITE=True to enable)")

# %%
"""
## Step 6: Compute City Profiles (Proportion of Node Types)
"""

print("\nSTEP 6: Computing country profiles from node cluster proportions")

# Compute proportion of nodes in each cluster for each country
country_profiles = []

for country in nodes_df_clean["country"].unique():
    country_nodes = nodes_df_clean[nodes_df_clean["country"] == country]
    n_nodes = len(country_nodes)

    profile = {
        "country": country,
        "n_nodes_sampled": n_nodes,
        "n_cities": country_nodes["city_label"].nunique(),
    }

    # Compute proportion in each cluster
    for c in range(N_NODE_CLUSTERS):
        pct = (country_nodes["node_cluster"] == c).sum() / n_nodes * 100
        profile[f"pct_cluster_{c}"] = pct

    country_profiles.append(profile)

country_profiles_df = pd.DataFrame(country_profiles)

# Compute country-level statistics
country_stats = []
for country in nodes_df_clean["country"].unique():
    country_data = city_df[city_df["country"] == country]
    if len(country_data) > 0:
        country_stats.append(
            {
                "country": country,
                "density_mean": country_data["density_mean"].mean(),
                "coverage_mean": country_data["coverage_mean"].mean(),
            }
        )

country_stats_df = pd.DataFrame(country_stats)
country_profiles_df = country_profiles_df.merge(country_stats_df, on="country", how="left")

print(f"  Computed profiles for {len(country_profiles_df)} countries")

print("\n  Cluster distribution across all nodes:")
for c in range(N_NODE_CLUSTERS):
    total_in_cluster = (nodes_df_clean["node_cluster"] == c).sum()
    pct = total_in_cluster / len(nodes_df_clean) * 100
    print(f"    Cluster {c + 1}: {total_in_cluster:,} nodes ({pct:.1f}%)")

# %%
"""
## Step 7: Node Cluster Profile Visualization
"""
print("\nSTEP 7: Generating node cluster profile visualization")

# Compute cluster means from nodes (recompute to ensure fresh data)
node_cluster_means = node_features_clean.copy()
node_cluster_means["node_cluster"] = nodes_df_clean["node_cluster"]

# Replace inf with NaN in the data BEFORE computing means
for col in MORPH_COLS:
    node_cluster_means[col] = node_cluster_means[col].replace([np.inf, -np.inf], np.nan)

# Compute cluster means manually to handle NaN properly
cluster_means_data = {}
for c in range(N_NODE_CLUSTERS):
    cluster_mask = node_cluster_means["node_cluster"] == c
    cluster_data = node_cluster_means[cluster_mask]
    cluster_means_data[c] = {}
    for col in MORPH_COLS:
        col_values = cluster_data[col].dropna()
        cluster_means_data[c][col] = col_values.mean() if len(col_values) > 0 else 0.0

cluster_means = pd.DataFrame(cluster_means_data).T
cluster_means.index.name = "node_cluster"

# Debug: Print shared walls values
print("  Shared walls cluster means:")
shared_walls_col = "cc_shared_walls_median_200_wt"
for c in range(N_NODE_CLUSTERS):
    print(f"    Cluster {c + 1}: {cluster_means.loc[c, shared_walls_col]:.4f}")

# Standardize cluster means for radar plot (z-scores across clusters)
cluster_means_standardized = (cluster_means - cluster_means.mean()) / cluster_means.std()

# --- 7a: Radar/Spider Plot for Cluster Profiles ---
print("  Creating radar plot of cluster profiles...")

# Set up radar chart
n_features = len(MORPH_COLS)
angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

# Create figure with subplots for each cluster (grid layout)
n_cols = 4
n_rows = math.ceil(N_NODE_CLUSTERS / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5 * n_rows), subplot_kw=dict(polar=True))
axes = axes.flatten()

cluster_colors = plt.cm.tab20(np.linspace(0, 1, N_NODE_CLUSTERS))

for c in range(N_NODE_CLUSTERS):
    ax = axes[c]
    values = cluster_means_standardized.loc[c].values.tolist()
    values += values[:1]  # Complete the circle

    ax.plot(angles, values, "o-", linewidth=2, color=cluster_colors[c])
    ax.fill(angles, values, alpha=0.25, color=cluster_colors[c])

    # Set feature labels with word wrapping at word boundaries
    feature_labels = []
    for col in MORPH_COLS:
        label = MORPH_LABELS.get(col, col.split("_")[1])
        # Wrap long labels by splitting on spaces or hyphens
        if len(label) > 12:
            words = label.split()
            if len(words) > 1:
                label = "\n".join(words)
        feature_labels.append(label)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_labels, size=7)
    ax.tick_params(axis="x", pad=10)

    # Subtler grid
    ax.grid(True, alpha=0.5, linewidth=0.5)
    ax.set_ylim(-2.5, 2.5)

    # Title with cluster number (1-based)
    n_nodes = cluster_profiles_df[cluster_profiles_df["cluster"] == c]["n_nodes"].iloc[0]
    pct = cluster_profiles_df[cluster_profiles_df["cluster"] == c]["pct_nodes"].iloc[0]
    ax.set_title(
        f"Cluster {c + 1}\n({n_nodes:,} nodes, {pct:.1f}%)",
        size=10,
        pad=25,
        weight="bold",
    )

    # Make radial grid labels slightly bigger
    ax.tick_params(axis="y", labelsize=5, colors="gray")

# Hide unused subplots
for i in range(N_NODE_CLUSTERS, len(axes)):
    axes[i].axis("off")

plt.suptitle("Node Morphology Cluster Profiles (Standardized Features)", fontsize=14, y=1.02)
plt.tight_layout(pad=2.0, h_pad=1.5)
plt.savefig(output_path / "cluster_radar_profiles.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: cluster_radar_profiles.png")

# --- 7c: Original bar charts (kept for detail) ---
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, col in enumerate(MORPH_COLS):
    ax = axes[i]
    label = MORPH_LABELS.get(col, col)
    values = cluster_means[col].values
    bars = ax.bar(range(N_NODE_CLUSTERS), values, color=cluster_colors)
    ax.set_xlabel("Cluster", fontsize=9)
    ax.set_ylabel(label, fontsize=9)
    ax.set_xticks(range(N_NODE_CLUSTERS))
    ax.set_xticklabels([f"{c + 1}" for c in range(N_NODE_CLUSTERS)])  # 1-based cluster labels
    ax.set_title(
        label,
        fontsize=10,
        weight="bold",
    )
    ax.tick_params(labelsize=8)
    # Set y-axis limits based on actual data range with padding (use nanmin/nanmax for safety)
    data_min = np.nanmin(values) if not np.all(np.isnan(values)) else 0
    data_max = np.nanmax(values) if not np.all(np.isnan(values)) else 1
    padding = (data_max - data_min) * 0.15
    ax.set_ylim(max(data_min - padding, 0), data_max + padding)

# Hide unused subplots
for i in range(len(MORPH_COLS), len(axes)):
    axes[i].axis("off")

plt.suptitle("Node Morphology Cluster Profiles (Raw Values)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(output_path / "node_cluster_profiles.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: node_cluster_profiles.png")

# %%
"""
## Step 7d: Cluster-Feature Correlation Heatmap
"""

print("\nSTEP 7d: Generating cluster-feature correlation heatmap")

# Create heatmap showing correlation between clusters and original features
fig, ax = plt.subplots(figsize=(14, 6))

# Prepare data for heatmap (transpose so features are rows)
corr_matrix = cluster_feature_corr_df.T.values
feature_labels = [MORPH_LABELS.get(col, col.replace("cc_", "").replace("_median_100_wt", "")) for col in MORPH_COLS]

im = ax.imshow(corr_matrix, cmap="RdBu_r", aspect="auto", vmin=-0.5, vmax=0.5)

# Use simple 1-based cluster numbering
cluster_labels = [f"Cluster {c + 1}" for c in range(N_NODE_CLUSTERS)]
ax.set_xticks(range(N_NODE_CLUSTERS))
ax.set_xticklabels(cluster_labels, fontsize=9, rotation=45, ha="right")
ax.set_yticks(range(len(feature_labels)))
ax.set_yticklabels(feature_labels, fontsize=9)
ax.set_xlabel("Cluster Type", fontsize=11)
ax.set_ylabel("Morphology Feature", fontsize=11)
ax.set_title(
    "Cluster-Feature Correlations\n(Point-biserial correlation between cluster membership and original features)",
    fontsize=12,
)

# Add correlation values as text
for i in range(len(feature_labels)):
    for j in range(N_NODE_CLUSTERS):
        val = corr_matrix[i, j]
        color = "white" if abs(val) > 0.25 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

plt.colorbar(im, ax=ax, label="Correlation", shrink=0.8)
plt.tight_layout()
plt.savefig(output_path / "cluster_feature_correlations.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: cluster_feature_correlations.png")

# %%
"""
## Step 8: Country Profile Heatmap (Hierarchically Ordered)
"""

print("\nSTEP 8: Generating country profile visualizations")

pct_cols = [f"pct_cluster_{c}" for c in range(N_NODE_CLUSTERS)]

# Remove any countries with missing data before clustering
country_profiles_clean = country_profiles_df.dropna(subset=pct_cols).copy()
print(f"  Countries with complete data: {len(country_profiles_clean)}/{len(country_profiles_df)}")

# --- 8a: Hierarchically cluster countries by their composition vectors ---
print("  Hierarchically clustering countries by morphology composition...")

# Prepare country composition matrix
country_composition = country_profiles_clean[pct_cols].values

# Hierarchical clustering of countries (Ward linkage for balanced clusters)
country_linkage = linkage(country_composition, method="ward")

# Get optimal ordering of countries
country_order = leaves_list(country_linkage)
country_profiles_ordered = country_profiles_clean.iloc[country_order].copy()

# Create cluster labels using 1-based numbering
cluster_labels = [f"Cluster {c + 1}" for c in range(N_NODE_CLUSTERS)]

fig, ax = plt.subplots(figsize=(12, max(8, len(country_profiles_ordered) * 0.15)))
im = ax.imshow(country_profiles_ordered[pct_cols].values, aspect="auto", cmap="YlOrRd")

ax.set_yticks(range(len(country_profiles_ordered)))
ax.set_yticklabels(
    [f"{row['country']} (n={row['n_cities']})" for _, row in country_profiles_ordered.iterrows()], fontsize=9
)
ax.set_xticks(range(N_NODE_CLUSTERS))
ax.set_xticklabels(cluster_labels, fontsize=8, rotation=45, ha="right")
ax.set_xlabel("Node Morphology Cluster Type")
ax.set_ylabel("Country (Hierarchically Ordered)")
ax.set_title(
    "Country Morphology Profiles: Node Proportions by Cluster\n(Countries ordered by compositional similarity)"
)

plt.colorbar(im, ax=ax, label="% of nodes")
plt.tight_layout()
plt.savefig(output_path / "country_profile_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: country_profile_heatmap.png")

# --- 8b: Stacked Bar Chart of Country Compositions ---
print("  Creating stacked bar chart of country compositions...")

# Sort by dominant cluster then by that cluster's percentage
country_composition_plot = country_profiles_clean.copy()
country_composition_plot["dominant_cluster"] = (
    country_composition_plot[pct_cols].idxmax(axis=1).str.replace("pct_cluster_", "").astype(int)
)
country_composition_plot = country_composition_plot.sort_values(
    ["dominant_cluster", "pct_cluster_0"], ascending=[True, False]
)

fig, ax = plt.subplots(figsize=(14, max(6, len(country_composition_plot) * 0.25)))

# Create stacked bars
x = np.arange(len(country_composition_plot))
bottom = np.zeros(len(country_composition_plot))

cluster_colors = plt.cm.tab20(np.linspace(0, 1, N_NODE_CLUSTERS))

for c in range(N_NODE_CLUSTERS):
    values = country_composition_plot[f"pct_cluster_{c}"].values
    ax.barh(x, values, left=bottom, label=f"Cluster {c + 1}", color=cluster_colors[c], height=0.8)
    bottom += values

ax.set_yticks(x)
ax.set_yticklabels([f"{row['country']}" for _, row in country_composition_plot.iterrows()], fontsize=9)
ax.set_xlabel("Percentage of Nodes (%)")
ax.set_ylabel("Country")
ax.set_title("Country Morphology Composition")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, title="Cluster Type")
ax.set_xlim(0, 100)

plt.tight_layout()
plt.savefig(output_path / "country_composition_stacked.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: country_composition_stacked.png")

# %%
"""
## Step 9: City Profiles Using Contrasting Cluster Pairs
Instead of PCA, we select cluster pairs that have the most extreme relationship
with external metrics (density, mixed uses) for more interpretable axes.
"""

print("\nSTEP 9: Generating city profile visualization using contrasting clusters")

# --- 9a: Create city profiles with cluster proportions ---
print("  Computing city-level cluster proportions...")

city_profiles = []
pct_cols = [f"pct_cluster_{c}" for c in range(N_NODE_CLUSTERS)]

for bounds_fid in nodes_df_clean["bounds_fid"].unique():
    city_nodes = nodes_df_clean[nodes_df_clean["bounds_fid"] == bounds_fid]
    if len(city_nodes) < 10:  # Skip cities with too few nodes
        continue

    city_label = city_nodes["city_label"].iloc[0]
    country = city_nodes["country"].iloc[0]
    n_nodes = len(city_nodes)

    profile = {
        "bounds_fid": bounds_fid,
        "city_label": city_label,
        "country": country,
        "n_nodes": n_nodes,
        # Also compute city-level means for external metrics
        "density_mean": city_nodes["density"].mean(),
        "mixed_use_mean": city_nodes["cc_hill_q0_200_wt"].mean(),
        "walkability_mean": city_nodes["cc_beta_1200"].mean(),
    }

    # Compute proportion in each cluster
    for c in range(N_NODE_CLUSTERS):
        pct = (city_nodes["node_cluster"] == c).sum() / n_nodes * 100
        profile[f"pct_cluster_{c}"] = pct

    city_profiles.append(profile)

city_profiles_df = pd.DataFrame(city_profiles)

# --- 9b: Find clusters with highest/lowest external metrics ---
print("\n  Finding clusters with most extreme external metric values...")

# Use the actual mean values from CLUSTER_EXTERNAL_CHARS (computed in Step 3c)
# This is more intuitive than correlation-based selection

# Rank by actual mean density
density_ranking = sorted(range(N_NODE_CLUSTERS), key=lambda c: CLUSTER_EXTERNAL_CHARS[c]["density_mean"])
low_density_cluster = density_ranking[0]  # Lowest mean density
high_density_cluster = density_ranking[-1]  # Highest mean density

# Rank by actual mean mixed use
mixed_use_ranking = sorted(range(N_NODE_CLUSTERS), key=lambda c: CLUSTER_EXTERNAL_CHARS[c]["mixed_use_mean"])
low_mixed_use_cluster = mixed_use_ranking[0]  # Lowest mean mixed use
high_mixed_use_cluster = mixed_use_ranking[-1]  # Highest mean mixed use

print("\n  Selected cluster pairs (based on mean values):")
print(
    f"    Density axis: Cluster {high_density_cluster + 1} (highest density, mean={CLUSTER_EXTERNAL_CHARS[high_density_cluster]['density_mean']:.0f}) vs Cluster {low_density_cluster + 1} (lowest density, mean={CLUSTER_EXTERNAL_CHARS[low_density_cluster]['density_mean']:.0f})"
)
print(
    f"    Mixed use axis: Cluster {high_mixed_use_cluster + 1} (highest mixed use, mean={CLUSTER_EXTERNAL_CHARS[high_mixed_use_cluster]['mixed_use_mean']:.2f}) vs Cluster {low_mixed_use_cluster + 1} (lowest mixed use, mean={CLUSTER_EXTERNAL_CHARS[low_mixed_use_cluster]['mixed_use_mean']:.2f})"
)

# Show available countries and filter to focus set
print(f"\n  Total cities in dataset: {len(city_profiles_df)}")
available_countries = [c for c in city_profiles_df["country"].unique() if c is not None]
print(f"  Available countries: {sorted(available_countries)}")

# Filter to focus countries
focus_countries = ["FR", "NL", "IT", "RO", "PL", "DE", "ES"]
city_focus = city_profiles_df[city_profiles_df["country"].isin(focus_countries)].copy()
print(f"  Cities in focus countries: {len(city_focus)}")

# If no cities match, show all cities
if len(city_focus) == 0:
    print("  WARNING: No cities found in focus countries, showing all cities")
    city_focus = city_profiles_df[city_profiles_df["country"].notna()].copy()
    available_countries = [c for c in city_focus["country"].unique() if c is not None]
    focus_countries = sorted(available_countries)[:7]

# Create color map for countries with high contrast colors
default_colors = [
    "#e41a1c",  # Red
    "#377eb8",  # Blue
    "#4daf4a",  # Green
    "#984ea3",  # Purple
    "#ff7f00",  # Orange
    "#a65628",  # Brown
    "#f781bf",  # Pink
]
country_colors_map = {country: default_colors[i % len(default_colors)] for i, country in enumerate(focus_countries)}

# --- 9c: Create two separate plots for density and mixed uses ---
fig, axes = plt.subplots(1, 2, figsize=(20, 9))

# Plot 1: Highest vs Lowest Density Clusters
ax = axes[0]
for country in focus_countries:
    subset = city_focus[city_focus["country"] == country]
    if len(subset) > 0:
        ax.scatter(
            subset[f"pct_cluster_{high_density_cluster}"],
            subset[f"pct_cluster_{low_density_cluster}"],
            c=country_colors_map[country],
            label=f"{country} ({len(subset)} cities)",
            alpha=0.6,
            s=100,
            edgecolors="black",
            linewidth=0.8,
        )

ax.grid(True, alpha=0.3, linestyle="--")
ax.set_xlabel(
    f"% in Highest Density Cluster ({high_density_cluster + 1})",
    fontsize=11,
    weight="bold",
)
ax.set_ylabel(
    f"% in Lowest Density Cluster ({low_density_cluster + 1})",
    fontsize=11,
    weight="bold",
)
ax.set_title("City Morphology: Highest vs Lowest Density Clusters", fontsize=14, weight="bold")
ax.legend(loc="best", fontsize=10, framealpha=0.9)

# Add diagonal reference line using actual data range
max_x = city_focus[f"pct_cluster_{high_density_cluster}"].max()
max_y = city_focus[f"pct_cluster_{low_density_cluster}"].max()
max_val = max(max_x, max_y) * 1.02  # Tight padding for diagonal
ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, linewidth=1)
ax.set_xlim(-1, max_x * 1.02)  # Small buffer on min, tight on max
ax.set_ylim(-1, max_y * 1.02)

# Plot 2: Highest vs Lowest Mixed-Use Clusters
ax = axes[1]
for country in focus_countries:
    subset = city_focus[city_focus["country"] == country]
    if len(subset) > 0:
        ax.scatter(
            subset[f"pct_cluster_{high_mixed_use_cluster}"],
            subset[f"pct_cluster_{low_mixed_use_cluster}"],
            c=country_colors_map[country],
            label=f"{country} ({len(subset)} cities)",
            alpha=0.6,
            s=100,
            edgecolors="black",
            linewidth=0.8,
        )

ax.grid(True, alpha=0.3, linestyle="--")
ax.set_xlabel(
    f"% in Highest Mixed-Use Cluster ({high_mixed_use_cluster + 1})",
    fontsize=11,
    weight="bold",
)
ax.set_ylabel(
    f"% in Lowest Mixed-Use Cluster ({low_mixed_use_cluster + 1})",
    fontsize=11,
    weight="bold",
)
ax.set_title("City Morphology: Highest vs Lowest Mixed-Use Clusters", fontsize=14, weight="bold")
ax.legend(loc="best", fontsize=10, framealpha=0.9)

# Add diagonal reference line using actual data range
max_x = city_focus[f"pct_cluster_{high_mixed_use_cluster}"].max()
max_y = city_focus[f"pct_cluster_{low_mixed_use_cluster}"].max()
max_val = max(max_x, max_y) * 1.02  # Tight padding for diagonal
ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, linewidth=1)
ax.set_xlim(-1, max_x * 1.02)  # Small buffer on min, tight on max
ax.set_ylim(-1, max_y * 1.02)

fig.suptitle(
    "City Morphology Profiles (FR, NL, IT, RO, PL, DE, ES)\nComparing clusters most correlated with density and mixed uses",
    fontsize=16,
    weight="bold",
)

plt.tight_layout()
plt.savefig(output_path / "city_profiles_clusters.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: city_profiles_clusters.png")

# Save cluster-external correlation summary
cluster_corr_df = pd.DataFrame([{"cluster": c, **cluster_external_corr[c]} for c in range(N_NODE_CLUSTERS)])
cluster_corr_df.to_csv(output_path / "cluster_external_correlations.csv", index=False)
print("  Saved: cluster_external_correlations.csv")

# %%
"""
## Step 10: Export Results
"""

print("\nSTEP 10: Exporting results")

# Prepare export dataframe with country profiles (use cleaned data)
export_cols = ["country", "n_cities", "n_nodes_sampled", "density_mean", "coverage_mean"]
export_cols += [f"pct_cluster_{c}" for c in range(N_NODE_CLUSTERS)]

export_df = country_profiles_clean[export_cols].copy()

# Rename columns for clarity
rename_map = {
    "country": "Country",
    "n_cities": "Number of Cities",
    "n_nodes_sampled": "Nodes Sampled",
    "density_mean": "Avg Population Density",
    "coverage_mean": "Avg Coverage Ratio",
}
for c in range(N_NODE_CLUSTERS):
    rename_map[f"pct_cluster_{c}"] = f"% Cluster {c + 1}"  # 1-based cluster numbering

export_df = export_df.rename(columns=rename_map)

# Sort by cluster proportions
sort_cols = [f"% Cluster {c + 1}" for c in range(N_NODE_CLUSTERS)]
export_df = export_df.sort_values(sort_cols, ascending=False)

# Save full results
export_df.to_csv(output_path / "country_morphology_profiles.csv", index=False)
print(f"  Saved: country_morphology_profiles.csv ({len(export_df)} countries)")

# Save node cluster summary
cluster_profiles_df.to_csv(output_path / "node_cluster_summary.csv", index=False)
print("  Saved: node_cluster_summary.csv")

# Export top countries by each cluster type
for c in range(N_NODE_CLUSTERS):
    top_countries = export_df.nlargest(10, f"% Cluster {c + 1}")[
        ["Country"] + [f"% Cluster {i + 1}" for i in range(N_NODE_CLUSTERS)]
    ]
    top_countries.to_csv(output_path / f"top_cluster_{c + 1}_countries.csv", index=False)
print(f"  Saved: top_cluster_{{1-{N_NODE_CLUSTERS}}}_countries.csv")

# %%
"""
## Step 11: Generate README Report
"""

print("\nSTEP 11: Generating README report")

# Compute summary statistics
n_total = len(city_df)
n_countries = len(country_profiles_df)

readme_content = f"""# EG6: Urban Density and Building Morphology Patterns

## Summary

Analysis of building morphology patterns across {n_total} European cities grouped by {n_countries} countries.
We cluster **nodes** by morphology profile, then characterize each **country** by the proportion
of its nodes in each cluster type. This reveals international morphology patterns.

## Methodology

1. Sample max({MIN_SAMPLE_NODES:,}, {SAMPLE_FRACTION * 100:.0f}%) nodes per city
2. Cluster nodes by {len(MORPH_COLS)} morphology features (BIRCH, k={N_NODE_CLUSTERS}):
   - Building Count, Block Count, Mean Height, Height Variation, Building Area, Fractal Dimension, Block Coverage, Shared Walls
3. For each country, aggregate proportion of nodes in each morphology cluster
4. Countries are characterized by their full cluster proportion profile
5. Countries are hierarchically clustered by their composition similarity


## Node Morphology Clusters

BIRCH clustering (k={N_NODE_CLUSTERS}) on {len(MORPH_COLS)} morphology features identified these node types.

**External characterization** uses three independent variables:
- **Population Density**: Average density (people/km²)
- **Network Density**: Street network density (beta coefficient)
- **Mixed Uses**: Landuse richness (Hill number q=0)

See cluster rankings visualization for relative ordering.

### Cluster Characteristics

| ID | Nodes | % Total | Pop Density | Network Density | Mixed Use |
|----|-------|---------|-------------|-------------|-----------|
"""

# Add cluster descriptions with external metrics
for c in range(N_NODE_CLUSTERS):
    profile = cluster_profiles_df[cluster_profiles_df["cluster"] == c].iloc[0]
    n_nodes = profile["n_nodes"]
    pct = profile["pct_nodes"]

    ext_chars = CLUSTER_EXTERNAL_CHARS.get(c, {})
    density_mean = ext_chars.get("density_mean", 0)
    walkability_mean = ext_chars.get("walkability_mean", 0)
    mixed_use_mean = ext_chars.get("mixed_use_mean", 0)

    readme_content += (
        f"| {c + 1} | {n_nodes:,} | {pct:.1f}% | {density_mean:.0f} | {walkability_mean:.2f} | {mixed_use_mean:.2f} |\n"
    )

readme_content += """

### Cluster External Characteristics
![Cluster Rankings](outputs/cluster_external_rankings.png)

### Cluster Profiles (Radar Plot)
![Cluster Radar Profiles](outputs/cluster_radar_profiles.png)

### Feature-Cluster Correlations
![Cluster Feature Correlations](outputs/cluster_feature_correlations.png)

## Country Morphology Profiles

Each country is characterized by its distribution across node clusters.

### Hierarchically Ordered Heatmap
Countries ordered by compositional similarity (Ward linkage):
![Country Profile Heatmap](outputs/country_profile_heatmap.png)

### Stacked Composition Chart
![Country Composition Stacked](outputs/country_composition_stacked.png)

### City Profiles by Cluster Proportions
![City Profiles](outputs/city_profiles_clusters.png)

## Morphology Patterns by Country

Countries are characterized by their distribution across morphology clusters.
"""

# Add top countries for each cluster
for c in range(N_NODE_CLUSTERS):
    readme_content += f"""
### Top Countries by Cluster {c + 1}

| Country | # Cities | Dominant Cluster | % in Cluster {c + 1} |
|---------|----------|------------------|---------------------|
"""
    for _, row in country_profiles_df.nlargest(5, f"pct_cluster_{c}").iterrows():
        dom_cluster = row[[f"pct_cluster_{i}" for i in range(N_NODE_CLUSTERS)]].idxmax()
        dom_id = int(dom_cluster.replace("pct_cluster_", ""))
        dom_pct = row[dom_cluster]
        cluster_pct = row[f"pct_cluster_{c}"]
        readme_content += f"| {row['country']} | {int(row['n_cities'])} | Cluster {dom_id + 1} | {cluster_pct:.1f}% |\n"

readme_content += f"""
## Outputs

### Data Files
- `country_morphology_profiles.csv`: Full country profiles with cluster proportions
- `node_cluster_summary.csv`: Node cluster characteristics
- `cluster_external_characterization.csv`: External metrics (density, network density, mixed uses) per cluster
- `cluster_external_correlations.csv`: Cluster-external metric correlation matrix
- `top_cluster_1-{N_NODE_CLUSTERS}_countries.csv`: Top countries by each cluster type
- `cluster_representatives.csv`: Representative city for each cluster

### Visualizations
- `cluster_radar_profiles.png`: Individual radar plots for each cluster
- `cluster_external_rankings.png`: Clusters ranked by external characteristics
- `cluster_feature_correlations.png`: Heatmap of cluster-feature correlations
- `node_cluster_profiles.png`: Bar charts of raw cluster means
- `country_profile_heatmap.png`: Hierarchically ordered country composition heatmap
- `country_composition_stacked.png`: Stacked bar chart of country compositions
- `city_profiles_clusters.png`: Cities plotted by contrasting cluster proportions
- `cluster_1-{N_NODE_CLUSTERS}_satellite_5x5.jpg`: Satellite imagery exemplars for each cluster

"""

with open(output_path / ".." / "README.md", "w") as f:
    f.write(readme_content)
print("  Saved: README.md")


print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)
print(f"\nTotal cities analyzed: {n_total}")
print(f"Countries profiled: {n_countries}")
print(f"Node clusters: {N_NODE_CLUSTERS}")
print(f"Morphology features: {len(MORPH_COLS)}")
print(f"Nodes sampled per city: max({MIN_SAMPLE_NODES:,}, {SAMPLE_FRACTION * 100:.0f}% of total)")
print(f"\nOutputs saved to: {output_path}")

# %%
