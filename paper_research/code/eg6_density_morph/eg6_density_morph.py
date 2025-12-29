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
1. Apply PCA for dimensionality reduction on node morphology features
2. Cluster NODES using Hierarchical Clustering (Ward linkage) for balanced clusters
3. Correlate clusters back to original features for interpretability
4. Characterize each CITY by proportions of node types

This reveals whether cities are homogeneous or mixed - e.g., a city might be
"40% dense-vertical, 30% suburban-sprawl, 30% historic-compact".

## Clustering Method
We use hierarchical clustering with adaptive algorithm selection for efficiency:

### Algorithm Selection
- **Small datasets (<50k nodes)**: Agglomerative Clustering with Ward linkage
  - Most accurate, O(n²) complexity
  - Ward minimizes within-cluster variance for balanced clusters
- **Large datasets (≥50k nodes)**: BIRCH (Balanced Iterative Reducing and Clustering)
  - Much faster, O(n) complexity
  - Builds CF-Tree structure for efficient incremental clustering
  - Final clustering uses AgglomerativeClustering on compressed features

### Additional Optimizations
- **PCA preprocessing**: Reduces dimensionality before clustering (95% variance retained)
- **Fastcluster library**: ~10x faster dendrogram computation (if installed: `pip install fastcluster`)
- **Sampled evaluation**: Cluster quality metrics computed on 20k sample
- **Sampled dendrogram**: Visualization uses 5k sample for readability

### Choosing the Number of Clusters
The optimal number of clusters is determined using multiple evaluation metrics:
- **Elbow Method**: Look for the "elbow" point where adding more clusters yields diminishing returns
- **Silhouette Score**: Measures how well samples fit in their clusters (higher is better)
- **Calinski-Harabasz Score**: Ratio of between-cluster to within-cluster variance (higher is better)
- **Davies-Bouldin Score**: Average similarity between clusters (lower is better)
- **Dendrogram**: Visual inspection of the hierarchical structure

Compare these metrics to find a consensus k value that balances cluster quality with interpretability.

## Steps
1. Load ALL city boundaries (no pre-filtering)
2. Load and sample morphology metrics at 400m scale
3. Apply PCA to standardized features
3a. Evaluate optimal number of clusters using multiple metrics
3b. Cluster in PCA space (hierarchical)
4. Correlate clusters back to original features
5. Extract satellite imagery for cluster representatives
6. Compute city profiles (% nodes in each cluster)
7. Visualize node cluster profiles, dendrogram, and cluster-feature correlations
8. Visualize city typology distributions
9. Export results
10. Generate README report

## SOAR Metrics Used (400m scale, weighted versions)
- **Building metrics**: area, perimeter, compactness, orientation, volume,
  form factor, corners, shape index, fractal dimension
- **Block metrics**: area, perimeter, compactness, orientation, coverage ratio
- **Population**: density (persons/km²)

All metrics use weighted neighborhood averages (_wt suffix) for spatial autocorrelation.

## Key Outputs
- **cluster_evaluation.png**: Metrics for choosing optimal number of clusters
- **cluster_dendrogram.png**: Hierarchical clustering dendrogram
- **city_morphology_profiles.csv**: Per-city node type proportions
- **node_cluster_summary.csv**: Node cluster characteristics
- **cluster_feature_correlations.csv/.png**: Correlation between clusters and features
- **pca_loadings.csv/.png**: PCA loadings showing feature contributions to components
- **city_profiles_heatmap.png**: Heatmap of city typology proportions
- **city_profiles_pca.png**: PCA of city morphology profiles
- **README.md**: Summary report with key findings
"""

# %%
import gc
import math
import urllib.request
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from PIL.Image import logger
from sklearn.cluster import Birch
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# %%
"""
## Configuration
"""
# Morphology metrics at 400m scale (using weighted versions)
MORPH_DATA = {
    "cc_area_median_{d}_nw": "Building Area Median",
    "cc_area_mad_{d}_nw": "Building Area MAD",
    "cc_mean_height_median_{d}_nw": "Mean Height Median",
    "cc_mean_height_mad_{d}_nw": "Mean Height MAD",
    # "cc_perimeter_median_{d}_nw": "Perimeter Median",
    # "cc_perimeter_mad_{d}_nw": "Perimeter MAD",
    "cc_compactness_median_{d}_nw": "Compactness Median",
    "cc_compactness_mad_{d}_nw": "Compactness MAD",
    # "cc_orientation_median_{d}_nw": "Orientation Median",
    # "cc_orientation_mad_{d}_nw": "Orientation MAD",
    "cc_volume_median_{d}_nw": "Volume Median",
    "cc_volume_mad_{d}_nw": "Volume MAD",
    "cc_floor_area_ratio_median_{d}_nw": "Floor Area Ratio Median",
    "cc_floor_area_ratio_mad_{d}_nw": "Floor Area Ratio MAD",
    # "cc_form_factor_median_{d}_nw": "Form Factor Median",
    # "cc_form_factor_mad_{d}_nw": "Form Factor MAD",
    # "cc_corners_median_{d}_nw": "Corners Median",
    # "cc_corners_mad_{d}_nw": "Corners MAD",
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
    # "cc_block_orientation_median_{d}_nw": "Block Orientation Median",
    # "cc_block_orientation_mad_{d}_nw": "Block Orientation MAD",
    "cc_block_covered_ratio_median_{d}_nw": "Block Covered Ratio Median",
    "cc_block_covered_ratio_mad_{d}_nw": "Block Covered Ratio MAD",
    "cc_block_{d}_nw": "Block Count",
}


# --- FEATURE SELECTION FOR CLUSTERING ---
# Use a small, interpretable subset for clustering
SELECTED_MORPH_COLS = [
    "cc_building_200_wt",
    "cc_block_200_wt",
    "cc_mean_height_median_200_wt",
    "cc_volume_median_200_wt",
    "cc_block_covered_ratio_median_200_wt",
    "cc_area_median_200_wt",
]
MORPH_COL_NAMES = [
    "Building Count",
    "Block Count",
    "Mean Height Median",
    "Volume Median",
    "Block Covered Ratio Median",
    "Building Area Median",
]
MORPH_COLS = SELECTED_MORPH_COLS  # Use only selected features for clustering
MORPH_LABELS = dict(zip(MORPH_COLS, MORPH_COL_NAMES))

# Population density column
DENSITY_COL = "density"

# Minimum nodes per city for reliable statistics
MIN_NODES = 100

# Number of node clusters for typology
# The script will evaluate k=2 to k=10 and show metrics to help choose optimal k
# Adjust this value based on cluster_evaluation.png and cluster_dendrogram.png
# Consider: silhouette score, elbow point, dendrogram structure, and interpretability
N_NODE_CLUSTERS = 12
COMPUTE_CLUSTER_SCORES = False  # Set to True to compute scores for all k (slower)

# Sample size per city for node clustering: max(20000, 25% of total nodes)
MIN_SAMPLE_NODES = 20000
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
load_cols = MORPH_COLS + [DENSITY_COL]

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
            gdf = gpd.read_file(metrics_file, columns=MORPH_COLS + [DENSITY_COL], layer="streets")
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
                    "coverage_mean": gdf["cc_block_covered_ratio_median_100_nw"].mean()
                    if "cc_block_covered_ratio_median_100_nw" in gdf.columns
                    else np.nan,
                    "volume_mean": gdf["cc_volume_median_100_nw"].mean()
                    if "cc_volume_median_100_nw" in gdf.columns
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
    CLUSTER_EVAL_SAMPLES = min(50000, len(X_nodes_scaled))  # BIRCH can handle more samples
    np.random.seed(42)
    eval_idx = np.random.choice(len(X_nodes_scaled), CLUSTER_EVAL_SAMPLES, replace=False)
    X_eval = X_nodes_scaled[eval_idx]

    # Test different numbers of clusters
    k_range = range(2, 31)  # Test 2-30 clusters
    silhouettes = []
    calinski_scores = []
    davies_bouldin_scores = []

    print(f"  Evaluating k={k_range.start} to k={k_range.stop - 1} on {CLUSTER_EVAL_SAMPLES:,} samples...")

    for k in k_range:
        # Use BIRCH with same parameters as actual clustering
        birch_eval = Birch(
            n_clusters=k,
            threshold=0.3,
            branching_factor=50,
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
            f"    k={k}: Silhouette={silhouette:.3f}, Calinski-Harabasz={calinski:.1f}, Davies-Bouldin={davies_bouldin:.3f}"
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

gc.collect()

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

# BIRCH: Balanced Iterative Reducing and Clustering using Hierarchies
# - Builds a CF-tree (hierarchical structure)
# - Memory efficient O(n)
# - Final step uses AgglomerativeClustering on subclusters
birch = Birch(
    n_clusters=N_NODE_CLUSTERS,
    threshold=0.3,  # Lower = more subclusters = better quality
    branching_factor=50,
)
nodes_df_clean["node_cluster"] = birch.fit_predict(X_nodes_scaled)
print(f"  BIRCH complete: {len(birch.subcluster_centers_)} subclusters → {N_NODE_CLUSTERS} clusters")
del birch
gc.collect()

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
        col_short = col.replace("cc_", "").replace("_100_nw", "").replace("_", " ").title()
        profile[f"{col_short} Mean"] = cluster_nodes[col].mean()

    cluster_profiles.append(profile)

    # Simple characterization based on selected features
    bldg_count = cluster_nodes["cc_building_100_nw"].mean()
    block_count = cluster_nodes["cc_block_100_nw"].mean()
    height = cluster_nodes["cc_mean_height_median_100_nw"].mean()

    bldg_cat = "dense" if bldg_count > node_features_clean["cc_building_100_nw"].median() else "sparse"
    height_cat = "tall" if height > node_features_clean["cc_mean_height_median_100_nw"].median() else "low"

    print(f"    Cluster {c}: {n_nodes:,} nodes ({profile['pct_nodes']:.1f}%) - {bldg_cat}, {height_cat}")

cluster_profiles_df = pd.DataFrame(cluster_profiles)

# %%
"""
## Step 4: Correlate Clusters to Original Features
"""

print("  Computing cluster-feature correlations...")

# Simple correlation summary for each cluster
cluster_feature_corr = []
for c in range(N_NODE_CLUSTERS):
    cluster_binary = (nodes_df_clean["node_cluster"] == c).astype(int).values
    corr_row = {"cluster": c}
    for col in MORPH_COLS:
        corr = np.corrcoef(node_features_clean[col].values, cluster_binary)[0, 1]
        col_short = col.replace("cc_", "").replace("_100_nw", "").replace("_", " ").title()
        corr_row[col_short] = corr
    cluster_feature_corr.append(corr_row)

cluster_feature_corr_df = pd.DataFrame(cluster_feature_corr).set_index("cluster")
cluster_feature_corr_df.to_csv(output_path / "cluster_feature_correlations.csv")
print("  Saved: cluster_feature_correlations.csv")

# %%
"""
## Step 5: Extract Representative Satellite Images for Each Cluster
Set DOWNLOAD_SATELLITE = False to skip if network is slow/unavailable.
"""

DOWNLOAD_SATELLITE = True  # Set to True to download satellite imagery

print("\nSTEP 5: Finding representative locations for each cluster")

# Find representative node for each cluster (closest to centroid in PCA space)
cluster_reps = []

for c in range(N_NODE_CLUSTERS):
    # Get nodes in this cluster
    cluster_mask = nodes_df_clean["node_cluster"] == c
    cluster_features = X_nodes_scaled[cluster_mask]

    # Find centroid in feature space
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
## Step 6: Compute City Profiles (Proportion of Node Types)
"""

print("\nSTEP 6: Computing city profiles from node cluster proportions")

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
## Step 7: Node Cluster Profile Visualization
"""

print("\nSTEP 7: Generating node cluster profile visualization")

# Compute cluster means from nodes
node_cluster_means = node_features_clean.copy()
node_cluster_means["node_cluster"] = nodes_df_clean["node_cluster"]
cluster_means = node_cluster_means.groupby("node_cluster")[MORPH_COLS].mean()

# Create bar chart of cluster profiles for all morphology features
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.flatten()

cluster_colors = plt.cm.viridis(np.linspace(0, 1, N_NODE_CLUSTERS))

for i, col in enumerate(MORPH_COLS):
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
for i in range(len(MORPH_COLS), len(axes)):
    axes[i].axis("off")

plt.suptitle("Node Morphology Cluster Profiles", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(output_path / "node_cluster_profiles.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: node_cluster_profiles.png")

# %%
"""
## Step 7b: Cluster-Feature Correlation Heatmap
"""

print("\nSTEP 7b: Generating cluster-feature correlation heatmap")

# Create heatmap showing correlation between clusters and original features
fig, ax = plt.subplots(figsize=(12, 6))

# Prepare data for heatmap (transpose so features are rows)
corr_matrix = cluster_feature_corr_df.T.values
feature_labels = [MORPH_LABELS.get(col, col.replace("cc_", "").replace("_median_100_wt", "")) for col in MORPH_COLS]

im = ax.imshow(corr_matrix, cmap="RdBu_r", aspect="auto", vmin=-0.5, vmax=0.5)

ax.set_xticks(range(N_NODE_CLUSTERS))
ax.set_xticklabels([f"Cluster {c}" for c in range(N_NODE_CLUSTERS)], fontsize=10)
ax.set_yticks(range(len(feature_labels)))
ax.set_yticklabels(feature_labels, fontsize=9)
ax.set_xlabel("Cluster", fontsize=11)
ax.set_ylabel("Original Feature", fontsize=11)
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
## Step 8: City Profile Heatmap (Proportions by Node Cluster)
"""

print("\nSTEP 8: Generating city profile heatmap")

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
## Step 9: PCA of City Profiles
"""

print("\nSTEP 9: Generating PCA visualization of city profiles")

# Scatter plot of city profiles by cluster dominance
fig, ax = plt.subplots(figsize=(12, 8))

# Color by country (top 5 countries by city count, rest as "Other")
top_countries = city_profiles_df["country"].value_counts().head(5).index
city_profiles_df["country_group"] = city_profiles_df["country"].apply(lambda x: x if x in top_countries else "Other")

colors = plt.cm.tab10(np.linspace(0, 1, len(city_profiles_df["country_group"].unique())))

# Use density and dominant cluster type for visualization
city_profiles_df["dominant_cluster"] = (
    city_profiles_df[pct_cols].idxmax(axis=1).str.replace("pct_cluster_", "").astype(int)
)

for i, country in enumerate(city_profiles_df["country_group"].unique()):
    subset = city_profiles_df[city_profiles_df["country_group"] == country]
    ax.scatter(
        subset["density_mean"],
        subset["dominant_cluster"],
        c=[colors[i]],
        label=f"{country} ({len(subset)})",
        alpha=0.7,
        s=40,
    )

ax.set_xlabel("Mean Population Density", fontsize=12)
ax.set_ylabel("Dominant Cluster Type", fontsize=12)
ax.set_title("City Morphology Profiles by Density", fontsize=14)
ax.set_yticks(range(N_NODE_CLUSTERS))
ax.legend(loc="best")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_path / "city_density_vs_cluster.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: city_density_vs_cluster.png")

# %%
"""
## Step 10: Export Results
"""

print("\nSTEP 10: Exporting results")

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
## Step 11: Generate README Report
"""

print("\nSTEP 11: Generating README report")

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
    volume = cluster_means.loc[c, "cc_volume_median_100_wt"]
    coverage = cluster_means.loc[c, "cc_block_covered_ratio_median_100_wt"]
    area = cluster_means.loc[c, "cc_area_median_100_wt"]

    # Simple characterization
    volume_cat = "High" if volume > cluster_means["cc_volume_median_100_wt"].median() else "Low"
    coverage_cat = "High" if coverage > cluster_means["cc_block_covered_ratio_median_100_wt"].median() else "Low"
    area_cat = "Large" if area > cluster_means["cc_area_median_100_wt"].median() else "Small"

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
2. Cluster nodes by {len(MORPH_COLS)} morphology features (K-means, k={N_NODE_CLUSTERS}):
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
print(f"Morphology features: {len(MORPH_COLS)}")
print(f"Nodes sampled per city: max({MIN_SAMPLE_NODES:,}, {SAMPLE_FRACTION * 100:.0f}% of total)")
print(f"\nOutputs saved to: {output_path}")
