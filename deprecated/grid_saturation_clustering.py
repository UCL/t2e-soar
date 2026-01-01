"""
Grid Saturation Clustering Module

Groups grid cells based on their raw z-score profiles across POI categories using
HDBSCAN clustering to discover natural spatial patterns with balanced cluster sizes.

## Design Rationale

### Why HDBSCAN?

**Advantages:**
- No preset k: Naturally discovers cluster count from data density
- min_cluster_size: Guarantees minimum cluster sizes (no tiny clusters)
- Handles varying density: Works well with real-world spatial data
- Outlier detection: Labels noise points separately for analysis

### Parameters

- min_cluster_size: Minimum points to form a cluster
  - Higher (500-2000) = fewer, larger clusters
  - Lower (50-200) = more, smaller clusters
- min_samples: Core point density threshold
  - Higher = denser, more conservative clusters
  - Lower = looser clustering
"""

import geopandas as gpd
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import hdbscan

    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


def analyze_grid_saturation_clustering(
    grid_gdf: gpd.GeoDataFrame,
    poi_categories: list,
    logger=None,
    min_cluster_size: int = 500,
    min_samples: int = 50,
    cluster_selection_epsilon: float = 0.0,
    feature_suffix: str = "_zscore",
) -> dict:
    """
    Cluster entities (grid cells or cities) by their z-score profiles using HDBSCAN.

    Parameters:
        grid_gdf: GeoDataFrame with data and z-score metrics
        poi_categories: List of POI category names
        logger: Logger instance (optional)
        min_cluster_size: Minimum number of points to form a cluster
            Higher = fewer, larger clusters (default 500 for grid, 5 for cities)
        min_samples: Number of samples in neighborhood for core points
            Higher = denser clusters required
        cluster_selection_epsilon: Distance threshold for cluster merging
            Higher = more aggressive merging of similar clusters
        feature_suffix: Column suffix for features ("_zscore" for grid, "_z_mean" for cities)

    Returns:
        Dictionary with clustering results
    """

    if not HDBSCAN_AVAILABLE:
        raise ImportError("hdbscan package required. Install with: pip install hdbscan")

    grid_results = grid_gdf.copy()
    log_fn = logger.info if logger else print

    log_fn("\n" + "=" * 100)
    log_fn("GRID SATURATION CLUSTERING (HDBSCAN)")
    log_fn("=" * 100)
    log_fn(f"\nTotal grid cells: {len(grid_results)}")
    log_fn(f"POI categories: {len(poi_categories)}")

    # Build feature matrix from z-scores
    log_fn("\nBuilding feature matrix from z-scores...")
    feature_cols = [
        f"{cat}{feature_suffix}" for cat in poi_categories if f"{cat}{feature_suffix}" in grid_results.columns
    ]
    log_fn(f"  Features: {len(feature_cols)} z-score columns")

    # Extract features, fill NaN with 0
    X_raw = grid_results[feature_cols].copy()
    n_missing = X_raw.isna().sum().sum()
    if n_missing > 0:
        log_fn(f"  Filling {n_missing} missing values with 0")
    X_raw = X_raw.fillna(0)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw.values)
    log_fn(f"  Feature matrix shape: {X_scaled.shape}")

    # Apply HDBSCAN clustering
    log_fn("\nApplying HDBSCAN clustering...")
    log_fn(f"  min_cluster_size: {min_cluster_size}")
    log_fn(f"  min_samples: {min_samples}")
    log_fn(f"  cluster_selection_epsilon: {cluster_selection_epsilon}")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method="eom",  # Excess of Mass - better for varying density
        metric="euclidean",
        core_dist_n_jobs=-1,
    )
    cluster_labels = clusterer.fit_predict(X_scaled)

    grid_results["cluster"] = cluster_labels

    # Count clusters (excluding noise labeled as -1)
    unique_clusters = sorted(set(cluster_labels))
    n_noise = (cluster_labels == -1).sum()
    n_clusters = len([c for c in unique_clusters if c >= 0])

    log_fn(f"\n  Found {n_clusters} natural clusters")
    log_fn(f"  Noise points (outliers): {n_noise} ({100 * n_noise / len(cluster_labels):.1f}%)")

    for c in unique_clusters:
        count = (cluster_labels == c).sum()
        label = f"Cluster {c}" if c >= 0 else "Noise/Outliers"
        log_fn(f"    {label}: {count} grid cells ({100 * count / len(cluster_labels):.1f}%)")

    # Compute cluster profiles
    log_fn("\nComputing cluster profiles...")
    cluster_profiles = pd.DataFrame(index=feature_cols, columns=unique_clusters, dtype=float)

    for cluster_id in unique_clusters:
        mask = grid_results["cluster"] == cluster_id
        if mask.sum() > 0:
            cluster_profiles[cluster_id] = X_raw[mask].mean()

    # Generate cluster names
    log_fn("\nGenerating cluster names...")
    cluster_names = {}

    for cluster_id in unique_clusters:
        if cluster_id == -1:
            cluster_names[cluster_id] = "Outliers/Mixed"
            continue

        profile = cluster_profiles[cluster_id]
        sorted_cats = profile.sort_values()

        name_parts = []

        # Strong positive signals
        strong_pos = sorted_cats.tail(2)[sorted_cats.tail(2) > 0.5]
        if len(strong_pos) > 0:
            top_cat = strong_pos.index[-1].replace(feature_suffix, "").replace("_", " ").title()
            if strong_pos.iloc[-1] > 1.0:
                name_parts.append(f"High {top_cat}")
            else:
                name_parts.append(f"Mod+ {top_cat}")

        # Strong negative signals
        strong_neg = sorted_cats.head(2)[sorted_cats.head(2) < -0.5]
        if len(strong_neg) > 0:
            top_cat = strong_neg.index[0].replace(feature_suffix, "").replace("_", " ").title()
            if strong_neg.iloc[0] < -1.0:
                name_parts.append(f"Low {top_cat}")
            else:
                name_parts.append(f"Mod- {top_cat}")

        # Fallback
        if not name_parts:
            mean_z = profile.mean()
            if mean_z > 0.3:
                name_parts.append("Generally Saturated")
            elif mean_z < -0.3:
                name_parts.append("Generally Unsaturated")
            else:
                name_parts.append("Balanced")

        cluster_names[cluster_id] = ", ".join(name_parts[:2])

    # Handle duplicate names
    from collections import Counter

    name_counts = Counter(cluster_names.values())
    if max(name_counts.values()) > 1:
        seen = {}
        for cluster_id in sorted(cluster_names.keys()):
            name = cluster_names[cluster_id]
            if name_counts[name] > 1:
                seen[name] = seen.get(name, 0) + 1
                cluster_names[cluster_id] = f"{name} ({seen[name]})"

    grid_results["saturation_group"] = grid_results["cluster"].map(cluster_names)

    # Log summary
    log_fn("\n" + "=" * 100)
    log_fn("NATURAL CLUSTERS (HDBSCAN on Grid Cells)")
    log_fn("=" * 100)
    group_counts = grid_results["saturation_group"].value_counts()
    for group, count in group_counts.items():
        log_fn(f"  {group}: {count} grid cells ({100 * count / len(grid_results):.1f}%)")

    # Log cluster profiles
    log_fn("\n" + "=" * 100)
    log_fn("CLUSTER PROFILES (Mean Z-Score per Category)")
    log_fn("=" * 100)

    cat_names = [c.replace(feature_suffix, "")[:12] for c in feature_cols]
    header = f"{'Cluster':<25s} " + " ".join(f"{c:>12s}" for c in cat_names)
    log_fn(header)
    log_fn("-" * len(header))

    for cluster_id in unique_clusters:
        name = cluster_names[cluster_id]
        values = cluster_profiles[cluster_id].values
        row = f"{name:<25s} " + " ".join(f"{v:>12.2f}" for v in values)
        log_fn(row)

    return {
        "grid_results": grid_results,
        "group_counts": group_counts.to_dict(),
        "cluster_profiles": cluster_profiles,
        "feature_cols": feature_cols,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "cluster_names": cluster_names,
        "scaler": scaler,
        "clusterer": clusterer,
        "probabilities": clusterer.probabilities_,
    }
