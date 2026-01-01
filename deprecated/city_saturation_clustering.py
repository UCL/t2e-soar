"""
City Saturation Clustering Module

Groups cities based on their raw z-score profiles across POI categories using
Mean Shift clustering to discover natural cluster structure.

## Design Rationale

### Why Mean Shift on Raw Z-Scores?

**Previous approaches:**
- KMeans: forced preset k
- Agglomerative + silhouette: still picking from preset k values
- OPTICS: requires density valleys that may not exist

**Current approach (Mean Shift):**
- Finds natural density modes (peaks) in the data - NO k parameter
- Clusters genuinely "bubble out" from the data structure
- Each cluster forms around a density peak
- Number of clusters is determined entirely by the data

### How Mean Shift Works

1. Start at each data point
2. Compute mean of all points within bandwidth
3. Shift to that mean
4. Repeat until convergence
5. Points that converge to same mode = same cluster

### Parameters

- bandwidth: Radius for mean calculation (auto-estimated if None)
  Larger bandwidth = fewer, broader clusters
  Smaller bandwidth = more, tighter clusters
- If bandwidth=None, sklearn estimates it from data using quantile

### Cluster Naming

Clusters are named based on their z-score profile:
- Identify categories with strongest positive/negative mean z-scores
- E.g., "High eating, Low retail" or "Generally saturated"
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler


def analyze_city_saturation_clustering(
    city_gdf: gpd.GeoDataFrame,
    thresholds_df: pd.DataFrame,
    threshold_lookup: dict,
    poi_categories: list,
    logger=None,
    bandwidth: float = None,
    bandwidth_quantile: float = 0.3,
) -> dict:
    """
    Cluster cities by their raw z-score profiles using Mean Shift.

    Mean Shift finds natural density modes in the data without requiring
    a preset number of clusters. Clusters genuinely emerge from the data.

    For each city and category, also classifies as:
    - Unsaturated: mean z-score < lower threshold (fewer POIs than expected)
    - Saturated: mean z-score > upper threshold (more POIs than expected)
    - Typical: between thresholds (normal POI coverage)

    Parameters:
        city_gdf: GeoDataFrame with city data and z-score metrics (from Step 4)
        thresholds_df: DataFrame with thresholds per category (from Step 3)
        threshold_lookup: Dict mapping categories to threshold results (from Step 3)
        poi_categories: List of POI category names
        logger: Logger instance (optional)
        bandwidth: Kernel bandwidth (if None, estimated from data)
        bandwidth_quantile: Quantile for bandwidth estimation (default 0.3)
            Lower = more clusters, Higher = fewer clusters

    Returns:
        Dictionary with:
        - city_results: GeoDataFrame with saturation profiles and cluster assignments
        - group_counts: Count of cities per cluster
        - status_cols: List of per-category status column names
        - cluster_profiles: DataFrame of mean z-score per cluster per category
        - feature_cols: List of z-score columns used for clustering
        - n_clusters: Number of clusters found
        - cluster_names: Dict mapping cluster IDs to interpretable names
        - bandwidth_used: The bandwidth value used for clustering
    """

    city_results = city_gdf.copy()
    log_fn = logger.info if logger else print

    log_fn("\n" + "=" * 100)
    log_fn("CITY SATURATION CLASSIFICATION")
    log_fn("=" * 100)
    log_fn(f"\nTotal cities: {len(city_results)}")
    log_fn(f"POI categories: {len(poi_categories)}")

    # Step 1: Classify each city-category pair using thresholds (for reporting)
    log_fn("\nClassifying cities by category using thresholds...")

    for cat in poi_categories:
        z_col = f"{cat}_z_mean"
        status_col = f"{cat}_status"

        if z_col not in city_results.columns or cat not in threshold_lookup:
            city_results[status_col] = "no_data"
            continue

        thresh = threshold_lookup[cat]
        if not thresh.get("success", False):
            city_results[status_col] = "no_data"
            continue

        lower = thresh["lower_threshold"]
        upper = thresh["upper_threshold"]

        def classify_city(z):
            if pd.isna(z):
                return "no_data"
            elif z < lower:
                return "unsaturated"
            elif z > upper:
                return "saturated"
            else:
                return "typical"

        city_results[status_col] = city_results[z_col].apply(classify_city)

    # Step 2: Count status across categories for each city (for reporting/visualization)
    status_cols = [f"{cat}_status" for cat in poi_categories if f"{cat}_status" in city_results.columns]

    city_results["n_unsaturated"] = city_results[status_cols].apply(
        lambda row: sum(1 for v in row if v == "unsaturated"), axis=1
    )
    city_results["n_typical"] = city_results[status_cols].apply(
        lambda row: sum(1 for v in row if v == "typical"), axis=1
    )
    city_results["n_saturated"] = city_results[status_cols].apply(
        lambda row: sum(1 for v in row if v == "saturated"), axis=1
    )
    city_results["n_categories"] = len(status_cols)

    # Calculate percentages (for visualization axes)
    city_results["pct_unsaturated"] = 100 * city_results["n_unsaturated"] / city_results["n_categories"]
    city_results["pct_typical"] = 100 * city_results["n_typical"] / city_results["n_categories"]
    city_results["pct_saturated"] = 100 * city_results["n_saturated"] / city_results["n_categories"]

    # Step 3: Build feature matrix from raw z-scores for clustering
    log_fn("\nBuilding feature matrix from raw z-scores...")
    feature_cols = [f"{cat}_z_mean" for cat in poi_categories if f"{cat}_z_mean" in city_results.columns]
    log_fn(f"  Features: {len(feature_cols)} z-score columns")

    # Extract features, fill NaN with 0 (no anomaly = typical)
    X_raw = city_results[feature_cols].copy()
    n_missing = X_raw.isna().sum().sum()
    if n_missing > 0:
        log_fn(f"  Filling {n_missing} missing values with 0 (no anomaly)")
    X_raw = X_raw.fillna(0)

    # Standardize features (z-scores may have different variances across cities)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw.values)

    log_fn(f"  Feature matrix shape: {X_scaled.shape}")

    # Step 4: Apply Mean Shift clustering (finds natural density modes)
    log_fn("\nApplying Mean Shift clustering (natural cluster discovery)...")

    # Estimate bandwidth if not provided
    if bandwidth is None:
        bandwidth = estimate_bandwidth(X_scaled, quantile=bandwidth_quantile, n_samples=min(500, len(X_scaled)))
        log_fn(f"  Auto-estimated bandwidth: {bandwidth:.4f} (quantile={bandwidth_quantile})")
    else:
        log_fn(f"  Using provided bandwidth: {bandwidth:.4f}")

    # Fit Mean Shift
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1)
    cluster_labels = ms.fit_predict(X_scaled)

    # Assign cluster labels
    city_results["cluster"] = cluster_labels

    unique_clusters = sorted(set(cluster_labels))
    n_clusters = len(unique_clusters)
    cluster_centers = ms.cluster_centers_

    log_fn(f"\n  Found {n_clusters} natural clusters")
    for c in unique_clusters:
        count = (cluster_labels == c).sum()
        log_fn(f"    Cluster {c}: {count} cities ({100*count/len(cluster_labels):.1f}%)")

    # Step 5: Compute cluster profiles (mean z-score per category per cluster)
    log_fn("\nComputing cluster profiles...")

    cluster_profiles = pd.DataFrame(index=feature_cols, columns=unique_clusters, dtype=float)

    for cluster_id in unique_clusters:
        mask = city_results["cluster"] == cluster_id
        if mask.sum() > 0:
            cluster_profiles[cluster_id] = X_raw[mask].mean()

    # Step 6: Generate interpretable cluster names based on profiles
    log_fn("\nGenerating cluster names from profiles...")

    cluster_names = {}
    for cluster_id in unique_clusters:
        profile = cluster_profiles[cluster_id]

        # Find categories with strongest signals
        sorted_cats = profile.sort_values()
        most_negative = sorted_cats.head(2)
        most_positive = sorted_cats.tail(2)

        # Build name based on dominant patterns
        name_parts = []

        # Check for strong positive signals (saturated)
        strong_pos = most_positive[most_positive > 0.5]
        if len(strong_pos) > 0:
            top_cat = strong_pos.index[-1].replace("_z_mean", "").replace("_", " ").title()
            if strong_pos.iloc[-1] > 1.0:
                name_parts.append(f"High {top_cat}")
            else:
                name_parts.append(f"Mod+ {top_cat}")

        # Check for strong negative signals (unsaturated)
        strong_neg = most_negative[most_negative < -0.5]
        if len(strong_neg) > 0:
            top_cat = strong_neg.index[0].replace("_z_mean", "").replace("_", " ").title()
            if strong_neg.iloc[0] < -1.0:
                name_parts.append(f"Low {top_cat}")
            else:
                name_parts.append(f"Mod- {top_cat}")

        # Fallback names based on overall pattern
        if not name_parts:
            mean_z = profile.mean()
            if mean_z > 0.3:
                name_parts.append("Generally Saturated")
            elif mean_z < -0.3:
                name_parts.append("Generally Unsaturated")
            else:
                name_parts.append("Balanced")

        cluster_names[cluster_id] = ", ".join(name_parts[:2])  # Max 2 parts

    # Handle duplicate names by adding cluster number
    from collections import Counter

    name_values = list(cluster_names.values())
    name_counts = Counter(name_values)
    if max(name_counts.values()) > 1:
        seen = {}
        for cluster_id in sorted(cluster_names.keys()):
            name = cluster_names[cluster_id]
            if name_counts[name] > 1:
                seen[name] = seen.get(name, 0) + 1
                cluster_names[cluster_id] = f"{name} ({seen[name]})"

    # Map cluster IDs to names
    city_results["saturation_group"] = city_results["cluster"].map(cluster_names)

    # Step 7: Log summary statistics
    log_fn("\n" + "=" * 100)
    log_fn("SATURATION STATUS BY CATEGORY")
    log_fn("=" * 100)
    log_fn(f"\n{'Category':<35s} {'Unsaturated':<12s} {'Typical':<12s} {'Saturated':<12s} {'No Data':<12s}")
    log_fn("-" * 85)

    for cat in poi_categories:
        status_col = f"{cat}_status"
        if status_col in city_results.columns:
            counts = city_results[status_col].value_counts()
            unsat = counts.get("unsaturated", 0)
            typical = counts.get("typical", 0)
            sat = counts.get("saturated", 0)
            no_data = counts.get("no_data", 0)
            log_fn(f"{cat:<35s} {unsat:<12d} {typical:<12d} {sat:<12d} {no_data:<12d}")

    log_fn("\n" + "=" * 100)
    log_fn("NATURAL CLUSTERS (Mean Shift)")
    log_fn("=" * 100)
    group_counts = city_results["saturation_group"].value_counts()
    for group, count in group_counts.items():
        log_fn(f"  {group}: {count} cities ({100 * count / len(city_results):.1f}%)")

    # Log cluster profiles - category importance per cluster
    log_fn("\n" + "=" * 100)
    log_fn("CLUSTER PROFILES (Mean Z-Score per Category)")
    log_fn("=" * 100)

    # Header
    cat_names = [c.replace("_z_mean", "")[:12] for c in feature_cols]
    header = f"{'Cluster':<25s} " + " ".join(f"{c:>12s}" for c in cat_names)
    log_fn(header)
    log_fn("-" * len(header))

    for cluster_id in unique_clusters:
        name = cluster_names[cluster_id]
        values = cluster_profiles[cluster_id].values
        row = f"{name:<25s} " + " ".join(f"{v:>12.2f}" for v in values)
        log_fn(row)

    return {
        "city_results": city_results,
        "group_counts": group_counts.to_dict(),
        "thresholds_df": thresholds_df,
        "threshold_lookup": threshold_lookup,
        "poi_categories": poi_categories,
        "status_cols": status_cols,
        "cluster_profiles": cluster_profiles,
        "feature_cols": feature_cols,
        "n_clusters": n_clusters,
        "cluster_names": cluster_names,
        "cluster_centers": cluster_centers,
        "scaler": scaler,
        "bandwidth_used": bandwidth,
        "meanshift_model": ms,
    }
