"""
Grid-Level Threshold Detection Module

Classifies grids as unsaturated/typical/saturated based on z-score thresholds.

## Design Rationale

### Why Fixed Thresholds Instead of GMM?

The original approach used 3-component Gaussian Mixture Models (GMM) to
"discover" thresholds from the z-score distribution. This was replaced with
simple fixed thresholds (±1 std dev) because:

1. **Z-scores are already standardized**: By construction, z-scores have
   mean=0 and std=1. They ARE the standard normal distribution.

2. **GMM would be redundant**: Fitting a GMM to standardized data just
   rediscovers what we already know — there's a center and two tails.
   The GMM components would converge to approximately μ ≈ {-1, 0, +1}.

3. **Circular logic**: The original code initialized GMM at percentiles
   [16, 50, 84], which correspond to z ≈ {-1, 0, +1}. The GMM would
   converge near these values, making the "discovery" predetermined.

4. **Fixed thresholds are statistically principled**:
   - z < -1: Lower ~16% of standard normal (1 std dev below mean)
   - -1 ≤ z ≤ 1: Middle ~68% (within 1 std dev)
   - z > 1: Upper ~16% (1 std dev above mean)

5. **Interpretability**: "More than 1 standard deviation below expected"
   is immediately understandable. GMM intersection points are opaque.

6. **Reproducibility**: Fixed thresholds give identical results every time.
   GMM has random initialization and convergence variability.

### Threshold Choices

Default ±1 std dev flags ~32% of grids as anomalous (16% each tail).
Users can adjust:
- ±1.5: ~13% flagged (~6.5% each tail)
- ±2.0: ~5% flagged (~2.5% each tail) — more conservative

Provides the core Step 3 functionality for the main analysis workflow.
"""

import geopandas as gpd
import numpy as np
import pandas as pd


def compute_thresholds(
    z_scores: np.ndarray,
    lower_threshold: float,
    upper_threshold: float,
) -> dict:
    """
    Compute classification thresholds for z-scores.

    Uses fixed thresholds based on standard normal distribution properties:
    - z < -1: ~16% of data (lower tail) → Unsaturated
    - -1 ≤ z ≤ 1: ~68% of data (center) → Typical
    - z > 1: ~16% of data (upper tail) → Saturated

    This is more principled than GMM for standardized data, since z-scores
    already have mean=0 and std=1 by construction. GMM would be redundant.

    Parameters:
        z_scores: Array of z-score values (NaN removed)
        lower_threshold: Z-score below which grids are "unsaturated" (default -1.0)
        upper_threshold: Z-score above which grids are "saturated" (default 1.0)

    Returns:
        Dictionary with thresholds and summary statistics
    """
    z_scores = z_scores[~np.isnan(z_scores)]

    if len(z_scores) < 10:
        return {
            "success": False,
            "error": f"Insufficient data ({len(z_scores)} points)",
            "lower_threshold": lower_threshold,
            "upper_threshold": upper_threshold,
        }

    # Compute actual distribution stats for reporting
    n_unsaturated = np.sum(z_scores < lower_threshold)
    n_typical = np.sum((z_scores >= lower_threshold) & (z_scores <= upper_threshold))
    n_saturated = np.sum(z_scores > upper_threshold)

    return {
        "success": True,
        "lower_threshold": lower_threshold,
        "upper_threshold": upper_threshold,
        "n_samples": len(z_scores),
        "n_unsaturated": int(n_unsaturated),
        "n_typical": int(n_typical),
        "n_saturated": int(n_saturated),
        "pct_unsaturated": 100 * n_unsaturated / len(z_scores),
        "pct_typical": 100 * n_typical / len(z_scores),
        "pct_saturated": 100 * n_saturated / len(z_scores),
        "z_mean": float(np.mean(z_scores)),
        "z_std": float(np.std(z_scores)),
    }


def extract_saturation_clusters(
    grid_gdf: gpd.GeoDataFrame,
    logger,
    lower_threshold: float = -1.0,
    upper_threshold: float = 1.0,
) -> tuple:
    """
    Classify grids as unsaturated/typical/saturated based on z-score thresholds.

    Uses fixed z-score thresholds (default ±1 std dev) which is statistically principled:
    - z < -1: Unsaturated (fewer POIs than expected, ~16% of normal distribution)
    - -1 ≤ z ≤ 1: Typical (within 1 std dev of expected, ~68% of normal distribution)
    - z > 1: Saturated (more POIs than expected, ~16% of normal distribution)

    All grids are retained with classification columns added.

    Parameters:
        grid_gdf: GeoDataFrame with z-score columns
        logger: Logger instance for output
        lower_threshold: Z-score below which grids are "unsaturated" (default -1.0)
        upper_threshold: Z-score above which grids are "saturated" (default 1.0)

    Returns:
        Tuple of (grid_gdf, thresholds_df, threshold_lookup, poi_categories)
    """

    # Get POI categories from columns
    poi_categories = sorted([col.replace("_zscore", "") for col in grid_gdf.columns if col.endswith("_zscore")])

    logger.info("\n" + "=" * 100)
    logger.info("STEP 3: Z-SCORE THRESHOLD CLASSIFICATION")
    logger.info("=" * 100)
    logger.info(f"\nUsing fixed thresholds: Unsaturated (z < {lower_threshold}), Saturated (z > {upper_threshold})")
    logger.info("Rationale: Z-scores are already standardized (mean=0, std=1), so fixed thresholds")
    logger.info("           based on standard normal distribution are statistically principled.")

    # Compute thresholds for each POI category
    logger.info(
        f"\n{'Category':<30s} {'Unsaturated %':<14s} {'Typical %':<12s} {'Saturated %':<14s} {'N Samples':<12s}"
    )
    logger.info("-" * 85)

    threshold_results = []

    for cat in poi_categories:
        zscore_col = f"{cat}_zscore"
        z_scores = grid_gdf[zscore_col].dropna().values

        result = compute_thresholds(z_scores, lower_threshold, upper_threshold)
        result["category"] = cat

        if result["success"]:
            logger.info(
                f"{cat:<30s} {result['pct_unsaturated']:>12.1f}%  {result['pct_typical']:>10.1f}%  "
                f"{result['pct_saturated']:>12.1f}%  {result['n_samples']:>10d}"
            )
        else:
            logger.warning(f"{cat:<30s} FAILED: {result.get('error', 'Unknown error')}")

        threshold_results.append(result)

    # Create thresholds dataframe
    thresholds_df = pd.DataFrame(
        [
            {
                "category": r["category"],
                "lower_threshold": r["lower_threshold"],
                "upper_threshold": r["upper_threshold"],
                "n_samples": r.get("n_samples", 0),
                "pct_unsaturated": r.get("pct_unsaturated", np.nan),
                "pct_typical": r.get("pct_typical", np.nan),
                "pct_saturated": r.get("pct_saturated", np.nan),
                "z_mean": r.get("z_mean", np.nan),
                "z_std": r.get("z_std", np.nan),
                "success": r["success"],
            }
            for r in threshold_results
        ]
    )

    # Classify grids
    logger.info("\n" + "=" * 100)
    logger.info("CLASSIFYING GRIDS")
    logger.info("=" * 100)

    threshold_lookup = {r["category"]: r for r in threshold_results}

    # Add classification columns to ALL grids using vectorized operations
    for cat in poi_categories:
        zscore_col = f"{cat}_zscore"
        status_col = f"{cat}_status"

        z = grid_gdf[zscore_col]
        # Vectorized classification
        conditions = [
            z.isna(),
            z < lower_threshold,
            z > upper_threshold,
        ]
        choices = ["no_data", "unsaturated", "saturated"]
        grid_gdf[status_col] = np.select(conditions, choices, default="typical")

    # Count classification distribution per category
    logger.info(f"\n{'Category':<30s} {'Unsaturated':<12s} {'Typical':<12s} {'Saturated':<12s} {'No Data':<12s}")
    logger.info("-" * 80)

    for cat in poi_categories:
        status_col = f"{cat}_status"
        counts = grid_gdf[status_col].value_counts()
        unsaturated = counts.get("unsaturated", 0)
        typical = counts.get("typical", 0)
        saturated = counts.get("saturated", 0)
        no_data = counts.get("no_data", 0)
        logger.info(f"{cat:<30s} {unsaturated:<12d} {typical:<12d} {saturated:<12d} {no_data:<12d}")

    logger.info(f"\nAll {len(grid_gdf)} grids classified with status columns")

    return grid_gdf, thresholds_df, threshold_lookup, poi_categories
