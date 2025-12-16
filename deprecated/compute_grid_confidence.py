"""Grid-based confidence scoring using buffered POI analysis."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

from src import landuse_categories, tools

logger = tools.get_logger(__name__)


def compute_grid_confidence_scores(
    grid_stats_path: str,
    output_dir: str,
    use_quantile_regression: bool = True,
    quantile: float = 0.75,
    strict_mode: bool = True,
    residual_threshold: float = -2.0,
) -> None:
    """Compute confidence scores at grid level and aggregate to cities.

    Parameters
    ----------
    grid_stats_path
        Path to grid statistics GeoPackage
    output_dir
        Output directory for results
    use_quantile_regression
        Use quantile regression instead of OLS
    quantile
        Quantile for quantile regression (e.g., 0.75)
    strict_mode
        Use only negative residuals for z-score calculation
    residual_threshold
        Z-score threshold for flagging grids

    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("GRID-LEVEL CONFIDENCE SCORING")
    logger.info("=" * 80)

    # Load grid statistics
    logger.info(f"Loading grid statistics from {grid_stats_path}")
    gdf = gpd.read_file(grid_stats_path)
    logger.info(f"  Loaded {len(gdf)} grid cells across {gdf['bounds_fid'].nunique()} cities")

    # Store geometry for later
    geometry = gdf["geometry"]
    df = pd.DataFrame(gdf.drop(columns="geometry"))

    # Filter to valid grids
    df_valid = df[df["population"].notna()].copy()
    logger.info(f"  Valid grids with population data: {len(df_valid)}")

    # Store regression results
    regression_results = {}

    # For each land-use category, fit global regression
    for cat in landuse_categories.COMMON_LANDUSE_CATEGORIES:
        col = f"{cat}_count"

        if col not in df_valid.columns:
            logger.warning(f"Column {col} not found, skipping")
            continue

        # Filter to grids with valid POI count data
        cat_data = df_valid.dropna(subset=[col]).copy()

        if len(cat_data) < 10:
            logger.warning(f"Insufficient data for {cat} ({len(cat_data)} grids), skipping")
            continue

        if use_quantile_regression:
            logger.info(f"Fitting quantile regression (q={quantile}) for {cat} ({len(cat_data)} grids)")
        else:
            logger.info(f"Fitting OLS regression for {cat} ({len(cat_data)} grids)")

        # Prepare features with log transformation
        pop_values = cat_data["population"].values.astype(float)
        y_raw = cat_data[col].values.astype(float)

        epsilon = 1.0
        X = np.log(pop_values + epsilon).reshape(-1, 1)
        y = np.log(y_raw + epsilon)

        # Compute sample weights to balance sparse/dense grids
        pop_bins = pd.qcut(pop_values, q=10, duplicates="drop", labels=False)
        bin_counts = pd.Series(pop_bins).value_counts()
        sample_weights = np.array([1.0 / bin_counts[bin_id] for bin_id in pop_bins])
        sample_weights = sample_weights * len(sample_weights) / sample_weights.sum()

        # Fit model
        if use_quantile_regression:
            model = QuantileRegressor(quantile=quantile, alpha=0, solver="highs")
            model.fit(X, y, sample_weight=sample_weights)
        else:
            from sklearn.linear_model import LinearRegression

            model = LinearRegression()
            model.fit(X, y, sample_weight=sample_weights)

        # Predict and compute residuals in original scale
        y_pred_log = model.predict(X)
        y_pred = np.exp(y_pred_log) - epsilon
        residuals = y_raw - y_pred

        # Standardize residuals
        if strict_mode and use_quantile_regression:
            negative_residuals = residuals[residuals < 0]
            if len(negative_residuals) > 0:
                residual_std = negative_residuals.std()
                residual_mean = 0
            else:
                residual_std = residuals.std()
                residual_mean = 0
        else:
            residual_std = residuals.std()
            residual_mean = residuals.mean()

        z_scores = (residuals - residual_mean) / residual_std if residual_std > 0 else residuals

        # Store results
        cat_data[f"{cat}_predicted"] = y_pred
        cat_data[f"{cat}_residual"] = residuals
        cat_data[f"{cat}_zscore"] = z_scores
        cat_data[f"{cat}_flagged"] = z_scores < residual_threshold

        # Merge back to main dataframe
        for col_name in [f"{cat}_predicted", f"{cat}_residual", f"{cat}_zscore", f"{cat}_flagged"]:
            df.loc[cat_data.index, col_name] = cat_data[col_name]

        # Store regression diagnostics
        r2_score = model.score(X, y)
        regression_results[cat] = {
            "r2": r2_score,
            "coef_population": model.coef_[0] if hasattr(model, "coef_") else model.coef_[0],
            "intercept": model.intercept_ if hasattr(model, "intercept_") else model.intercept_,
            "n_grids": len(cat_data),
            "mean_residual": residual_mean,
            "std_residual": residual_std,
            "regression_type": f"log_quantile_{quantile}" if use_quantile_regression else "log_ols",
            "transform": "log",
            "n_flagged": int((z_scores < residual_threshold).sum()),
        }

        logger.info(
            f"  {cat}: R² = {r2_score:.3f}, "
            f"{int((z_scores < residual_threshold).sum())} / {len(cat_data)} grids flagged"
        )

    # Count flagged categories per grid
    df["n_flagged_categories"] = 0
    for cat in landuse_categories.COMMON_LANDUSE_CATEGORIES:
        flag_col = f"{cat}_flagged"
        if flag_col in df.columns:
            df["n_flagged_categories"] += df[flag_col].fillna(False).astype(int)

    # Save grid-level results
    result_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=gdf.crs)
    grid_output = output_path / "grid_confidence.gpkg"
    result_gdf.to_file(grid_output, driver="GPKG", layer="grid_confidence")
    logger.info(f"Saved grid-level confidence scores to {grid_output}")

    # Aggregate to city level
    logger.info("Aggregating grid scores to city level...")
    city_stats = []

    for bounds_fid in df["bounds_fid"].unique():
        city_grids = df[df["bounds_fid"] == bounds_fid]
        total_grids = len(city_grids)

        city_stat = {
            "bounds_fid": bounds_fid,
            "total_grids": total_grids,
            "total_population": city_grids["population"].sum(),
        }

        # Overall flagging percentage
        grids_with_any_flag = (city_grids["n_flagged_categories"] > 0).sum()
        city_stat["pct_flagged_overall"] = (grids_with_any_flag / total_grids * 100) if total_grids > 0 else 0

        # Category-specific flagging percentages
        for cat in landuse_categories.COMMON_LANDUSE_CATEGORIES:
            flag_col = f"{cat}_flagged"
            if flag_col in city_grids.columns:
                flagged_count = city_grids[flag_col].fillna(False).sum()
                city_stat[f"pct_flagged_{cat}"] = (flagged_count / total_grids * 100) if total_grids > 0 else 0
                city_stat[f"n_flagged_{cat}"] = int(flagged_count)

        city_stats.append(city_stat)

    city_df = pd.DataFrame(city_stats)

    # Save city-level aggregated results
    city_output = output_path / "city_confidence.csv"
    city_df.to_csv(city_output, index=False)
    logger.info(f"Saved city-level aggregated scores to {city_output}")

    # Save regression diagnostics
    regression_df = pd.DataFrame(regression_results).T
    regression_df.to_csv(output_path / "regression_diagnostics.csv")
    logger.info(f"Saved regression diagnostics to {output_path / 'regression_diagnostics.csv'}")

    # Summary statistics
    logger.info("\nCity-Level Summary:")
    logger.info(f"  Total cities: {len(city_df)}")
    logger.info(f"  Mean % flagged grids: {city_df['pct_flagged_overall'].mean():.1f}%")
    logger.info(f"  Median % flagged grids: {city_df['pct_flagged_overall'].median():.1f}%")
    logger.info(f"  Cities with >50% flagged grids: {(city_df['pct_flagged_overall'] > 50).sum()}")

    logger.info("\nGrid confidence scoring complete!")
