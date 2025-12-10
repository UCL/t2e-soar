"""Regression-based confidence scoring for city-level POI data quality."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src import landuse_categories, tools

logger = tools.get_logger(__name__)


def compute_confidence_scores(
    city_stats_path: str,
    output_dir: str,
    residual_threshold: float = -2.0,
) -> None:
    """
    Compute regression-based confidence scores for city POI data quality.

    Uses linear regression (POI_count ~ population + area) to identify cities
    with unexpectedly low land-use counts, which likely indicate data quality
    issues rather than genuine urban characteristics.

    Args:
        city_stats_path: Path to city_stats.gpkg
        output_dir: Output directory for confidence scores
        residual_threshold: Z-score threshold for flagging (default: -2.0)
    """
    tools.validate_filepath(city_stats_path)
    output_path = Path(output_dir)
    tools.validate_directory(output_path, create=True)

    # Load data
    logger.info(f"Loading city statistics from {city_stats_path}")
    gdf = gpd.read_file(city_stats_path, layer="city_stats")
    logger.info(f"Loaded {len(gdf)} cities with geometries")

    # Reset index to ensure all columns are available
    if gdf.index.name:
        gdf = gdf.reset_index()

    # Convert to DataFrame for calculations (keeping geometry for later)
    df = pd.DataFrame(gdf.drop(columns="geometry"))
    geometry = gdf["geometry"].copy()

    # Filter out cities with missing population or area
    df_valid = df.dropna(subset=["population", "area_km2"]).copy()
    logger.info(f"Using {len(df_valid)} cities with valid population and area data")

    # Store regression results and residuals
    regression_results = {}

    # For each common land-use category, fit regression model
    for cat in landuse_categories.COMMON_LANDUSE_CATEGORIES:
        col = f"{cat}_count"

        if col not in df_valid.columns:
            logger.warning(f"Column {col} not found, skipping")
            continue

        # Filter to cities with valid POI count data
        cat_data = df_valid.dropna(subset=[col]).copy()

        if len(cat_data) < 10:
            logger.warning(f"Insufficient data for {cat} ({len(cat_data)} cities), skipping")
            continue

        logger.info(f"Fitting regression model for {cat} ({len(cat_data)} cities)")

        # Prepare features and target
        X = cat_data[["population", "area_km2"]].values
        y = cat_data[col].values

        # Fit linear regression
        model = LinearRegression()
        model.fit(X, y)

        # Predict and compute residuals
        y_pred = model.predict(X)
        residuals = y - y_pred

        # Standardize residuals (z-scores)
        residual_std = residuals.std()
        residual_mean = residuals.mean()
        z_scores = (residuals - residual_mean) / residual_std if residual_std > 0 else residuals

        # Store results
        cat_data[f"{cat}_predicted"] = y_pred
        cat_data[f"{cat}_residual"] = residuals
        cat_data[f"{cat}_zscore"] = z_scores

        # Merge back to main dataframe
        for col_name in [f"{cat}_predicted", f"{cat}_residual", f"{cat}_zscore"]:
            df.loc[cat_data.index, col_name] = cat_data[col_name]

        # Store regression diagnostics
        r2_score = model.score(X, y)
        regression_results[cat] = {
            "r2": r2_score,
            "coef_population": model.coef_[0],
            "coef_area": model.coef_[1],
            "intercept": model.intercept_,
            "n_cities": len(cat_data),
            "mean_residual": residual_mean,
            "std_residual": residual_std,
        }

        logger.info(f"  {cat}: R² = {r2_score:.3f}, n = {len(cat_data)}")

    # FLAG CITIES: identify those with residuals below threshold
    logger.info(f"Flagging cities with residuals < {residual_threshold}")

    df["n_flagged_categories"] = 0
    df["flagged_categories"] = ""
    df["mean_zscore"] = np.nan
    df["min_zscore"] = np.nan

    for idx, row in df.iterrows():
        flagged_cats = []
        z_scores = []

        for cat in landuse_categories.COMMON_LANDUSE_CATEGORIES:
            zscore_col = f"{cat}_zscore"
            if zscore_col in df.columns and pd.notna(row[zscore_col]):
                z_scores.append(row[zscore_col])
                if row[zscore_col] < residual_threshold:
                    flagged_cats.append(cat)

        df.loc[idx, "n_flagged_categories"] = len(flagged_cats)
        df.loc[idx, "flagged_categories"] = ", ".join(flagged_cats)

        if z_scores:
            df.loc[idx, "mean_zscore"] = np.mean(z_scores)
            df.loc[idx, "min_zscore"] = np.min(z_scores)

    # COMPUTE CONFIDENCE SCORE (0-1)
    # Based on: (1) number of flagged categories, (2) severity of residuals
    max_categories = len(landuse_categories.COMMON_LANDUSE_CATEGORIES)

    # Confidence decreases with more flags and more severe negative z-scores
    df["confidence_score"] = 1.0

    for idx, row in df.iterrows():
        if pd.notna(row["n_flagged_categories"]) and row["n_flagged_categories"] > 0:
            # Penalty for number of flags (0 to 1)
            flag_penalty = row["n_flagged_categories"] / max_categories

            # Penalty for severity (z-scores below threshold)
            if pd.notna(row["min_zscore"]):
                severity_penalty = max(0, (residual_threshold - row["min_zscore"]) / abs(residual_threshold))
                severity_penalty = min(1.0, severity_penalty)  # Cap at 1
            else:
                severity_penalty = 0

            # Combined confidence (weighted average, more weight on flags)
            confidence = 1.0 - (0.6 * flag_penalty + 0.4 * severity_penalty)
            df.loc[idx, "confidence_score"] = max(0, confidence)

    # Handle cities with missing data
    missing_mask = df["population"].isna() | df["area_km2"].isna()
    df.loc[missing_mask, "confidence_score"] = 0.0
    df.loc[missing_mask, "flagged_categories"] = "missing_population_or_area"

    logger.info("Confidence score statistics:")
    logger.info(f"  Mean: {df['confidence_score'].mean():.3f}")
    logger.info(f"  Median: {df['confidence_score'].median():.3f}")
    logger.info(f"  Min: {df['confidence_score'].min():.3f}")
    logger.info(f"  Max: {df['confidence_score'].max():.3f}")
    logger.info(f"  Cities flagged (>0 categories): {(df['n_flagged_categories'] > 0).sum()}")
    logger.info(f"  Cities with confidence < 0.5: {(df['confidence_score'] < 0.5).sum()}")

    # SAVE RESULTS
    # Recreate GeoDataFrame with updated data
    result_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=gdf.crs)

    confidence_output = output_path / "city_confidence.gpkg"
    result_gdf.to_file(confidence_output, driver="GPKG", layer="city_confidence")
    logger.info(f"Saved confidence scores with geometries to {confidence_output}")

    # Save regression diagnostics (CSV is fine for this summary table)
    regression_df = pd.DataFrame(regression_results).T
    regression_df.to_csv(output_path / "regression_diagnostics.csv")
    logger.info(f"Saved regression diagnostics to {output_path / 'regression_diagnostics.csv'}")

    # Summary statistics by flag count
    flag_summary = (
        df.groupby("n_flagged_categories")
        .agg(
            {
                "population": ["count", "mean"],
                "confidence_score": ["mean", "min", "max"],
            }
        )
        .round(3)
    )
    flag_summary.columns = ["_".join(col).strip("_") for col in flag_summary.columns]
    flag_summary = flag_summary.rename(columns={"population_count": "n_cities"})
    flag_summary.to_csv(output_path / "flag_summary.csv")
    logger.info(f"Saved flag summary to {output_path / 'flag_summary.csv'}")

    logger.info("Confidence scoring complete!")
