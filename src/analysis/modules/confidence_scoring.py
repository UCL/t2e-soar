"""Regression-based confidence scoring for city-level POI data quality."""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, QuantileRegressor

from src import landuse_categories, tools

logger = tools.get_logger(__name__)


def compute_confidence_scores(
    city_stats_path: str,
    output_dir: str,
    residual_threshold: float = -2.0,
    use_quantile_regression: bool = True,
    quantile: float = 0.75,
    strict_mode: bool = True,
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
        use_quantile_regression: If True, fit to upper quantile instead of mean (default: True)
        quantile: Quantile to fit when use_quantile_regression=True (default: 0.75)
        strict_mode: If True, use more aggressive flagging for low POI counts (default: True)
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

        if use_quantile_regression:
            logger.info(f"Fitting quantile regression (q={quantile}) for {cat} ({len(cat_data)} cities)")
        else:
            logger.info(f"Fitting OLS regression for {cat} ({len(cat_data)} cities)")

        # Prepare features and target with log transformation for scale sensitivity
        # Log transformation handles non-linear relationships and scale effects
        X_raw = cat_data[["population", "area_km2"]].values
        y_raw = cat_data[col].values.astype(float)

        # Add small constant to avoid log(0)
        epsilon = 1.0
        X = np.log(X_raw + epsilon)
        y = np.log(y_raw + epsilon)

        # Compute sample weights to balance influence across population range
        # This prevents bunched-up small cities from dominating the fit
        pop_values = cat_data["population"].values
        pop_bins = pd.qcut(pop_values, q=10, duplicates="drop", labels=False)
        bin_counts = pd.Series(pop_bins).value_counts()
        # Inverse frequency weighting: rare bins get higher weight
        sample_weights = np.array([1.0 / bin_counts[bin_id] for bin_id in pop_bins])
        # Normalize weights to sum to n_samples
        sample_weights = sample_weights * len(sample_weights) / sample_weights.sum()

        # Fit regression model (quantile or OLS) in log space with sample weights
        if use_quantile_regression:
            # Fit to upper quantile to establish what "good" coverage looks like
            # Note: QuantileRegressor doesn't support sample_weight in all solvers
            # We'll use a stratified approach instead
            model = QuantileRegressor(quantile=quantile, alpha=0, solver="highs")
            model.fit(X, y, sample_weight=sample_weights)
        else:
            # Standard OLS regression with weights
            model = LinearRegression()
            model.fit(X, y, sample_weight=sample_weights)

        # Predict and compute residuals in original scale
        y_pred_log = model.predict(X)
        y_pred = np.exp(y_pred_log) - epsilon  # Transform back to original scale
        residuals = y_raw - y_pred  # Residuals in original scale

        # Standardize residuals (z-scores)
        if strict_mode and use_quantile_regression:
            # In strict mode with quantile regression, standardize using only negative residuals
            # This makes the threshold more aggressive for cities below the quantile line
            negative_residuals = residuals[residuals < 0]
            if len(negative_residuals) > 0:
                residual_std = negative_residuals.std()
                residual_mean = 0  # Center around the regression line, not the mean
            else:
                residual_std = residuals.std()
                residual_mean = 0
        else:
            # Standard z-score calculation
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

        # Store regression diagnostics (note: coefficients are in log-space)
        r2_score = model.score(X, y)
        regression_results[cat] = {
            "r2": r2_score,
            "coef_population": model.coef_[0] if hasattr(model, "coef_") else model.coef_[0],
            "coef_area": model.coef_[1] if hasattr(model, "coef_") else model.coef_[1],
            "intercept": model.intercept_ if hasattr(model, "intercept_") else model.intercept_,
            "n_cities": len(cat_data),
            "mean_residual": residual_mean,
            "std_residual": residual_std,
            "regression_type": f"log_quantile_{quantile}" if use_quantile_regression else "log_ols",
            "transform": "log",
        }

        logger.info(f"  {cat}: R² = {r2_score:.3f}, n = {len(cat_data)}")

    # Generate regression fit plots
    logger.info("Generating regression fit visualizations")
    plots_dir = output_path / "regression_plots"
    plots_dir.mkdir(exist_ok=True)

    for cat in landuse_categories.COMMON_LANDUSE_CATEGORIES:
        col = f"{cat}_count"
        pred_col = f"{cat}_predicted"
        zscore_col = f"{cat}_zscore"

        if pred_col not in df.columns:
            continue

        # Get valid data for plotting
        plot_data = df[[col, pred_col, zscore_col, "population"]].dropna()
        if len(plot_data) == 0:
            continue

        # Identify flagged cities
        flagged_mask = plot_data[zscore_col] < residual_threshold

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot non-flagged cities
        ax.scatter(
            plot_data.loc[~flagged_mask, "population"],
            plot_data.loc[~flagged_mask, col],
            alpha=0.5,
            s=30,
            c="blue",
            label="Non-flagged cities",
        )

        # Plot flagged cities
        if flagged_mask.sum() > 0:
            ax.scatter(
                plot_data.loc[flagged_mask, "population"],
                plot_data.loc[flagged_mask, col],
                alpha=0.7,
                s=50,
                c="red",
                marker="x",
                label=f"Flagged cities (z < {residual_threshold})",
            )

        # Plot regression line (accounting for log transformation)
        pop_range = np.linspace(plot_data["population"].min(), plot_data["population"].max(), 100)
        # Use median area for visualization
        median_area = df["area_km2"].median()

        # Get the model from regression_results to predict
        if cat in regression_results:
            coef_pop = regression_results[cat]["coef_population"]
            coef_area = regression_results[cat]["coef_area"]
            intercept = regression_results[cat]["intercept"]

            # Model is in log-space: log(y) = coef_pop * log(pop) + coef_area * log(area) + intercept
            # Transform back to original scale for plotting
            epsilon = 1.0
            log_pop = np.log(pop_range + epsilon)
            log_area = np.log(median_area + epsilon)
            y_line_log = coef_pop * log_pop + coef_area * log_area + intercept
            y_line = np.exp(y_line_log) - epsilon

            reg_label = f"Quantile {quantile}" if use_quantile_regression else "OLS"
            ax.plot(pop_range, y_line, "g--", linewidth=2, label=f"{reg_label} regression line (log-transformed)")

        ax.set_xlabel("Population", fontsize=12)
        ax.set_ylabel(f"{cat.replace('_', ' ').title()} Count", fontsize=12)
        ax.set_title(
            f"{cat.replace('_', ' ').title()}: Regression Fit and Flagged Cities\n"
            f"R² = {regression_results[cat]['r2']:.3f}, "
            f"{flagged_mask.sum()} flagged / {len(plot_data)} cities",
            fontsize=13,
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / f"{cat}_regression_fit.png", dpi=150, bbox_inches="tight")
        plt.close()

    logger.info(f"Saved regression fit plots to {plots_dir}")

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

    # Handle cities with missing data
    missing_mask = df["population"].isna() | df["area_km2"].isna()
    df.loc[missing_mask, "flagged_categories"] = "missing_population_or_area"

    logger.info("Flagged category statistics:")
    logger.info(f"  Cities flagged (>0 categories): {(df['n_flagged_categories'] > 0).sum()}")
    logger.info(f"  Mean flagged categories: {df['n_flagged_categories'].mean():.2f}")
    logger.info(f"  Median flagged categories: {df['n_flagged_categories'].median():.1f}")
    logger.info(f"  Max flagged categories: {int(df['n_flagged_categories'].max())}")

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
                "n_flagged_categories": ["count"],
            }
        )
        .round(3)
    )
    flag_summary.columns = ["_".join(col).strip("_") for col in flag_summary.columns]
    flag_summary = flag_summary.rename(columns={"population_count": "n_cities"})
    flag_summary.to_csv(output_path / "flag_summary.csv")
    logger.info(f"Saved flag summary to {output_path / 'flag_summary.csv'}")

    logger.info("Confidence scoring complete!")
