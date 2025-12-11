"""Exploratory data analysis for city-level statistics."""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import QuantileRegressor

from src import landuse_categories, tools

logger = tools.get_logger(__name__)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


def explore_city_stats(
    city_stats_path: str,
    output_dir: str,
) -> None:
    """
    Generate exploratory data analysis visualizations and tables.

    Args:
        city_stats_path: Path to city_stats.gpkg
        output_dir: Output directory for EDA results
    """
    tools.validate_filepath(city_stats_path)
    output_path = Path(output_dir)
    tools.validate_directory(output_path, create=True)
    eda_dir = output_path / "eda"
    eda_dir.mkdir(exist_ok=True)

    # Load data
    logger.info(f"Loading city statistics from {city_stats_path}")
    gdf = gpd.read_file(city_stats_path, layer="city_stats")
    # Reset index to ensure all columns are available
    if gdf.index.name:
        gdf = gdf.reset_index()
    # Convert to DataFrame for analysis (drop geometry)
    df = pd.DataFrame(gdf.drop(columns="geometry"))
    logger.info(f"Loaded {len(df)} cities with {len(df.columns)} columns")

    # 1. DESCRIPTIVE STATISTICS
    logger.info("Computing descriptive statistics")
    desc_stats = df.describe()
    desc_stats.to_csv(eda_dir / "descriptive_statistics.csv")
    logger.info(f"Saved descriptive statistics to {eda_dir / 'descriptive_statistics.csv'}")

    # Missing value summary
    missing_summary = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isnull().sum().values,
            "missing_pct": (df.isnull().sum() / len(df) * 100).values,
        }
    )
    missing_summary.to_csv(eda_dir / "missing_values.csv", index=False)
    logger.info(f"Missing values summary: \n{missing_summary[missing_summary['missing_count'] > 0]}")

    # 2. IDENTIFY ZERO COUNT CITIES
    logger.info("Identifying cities with zero POI counts")
    zero_count_data = []
    for cat in landuse_categories.COMMON_LANDUSE_CATEGORIES:
        col = f"{cat}_count"
        if col in df.columns:
            zero_cities = df[df[col] == 0]
            for _, row in zero_cities.iterrows():
                zero_count_data.append(
                    {
                        "bounds_fid": row["bounds_fid"],
                        "label": row["label"],
                        "category": cat,
                        "population": row["population"],
                        "area_km2": row["area_km2"],
                    }
                )

    if zero_count_data:
        zero_df = pd.DataFrame(zero_count_data)
        zero_df.to_csv(eda_dir / "zero_count_cities.csv", index=False)
        logger.info(f"Found {len(zero_df)} city-category pairs with zero POIs")
        logger.info(f"Saved to {eda_dir / 'zero_count_cities.csv'}")

    # 3. COMPUTE NORMALIZED DENSITIES
    logger.info("Computing normalized POI densities")
    for cat in landuse_categories.COMMON_LANDUSE_CATEGORIES:
        col = f"{cat}_count"
        if col in df.columns:
            # POIs per 1000 residents
            df[f"{cat}_per_1000_pop"] = (df[col] / df["population"] * 1000).replace([np.inf, -np.inf], np.nan)
            # POIs per km²
            df[f"{cat}_per_km2"] = (df[col] / df["area_km2"]).replace([np.inf, -np.inf], np.nan)

    # Save enhanced dataframe
    df.to_csv(eda_dir / "city_stats_with_densities.csv", index=False)

    # 4. SCATTER PLOTS: POI counts vs Population and Area
    logger.info("Generating scatter plots")
    for cat in landuse_categories.COMMON_LANDUSE_CATEGORIES:
        col = f"{cat}_count"
        if col not in df.columns:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # vs Population
        valid_data = df.dropna(subset=[col, "population"])
        if len(valid_data) > 0:
            axes[0].scatter(valid_data["population"], valid_data[col], alpha=0.5, s=20, c="blue")
            axes[0].set_xlabel("Population")
            axes[0].set_ylabel(f"{cat} Count")
            axes[0].set_title(f"{cat.replace('_', ' ').title()} vs Population")

            # Add regression lines with log transformation and weighting
            if len(valid_data) > 2:
                # Log-transformed quantile regression (75th percentile)
                epsilon = 1.0
                X_log = np.log(valid_data[["population"]].values + epsilon)
                y_log = np.log(valid_data[col].values.astype(float) + epsilon)
                
                # Compute sample weights to balance small vs large cities
                pop_bins = pd.qcut(valid_data["population"], q=10, duplicates="drop", labels=False)
                bin_counts = pop_bins.value_counts()
                sample_weights = np.array([1.0 / bin_counts[bin_id] for bin_id in pop_bins])
                sample_weights = sample_weights * len(sample_weights) / sample_weights.sum()
                
                qr_model = QuantileRegressor(quantile=0.75, alpha=0, solver="highs")
                qr_model.fit(X_log, y_log, sample_weight=sample_weights)
                
                # Generate smooth line for plotting
                x_line = np.linspace(valid_data["population"].min(), valid_data["population"].max(), 100)
                x_line_log = np.log(x_line + epsilon)
                y_q75_log = qr_model.predict(x_line_log.reshape(-1, 1))
                y_q75 = np.exp(y_q75_log) - epsilon
                
                axes[0].plot(
                    x_line,
                    y_q75,
                    "g--",
                    alpha=0.8,
                    linewidth=2,
                    label=f"Q75 (weighted log): coef={qr_model.coef_[0]:.2f}",
                )

                axes[0].legend()

        # vs Area
        valid_data = df.dropna(subset=[col, "area_km2"])
        if len(valid_data) > 0:
            axes[1].scatter(valid_data["area_km2"], valid_data[col], alpha=0.5, s=20, c="blue")
            axes[1].set_xlabel("Area (km²)")
            axes[1].set_ylabel(f"{cat} Count")
            axes[1].set_title(f"{cat.replace('_', ' ').title()} vs Area")

            # Add log-transformed quantile regression line with weighting
            if len(valid_data) > 2:
                epsilon = 1.0
                X_log = np.log(valid_data[["area_km2"]].values + epsilon)
                y_log = np.log(valid_data[col].values.astype(float) + epsilon)
                
                # Weight by area bins
                area_bins = pd.qcut(valid_data["area_km2"], q=10, duplicates="drop", labels=False)
                bin_counts = area_bins.value_counts()
                sample_weights = np.array([1.0 / bin_counts[bin_id] for bin_id in area_bins])
                sample_weights = sample_weights * len(sample_weights) / sample_weights.sum()
                
                qr_model = QuantileRegressor(quantile=0.75, alpha=0, solver="highs")
                qr_model.fit(X_log, y_log, sample_weight=sample_weights)
                
                x_line = np.linspace(valid_data["area_km2"].min(), valid_data["area_km2"].max(), 100)
                x_line_log = np.log(x_line + epsilon)
                y_q75_log = qr_model.predict(x_line_log.reshape(-1, 1))
                y_q75 = np.exp(y_q75_log) - epsilon
                
                axes[1].plot(
                    x_line,
                    y_q75,
                    "g--",
                    alpha=0.8,
                    linewidth=2,
                    label=f"Q75 (weighted log): coef={qr_model.coef_[0]:.2f}",
                )

                axes[1].legend()

        plt.tight_layout()
        plt.savefig(eda_dir / f"scatter_{cat}.png", dpi=150, bbox_inches="tight")
        plt.close()

    logger.info(f"Saved scatter plots to {eda_dir}")

    # 5. CORRELATION MATRIX
    logger.info("Computing correlation matrix")
    corr_cols = ["population", "area_km2"] + [
        f"{cat}_count" for cat in landuse_categories.COMMON_LANDUSE_CATEGORIES if f"{cat}_count" in df.columns
    ]
    corr_df = df[corr_cols].dropna()

    if len(corr_df) > 0:
        corr_matrix = corr_df.corr()

        plt.figure(figsize=(14, 12))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
        )
        plt.title("Correlation Matrix: City Size and POI Counts", fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(eda_dir / "correlation_matrix.png", dpi=150, bbox_inches="tight")
        plt.close()

        corr_matrix.to_csv(eda_dir / "correlation_matrix.csv")
        logger.info(f"Saved correlation matrix to {eda_dir}")

    # 6. DISTRIBUTION HISTOGRAMS
    logger.info("Generating distribution histograms")

    # Population and area distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    df["population"].dropna().hist(bins=50, ax=axes[0, 0])
    axes[0, 0].set_title("Population Distribution")
    axes[0, 0].set_xlabel("Population")

    df["population"].dropna().hist(bins=50, ax=axes[0, 1], log=True)
    axes[0, 1].set_title("Population Distribution (Log Scale)")
    axes[0, 1].set_xlabel("Population")
    axes[0, 1].set_ylabel("Frequency (log)")

    df["area_km2"].dropna().hist(bins=50, ax=axes[1, 0])
    axes[1, 0].set_title("Area Distribution")
    axes[1, 0].set_xlabel("Area (km²)")

    df["area_km2"].dropna().hist(bins=50, ax=axes[1, 1], log=True)
    axes[1, 1].set_title("Area Distribution (Log Scale)")
    axes[1, 1].set_xlabel("Area (km²)")
    axes[1, 1].set_ylabel("Frequency (log)")

    plt.tight_layout()
    plt.savefig(eda_dir / "size_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()

    # POI density distributions
    for cat in landuse_categories.COMMON_LANDUSE_CATEGORIES:
        per_1000_col = f"{cat}_per_1000_pop"
        per_km2_col = f"{cat}_per_km2"

        if per_1000_col not in df.columns:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Per 1000 pop
        valid_data = df[per_1000_col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid_data) > 0:
            valid_data.hist(bins=50, ax=axes[0])
            axes[0].set_title(f"{cat.replace('_', ' ').title()} per 1000 Population")
            axes[0].set_xlabel(f"{cat} per 1000 residents")

        # Per km²
        valid_data = df[per_km2_col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid_data) > 0:
            valid_data.hist(bins=50, ax=axes[1])
            axes[1].set_title(f"{cat.replace('_', ' ').title()} per km²")
            axes[1].set_xlabel(f"{cat} per km²")

        plt.tight_layout()
        plt.savefig(eda_dir / f"density_hist_{cat}.png", dpi=150, bbox_inches="tight")
        plt.close()

    logger.info(f"Saved distribution histograms to {eda_dir}")

    # 7. COUNTRY-LEVEL ANALYSIS
    if "country" in df.columns and df["country"].notna().sum() > 0:
        logger.info("Generating country-level analysis")

        country_df = df[df["country"].notna()].copy()

        # Basic country statistics
        country_stats = (
            country_df.groupby("country")
            .agg(
                {"bounds_fid": "count", "population": ["sum", "mean", "median"], "area_km2": ["sum", "mean", "median"]}
            )
            .round(2)
        )
        country_stats.columns = ["_".join(col).strip() for col in country_stats.columns.values]
        country_stats = country_stats.sort_values("population_sum", ascending=False)
        country_stats.to_csv(eda_dir / "country_statistics.csv")
        logger.info(f"Saved country statistics to {eda_dir / 'country_statistics.csv'}")

        # POI statistics by country
        poi_cols = [
            f"{cat}_count" for cat in landuse_categories.COMMON_LANDUSE_CATEGORIES if f"{cat}_count" in df.columns
        ]

        if poi_cols:
            country_poi_stats = country_df.groupby("country")[poi_cols].agg(["sum", "mean"]).round(1)
            country_poi_stats.columns = ["_".join(col).strip() for col in country_poi_stats.columns.values]
            country_poi_stats.to_csv(eda_dir / "country_poi_statistics.csv")
            logger.info(f"Saved country POI statistics to {eda_dir / 'country_poi_statistics.csv'}")

            # Create country comparison plots for key categories
            key_categories = ["eat_and_drink", "retail", "education", "healthcare"]
            for cat in key_categories:
                col = f"{cat}_count"
                if col in df.columns:
                    country_summary = country_df.groupby("country")[col].sum().sort_values(ascending=False)

                    if len(country_summary) > 0:
                        fig, ax = plt.subplots(figsize=(12, max(6, len(country_summary) * 0.3)))
                        country_summary.plot(kind="barh", ax=ax)
                        ax.set_title(f"Total {cat.replace('_', ' ').title()} POIs by Country")
                        ax.set_xlabel("Total POI Count")
                        ax.set_ylabel("Country")
                        plt.tight_layout()
                        plt.savefig(eda_dir / f"country_comparison_{cat}.png", dpi=150, bbox_inches="tight")
                        plt.close()

            logger.info(f"Saved country comparison plots to {eda_dir}")

    # 8. SUMMARY REPORT
    logger.info("Generating summary report")
    with open(eda_dir / "summary_report.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("CITY-LEVEL DATA EXPLORATORY ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total cities: {len(df)}\n\n")

        f.write("POPULATION STATISTICS:\n")
        f.write(f"  Mean: {df['population'].mean():.0f}\n")
        f.write(f"  Median: {df['population'].median():.0f}\n")
        f.write(f"  Std: {df['population'].std():.0f}\n")
        f.write(f"  Min: {df['population'].min():.0f}\n")
        f.write(f"  Max: {df['population'].max():.0f}\n\n")

        f.write("AREA STATISTICS (km²):\n")
        f.write(f"  Mean: {df['area_km2'].mean():.2f}\n")
        f.write(f"  Median: {df['area_km2'].median():.2f}\n")
        f.write(f"  Std: {df['area_km2'].std():.2f}\n")
        f.write(f"  Min: {df['area_km2'].min():.2f}\n")
        f.write(f"  Max: {df['area_km2'].max():.2f}\n\n")

        f.write("POI COUNT STATISTICS BY CATEGORY:\n")
        for cat in landuse_categories.COMMON_LANDUSE_CATEGORIES:
            col = f"{cat}_count"
            if col in df.columns:
                f.write(f"\n  {cat.replace('_', ' ').title()}:\n")
                f.write(f"    Mean: {df[col].mean():.1f}\n")
                f.write(f"    Median: {df[col].median():.1f}\n")
                f.write(f"    Std: {df[col].std():.1f}\n")
                f.write(f"    Min: {df[col].min():.0f}\n")
                f.write(f"    Max: {df[col].max():.0f}\n")
                f.write(f"    Cities with zero: {(df[col] == 0).sum()}\n")

        # Add country statistics if available
        if "country" in df.columns and df["country"].notna().sum() > 0:
            f.write("\n" + "=" * 80 + "\n")
            f.write("COUNTRY-LEVEL SUMMARY:\n")
            f.write("=" * 80 + "\n\n")

            country_df = df[df["country"].notna()]
            f.write(f"Cities with country data: {len(country_df)} ({len(country_df) / len(df) * 100:.1f}%)\n")
            f.write(f"Number of countries: {country_df['country'].nunique()}\n\n")

            # Top countries by city count
            f.write("Top 10 Countries by Number of Cities:\n")
            top_countries = country_df["country"].value_counts().head(10)
            for country, count in top_countries.items():
                f.write(f"  {country}: {count} cities\n")

            f.write("\nTop 10 Countries by Total Population:\n")
            pop_by_country = country_df.groupby("country")["population"].sum().sort_values(ascending=False).head(10)
            for country, pop in pop_by_country.items():
                f.write(f"  {country}: {pop:,.0f}\n")

    logger.info(f"Saved summary report to {eda_dir / 'summary_report.txt'}")
    logger.info("Exploratory data analysis complete!")
