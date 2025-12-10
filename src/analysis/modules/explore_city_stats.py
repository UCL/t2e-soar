"""Exploratory data analysis for city-level statistics."""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
            axes[0].scatter(valid_data["population"], valid_data[col], alpha=0.5, s=20)
            axes[0].set_xlabel("Population")
            axes[0].set_ylabel(f"{cat} Count")
            axes[0].set_title(f"{cat.replace('_', ' ').title()} vs Population")

            # Add regression line
            if len(valid_data) > 2:
                z = np.polyfit(valid_data["population"], valid_data[col], 1)
                p = np.poly1d(z)
                x_line = np.linspace(valid_data["population"].min(), valid_data["population"].max(), 100)
                axes[0].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f"y={z[0]:.2e}x+{z[1]:.2f}")
                axes[0].legend()

        # vs Area
        valid_data = df.dropna(subset=[col, "area_km2"])
        if len(valid_data) > 0:
            axes[1].scatter(valid_data["area_km2"], valid_data[col], alpha=0.5, s=20)
            axes[1].set_xlabel("Area (km²)")
            axes[1].set_ylabel(f"{cat} Count")
            axes[1].set_title(f"{cat.replace('_', ' ').title()} vs Area")

            # Add regression line
            if len(valid_data) > 2:
                z = np.polyfit(valid_data["area_km2"], valid_data[col], 1)
                p = np.poly1d(z)
                x_line = np.linspace(valid_data["area_km2"].min(), valid_data["area_km2"].max(), 100)
                axes[1].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f"y={z[0]:.2f}x+{z[1]:.2f}")
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

    # 7. SUMMARY REPORT
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

    logger.info(f"Saved summary report to {eda_dir / 'summary_report.txt'}")
    logger.info("Exploratory data analysis complete!")
