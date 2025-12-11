"""Generate markdown report with city confidence rankings."""

from pathlib import Path

import geopandas as gpd
import pandas as pd

from src import landuse_categories, tools

logger = tools.get_logger(__name__)


def generate_confidence_report(
    confidence_path: str,
    regression_diagnostics_path: str,
    output_dir: str,
    top_n: int = 50,
    bottom_n: int = 50,
) -> None:
    """
    Generate markdown report with confidence rankings and analysis.

    Args:
        confidence_path: Path to city_confidence.gpkg
        regression_diagnostics_path: Path to regression_diagnostics.csv
        output_dir: Output directory for markdown report
        top_n: Number of top cities to include (default: 50)
        bottom_n: Number of bottom cities to include (default: 50)
    """
    tools.validate_filepath(confidence_path)
    tools.validate_filepath(regression_diagnostics_path)
    output_path = Path(output_dir)
    tools.validate_directory(output_path, create=True)

    # Load data
    logger.info(f"Loading confidence scores from {confidence_path}")
    gdf = gpd.read_file(confidence_path, layer="city_confidence")
    # Reset index to ensure all columns are available
    if gdf.index.name:
        gdf = gdf.reset_index()

    # Convert to DataFrame for analysis (drop geometry for report generation)
    df = pd.DataFrame(gdf.drop(columns="geometry"))

    logger.info(f"Available columns: {list(df.columns)}")  # Debug output

    logger.info(f"Loading regression diagnostics from {regression_diagnostics_path}")
    reg_df = pd.read_csv(regression_diagnostics_path, index_col=0)

    # Sort by number of flagged categories (ascending = fewer flags first)
    df_sorted = df.sort_values("n_flagged_categories", ascending=True)

    # Select cities with fewest and most flags
    top_cities = df_sorted.head(top_n)  # Cities with fewest flags
    bottom_cities = df_sorted.tail(bottom_n)  # Cities with most flags

    # Generate markdown report
    report_path = output_path / "confidence_report.md"
    logger.info(f"Generating markdown report at {report_path}")

    with open(report_path, "w") as f:
        # Header
        f.write("# City-Level Land Use Data Confidence Report\n\n")
        f.write("This report provides a comprehensive analysis of data quality for POI (Point of Interest) ")
        f.write("data across 690 EU cities, using regression-based confidence scoring.\n\n")
        f.write("---\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total cities analyzed**: {len(df)}\n")
        f.write(
            f"- **Cities with at least one flagged category**: {(df['n_flagged_categories'] > 0).sum()} "
            f"({(df['n_flagged_categories'] > 0).sum() / len(df) * 100:.1f}%)\n"
        )
        f.write(f"- **Mean flagged categories per city**: {df['n_flagged_categories'].mean():.2f}\n")
        f.write(f"- **Median flagged categories per city**: {df['n_flagged_categories'].median():.1f}\n")
        f.write(f"- **Max flagged categories**: {int(df['n_flagged_categories'].max())}\n\n")

        # Methodology
        f.write("## Methodology\n\n")

        # Detect regression type from diagnostics
        regression_type = "OLS (Ordinary Least Squares)"
        if "regression_type" in reg_df.columns and not reg_df.empty:
            first_type = reg_df["regression_type"].iloc[0]
            if "quantile" in str(first_type):
                quantile_val = str(first_type).split("_")[1]
                regression_type = f"Quantile Regression (q={quantile_val})"

        f.write("Confidence scores are computed using regression analysis to identify cities with ")
        f.write("unexpectedly low POI counts given their population and area:\n\n")
        f.write(f"1. **Regression type**: {regression_type}\n")
        f.write("2. **Regression models**: For each common land-use category (eat_and_drink, retail, ")
        f.write("education, etc.), fit regression: `POI_count ~ population + area_km2`\n")

        if "quantile" in regression_type.lower():
            f.write("   - **Quantile regression**: Fits to cities with HIGHER POI coverage (above the quantile), ")
            f.write("not the mean. This defines what 'good coverage' looks like.\n")
            f.write("   - Cities below this line are more likely to have data quality issues.\n")

        f.write("3. **Residual analysis**: Compute standardized residuals (z-scores) for each city\n")
        f.write("4. **Flagging**: Cities with z-scores < -2.0 in a category are flagged as having ")
        f.write("likely data quality issues\n")
        f.write("5. **Quality assessment**: Based on number of flagged categories and severity of z-scores\n\n")

        # Regression Diagnostics
        f.write("## Regression Model Diagnostics\n\n")

        # Add regression type info if available
        if "regression_type" in reg_df.columns:
            f.write(f"**Regression Type**: {regression_type}\n\n")

        f.write("Model fit quality (R²) for each land-use category:\n\n")
        f.write("| Land Use Category | R² Score | N Cities | Coef (Population) | Coef (Area) | Intercept |\n")
        f.write("|-------------------|----------|----------|-------------------|-------------|----------|\n")

        for cat, row in reg_df.iterrows():
            cat_name = str(cat).replace("_", " ").title()
            f.write(
                f"| {cat_name} | {row['r2']:.3f} | {int(row['n_cities'])} | "
                f"{row['coef_population']:.2e} | {row['coef_area']:.2f} | {row['intercept']:.1f} |\n"
            )

        f.write("\n")

        # Add regression fit visualizations if available
        plots_dir = output_path / "regression_plots"
        if plots_dir.exists():
            f.write("### Regression Fit Visualizations\n\n")
            f.write(
                "The following plots show the regression fit for each category. "
                "Blue dots represent cities with acceptable POI coverage, while red X marks "
                "indicate flagged cities (those significantly below the expected line).\n\n"
            )

            # Add plots for key categories
            key_categories = ["eat_and_drink", "retail", "education", "business_and_services"]
            for cat in key_categories:
                plot_file = plots_dir / f"{cat}_regression_fit.png"
                if plot_file.exists():
                    cat_name = cat.replace("_", " ").title()
                    # Use relative path from output_dir
                    rel_path = f"regression_plots/{cat}_regression_fit.png"
                    f.write(f"#### {cat_name}\n\n")
                    f.write(f"![{cat_name} Regression Fit]({rel_path})\n\n")

            f.write("*Additional regression plots available in the `regression_plots/` directory.*\n\n")

        # Category-specific issues
        f.write("## Most Commonly Flagged Categories\n\n")
        flagged_counts = {}
        for _, row in df.iterrows():
            if row["flagged_categories"]:
                cats = [c.strip() for c in str(row["flagged_categories"]).split(",")]
                for cat in cats:
                    if cat and cat != "missing_population_or_area":
                        flagged_counts[cat] = flagged_counts.get(cat, 0) + 1

        if flagged_counts:
            sorted_flags = sorted(flagged_counts.items(), key=lambda x: x[1], reverse=True)
            f.write("Cities with data quality issues by category:\n\n")
            for cat, count in sorted_flags:
                cat_name = cat.replace("_", " ").title()
                pct = count / len(df) * 100
                f.write(f"- **{cat_name}**: {count} cities ({pct:.1f}%)\n")
            f.write("\n")

        # Data Quality Patterns
        f.write("## Data Quality Patterns\n\n")

        # By population size
        df["pop_bin"] = pd.cut(
            df["population"],
            bins=[0, 100000, 500000, float("inf")],
            labels=["Small (<100k)", "Medium (100k-500k)", "Large (>500k)"],
        )
        pop_summary = df.groupby("pop_bin", observed=True)["n_flagged_categories"].agg(["mean", "count"])

        f.write("### Flagged Categories by City Size\n\n")
        f.write("| City Size | Mean Flagged Categories | N Cities |\n")
        f.write("|-----------|-------------------------|----------|\n")
        for size, row in pop_summary.iterrows():
            f.write(f"| {size} | {row['mean']:.2f} | {int(row['count'])} |\n")
        f.write("\n")

        # By country
        if "country" in df.columns and df["country"].notna().sum() > 0:
            country_summary = (
                df[df["country"].notna()]
                .groupby("country")["n_flagged_categories"]
                .agg(["mean", "median", "count"])
                .sort_values("mean", ascending=True)  # Fewer flags is better
            )

            f.write("### Flagged Categories by Country\n\n")
            f.write("Countries ranked by mean flagged categories (fewer is better):\n\n")
            f.write("| Country | Mean Flags | Median Flags | N Cities |\n")
            f.write("|---------|------------|--------------|----------|\n")
            for country, row in country_summary.iterrows():
                f.write(f"| {country} | {row['mean']:.2f} | {row['median']:.1f} | {int(row['count'])} |\n")
            f.write("\n")

            # Country-level statistics
            countries_with_data = df["country"].notna().sum()
            f.write("**Summary:**\n\n")
            f.write(
                f"- **Cities with country data**: {countries_with_data} ({countries_with_data / len(df) * 100:.1f}%)\n"
            )
            f.write(f"- **Number of countries**: {df[df['country'].notna()]['country'].nunique()}\n")

            # Identify countries with potential data quality issues
            low_conf_countries = country_summary[country_summary["mean"] < 0.5]
            if len(low_conf_countries) > 0:
                f.write(f"- **Countries with mean confidence < 0.5**: {len(low_conf_countries)}\n")
                for country, row in low_conf_countries.iterrows():
                    f.write(f"  - {country}: {row['mean']:.3f} (n={int(row['count'])})\n")
            f.write("\n")

        # Geographic patterns (if label available)
        if "label" in df.columns and df["label"].notna().sum() > 0:
            # Count cities by country (extract from label if possible)
            f.write("### Cities with Labels\n\n")
            labeled_cities = df["label"].notna().sum()
            f.write(f"- **Cities with geographic labels**: {labeled_cities} ({labeled_cities / len(df) * 100:.1f}%)\n")
            f.write(
                f"- **Mean flagged categories (labeled cities)**: {df[df['label'].notna()]['n_flagged_categories'].mean():.2f}\n"
            )
            f.write(
                f"- **Mean flagged categories (unlabeled cities)**: {df[df['label'].isna()]['n_flagged_categories'].mean():.2f}\n\n"
            )

        # Top 50 Cities with Fewest Flags
        f.write(f"## Top {top_n} Cities with Fewest Flagged Categories\n\n")
        f.write("Cities with best data quality (fewest flagged categories):\n\n")
        f.write(
            "| Rank | Bounds FID | City Label | Population | Area (km²) | N Flags | "
            "Mean Z-Score | Flagged Categories |\n"
        )
        f.write(
            "|------|------------|------------|------------|------------|---------|--------------|-------------------|\n"
        )

        for rank, (_, row) in enumerate(top_cities.iterrows(), 1):
            label = row["label"] if pd.notna(row["label"]) else "N/A"
            pop = f"{int(row['population']):,}" if pd.notna(row["population"]) else "N/A"
            area = f"{row['area_km2']:.1f}" if pd.notna(row["area_km2"]) else "N/A"
            n_flags = int(row["n_flagged_categories"]) if pd.notna(row["n_flagged_categories"]) else 0
            mean_z = f"{row['mean_zscore']:.2f}" if pd.notna(row["mean_zscore"]) else "N/A"
            flags = row["flagged_categories"] if row["flagged_categories"] else "None"

            f.write(f"| {rank} | {row['bounds_fid']} | {label} | {pop} | {area} | {n_flags} | {mean_z} | {flags} |\n")

        f.write("\n")

        # Bottom 50 Cities with Most Flags
        f.write(f"## Bottom {bottom_n} Cities with Most Flagged Categories\n\n")
        f.write("Cities with likely data quality issues (most flagged categories):\n\n")
        f.write("| Rank | Bounds FID | City Label | Population | Area (km²) | N Flags | Flagged Categories |\n")
        f.write("|------|------------|------------|------------|------------|---------|-------------------|\n")

        for rank, (_, row) in enumerate(bottom_cities[::-1].iterrows(), 1):
            label = row["label"] if pd.notna(row["label"]) else "N/A"
            pop = f"{int(row['population']):,}" if pd.notna(row["population"]) else "N/A"
            area = f"{row['area_km2']:.1f}" if pd.notna(row["area_km2"]) else "N/A"
            n_flags = int(row["n_flagged_categories"]) if pd.notna(row["n_flagged_categories"]) else 0
            flags = row["flagged_categories"] if row["flagged_categories"] else "None"

            # Truncate long flag lists
            if len(flags) > 60:
                flags = flags[:57] + "..."

            f.write(f"| {rank} | {row['bounds_fid']} | {label} | {pop} | {area} | {n_flags} | {flags} |\n")

        f.write("\n")

        # Detailed Breakdown of Bottom Cities
        f.write("### Detailed Category Breakdown (Bottom 10 Cities)\n\n")
        worst_10 = df_sorted.tail(10)

        for idx, (_, row) in enumerate(worst_10[::-1].iterrows(), 1):
            f.write(f"#### {idx}. Bounds FID {row['bounds_fid']}")
            if pd.notna(row["label"]):
                f.write(f" - {row['label']}")
            f.write("\n\n")

            f.write(
                f"- **Population**: {int(row['population']):,}"
                if pd.notna(row["population"])
                else "- **Population**: N/A\n"
            )
            f.write(f"\n- **Area**: {row['area_km2']:.1f} km²" if pd.notna(row["area_km2"]) else "\n- **Area**: N/A\n")
            f.write(f"\n- **Flagged Categories**: {int(row['n_flagged_categories'])}\n")

            # Show residuals for each category
            f.write("\n**Category Z-Scores:**\n\n")
            has_data = False
            for cat in landuse_categories.COMMON_LANDUSE_CATEGORIES:
                zscore_col = f"{cat}_zscore"
                count_col = f"{cat}_count"
                pred_col = f"{cat}_predicted"

                if zscore_col in row.index and pd.notna(row[zscore_col]):
                    has_data = True
                    cat_name = cat.replace("_", " ").title()
                    z_score = row[zscore_col]
                    count = int(row[count_col]) if pd.notna(row[count_col]) else "N/A"
                    predicted = f"{row[pred_col]:.1f}" if pd.notna(row[pred_col]) else "N/A"

                    flag = " ⚠️ **FLAGGED**" if z_score < -2.0 else ""
                    f.write(f"- {cat_name}: Z={z_score:.2f} (Observed={count}, Expected={predicted}){flag}\n")

            if not has_data:
                f.write("- No z-score data available\n")

            f.write("\n")

        # Recommendations
        f.write("## Recommendations for Dataset Usage\n\n")

        high_flags = (df["n_flagged_categories"] >= 3).sum()
        moderate_flags = ((df["n_flagged_categories"] >= 1) & (df["n_flagged_categories"] < 3)).sum()
        low_flags = (df["n_flagged_categories"] == 0).sum()

        f.write("### Data Quality Tiers\n\n")
        f.write(f"1. **High Quality (0 flags)**: {low_flags} cities - ")
        f.write("Suitable for all analyses including detailed land-use studies\n")
        f.write(f"2. **Moderate Quality (1-2 flags)**: {moderate_flags} cities - ")
        f.write("Suitable for aggregate analyses; use caution for category-specific studies\n")
        f.write(f"3. **Low Quality (≥3 flags)**: {high_flags} cities - ")
        f.write("Recommend excluding from analyses or treating as missing data\n\n")

        f.write("### Specific Recommendations\n\n")

        if high_flags > 0:
            f.write(f"1. **Exclusion criterion**: Consider excluding {high_flags} cities with ")
            f.write("3+ flagged categories from regression analyses to avoid biasing results\n\n")

        f.write("2. **Category-specific caution**: For analyses focused on specific land uses, ")
        f.write("filter cities based on category-specific z-scores rather than overall flag count\n\n")

        f.write("3. **Robust methods**: Use robust regression techniques (e.g., M-estimators) that ")
        f.write("downweight outliers when including all cities\n\n")

        f.write("4. **Data improvement**: Cities with many flagged categories may benefit from manual ")
        f.write("validation or supplementary data sources (e.g., official business registries)\n\n")

        # Footer
        f.write("---\n\n")
        f.write("*Report generated automatically from city-level POI aggregation and regression analysis.*\n")

    logger.info(f"Report generated successfully at {report_path}")

    # Also save simplified CSV summaries
    summary_cols = [
        "bounds_fid",
        "label",
        "population",
        "area_km2",
        "n_flagged_categories",
        "mean_zscore",
        "flagged_categories",
    ]
    # Add country column if available
    if "country" in top_cities.columns:
        summary_cols.insert(2, "country")

    top_cities_simple = top_cities[summary_cols]
    top_cities_simple.to_csv(output_path / "top_50_cities.csv", index=False)

    bottom_cities_simple = bottom_cities[summary_cols]
    bottom_cities_simple.to_csv(output_path / "bottom_50_cities.csv", index=False)

    logger.info("Saved CSV summaries: top_50_cities.csv, bottom_50_cities.csv")

    # Save country summary if available
    if "country" in df.columns and df["country"].notna().sum() > 0:
        country_summary = (
            df[df["country"].notna()]
            .groupby("country")
            .agg(
                {
                    "n_flagged_categories": ["mean", "median", "std", "min", "max"],
                    "bounds_fid": "count",
                    "population": "sum",
                }
            )
            .round(3)
        )
        country_summary.columns = ["_".join(col).strip() for col in country_summary.columns.values]
        country_summary = country_summary.sort_values("n_flagged_categories_mean", ascending=True)
        country_summary.to_csv(output_path / "country_summary.csv")
        logger.info("Saved country summary: country_summary.csv")
