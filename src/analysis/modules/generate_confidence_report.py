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

    # Sort by confidence score
    df_sorted = df.sort_values("confidence_score", ascending=False)

    # Select top and bottom cities
    top_cities = df_sorted.head(top_n)
    bottom_cities = df_sorted.tail(bottom_n)

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
        f.write(f"- **Mean confidence score**: {df['confidence_score'].mean():.3f}\n")
        f.write(f"- **Median confidence score**: {df['confidence_score'].median():.3f}\n")
        f.write(
            f"- **Cities with confidence < 0.5**: {(df['confidence_score'] < 0.5).sum()} "
            f"({(df['confidence_score'] < 0.5).sum() / len(df) * 100:.1f}%)\n"
        )
        f.write(
            f"- **Cities with at least one flagged category**: {(df['n_flagged_categories'] > 0).sum()} "
            f"({(df['n_flagged_categories'] > 0).sum() / len(df) * 100:.1f}%)\n"
        )
        f.write(f"- **Mean flagged categories per city**: {df['n_flagged_categories'].mean():.2f}\n\n")

        # Methodology
        f.write("## Methodology\n\n")
        f.write("Confidence scores are computed using regression analysis to identify cities with ")
        f.write("unexpectedly low POI counts given their population and area:\n\n")
        f.write("1. **Regression models**: For each common land-use category (eat_and_drink, retail, ")
        f.write("education, etc.), fit linear regression: `POI_count ~ population + area_km2`\n")
        f.write("2. **Residual analysis**: Compute standardized residuals (z-scores) for each city\n")
        f.write("3. **Flagging**: Cities with z-scores < -2.0 in a category are flagged as having ")
        f.write("likely data quality issues\n")
        f.write("4. **Confidence score**: Computed as 1.0 minus a weighted penalty based on number of ")
        f.write("flagged categories (60%) and severity of residuals (40%)\n\n")

        # Regression Diagnostics
        f.write("## Regression Model Diagnostics\n\n")
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
        pop_summary = df.groupby("pop_bin", observed=True)["confidence_score"].agg(["mean", "count"])

        f.write("### Confidence by City Size\n\n")
        f.write("| City Size | Mean Confidence | N Cities |\n")
        f.write("|-----------|-----------------|----------|\n")
        for size, row in pop_summary.iterrows():
            f.write(f"| {size} | {row['mean']:.3f} | {int(row['count'])} |\n")
        f.write("\n")

        # By country
        if "country" in df.columns and df["country"].notna().sum() > 0:
            country_summary = (
                df[df["country"].notna()]
                .groupby("country")["confidence_score"]
                .agg(["mean", "median", "count"])
                .sort_values("mean", ascending=False)
            )

            f.write("### Confidence by Country\n\n")
            f.write("Countries ranked by mean confidence score:\n\n")
            f.write("| Country | Mean Confidence | Median Confidence | N Cities |\n")
            f.write("|---------|-----------------|-------------------|----------|\n")
            for country, row in country_summary.iterrows():
                f.write(f"| {country} | {row['mean']:.3f} | {row['median']:.3f} | {int(row['count'])} |\n")
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
                f"- **Mean confidence (labeled cities)**: {df[df['label'].notna()]['confidence_score'].mean():.3f}\n"
            )
            f.write(
                f"- **Mean confidence (unlabeled cities)**: {df[df['label'].isna()]['confidence_score'].mean():.3f}\n\n"
            )

        # Top 50 Most Robust Cities
        f.write(f"## Top {top_n} Most Robust Cities\n\n")
        f.write("Cities with highest confidence scores (best data quality):\n\n")
        f.write(
            "| Rank | Bounds FID | City Label | Population | Area (km²) | Confidence | "
            "Mean Z-Score | Flagged Categories |\n"
        )
        f.write(
            "|------|------------|------------|------------|------------|------------|--------------|-------------------|\n"
        )

        for rank, (_, row) in enumerate(top_cities.iterrows(), 1):
            label = row["label"] if pd.notna(row["label"]) else "N/A"
            pop = f"{int(row['population']):,}" if pd.notna(row["population"]) else "N/A"
            area = f"{row['area_km2']:.1f}" if pd.notna(row["area_km2"]) else "N/A"
            conf = f"{row['confidence_score']:.3f}"
            mean_z = f"{row['mean_zscore']:.2f}" if pd.notna(row["mean_zscore"]) else "N/A"
            flags = row["flagged_categories"] if row["flagged_categories"] else "None"

            f.write(f"| {rank} | {row['bounds_fid']} | {label} | {pop} | {area} | {conf} | {mean_z} | {flags} |\n")

        f.write("\n")

        # Bottom 50 Least Confident Cities
        f.write(f"## Bottom {bottom_n} Least Confident Cities\n\n")
        f.write("Cities with lowest confidence scores (likely data quality issues):\n\n")
        f.write(
            "| Rank | Bounds FID | City Label | Population | Area (km²) | Confidence | N Flags | Flagged Categories |\n"
        )
        f.write(
            "|------|------------|------------|------------|------------|------------|---------|-------------------|\n"
        )

        for rank, (_, row) in enumerate(bottom_cities[::-1].iterrows(), 1):
            label = row["label"] if pd.notna(row["label"]) else "N/A"
            pop = f"{int(row['population']):,}" if pd.notna(row["population"]) else "N/A"
            area = f"{row['area_km2']:.1f}" if pd.notna(row["area_km2"]) else "N/A"
            conf = f"{row['confidence_score']:.3f}"
            n_flags = int(row["n_flagged_categories"]) if pd.notna(row["n_flagged_categories"]) else 0
            flags = row["flagged_categories"] if row["flagged_categories"] else "None"

            # Truncate long flag lists
            if len(flags) > 60:
                flags = flags[:57] + "..."

            f.write(f"| {rank} | {row['bounds_fid']} | {label} | {pop} | {area} | {conf} | {n_flags} | {flags} |\n")

        f.write("\n")

        # Detailed Breakdown of Bottom Cities
        f.write("### Detailed Category Breakdown (Bottom 10 Cities)\n\n")
        worst_10 = df_sorted.tail(10)

        for idx, (_, row) in enumerate(worst_10[::-1].iterrows(), 1):
            f.write(f"#### {idx}. Bounds FID {row['bounds_fid']}")
            if pd.notna(row["label"]):
                f.write(f" - {row['label']}")
            f.write("\n\n")

            f.write(f"- **Confidence Score**: {row['confidence_score']:.3f}\n")
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

        low_conf_count = (df["confidence_score"] < 0.4).sum()
        moderate_conf_count = ((df["confidence_score"] >= 0.4) & (df["confidence_score"] < 0.7)).sum()

        f.write("### Data Quality Tiers\n\n")
        f.write(f"1. **High Confidence (≥0.7)**: {(df['confidence_score'] >= 0.7).sum()} cities - ")
        f.write("Suitable for all analyses including detailed land-use studies\n")
        f.write(f"2. **Moderate Confidence (0.4-0.7)**: {moderate_conf_count} cities - ")
        f.write("Suitable for aggregate analyses; use caution for category-specific studies\n")
        f.write(f"3. **Low Confidence (<0.4)**: {low_conf_count} cities - ")
        f.write("Recommend excluding from analyses or treating as missing data\n\n")

        f.write("### Specific Recommendations\n\n")

        if low_conf_count > 0:
            f.write(f"1. **Exclusion criterion**: Consider excluding {low_conf_count} cities with ")
            f.write("confidence < 0.4 from regression analyses to avoid biasing results\n\n")

        f.write("2. **Category-specific caution**: For analyses focused on specific land uses, ")
        f.write("filter cities based on category-specific z-scores rather than overall confidence\n\n")

        f.write("3. **Robust methods**: Use robust regression techniques (e.g., M-estimators) that ")
        f.write("downweight outliers when including all cities\n\n")

        f.write("4. **Data improvement**: Cities with low confidence scores may benefit from manual ")
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
        "confidence_score",
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
                    "confidence_score": ["mean", "median", "std", "min", "max"],
                    "bounds_fid": "count",
                    "n_flagged_categories": "mean",
                    "population": "sum",
                }
            )
            .round(3)
        )
        country_summary.columns = ["_".join(col).strip() for col in country_summary.columns.values]
        country_summary = country_summary.sort_values("confidence_score_mean", ascending=False)
        country_summary.to_csv(output_path / "country_summary.csv")
        logger.info("Saved country summary: country_summary.csv")
