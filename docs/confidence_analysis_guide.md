# City-Level Data Quality Analysis - Quick Start Guide

This guide explains how to assess POI data quality across your 690 EU cities using regression-based confidence scoring.

## Overview

The confidence analysis identifies cities with suspiciously low land-use accessibility counts by:

1. Aggregating POI counts per city by land-use category (eat_and_drink, retail, education, etc.)
2. Fitting regression models: `POI_count ~ population + area_km2`
3. Computing standardized residuals (z-scores) to flag cities below expected counts
4. Generating confidence scores (0-1) based on number and severity of flags

## Prerequisites

Ensure you have completed the following data processing steps:

- Generated boundaries: `temp/datasets/boundaries.gpkg`
- Loaded Overture POI data: `temp/cities_data/overture/overture_{fid}.gpkg` (per city)
- Downloaded census data: `temp/Eurostat_Census-GRID_2021_V2/ESTAT_Census_2021_V2.gpkg`

## Quick Start

1. **Open the notebook**: `src/analysis/analysis_notebook.py` in VS Code
2. **Configure paths** in the second cell:
   - `BOUNDS_PATH` - Path to boundaries GPKG
   - `OVERTURE_DATA_DIR` - Directory with Overture data
   - `CENSUS_PATH` - Path to census GPKG
   - `OUTPUT_DIR` - Where to save results (e.g., `temp/analysis`)
3. **Run all cells** sequentially using the "Run Cell" button or keyboard shortcuts
4. **Review results** in `temp/analysis/confidence_report.md`

**Output directory:** `temp/analysis/`

**Key outputs:**

- `confidence_report.md` - Main report with top/bottom 50 cities
- `city_confidence.gpkg` - Full confidence scores and z-scores per city (with geometries)
- `city_stats.gpkg` - City-level statistics (with geometries)
- `eda/` - Exploratory visualizations

### Notebook Options

**Skip exploratory data analysis** (faster execution):

In the configuration cell, set:

```python
SKIP_EDA = True
```

**Skip aggregation if already completed**:

The notebook automatically detects if `city_stats.gpkg` exists and skips re-aggregation. To force re-run, delete the file first.

### Workflow Steps

The workflow performs these steps automatically:

**Step 1: Aggregate city-level data**

- Loads boundaries, intersects with census data, counts POIs per category
- Creates: `temp/analysis/city_stats.gpkg` (with city boundary geometries)

**Step 2: Exploratory data analysis** (optional, skip with `--skip_eda`)

- Generates scatter plots, correlation matrices, histograms
- Creates: `temp/analysis/eda/` directory with visualizations

**Step 3: Compute confidence scores**

- Fits regression models per land-use category
- Computes standardized residuals and flags outliers
- Creates: `temp/analysis/city_confidence.gpkg` (with geometries), `temp/analysis/regression_diagnostics.csv`

**Step 4: Generate report**

- Creates markdown report with top/bottom 50 cities
- Creates: `temp/analysis/confidence_report.md`, `temp/analysis/top_50_cities.csv`, `temp/analysis/bottom_50_cities.csv`

## Understanding the Output

### Confidence Report (`confidence_report.md`)

The main report includes:

1. **Executive Summary**: Overall data quality statistics
2. **Regression Diagnostics**: Model fit (R²) for each land-use category
3. **Top 50 Cities**: Cities with best data quality (confidence ≥ 0.9)
4. **Bottom 50 Cities**: Cities with data quality issues
5. **Detailed Breakdown**: Category-specific z-scores for worst cities
6. **Recommendations**: Guidelines for using data in downstream analyses

### Spatial Data Files (GeoPackage format)

**`city_stats.gpkg`** and **`city_confidence.gpkg`** contain city boundary geometries along with all statistics. You can:

- **Open in QGIS** for spatial visualization and exploration
- **Style by confidence score** to visualize data quality patterns geographically
- **Filter and select** cities with specific characteristics
- **Export subsets** for further analysis
- **Overlay with other spatial data** for context

**Tip**: In QGIS, symbolize `city_confidence.gpkg` using the `confidence_score` field with a graduated color scheme to quickly identify areas with data quality issues.

### Confidence Scores Attributes

Key attributes in `city_confidence.gpkg`:

- `bounds_fid` - City identifier
- `label` - City name (if available)
- `population` - Total population from census
- `area_km2` - City area
- `confidence_score` - Overall quality score (0-1)
- `n_flagged_categories` - Number of categories with z-score < -2.0
- `flagged_categories` - Comma-separated list of problematic categories
- `{category}_count` - Observed POI count per category
- `{category}_predicted` - Expected POI count from regression
- `{category}_residual` - Difference (observed - expected)
- `{category}_zscore` - Standardized residual

### Regression Diagnostics (`regression_diagnostics.csv`)

Shows model quality per land-use category:

- `r2` - Proportion of variance explained (higher = better fit)
- `n_cities` - Number of cities used in model
- `coef_population` - Population coefficient (POIs per person)
- `coef_area` - Area coefficient (POIs per km²)

**Typical R² values:**

- Eat & Drink: 0.85-0.95 (strong relationship)
- Retail: 0.80-0.90
- Education: 0.70-0.85
- Accommodation: 0.60-0.75 (more variable)

## Interpreting Results

### Confidence Score Tiers

- **≥ 0.7 (High)**: Reliable data suitable for all analyses
- **0.4 - 0.7 (Moderate)**: Use with caution; check category-specific flags
- **< 0.4 (Low)**: Likely incomplete data; consider excluding

### Common Flagged Categories

Cities may be flagged for:

1. **Systematic undercounting**: Multiple categories flagged (incomplete OSM coverage)
2. **Category-specific gaps**: Single category flagged (e.g., missing education data)
3. **Missing base data**: No population or area data available

### Example Interpretations

**Case 1: High confidence (0.92)**

```
bounds_fid: 123
label: Barcelona
flagged_categories: None
mean_zscore: 0.85
```

→ Reliable data across all categories

**Case 2: Moderate confidence (0.58)**

```
bounds_fid: 456
label: Small Town X
flagged_categories: accommodation, health_and_medical
mean_zscore: -0.35
```

→ Good for most analyses, but limited tourism/healthcare data

**Case 3: Low confidence (0.22)**

```
bounds_fid: 789
label: Remote City Y
flagged_categories: eat_and_drink, retail, education, health_and_medical
mean_zscore: -2.45
```

→ Incomplete OSM coverage; exclude from analyses

## Recommendations for Downstream Analysis

### For Regression Models

**Option A: Exclude low-confidence cities**

```python
df_filtered = df[df['confidence_score'] >= 0.4]
```

**Option B: Use confidence as weight**

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y, sample_weight=df['confidence_score'])
```

**Option C: Use robust regression**

```python
from sklearn.linear_model import HuberRegressor
model = HuberRegressor()  # Downweights outliers automatically
```

### For Category-Specific Studies

Filter based on category z-scores rather than overall confidence:

```python
# Study of retail accessibility - only use cities with good retail data
df_retail = df[df['retail_zscore'] > -2.0]
```

### For Clustering/Classification

Exclude or flag low-confidence cities:

```python
df['data_quality_flag'] = df['confidence_score'].apply(
    lambda x: 'high' if x >= 0.7 else 'moderate' if x >= 0.4 else 'low'
)
```

## Advanced Usage

### Programmatic Access

You can also call the analysis functions directly in Python:

```python
from src.analysis.modules import (
    aggregate_city_stats,
    explore_city_stats,
    confidence_scoring,
    generate_confidence_report,
)

# Run individual steps
aggregate_city_stats.aggregate_city_stats(
    "temp/datasets/boundaries.gpkg",
    "temp/cities_data/overture",
    "temp/Eurostat_Census-GRID_2021_V2/ESTAT_Census_2021_V2.gpkg",
    "temp/analysis",
)

# Adjust confidence scoring threshold
confidence_scoring.compute_confidence_scores(
    "temp/analysis/city_stats.gpkg",
    "temp/analysis",
    residual_threshold=-2.5,  # More conservative
)

# Customize report output
generate_confidence_report.generate_confidence_report(
    "temp/analysis/city_confidence.gpkg",
    "temp/analysis/regression_diagnostics.csv",
    "temp/analysis",
    top_n=100,  # Show more cities
    bottom_n=100,
)
```

## Troubleshooting

**Issue: "Missing overture file for bounds_fid X"**

- Some cities may not have Overture data processed yet
- Check `temp/cities_data/overture/` directory
- Re-run `load_overture.py` for missing cities

**Issue: Low R² values (<0.5)**

- May indicate non-linear relationships
- Check scatter plots in `eda/` directory
- Consider log-transforming POI counts

**Issue: Many cities flagged**

- Check if threshold is too strict (try -2.5)
- Verify census data loaded correctly
- Review regression diagnostic plots

## Next Steps

After reviewing confidence scores:

1. **Filter your dataset** based on confidence tiers
2. **Document exclusions** in your analysis methodology
3. **Use robust methods** if retaining low-confidence cities
4. **Consider manual validation** for critical cities
5. **Re-run periodically** as OSM data improves

## Additional Resources

- Land-use category definitions: `src/landuse_categories.py`
- Regression implementation: `src/analysis/confidence_scoring.py`
- Report generation: `src/analysis/generate_confidence_report.py`
