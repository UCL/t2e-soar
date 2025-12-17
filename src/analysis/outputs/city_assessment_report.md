# POI Quality Assessment Report

## Executive Summary

- **Total cities analyzed**: 699
- **Total grid cells**: 43038
- **POI categories**: 11

**Note**: Z-scores represent continuous deviations from expected POI counts.
No arbitrary thresholds are applied. The quadrant analysis identifies cities
based on their mean z-score (saturation level) and variability (consistency).

---

## City Quadrant Classification

### Consistently Undersaturated
Low POI coverage with uniform spatial distribution across categories.

| City | Country | Mean Z | Between-Cat Std |
|------|---------|--------|-----------------|
| Parla | ES | -1.0161 | 0.3234 |
| Fuenlabrada | ES | -0.8996 | 0.2224 |
| Vallès Occidental | ES | -0.7188 | 0.1485 |
| Torrejón de Ardoz | ES | -0.6203 | 0.1718 |
| Gasteizko kuadrilla / Cuadrilla de Vitoria | ES | -0.6186 | 0.1928 |
| Pinto | ES | -0.5827 | 0.2223 |
| Galați | RO | -0.5689 | 0.1623 |
| Madrid | ES | -0.5508 | 0.1606 |
| Alcalá de Henares | ES | -0.5345 | 0.1431 |
| Bilbao | ES | -0.5339 | 0.1084 |

### Consistently Saturated
High & Uniform

| City | Country | Mean Z | Between-Cat Std |
|------|---------|--------|-----------------|
| Almere | NL | 0.0005 | 0.0632 |
| Lugo | ES | 0.0005 | 0.1712 |
| Prato | IT | 0.0008 | 0.0969 |
| Carpi | IT | 0.0022 | 0.1140 |
| Caserta | IT | 0.0030 | 0.0855 |
| Gouda | NL | 0.0032 | 0.1313 |
| Halle-Vilvoorde | BE | 0.0039 | 0.0725 |
| Bönebüttel | DE | 0.0044 | 0.0902 |
| Lüdenscheid | DE | 0.0049 | 0.0924 |
| Bregenz | AT | 0.0063 | 0.0474 |

### Variable Saturated
High & Variable

| City | Country | Mean Z | Between-Cat Std |
|------|---------|--------|-----------------|
| Matera | IT | 0.3987 | 0.6976 |
| Rimini | IT | 0.4078 | 0.6134 |
| Venezia | IT | 1.1563 | 1.1773 |

---

## Model Performance by Category

| Category | R² Score | Local Importance | Intermediate Importance | Large Importance |
|----------|----------|------------------|-------------------------|------------------|
| Accommodation | 0.3892 | 0.2632 | 0.3970 | 0.3398 |
| Active Life | 0.6727 | 0.6428 | 0.2092 | 0.1480 |
| Arts And Entertainment | 0.5525 | 0.2528 | 0.4828 | 0.2644 |
| Attractions And Activities | 0.5321 | 0.2140 | 0.4254 | 0.3606 |
| Business And Services | 0.6447 | 0.5907 | 0.2181 | 0.1912 |
| Eat And Drink | 0.5998 | 0.3448 | 0.3889 | 0.2663 |
| Education | 0.6598 | 0.6158 | 0.2161 | 0.1681 |
| Health And Medical | 0.5975 | 0.6360 | 0.1882 | 0.1758 |
| Public Services | 0.5789 | 0.4907 | 0.2806 | 0.2287 |
| Religious | 0.5260 | 0.5489 | 0.2475 | 0.2036 |
| Retail | 0.6031 | 0.6093 | 0.1704 | 0.2204 |

---

## Visualizations

The following visualizations have been generated to support this analysis:

### Exploratory Data Analysis
![EDA Analysis](eda_analysis.png)

Key insights:
- **Z-Score Distribution**: Distribution of z-scores across grid cells per category
- **Population Distribution**: Distribution of local population across census grid cells
- **Model Fit (R²)**: Model fit quality for each POI category
- **City Z-Score Distribution**: Distribution of mean z-scores across cities

### Feature Importance Analysis
![Feature Importance](feature_importance.png)

Shows which population scale (local, intermediate, large) is most predictive for each POI category.
Higher values indicate the scale is more important for predicting POI distribution.

### Regression Diagnostics
![Regression Diagnostics](regression_diagnostics.png)

Predicted vs observed POI counts for each category. Shows model fit quality and outliers.
Points closer to the diagonal line indicate better predictions.

### City Quadrant Analysis
![City Quadrant Analysis](city_quadrant_analysis.png)

12-panel visualization (4×3 grid) showing city quadrant classification by POI category:
- **First 11 panels**: Per-category analysis (mean z-score vs spatial std within category)
- **12th panel**: Between-category summary (mean across categories vs std between categories)

Each panel uses consistent color coding for quadrants:
- **Red** (bottom-left): Consistently Undersaturated
- **Green** (bottom-right): Consistently Saturated
- **Orange** (top-left): Variable Undersaturated
- **Blue** (top-right): Variable Saturated

---

## Output Files

### Data Files
- **grid_multiscale.gpkg**: Vector grid dataset with z-scores and predictions
- **city_analysis_results.gpkg**: City-level z-score statistics and per-category + between-category quadrant classifications

### Visualization Files
- **eda_analysis.png**: Exploratory data analysis
- **feature_importance.png**: Random Forest feature importance comparison
- **regression_diagnostics.png**: Predicted vs observed plots for all categories
- **city_quadrant_analysis.png**: 12-panel per-category and between-category quadrant analysis
