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
| Parla | ES | -1.1088 | 0.3453 |
| Brăila | RO | -0.8894 | 0.2762 |
| Добрич | BG | -0.8719 | 0.2882 |
| Ceuta | ES | -0.8598 | 0.4272 |
| Alcorcón | ES | -0.8347 | 0.4005 |
| Valdemoro | ES | -0.8154 | 0.2876 |
| Fuenlabrada | ES | -0.7287 | 0.2514 |
| Vallès Occidental | ES | -0.7254 | 0.2563 |
| Arganda del Rey | ES | -0.7190 | 0.5006 |
| Gasteizko kuadrilla / Cuadrilla de Vitoria | ES | -0.6784 | 0.1777 |
| Сливен | BG | -0.6461 | 0.3775 |
| Bydgoszcz | PL | -0.6337 | 0.4007 |
| Melilla | ES | -0.6179 | 0.2706 |
| Bilbao | ES | -0.6099 | 0.1792 |
| Andria | IT | -0.5991 | 0.3289 |
| Magheru | RO | -0.5830 | 0.2546 |
| Coslada | ES | -0.5811 | 0.2962 |
| Mantes-la-Jolie | FR | -0.5767 | 0.2091 |
| Ostrava | CZ | -0.5758 | 0.2527 |
| Хасково | BG | -0.5691 | 0.2959 |
| Cerignola | IT | -0.5671 | 0.2463 |
| None | None | -0.5633 | 0.0322 |
| Galați | RO | -0.5592 | 0.1693 |
| Ploiești | RO | -0.5467 | 0.2517 |
| Плевен | BG | -0.5386 | 0.2381 |
| Vallès Occidental | ES | -0.5368 | 0.1852 |
| Buzău | RO | -0.5351 | 0.2193 |
| Torrejón de Ardoz | ES | -0.5346 | 0.3601 |
| Panevėžys | LT | -0.5322 | 0.3304 |
| Ortsbeirat 6 : Evershagen | DE | -0.5293 | 0.1875 |

### Consistently Saturated
High & Uniform

| City | Country | Mean Z | Between-Cat Std |
|------|---------|--------|-----------------|
| Enschede | NL | 0.0001 | 0.1873 |
| Comarca de la Vega de Granada | ES | 0.0003 | 0.2311 |
| Gütersloh | DE | 0.0006 | 0.1955 |
| Debrecen | HU | 0.0008 | 0.2163 |
| Västerås | SE | 0.0010 | 0.1489 |
| Kaunas | LT | 0.0013 | 0.1749 |
| Košice | SK | 0.0018 | 0.2072 |
| Iserlohn | DE | 0.0026 | 0.2062 |
| Tatabánya | HU | 0.0028 | 0.2565 |
| Düren | DE | 0.0028 | 0.2359 |
| České Budějovice | CZ | 0.0032 | 0.2172 |
| Jürgensby | DE | 0.0040 | 0.1532 |
| Wolfsburg | DE | 0.0046 | 0.1254 |
| Zamora | ES | 0.0050 | 0.2695 |
| Ridderkerk | NL | 0.0060 | 0.2735 |
| El Bierzo | ES | 0.0075 | 0.2697 |
| Aix-en-Provence | FR | 0.0081 | 0.2268 |
| Anzio | IT | 0.0087 | 0.2665 |
| Hilden | DE | 0.0096 | 0.2160 |
| Jena-Zentrum | DE | 0.0097 | 0.2742 |
| Κατσικάς | GR | 0.0104 | 0.3140 |
| Milano | IT | 0.0125 | 0.1557 |
| Saint-Julien-en-Genevois | CH | 0.0126 | 0.1235 |
| Veenendaal | NL | 0.0128 | 0.2645 |
| Roosendaal | NL | 0.0129 | 0.3499 |
| Durach | DE | 0.0142 | 0.1764 |
| Antwerpen | BE | 0.0152 | 0.1317 |
| Tours | FR | 0.0154 | 0.1542 |
| Zagreb | HR | 0.0156 | 0.1462 |
| Brackel | DE | 0.0156 | 0.2964 |

---

## Model Performance by Category

| Category | R² Score | Local Importance | Intermediate Importance | Large Importance |
|----------|----------|------------------|-------------------------|------------------|
| Accommodation | 0.5572 | 0.4040 | 0.3371 | 0.2589 |
| Active Life | 0.6551 | 0.6411 | 0.2068 | 0.1521 |
| Arts And Entertainment | 0.6315 | 0.4450 | 0.3711 | 0.1839 |
| Attractions And Activities | 0.6031 | 0.2641 | 0.5289 | 0.2071 |
| Business And Services | 0.7283 | 0.7559 | 0.1398 | 0.1043 |
| Eat And Drink | 0.7148 | 0.7230 | 0.1549 | 0.1221 |
| Education | 0.7255 | 0.7232 | 0.1572 | 0.1196 |
| Health And Medical | 0.6933 | 0.7211 | 0.1436 | 0.1353 |
| Public Services | 0.6837 | 0.6408 | 0.2079 | 0.1513 |
| Religious | 0.5860 | 0.5569 | 0.2323 | 0.2108 |
| Retail | 0.7016 | 0.7472 | 0.1367 | 0.1161 |

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
- **grid_counts_regress.gpkg**: Vector grid dataset with z-scores and predictions
- **city_analysis_results.gpkg**: City-level z-score statistics and per-category + between-category quadrant classifications

### Visualization Files
- **eda_analysis.png**: Exploratory data analysis
- **feature_importance.png**: Random Forest feature importance comparison
- **regression_diagnostics.png**: Predicted vs observed plots for all categories
- **city_quadrant_analysis.png**: 12-panel per-category and between-category quadrant analysis
