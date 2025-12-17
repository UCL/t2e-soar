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
| Parla | ES | -1.1013 | 0.3346 |
| Brăila | RO | -0.8890 | 0.2680 |
| Добрич | BG | -0.8739 | 0.2879 |
| Ceuta | ES | -0.8649 | 0.4262 |
| Valdemoro | ES | -0.8131 | 0.2979 |
| Alcorcón | ES | -0.8128 | 0.3989 |
| Fuenlabrada | ES | -0.7278 | 0.2478 |
| Vallès Occidental | ES | -0.7264 | 0.2573 |
| Arganda del Rey | ES | -0.7262 | 0.4988 |
| Gasteizko kuadrilla / Cuadrilla de Vitoria | ES | -0.6792 | 0.1772 |
| Сливен | BG | -0.6412 | 0.3742 |
| Bydgoszcz | PL | -0.6337 | 0.4032 |
| Melilla | ES | -0.6218 | 0.2717 |
| Bilbao | ES | -0.6101 | 0.1781 |
| Andria | IT | -0.6012 | 0.3279 |
| Ostrava | CZ | -0.5817 | 0.2540 |
| Coslada | ES | -0.5798 | 0.2925 |
| Magheru | RO | -0.5782 | 0.2496 |
| Mantes-la-Jolie | FR | -0.5769 | 0.2095 |
| Cerignola | IT | -0.5700 | 0.2408 |
| Хасково | BG | -0.5695 | 0.2984 |
| Galați | RO | -0.5569 | 0.1625 |
| None | None | -0.5550 | 0.0387 |
| Ploiești | RO | -0.5455 | 0.2506 |
| Torrejón de Ardoz | ES | -0.5435 | 0.3567 |
| Плевен | BG | -0.5412 | 0.2344 |
| Vallès Occidental | ES | -0.5389 | 0.1888 |
| Buzău | RO | -0.5299 | 0.2169 |
| Ortsbeirat 6 : Evershagen | DE | -0.5297 | 0.1855 |
| Panevėžys | LT | -0.5282 | 0.3294 |

### Consistently Saturated
High & Uniform

| City | Country | Mean Z | Between-Cat Std |
|------|---------|--------|-----------------|
| Enschede | NL | 0.0004 | 0.1896 |
| Ridderkerk | NL | 0.0007 | 0.2651 |
| Düren | DE | 0.0008 | 0.2378 |
| Kaunas | LT | 0.0010 | 0.1732 |
| Jürgensby | DE | 0.0024 | 0.1516 |
| Västerås | SE | 0.0026 | 0.1511 |
| Iserlohn | DE | 0.0027 | 0.2084 |
| Košice | SK | 0.0028 | 0.2126 |
| Porz | DE | 0.0031 | 0.2807 |
| Wolfsburg | DE | 0.0042 | 0.1228 |
| Anzio | IT | 0.0057 | 0.2693 |
| Zamora | ES | 0.0065 | 0.2709 |
| Jena-Zentrum | DE | 0.0077 | 0.2730 |
| Aix-en-Provence | FR | 0.0087 | 0.2278 |
| Hilden | DE | 0.0090 | 0.2166 |
| Veenendaal | NL | 0.0096 | 0.2608 |
| Antwerpen | BE | 0.0105 | 0.1257 |
| Roosendaal | NL | 0.0106 | 0.3488 |
| Durach | DE | 0.0116 | 0.1730 |
| Perpignan | FR | 0.0123 | 0.1175 |
| Milano | IT | 0.0125 | 0.1557 |
| Neusäß | DE | 0.0142 | 0.1882 |
| Tours | FR | 0.0144 | 0.1541 |
| Saint-Julien-en-Genevois | CH | 0.0146 | 0.1208 |
| Zagreb | HR | 0.0153 | 0.1456 |
| Brackel | DE | 0.0158 | 0.2920 |
| Κατσικάς | GR | 0.0158 | 0.3111 |
| El Bierzo | ES | 0.0163 | 0.2655 |
| Grenoble | FR | 0.0170 | 0.1993 |
| Zielona Góra | PL | 0.0171 | 0.3045 |

---

## Model Performance by Category

| Category | R² Score | Local Importance | Intermediate Importance | Large Importance |
|----------|----------|------------------|-------------------------|------------------|
| Accommodation | 0.5572 | 0.4040 | 0.3371 | 0.2589 |
| Active Life | 0.6558 | 0.6385 | 0.2083 | 0.1532 |
| Arts And Entertainment | 0.6320 | 0.4509 | 0.3644 | 0.1848 |
| Attractions And Activities | 0.6029 | 0.2641 | 0.5289 | 0.2070 |
| Business And Services | 0.7283 | 0.7559 | 0.1398 | 0.1043 |
| Eat And Drink | 0.7148 | 0.7230 | 0.1549 | 0.1221 |
| Education | 0.7256 | 0.7232 | 0.1572 | 0.1197 |
| Health And Medical | 0.6933 | 0.7212 | 0.1435 | 0.1354 |
| Public Services | 0.6848 | 0.6385 | 0.2109 | 0.1506 |
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
