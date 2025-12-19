# Amenity Supply Prediction Using Network Centrality

**Analysis Date:** 2025-12-19

## Overview

This analysis uses Extra Trees regression to predict amenity supply based on
multi-scale network centrality metrics and census variables. The goal is to demonstrate that street
network structure can predict POI distributions, with per-city accuracy metrics
showing how well this relationship generalizes across different urban contexts.

## Summary Statistics

- **Total Street Network Nodes Analyzed:** 999,180
- **Cities Analyzed:** 21
- **POI Categories:** 2
- **Network Centrality Scales:** 6 (400–9600m)
- **Census Features:** Population Density, Population under 15, Population 15-64, Population 65 and over, Employment Ratio

## Overall Model Performance

| Category | R² (train) | R² (test) | MAE (test) | RMSE (test) |
|----------|-----------|----------|-----------|------------|
| Eat & Drink 400M | 0.740 | 0.723 | 0.447 | 0.603 |
| Business & Services 400M | 0.738 | 0.724 | 0.590 | 0.778 |

## Per-City Prediction Accuracy

R² scores computed separately for each city show how well the model
generalizes across different urban contexts.

### Eat & Drink 400M

- **Median City R²:** 0.724
- **Mean City R²:** 0.685
- **R² Range:** 0.346 to 0.821
- **Cities with R² > 0.5:** 19 (90.5%)

#### Top 10 Best Predicted Cities

| City | R² | MAE | RMSE | N Nodes |
|------|-----|-----|------|---------|
| Alessandria | 0.821 | 0.408 | 0.576 | 7,280 |
| Maresme | 0.815 | 0.407 | 0.598 | 8,539 |
| Torino | 0.807 | 0.423 | 0.588 | 110,122 |
| Pavia | 0.778 | 0.425 | 0.617 | 9,291 |
| Cremona | 0.776 | 0.408 | 0.556 | 7,784 |
| Milano | 0.766 | 0.445 | 0.593 | 387,236 |
| Brescia | 0.765 | 0.357 | 0.512 | 55,308 |
| Seiersberg-Pirka | 0.764 | 0.391 | 0.504 | 53,854 |
| Liberec | 0.738 | 0.375 | 0.521 | 20,425 |
| Modena | 0.725 | 0.464 | 0.619 | 18,207 |

#### Top 10 Worst Predicted Cities

| City | R² | MAE | RMSE | N Nodes |
|------|-----|-----|------|---------|
| Gallarate | 0.346 | 0.493 | 0.681 | 31,654 |
| Heerlen | 0.448 | 0.461 | 0.595 | 43,911 |
| Busto Arsizio | 0.551 | 0.472 | 0.619 | 66,754 |
| Bergamo | 0.588 | 0.451 | 0.618 | 65,130 |
| Prato | 0.645 | 0.421 | 0.585 | 30,848 |
| Thalheim bei Wels | 0.652 | 0.411 | 0.543 | 12,880 |
| Pordenone / Pordenon | 0.653 | 0.390 | 0.545 | 16,929 |
| Ravenna | 0.674 | 0.465 | 0.621 | 11,673 |
| Halle-Vilvoorde | 0.675 | 0.447 | 0.573 | 19,640 |
| Treviso | 0.678 | 0.495 | 0.641 | 13,430 |

#### R² Distribution

![R² Distribution](outputs/Eat%20&%20Drink%20400m_city_r2_distribution.png)

### Business & Services 400M

- **Median City R²:** 0.693
- **Mean City R²:** 0.695
- **R² Range:** 0.398 to 0.835
- **Cities with R² > 0.5:** 20 (95.2%)

#### Top 10 Best Predicted Cities

| City | R² | MAE | RMSE | N Nodes |
|------|-----|-----|------|---------|
| Maresme | 0.835 | 0.519 | 0.711 | 8,539 |
| Cremona | 0.810 | 0.487 | 0.660 | 7,784 |
| Alessandria | 0.802 | 0.519 | 0.744 | 7,280 |
| Torino | 0.801 | 0.559 | 0.759 | 110,122 |
| Pavia | 0.783 | 0.524 | 0.732 | 9,291 |
| Liberec | 0.768 | 0.475 | 0.621 | 20,425 |
| Brescia | 0.767 | 0.514 | 0.699 | 55,308 |
| Milano | 0.760 | 0.586 | 0.771 | 387,236 |
| Seiersberg-Pirka | 0.747 | 0.537 | 0.682 | 53,854 |
| Ravenna | 0.723 | 0.554 | 0.724 | 11,673 |

#### Top 10 Worst Predicted Cities

| City | R² | MAE | RMSE | N Nodes |
|------|-----|-----|------|---------|
| Gallarate | 0.398 | 0.656 | 0.906 | 31,654 |
| Busto Arsizio | 0.590 | 0.624 | 0.809 | 66,754 |
| Treviso | 0.595 | 0.696 | 0.881 | 13,430 |
| Modena | 0.611 | 0.648 | 0.879 | 18,207 |
| Heerlen | 0.629 | 0.543 | 0.702 | 43,911 |
| Halle-Vilvoorde | 0.631 | 0.568 | 0.717 | 19,640 |
| Prato | 0.642 | 0.623 | 0.810 | 30,848 |
| Pordenone / Pordenon | 0.647 | 0.566 | 0.753 | 16,929 |
| Bergamo | 0.671 | 0.582 | 0.737 | 65,130 |
| Carpi | 0.685 | 0.567 | 0.762 | 8,285 |

#### R² Distribution

![R² Distribution](outputs/Business%20&%20Services%20400m_city_r2_distribution.png)

## Top Features by Category

### Eat & Drink 400M

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | cc_beta_1600 | 0.1222 |
| 2 | cc_beta_1200 | 0.1190 |
| 3 | emp | 0.1104 |
| 4 | density | 0.1093 |
| 5 | y_ge65 | 0.1055 |
| 6 | y_1564 | 0.0983 |
| 7 | cc_beta_800 | 0.0948 |
| 8 | cc_beta_4800 | 0.0667 |
| 9 | y_lt15 | 0.0661 |
| 10 | cc_beta_9600 | 0.0570 |

![Feature Importance](outputs/Eat%20&%20Drink%20400m_feature_importance.png)

### Business & Services 400M

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | cc_beta_1600 | 0.1486 |
| 2 | cc_beta_1200 | 0.1456 |
| 3 | y_1564 | 0.0993 |
| 4 | cc_beta_800 | 0.0984 |
| 5 | emp | 0.0950 |
| 6 | density | 0.0897 |
| 7 | y_ge65 | 0.0847 |
| 8 | cc_beta_4800 | 0.0788 |
| 9 | y_lt15 | 0.0531 |
| 10 | cc_beta_9600 | 0.0450 |

![Feature Importance](outputs/Business%20&%20Services%20400m_feature_importance.png)

## Methods

### Feature Engineering
- Closeness centrality at 6 scales (400–9600m)
- Betweenness centrality at 6 scales (400–9600m)
- Census features:
  - Population Density
  - Population under 15
  - Population 15-64
  - Population 65 and over
  - Employment Ratio
- Log-transformation of all features to stabilize variance

### Model Training
- Algorithm: Extra Trees (100 estimators)
- Max depth: 20
- Test size: 10%
- Stratified splitting by city to ensure representative train/test sets

## Output Files

### Eat & Drink 400M

- `Eat & Drink 400m_city_accuracy.csv`: Per-city R², MAE, RMSE
- `Eat & Drink 400m_best_predicted_cities.csv`: Top 20 cities by R²
- `Eat & Drink 400m_worst_predicted_cities.csv`: Bottom 20 cities by R²
- `Eat & Drink 400m_feature_importance.png`: Feature importance plot
- `Eat & Drink 400m_city_r2_distribution.png`: Distribution of city R² scores

### Business & Services 400M

- `Business & Services 400m_city_accuracy.csv`: Per-city R², MAE, RMSE
- `Business & Services 400m_best_predicted_cities.csv`: Top 20 cities by R²
- `Business & Services 400m_worst_predicted_cities.csv`: Bottom 20 cities by R²
- `Business & Services 400m_feature_importance.png`: Feature importance plot
- `Business & Services 400m_city_r2_distribution.png`: Distribution of city R² scores
