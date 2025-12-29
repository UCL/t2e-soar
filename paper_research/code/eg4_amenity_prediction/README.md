# Amenity Supply Prediction Using Network Centrality

**Analysis Date:** 2025-12-29

## Overview

This analysis uses Extra Trees regression to predict amenity supply based on
multi-scale network centrality metrics and census variables. The goal is to demonstrate that street
network structure can predict POI distributions, with per-city accuracy metrics
showing how well this relationship generalizes across different urban contexts.

## Summary Statistics

- **Total Street Network Nodes Analyzed:** 694,527
- **Cities Analyzed:** 27
- **POI Categories:** 2
- **Network Centrality Scales:** 5 (400–4800m)
- **Census Features:** Population Density, Population under 15, Population 15-64, Population 65 and over, Employment Ratio

## Overall Model Performance

| Category | R² (train) | R² (test) | MAE (test) | RMSE (test) |
|----------|-----------|----------|-----------|------------|
| Eat & Drink 400M | 0.746 | 0.731 | 0.496 | 0.642 |
| Business & Services 400M | 0.747 | 0.731 | 0.585 | 0.755 |

## Per-City Prediction Accuracy

R² scores computed separately for each city show how well the model
generalizes across different urban contexts.

### Eat & Drink 400M

- **Median City R²:** 0.709
- **Mean City R²:** 0.704
- **R² Range:** 0.439 to 0.857
- **Cities with R² > 0.5:** 26 (96.3%)

#### Top 10 Best Predicted Cities

| City | R² | MAE | RMSE | N Nodes |
|------|-----|-----|------|---------|
| Bari | 0.857 | 0.430 | 0.584 | 14,855 |
| la Safor | 0.822 | 0.423 | 0.601 | 5,435 |
| Ragusa | 0.819 | 0.374 | 0.511 | 6,693 |
| la Plana Alta | 0.817 | 0.446 | 0.627 | 10,265 |
| Alessandria | 0.809 | 0.457 | 0.590 | 5,417 |
| Torino | 0.790 | 0.485 | 0.638 | 70,391 |
| Cremona | 0.780 | 0.431 | 0.565 | 6,138 |
| Gent | 0.772 | 0.482 | 0.605 | 30,182 |
| Brescia | 0.764 | 0.434 | 0.564 | 30,497 |
| Pavia | 0.760 | 0.492 | 0.660 | 7,255 |

#### Top 10 Worst Predicted Cities

| City | R² | MAE | RMSE | N Nodes |
|------|-----|-----|------|---------|
| Heerlen | 0.439 | 0.488 | 0.612 | 33,426 |
| Pordenone / Pordenon | 0.582 | 0.499 | 0.657 | 10,823 |
| Gallarate | 0.585 | 0.479 | 0.606 | 17,720 |
| Bergamo | 0.590 | 0.496 | 0.638 | 41,833 |
| Busto Arsizio | 0.617 | 0.482 | 0.616 | 36,203 |
| Ravenna | 0.626 | 0.524 | 0.664 | 8,922 |
| Lecco | 0.648 | 0.444 | 0.636 | 7,149 |
| Modena | 0.651 | 0.534 | 0.701 | 14,782 |
| Prato | 0.657 | 0.468 | 0.613 | 20,100 |
| Vigevano | 0.668 | 0.478 | 0.629 | 6,019 |

#### R² Distribution

![R² Distribution](outputs/Eat%20&%20Drink%20400m_city_r2_distribution.png)

### Business & Services 400M

- **Median City R²:** 0.721
- **Mean City R²:** 0.721
- **R² Range:** 0.565 to 0.865
- **Cities with R² > 0.5:** 27 (100.0%)

#### Top 10 Best Predicted Cities

| City | R² | MAE | RMSE | N Nodes |
|------|-----|-----|------|---------|
| Ragusa | 0.865 | 0.406 | 0.570 | 6,693 |
| Cremona | 0.816 | 0.499 | 0.643 | 6,138 |
| Bari | 0.810 | 0.548 | 0.724 | 14,855 |
| la Safor | 0.804 | 0.550 | 0.759 | 5,435 |
| Alessandria | 0.799 | 0.535 | 0.710 | 5,417 |
| Torino | 0.792 | 0.558 | 0.739 | 70,391 |
| Pavia | 0.789 | 0.566 | 0.722 | 7,255 |
| la Plana Alta | 0.777 | 0.585 | 0.801 | 10,265 |
| Bellizzi | 0.768 | 0.600 | 0.768 | 5,715 |
| Milano | 0.752 | 0.563 | 0.731 | 227,228 |

#### Top 10 Worst Predicted Cities

| City | R² | MAE | RMSE | N Nodes |
|------|-----|-----|------|---------|
| Modena | 0.565 | 0.679 | 0.892 | 14,782 |
| Heerlen | 0.595 | 0.566 | 0.726 | 33,426 |
| Pordenone / Pordenon | 0.623 | 0.600 | 0.771 | 10,823 |
| Prato | 0.636 | 0.639 | 0.797 | 20,100 |
| Gallarate | 0.661 | 0.565 | 0.704 | 17,720 |
| Bergamo | 0.667 | 0.601 | 0.742 | 41,833 |
| Treviso | 0.673 | 0.626 | 0.783 | 9,625 |
| Carpi | 0.676 | 0.580 | 0.741 | 6,372 |
| Seiersberg-Pirka | 0.677 | 0.600 | 0.756 | 37,891 |
| Liberec | 0.682 | 0.542 | 0.701 | 16,157 |

#### R² Distribution

![R² Distribution](outputs/Business%20&%20Services%20400m_city_r2_distribution.png)

## Top Features by Category

### Eat & Drink 400M

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | cc_beta_1200 | 0.1707 |
| 2 | cc_beta_1600 | 0.1572 |
| 3 | y_1564 | 0.1170 |
| 4 | cc_beta_800 | 0.1150 |
| 5 | y_ge65 | 0.0918 |
| 6 | density | 0.0853 |
| 7 | emp | 0.0833 |
| 8 | y_lt15 | 0.0694 |
| 9 | cc_beta_4800 | 0.0531 |
| 10 | cc_beta_400 | 0.0307 |

![Feature Importance](outputs/Eat%20&%20Drink%20400m_feature_importance.png)

### Business & Services 400M

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | cc_beta_1200 | 0.2140 |
| 2 | cc_beta_1600 | 0.1695 |
| 3 | cc_beta_800 | 0.1466 |
| 4 | y_1564 | 0.0996 |
| 5 | emp | 0.0745 |
| 6 | y_lt15 | 0.0677 |
| 7 | y_ge65 | 0.0593 |
| 8 | density | 0.0529 |
| 9 | cc_beta_4800 | 0.0451 |
| 10 | cc_beta_400 | 0.0402 |

![Feature Importance](outputs/Business%20&%20Services%20400m_feature_importance.png)

## Methods

### Feature Engineering
- Closeness centrality at 5 scales (400–4800m)
- Betweenness centrality at 5 scales (400–4800m)
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
