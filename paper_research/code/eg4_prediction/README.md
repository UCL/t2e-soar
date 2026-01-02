# Amenity Supply Prediction

**Analysis Date:** 2026-01-02

## Vignette Purpose

Large-scale datasets from multiple cities can enable training of generalisable models.
This vignette uses network centrality and population density to predict commercial amenity
intensities, demonstrating how consistent feature sets across cities support transfer learning.

## Analysis Overview

Using 27 cities with saturated POI coverage (694,527 nodes), we train Extra Trees regressors to
predict 'Eat & Drink' and 'Business & Services' POI counts from multi-scale closeness centrality
(400m-4800m), betweenness centrality, and census demographics. Models achieve median R²=0.709-0.721
on held-out data, with 96-100% of cities exceeding R²>0.5. Feature importance analysis reveals
intermediate-scale centrality (1200-1600m) dominates predictions.

## Summary Statistics

- **Total Street Network Nodes Analyzed:** 694,527
- **Cities Analyzed:** 27
- **POI Categories:** 2
- **Network Centrality Scales:** 5 (400–4800m)
- **Census Features:** Population Density, Population under 15, Population 15-64, Population 65 and over, Employment Ratio

## Overall Model Performance

| Category                 | R² (train) | R² (test) | MAE (test) | RMSE (test) |
| ------------------------ | ---------- | --------- | ---------- | ----------- |
| Eat & Drink 400M         | 0.746      | 0.731     | 0.496      | 0.643       |
| Business & Services 400M | 0.747      | 0.731     | 0.585      | 0.755       |

## Per-City Prediction Accuracy

R² scores computed separately for each city show how well the model
generalizes across different urban contexts.

### Eat & Drink 400M

- **Median City R²:** 0.709
- **Mean City R²:** 0.704
- **R² Range:** 0.440 to 0.857
- **Cities with R² > 0.5:** 26 (96.3%)

#### Top 10 Best Predicted Cities

| City          | R²    | MAE   | RMSE  | N Nodes |
| ------------- | ----- | ----- | ----- | ------- |
| Bari          | 0.857 | 0.430 | 0.584 | 14,855  |
| la Safor      | 0.821 | 0.425 | 0.603 | 5,435   |
| Ragusa        | 0.819 | 0.375 | 0.512 | 6,693   |
| la Plana Alta | 0.816 | 0.448 | 0.628 | 10,265  |
| Alessandria   | 0.809 | 0.458 | 0.590 | 5,417   |
| Torino        | 0.790 | 0.485 | 0.639 | 70,391  |
| Cremona       | 0.780 | 0.431 | 0.565 | 6,138   |
| Gent          | 0.771 | 0.483 | 0.606 | 30,182  |
| Brescia       | 0.764 | 0.435 | 0.565 | 30,497  |
| Pavia         | 0.759 | 0.493 | 0.662 | 7,255   |

#### Top 10 Worst Predicted Cities

| City                 | R²    | MAE   | RMSE  | N Nodes |
| -------------------- | ----- | ----- | ----- | ------- |
| Heerlen              | 0.440 | 0.488 | 0.611 | 33,426  |
| Pordenone / Pordenon | 0.585 | 0.498 | 0.655 | 10,823  |
| Gallarate            | 0.586 | 0.479 | 0.606 | 17,720  |
| Bergamo              | 0.590 | 0.496 | 0.638 | 41,833  |
| Busto Arsizio        | 0.618 | 0.482 | 0.616 | 36,203  |
| Ravenna              | 0.626 | 0.524 | 0.664 | 8,922   |
| Lecco                | 0.646 | 0.444 | 0.637 | 7,149   |
| Modena               | 0.650 | 0.535 | 0.702 | 14,782  |
| Prato                | 0.657 | 0.468 | 0.613 | 20,100  |
| Liberec              | 0.668 | 0.458 | 0.599 | 16,157  |

#### R² Distribution

![R² Distribution](outputs/Eat%20&%20Drink%20400m_city_r2_distribution.png)

### Business & Services 400M

- **Median City R²:** 0.721
- **Mean City R²:** 0.721
- **R² Range:** 0.564 to 0.865
- **Cities with R² > 0.5:** 27 (100.0%)

#### Top 10 Best Predicted Cities

| City          | R²    | MAE   | RMSE  | N Nodes |
| ------------- | ----- | ----- | ----- | ------- |
| Ragusa        | 0.865 | 0.405 | 0.571 | 6,693   |
| Cremona       | 0.816 | 0.497 | 0.642 | 6,138   |
| Bari          | 0.809 | 0.549 | 0.725 | 14,855  |
| la Safor      | 0.804 | 0.550 | 0.759 | 5,435   |
| Alessandria   | 0.798 | 0.535 | 0.711 | 5,417   |
| Torino        | 0.792 | 0.558 | 0.739 | 70,391  |
| Pavia         | 0.789 | 0.567 | 0.722 | 7,255   |
| la Plana Alta | 0.779 | 0.583 | 0.797 | 10,265  |
| Bellizzi      | 0.766 | 0.605 | 0.772 | 5,715   |
| Milano        | 0.752 | 0.563 | 0.731 | 227,228 |

#### Top 10 Worst Predicted Cities

| City                 | R²    | MAE   | RMSE  | N Nodes |
| -------------------- | ----- | ----- | ----- | ------- |
| Modena               | 0.564 | 0.679 | 0.892 | 14,782  |
| Heerlen              | 0.594 | 0.566 | 0.726 | 33,426  |
| Pordenone / Pordenon | 0.622 | 0.601 | 0.771 | 10,823  |
| Prato                | 0.637 | 0.639 | 0.796 | 20,100  |
| Gallarate            | 0.661 | 0.566 | 0.704 | 17,720  |
| Bergamo              | 0.667 | 0.601 | 0.742 | 41,833  |
| Treviso              | 0.672 | 0.628 | 0.785 | 9,625   |
| Carpi                | 0.675 | 0.581 | 0.741 | 6,372   |
| Seiersberg-Pirka     | 0.677 | 0.601 | 0.757 | 37,891  |
| Liberec              | 0.682 | 0.542 | 0.700 | 16,157  |

#### R² Distribution

![R² Distribution](outputs/Business%20&%20Services%20400m_city_r2_distribution.png)

## Top Features by Category

### Eat & Drink 400M

| Rank | Feature      | Importance |
| ---- | ------------ | ---------- |
| 1    | cc_beta_1200 | 0.1733     |
| 2    | cc_beta_1600 | 0.1502     |
| 3    | y_1564       | 0.1257     |
| 4    | cc_beta_800  | 0.1172     |
| 5    | y_ge65       | 0.0919     |
| 6    | density      | 0.0824     |
| 7    | emp          | 0.0795     |
| 8    | y_lt15       | 0.0724     |
| 9    | cc_beta_4800 | 0.0488     |
| 10   | cc_beta_400  | 0.0317     |

![Feature Importance](outputs/Eat%20&%20Drink%20400m_feature_importance.png)

### Business & Services 400M

| Rank | Feature      | Importance |
| ---- | ------------ | ---------- |
| 1    | cc_beta_1200 | 0.2171     |
| 2    | cc_beta_1600 | 0.1700     |
| 3    | cc_beta_800  | 0.1460     |
| 4    | y_1564       | 0.0973     |
| 5    | y_lt15       | 0.0683     |
| 6    | emp          | 0.0652     |
| 7    | y_ge65       | 0.0645     |
| 8    | density      | 0.0559     |
| 9    | cc_beta_4800 | 0.0437     |
| 10   | cc_beta_400  | 0.0404     |

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
