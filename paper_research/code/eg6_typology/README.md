# EG6: Urban Density and Building Morphology Patterns

## Summary

Analysis of building morphology patterns across 654 European cities grouped by 30 countries.
We cluster **nodes** by morphology profile, then characterize each **country** by the proportion
of its nodes in each cluster type. This reveals international morphology patterns.

## Methodology

1. Sample max(5,000, 25%) nodes per city
2. Cluster nodes by 8 morphology features (BIRCH, k=8):
   - Building Count, Block Count, Mean Height, Height Variation, Building Area, Fractal Dimension, Block Coverage, Shared Walls
3. For each country, aggregate proportion of nodes in each morphology cluster
4. Countries are characterized by their full cluster proportion profile
5. Countries are hierarchically clustered by their composition similarity


## Node Morphology Clusters

BIRCH clustering (k=8) on 8 morphology features identified these node types.

**External characterization** uses three independent variables:
- **Population Density**: Average density (people/km²)
- **Network Density**: Street network density (density coefficient)
- **Mixed Uses**: Landuse richness (Hill number q=0)

See cluster rankings visualization for relative ordering.

### Cluster Characteristics

| ID | Nodes | % Total | Pop Density | Network Density | Mixed Use |
|----|-------|---------|-------------|-------------|-----------|
| 1 | 10,641.0 | 5.3% | 5405 | 909.86 | 2.50 |
| 2 | 21,660.0 | 10.8% | 13032 | 1343.16 | 6.87 |
| 3 | 16,778.0 | 8.4% | 12662 | 1277.75 | 5.08 |
| 4 | 26,133.0 | 13.1% | 6724 | 1076.86 | 4.15 |
| 5 | 5,513.0 | 2.8% | 8045 | 920.80 | 2.89 |
| 6 | 39,268.0 | 19.6% | 4308 | 969.35 | 2.72 |
| 7 | 72,122.0 | 36.1% | 4501 | 815.96 | 2.27 |
| 8 | 7,885.0 | 3.9% | 17180 | 1099.07 | 4.96 |


### Cluster External Characteristics
![Cluster Rankings](outputs/cluster_external_rankings.png)

### Cluster Profiles (Radar Plot)
![Cluster Radar Profiles](outputs/cluster_radar_profiles.png)

### Feature-Cluster Correlations
![Cluster Feature Correlations](outputs/cluster_feature_correlations.png)

## Country Morphology Profiles

Each country is characterized by its distribution across node clusters.

### Hierarchically Ordered Heatmap
Countries ordered by compositional similarity (Ward linkage):
![Country Profile Heatmap](outputs/country_profile_heatmap.png)

### Stacked Composition Chart
![Country Composition Stacked](outputs/country_composition_stacked.png)

### City Profiles by Cluster Proportions
![City Profiles](outputs/city_profiles_clusters.png)

## Morphology Patterns by Country

Countries are characterized by their distribution across morphology clusters.

### Top Countries by Cluster 1

| Country | # Cities | Dominant Cluster | % in Cluster 1 |
|---------|----------|------------------|---------------------|
| DK | 4 | Cluster 7 | 12.6% |
| SE | 15 | Cluster 7 | 11.2% |
| FI | 4 | Cluster 7 | 11.0% |
| EE | 3 | Cluster 7 | 9.4% |
| LT | 4 | Cluster 7 | 8.9% |

### Top Countries by Cluster 2

| Country | # Cities | Dominant Cluster | % in Cluster 2 |
|---------|----------|------------------|---------------------|
| GR | 8 | Cluster 2 | 37.1% |
| ES | 86 | Cluster 7 | 23.9% |
| BE | 13 | Cluster 6 | 19.8% |
| LU | 1 | Cluster 4 | 18.1% |
| PT | 10 | Cluster 7 | 15.1% |

### Top Countries by Cluster 3

| Country | # Cities | Dominant Cluster | % in Cluster 3 |
|---------|----------|------------------|---------------------|
| LT | 4 | Cluster 7 | 24.4% |
| EE | 3 | Cluster 7 | 18.4% |
| LV | 3 | Cluster 7 | 16.9% |
| SK | 6 | Cluster 7 | 16.5% |
| BG | 13 | Cluster 7 | 16.3% |

### Top Countries by Cluster 4

| Country | # Cities | Dominant Cluster | % in Cluster 4 |
|---------|----------|------------------|---------------------|
| LU | 1 | Cluster 4 | 31.5% |
| DE | 103 | Cluster 7 | 21.2% |
| SK | 6 | Cluster 7 | 20.7% |
| CZ | 14 | Cluster 7 | 20.6% |
| CH | 12 | Cluster 7 | 17.6% |

### Top Countries by Cluster 5

| Country | # Cities | Dominant Cluster | % in Cluster 5 |
|---------|----------|------------------|---------------------|
| LV | 3 | Cluster 7 | 10.5% |
| EE | 3 | Cluster 7 | 8.3% |
| LT | 4 | Cluster 7 | 7.8% |
| SE | 15 | Cluster 7 | 6.8% |
| LU | 1 | Cluster 4 | 5.5% |

### Top Countries by Cluster 6

| Country | # Cities | Dominant Cluster | % in Cluster 6 |
|---------|----------|------------------|---------------------|
| NL | 43 | Cluster 6 | 59.9% |
| IE | 5 | Cluster 6 | 55.1% |
| FR | 72 | Cluster 6 | 40.0% |
| BE | 13 | Cluster 6 | 39.0% |
| DE | 103 | Cluster 7 | 25.7% |

### Top Countries by Cluster 7

| Country | # Cities | Dominant Cluster | % in Cluster 7 |
|---------|----------|------------------|---------------------|
| CY | 2 | Cluster 7 | 73.5% |
| DK | 4 | Cluster 7 | 69.3% |
| FI | 4 | Cluster 7 | 63.8% |
| SI | 2 | Cluster 7 | 62.2% |
| HU | 12 | Cluster 7 | 61.4% |

### Top Countries by Cluster 8

| Country | # Cities | Dominant Cluster | % in Cluster 8 |
|---------|----------|------------------|---------------------|
| MT | 1 | Cluster 7 | 16.1% |
| GR | 8 | Cluster 2 | 12.8% |
| IT | 83 | Cluster 7 | 11.6% |
| BG | 13 | Cluster 7 | 9.1% |
| CY | 2 | Cluster 7 | 7.3% |

## Outputs

### Data Files
- `country_morphology_profiles.csv`: Full country profiles with cluster proportions
- `node_cluster_summary.csv`: Node cluster characteristics
- `cluster_external_characterization.csv`: External metrics (density, network density, mixed uses) per cluster
- `cluster_external_correlations.csv`: Cluster-external metric correlation matrix
- `top_cluster_1-8_countries.csv`: Top countries by each cluster type
- `cluster_representatives.csv`: Representative city for each cluster

### Visualizations
- `cluster_radar_profiles.png`: Individual radar plots for each cluster
- `cluster_external_rankings.png`: Clusters ranked by external characteristics
- `cluster_feature_correlations.png`: Heatmap of cluster-feature correlations
- `node_cluster_profiles.png`: Bar charts of raw cluster means
- `country_profile_heatmap.png`: Hierarchically ordered country composition heatmap
- `country_composition_stacked.png`: Stacked bar chart of country compositions
- `city_profiles_clusters.png`: Cities plotted by contrasting cluster proportions
- `cluster_1-8_satellite_5x5.jpg`: Satellite imagery exemplars for each cluster

