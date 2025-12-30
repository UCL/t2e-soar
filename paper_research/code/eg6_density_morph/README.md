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
- **Network Density**: Street network density (beta coefficient)
- **Mixed Uses**: Landuse richness (Hill number q=0)

See cluster rankings visualization for relative ordering.

### Cluster Characteristics

| ID | Nodes | % Total | Pop Density | Network Density | Mixed Use |
|----|-------|---------|-------------|-------------|-----------|
| 1 | 68,467.0 | 34.2% | 3811 | 99.90 | 0.36 |
| 2 | 20,297.0 | 10.1% | 12449 | 161.31 | 1.33 |
| 3 | 26,133.0 | 13.1% | 6724 | 129.19 | 0.87 |
| 4 | 11,813.0 | 5.9% | 5004 | 109.02 | 0.44 |
| 5 | 32,536.0 | 16.3% | 8715 | 120.97 | 0.75 |
| 6 | 20,141.0 | 10.1% | 11412 | 156.46 | 1.51 |
| 7 | 2,741.0 | 1.4% | 14161 | 139.72 | 0.76 |
| 8 | 17,872.0 | 8.9% | 5650 | 93.99 | 0.41 |


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
| IE | 5 | Cluster 1 | 61.1% |
| NL | 43 | Cluster 1 | 53.5% |
| NO | 3 | Cluster 1 | 48.7% |
| FR | 72 | Cluster 1 | 48.7% |
| BE | 13 | Cluster 1 | 46.0% |

### Top Countries by Cluster 2

| Country | # Cities | Dominant Cluster | % in Cluster 2 |
|---------|----------|------------------|---------------------|
| LT | 4 | Cluster 1 | 21.0% |
| SK | 6 | Cluster 1 | 19.7% |
| AT | 6 | Cluster 1 | 18.0% |
| EE | 3 | Cluster 5 | 16.2% |
| CZ | 14 | Cluster 1 | 16.1% |

### Top Countries by Cluster 3

| Country | # Cities | Dominant Cluster | % in Cluster 3 |
|---------|----------|------------------|---------------------|
| LU | 1 | Cluster 3 | 31.5% |
| DE | 103 | Cluster 1 | 21.2% |
| SK | 6 | Cluster 1 | 20.7% |
| CZ | 14 | Cluster 1 | 20.6% |
| CH | 12 | Cluster 5 | 17.6% |

### Top Countries by Cluster 4

| Country | # Cities | Dominant Cluster | % in Cluster 4 |
|---------|----------|------------------|---------------------|
| DK | 4 | Cluster 1 | 16.1% |
| FI | 4 | Cluster 1 | 11.6% |
| SE | 15 | Cluster 1 | 10.3% |
| FR | 72 | Cluster 1 | 9.4% |
| NO | 3 | Cluster 1 | 7.9% |

### Top Countries by Cluster 5

| Country | # Cities | Dominant Cluster | % in Cluster 5 |
|---------|----------|------------------|---------------------|
| CY | 2 | Cluster 5 | 47.6% |
| MT | 1 | Cluster 5 | 36.2% |
| HR | 5 | Cluster 5 | 34.4% |
| BG | 13 | Cluster 5 | 32.9% |
| CH | 12 | Cluster 5 | 32.5% |

### Top Countries by Cluster 6

| Country | # Cities | Dominant Cluster | % in Cluster 6 |
|---------|----------|------------------|---------------------|
| GR | 8 | Cluster 6 | 36.9% |
| BE | 13 | Cluster 1 | 22.3% |
| ES | 86 | Cluster 6 | 21.4% |
| NL | 43 | Cluster 1 | 19.6% |
| LU | 1 | Cluster 3 | 16.4% |

### Top Countries by Cluster 7

| Country | # Cities | Dominant Cluster | % in Cluster 7 |
|---------|----------|------------------|---------------------|
| LT | 4 | Cluster 1 | 5.6% |
| EE | 3 | Cluster 5 | 3.4% |
| ES | 86 | Cluster 6 | 2.4% |
| MT | 1 | Cluster 5 | 2.4% |
| IT | 83 | Cluster 5 | 2.3% |

### Top Countries by Cluster 8

| Country | # Cities | Dominant Cluster | % in Cluster 8 |
|---------|----------|------------------|---------------------|
| FI | 4 | Cluster 1 | 18.9% |
| SE | 15 | Cluster 1 | 18.7% |
| LT | 4 | Cluster 1 | 18.1% |
| EE | 3 | Cluster 5 | 17.1% |
| LV | 3 | Cluster 1 | 16.3% |

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

