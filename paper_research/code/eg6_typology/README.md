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
| 1 | 18,957.0 | 9.5% | 13122 | 1072.20 | 4.27 |
| 2 | 42,642.0 | 21.3% | 4721 | 976.38 | 2.84 |
| 3 | 23,955.0 | 12.0% | 7457 | 1169.95 | 4.46 |
| 4 | 23,203.0 | 11.6% | 13441 | 1324.84 | 6.80 |
| 5 | 62,839.0 | 31.4% | 4127 | 788.02 | 2.14 |
| 6 | 7,268.0 | 3.6% | 5105 | 902.44 | 2.42 |
| 7 | 10,928.0 | 5.5% | 10931 | 1230.90 | 4.71 |
| 8 | 10,208.0 | 5.1% | 4434 | 857.63 | 2.38 |


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
| MT | 1 | Cluster 5 | 21.3% |
| EE | 3 | Cluster 5 | 20.8% |
| IT | 83 | Cluster 5 | 20.0% |
| LV | 3 | Cluster 5 | 19.6% |
| LT | 4 | Cluster 5 | 18.6% |

### Top Countries by Cluster 2

| Country | # Cities | Dominant Cluster | % in Cluster 2 |
|---------|----------|------------------|---------------------|
| NL | 43 | Cluster 2 | 57.4% |
| IE | 5 | Cluster 2 | 52.0% |
| FR | 72 | Cluster 2 | 42.8% |
| BE | 13 | Cluster 2 | 41.4% |
| DE | 103 | Cluster 5 | 27.4% |

### Top Countries by Cluster 3

| Country | # Cities | Dominant Cluster | % in Cluster 3 |
|---------|----------|------------------|---------------------|
| SK | 6 | Cluster 3 | 25.7% |
| LU | 1 | Cluster 3 | 22.3% |
| CZ | 14 | Cluster 5 | 21.4% |
| DE | 103 | Cluster 5 | 19.0% |
| PL | 50 | Cluster 5 | 18.9% |

### Top Countries by Cluster 4

| Country | # Cities | Dominant Cluster | % in Cluster 4 |
|---------|----------|------------------|---------------------|
| GR | 8 | Cluster 4 | 30.8% |
| ES | 86 | Cluster 4 | 27.1% |
| BE | 13 | Cluster 2 | 20.3% |
| LU | 1 | Cluster 3 | 19.7% |
| AT | 6 | Cluster 5 | 16.6% |

### Top Countries by Cluster 5

| Country | # Cities | Dominant Cluster | % in Cluster 5 |
|---------|----------|------------------|---------------------|
| DK | 4 | Cluster 5 | 65.2% |
| CY | 2 | Cluster 5 | 65.1% |
| FI | 4 | Cluster 5 | 59.8% |
| SI | 2 | Cluster 5 | 58.3% |
| HU | 12 | Cluster 5 | 56.9% |

### Top Countries by Cluster 6

| Country | # Cities | Dominant Cluster | % in Cluster 6 |
|---------|----------|------------------|---------------------|
| DK | 4 | Cluster 5 | 8.3% |
| SE | 15 | Cluster 5 | 8.2% |
| LT | 4 | Cluster 5 | 6.7% |
| FI | 4 | Cluster 5 | 6.6% |
| EE | 3 | Cluster 5 | 6.6% |

### Top Countries by Cluster 7

| Country | # Cities | Dominant Cluster | % in Cluster 7 |
|---------|----------|------------------|---------------------|
| BG | 13 | Cluster 5 | 14.8% |
| LT | 4 | Cluster 5 | 14.1% |
| EE | 3 | Cluster 5 | 13.0% |
| LV | 3 | Cluster 5 | 12.1% |
| SK | 6 | Cluster 3 | 10.7% |

### Top Countries by Cluster 8

| Country | # Cities | Dominant Cluster | % in Cluster 8 |
|---------|----------|------------------|---------------------|
| IE | 5 | Cluster 2 | 12.1% |
| FR | 72 | Cluster 2 | 8.2% |
| DE | 103 | Cluster 5 | 6.4% |
| SK | 6 | Cluster 3 | 6.2% |
| CZ | 14 | Cluster 5 | 6.2% |

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

