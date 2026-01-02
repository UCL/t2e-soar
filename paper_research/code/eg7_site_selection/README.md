# Site Selection for Development Opportunities

**Analysis Date:** 2026-01-02

## Overview

This analysis identifies candidate locations for development by filtering street network
nodes based on multiple criteria: centrality, land-use diversity, transport access, and
current population density. Nodes with high connectivity and diversity but lower
population density represent potential opportunities for sustainable densification.

## Summary Statistics

- **Cities Analyzed:** 339
- **Total Street Network Nodes:** 9,374,220
- **Mean Nodes per City:** 27653
- **Minimum Nodes per City:** 100

## Typology Distribution

Average percentage of nodes in each typology across all cities:

- **Mixed-Use Dense:** 18.8% (high diversity + high density)
- **Mixed-Use Opportunity:** 1.2% (high diversity + low density)
- **Single-Use Dense:** 2.4% (low diversity + high density)
- **Single-Use Low-Density:** 19.4% (low diversity + low density)
- **Intermediate:** 58.2% (all other combinations)

## Development Opportunity Statistics

Nodes classified as development opportunities meet all criteria:

- High diversity score (≥70th percentile)
- High network centrality (≥70th percentile)
- Good transport access (≥70th percentile)
- Low population density (≤30th percentile)

- **Mean % Opportunity Nodes:** 0.74%
- **Median % Opportunity Nodes:** 0.00%
- **Range:** 0.00% to 21.32%

## Top 10 Mixed-Use Cities

Cities with highest mixed-use character score (combination of diversity and mixed-use typology proportions):

| Rank | City                          | Country | Mixed-Use Score | % Mixed Dense | % Mixed Opp. |
| ---- | ----------------------------- | ------- | --------------- | ------------- | ------------ |
| 1    | Den Haag                      | NL      | 0.42            | 30.0          | 0.0          |
| 2    | Leiden                        | NL      | 0.42            | 30.0          | 0.0          |
| 3    | Venezia                       | IT      | 0.41            | 8.3           | 3.4          |
| 4    | Νέα Αλικαρνασσός              | GR      | 0.40            | 21.5          | 1.7          |
| 5    | La Rochelle                   | FR      | 0.40            | 30.0          | 0.0          |
| 6    | Grasse                        | FR      | 0.40            | 30.0          | 0.0          |
| 7    | Bayonne                       | FR      | 0.39            | 30.0          | 0.0          |
| 8    | Bahía de Cádiz                | ES      | 0.39            | 7.6           | 5.1          |
| 9    | el Baix Segura / La Vega Baja | ES      | 0.38            | 20.2          | 2.7          |
| 10   | Bari                          | IT      | 0.38            | 19.3          | 1.1          |

## Top 10 Development Opportunity Cities

Cities with highest proportion of nodes identified as development opportunities:

| Rank | City        | Country | % Opportunity | Mean Centrality | Mean Transport |
| ---- | ----------- | ------- | ------------- | --------------- | -------------- |
| 1    | Grasse      | FR      | 21.3%         | 0.37            | 0.50           |
| 2    | Draguignan  | FR      | 19.8%         | 0.39            | 0.50           |
| 3    | Grasse      | FR      | 19.3%         | 0.37            | 0.50           |
| 4    | Bayonne     | FR      | 18.7%         | 0.49            | 0.50           |
| 5    | Cherbourg   | FR      | 18.5%         | 0.50            | 0.50           |
| 6    | Vannes      | FR      | 17.6%         | 0.56            | 0.50           |
| 7    | Søholt      | DK      | 17.1%         | 0.48            | 0.50           |
| 8    | Rætebøl     | DK      | 17.0%         | 0.48            | 0.50           |
| 9    | Toulon      | FR      | 16.9%         | 0.44            | 0.50           |
| 10   | La Rochelle | FR      | 16.3%         | 0.51            | 0.50           |

## Methodology

### Node Typology Classification

Each street network node is classified based on within-city percentile thresholds:

- **Mixed-Use Dense**: High diversity (≥70th percentile) + High density (≥70th percentile)
- **Mixed-Use Opportunity**: High diversity + High centrality + Good transport access, but Low density (≤30th percentile)
- **Single-Use Dense**: Low diversity (≤30th percentile) + High density
- **Single-Use Low-Density**: Low diversity + Low density
- **Intermediate**: All other combinations

### Diversity Scoring

Composite diversity score computed from three Hill numbers at 400m scale:

- `cc_hill_q0_400_nw`: Species richness (count of distinct land-use types)
- `cc_hill_q1_400_nw`: Exponential Shannon entropy (balanced diversity)
- `cc_hill_q2_400_nw`: Inverse Simpson index (dominance-adjusted)

Each metric is normalized within-city to [0,1] and averaged to create diversity score.

### Development Opportunity Criteria

Nodes flagged as 'Mixed-Use Opportunity' must simultaneously meet:

1. High diversity score (≥70th percentile within city)
2. High network centrality at 1,600m scale (≥70th percentile)
3. Good transport access—low distance to nearest stop (≥70th percentile accessibility)
4. Low current population density (≤30th percentile)

## Key Outputs

### Data Files

- **city_site_profiles.csv**: Per-city summary with typology proportions

### Visualization Files

- **city_mixed_use_ranking.png**: Top 20 cities by mixed-use score
- **city_opportunity_ranking.png**: Top 20 cities by development opportunity proportion
- **typology_distribution.png**: Average typology distribution across all analyzed cities

### LaTeX Tables

- **table_mixed_use_cities.tex**: Top 10 mixed-use cities
- **table_opportunity_cities.tex**: Top 10 development opportunity cities

## Interpretation

### Mixed-Use Cities

Cities with high proportions of nodes classified as 'Mixed-Use Dense' demonstrate
fine-grained integration of residential, commercial, and service functions. These
cities typically have pedestrian-oriented development patterns with diverse amenities
accessible within short walking distances.

### Development Opportunities

Locations classified as 'Mixed-Use Opportunity' represent areas where infrastructure
(street network connectivity, land-use diversity, transport access) supports higher
utilization than current population density suggests. These are not recommendations
for development—such decisions require local planning knowledge, market analysis,
zoning compatibility, and community input—but rather a filtering mechanism to identify
areas warranting further investigation.

## Caveats

1. **Threshold sensitivity**: 70th/30th percentile cutoffs are illustrative; different
   thresholds would identify different opportunity areas.

2. **Within-city normalization**: All metrics are normalized within each city, so
   'high diversity' is relative to that city's range, not an absolute standard.

3. **Infrastructure ≠ suitability**: High connectivity and diversity do not imply that
   densification is appropriate—environmental constraints, heritage protection,
   infrastructure capacity, and community preferences must be considered.

4. **POI data quality**: Relies on cities with reliable POI coverage from EG1
   saturation analysis.

## Reproducibility

Code: `eg7_site_selection.py`  
Outputs: `outputs/`

All processing is deterministic and uses fixed thresholds. The analysis can be
regenerated by running the Python script with access to SOAR metrics and EG1
saturation results.
