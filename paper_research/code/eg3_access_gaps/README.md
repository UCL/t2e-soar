# Access Gap Analysis: Education and Transport

**Analysis Date:** 2026-01-02

## Vignette Purpose

Distance-to-nearest metrics can reveal locations where distances to amenities or services
are greater than average or exceed targeted thresholds. This vignette examines two types
of access gaps: **education** (spatial equity and absolute access) and **transport**
(demand-supply mismatches).

## Analysis Overview

This analysis examines two complementary types of access gaps:

**Education Access:** For cities with reliable POI coverage, we compute mean/median distances
to nearest schools, P75/P25 equity ratios, and identify severely underserved areas (>2× city mean).
This reveals both absolute access levels and within-city spatial equity.

**Transport Gaps:** We identify high-demand locations (top 30% centrality+density) with poor supply
(bottom 30% proximity), flagging critical gaps where supply falls below the 15th percentile.
This highlights demand-supply mismatches requiring infrastructure intervention.

---

## Part 1: Education Access Analysis

### Summary Statistics

- **Cities Analyzed:** 123 (saturated education POI coverage only)
- **Total Street Network Nodes:** 2,434,058
- **Mean Distance to Education (cross-city):** 491.9m
- **Median Distance to Education (cross-city):** 493.2m

### Cross-City Comparison

#### Best Access (Top 10 Cities)

| City          | Country | Mean Dist (m) | % within 400m |
| ------------- | ------- | ------------- | ------------- |
| Küsnacht (ZH) | CH      | 327.7         | 69.8%         |
| Płock         | PL      | 376.1         | 69.4%         |
| Hoorn         | NL      | 387.9         | 64.3%         |
| A Coruña      | ES      | 394.7         | 61.8%         |
| Leiden        | NL      | 395.6         | 65.0%         |
| Almere        | NL      | 396.8         | 65.3%         |
| Rotterdam     | NL      | 399.1         | 63.4%         |
| Kalisz        | PL      | 399.4         | 66.1%         |
| Częstochowa   | PL      | 399.4         | 63.3%         |
| Nieuwegein    | NL      | 403.4         | 63.2%         |

#### Worst Access (Bottom 10 Cities)

| City                                                    | Country | Mean Dist (m) | % within 400m |
| ------------------------------------------------------- | ------- | ------------- | ------------- |
| Aschaffenburg                                           | DE      | 582.8         | 44.1%         |
| Lüdenscheid                                             | DE      | 583.8         | 41.2%         |
| Sosnowiec                                               | PL      | 593.0         | 39.9%         |
| Wołomin                                                 | PL      | 595.5         | 40.0%         |
| Schweinfurt                                             | DE      | 612.5         | 35.5%         |
| Douai                                                   | FR      | 613.6         | 37.1%         |
| Rönninge By                                             | SE      | 616.4         | 38.1%         |
| Vereinbarte Verwaltungsgemeinschaft der Stadt Göppingen | DE      | 623.7         | 37.4%         |
| Iserlohn                                                | DE      | 635.0         | 37.0%         |
| Como                                                    | IT      | 644.2         | 34.4%         |

### Within-City Equity Analysis

The equity ratio (P75/P25) measures inequality of access within each city.
Higher values indicate greater disparity between well-served and underserved areas.

#### Most Equitable Cities (Lowest P75/P25 Ratio)

| City                | Country | Equity Ratio | % Severely Underserved |
| ------------------- | ------- | ------------ | ---------------------- |
| Hoorn               | NL      | 2.66         | 10.0%                  |
| Berkel en Rodenrijs | NL      | 2.63         | 10.4%                  |
| Agedrup             | DK      | 2.59         | 8.0%                   |
| Apeldoorn           | NL      | 2.59         | 11.4%                  |
| Heerhugowaard       | NL      | 2.58         | 11.6%                  |
| Roosendaal          | NL      | 2.58         | 12.8%                  |
| Küsnacht (ZH)       | CH      | 2.54         | 8.2%                   |
| Almelo              | NL      | 2.54         | 10.1%                  |
| Almere              | NL      | 2.47         | 10.9%                  |
| Nieuwegein          | NL      | 2.44         | 9.8%                   |

#### Least Equitable Cities (Highest P75/P25 Ratio)

| City                            | Country | Equity Ratio | % Severely Underserved |
| ------------------------------- | ------- | ------------ | ---------------------- |
| Ξάνθη                           | GR      | 6.21         | 18.5%                  |
| Toledo                          | ES      | 5.99         | 19.4%                  |
| Hoya de Huesca / Plana de Uesca | ES      | 5.69         | 18.3%                  |
| A Coruña                        | ES      | 4.69         | 14.4%                  |
| Guadalajara                     | ES      | 4.65         | 17.4%                  |
| Alessandria                     | IT      | 4.64         | 16.0%                  |
| Focșani                         | RO      | 4.61         | 16.0%                  |
| Baix Camp                       | ES      | 4.43         | 14.6%                  |
| Lubin                           | PL      | 4.41         | 15.8%                  |
| Szombathely                     | HU      | 4.30         | 14.4%                  |

---

## Part 2: Transport Access Gap Analysis

- **Cities Analyzed:** 339
- **Total Street Network Nodes:** 9,374,220
- **Mean Critical Gap %:** 0.4%

### Cities with Largest Transport Gaps (Top 10)

High-demand areas (top 30% centrality + density) with poor transport access (bottom 15% supply).

| City           | Country | % Critical Gap | Mean Demand | Mean Supply |
| -------------- | ------- | -------------- | ----------- | ----------- |
| Küsnacht (ZH)  | CH      | 5.8%           | 0.45        | 0.64        |
| Lahti          | FI      | 4.4%           | 0.32        | 0.71        |
| Rzeszów        | PL      | 3.7%           | 0.38        | 0.54        |
| Grosseto       | IT      | 3.3%           | 0.43        | 0.36        |
| Zagreb         | HR      | 3.3%           | 0.41        | 0.73        |
| Anzio          | IT      | 2.9%           | 0.26        | 0.46        |
| Vila do Conde  | PT      | 2.9%           | 0.45        | 0.70        |
| Carpi          | IT      | 2.7%           | 0.47        | 0.52        |
| Sassari        | IT      | 2.7%           | 0.31        | 0.50        |
| Bahía de Cádiz | ES      | 2.6%           | 0.53        | 0.81        |

## Methodology

### Education Access

- Analyzed cities with 'Consistently Saturated' education POI coverage
- Computed mean/median distances to nearest school
- Calculated proportion within 400m (5-min walk) and 800m (10-min walk)
- Equity ratio = P75/P25 distance (higher = more unequal)
- Underserved = nodes with distance > 2× city mean

### Transport Gap Identification

- Demand score = (normalized centrality + normalized density) / 2
- Supply score = 1 - normalized distance to transport
- Gap areas = high demand (≥70th percentile) + low supply (≤30th percentile)
- Critical gaps = high demand + critically low supply (≤15th percentile)

## Key Findings

### Education Access Patterns

- Best-performing cities achieve <330m mean distance with >69% nodes within 400m
- Worst-performing cities exceed 600m mean distance with <40% within 400m
- Cross-city range: 328m to 644m

### Education Equity Patterns

- Most equitable cities have P75/P25 ratios around 2.4-2.6
- Least equitable cities exceed 4.5-6.0 ratios
- Even equitable cities have 8-12% underserved nodes

### Transport Gap Patterns

- Cities show 0.0% to 5.8% critical gap nodes
- Mean demand across cities: 0.39
- Mean supply across cities: 0.73

## Outputs

### Data Files

- `education_city_access.csv`: Mean education access statistics per city
- `education_equity_analysis.csv`: Within-city equity metrics (P75/P25, underserved %)
- `transport_gap_profiles.csv`: Transport gap statistics per city

### LaTeX Tables

- `table_access.tex`: Top/bottom cities by education access
- `table_equity.tex`: Most/least equitable cities by P75/P25 ratio
- `table_transport_gaps.tex`: Cities with largest transport gaps

### Visualizations

- `education_access_ranking.png`: Side-by-side best/worst education access
- `transport_gap_ranking.png`: Top 20 cities by critical transport gaps
