# Educational Infrastructure Gap Analysis Report

**Analysis Date:** 2025-12-18

## Overview

This analysis examines educational facility accessibility across European cities,
restricted to cities with reliable (saturated) POI coverage from Demonstrator 1.

## Summary Statistics

- **Cities Analyzed:** 308 (saturated education POI coverage only)
- **Filter:** Analysis restricted to `Consistently Saturated` cities (low POI variability) to ensure within-city equity metrics reflect real infrastructure gaps rather than data artefacts.
- **Total Street Network Nodes:** 11,379,863
- **Mean Distance to Education (cross-city):** 573.9m
- **Median Distance to Education (cross-city):** 462.4m

## Cross-City Comparison

### Best Access (Top 10 Cities)

| City       | Country | Mean Dist (m) | % within 400m |
| ---------- | ------- | ------------- | ------------- |
| Venezia    | IT      | 311.6         | 72.6%         |
| Warszawa   | PL      | 419.4         | 63.4%         |
| Almere     | NL      | 432.8         | 61.7%         |
| Utrecht    | NL      | 434.1         | 62.7%         |
| Lublin     | PL      | 441.2         | 63.0%         |
| Bratislava | SK      | 442.1         | 61.9%         |
| Hoorn      | NL      | 443.3         | 58.0%         |
| Gdańsk     | PL      | 445.5         | 60.1%         |
| Leiden     | NL      | 445.8         | 59.9%         |
| Kraków     | PL      | 446.7         | 59.7%         |

### Worst Access (Bottom 10 Cities)

| City          | Country | Mean Dist (m) | % within 400m |
| ------------- | ------- | ------------- | ------------- |
| Legionowo     | PL      | 701.2         | 34.2%         |
| Harburg       | DE      | 702.0         | 31.3%         |
| Wołomin       | PL      | 702.7         | 32.0%         |
| Aschaffenburg | DE      | 704.4         | 32.9%         |
| Douai         | FR      | 708.0         | 28.6%         |
| Ludwigsburg   | DE      | 715.6         | 32.4%         |
| Brackel       | DE      | 716.0         | 27.5%         |
| Tampere       | FI      | 720.7         | 25.7%         |
| Como          | IT      | 725.9         | 27.5%         |
| None          | None    | 738.1         | 28.4%         |

## Within-City Equity Analysis

The equity ratio (P75/P25) measures inequality of access within each city.
Higher values indicate greater disparity between well-served and underserved areas.

### Most Equitable Cities (Lowest P75/P25 Ratio)

| City                         | Country | Equity Ratio | % Severely Underserved |
| ---------------------------- | ------- | ------------ | ---------------------- |
| Almere                       | NL      | 2.58         | 12.1%                  |
| Tampere                      | FI      | 2.61         | 4.8%                   |
| Västerås                     | SE      | 2.64         | 9.7%                   |
| Agedrup                      | DK      | 2.69         | 10.2%                  |
| Brackel                      | DE      | 2.70         | 7.2%                   |
| Stora Hultet                 | SE      | 2.70         | 9.5%                   |
| Vallkärra by                 | SE      | 2.73         | 11.0%                  |
| Stadtbezirk Bremerhaven-Nord | DE      | 2.78         | 9.0%                   |
| Hasbergen                    | DE      | 2.81         | 8.2%                   |
| Turku                        | FI      | 2.81         | 7.7%                   |

### Least Equitable Cities (Highest P75/P25 Ratio)

| City      | Country | Equity Ratio | % Severely Underserved |
| --------- | ------- | ------------ | ---------------------- |
| Lugo      | ES      | 6.16         | 18.3%                  |
| A Coruña  | ES      | 5.11         | 16.2%                  |
| Focșani   | RO      | 4.89         | 15.7%                  |
| Santiago  | ES      | 4.85         | 16.2%                  |
| Asti      | IT      | 4.82         | 15.9%                  |
| Miroslava | RO      | 4.76         | 15.7%                  |
| Pécs      | HU      | 4.64         | 17.4%                  |
| Ełk       | PL      | 4.59         | 19.1%                  |
| Győr      | HU      | 4.56         | 14.8%                  |
| Pavia     | IT      | 4.55         | 15.2%                  |

## Key Findings

1. **Cross-city variation**: Mean distance to education varies substantially across
   European cities, from 311.6m to 738.1m.

2. **Within-city inequality**: Even in cities with good average access, significant
   portions of the population may be underserved. The equity ratio captures this disparity.

3. **Data quality matters**: By filtering to saturated cities only, these results
   reflect actual infrastructure gaps rather than POI data incompleteness.

## Output Files

- `cross_city_education_access.csv`: Full cross-city comparison table
- `within_city_below_average.csv`: Within-city equity metrics
