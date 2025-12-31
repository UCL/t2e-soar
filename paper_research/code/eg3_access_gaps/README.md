# Educational Infrastructure Gap Analysis Report

**Analysis Date:** 2025-12-31

## Overview

This analysis examines educational facility accessibility across European cities,
restricted to cities with reliable (saturated) POI coverage from Demonstrator 1.

## Summary Statistics

- **Cities Analyzed:** 123 (saturated education POI coverage only)
- **Total Street Network Nodes:** 2,434,058
- **Mean Distance to Education (cross-city):** 491.9m
- **Median Distance to Education (cross-city):** 386.3m

## Cross-City Comparison

### Best Access (Top 10 Cities)

| City | Country | Mean Dist (m) | % within 400m |
|------|---------|---------------|---------------|
| Küsnacht (ZH) | CH | 327.7 | 69.8% |
| Płock | PL | 376.1 | 69.4% |
| Hoorn | NL | 387.9 | 64.3% |
| A Coruña | ES | 394.7 | 61.8% |
| Leiden | NL | 395.6 | 65.0% |
| Almere | NL | 396.8 | 65.3% |
| Rotterdam | NL | 399.1 | 63.4% |
| Kalisz | PL | 399.4 | 66.1% |
| Częstochowa | PL | 399.4 | 63.3% |
| Nieuwegein | NL | 403.4 | 63.2% |

### Worst Access (Bottom 10 Cities)

| City | Country | Mean Dist (m) | % within 400m |
|------|---------|---------------|---------------|
| Aschaffenburg | DE | 582.8 | 44.1% |
| Lüdenscheid | DE | 583.8 | 41.2% |
| Sosnowiec | PL | 593.0 | 39.9% |
| Wołomin | PL | 595.5 | 40.0% |
| Schweinfurt | DE | 612.5 | 35.5% |
| Douai | FR | 613.6 | 37.1% |
| Rönninge By | SE | 616.4 | 38.1% |
| Vereinbarte Verwaltungsgemeinschaft der Stadt Göppingen | DE | 623.7 | 37.4% |
| Iserlohn | DE | 635.0 | 37.0% |
| Como | IT | 644.2 | 34.4% |

## Within-City Equity Analysis

The equity ratio (P75/P25) measures inequality of access within each city.
Higher values indicate greater disparity between well-served and underserved areas.

### Most Equitable Cities (Lowest P75/P25 Ratio)

| City | Country | Equity Ratio | % Severely Underserved |
|------|---------|--------------|------------------------|
| Nieuwegein | NL | 2.44 | 9.8% |
| Almere | NL | 2.47 | 10.9% |
| Almelo | NL | 2.54 | 10.1% |
| Küsnacht (ZH) | CH | 2.54 | 8.2% |
| Roosendaal | NL | 2.58 | 12.8% |
| Heerhugowaard | NL | 2.58 | 11.6% |
| Apeldoorn | NL | 2.59 | 11.4% |
| Agedrup | DK | 2.59 | 8.0% |
| Berkel en Rodenrijs | NL | 2.63 | 10.4% |
| Hoorn | NL | 2.66 | 10.0% |

### Least Equitable Cities (Highest P75/P25 Ratio)

| City | Country | Equity Ratio | % Severely Underserved |
|------|---------|--------------|------------------------|
| Ξάνθη | GR | 6.21 | 18.5% |
| Toledo | ES | 5.99 | 19.4% |
| Hoya de Huesca / Plana de Uesca | ES | 5.69 | 18.3% |
| A Coruña | ES | 4.69 | 14.4% |
| Guadalajara | ES | 4.65 | 17.4% |
| Alessandria | IT | 4.64 | 16.0% |
| Focșani | RO | 4.61 | 16.0% |
| Baix Camp | ES | 4.43 | 14.6% |
| Lubin | PL | 4.41 | 15.8% |
| Szombathely | HU | 4.30 | 14.4% |

## Key Findings

1. **Cross-city variation**: Mean distance to education varies substantially across
   European cities, from 327.7m to 644.2m.

2. **Within-city inequality**: Even in cities with good average access, significant
   portions of the population may be underserved. The equity ratio captures this disparity.

3. **Data quality matters**: By filtering to saturated cities only, these results
   reflect actual infrastructure gaps rather than POI data incompleteness.

## Output Files

- `cross_city_education_access.csv`: Full cross-city comparison table
- `within_city_below_average.csv`: Within-city equity metrics
