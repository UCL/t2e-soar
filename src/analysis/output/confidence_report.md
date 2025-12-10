# City-Level Land Use Data Confidence Report

This report provides a comprehensive analysis of data quality for POI (Point of Interest) data across 690 EU cities, using regression-based confidence scoring.

---

## Executive Summary

- **Total cities analyzed**: 699
- **Mean confidence score**: 0.990
- **Median confidence score**: 1.000
- **Cities with confidence < 0.5**: 6 (0.9%)
- **Cities with at least one flagged category**: 18 (2.6%)
- **Mean flagged categories per city**: 0.08

## Methodology

Confidence scores are computed using regression analysis to identify cities with unexpectedly low POI counts given their population and area:

1. **Regression models**: For each common land-use category (eat_and_drink, retail, education, etc.), fit linear regression: `POI_count ~ population + area_km2`
2. **Residual analysis**: Compute standardized residuals (z-scores) for each city
3. **Flagging**: Cities with z-scores < -2.0 in a category are flagged as having likely data quality issues
4. **Confidence score**: Computed as 1.0 minus a weighted penalty based on number of flagged categories (60%) and severity of residuals (40%)

## Regression Model Diagnostics

Model fit quality (R²) for each land-use category:

| Land Use Category | R² Score | N Cities | Coef (Population) | Coef (Area) | Intercept |
|-------------------|----------|----------|-------------------|-------------|----------|
| Business And Services | 0.948 | 568 | 6.80e-03 | 27.77 | -293.9 |
| Active Life | 0.937 | 568 | 5.59e-04 | 2.91 | -11.7 |
| Arts And Entertainment | 0.948 | 568 | 4.58e-04 | 1.13 | -12.9 |
| Public Services | 0.953 | 568 | 1.32e-03 | 1.33 | 10.9 |
| Retail | 0.961 | 568 | 5.27e-03 | 6.49 | 99.0 |
| Health And Medical | 0.889 | 568 | 3.78e-04 | 7.00 | -70.2 |
| Eat And Drink | 0.960 | 568 | 4.59e-03 | -0.98 | 51.2 |
| Education | 0.939 | 568 | 1.03e-03 | 1.89 | 17.0 |
| Attractions And Activities | 0.903 | 568 | 8.99e-04 | 0.56 | 0.1 |
| Religious | 0.956 | 568 | 1.74e-04 | 0.82 | -9.1 |
| Accommodation | 0.804 | 568 | 6.99e-04 | -0.24 | 35.3 |

## Most Commonly Flagged Categories

Cities with data quality issues by category:

- **Religious**: 9 cities (1.3%)
- **Retail**: 6 cities (0.9%)
- **Eat And Drink**: 6 cities (0.9%)
- **Health And Medical**: 5 cities (0.7%)
- **Business And Services**: 5 cities (0.7%)
- **Arts And Entertainment**: 5 cities (0.7%)
- **Public Services**: 5 cities (0.7%)
- **Education**: 4 cities (0.6%)
- **Attractions And Activities**: 4 cities (0.6%)
- **Active Life**: 3 cities (0.4%)
- **Accommodation**: 3 cities (0.4%)

## Data Quality Patterns

### Confidence by City Size

| City Size | Mean Confidence | N Cities |
|-----------|-----------------|----------|
| Small (<100k) | 1.000 | 339 |
| Medium (100k-500k) | 1.000 | 298 |
| Large (>500k) | 0.886 | 62 |

### Confidence by Country

Countries ranked by mean confidence score:

| Country | Mean Confidence | Median Confidence | N Cities |
|---------|-----------------|-------------------|----------|
| AT | 1.000 | 1.000 | 8 |
| GR | 1.000 | 1.000 | 13 |
| SI | 1.000 | 1.000 | 2 |
| NO | 1.000 | 1.000 | 3 |
| NL | 1.000 | 1.000 | 51 |
| MT | 1.000 | 1.000 | 1 |
| LV | 1.000 | 1.000 | 3 |
| LU | 1.000 | 1.000 | 1 |
| LT | 1.000 | 1.000 | 4 |
| BE | 1.000 | 1.000 | 16 |
| HR | 1.000 | 1.000 | 5 |
| HU | 1.000 | 1.000 | 14 |
| CZ | 1.000 | 1.000 | 16 |
| EE | 1.000 | 1.000 | 3 |
| CH | 1.000 | 1.000 | 20 |
| CY | 1.000 | 1.000 | 3 |
| DK | 1.000 | 1.000 | 4 |
| SK | 1.000 | 1.000 | 7 |
| ES | 0.999 | 1.000 | 111 |
| IT | 0.996 | 1.000 | 87 |
| SE | 0.996 | 1.000 | 16 |
| DE | 0.991 | 1.000 | 110 |
| PT | 0.989 | 1.000 | 12 |
| RO | 0.981 | 1.000 | 32 |
| PL | 0.978 | 1.000 | 54 |
| FR | 0.972 | 1.000 | 76 |
| BG | 0.971 | 1.000 | 13 |
| IE | 0.887 | 1.000 | 5 |
| FI | 0.845 | 1.000 | 4 |

**Summary:**

- **Cities with country data**: 694 (99.3%)
- **Number of countries**: 29

### Cities with Labels

- **Cities with geographic labels**: 694 (99.3%)
- **Mean confidence (labeled cities)**: 0.990
- **Mean confidence (unlabeled cities)**: 1.000

## Top 50 Most Robust Cities

Cities with highest confidence scores (best data quality):

| Rank | Bounds FID | City Label | Population | Area (km²) | Confidence | Mean Z-Score | Flagged Categories |
|------|------------|------------|------------|------------|------------|--------------|-------------------|
| 1 | 0 | N/A | 97,244 | 62.8 | 1.000 | -0.15 | None |
| 2 | 468 | Alessandria | 65,425 | 28.5 | 1.000 | 0.14 | None |
| 3 | 460 | Добрич | 72,154 | 30.6 | 1.000 | -0.37 | None |
| 4 | 461 | Ferrol | 92,683 | 42.5 | 1.000 | -0.17 | None |
| 5 | 462 | Русе | 119,719 | 41.7 | 1.000 | -0.25 | None |
| 6 | 463 | Valence | 94,635 | 50.9 | 1.000 | -0.02 | None |
| 7 | 464 | Piacenza | 94,079 | 39.0 | 1.000 | 0.21 | None |
| 8 | 465 | Torino | 1,161,395 | 270.2 | 1.000 | 1.70 | None |
| 9 | 466 | A Coruña | 298,793 | 78.9 | 1.000 | 0.13 | None |
| 10 | 467 | Asti | 60,503 | 26.8 | 1.000 | 0.14 | None |
| 11 | 469 | Ferrara | 99,971 | 60.4 | 1.000 | 0.29 | None |
| 12 | 458 | Pavia | 66,257 | 36.8 | 1.000 | 0.32 | None |
| 13 | 470 | Parma | 157,498 | 56.0 | 1.000 | 0.51 | None |
| 14 | 471 | Carpi | 60,323 | 35.6 | 1.000 | -0.09 | None |
| 15 | 472 | Avilés | 90,389 | 34.3 | 1.000 | -0.17 | None |
| 16 | 473 | Варна | 271,373 | 67.9 | 1.000 | -0.05 | None |
| 17 | 474 | Gijón / Xixón | 252,852 | 46.2 | 1.000 | 0.16 | None |
| 18 | 475 | Reggio Emilia | 123,556 | 54.0 | 1.000 | 0.22 | None |
| 19 | 476 | Шумен | 62,787 | 35.6 | 1.000 | -0.30 | None |
| 20 | 477 | Modena | 160,853 | 56.0 | 1.000 | 0.45 | None |
| 21 | 459 | Cremona | 63,416 | 30.8 | 1.000 | 0.17 | None |
| 22 | 457 | Mofleni | 230,768 | 67.3 | 1.000 | -0.63 | None |
| 23 | 433 | Lyon | 57,850 | 33.8 | 1.000 | -0.33 | None |
| 24 | 444 | Brescia | 241,357 | 123.8 | 1.000 | 0.23 | None |
| 25 | 435 | Treviso | 85,582 | 42.0 | 1.000 | 0.28 | None |
| 26 | 436 | Bergamo | 353,630 | 192.8 | 1.000 | -0.37 | None |
| 27 | 437 | Gallarate | 130,471 | 79.5 | 1.000 | -0.49 | None |
| 28 | 438 | Chambéry | 102,812 | 56.3 | 1.000 | -0.18 | None |
| 29 | 439 | Călărași | 57,633 | 31.5 | 1.000 | -0.37 | None |
| 30 | 441 | Saint-Étienne | 196,196 | 74.9 | 1.000 | -0.25 | None |
| 31 | 442 | Busto Arsizio | 312,493 | 154.7 | 1.000 | -0.87 | None |
| 32 | 443 | Vicenza | 100,820 | 52.6 | 1.000 | 0.27 | None |
| 33 | 445 | Venezia | 180,900 | 69.0 | 1.000 | -0.00 | None |
| 34 | 456 | Grenoble | 348,749 | 109.1 | 1.000 | -0.15 | None |
| 35 | 446 | Novara | 94,759 | 43.0 | 1.000 | 0.02 | None |
| 36 | 447 | Verona | 202,774 | 71.5 | 1.000 | 1.14 | None |
| 37 | 448 | Rijeka | 116,764 | 60.9 | 1.000 | -0.09 | None |
| 38 | 449 | Venezia | 66,715 | 58.2 | 1.000 | 1.90 | None |
| 39 | 450 | Slatina | 62,526 | 25.8 | 1.000 | -0.22 | None |
| 40 | 451 | Magheru | 72,947 | 27.9 | 1.000 | -0.30 | None |
| 41 | 452 | Padova | 261,786 | 119.8 | 1.000 | 0.73 | None |
| 42 | 454 | Vigevano | 60,432 | 35.1 | 1.000 | -0.14 | None |
| 43 | 478 | Oviedo | 210,156 | 47.1 | 1.000 | 0.28 | None |
| 44 | 479 | Santiago | 82,694 | 33.2 | 1.000 | 0.62 | None |
| 45 | 480 | Lugo | 89,446 | 25.9 | 1.000 | 0.13 | None |
| 46 | 513 | Pisa | 130,420 | 86.5 | 1.000 | 0.41 | None |
| 47 | 505 | Ourense | 110,291 | 36.8 | 1.000 | 0.07 | None |
| 48 | 506 | Prato | 209,792 | 91.6 | 1.000 | -0.13 | None |
| 49 | 507 | Pau | 119,986 | 54.5 | 1.000 | -0.04 | None |
| 50 | 508 | León | 164,497 | 53.3 | 1.000 | 0.05 | None |

## Bottom 50 Least Confident Cities

Cities with lowest confidence scores (likely data quality issues):

| Rank | Bounds FID | City Label | Population | Area (km²) | Confidence | N Flags | Flagged Categories |
|------|------------|------------|------------|------------|------------|---------|-------------------|
| 1 | 307 | Palaiseau | 9,773,324 | 1835.9 | 0.055 | 10 | business_and_services, active_life, arts_and_entertainmen... |
| 2 | 237 | Katowice | 1,148,965 | 448.9 | 0.109 | 9 | business_and_services, active_life, arts_and_entertainmen... |
| 3 | 170 | Innenstadt West | 2,946,855 | 1199.3 | 0.109 | 9 | business_and_services, active_life, arts_and_entertainmen... |
| 4 | 7 | Helsinki | 1,056,200 | 464.9 | 0.382 | 4 | retail, health_and_medical, education, religious |
| 5 | 440 | Sector 1 | 1,922,595 | 343.9 | 0.382 | 4 | public_services, retail, eat_and_drink, accommodation |
| 6 | 51 | Dublin | 1,246,189 | 479.3 | 0.436 | 3 | business_and_services, retail, health_and_medical |
| 7 | 221 | Lille | 999,401 | 322.9 | 0.538 | 5 | business_and_services, arts_and_entertainment, health_and... |
| 8 | 455 | Bordeaux | 738,916 | 268.3 | 0.596 | 1 | religious |
| 9 | 524 | София | 1,121,633 | 226.8 | 0.619 | 1 | religious |
| 10 | 453 | Milano | 3,150,317 | 850.9 | 0.649 | 1 | education |
| 11 | 503 | Toulouse | 722,276 | 271.5 | 0.709 | 1 | religious |
| 12 | 128 | Łódź | 633,748 | 177.1 | 0.780 | 1 | eat_and_drink |
| 13 | 542 | Cedofeita, Santo Ildefonso, Sé, Miragaia, São Nicolau e Vitória | 917,584 | 311.1 | 0.872 | 1 | arts_and_entertainment |
| 14 | 181 | Ost | 576,826 | 206.8 | 0.900 | 1 | religious |
| 15 | 92 | Warszawa | 1,889,245 | 457.3 | 0.914 | 1 | eat_and_drink |
| 16 | 15 | Sjöberg | 1,462,756 | 480.8 | 0.932 | 1 | retail |
| 17 | 495 | Bilbao | 786,311 | 140.7 | 0.934 | 1 | public_services |
| 18 | 228 | Lens | 195,803 | 115.3 | 0.940 | 1 | religious |
| 19 | 225 | Verviers | 68,241 | 34.7 | 1.000 | 0 | None |
| 20 | 227 | Sosnowiec | 55,800 | 19.8 | 1.000 | 0 | None |
| 21 | 234 | Mons | 95,613 | 64.0 | 1.000 | 0 | None |
| 22 | 224 | Liège | 417,400 | 191.7 | 1.000 | 0 | None |
| 23 | 229 | Gliwice | 125,928 | 53.0 | 1.000 | 0 | None |
| 24 | 223 | Stadtbezirk Bonn | 545,494 | 274.5 | 1.000 | 0 | None |
| 25 | 230 | La Louvière | 102,783 | 70.7 | 1.000 | 0 | None |
| 26 | 231 | Sosnowiec | 80,378 | 47.3 | 1.000 | 0 | None |
| 27 | 232 | Gießen | 75,214 | 45.4 | 1.000 | 0 | None |
| 28 | 222 | Ústí nad Labem | 74,984 | 40.0 | 1.000 | 0 | None |
| 29 | 233 | Douai | 94,282 | 62.4 | 1.000 | 0 | None |
| 30 | 226 | Rzeszów | 167,098 | 73.5 | 1.000 | 0 | None |
| 31 | 235 | Namur | 80,326 | 51.0 | 1.000 | 0 | None |
| 32 | 251 | Pardubice | 80,843 | 39.4 | 1.000 | 0 | None |
| 33 | 255 | Ostrava | 71,248 | 29.6 | 1.000 | 0 | None |
| 34 | 250 | Kladno | 62,930 | 30.5 | 1.000 | 0 | None |
| 35 | 249 | Cherbourg | 70,657 | 48.3 | 1.000 | 0 | None |
| 36 | 248 | Jastrzębie-Zdrój | 62,422 | 31.7 | 1.000 | 0 | None |
| 37 | 247 | Hradec Králové | 83,537 | 46.0 | 1.000 | 0 | None |
| 38 | 246 | Lahnstein | 121,650 | 79.9 | 1.000 | 0 | None |
| 39 | 245 | Kraków | 755,288 | 209.1 | 1.000 | 0 | None |
| 40 | 244 | Tychy | 115,956 | 44.7 | 1.000 | 0 | None |
| 41 | 243 | Arras | 79,026 | 46.3 | 1.000 | 0 | None |
| 42 | 242 | Plauen | 58,024 | 41.1 | 1.000 | 0 | None |
| 43 | 241 | Valenciennes | 121,357 | 74.4 | 1.000 | 0 | None |
| 44 | 253 | Amiens | 149,696 | 63.5 | 1.000 | 0 | None |
| 45 | 240 | Jirkov | 64,281 | 35.5 | 1.000 | 0 | None |
| 46 | 254 | Ostrava | 66,025 | 35.1 | 1.000 | 0 | None |
| 47 | 239 | Charleroi | 244,178 | 133.7 | 1.000 | 0 | None |
| 48 | 252 | Bielsko-Biała | 141,588 | 57.4 | 1.000 | 0 | None |
| 49 | 236 | Most | 61,106 | 27.0 | 1.000 | 0 | None |
| 50 | 238 | Tarnów | 85,834 | 36.4 | 1.000 | 0 | None |

### Detailed Category Breakdown (Bottom 10 Cities)

#### 1. Bounds FID 307 - Palaiseau

- **Confidence Score**: 0.055
- **Population**: 9,773,324
- **Area**: 1835.9 km²
- **Flagged Categories**: 10

**Category Z-Scores:**

- Business And Services: Z=-8.33 (Observed=103597, Expected=117154.5) ⚠️ **FLAGGED**
- Active Life: Z=-7.23 (Observed=9583, Expected=10805.2) ⚠️ **FLAGGED**
- Arts And Entertainment: Z=-7.36 (Observed=5891, Expected=6542.5) ⚠️ **FLAGGED**
- Public Services: Z=-4.18 (Observed=14537, Expected=15337.9) ⚠️ **FLAGGED**
- Retail: Z=-6.89 (Observed=58573, Expected=63559.1) ⚠️ **FLAGGED**
- Health And Medical: Z=-9.27 (Observed=13009, Expected=16485.9) ⚠️ **FLAGGED**
- Eat And Drink: Z=-3.83 (Observed=41241, Expected=43056.2) ⚠️ **FLAGGED**
- Education: Z=-6.37 (Observed=12290, Expected=13543.3) ⚠️ **FLAGGED**
- Attractions And Activities: Z=-4.50 (Observed=9015, Expected=9818.3) ⚠️ **FLAGGED**
- Religious: Z=-2.71 (Observed=3085, Expected=3196.8) ⚠️ **FLAGGED**
- Accommodation: Z=-0.83 (Observed=6280, Expected=6419.9)

#### 2. Bounds FID 237 - Katowice

- **Confidence Score**: 0.109
- **Population**: 1,148,965
- **Area**: 448.9 km²
- **Flagged Categories**: 9

**Category Z-Scores:**

- Business And Services: Z=-2.05 (Observed=16654, Expected=19984.6) ⚠️ **FLAGGED**
- Active Life: Z=-3.77 (Observed=1302, Expected=1939.0) ⚠️ **FLAGGED**
- Arts And Entertainment: Z=-5.64 (Observed=522, Expected=1021.1) ⚠️ **FLAGGED**
- Public Services: Z=-2.97 (Observed=1555, Expected=2123.4) ⚠️ **FLAGGED**
- Retail: Z=-1.51 (Observed=7983, Expected=9072.5)
- Health And Medical: Z=-3.48 (Observed=2204, Expected=3508.3) ⚠️ **FLAGGED**
- Eat And Drink: Z=-5.45 (Observed=2294, Expected=4877.6) ⚠️ **FLAGGED**
- Education: Z=-0.19 (Observed=2009, Expected=2046.6)
- Attractions And Activities: Z=-3.63 (Observed=637, Expected=1285.0) ⚠️ **FLAGGED**
- Religious: Z=-5.67 (Observed=325, Expected=559.4) ⚠️ **FLAGGED**
- Accommodation: Z=-2.32 (Observed=336, Expected=729.6) ⚠️ **FLAGGED**

#### 3. Bounds FID 170 - Innenstadt West

- **Confidence Score**: 0.109
- **Population**: 2,946,855
- **Area**: 1199.3 km²
- **Flagged Categories**: 9

**Category Z-Scores:**

- Business And Services: Z=-5.78 (Observed=43643, Expected=53051.7) ⚠️ **FLAGGED**
- Active Life: Z=-8.11 (Observed=3761, Expected=5131.5) ⚠️ **FLAGGED**
- Arts And Entertainment: Z=-3.03 (Observed=2426, Expected=2693.6) ⚠️ **FLAGGED**
- Public Services: Z=-8.08 (Observed=3945, Expected=5493.0) ⚠️ **FLAGGED**
- Retail: Z=-6.03 (Observed=19067, Expected=23426.5) ⚠️ **FLAGGED**
- Health And Medical: Z=-1.42 (Observed=8914, Expected=9444.9)
- Eat And Drink: Z=-8.12 (Observed=8532, Expected=12382.6) ⚠️ **FLAGGED**
- Education: Z=-4.62 (Observed=4405, Expected=5313.3) ⚠️ **FLAGGED**
- Attractions And Activities: Z=-8.66 (Observed=1776, Expected=3322.7) ⚠️ **FLAGGED**
- Religious: Z=0.55 (Observed=1511, Expected=1488.4)
- Accommodation: Z=-5.35 (Observed=897, Expected=1804.4) ⚠️ **FLAGGED**

#### 4. Bounds FID 7 - Helsinki

- **Confidence Score**: 0.382
- **Population**: 1,056,200
- **Area**: 464.9 km²
- **Flagged Categories**: 4

**Category Z-Scores:**

- Business And Services: Z=-1.95 (Observed=16618, Expected=19798.1)
- Active Life: Z=1.46 (Observed=2181, Expected=1933.7)
- Arts And Entertainment: Z=-0.31 (Observed=969, Expected=996.7)
- Public Services: Z=2.95 (Observed=2587, Expected=2022.4)
- Retail: Z=-2.82 (Observed=6647, Expected=8687.2) ⚠️ **FLAGGED**
- Health And Medical: Z=-4.40 (Observed=1933, Expected=3585.3) ⚠️ **FLAGGED**
- Eat And Drink: Z=1.48 (Observed=5139, Expected=4436.5)
- Education: Z=-2.30 (Observed=1529, Expected=1981.3) ⚠️ **FLAGGED**
- Attractions And Activities: Z=6.84 (Observed=2433, Expected=1210.6)
- Religious: Z=-2.09 (Observed=470, Expected=556.4) ⚠️ **FLAGGED**
- Accommodation: Z=-0.49 (Observed=578, Expected=660.9)

#### 5. Bounds FID 440 - Sector 1

- **Confidence Score**: 0.382
- **Population**: 1,922,595
- **Area**: 343.9 km²
- **Flagged Categories**: 4

**Category Z-Scores:**

- Business And Services: Z=1.09 (Observed=24107, Expected=22329.9)
- Active Life: Z=0.13 (Observed=2087, Expected=2065.8)
- Arts And Entertainment: Z=2.61 (Observed=1488, Expected=1257.1)
- Public Services: Z=-4.03 (Observed=2231, Expected=3003.0) ⚠️ **FLAGGED**
- Retail: Z=-2.13 (Observed=10932, Expected=12470.5) ⚠️ **FLAGGED**
- Health And Medical: Z=5.17 (Observed=5006, Expected=3065.5)
- Eat And Drink: Z=-5.93 (Observed=5716, Expected=8528.1) ⚠️ **FLAGGED**
- Education: Z=2.48 (Observed=3132, Expected=2645.3)
- Attractions And Activities: Z=-0.48 (Observed=1836, Expected=1921.8)
- Religious: Z=-1.46 (Observed=547, Expected=607.4)
- Accommodation: Z=-2.91 (Observed=802, Expected=1295.4) ⚠️ **FLAGGED**

#### 6. Bounds FID 51 - Dublin

- **Confidence Score**: 0.436
- **Population**: 1,246,189
- **Area**: 479.3 km²
- **Flagged Categories**: 3

**Category Z-Scores:**

- Business And Services: Z=-4.38 (Observed=14366, Expected=21489.9) ⚠️ **FLAGGED**
- Active Life: Z=0.77 (Observed=2212, Expected=2082.0)
- Arts And Entertainment: Z=-0.25 (Observed=1078, Expected=1100.1)
- Public Services: Z=-1.73 (Observed=1960, Expected=2292.0)
- Retail: Z=-3.34 (Observed=7367, Expected=9782.6) ⚠️ **FLAGGED**
- Health And Medical: Z=-3.40 (Observed=2482, Expected=3758.0) ⚠️ **FLAGGED**
- Eat And Drink: Z=0.63 (Observed=5591, Expected=5293.5)
- Education: Z=-0.84 (Observed=2038, Expected=2204.1)
- Attractions And Activities: Z=0.20 (Observed=1426, Expected=1389.5)
- Religious: Z=2.85 (Observed=719, Expected=601.3)
- Accommodation: Z=0.75 (Observed=918, Expected=790.1)

#### 7. Bounds FID 221 - Lille

- **Confidence Score**: 0.538
- **Population**: 999,401
- **Area**: 322.9 km²
- **Flagged Categories**: 5

**Category Z-Scores:**

- Business And Services: Z=-2.40 (Observed=11567, Expected=15468.3) ⚠️ **FLAGGED**
- Active Life: Z=-1.40 (Observed=1252, Expected=1488.2)
- Arts And Entertainment: Z=-2.14 (Observed=621, Expected=810.1) ⚠️ **FLAGGED**
- Public Services: Z=-0.19 (Observed=1721, Expected=1758.3)
- Retail: Z=-1.04 (Observed=6715, Expected=7465.6)
- Health And Medical: Z=-2.26 (Observed=1720, Expected=2569.2) ⚠️ **FLAGGED**
- Eat And Drink: Z=-0.61 (Observed=4028, Expected=4315.8)
- Education: Z=-0.88 (Observed=1481, Expected=1655.0)
- Attractions And Activities: Z=-2.00 (Observed=722, Expected=1079.8) ⚠️ **FLAGGED**
- Religious: Z=-2.95 (Observed=308, Expected=429.8) ⚠️ **FLAGGED**
- Accommodation: Z=-1.31 (Observed=434, Expected=655.5)

#### 8. Bounds FID 455 - Bordeaux

- **Confidence Score**: 0.596
- **Population**: 738,916
- **Area**: 268.3 km²
- **Flagged Categories**: 1

**Category Z-Scores:**

- Business And Services: Z=-0.66 (Observed=11107, Expected=12181.4)
- Active Life: Z=0.36 (Observed=1245, Expected=1183.4)
- Arts And Entertainment: Z=-0.63 (Observed=573, Expected=629.0)
- Public Services: Z=0.66 (Observed=1469, Expected=1342.3)
- Retail: Z=0.34 (Observed=5980, Expected=5737.7)
- Health And Medical: Z=-0.66 (Observed=1839, Expected=2088.4)
- Eat And Drink: Z=0.82 (Observed=3563, Expected=3175.2)
- Education: Z=-0.16 (Observed=1252, Expected=1283.8)
- Attractions And Activities: Z=-0.88 (Observed=657, Expected=815.0)
- Religious: Z=-3.75 (Observed=185, Expected=339.8) ⚠️ **FLAGGED**
- Accommodation: Z=0.72 (Observed=608, Expected=486.7)

#### 9. Bounds FID 524 - София

- **Confidence Score**: 0.619
- **Population**: 1,121,633
- **Area**: 226.8 km²
- **Flagged Categories**: 1

**Category Z-Scores:**

- Business And Services: Z=2.34 (Observed=17442, Expected=13632.4)
- Active Life: Z=2.48 (Observed=1695, Expected=1276.7)
- Arts And Entertainment: Z=3.86 (Observed=1099, Expected=757.6)
- Public Services: Z=-1.39 (Observed=1525, Expected=1791.4)
- Retail: Z=2.19 (Observed=9073, Expected=7486.6)
- Health And Medical: Z=0.59 (Observed=2165, Expected=1942.7)
- Eat And Drink: Z=0.24 (Observed=5084, Expected=4970.8)
- Education: Z=2.31 (Observed=2054, Expected=1599.7)
- Attractions And Activities: Z=0.86 (Observed=1290, Expected=1135.9)
- Religious: Z=-3.63 (Observed=222, Expected=372.1) ⚠️ **FLAGGED**
- Accommodation: Z=-0.86 (Observed=618, Expected=764.1)

#### 10. Bounds FID 453 - Milano

- **Confidence Score**: 0.649
- **Population**: 3,150,317
- **Area**: 850.9 km²
- **Flagged Categories**: 1

**Category Z-Scores:**

- Business And Services: Z=5.18 (Observed=53193, Expected=44760.5)
- Active Life: Z=2.11 (Observed=4586, Expected=4230.1)
- Arts And Entertainment: Z=-1.12 (Observed=2294, Expected=2393.1)
- Public Services: Z=2.18 (Observed=5714, Expected=5296.8)
- Retail: Z=10.00 (Observed=29473, Expected=22237.3)
- Health And Medical: Z=1.58 (Observed=7676, Expected=7081.5)
- Eat And Drink: Z=7.44 (Observed=17188, Expected=13658.3)
- Education: Z=-3.48 (Observed=4181, Expected=4865.8) ⚠️ **FLAGGED**
- Attractions And Activities: Z=0.12 (Observed=3332, Expected=3310.2)
- Religious: Z=2.15 (Observed=1326, Expected=1237.3)
- Accommodation: Z=2.04 (Observed=2377, Expected=2030.7)

## Recommendations for Dataset Usage

### Data Quality Tiers

1. **High Confidence (≥0.7)**: 689 cities - Suitable for all analyses including detailed land-use studies
2. **Moderate Confidence (0.4-0.7)**: 5 cities - Suitable for aggregate analyses; use caution for category-specific studies
3. **Low Confidence (<0.4)**: 5 cities - Recommend excluding from analyses or treating as missing data

### Specific Recommendations

1. **Exclusion criterion**: Consider excluding 5 cities with confidence < 0.4 from regression analyses to avoid biasing results

2. **Category-specific caution**: For analyses focused on specific land uses, filter cities based on category-specific z-scores rather than overall confidence

3. **Robust methods**: Use robust regression techniques (e.g., M-estimators) that downweight outliers when including all cities

4. **Data improvement**: Cities with low confidence scores may benefit from manual validation or supplementary data sources (e.g., official business registries)

---

*Report generated automatically from city-level POI aggregation and regression analysis.*
