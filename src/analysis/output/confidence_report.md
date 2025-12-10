# City-Level Land Use Data Confidence Report

This report provides a comprehensive analysis of data quality for POI (Point of Interest) data across 690 EU cities, using regression-based confidence scoring.

---

## Executive Summary

- **Total cities analyzed**: 699
- **Mean confidence score**: 0.992
- **Median confidence score**: 1.000
- **Cities with confidence < 0.5**: 6 (0.9%)
- **Cities with at least one flagged category**: 13 (1.9%)
- **Mean flagged categories per city**: 0.05

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
| Business And Services | 0.953 | 244 | 1.77e-02 | -1.88 | 481.7 |
| Active Life | 0.938 | 244 | 1.44e-03 | 0.64 | 61.3 |
| Arts And Entertainment | 0.955 | 244 | 8.24e-04 | 0.28 | -1.4 |
| Public Services | 0.940 | 244 | 2.53e-03 | -2.10 | 102.1 |
| Retail | 0.963 | 244 | 8.57e-03 | -3.38 | 399.4 |
| Health And Medical | 0.936 | 244 | 2.54e-03 | 1.53 | -5.1 |
| Eat And Drink | 0.942 | 244 | 5.94e-03 | -5.17 | 93.6 |
| Education | 0.945 | 244 | 2.14e-03 | -1.15 | 88.8 |
| Attractions And Activities | 0.853 | 244 | 1.63e-03 | -1.39 | 54.0 |
| Religious | 0.934 | 244 | 2.04e-04 | 0.84 | -8.2 |
| Accommodation | 0.689 | 244 | 8.91e-04 | -0.96 | 53.2 |

## Most Commonly Flagged Categories

Cities with data quality issues by category:

- **Health And Medical**: 6 cities (0.9%)
- **Religious**: 5 cities (0.7%)
- **Business And Services**: 5 cities (0.7%)
- **Retail**: 3 cities (0.4%)
- **Public Services**: 3 cities (0.4%)
- **Attractions And Activities**: 3 cities (0.4%)
- **Arts And Entertainment**: 3 cities (0.4%)
- **Eat And Drink**: 3 cities (0.4%)
- **Active Life**: 3 cities (0.4%)
- **Education**: 2 cities (0.3%)
- **Accommodation**: 1 cities (0.1%)

## Data Quality Patterns

### Confidence by City Size

| City Size | Mean Confidence | N Cities |
|-----------|-----------------|----------|
| Small (<100k) | 1.000 | 304 |
| Medium (100k-500k) | 1.000 | 331 |
| Large (>500k) | 0.914 | 64 |

### Cities with Labels

- **Cities with geographic labels**: 694 (99.3%)
- **Mean confidence (labeled cities)**: 0.992
- **Mean confidence (unlabeled cities)**: 1.000

## Top 50 Most Robust Cities

Cities with highest confidence scores (best data quality):

| Rank | Bounds FID | City Label | Population | Area (km²) | Confidence | Mean Z-Score | Flagged Categories |
|------|------------|------------|------------|------------|------------|--------------|-------------------|
| 1 | 0 | N/A | 102,792 | 62.8 | 1.000 | 0.01 | None |
| 2 | 469 | Ferrara | 104,855 | 60.4 | 1.000 | N/A | None |
| 3 | 461 | Ferrol | 99,747 | 42.5 | 1.000 | N/A | None |
| 4 | 462 | Русе | 120,796 | 41.7 | 1.000 | N/A | None |
| 5 | 463 | Valence | 97,670 | 50.9 | 1.000 | N/A | None |
| 6 | 464 | Piacenza | 97,694 | 39.0 | 1.000 | N/A | None |
| 7 | 465 | Torino | 1,189,078 | 270.2 | 1.000 | N/A | None |
| 8 | 466 | A Coruña | 308,019 | 78.9 | 1.000 | N/A | None |
| 9 | 467 | Asti | 62,198 | 26.8 | 1.000 | N/A | None |
| 10 | 468 | Alessandria | 66,599 | 28.5 | 1.000 | N/A | None |
| 11 | 470 | Parma | 163,531 | 56.0 | 1.000 | N/A | None |
| 12 | 459 | Cremona | 67,360 | 30.8 | 1.000 | N/A | None |
| 13 | 471 | Carpi | 66,232 | 35.6 | 1.000 | N/A | None |
| 14 | 472 | Avilés | 96,051 | 34.3 | 1.000 | N/A | None |
| 15 | 473 | Варна | 277,393 | 67.9 | 1.000 | N/A | None |
| 16 | 474 | Gijón / Xixón | 256,259 | 46.2 | 1.000 | N/A | None |
| 17 | 475 | Reggio Emilia | 128,161 | 54.0 | 1.000 | N/A | None |
| 18 | 476 | Шумен | 64,407 | 35.6 | 1.000 | N/A | None |
| 19 | 477 | Modena | 164,441 | 56.0 | 1.000 | N/A | None |
| 20 | 478 | Oviedo | 216,609 | 47.1 | 1.000 | N/A | None |
| 21 | 460 | Добрич | 72,255 | 30.6 | 1.000 | N/A | None |
| 22 | 458 | Pavia | 72,809 | 36.8 | 1.000 | N/A | None |
| 23 | 480 | Lugo | 90,400 | 25.9 | 1.000 | N/A | None |
| 24 | 447 | Verona | 218,185 | 71.5 | 1.000 | N/A | None |
| 25 | 439 | Călărași | 57,835 | 31.5 | 1.000 | N/A | None |
| 26 | 440 | Sector 1 | 1,949,460 | 343.9 | 1.000 | N/A | None |
| 27 | 441 | Saint-Étienne | 203,061 | 74.9 | 1.000 | N/A | None |
| 28 | 442 | Busto Arsizio | 341,660 | 154.7 | 1.000 | N/A | None |
| 29 | 443 | Vicenza | 110,485 | 52.6 | 1.000 | N/A | None |
| 30 | 444 | Brescia | 261,170 | 123.8 | 1.000 | N/A | None |
| 31 | 445 | Venezia | 194,689 | 69.0 | 1.000 | N/A | None |
| 32 | 446 | Novara | 96,545 | 43.0 | 1.000 | N/A | None |
| 33 | 448 | Rijeka | 122,335 | 60.9 | 1.000 | N/A | None |
| 34 | 457 | Mofleni | 235,542 | 67.3 | 1.000 | N/A | None |
| 35 | 449 | Venezia | 66,929 | 58.2 | 1.000 | N/A | None |
| 36 | 450 | Slatina | 63,115 | 25.8 | 1.000 | N/A | None |
| 37 | 451 | Magheru | 77,662 | 27.9 | 1.000 | N/A | None |
| 38 | 452 | Padova | 286,230 | 119.8 | 1.000 | N/A | None |
| 39 | 453 | Milano | 3,246,677 | 850.9 | 1.000 | N/A | None |
| 40 | 454 | Vigevano | 61,330 | 35.1 | 1.000 | N/A | None |
| 41 | 455 | Bordeaux | 766,593 | 268.3 | 1.000 | N/A | None |
| 42 | 456 | Grenoble | 356,658 | 109.1 | 1.000 | N/A | None |
| 43 | 479 | Santiago | 88,081 | 33.2 | 1.000 | N/A | None |
| 44 | 481 | Bologna | 471,395 | 146.5 | 1.000 | N/A | None |
| 45 | 524 | София | 1,131,842 | 226.8 | 1.000 | N/A | None |
| 46 | 513 | Pisa | 137,932 | 86.5 | 1.000 | N/A | None |
| 47 | 505 | Ourense | 112,971 | 36.8 | 1.000 | N/A | None |
| 48 | 506 | Prato | 225,125 | 91.6 | 1.000 | N/A | None |
| 49 | 507 | Pau | 125,886 | 54.5 | 1.000 | N/A | None |
| 50 | 508 | León | 168,719 | 53.3 | 1.000 | N/A | None |

## Bottom 50 Least Confident Cities

Cities with lowest confidence scores (likely data quality issues):

| Rank | Bounds FID | City Label | Population | Area (km²) | Confidence | N Flags | Flagged Categories |
|------|------------|------------|------------|------------|------------|---------|-------------------|
| 1 | 100 | Mitte | 3,622,646 | 837.3 | 0.273 | 6 | business_and_services, active_life, public_services, reta... |
| 2 | 221 | Lille | 1,027,354 | 322.9 | 0.283 | 6 | business_and_services, active_life, arts_and_entertainmen... |
| 3 | 170 | Innenstadt West | 3,039,424 | 1199.3 | 0.327 | 5 | active_life, public_services, eat_and_drink, attractions_... |
| 4 | 213 | Brussel-Hoofdstad - Bruxelles-Capitale | 1,438,593 | 270.5 | 0.436 | 3 | business_and_services, health_and_medical, education |
| 5 | 51 | Dublin | 1,269,195 | 479.3 | 0.436 | 3 | business_and_services, retail, health_and_medical |
| 6 | 7 | Helsinki | 1,079,410 | 464.9 | 0.491 | 2 | health_and_medical, religious |
| 7 | 92 | Warszawa | 1,943,995 | 457.3 | 0.549 | 2 | arts_and_entertainment, eat_and_drink |
| 8 | 128 | Łódź | 649,615 | 177.1 | 0.621 | 2 | arts_and_entertainment, eat_and_drink |
| 9 | 15 | Sjöberg | 1,499,000 | 480.8 | 0.622 | 3 | retail, health_and_medical, religious |
| 10 | 65 | Hamburg-Mitte | 1,740,612 | 532.5 | 0.721 | 2 | public_services, attractions_and_activities |
| 11 | 123 | Den Haag | 865,515 | 209.1 | 0.895 | 1 | business_and_services |
| 12 | 25 | Latgales apkaime | 556,989 | 175.8 | 0.902 | 1 | health_and_medical |
| 13 | 181 | Ost | 602,158 | 206.8 | 0.926 | 1 | religious |
| 14 | 237 | Katowice | 1,199,248 | 448.9 | 1.000 | 0 | None |
| 15 | 236 | Most | 61,731 | 27.0 | 1.000 | 0 | None |
| 16 | 235 | Namur | 86,381 | 51.0 | 1.000 | 0 | None |
| 17 | 238 | Tarnów | 91,508 | 36.4 | 1.000 | 0 | None |
| 18 | 233 | Douai | 103,514 | 62.4 | 1.000 | 0 | None |
| 19 | 239 | Charleroi | 259,702 | 133.7 | 1.000 | 0 | None |
| 20 | 232 | Gießen | 82,850 | 45.4 | 1.000 | 0 | None |
| 21 | 231 | Sosnowiec | 103,588 | 47.3 | 1.000 | 0 | None |
| 22 | 230 | La Louvière | 112,643 | 70.7 | 1.000 | 0 | None |
| 23 | 229 | Gliwice | 143,732 | 53.0 | 1.000 | 0 | None |
| 24 | 228 | Lens | 215,679 | 115.3 | 1.000 | 0 | None |
| 25 | 227 | Sosnowiec | 99,571 | 19.8 | 1.000 | 0 | None |
| 26 | 234 | Mons | 108,073 | 64.0 | 1.000 | 0 | None |
| 27 | 240 | Jirkov | 65,106 | 35.5 | 1.000 | 0 | None |
| 28 | 243 | Arras | 82,779 | 46.3 | 1.000 | 0 | None |
| 29 | 242 | Plauen | 59,725 | 41.1 | 1.000 | 0 | None |
| 30 | 260 | Havířov | 76,218 | 39.2 | 1.000 | 0 | None |
| 31 | 259 | Praha | 1,228,032 | 330.1 | 1.000 | 0 | None |
| 32 | 258 | Innenstadt 1 | 963,622 | 290.5 | 1.000 | 0 | None |
| 33 | 257 | Hanau | 96,667 | 56.7 | 1.000 | 0 | None |
| 34 | 256 | Nowy Sącz | 76,896 | 34.7 | 1.000 | 0 | None |
| 35 | 255 | Ostrava | 75,123 | 29.6 | 1.000 | 0 | None |
| 36 | 254 | Ostrava | 74,531 | 35.1 | 1.000 | 0 | None |
| 37 | 253 | Amiens | 155,730 | 63.5 | 1.000 | 0 | None |
| 38 | 252 | Bielsko-Biała | 149,251 | 57.4 | 1.000 | 0 | None |
| 39 | 251 | Pardubice | 84,685 | 39.4 | 1.000 | 0 | None |
| 40 | 250 | Kladno | 69,450 | 30.5 | 1.000 | 0 | None |
| 41 | 249 | Cherbourg | 75,905 | 48.3 | 1.000 | 0 | None |
| 42 | 248 | Jastrzębie-Zdrój | 69,269 | 31.7 | 1.000 | 0 | None |
| 43 | 247 | Hradec Králové | 86,959 | 46.0 | 1.000 | 0 | None |
| 44 | 246 | Lahnstein | 137,225 | 79.9 | 1.000 | 0 | None |
| 45 | 245 | Kraków | 787,456 | 209.1 | 1.000 | 0 | None |
| 46 | 244 | Tychy | 118,409 | 44.7 | 1.000 | 0 | None |
| 47 | 241 | Valenciennes | 133,355 | 74.4 | 1.000 | 0 | None |
| 48 | 226 | Rzeszów | 176,377 | 73.5 | 1.000 | 0 | None |
| 49 | 698 | Il-Gudja | 379,126 | 131.0 | 1.000 | 0 | None |
| 50 | 224 | Liège | 443,283 | 191.7 | 1.000 | 0 | None |

### Detailed Category Breakdown (Bottom 10 Cities)

#### 1. Bounds FID 100 - Mitte

- **Confidence Score**: 0.273
- **Population**: 3,622,646
- **Area**: 837.3 km²
- **Flagged Categories**: 6

**Category Z-Scores:**

- Business And Services: Z=-3.26 (Observed=57860, Expected=62847.4) ⚠️ **FLAGGED**
- Active Life: Z=-4.41 (Observed=5075, Expected=5811.2) ⚠️ **FLAGGED**
- Arts And Entertainment: Z=-1.26 (Observed=3115, Expected=3214.0)
- Public Services: Z=-3.78 (Observed=6768, Expected=7506.6) ⚠️ **FLAGGED**
- Retail: Z=-4.94 (Observed=25667, Expected=28602.3) ⚠️ **FLAGGED**
- Health And Medical: Z=1.04 (Observed=10781, Expected=10458.8)
- Eat And Drink: Z=-0.79 (Observed=16932, Expected=17283.6)
- Education: Z=-4.81 (Observed=6030, Expected=6875.5) ⚠️ **FLAGGED**
- Attractions And Activities: Z=-1.22 (Observed=4559, Expected=4810.8)
- Religious: Z=-3.10 (Observed=1288, Expected=1437.5) ⚠️ **FLAGGED**
- Accommodation: Z=-1.77 (Observed=2183, Expected=2480.1)

#### 2. Bounds FID 221 - Lille

- **Confidence Score**: 0.283
- **Population**: 1,027,354
- **Area**: 322.9 km²
- **Flagged Categories**: 6

**Category Z-Scores:**

- Business And Services: Z=-3.44 (Observed=12748, Expected=18007.2) ⚠️ **FLAGGED**
- Active Life: Z=-2.06 (Observed=1402, Expected=1746.9) ⚠️ **FLAGGED**
- Arts And Entertainment: Z=-3.13 (Observed=688, Expected=934.0) ⚠️ **FLAGGED**
- Public Services: Z=-0.79 (Observed=1868, Expected=2022.8)
- Retail: Z=-1.29 (Observed=7345, Expected=8109.1)
- Health And Medical: Z=-3.95 (Observed=1865, Expected=3093.0) ⚠️ **FLAGGED**
- Eat And Drink: Z=-0.52 (Observed=4295, Expected=4527.1)
- Education: Z=-1.91 (Observed=1579, Expected=1915.4)
- Attractions And Activities: Z=-2.47 (Observed=774, Expected=1284.3) ⚠️ **FLAGGED**
- Religious: Z=-3.03 (Observed=328, Expected=473.8) ⚠️ **FLAGGED**
- Accommodation: Z=-1.09 (Observed=477, Expected=659.7)

#### 3. Bounds FID 170 - Innenstadt West

- **Confidence Score**: 0.327
- **Population**: 3,039,424
- **Area**: 1199.3 km²
- **Flagged Categories**: 5

**Category Z-Scores:**

- Business And Services: Z=-1.71 (Observed=49249, Expected=51871.1)
- Active Life: Z=-5.09 (Observed=4355, Expected=5205.0) ⚠️ **FLAGGED**
- Arts And Entertainment: Z=-1.72 (Observed=2698, Expected=2833.2)
- Public Services: Z=-4.31 (Observed=4429, Expected=5271.9) ⚠️ **FLAGGED**
- Retail: Z=-1.97 (Observed=21211, Expected=22384.3)
- Health And Medical: Z=1.40 (Observed=9969, Expected=9533.9)
- Eat And Drink: Z=-5.31 (Observed=9574, Expected=11948.6) ⚠️ **FLAGGED**
- Education: Z=-1.57 (Observed=4936, Expected=5212.7)
- Attractions And Activities: Z=-6.32 (Observed=2048, Expected=3354.2) ⚠️ **FLAGGED**
- Religious: Z=1.55 (Observed=1698, Expected=1623.4)
- Accommodation: Z=-3.61 (Observed=1009, Expected=1613.8) ⚠️ **FLAGGED**

#### 4. Bounds FID 213 - Brussel-Hoofdstad - Bruxelles-Capitale

- **Confidence Score**: 0.436
- **Population**: 1,438,593
- **Area**: 270.5 km²
- **Flagged Categories**: 3

**Category Z-Scores:**

- Business And Services: Z=-3.98 (Observed=19277, Expected=25364.5) ⚠️ **FLAGGED**
- Active Life: Z=-1.33 (Observed=2082, Expected=2304.8)
- Arts And Entertainment: Z=0.94 (Observed=1332, Expected=1258.4)
- Public Services: Z=6.98 (Observed=4537, Expected=3172.5)
- Retail: Z=-1.54 (Observed=10892, Expected=11808.3)
- Health And Medical: Z=-4.11 (Observed=2777, Expected=4055.5) ⚠️ **FLAGGED**
- Eat And Drink: Z=1.76 (Observed=8029, Expected=7240.1)
- Education: Z=-2.96 (Observed=2334, Expected=2855.0) ⚠️ **FLAGGED**
- Attractions And Activities: Z=1.17 (Observed=2271, Expected=2029.2)
- Religious: Z=4.41 (Observed=726, Expected=513.7)
- Accommodation: Z=-1.31 (Observed=856, Expected=1076.3)

#### 5. Bounds FID 51 - Dublin

- **Confidence Score**: 0.436
- **Population**: 1,269,195
- **Area**: 479.3 km²
- **Flagged Categories**: 3

**Category Z-Scores:**

- Business And Services: Z=-4.16 (Observed=15611, Expected=21981.3) ⚠️ **FLAGGED**
- Active Life: Z=1.67 (Observed=2474, Expected=2195.4)
- Arts And Entertainment: Z=-0.36 (Observed=1148, Expected=1176.3)
- Public Services: Z=-1.10 (Observed=2092, Expected=2306.2)
- Retail: Z=-2.87 (Observed=7950, Expected=9652.6) ⚠️ **FLAGGED**
- Health And Medical: Z=-4.05 (Observed=2686, Expected=3945.2) ⚠️ **FLAGGED**
- Eat And Drink: Z=1.81 (Observed=5964, Expected=5155.3)
- Education: Z=-0.43 (Observed=2178, Expected=2253.1)
- Attractions And Activities: Z=0.36 (Observed=1537, Expected=1462.1)
- Religious: Z=2.55 (Observed=778, Expected=655.0)
- Accommodation: Z=1.43 (Observed=965, Expected=725.5)

#### 6. Bounds FID 7 - Helsinki

- **Confidence Score**: 0.491
- **Population**: 1,079,410
- **Area**: 464.9 km²
- **Flagged Categories**: 2

**Category Z-Scores:**

- Business And Services: Z=-0.21 (Observed=18337, Expected=18658.6)
- Active Life: Z=2.41 (Observed=2316, Expected=1913.1)
- Arts And Entertainment: Z=-0.15 (Observed=1004, Expected=1016.0)
- Public Services: Z=4.11 (Observed=2660, Expected=1856.4)
- Retail: Z=-1.38 (Observed=7253, Expected=8075.6)
- Health And Medical: Z=-4.64 (Observed=1999, Expected=3442.1) ⚠️ **FLAGGED**
- Eat And Drink: Z=3.06 (Observed=5474, Expected=4102.5)
- Education: Z=-1.47 (Observed=1605, Expected=1863.8)
- Attractions And Activities: Z=7.07 (Observed=2632, Expected=1171.9)
- Religious: Z=-2.39 (Observed=489, Expected=604.1) ⚠️ **FLAGGED**
- Accommodation: Z=0.41 (Observed=638, Expected=570.1)

#### 7. Bounds FID 92 - Warszawa

- **Confidence Score**: 0.549
- **Population**: 1,943,995
- **Area**: 457.3 km²
- **Flagged Categories**: 2

**Category Z-Scores:**

- Business And Services: Z=5.63 (Observed=42551, Expected=33933.4)
- Active Life: Z=1.01 (Observed=3321, Expected=3152.0)
- Arts And Entertainment: Z=-3.50 (Observed=1451, Expected=1726.2) ⚠️ **FLAGGED**
- Public Services: Z=2.54 (Observed=4556, Expected=4058.7)
- Retail: Z=3.48 (Observed=17576, Expected=15506.6)
- Health And Medical: Z=0.82 (Observed=5878, Expected=5622.3)
- Eat And Drink: Z=-3.71 (Observed=7616, Expected=9276.6) ⚠️ **FLAGGED**
- Education: Z=10.14 (Observed=5504, Expected=3721.5)
- Attractions And Activities: Z=-1.00 (Observed=2388, Expected=2595.5)
- Religious: Z=-1.96 (Observed=680, Expected=774.3)
- Accommodation: Z=-0.58 (Observed=1251, Expected=1347.8)

#### 8. Bounds FID 128 - Łódź

- **Confidence Score**: 0.621
- **Population**: 649,615
- **Area**: 177.1 km²
- **Flagged Categories**: 2

**Category Z-Scores:**

- Business And Services: Z=0.17 (Observed=11873, Expected=11614.4)
- Active Life: Z=-1.98 (Observed=779, Expected=1109.7)
- Arts And Entertainment: Z=-2.17 (Observed=412, Expected=582.6) ⚠️ **FLAGGED**
- Public Services: Z=-1.49 (Observed=1083, Expected=1373.3)
- Retail: Z=0.06 (Observed=5403, Expected=5365.7)
- Health And Medical: Z=-0.95 (Observed=1617, Expected=1912.5)
- Eat And Drink: Z=-3.35 (Observed=1538, Expected=3036.8) ⚠️ **FLAGGED**
- Education: Z=0.71 (Observed=1400, Expected=1274.8)
- Attractions And Activities: Z=-1.52 (Observed=556, Expected=869.5)
- Religious: Z=-1.68 (Observed=193, Expected=273.8)
- Accommodation: Z=-0.84 (Observed=322, Expected=462.6)

#### 9. Bounds FID 15 - Sjöberg

- **Confidence Score**: 0.622
- **Population**: 1,499,000
- **Area**: 480.8 km²
- **Flagged Categories**: 3

**Category Z-Scores:**

- Business And Services: Z=-0.41 (Observed=25408, Expected=26034.8)
- Active Life: Z=2.92 (Observed=3015, Expected=2526.9)
- Arts And Entertainment: Z=-0.57 (Observed=1321, Expected=1366.1)
- Public Services: Z=3.24 (Observed=3518, Expected=2884.2)
- Retail: Z=-2.97 (Observed=9852, Expected=11615.9) ⚠️ **FLAGGED**
- Health And Medical: Z=-3.07 (Observed=3574, Expected=4530.1) ⚠️ **FLAGGED**
- Eat And Drink: Z=3.51 (Observed=8083, Expected=6512.5)
- Education: Z=-0.04 (Observed=2736, Expected=2742.9)
- Attractions And Activities: Z=4.36 (Observed=2737, Expected=1835.6)
- Religious: Z=-2.06 (Observed=604, Expected=703.2) ⚠️ **FLAGGED**
- Accommodation: Z=2.44 (Observed=1337, Expected=928.8)

#### 10. Bounds FID 65 - Hamburg-Mitte

- **Confidence Score**: 0.721
- **Population**: 1,740,612
- **Area**: 532.5 km²
- **Flagged Categories**: 2

**Category Z-Scores:**

- Business And Services: Z=5.57 (Observed=38730, Expected=30201.9)
- Active Life: Z=0.40 (Observed=2975, Expected=2907.8)
- Arts And Entertainment: Z=-0.22 (Observed=1562, Expected=1579.4)
- Public Services: Z=-2.85 (Observed=2830, Expected=3386.6) ⚠️ **FLAGGED**
- Retail: Z=3.79 (Observed=15762, Expected=13510.7)
- Health And Medical: Z=4.85 (Observed=6732, Expected=5221.7)
- Eat And Drink: Z=0.37 (Observed=7846, Expected=7680.1)
- Education: Z=-0.92 (Observed=3039, Expected=3200.2)
- Attractions And Activities: Z=-2.25 (Observed=1693, Expected=2158.5) ⚠️ **FLAGGED**
- Religious: Z=-1.27 (Observed=735, Expected=796.2)
- Accommodation: Z=-0.31 (Observed=1042, Expected=1094.6)

## Recommendations for Dataset Usage

### Data Quality Tiers

1. **High Confidence (≥0.7)**: 690 cities - Suitable for all analyses including detailed land-use studies
2. **Moderate Confidence (0.4-0.7)**: 6 cities - Suitable for aggregate analyses; use caution for category-specific studies
3. **Low Confidence (<0.4)**: 3 cities - Recommend excluding from analyses or treating as missing data

### Specific Recommendations

1. **Exclusion criterion**: Consider excluding 3 cities with confidence < 0.4 from regression analyses to avoid biasing results

2. **Category-specific caution**: For analyses focused on specific land uses, filter cities based on category-specific z-scores rather than overall confidence

3. **Robust methods**: Use robust regression techniques (e.g., M-estimators) that downweight outliers when including all cities

4. **Data improvement**: Cities with low confidence scores may benefit from manual validation or supplementary data sources (e.g., official business registries)

---

*Report generated automatically from city-level POI aggregation and regression analysis.*
