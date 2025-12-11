# Strict Confidence Scoring Guide

## Overview

The confidence scoring has been updated to support **strict mode** that fits regression lines to cities with higher POI coverage, making it more aggressive at flagging cities with low POI counts.

## New Parameters

### `use_quantile_regression` (default: `True`)
- **What it does**: Instead of fitting to the mean (OLS regression), fits to an upper quantile
- **Why it matters**: Cities with good POI coverage define the "expected" line, not the average
- **Effect**: More cities will be flagged as having low coverage

### `quantile` (default: `0.75`)
- **What it does**: Which quantile to fit when using quantile regression
- **Recommended values**:
  - `0.75` (75th percentile) - Moderately strict, good balance
  - `0.80` (80th percentile) - More strict
  - `0.90` (90th percentile) - Very strict, only best cities define the line
- **Effect**: Higher values = stricter flagging

### `strict_mode` (default: `True`)
- **What it does**: Changes how z-scores are calculated for cities below the regression line
- **When enabled**: Uses only negative residuals to compute standard deviation
- **Effect**: Makes flagging thresholds more sensitive to low POI counts

### `residual_threshold` (default: `-2.0`)
- **What it does**: Z-score below which cities are flagged
- **Recommended values**:
  - `-2.0` - Standard (95% confidence)
  - `-1.5` - More aggressive flagging
  - `-1.0` - Very aggressive flagging
- **Effect**: Lower (more negative) = less strict, Higher (closer to 0) = more strict

## Usage Examples

### Default (Strict Mode)
```python
from src.analysis.modules import compute_confidence_scores

# Fits to 75th percentile, strict z-score calculation
compute_confidence_scores(
    "temp/city_stats.gpkg",
    "output/",
    use_quantile_regression=True,
    quantile=0.75,
    strict_mode=True,
    residual_threshold=-2.0
)
```

### Very Strict Mode
```python
# Fits to 90th percentile with aggressive flagging
compute_confidence_scores(
    "temp/city_stats.gpkg",
    "output/",
    use_quantile_regression=True,
    quantile=0.90,
    strict_mode=True,
    residual_threshold=-1.5
)
```

### Legacy Mode (Original Behavior)
```python
# Standard OLS regression, traditional z-scores
compute_confidence_scores(
    "temp/city_stats.gpkg",
    "output/",
    use_quantile_regression=False,
    strict_mode=False,
    residual_threshold=-2.0
)
```

## How It Works

### Quantile Regression Approach

1. **Fit regression line to upper quantile**: Instead of fitting `POI_count ~ population + area` to minimize overall error, we fit to the Qth quantile (e.g., 75th percentile)

2. **This line represents "good coverage"**: Cities at the 75th percentile or higher have good POI data

3. **Cities below are flagged more aggressively**: Any city significantly below this line (< -2 SD) is flagged

### Strict Z-Score Calculation

In strict mode:
- **Standard deviation** is computed using only cities BELOW the regression line
- **Mean** is set to 0 (the regression line itself)
- This makes the threshold more sensitive to low values

### Visual Example

```
POI Count
    |     o  o  <- 90th percentile line (very strict)
    |   o  o  o
    |  o  o  o  <- 75th percentile line (strict default)
    | o  o  o  o
    |o  o  o  o  <- OLS mean line (legacy)
    |o  o  o
    |o  x  <- These cities flagged in strict mode
    |x  x     but might not be in legacy mode
    +-------------------> Population
```

## Choosing Settings

### For Quality Control (Recommended)
- `use_quantile_regression=True`
- `quantile=0.75`
- `strict_mode=True`
- `residual_threshold=-2.0`

This identifies cities that are clearly underperforming compared to well-covered cities.

### For Data Cleaning (Very Strict)
- `use_quantile_regression=True`
- `quantile=0.90`
- `strict_mode=True`
- `residual_threshold=-1.5`

This is very aggressive and will flag many cities. Use when you want to identify ANY potential data issues.

### For Analysis (Balanced)
- `use_quantile_regression=True`
- `quantile=0.75`
- `strict_mode=False`
- `residual_threshold=-2.0`

Good balance for analytical work where you want to account for well-covered cities but not be overly aggressive.

## Output Changes

The `regression_diagnostics.csv` now includes:
- `regression_type`: Shows whether OLS or quantile regression was used (e.g., "quantile_0.75")

This helps you track which method was used for each analysis run.

## Notes

- **More flags ≠ bad data**: In strict mode, you'll flag more cities. This is intentional - it helps identify potential issues.
- **Context matters**: A flagged city in strict mode might still be usable for some analyses
- **Compare modes**: Run both legacy and strict mode to see the difference
- **Category-specific**: Different categories may need different thresholds
