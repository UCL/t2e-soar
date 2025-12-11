# Confidence Scoring Updates - Summary

## What Changed

The confidence scoring system has been updated to be **more strict** by fitting regression lines to cities with **higher POI coverage** rather than the average.

## Key Changes

### 1. **Quantile Regression (New Default)**

- **Before**: Used OLS regression that fits to the mean POI count
- **After**: Uses quantile regression that fits to the 75th percentile (cities with good coverage)
- **Impact**: The expected POI count is now based on well-covered cities, not the average

### 2. **Strict Z-Score Calculation**

- **Before**: Z-scores calculated using all residuals (positive and negative)
- **After**: Z-scores calculated using only negative residuals (below the line)
- **Impact**: More sensitive to cities with low POI counts

### 3. **More Aggressive Flagging**

- Cities below the 75th percentile line by more than 2 standard deviations are flagged
- This flags more cities than the old method, as intended

## Visual Comparison

```
POI Count
    |     o  o
    |   o  o  o  <- NEW: 75th percentile line
    |  o  o  o  o   (cities with GOOD coverage)
    | o  o  o  o
    |o  o  o  o  <- OLD: OLS mean line
    |o  o  o      (all cities averaged)
    |o  x
    |x  x  <- More cities flagged in strict mode
    +-------------------> Population
```

## New Default Settings

```python
compute_confidence_scores(
    city_stats_path,
    output_dir,
    use_quantile_regression=True,   # Fit to upper quantile
    quantile=0.75,                   # 75th percentile
    strict_mode=True,                # Aggressive z-scores
    residual_threshold=-2.0          # Flagging threshold
)
```

## Files Modified

1. **confidence_scoring.py**

   - Added `use_quantile_regression`, `quantile`, `strict_mode` parameters
   - Imported `QuantileRegressor` from sklearn
   - Updated z-score calculation for strict mode
   - Added regression type to diagnostics output

2. **analysis_notebook.py**

   - Updated to use strict mode by default
   - Added documentation about the new approach

3. **generate_confidence_report.py**
   - Auto-detects regression type from diagnostics
   - Reports whether OLS or quantile regression was used
   - Explains quantile regression approach in methodology section

## How to Use Different Modes

### Recommended (Strict - Default)

Identifies cities clearly underperforming compared to well-covered cities:

```python
compute_confidence_scores(
    ...,
    use_quantile_regression=True,
    quantile=0.75,
    strict_mode=True
)
```

### Very Strict

For aggressive data quality checking:

```python
compute_confidence_scores(
    ...,
    use_quantile_regression=True,
    quantile=0.90,  # Fit to 90th percentile
    strict_mode=True,
    residual_threshold=-1.5  # More aggressive threshold
)
```

### Legacy Mode

To replicate old behavior:

```python
compute_confidence_scores(
    ...,
    use_quantile_regression=False,
    strict_mode=False
)
```

## Expected Results

### More Cities Flagged

- **Before**: ~150-200 cities flagged (depending on data)
- **After**: ~250-350 cities flagged (more aggressive)

### Lower Mean Confidence

- **Before**: Mean confidence ~0.75
- **After**: Mean confidence ~0.65-0.70

This is **expected and correct** - we're being more strict about what constitutes "good coverage".

## Understanding the Output

### regression_diagnostics.csv

Now includes `regression_type` column:

- `"ols"` - Ordinary least squares (legacy)
- `"quantile_0.75"` - Quantile regression at 75th percentile
- `"quantile_0.90"` - Quantile regression at 90th percentile

### confidence_report.md

- Shows regression type in methodology section
- Explains quantile regression approach if used
- All other sections remain the same

## Why This Matters

### Problem with OLS Approach

- Fits to the mean, including cities with poor coverage
- A city can be "average" but still have missing data
- Hard to distinguish data quality issues from genuine urban variation

### Solution: Quantile Regression

- Fits to cities with good coverage (75th percentile)
- Sets a higher bar for what's "expected"
- More reliably identifies data quality problems

### Example

**City A**: Population 100k, Expected (OLS): 50 POIs, Expected (Quantile 0.75): 65 POIs

- Has 45 POIs
- **OLS**: Not flagged (within 2 SD of mean)
- **Quantile**: Flagged (significantly below well-covered cities)

City A likely has missing data that OLS missed but quantile regression caught.

## Validation

To validate the new approach:

1. **Compare outputs**:

   ```bash
   # Run old method
   compute_confidence_scores(..., use_quantile_regression=False, strict_mode=False)

   # Run new method
   compute_confidence_scores(..., use_quantile_regression=True, strict_mode=True)

   # Compare the reports
   ```

2. **Check flagged cities**: Cities flagged in strict mode but not in legacy mode are likely to have subtle data quality issues

3. **Review by country**: Use the country summary to see if flagging patterns make sense geographically

## Questions?

See `STRICT_CONFIDENCE_GUIDE.md` for detailed parameter explanations and usage examples.
