# Quick Reference: Strict Confidence Scoring

## TL;DR

**What changed**: Regression now fits to cities with **high POI coverage** (75th percentile) instead of average, making flagging more aggressive for cities below the line.

## Quick Parameter Guide

| Parameter | Default | What it does | Effect |
|-----------|---------|--------------|--------|
| `use_quantile_regression` | `True` | Fit to upper quantile vs mean | More strict |
| `quantile` | `0.75` | Which percentile to fit (0.0-1.0) | Higher = stricter |
| `strict_mode` | `True` | Use only negative residuals for SD | More sensitive |
| `residual_threshold` | `-2.0` | Z-score cutoff for flagging | Higher = stricter |

## Common Configurations

```python
# Default (Recommended)
compute_confidence_scores(city_stats, output)
# Fits to 75th percentile, strict z-scores

# Very Strict
compute_confidence_scores(city_stats, output, quantile=0.90, residual_threshold=-1.5)
# Fits to 90th percentile, aggressive flagging

# Moderate
compute_confidence_scores(city_stats, output, quantile=0.75, strict_mode=False)
# Fits to 75th percentile, standard z-scores

# Legacy (Old Behavior)
compute_confidence_scores(city_stats, output, use_quantile_regression=False, strict_mode=False)
# Standard OLS regression
```

## What to Expect

- **More cities flagged** (good - we want to be strict)
- **Lower mean confidence** (expected - higher standards)
- **Better identification** of subtle data quality issues

## When to Adjust

- **Too many false positives**: Lower `quantile` to 0.65 or set `strict_mode=False`
- **Missing real issues**: Raise `quantile` to 0.85 or `residual_threshold` to -1.5
- **Need to match old results**: Use legacy mode

## Check Your Results

```python
# Compare old vs new
import pandas as pd

old_conf = pd.read_csv("output_old/city_confidence.csv")
new_conf = pd.read_csv("output_new/city_confidence.csv")

print(f"Old: {(old_conf['n_flagged_categories'] > 0).sum()} cities flagged")
print(f"New: {(new_conf['n_flagged_categories'] > 0).sum()} cities flagged")
print(f"Old mean confidence: {old_conf['confidence_score'].mean():.3f}")
print(f"New mean confidence: {new_conf['confidence_score'].mean():.3f}")
```

## Documentation

- Full guide: `STRICT_CONFIDENCE_GUIDE.md`
- Summary: `CONFIDENCE_UPDATES_SUMMARY.md`
