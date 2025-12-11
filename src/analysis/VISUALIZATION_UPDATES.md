# Visualization Updates: Regression Fit Plots

## Summary

Added comprehensive visualizations showing quantile regression fits and flagged cities to both the exploratory data analysis and confidence scoring outputs.

## Changes Made

### 1. Confidence Scoring Module (`confidence_scoring.py`)

**New Feature**: Generates regression fit plots for each POI category

**What's Created**:

- Directory: `output/regression_plots/`
- Files: One plot per category (e.g., `eat_and_drink_regression_fit.png`)

**Plot Features**:

- **Blue dots**: Cities with acceptable POI coverage (non-flagged)
- **Red X marks**: Flagged cities (z-score < -2.0)
- **Green dashed line**: Quantile regression fit (75th percentile by default)
- **Title includes**: R² score and number of flagged cities
- **X-axis**: Population (using median area for visualization)
- **Y-axis**: POI count for that category

**Benefits**:

- Visually identifies which cities fall significantly below the expected line
- Shows the quality of the regression fit
- Highlights the distribution of flagged vs non-flagged cities

### 2. Confidence Report (`generate_confidence_report.py`)

**New Section**: "Regression Fit Visualizations"

**Location**: Added after the regression diagnostics table

**Content**:

- Embeds plots for key categories: eat_and_drink, retail, education, business_and_services
- Includes explanatory text about what the plots show
- Uses markdown image syntax with relative paths
- Links to full plot directory for additional categories

**Example Output**:

```markdown
### Regression Fit Visualizations

The following plots show the regression fit for each category. Blue dots represent
cities with acceptable POI coverage, while red X marks indicate flagged cities
(those significantly below the expected line).

#### Eat And Drink

![Eat And Drink Regression Fit](regression_plots/eat_and_drink_regression_fit.png)

...
```

### 3. Exploratory Data Analysis (`explore_city_stats.py`)

**Enhanced Feature**: Scatter plots now include both OLS and Q75 regression lines

**Updates to Existing Plots**:

- **Red dashed line**: OLS (Ordinary Least Squares) regression - traditional mean fit
- **Green dashed line**: Q75 (75th percentile) quantile regression - strict mode fit
- Both lines labeled with equations
- Shows the difference between mean-based and quantile-based approaches

**Benefits**:

- Direct comparison of OLS vs quantile regression
- Helps understand why quantile regression is stricter
- Visualizes the gap between "average" and "good coverage"

## How to Use

### Running the Analysis

The plots are generated automatically when you run the analysis pipeline:

```python
from src.analysis.modules import compute_confidence_scores

# This now generates plots in addition to scoring
compute_confidence_scores(
    "city_stats.gpkg",
    "output/",
    use_quantile_regression=True,
    quantile=0.75
)
```

### Viewing the Plots

**Option 1: In the Confidence Report**

- Open `output/confidence_report.md` in a markdown viewer
- Key plots are embedded in the report

**Option 2: In the Plots Directory**

- Navigate to `output/regression_plots/`
- View all category plots individually

**Option 3: In EDA Scatter Plots**

- Navigate to `output/eda/`
- View `scatter_<category>.png` files
- These show both OLS and Q75 lines

## Understanding the Visualizations

### Regression Fit Plots (from confidence_scoring)

**What They Show**:

```
POI Count
    |     o  o  o  <- Cities above the line (good coverage)
    |   o  o  o
    |  o  o  /  <- Green line: Q75 regression
    | o  o /o
    |o  o/  o
    |o /x  x  <- Red X: Flagged cities (below line)
    |x/  x
    +-------------------> Population
```

**Interpretation**:

- **Tight clustering around line**: Good model fit (high R²)
- **Many red X below line**: More cities flagged as having data issues
- **Scattered points**: More variation in POI coverage

### EDA Scatter Plots

**What They Show**:

```
POI Count
    |     o  o  o
    |   o  o  /o  <- Green: Q75 (strict)
    |  o  o//o
    | o  o/o  o
    |o  /o  o  <- Red: OLS (traditional)
    |o/  o
    |/o  o
    +-------------------> Population/Area
```

**Interpretation**:

- **Green above red**: Q75 sets higher expectations
- **Gap between lines**: Shows strictness difference
- Cities between the lines: Flagged in strict mode, not in legacy mode

## Plot Customization

### Adjust Quantile Level

To see fits at different quantiles:

```python
# More strict (90th percentile)
compute_confidence_scores(..., quantile=0.90)

# Less strict (60th percentile)
compute_confidence_scores(..., quantile=0.60)
```

The plots will reflect the new quantile in:

- The regression line position
- The legend label
- The title

### Adjust Flagging Threshold

To see more/fewer flagged cities:

```python
# More aggressive flagging
compute_confidence_scores(..., residual_threshold=-1.5)

# Less aggressive flagging
compute_confidence_scores(..., residual_threshold=-2.5)
```

The red X marks will update based on the new threshold.

## Technical Details

### Plot Generation

**When**: After regression models are fit, before final confidence scores computed
**Where**: `confidence_scoring.py` line ~145
**Format**: PNG, 150 DPI, 10x6 inches

### Dependencies

- `matplotlib`: Plotting library
- `sklearn.linear_model.QuantileRegressor`: For Q75 fits in EDA

### Performance

- Generates ~11 plots (one per category)
- Takes ~5-10 seconds additional runtime
- Total file size: ~5-10 MB

## Files Modified

1. `/src/analysis/modules/confidence_scoring.py`

   - Added matplotlib import
   - Added plotting loop after regression fitting
   - Creates `regression_plots/` directory

2. `/src/analysis/modules/generate_confidence_report.py`

   - Added "Regression Fit Visualizations" section
   - Embeds key plots with markdown image syntax

3. `/src/analysis/modules/explore_city_stats.py`
   - Added QuantileRegressor import
   - Updated scatter plot generation to include Q75 line
   - Both vs Population and vs Area plots updated

## Example Output Structure

```
output/
├── confidence_report.md         (now includes embedded plots)
├── city_confidence.gpkg
├── regression_diagnostics.csv
├── regression_plots/            (NEW)
│   ├── eat_and_drink_regression_fit.png
│   ├── retail_regression_fit.png
│   ├── education_regression_fit.png
│   ├── business_and_services_regression_fit.png
│   ├── active_life_regression_fit.png
│   ├── ... (all categories)
└── eda/
    ├── scatter_eat_and_drink.png  (updated with Q75 line)
    ├── scatter_retail.png         (updated with Q75 line)
    └── ...
```

## Benefits Summary

1. **Visual validation**: Quickly see if regression fits make sense
2. **Flagging transparency**: Understand which cities are flagged and why
3. **Comparison**: See OLS vs quantile regression differences
4. **Quality assessment**: R² scores visible in plot titles
5. **Documentation**: Plots embedded in markdown reports
6. **Debugging**: Easy to spot data quality issues or model problems
