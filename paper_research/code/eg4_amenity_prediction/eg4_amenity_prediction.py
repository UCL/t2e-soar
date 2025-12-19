# %% [markdown];
"""
# Amenity Supply Prediction Using Network Centrality

Trains Extra Trees models to predict POI distribution based on network centrality,
demonstrating that street network structure can predict amenity locations.

## Steps
1. Filter to cities with 'Consistently Saturated' POI coverage (from EG1)
2. Load node-level data (network centrality, POI counts, population)
3. Train Extra Trees models (eat & drink, business & services)
4. Generate predictions for all data
5. Compute per-city prediction accuracy (R², MAE, RMSE)
6. Export tables and visualizations
7. Generate README report

## Key Outputs
- **{category}_city_accuracy.csv**: Per-city R², MAE, RMSE scores
- **{category}_best_predicted_cities.csv**: Top 20 cities by R²
- **{category}_worst_predicted_cities.csv**: Bottom 20 cities by R²
- **{category}_feature_importance.png**: Centrality scale importance
- **{category}_city_r2_distribution.png**: Distribution of per-city R² scores
- **README.md**: Full analysis report with accuracy tables

## Interpretation
- High R² indicates network centrality strongly predicts POI locations
- Per-city accuracy shows how well this relationship generalizes
- Feature importance reveals which centrality scales matter most
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# %%
"""
## Configuration
"""

# Analysis parameters
# NOTE: Cent column names use templates {d} that get substituted with distances
DISTANCES = [400, 800, 1200, 1600, 4800, 9600]
CENT_COLUMNS = ["cc_beta_{d}", "cc_betweenness_beta_{d}"]
CENT_NAMES = ["Closeness", "Betweenness"]
CENSUS_COLUMNS = [
    "density",
    "y_lt15",
    "y_1564",
    "y_ge65",
    "emp",
]
CENSUS_LABELS = [
    "Population Density",
    "Population under 15",
    "Population 15-64",
    "Population 65 and over",
    "Employment Ratio",
]
POI_COLUMNS = ["cc_eat_and_drink_400_nw", "cc_business_and_services_400_nw"]
POI_CATEGORY_NAMES = ["Eat & Drink 400m", "Business & Services 400m"]
# Extract base category names for saturation file lookups (e.g., "eat_and_drink")
POI_CATEGORIES = [col.replace("cc_", "").replace("_400_nw", "") for col in POI_COLUMNS]

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.1
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 20,
    "min_samples_leaf": 50,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Configuration paths - modify these as needed
BOUNDS_PATH = "temp/datasets/boundaries.gpkg"
METRICS_DIR = "temp/cities_data/processed"
SATURATION_RESULTS_PATH = "paper_research/code/eg1_poi_compare/outputs/city_analysis_results.gpkg"
OUTPUT_DIR = "paper_research/code/eg4_amenity_prediction/outputs"
TEMP_DIR = "temp/egs/eg4_amenity_prediction"

# Saturation quadrants to include (reliable POI data)
SATURATED_QUADRANTS = ["Consistently Saturated"]

# %%
"""
## Setup Paths
"""

# Create output and temp directories
output_path = Path(OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)
temp_path = Path(TEMP_DIR)
temp_path.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Exploratory Question 4: Amenity Prediction")
print("=" * 80)
print(f"\nOutput directory: {output_path}")
print(f"Temp directory: {temp_path}")

# %%
"""
## Step 1: Load Saturation Results and Filter to Saturated Cities
"""

output_path = Path(OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)
metrics_dir = Path(METRICS_DIR)
temp_path = Path(TEMP_DIR)
temp_path.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("AMENITY PREDICTION")
print("=" * 80)

print("\nSTEP 1: Loading saturation results and filtering cities")

# Load saturation results from EG1
saturation_gdf = gpd.read_file(SATURATION_RESULTS_PATH)
print(f"  Loaded saturation results for {len(saturation_gdf)} cities")

# Filter to cities with saturated POI coverage for ALL categories
# (intersection of saturated cities across all categories)
saturated_by_category = {}
for i, cat in enumerate(POI_CATEGORIES):
    category_col = f"{cat}_quadrant"
    cat_name = POI_CATEGORY_NAMES[i]
    if category_col in saturation_gdf.columns:
        saturated_cities = saturation_gdf[saturation_gdf[category_col].isin(SATURATED_QUADRANTS)].copy()
        saturated_by_category[cat] = set(saturated_cities["bounds_fid"].tolist())
        print(f"\n  {cat_name}:")
        print(f"    Cities with saturated {cat_name} data: {len(saturated_cities)}")
        for quadrant in SATURATED_QUADRANTS:
            count = (saturated_cities[category_col] == quadrant).sum()
            print(f"      - {quadrant}: {count}")

# Find cities that are saturated for ALL categories (intersection)
if saturated_by_category:
    saturated_fids = set.intersection(*saturated_by_category.values())
    saturated_cities = saturation_gdf[saturation_gdf["bounds_fid"].isin(saturated_fids)].copy()
    print(f"\n  Cities saturated for ALL categories: {len(saturated_cities)}")
else:
    saturated_cities = pd.DataFrame()
    print("\n  WARNING: No saturated cities found")

# %%
"""
## Step 2: Load Node Data for Saturated Cities and Prepare Features
"""

print("\nSTEP 2: Loading node data and preparing features for modelling")

# Check for cached model data
model_data_cache_file = temp_path / "model_data_by_category.parquet"

if model_data_cache_file.exists():
    print("  Loading cached model data...")
    cached_df = pd.read_parquet(model_data_cache_file)
    model_data = {}
    for cat_name in POI_CATEGORY_NAMES:
        cat_data = cached_df[cached_df["category"] == cat_name].drop(columns=["category"])

        # Get all feature columns (centrality + census)
        feature_cols = [
            c
            for c in cat_data.columns
            if c.startswith("cc_beta") or c.startswith("cc_betweenness") or c in CENSUS_COLUMNS
        ]

        # Reconstruct X and y
        X = cat_data[feature_cols].copy()
        y = cat_data["y"].values
        cities = cat_data["city_name"].values

        model_data[cat_name] = {"X": X.reset_index(drop=True), "y": y, "cities": pd.Series(cities)}

    print(f"  Loaded cached model data for {len(model_data)} categories")
else:
    print("  Loading individual city metrics files...")

    # Generate centrality column names by substituting distances into templates
    centrality_cols = [col_template.replace("{d}", str(d)) for col_template in CENT_COLUMNS for d in DISTANCES]

    # Generate POI column names for 400m distance
    poi_cols = []
    poi_col_to_category = {}
    for col_template, cat_name in zip(POI_COLUMNS, POI_CATEGORY_NAMES):
        col_name = col_template.replace("{d}", "400")
        poi_cols.append(col_name)
        poi_col_to_category[col_name] = cat_name

    # Load all columns at once for each city
    all_cols_to_load = centrality_cols + poi_cols + CENSUS_COLUMNS
    all_nodes = []

    for idx, row in saturated_cities.iterrows():
        bounds_fid = row["bounds_fid"]
        city_label = row.get("label", str(bounds_fid))

        metrics_file = metrics_dir / f"metrics_{bounds_fid}.gpkg"
        if not metrics_file.exists():
            continue

        try:
            gdf = gpd.read_file(metrics_file, columns=all_cols_to_load, layer="streets")

            # Filter out invalid values
            for col in all_cols_to_load:
                gdf = gdf[gdf[col].notna() & np.isfinite(gdf[col])]

            if len(gdf) > 0:
                gdf["city_name"] = city_label
                all_nodes.append(gdf)
        except Exception:
            pass

    if not all_nodes:
        raise ValueError("No valid data loaded for any city")

    # Concatenate all cities
    df_all = pd.concat(all_nodes, ignore_index=True)
    print(f"  Loaded {len(df_all):,} nodes from {len(all_nodes)} cities")

    # Prepare model data for each category
    model_data = {}
    cache_records = []

    for poi_col, cat_name in zip(poi_cols, POI_CATEGORY_NAMES):
        print(f"\n  Category: {cat_name}")

        # Features: centrality + census (log-transformed)
        X = df_all[centrality_cols + CENSUS_COLUMNS].copy()
        X = np.log1p(X)
        X = X.reset_index(drop=True)

        # Target: POI counts (log-transformed)
        y = np.log1p(df_all[poi_col].values)

        # City labels
        cities = df_all["city_name"].reset_index(drop=True)

        model_data[cat_name] = {"X": X, "y": y, "cities": cities}

        print(f"    Features shape: {X.shape}")
        print(
            f"    Features: {len(centrality_cols)} centrality + {len(CENSUS_COLUMNS)} census = {len(centrality_cols) + len(CENSUS_COLUMNS)}"
        )
        print(f"    Target range: {y.min():.2f} to {y.max():.2f} (log-space)")

        # Cache the data
        for idx in range(len(df_all)):
            record = {"category": cat_name, "city_name": cities.iloc[idx], "y": y[idx]}
            for col in centrality_cols + CENSUS_COLUMNS:
                record[col] = X.iloc[idx][col]
            cache_records.append(record)

    # Save to cache
    if cache_records:
        cache_df = pd.DataFrame(cache_records)
        cache_df.to_parquet(model_data_cache_file)
        print(f"\n  Saved model data cache to {model_data_cache_file}")

# %%
"""
## Step 3: Train Extra Trees Models
"""

print("\nSTEP 3: Training Extra Trees models...")

models = {}

for cat_name in POI_CATEGORY_NAMES:
    print(f"\n  Category: {cat_name}")

    X = model_data[cat_name]["X"]
    y = model_data[cat_name]["y"]
    cities = model_data[cat_name]["cities"]

    # Train-test split (stratified by city)
    city_counts = cities.value_counts()
    city_bins = pd.qcut(city_counts, q=5, labels=False, duplicates="drop")
    city_bin_map = city_bins.to_dict()
    stratify_labels = cities.map(city_bin_map)

    X_train, X_test, y_train, y_test, cities_train, cities_test = train_test_split(
        X, y, cities, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify_labels
    )

    print(f"    Training set: {len(X_train):,} nodes from {cities_train.nunique()} cities")
    print(f"    Test set: {len(X_test):,} nodes from {cities_test.nunique()} cities")

    # Train model
    print("    Training Extra Trees...")
    et = ExtraTreesRegressor(**RF_PARAMS)
    et.fit(X_train, y_train)

    # Evaluate
    y_pred_train = et.predict(X_train)
    y_pred_test = et.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"    R² (train): {r2_train:.3f}")
    print(f"    R² (test): {r2_test:.3f}")
    print(f"    MAE (test): {mae_test:.3f}")
    print(f"    RMSE (test): {rmse_test:.3f}")

    # Feature importance
    feature_names = X.columns.tolist()
    importance = pd.DataFrame({"feature": feature_names, "importance": et.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    print("    Top features:")
    for _, row in importance.iterrows():
        print(f"      {row['feature']}: {row['importance']:.3f}")

    models[cat_name] = {
        "model": et,
        "r2_train": r2_train,
        "r2_test": r2_test,
        "mae_test": mae_test,
        "rmse_test": rmse_test,
        "feature_importance": importance,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred_test": y_pred_test,
    }

# %%
"""
## Step 4: Generate Predictions for All Data
"""

print("\nSTEP 4: Generating predictions...")

prediction_results = {}

for cat_name in POI_CATEGORY_NAMES:
    print(f"\n  Category: {cat_name}")

    X = model_data[cat_name]["X"]
    y = model_data[cat_name]["y"]
    cities = model_data[cat_name]["cities"]
    model = models[cat_name]["model"]

    # Predict for all nodes
    y_pred = model.predict(X)

    # Create results dataframe
    df_result = pd.DataFrame(
        {
            "city_name": cities,
            "y_true": y,
            "y_pred": y_pred,
        }
    )

    # Overall prediction accuracy
    overall_r2 = r2_score(y, y_pred)
    print(f"    Overall R² (all data): {overall_r2:.3f}")

    prediction_results[cat_name] = df_result

# %%
"""
## Step 5: Compute Per-City Prediction Accuracy
"""

print("\nSTEP 5: Computing per-city prediction accuracy...")

city_results = {}

for cat_name in POI_CATEGORY_NAMES:
    print(f"\n  Category: {cat_name}")

    df = prediction_results[cat_name]

    # Compute R² for each city
    city_accuracy = []
    for city_name, city_df in df.groupby("city_name"):
        y_true = city_df["y_true"].values
        y_pred = city_df["y_pred"].values
        n_nodes = len(city_df)

        # Need at least 2 samples and some variance for R²
        if n_nodes >= 10 and np.std(y_true) > 0:
            city_r2 = r2_score(y_true, y_pred)
            city_mae = mean_absolute_error(y_true, y_pred)
            city_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        else:
            city_r2 = np.nan
            city_mae = np.nan
            city_rmse = np.nan

        city_accuracy.append(
            {
                "city_name": city_name,
                "r2": city_r2,
                "mae": city_mae,
                "rmse": city_rmse,
                "n_nodes": n_nodes,
            }
        )

    city_agg = pd.DataFrame(city_accuracy)
    city_agg = city_agg.dropna(subset=["r2"])  # Remove cities with insufficient data
    city_agg = city_agg.sort_values("r2", ascending=False)

    print(f"    Cities with valid R²: {len(city_agg)}")
    print(f"    R² range: {city_agg['r2'].min():.3f} to {city_agg['r2'].max():.3f}")
    print(f"    Median R²: {city_agg['r2'].median():.3f}")
    print(f"    Mean R²: {city_agg['r2'].mean():.3f}")

    city_results[cat_name] = city_agg

# %%
"""
## Step 6: Export Results and Visualizations
"""

print("\nSTEP 6: Exporting results...")

for cat_name in POI_CATEGORY_NAMES:
    print(f"\n  Category: {cat_name}")

    city_agg = city_results[cat_name]

    # Export top 20 best predicted cities (highest R²)
    best_predicted = city_agg.head(20)
    out_file = output_path / f"{cat_name}_best_predicted_cities.csv"
    best_predicted.to_csv(out_file, index=False)
    print(f"    Exported best predicted cities: {out_file}")

    # Export top 20 worst predicted cities (lowest R²)
    worst_predicted = city_agg.tail(20).iloc[::-1]  # Reverse to show worst first
    out_file = output_path / f"{cat_name}_worst_predicted_cities.csv"
    worst_predicted.to_csv(out_file, index=False)
    print(f"    Exported worst predicted cities: {out_file}")

    # Export full city results
    out_file = output_path / f"{cat_name}_city_accuracy.csv"
    city_agg.to_csv(out_file, index=False)
    print(f"    Exported all city accuracies: {out_file}")

    # Feature importance plot
    fig, ax = plt.subplots(figsize=(10, 6))
    importance = models[cat_name]["feature_importance"].head(10)
    ax.barh(range(len(importance)), importance["importance"])
    ax.set_yticks(range(len(importance)))
    ax.set_yticklabels(importance["feature"])
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Feature Importance: {cat_name.replace('_', ' ').title()}")
    ax.invert_yaxis()
    plt.tight_layout()
    out_file = output_path / f"{cat_name}_feature_importance.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved feature importance plot: {out_file}")

    # Per-city R² distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(city_agg["r2"], bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(
        city_agg["r2"].median(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Median R²: {city_agg['r2'].median():.3f}",
    )
    ax.set_xlabel("R² Score")
    ax.set_ylabel("Number of Cities")
    ax.set_title(f"Distribution of Per-City R²: {cat_name.replace('_', ' ').title()}")
    ax.legend()
    plt.tight_layout()
    out_file = output_path / f"{cat_name}_city_r2_distribution.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved city R² distribution: {out_file}")

# %%
"""
## Step 7: Generate Markdown Report
"""

print("\nSTEP 7: Generating markdown report...")

# Collect summary statistics
total_nodes = len(prediction_results[POI_CATEGORY_NAMES[0]])
total_cities = len(city_results[POI_CATEGORY_NAMES[0]])

report_lines = [
    "# Amenity Supply Prediction Using Network Centrality",
    "",
    f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}",
    "",
    "## Overview",
    "",
    "This analysis uses Extra Trees regression to predict amenity supply based on",
    "multi-scale network centrality metrics and census variables. The goal is to demonstrate that street",
    "network structure can predict POI distributions, with per-city accuracy metrics",
    "showing how well this relationship generalizes across different urban contexts.",
    "",
    "## Summary Statistics",
    "",
    f"- **Total Street Network Nodes Analyzed:** {total_nodes:,}",
    f"- **Cities Analyzed:** {total_cities}",
    f"- **POI Categories:** {len(POI_CATEGORY_NAMES)}",
    f"- **Network Centrality Scales:** {len(DISTANCES)} ({DISTANCES[0]}–{DISTANCES[-1]}m)",
    f"- **Census Features:** {', '.join(CENSUS_LABELS)}",
    "",
    "## Overall Model Performance",
    "",
    "| Category | R² (train) | R² (test) | MAE (test) | RMSE (test) |",
    "|----------|-----------|----------|-----------|------------|",
]

for cat_name in POI_CATEGORY_NAMES:
    model_info = models[cat_name]
    report_lines.append(
        f"| {cat_name.replace('_', ' ').title()} | {model_info['r2_train']:.3f} | {model_info['r2_test']:.3f} | {model_info['mae_test']:.3f} | {model_info['rmse_test']:.3f} |"
    )

report_lines.extend(
    [
        "",
        "## Per-City Prediction Accuracy",
        "",
        "R² scores computed separately for each city show how well the model",
        "generalizes across different urban contexts.",
        "",
    ]
)

for cat_name in POI_CATEGORY_NAMES:
    city_agg = city_results[cat_name]
    report_lines.append(f"### {cat_name.replace('_', ' ').title()}")
    report_lines.append("")
    report_lines.append(f"- **Median City R²:** {city_agg['r2'].median():.3f}")
    report_lines.append(f"- **Mean City R²:** {city_agg['r2'].mean():.3f}")
    report_lines.append(f"- **R² Range:** {city_agg['r2'].min():.3f} to {city_agg['r2'].max():.3f}")
    report_lines.append(
        f"- **Cities with R² > 0.5:** {(city_agg['r2'] > 0.5).sum()} ({(city_agg['r2'] > 0.5).mean() * 100:.1f}%)"
    )
    report_lines.append("")
    report_lines.append("#### Top 10 Best Predicted Cities")
    report_lines.append("")
    report_lines.append("| City | R² | MAE | RMSE | N Nodes |")
    report_lines.append("|------|-----|-----|------|---------|")

    for _, row in city_agg.head(10).iterrows():
        report_lines.append(
            f"| {row['city_name']} | {row['r2']:.3f} | {row['mae']:.3f} | {row['rmse']:.3f} | {int(row['n_nodes']):,} |"
        )

    report_lines.append("")
    report_lines.append("#### Top 10 Worst Predicted Cities")
    report_lines.append("")
    report_lines.append("| City | R² | MAE | RMSE | N Nodes |")
    report_lines.append("|------|-----|-----|------|---------|")

    for _, row in city_agg.tail(10).iloc[::-1].iterrows():
        report_lines.append(
            f"| {row['city_name']} | {row['r2']:.3f} | {row['mae']:.3f} | {row['rmse']:.3f} | {int(row['n_nodes']):,} |"
        )

    report_lines.append("")
    # Add R² distribution plot
    report_lines.append("#### R² Distribution")
    report_lines.append("")
    # URL-encode spaces in filename
    cat_name_encoded = cat_name.replace(" ", "%20")
    report_lines.append(f"![R² Distribution](outputs/{cat_name_encoded}_city_r2_distribution.png)")
    report_lines.append("")

report_lines.extend(
    [
        "## Top Features by Category",
        "",
    ]
)

for cat_name in POI_CATEGORY_NAMES:
    report_lines.append(f"### {cat_name.replace('_', ' ').title()}")
    report_lines.append("")
    report_lines.append("| Rank | Feature | Importance |")
    report_lines.append("|------|---------|------------|")

    importance = models[cat_name]["feature_importance"].head(10)
    for rank, (_, row) in enumerate(importance.iterrows(), 1):
        report_lines.append(f"| {rank} | {row['feature']} | {row['importance']:.4f} |")

    report_lines.append("")
    # Add feature importance plot
    cat_name_encoded = cat_name.replace(" ", "%20")
    report_lines.append(f"![Feature Importance](outputs/{cat_name_encoded}_feature_importance.png)")
    report_lines.append("")

report_lines.extend(
    [
        "## Methods",
        "",
        "### Feature Engineering",
        f"- Closeness centrality at {len(DISTANCES)} scales ({DISTANCES[0]}–{DISTANCES[-1]}m)",
        f"- Betweenness centrality at {len(DISTANCES)} scales ({DISTANCES[0]}–{DISTANCES[-1]}m)",
        "- Census features:",
    ]
)
for label in CENSUS_LABELS:
    report_lines.append(f"  - {label}")
report_lines.extend(
    [
        "- Log-transformation of all features to stabilize variance",
        "",
        "### Model Training",
        f"- Algorithm: Extra Trees ({RF_PARAMS['n_estimators']} estimators)",
        f"- Max depth: {RF_PARAMS['max_depth']}",
        f"- Test size: {TEST_SIZE * 100:.0f}%",
        "- Stratified splitting by city to ensure representative train/test sets",
        "",
        "## Output Files",
        "",
    ]
)

for cat_name in POI_CATEGORY_NAMES:
    report_lines.extend(
        [
            f"### {cat_name.replace('_', ' ').title()}",
            "",
            f"- `{cat_name}_city_accuracy.csv`: Per-city R², MAE, RMSE",
            f"- `{cat_name}_best_predicted_cities.csv`: Top 20 cities by R²",
            f"- `{cat_name}_worst_predicted_cities.csv`: Bottom 20 cities by R²",
            f"- `{cat_name}_feature_importance.png`: Feature importance plot",
            f"- `{cat_name}_city_r2_distribution.png`: Distribution of city R² scores",
            "",
        ]
    )

# Write markdown report
report_text = "\n".join(report_lines)
report_path = output_path.parent / "README.md"
with open(report_path, "w") as f:
    f.write(report_text)

print(f"✓ Markdown report saved: {report_path}")

# Also save plain text summary for console review
summary_lines = [
    "=" * 80,
    "EXPLORATORY QUESTION 4: AMENITY SUPPLY PREDICTION",
    "Summary Report",
    "=" * 80,
    "",
]

for cat_name in POI_CATEGORY_NAMES:
    summary_lines.append(f"\n{cat_name.replace('_', ' ').upper()}")
    summary_lines.append("-" * 80)
    summary_lines.append(f"Overall R² (test): {models[cat_name]['r2_test']:.3f}")
    summary_lines.append(f"Overall RMSE (test): {models[cat_name]['rmse_test']:.3f}")

    city_agg = city_results[cat_name]
    summary_lines.append("\nPer-City Accuracy:")
    summary_lines.append(f"  Median R²: {city_agg['r2'].median():.3f}")
    summary_lines.append(f"  Mean R²: {city_agg['r2'].mean():.3f}")
    summary_lines.append(f"  Cities with R² > 0.5: {(city_agg['r2'] > 0.5).sum()}/{len(city_agg)}")

    summary_lines.append("\nTop 3 Features:")
    for i, (_, row) in enumerate(models[cat_name]["feature_importance"].head(3).iterrows(), 1):
        summary_lines.append(f"  {i}. {row['feature']}: {row['importance']:.4f}")

    summary_lines.append("\nBest Predicted Cities:")
    for _, row in city_agg.head(3).iterrows():
        summary_lines.append(f"  {row['city_name']}: R²={row['r2']:.3f}")

    summary_lines.append("\nWorst Predicted Cities:")
    for _, row in city_agg.tail(3).iloc[::-1].iterrows():
        summary_lines.append(f"  {row['city_name']}: R²={row['r2']:.3f}")

summary_lines.extend(["", "=" * 80])

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)

# %%
"""
## Analysis Complete
"""

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)
print(f"\nOutputs saved to: {output_path}")
print("\nKey files:")
for cat_name in POI_CATEGORY_NAMES:
    print(f"  - {cat_name}_city_accuracy.csv")
    print(f"  - {cat_name}_best_predicted_cities.csv")
    print(f"  - {cat_name}_feature_importance.png")
    print(f"  - {cat_name}_city_r2_distribution.png")
