# ebdp-lightweight

Lightweight modelling workflow for EBDP TWIN2EXPAND project for evidence based approaches to urban design and planning.

## Development

Project configuration is managed using a `pyproject.toml` file. For development purposes: `uv` is used for installation and management of the packages and related upgrades. For example: `uv sync` will install packages listed in the `pyproject.toml` file and creates a self-contained development environment in a `.venv` folder.

## Data Loading

See the [data_loading.md](data_loading.md) markdown file for data loading guidelines.

## Licenses

This repo depends on copy-left open source packages licensed as AGPLv3 and therefore adopts the same license. This is also in keeping with the intention of the TWIN2EXPAND project to create openly reproducible workflows.

The Overture Maps data source is licensed [Community Data License Agreement – Permissive, Version 2.0](https://cdla.dev) with some layers licensed as [Open Data Commons Open Database License](https://opendatacommons.org/licenses/odbl/). OpenStreetMap data is [© OpenStreetMap contributors](https://osmfoundation.org/wiki/Licence/Attribution_Guidelines#Attribution_text)

## Loading Notes

The data source is a combination of EU Copernicus data and [Overture Maps](https://overturemaps.org), which largely resembles [OpenStreetMap](https://www.openstreetmap.org). Overture intends to provide a higher degree of data verification and issues fixed releases.

### Boundaries

Boundaries are extracted from the [2021 Urban Centres / High Density Clusters](https://ec.europa.eu/eurostat/web/gisco/geodata/population-distribution/clusters) dataset. This is 1x1km raster with high density clusters [described as](https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Territorial_typologies#Typologies) contiguous 1km2 cells with at least 1,500 residents per km2 and consisting of cumulative urban clusters with at least 50,000 people.

Download the dataset from the above link. Then run the `generate_boundary_polys.py` script to generate the vector boundaries from the raster source. Provide the input datapath to the TIFF file and the output file path for the generated vector boundaries in GPKG format. The generated GPKG will contain three layers named `bounds`, `unioned_bounds_2000`, and `unioned_bounds_10000`. This script will automatically remove boundaries intersecting the UK.

Example:

```bash
python -m src.data.generate_boundary_polys temp/HDENS-CLST-2021/HDENS_CLST_2021.tif temp/datasets/boundaries.gpkg
```

### Urban Atlas

[urban atlas](https://land.copernicus.eu/local/urban-atlas/urban-atlas-2018) (~37GB vectors)

Run the `load_urban_atlas_blocks.py` script to generate the blocks data. Provide the path to the boundaries GPKG generated previously, as well as the downloaded Urban Atlas data and an output path for the generated blocks GPKG.

Example:

```bash
python -m src.data.load_urban_atlas_blocks \
        temp/datasets/boundaries.gpkg \
            temp/UA_2018_3035_eu \
                temp/datasets/blocks.gpkg
```

### Tree cover

[Tree cover](https://land.copernicus.eu/local/urban-atlas/street-tree-layer-stl-2018) (~36GB vectors).

Run the `load_urban_atlas_trees.py` script to generate the tree cover data. Provide the path to the boundaries GPKG generated previously, as well as the downloaded STL data and an output path for the generated tree cover GPKG.

Example:

```bash
python -m src.data.load_urban_atlas_trees \
    temp/datasets/boundaries.gpkg \
        temp/STL_2018_3035_eu \
            temp/datasets/tree_canopies.gpkg
```

### Building Heights

[Digital Height Model](https://land.copernicus.eu/local/urban-atlas/building-height-2012) (~ 1GB raster).

Run the `load_bldg_hts_raster.py` script to generate the building heights data. Provide the path to the boundaries GPKG generated previously, as well as the downloaded building height data and an output folder path for the extracted building heights TIFF files.

Example:

```bash
python -m src.data.load_bldg_hts_raster \
    temp/datasets/boundaries.gpkg \
        temp/Results-Building_Height_2012_3035_eu \
            temp/cities_data/heights
```

### Ingesting Overture data

Run the `load_overture.py` script to download and prepare the overture data. The script will download the relevant Overture GPKG files for each boundary, clip them to the boundary, and save them to the output directory. Provide the path to the boundaries GPKG generated previously, as well as an output directory for the clipped Overture data. Optionally, you can specify the number of parallel workers to speed up the processing. By default, it uses 2 workers. Pass an additional argument `--overwrite` to redo processing for boundaries that already have corresponding Overture data in the output directory. Otherwise, existing data will be skipped. Each boundary will be saved as a separate GPKG file named with the boundary ID, containing layers for `buildings`, street `edges`, street `nodes`, a cleaned version of street edges `clean_edges`, POI `places`, and `infrastructure`.

```bash
python -m src.data.load_overture \
    temp/datasets/boundaries.gpkg \
        temp/cities_data/overture \
            --parallel_workers 6
```

docs/schema/concepts/by-theme/places/overture_categories.csv

> The Overture POI schema is based on [`overture_categories.csv`](https://github.com/OvertureMaps/schema/blob/dev/docs/schema/concepts/by-theme/places/overture_categories.csv).

### Census Data (2021)

GeoStat Census data for 2021 is [downloaded from](https://ec.europa.eu/eurostat/web/gisco/geodata/population-distribution/population-grids). These census statistics are aggregated to 1km2 cells.

Download the census ZIP dataset for Version 2021 (22 January 2025).

### Metrics

Compute metrics using the `generate_metrics.py` script. Provide the path to the boundaries GPKG, the directory containing the processed Overture data, the blocks GPKG, the tree canopies GPKG, the census GPKG, and an output directory for the generated metrics GPKG files.

```bash
python -m src.processing.generate_metrics \
    temp/datasets/boundaries.gpkg \
        temp/cities_data/overture \
            temp/datasets/blocks.gpkg \
                temp/datasets/tree_canopies.gpkg \
                    temp/cities_data/heights \
                        temp/Eurostat_Census-GRID_2021_V2/ESTAT_Census_2021_V2.gpkg \
                            temp/cities_data/processed
```

## Data Quality Analysis

After computing metrics, you can assess POI (Point of Interest) data saturation across all cities using a grid-based multi-scale regression approach. This analysis compares each city's POI density against population-based expectations, identifying cities that are undersaturated (fewer POIs than expected) or saturated (at or above expected levels).

### POI Saturation Assessment Workflow

Open `src/analysis/poi_saturation_notebook.py` in VS Code and run all cells sequentially. The notebook uses `# %%` cell markers for interactive execution.

**Configuration**: Modify the paths in the configuration cell:

- `BOUNDS_PATH` - Path to boundaries GPKG
- `OVERTURE_DATA_DIR` - Directory with Overture data
- `CENSUS_PATH` - Path to census GPKG
- `OUTPUT_DIR` - Where to save results

This workflow performs seven steps:

1. **Grid-level aggregation**: Counts POIs within 1km² census grid cells and computes multi-scale population neighborhoods (local, intermediate, large radii)
2. **Random Forest regression**: For each POI category, fits a model in log-space: `log(POI_count) ~ log(pop_local) + log(pop_intermediate) + log(pop_large)`. Log transformation linearizes power-law relationships between population and POI counts.
3. **Z-score computation**: Standardized residuals identify grid cells with more/fewer POIs than expected given their multi-scale population context
4. **City-level aggregation**: Aggregates grid z-scores per city, computing mean (saturation level) and standard deviation (spatial variability)
5. **Quadrant classification**: Cities are classified into four quadrants based on mean z-score × variability:
   - **Consistently Undersaturated**: Low POI coverage, uniform distribution (potential data gap)
   - **Variable Undersaturated**: Low coverage, high spatial variability
   - **Consistently Saturated**: Expected or above POI coverage, uniform distribution
   - **Variable Saturated**: High coverage, high spatial variability
6. **Feature importance analysis**: Compares which population scale (local, intermediate, large) best predicts each POI category
7. **Report generation**: Creates comprehensive markdown report with quadrant classifications and visualizations

### Output Files

The analysis generates:

- `grid_stats.gpkg` - Grid-level POI counts with multi-scale population neighborhoods
- `grid_counts_regress.gpkg` - Grid cells with z-scores and predicted values per category
- `city_analysis_results.gpkg` - City-level z-score statistics and quadrant classifications
- `city_assessment_report.md` - Comprehensive analysis report

**Visualizations**:

- `eda_analysis.png` - Model fit (R²) and z-score distributions by category
- `regression_diagnostics.png` - Predicted vs observed POI counts per category
- `feature_importance.png` - Population scale importance for each POI type
- `city_quadrant_analysis.png` - 12-panel visualization showing per-category and between-category quadrant classification

**Note**: All GeoPackage outputs can be opened in QGIS for spatial visualization and exploration.

### Interpreting Results

Z-scores represent continuous deviations from expected POI counts:

- **z < 0**: Fewer POIs than expected (undersaturated)
- **z > 0**: More POIs than expected (saturated)

The quadrant analysis combines:

- **Mean z-score**: Overall saturation level (negative = undersaturated, positive = saturated)
- **Std z-score**: Spatial variability (low = consistent across grids, high = variable)

**Recommended interpretation**:

- **Consistently Undersaturated cities**: May indicate data quality issues; use with caution
- **Variable Undersaturated cities**: Partial coverage; some areas may be reliable
- **Consistently/Variable Saturated cities**: Suitable for most analyses
