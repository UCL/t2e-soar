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

After computing metrics, you can assess the quality and completeness of POI (Point of Interest) data across all cities using regression-based confidence scoring. This analysis identifies cities with unexpectedly low land-use counts that likely indicate data quality issues rather than genuine urban characteristics.

### Confidence Scoring Workflow

Open `src/analysis/analysis_notebook.py` in VS Code and run all cells sequentially. The notebook uses `# %%` cell markers for interactive execution.

**Configuration**: Modify the paths in the second cell:

- `BOUNDS_PATH` - Path to boundaries GPKG
- `OVERTURE_DATA_DIR` - Directory with Overture data
- `CENSUS_PATH` - Path to census GPKG
- `OUTPUT_DIR` - Where to save results

This workflow performs four steps:

1. **City-level aggregation**: Aggregates population, area, and POI counts by land-use category for each city boundary
2. **Exploratory data analysis**: Generates descriptive statistics, scatter plots, correlation matrices, and distribution histograms
3. **Confidence scoring**: Fits regression models (`POI_count ~ population + area`) for each land-use category and computes standardized residuals to flag cities with unexpectedly low counts
4. **Report generation**: Creates a comprehensive markdown report ranking cities by data quality

### Output Files

The analysis generates:

- `city_stats.gpkg` - City-level aggregated statistics (population, area, POI counts) with city boundary geometries
- `city_confidence.gpkg` - Confidence scores, z-scores, and flagged categories per city with geometries
- `confidence_report.md` - Main analysis report with top/bottom 50 cities and recommendations
- `regression_diagnostics.csv` - Model fit statistics (R², coefficients) per land-use category
- `top_50_cities.csv` / `bottom_50_cities.csv` - Ranked city lists
- `eda/` - Exploratory data analysis visualizations and tables

**Note**: The main output files (`city_stats.gpkg` and `city_confidence.gpkg`) are GeoPackages that can be opened in QGIS for spatial visualization and exploration.

### Workflow Options

**Skip exploratory data analysis** (faster execution):

Set `SKIP_EDA = True` in the configuration cell (second cell) before running.

**Skip aggregation if already completed**:

The notebook automatically detects if `city_stats.gpkg` exists and skips re-aggregation. To force re-aggregation, delete the file first.

### Interpreting Results

The confidence score (0-1) is computed based on:

- **Number of flagged categories**: Cities with z-scores < -2.0 in multiple land-use categories
- **Severity of residuals**: How far below expected POI counts the city falls

**Recommended usage:**

- **High confidence (≥0.7)**: Suitable for all analyses
- **Moderate confidence (0.4-0.7)**: Use with caution for category-specific studies
- **Low confidence (<0.4)**: Consider excluding from analyses or treating as missing data
