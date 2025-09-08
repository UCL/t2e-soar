# Loading Notes

The data source is a combination of EU Copernicus data and [Overture Maps](https://overturemaps.org), which largely resembles [OpenStreetMap](https://www.openstreetmap.org). Overture intends to provide a higher degree of data verification and uses fixed releases.

## Boundaries

Boundaries are extracted from the [2021 Urban Centres / High Density Clusters](https://ec.europa.eu/eurostat/web/gisco/geodata/population-distribution/clusters) dataset. This is 1x1km raster with high density clusters [described as](https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Territorial_typologies#Typologies) contiguous 1km2 cells with at least 1,500 residents per km2 and consisting of cumulative urban clusters with at least 50,000 people.

- Download the dataset from the above link.
- Run the `generate_boundary_polys.py` script to generate the vector boundaries from the raster source. Provide the source schema name and the table name of the high density cluster raster per above upload. The output polygons will be generated to the `eu` schema in tables named `bounds`, `unioned_bounds_2000`, and `unioned_bounds_10000`. This script will automatically remove boundaries intersecting the UK.

```bash
python -m src.data.generate_boundary_polys "temp/HDENS-CLST-2021/HDENS_CLST_2021.tif" "temp/datasets/boundaries.gpkg"
```

## Census Data (2021)

GeoStat Census data for 2021 is [downloaded from](https://ec.europa.eu/eurostat/web/gisco/geodata/population-distribution/population-grids). These census statistics are aggregated to 1km2 cells.

- Download the census ZIP dataset for Version 2021 (22 January 2025).

## Urban Atlas

[urban atlas](https://land.copernicus.eu/local/urban-atlas/urban-atlas-2018) (~37GB vectors)

- Run the `load_urban_atlas_blocks.py` script to upload the data. Provide the path to the input directory with the zipped data files. The blocks will be loaded to the `blocks` table in the `eu` schema.

```bash
python -m src.data.load_urban_atlas_blocks "./temp/urban atlas"
```

## Tree cover

[Tree cover](https://land.copernicus.eu/local/urban-atlas/street-tree-layer-stl-2018) (~36GB vectors).

- Run the `load_urban_atlas_trees.py` script to upload the data. Provide the path to the input directory with the zipped data files. The trees will be loaded to the `trees` table in the `eu` schema.

```bash
python -m src.data.load_urban_atlas_trees "./temp/urban atlas trees"
```

## Ingesting Overture data

Upload overture data. Pass the `--drop` flag to drop and therefore replace existing tables. The loading scripts will otherwise track which boundary extents are loaded and will resume if interrupted. The tables will be uploaded to the `overture` schema.

Places:

```bash
python -m src.data.ingest_overture_places
```

Places:

```bash
python -m src.data.ingest_overture_infrast
```

Buildings:

```bash
python -m src.data.ingest_overture_buildings
```

Network (cleaned) - in this case there is an optional parallel workers argument:

```bash
python -m src.data.ingest_overture_networks all --parallel_workers 4
```

### Building Heights

[Digital Height Model](https://land.copernicus.eu/local/urban-atlas/building-height-2012) (~ 1GB raster).

- Run the `load_bldg_hts_raster.py` script to upload the building heights data. Provide the path to the input directory with the zipped data files. Use the optional argument `--bin_path` to provide a path to the `bin` directory for your `postgres` installation. The raster will be loaded to the `bldg_hts` table in the `eu` schema.

```bash
python -m src.data.load_bldg_hts_raster "./temp/Digital height Model EU" --bin_path /Applications/Postgres.app/Contents/Versions/15/bin/
```

## Metrics

Once the datasets are uploaded, boundaries extracted, and networks prepared, it becomes possible to compute the metrics.

`python -m src.processing.generate_metrics all`
