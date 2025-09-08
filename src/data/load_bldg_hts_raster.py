""" """

import argparse
import os
import shutil
import zipfile
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.merge import merge
from shapely import geometry
from tqdm import tqdm

from src import tools

logger = tools.get_logger(__name__)


def load_bldg_hts(bounds_in_path: str, data_dir_path: str, cities_data_out_dir: str) -> None:
    """ """
    tools.validate_filepath(bounds_in_path)
    tools.validate_directory(data_dir_path)
    tools.validate_directory(cities_data_out_dir, create=True)
    # load bounds
    bounds_gdf = gpd.read_file(bounds_in_path, layer="bounds")
    bounds_gdf = bounds_gdf.to_crs(3035)
    bounds_gdf.geometry = bounds_gdf.geometry.buffer(2000)
    bounds_geom = bounds_gdf.union_all()
    # Loop through each ZIP file and upload TIFs
    logger.info("Loading building heights rasters")
    raster_tiles = []
    dir_path: Path = Path(data_dir_path)
    unzip_dir = dir_path / "temp_unzipped/"
    for zip_file_name in tqdm(os.listdir(dir_path)):
        if zip_file_name.endswith(".zip"):
            temp_unzip_dir = unzip_dir / zip_file_name.rstrip(".zip")
            os.makedirs(temp_unzip_dir, exist_ok=True)
            src_zip_path = dir_path / zip_file_name
            with zipfile.ZipFile(src_zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_unzip_dir)
            # Find the tif file
            for walk_dir_path, _dir_names, file_names in os.walk(temp_unzip_dir):
                for raster_file_name in file_names:
                    if raster_file_name.endswith(".tif"):
                        full_raster_path = str((Path(walk_dir_path) / raster_file_name).resolve())
                        # load and check itx
                        with rasterio.open(full_raster_path) as src:
                            if src.crs.to_epsg() != 3035:
                                logger.warning(f"Skipping raster with unexpected CRS: {src.crs}")
                                continue
                            if not geometry.box(*list(src.bounds)).intersects(bounds_geom):
                                continue
                            raster_tiles.append(full_raster_path)
    logger.info(f"Found {len(raster_tiles)} rasters intersecting bounds")

    logger.info("Merging and saving rasters per bounds")
    # iter bounds
    for bounds_fid, bounds_row in tqdm(bounds_gdf.iterrows(), total=len(bounds_gdf)):
        # filter rasters intersecting bounds
        intersecting_rasters = []
        for raster_fp in raster_tiles:
            with rasterio.open(raster_fp) as src:
                if geometry.box(*list(src.bounds)).intersects(bounds_row.geometry):
                    intersecting_rasters.append(raster_fp)
        if not intersecting_rasters:
            continue
        intersecting_rasters = [rasterio.open(fp) for fp in intersecting_rasters]
        mosaic, out_trans = merge(intersecting_rasters)
        out_meta = intersecting_rasters[0].meta.copy()
        out_meta.update(
            {"driver": "GTiff", "height": mosaic.shape[1], "width": mosaic.shape[2], "transform": out_trans}
        )
        out_path = Path(cities_data_out_dir) / f"bldg_hts_{bounds_fid}.tif"
        with rasterio.open(out_path, "w", **out_meta) as dest:
            dest.write(mosaic)
        # Close all opened files
        for src in intersecting_rasters:
            src.close()
    # Delete the unzipped files
    shutil.rmtree(unzip_dir)


if __name__ == "__main__":
    """
    python -m src.data.load_bldg_hts_raster temp/datasets/boundaries.gpkg temp/Results-Building_Height_2012_3035_eu temp/cities_data 
    """
    if True:
        parser = argparse.ArgumentParser(description="Load building heights raster data.")
        parser.add_argument("bounds_in_path", type=str, help="Input data directory with boundary GPKG.")
        parser.add_argument(
            "data_dir_path", type=str, help="Input data directory with zipped building heights rasters."
        )
        parser.add_argument("cities_data_out_dir", type=str, help="Output data directory for building heights TIFs.")
        args = parser.parse_args()
        logger.info(f"Loading building heights data from path: {args.data_dir_path}")
        load_bldg_hts(args.bounds_in_path, args.data_dir_path, args.cities_data_out_dir)
    else:
        load_bldg_hts(
            "temp/datasets/boundaries.gpkg",
            "temp/Results-Building_Height_2012_3035_eu",
            "temp/cities_data",
        )
