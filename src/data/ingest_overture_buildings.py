""" """

import argparse
from pathlib import Path

import geopandas as gpd
from tqdm import tqdm

from src import tools
from src.data import loaders

logger = tools.get_logger(__name__)


def load_overture_buildings(bounds_in_path: str, cities_data_out_dir: str, overwrite: bool = False) -> None:
    """ """
    tools.validate_filepath(bounds_in_path)
    tools.validate_directory(cities_data_out_dir, create=True)
    logger.info("Loading overture buildings")
    bounds_gdf = gpd.read_file(bounds_in_path, layer="bounds")
    bounds_gdf = bounds_gdf.to_crs(3035)
    bounds_gdf.geometry = bounds_gdf.geometry.buffer(2000)
    bounds_gdf = bounds_gdf.to_crs(4326)
    # loop through bounds and load buildings
    for bounds_fid, bounds_row in tqdm(bounds_gdf.iterrows(), total=len(bounds_gdf)):
        output_path = Path(cities_data_out_dir) / f"overture_{bounds_fid}.gpkg"
        if not overwrite and output_path.exists():
            logger.info(f"Skipping existing file at path: {output_path}")
            continue
        buildings_gdf = loaders.load_buildings(bounds_row.geometry, 3035)
        buildings_gdf["bounds_fid"] = bounds_fid
        buildings_gdf.to_file(output_path, driver="GPKG", layer="buildings", overwrite=True)


if __name__ == "__main__":
    """
    python -m src.data.ingest_overture_buildings temp/datasets/boundaries.gpkg temp/cities_data False
    """
    if True:
        parser = argparse.ArgumentParser(description="Load overture buildings.")
        parser.add_argument("bounds_in_path", type=str, help="Input data directory with boundary GPKG.")
        parser.add_argument("cities_data_out_dir", type=str, help="Output data directory for city GPKG files.")
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Overwrite existing files (default: False)",
            default=False,
        )
        args = parser.parse_args()
        load_overture_buildings(
            bounds_in_path=args.bounds_in_path,
            cities_data_out_dir=args.cities_data_out_dir,
            overwrite=args.overwrite,
        )
    else:
        load_overture_buildings(
            bounds_in_path="temp/datasets/boundaries.gpkg",
            cities_data_out_dir="temp/cities_data",
            overwrite=False,
        )
