""" """

import argparse
from pathlib import Path

import geopandas as gpd
from tqdm import tqdm

from src import tools
from src.data import loaders

logger = tools.get_logger(__name__)


OVERTURE_SCHEMA = tools.generate_overture_schema()


def load_overture_places(bounds_in_path: str, cities_data_out_dir: str) -> None:
    """ """
    tools.validate_filepath(bounds_in_path)
    tools.validate_directory(cities_data_out_dir, create=True)
    logger.info("Loading overture places")
    bounds_gdf = gpd.read_file(bounds_in_path, layer="bounds")
    bounds_gdf = bounds_gdf.to_crs(3035)
    bounds_gdf.geometry = bounds_gdf.geometry.buffer(2000)
    bounds_gdf = bounds_gdf.to_crs(4326)
    # loop through bounds and load places
    for bounds_fid, bounds_row in tqdm(bounds_gdf.iterrows(), total=len(bounds_gdf)):
        output_path = Path(cities_data_out_dir) / f"overture_{bounds_fid}.gpkg"
        places_gdf = loaders.load_places(bounds_row.geometry, 3035)
        places_gdf["bounds_fid"] = bounds_fid
        places_gdf.to_file(output_path, driver="GPKG", layer="places", overwrite=True)


if __name__ == "__main__":
    """
    python -m src.data.ingest_overture_places temp/datasets/boundaries.gpkg temp/cities_data
    """
    if True:
        parser = argparse.ArgumentParser(description="Load overture places.")
        parser.add_argument("bounds_in_path", type=str, help="Input data directory with boundary GPKG.")
        parser.add_argument("cities_data_out_dir", type=str, help="Output data directory for city GPKG files.")
        args = parser.parse_args()
        load_overture_places(
            bounds_in_path=args.bounds_in_path,
            cities_data_out_dir=args.cities_data_out_dir,
        )
    else:
        load_overture_places(
            bounds_in_path="temp/datasets/boundaries.gpkg",
            cities_data_out_dir="temp/cities_data",
        )
