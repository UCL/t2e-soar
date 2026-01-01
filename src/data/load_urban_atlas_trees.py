""" """

import argparse
import os
import shutil
import warnings
import zipfile
from pathlib import Path

import fiona
import geopandas as gpd
import pandas as pd
from shapely import geometry
from tqdm import tqdm

from src import tools

warnings.filterwarnings("ignore", message="More than one layer found", category=UserWarning)

logger = tools.get_logger(__name__)


def load_tree_canopies(bounds_in_path: str, data_dir_path: str, trees_out_path: str) -> None:
    """ """
    logger.info(f"Loading urban atlas blocks data from path: {bounds_in_path}")
    tools.validate_filepath(bounds_in_path)
    tools.validate_directory(data_dir_path)
    tools.validate_directory(trees_out_path, create=True)
    # load bounds
    bounds_gdf = gpd.read_file(bounds_in_path, layer="bounds")
    bounds_gdf.geometry = bounds_gdf.geometry.buffer(2000)
    bounds_geom = bounds_gdf.union_all()
    # gather canopies
    all_canopies = []
    # iter zip files and load if intersecting bounds
    dir_path: Path = Path(data_dir_path)
    unzip_dir = dir_path / "temp_unzipped/"
    for zip_file_name in tqdm(os.listdir(dir_path)):
        if not zip_file_name.endswith(".zip"):
            continue
        # Create directory for unzipped files
        if os.path.exists(unzip_dir):
            shutil.rmtree(unzip_dir)
        os.makedirs(unzip_dir)
        full_zip_path = dir_path / zip_file_name
        # Unzip
        with zipfile.ZipFile(full_zip_path, "r") as zip_ref:
            zip_ref.extractall(unzip_dir)
        # Extract features from the geopackage file
        for walk_dir_path, _dir_names, file_names in os.walk(unzip_dir):
            for file_name in file_names:
                if not file_name.endswith(".gpkg"):
                    continue
                full_gpkg_path = str((Path(walk_dir_path) / file_name).resolve())
                # use fiona for quick bbox check
                try:
                    with fiona.open(full_gpkg_path) as src:  # type: ignore
                        if not geometry.box(*src.bounds).intersects(bounds_geom):  # type: ignore
                            continue
                except Exception:
                    # skip files that can't be opened
                    continue
                # read only features within the bounds bbox when possible
                try:
                    gdf = gpd.read_file(full_gpkg_path, bbox=bounds_geom.bounds)  # type: ignore
                except Exception:
                    gdf = gpd.read_file(full_gpkg_path)  # type: ignore
                if gdf.empty:
                    continue
                # filter spatially using bbox envelope for speed
                gdf = gdf.loc[gdf.geometry.notna()].copy()
                gdf["bbox"] = gdf["geometry"].envelope
                gdf_itx = gdf.set_geometry("bbox")
                gdf_itx = gdf_itx.loc[gdf_itx.intersects(bounds_geom)].copy()
                if gdf_itx.empty:
                    continue
                # rename geometry column and set it as active geometry
                gdf_itx = gdf_itx.rename(columns={"geometry": "geom"})
                gdf_itx = gdf_itx.set_geometry("geom")
                # explode multipolygons
                gdf_exp = gdf_itx.explode(index_parts=False)
                # select available columns
                cols = [c for c in ["fua_name", "fua_code", "geom"] if c in gdf_exp.columns]
                if not cols:
                    continue
                all_canopies.append(gdf_exp[cols])
        # Delete the unzipped files
        shutil.rmtree(unzip_dir)
    # save to file
    if all_canopies:
        final_gdf = gpd.GeoDataFrame(pd.concat(all_canopies, ignore_index=True))
        final_gdf.to_file(trees_out_path, driver="GPKG")


if __name__ == "__main__":
    """
    python -m src.data.load_urban_atlas_trees \
        temp/datasets/boundaries.gpkg \
            temp/STL_2018_3035_eu \
                temp/datasets/tree_canopies.gpkg
    """
    if True:
        parser = argparse.ArgumentParser(description="Load tree canopy data.")
        parser.add_argument("bounds_in_path", type=str, help="Input data directory with boundary GPKG.")
        parser.add_argument(
            "data_dir_path", type=str, help="Input data directory with zipped Urban Atlas tree canopy files."
        )
        parser.add_argument("trees_out_path", type=str, help="Output path for urban trees GPKG.")
        args = parser.parse_args()
        load_tree_canopies(
            bounds_in_path=args.bounds_in_path,
            data_dir_path=args.data_dir_path,
            trees_out_path=args.trees_out_path,
        )
    else:
        load_tree_canopies(
            bounds_in_path="temp/datasets/boundaries.gpkg",
            data_dir_path="temp/STL_2018_3035_eu",
            trees_out_path="temp/datasets/tree_canopies.gpkg",
        )
