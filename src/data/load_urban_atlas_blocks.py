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


def load_urban_blocks(bounds_in_path: str, data_dir_path: str, blocks_out_path: str) -> None:
    """ """
    logger.info(f"Loading urban atlas blocks data from path: {bounds_in_path}")
    tools.validate_filepath(bounds_in_path)
    tools.validate_directory(data_dir_path)
    tools.validate_directory(blocks_out_path, create=True)
    # load bounds
    bounds_gdf = gpd.read_file(bounds_in_path, layer="bounds")
    bounds_gdf.geometry = bounds_gdf.geometry.buffer(2000)
    bounds_geom = bounds_gdf.union_all()
    # filter out unwanted block types
    filter_classes = [
        "Fast transit roads and associated land",
        "Other roads and associated land",
        "Railways and associated land",
    ]
    all_blocks = []
    # iter zip files and load if intersecting bounds
    dir_path: Path = Path(data_dir_path)
    unzip_dir = dir_path / "temp_unzipped/"
    for zip_file_name in tqdm(os.listdir(dir_path)):  # type: ignore
        if not zip_file_name.endswith(".zip"):
            continue
        # Create directory for unzipped files
        if os.path.exists(unzip_dir):
            shutil.rmtree(unzip_dir)
        os.makedirs(unzip_dir, exist_ok=True)
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
                    with fiona.open(full_gpkg_path) as src:
                        if not geometry.box(*src.bounds).intersects(bounds_geom):  # type: ignore
                            continue
                except Exception:
                    # skip files that can't be opened
                    continue
                # read only features within the bounds bbox when possible
                try:
                    gdf = gpd.read_file(full_gpkg_path, bbox=bounds_geom.bounds)
                except Exception:
                    gdf = gpd.read_file(full_gpkg_path)
                if gdf.empty:
                    continue
                # discard rows if in filtered classes and avoid chained assignment
                gdf = gdf.loc[~gdf["class_2018"].isin(filter_classes)]
                if gdf.empty:
                    continue
                # filter spatially using envelope bbox for speed, then refine
                gdf["bbox"] = gdf["geometry"].envelope
                gdf_itx = gdf.set_geometry("bbox")
                gdf_itx = gdf_itx.loc[gdf_itx.intersects(bounds_geom)]
                if gdf_itx.empty:
                    continue
                # rename geometry column and set it as active geometry
                gdf_itx = gdf_itx.rename(columns={"geometry": "geom", "Pop2018": "pop2018"})
                gdf_itx = gdf_itx.set_geometry("geom")
                # explode multipolygons
                gdf_exp = gdf_itx.explode(index_parts=False)
                # keep all even large geoms as these are sometimes shorelines
                # write to postgis
                cols = [
                    "country",
                    "fua_name",
                    "fua_code",
                    "code_2018",
                    "class_2018",
                    "identifier",
                    "comment",
                    "pop2018",
                    "geom",
                ]
                available_cols = [c for c in cols if c in gdf_exp.columns]
                if not available_cols:
                    continue
                all_blocks.append(gdf_exp[available_cols])
        # Delete the unzipped files
        shutil.rmtree(unzip_dir)
    # save to file
    if all_blocks:
        final_gdf = gpd.GeoDataFrame(pd.concat(all_blocks, ignore_index=True))
        final_gdf.to_file(blocks_out_path, driver="GPKG")


if __name__ == "__main__":
    """
    python -m src.data.load_urban_atlas_blocks \
        temp/datasets/boundaries.gpkg \
            temp/UA_2018_3035_eu \
                temp/datasets/blocks.gpkg
    """
    if True:
        parser = argparse.ArgumentParser(description="Load Urban Atlas data.")
        parser.add_argument("bounds_in_path", type=str, help="Input data directory with boundary GPKG.")
        parser.add_argument(
            "data_dir_path", type=str, help="Input data directory with zipped Urban Atlas blocks files."
        )
        parser.add_argument("blocks_out_path", type=str, help="Output path for urban blocks GPKG.")
        args = parser.parse_args()
        load_urban_blocks(
            bounds_in_path=args.bounds_in_path,
            data_dir_path=args.data_dir_path,
            blocks_out_path=args.blocks_out_path,
        )
    else:
        load_urban_blocks(
            bounds_in_path="temp/datasets/boundaries.gpkg",
            data_dir_path="temp/UA_2018_3035_eu",
            blocks_out_path="temp/datasets/blocks.gpkg",
        )
