""" """

import argparse
import traceback
from concurrent import futures
from pathlib import Path

import geopandas as gpd
from shapely import geometry, wkt
from tqdm import tqdm

from src import tools
from src.data import overture_data

logger = tools.get_logger(__name__)


WORKING_CRS = 3035

# Layers expected to be present in each per-boundary GeoPackage
REQUIRED_LAYERS = [
    "nodes",
    "edges",
    "clean_edges",
    "infrastructure",
    "places",
    "buildings",
]


def load_overture_layers(bounds_fid: str, bounds_geom_wkt: str, output_path: str) -> None:
    """ """
    # Reconstruct geometry from WKT (safe for multiprocessing)
    bounds_geom: geometry.Polygon = wkt.loads(bounds_geom_wkt)  # type: ignore
    bounds_geom_2km = bounds_geom.buffer(2000)
    bounds_geom_2km_wgs = tools.reproject_geometry(bounds_geom_2km, from_crs=WORKING_CRS, to_crs=4326)
    bounds_geom_10km = bounds_geom.buffer(10000)
    bounds_geom_10km_wgs = tools.reproject_geometry(bounds_geom_10km, from_crs=WORKING_CRS, to_crs=4326)
    # NETWORK
    nodes_gdf, edges_gdf, clean_edges_gdf = overture_data.load_network(bounds_geom_10km_wgs, to_crs=WORKING_CRS)
    # nodes
    nodes_gdf["bounds_fid"] = bounds_fid
    nodes_gdf.to_file(output_path, driver="GPKG", layer="nodes")
    # edges
    edges_gdf["bounds_fid"] = bounds_fid
    edges_gdf.to_file(output_path, driver="GPKG", layer="edges")
    # clean edges
    clean_edges_gdf["bounds_fid"] = bounds_fid
    clean_edges_gdf.to_file(output_path, driver="GPKG", layer="clean_edges")

    # OVERTURE INFRASTRUCTURE
    infrast_gdf = overture_data.load_infrastructure(bounds_geom_2km_wgs, to_crs=WORKING_CRS)
    infrast_gdf["bounds_fid"] = bounds_fid
    infrast_gdf.to_file(output_path, driver="GPKG", layer="infrastructure")

    # OVERTURE PLACES
    places_gdf = overture_data.load_places(bounds_geom_2km_wgs, to_crs=WORKING_CRS)
    places_gdf["bounds_fid"] = bounds_fid
    places_gdf.to_file(output_path, driver="GPKG", layer="places")

    # OVERTURE BUILDINGS
    buildings_gdf = overture_data.load_buildings(bounds_geom_2km_wgs, to_crs=WORKING_CRS)
    buildings_gdf["bounds_fid"] = bounds_fid
    buildings_gdf.to_file(output_path, driver="GPKG", layer="buildings")


def load_overture_for_bounds(bounds_in_path: str, data_out_dir: str, parallel_workers: int, overwrite: bool) -> None:
    """Dispatch workers to produce per-boundary Overture GeoPackages.

    Reads a `bounds` layer from `bounds_in_path`, buffers each boundary in a
    projected CRS to produce a stable metric buffer, and then submits a
    worker task to create a GeoPackage for each boundary.
    """
    # set to quiet mode
    tools.validate_filepath(bounds_in_path)
    tools.validate_directory(data_out_dir, create=True)
    logger.info("Loading overture networks")
    bounds_gdf = gpd.read_file(bounds_in_path, layer="bounds")
    bounds_gdf = bounds_gdf.to_crs(WORKING_CRS)
    # use futures to parallelize
    futs = {}
    with futures.ProcessPoolExecutor(max_workers=parallel_workers) as executor:
        try:
            # loop through bounds and load networks
            for bounds_fid, bounds_row in bounds_gdf.iterrows():
                output_path = Path(data_out_dir) / f"overture_{bounds_fid}.gpkg"
                # If the file exists and overwrite is False, check whether it
                # already contains all required layers. If so, skip. If not,
                # rebuild (i.e., set overwrite for layers to True so existing
                # incomplete layers are replaced).
                if output_path.exists() and not overwrite:
                    has_all = tools.gpkg_has_all_layers(str(output_path), REQUIRED_LAYERS)
                    if has_all:
                        logger.info(f"Skipping existing file with all layers: {output_path}")
                        continue
                    else:
                        logger.info(f"File missing some layers, will overwrite: {output_path}")
                # Pass WKT to workers to avoid pickling Shapely geometry objects
                args = (bounds_fid, bounds_row.geometry.wkt, output_path)
                futs[executor.submit(load_overture_layers, *args)] = args  # type: ignore
            # iterate over completed futures and update progress with tqdm
            for fut in tqdm(futures.as_completed(futs), total=len(futs), desc="Loading Overture"):
                # Immediately raise any exception that occurred in the worker
                # This will stop processing and report the error right away
                args = futs[fut]
                bounds_fid = args[0]
                try:
                    fut.result()
                    logger.info(f"Successfully completed: {bounds_fid}")
                except Exception as exc:
                    logger.error(f"Error processing {bounds_fid}:")
                    logger.error(traceback.format_exc())
                    # Immediately raise to stop all processing
                    raise RuntimeError(f"Error processing {bounds_fid}") from exc
        except KeyboardInterrupt:
            executor.shutdown(wait=True, cancel_futures=True)
            raise


if __name__ == "__main__":
    """
    python -m src.data.load_overture \
        temp/datasets/boundaries.gpkg \
            temp/cities_data/overture \
                --parallel_workers 6
    """
    if True:
        parser = argparse.ArgumentParser(description="Load overture networks.")
        parser.add_argument("bounds_in_path", type=str, help="Input data directory with boundary GPKG.")
        parser.add_argument("data_out_dir", type=str, help="Output data directory for city GPKG files.")
        parser.add_argument(
            "--parallel_workers",
            type=int,
            default=2,  # Set your desired default value here
            help="The number of CPU cores to use for processing bounds in parallel. Defaults to 2.",
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Overwrite existing files (default: False)",
            default=False,
        )
        args = parser.parse_args()
        load_overture_for_bounds(
            bounds_in_path=args.bounds_in_path,
            data_out_dir=args.data_out_dir,
            parallel_workers=args.parallel_workers,
            overwrite=args.overwrite,
        )
    else:
        load_overture_for_bounds(
            bounds_in_path="temp/datasets/boundaries.gpkg",
            data_out_dir="temp/cities_data/overture",
            parallel_workers=4,
            overwrite=False,
        )
