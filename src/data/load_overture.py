"""ingest_overture

Utilities to build Overture-derived GeoPackage outputs for a set of
boundary geometries. Each output file contains network (primal + dual),
infrastructure, places and buildings layers for a single boundary.

The module intentionally keeps logic small and delegates heavy lifting to
`src.data.loaders` and `cityseer.tools`.
"""

import argparse
import os
import traceback
from concurrent import futures
from pathlib import Path

import geopandas as gpd
from cityseer.tools import graphs, io
from shapely import geometry, wkt
from tqdm import tqdm

from src import tools
from src.data import overture_data

logger = tools.get_logger(__name__)


WORKING_CRS = 3035


def load_overture_layers(bounds_fid: str, bounds_geom_wgs_wkt: str, output_path: str) -> None:
    """Create and write per-boundary GeoPackage layers.

    This function receives a WKT representation of the boundary geometry
    (so it can be safely sent to worker processes), reconstructs the
    Shapely geometry, and uses loader helpers to read and write the
    resulting GeoDataFrames into `output_path`.
    """

    # Reconstruct geometry from WKT (safe for multiprocessing)
    bounds_geom_wgs: geometry.Polygon = wkt.loads(bounds_geom_wgs_wkt)  # type: ignore
    # NETWORK
    nodes_gdf, edges_gdf, clean_edges_gdf = overture_data.load_network(bounds_geom_wgs, to_crs=WORKING_CRS)
    # nodes
    nodes_gdf["bounds_fid"] = bounds_fid
    nodes_gdf.to_file(output_path, driver="GPKG", layer="nodes")
    # edges
    edges_gdf["bounds_fid"] = bounds_fid
    edges_gdf.to_file(output_path, driver="GPKG", layer="edges")
    # clean edges
    clean_edges_gdf["bounds_fid"] = bounds_fid
    clean_edges_gdf.to_file(output_path, driver="GPKG", layer="clean_edges")
    # DUAL CLEAN NETWORK
    nx_clean = io.nx_from_generic_geopandas(clean_edges_gdf)
    # cast to dual
    nx_dual = graphs.nx_to_dual(nx_clean)
    # back to GDF
    nodes_dual_gdf, edges_dual_gdf, _network_structure = io.network_structure_from_nx(nx_dual)
    # write dual nodes
    nodes_dual_gdf["bounds_fid"] = bounds_fid
    nodes_dual_gdf.to_file(output_path, driver="GPKG", layer="dual_nodes")
    # write dual edges
    edges_dual_gdf["bounds_fid"] = bounds_fid
    edges_dual_gdf.to_file(output_path, driver="GPKG", layer="dual_edges")

    # OVERTURE INFRASTRUCTURE
    infrast_gdf = overture_data.load_infrastructure(bounds_geom_wgs, to_crs=WORKING_CRS)
    infrast_gdf["bounds_fid"] = bounds_fid
    infrast_gdf.to_file(output_path, driver="GPKG", layer="infrastructure")

    # OVERTURE PLACES
    places_gdf = overture_data.load_places(bounds_geom_wgs, to_crs=WORKING_CRS)
    places_gdf["bounds_fid"] = bounds_fid
    places_gdf.to_file(output_path, driver="GPKG", layer="places")

    # OVERTURE BUILDINGS
    buildings_gdf = overture_data.load_buildings(bounds_geom_wgs, to_crs=WORKING_CRS)
    buildings_gdf["bounds_fid"] = bounds_fid
    buildings_gdf.to_file(output_path, driver="GPKG", layer="buildings")


def load_overture_for_bounds(
    bounds_in_path: str, cities_data_out_dir: str, parallel_workers: int, overwrite: bool
) -> None:
    """Dispatch workers to produce per-boundary Overture GeoPackages.

    Reads a `bounds` layer from `bounds_in_path`, buffers each boundary in a
    projected CRS to produce a stable metric buffer, and then submits a
    worker task to create a GeoPackage for each boundary.
    """
    # set to quiet mode
    os.environ["CITYSEER_QUIET_MODE"] = "true"
    tools.validate_filepath(bounds_in_path)
    tools.validate_directory(cities_data_out_dir, create=True)
    logger.info("Loading overture networks")
    bounds_gdf = gpd.read_file(bounds_in_path, layer="bounds")
    # Buffer in a projected CRS so the 2000-unit buffer is in metres.
    bounds_gdf = bounds_gdf.to_crs(WORKING_CRS)
    bounds_gdf.geometry = bounds_gdf.geometry.buffer(2000)
    # Convert back to WGS84 for downstream loaders that expect geographic CRS
    bounds_gdf = bounds_gdf.to_crs(4326)
    # use futures to parallelize
    futs = {}
    with futures.ProcessPoolExecutor(max_workers=parallel_workers) as executor:
        try:
            # loop through bounds and load networks
            for bounds_fid, bounds_row in bounds_gdf.iterrows():
                output_path = Path(cities_data_out_dir) / f"overture_{bounds_fid}.gpkg"
                if output_path.exists() and not overwrite:
                    logger.info(f"Skipping existing file: {output_path}")
                    continue
                # Pass WKT to workers to avoid pickling Shapely geometry objects
                args = (bounds_fid, bounds_row.geometry.wkt, output_path)
                futs[executor.submit(load_overture_layers, *args)] = args  # type: ignore
            # iterate over completed futures and update progress with tqdm
            for fut in tqdm(futures.as_completed(futs), total=len(futs), desc="Loading Overture"):
                try:
                    fut.result()
                except Exception as exc:
                    logger.error(traceback.format_exc())
                    raise RuntimeError("An error occurred in the background task") from exc
        except KeyboardInterrupt:
            executor.shutdown(wait=True, cancel_futures=True)
            raise


if __name__ == "__main__":
    """
    python -m src.data.load_overture temp/datasets/boundaries.gpkg temp/cities_data --parallel_workers 4
    """
    if True:
        parser = argparse.ArgumentParser(description="Load overture networks.")
        parser.add_argument("bounds_in_path", type=str, help="Input data directory with boundary GPKG.")
        parser.add_argument("cities_data_out_dir", type=str, help="Output data directory for city GPKG files.")
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
            cities_data_out_dir=args.cities_data_out_dir,
            parallel_workers=args.parallel_workers,
            overwrite=args.overwrite,
        )
    else:
        load_overture_for_bounds(
            bounds_in_path="temp/datasets/boundaries.gpkg",
            cities_data_out_dir="temp/cities_data",
            parallel_workers=4,
            overwrite=False,
        )
