""" """

import argparse

import geopandas as gpd
import osmnx as ox
from rasterio.features import shapes
from rasterio.io import MemoryFile
from shapely import geometry

from src import tools

logger = tools.get_logger(__name__)


def extract_boundary_polys(raster_in_path: str, bounds_out_path: str) -> None:
    """ """
    tools.validate_filepath(raster_in_path)
    tools.validate_directory(bounds_out_path, create=True)
    # fetch UK boundary to filter out
    uk_boundary = ox.geocode_to_gdf("United Kingdom").to_crs("3035").iloc[0].geometry
    # and EU boundary to filter out remote islands (including Madeira)
    eu_bounds = [2500000, 1250000, 7000000, 5000000]  # E, S, W, N - EPSG:3035
    logger.info(f"Clipping polygons outside of hard-coded EU boundary: {eu_bounds} (ESWN / EPSG:3035)")
    eu_boundary = geometry.box(*eu_bounds)  # type: ignore
    # extract polygons from raster
    polys: list[geometry.Polygon] = []
    with MemoryFile(open(raster_in_path, "rb")) as memfile, memfile.open() as dataset:
        rast_array = dataset.read(1)
        for geom, value in shapes(rast_array, transform=dataset.transform):
            # -21474836... represents no value
            if value < 0:
                continue
            # extract polygon features from raster blobs
            poly = geometry.shape(geom)
            # log if anything problematic found
            if not isinstance(poly, geometry.Polygon):
                logger.warning(f"Discarding extracted geom of type {poly.geom_type}")
                continue
            # don't load if intersecting UK
            if uk_boundary.contains(poly):
                continue
            # don't load if outside EU
            if not eu_boundary.contains(poly):
                continue
            # buffer and reverse buffer to smooth edges
            poly = poly.buffer(2000).buffer(-1000)
            # agg
            polys.append(poly)

    # generate the gdf
    data = {"geom": polys}
    bounds_gdf = gpd.GeoDataFrame(data, geometry="geom", crs=dataset.crs)  # type:ignore
    bounds_gdf["geom_2000"] = bounds_gdf["geom"].buffer(2000)  # type:ignore
    bounds_gdf["geom_10000"] = bounds_gdf["geom"].buffer(10000)  # type:ignore
    # unioned boundaries using GeoPandas/Shapely
    unioned_2000 = bounds_gdf["geom_2000"].union_all()
    unioned_10000 = bounds_gdf["geom_10000"].union_all()
    # save bounds
    bounds_gdf = bounds_gdf.drop(columns=["geom_2000", "geom_10000"])
    bounds_gdf.to_file(bounds_out_path, driver="GPKG", layer="bounds")
    # 2km buffer
    unioned_2000_gdf = gpd.GeoDataFrame({"geom": [unioned_2000]}, geometry="geom", crs=bounds_gdf.crs)
    unioned_2000_gdf = unioned_2000_gdf.explode(index_parts=False).reset_index(drop=True)
    unioned_2000_gdf.to_file(bounds_out_path, driver="GPKG", layer="unioned_bounds_2000")
    # 10km buffer
    unioned_10000_gdf = gpd.GeoDataFrame({"geom": [unioned_10000]}, geometry="geom", crs=bounds_gdf.crs)
    unioned_10000_gdf = unioned_10000_gdf.explode(index_parts=False).reset_index(drop=True)
    unioned_10000_gdf.to_file(bounds_out_path, driver="GPKG", layer="unioned_bounds_10000")


if __name__ == "__main__":
    """ """
    logger.info("Converting raster boundaries to polygons.")
    if True:
        parser = argparse.ArgumentParser(description="Load building heights raster data.")
        parser.add_argument("raster_data_path", type=str, help="Path to the raster data.")
        parser.add_argument("bounds_output_path", type=str, help="Path to the data output.")
        args = parser.parse_args()
        extract_boundary_polys(args.raster_data_path, args.bounds_output_path)
    else:
        extract_boundary_polys("temp/HDENS-CLST-2021/HDENS_CLST_2021.tif", "temp/datasets/boundaries.gpkg")
