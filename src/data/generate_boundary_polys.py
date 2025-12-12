""" """

import argparse

import geopandas as gpd
import osmnx as ox
from overturemaps import core
from rasterio.features import shapes
from rasterio.io import MemoryFile
from shapely import geometry

from src import tools

logger = tools.get_logger(__name__)
WORKING_CRS = 3035


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
    bounds_gdf = bounds_gdf.to_crs(WORKING_CRS)
    bounds_gdf["label"] = None

    # query overture divisions for entire bounds_gdf at once
    overall_bounds_wgs = tools.reproject_geometry(
        geometry.box(*bounds_gdf.total_bounds),  # type: ignore
        from_crs=bounds_gdf.crs.to_epsg(),  # type: ignore
        to_crs=4326,  # type: ignore
    )
    logger.info("Querying divisions for entire dataset bounds")
    division_gdf = core.geodataframe("division", overall_bounds_wgs.bounds, stac=True)  # type:ignore

    # helper function to extract first locality from hierarchy
    def get_first_localadmin(hierarchies_value):
        """Extract the first (parent) locality from the hierarchy."""
        try:
            if len(hierarchies_value) == 0:
                return None
            hierarchy = hierarchies_value[0]
            localities = [d.get("name") for d in hierarchy if d.get("subtype") == "localadmin"]
            if not localities:
                localities = [d.get("name") for d in hierarchy if d.get("subtype") == "locality"]
            return localities[0] if localities else None
        except (IndexError, TypeError, AttributeError):
            return None

    if not division_gdf.empty:
        division_gdf = division_gdf.set_crs(4326)
        division_gdf = division_gdf.to_crs(WORKING_CRS)
        division_gdf.to_file("temp/divisions.gpkg", driver="GPKG", layer="divisions")
        division_gdf["admin_name"] = division_gdf["hierarchies"].apply(get_first_localadmin)
        # now filter for each boundary polygon
        for bounds_fid, bounds_row in bounds_gdf.iterrows():
            # filter divisions that are contained within this boundary
            contained_divisions = division_gdf[bounds_row.geom.contains(division_gdf.geometry)]
            if not contained_divisions.empty:
                locality_counts = contained_divisions["admin_name"].value_counts()
                most_common_locality = locality_counts.index[0] if not locality_counts.empty else None
            else:
                most_common_locality = None
            bounds_gdf.loc[bounds_fid, "label"] = most_common_locality  # type: ignore
            if not contained_divisions.empty:
                locality_counts = contained_divisions["country"].value_counts()
                most_common_country = locality_counts.index[0] if not locality_counts.empty else None
            else:
                most_common_country = None
            bounds_gdf.loc[bounds_fid, "country"] = most_common_country  # type: ignore
            # filter country codes
            logger.info(
                f"Assigned label '{most_common_locality}' to boundary {bounds_fid} and country '{most_common_country}'"
            )
    bounds_gdf["bounds_fid"] = bounds_gdf.index
    bounds_gdf.to_file(bounds_out_path, driver="GPKG", layer="bounds")


if __name__ == "__main__":
    """
    python -m src.data.generate_boundary_polys temp/HDENS-CLST-2021/HDENS_CLST_2021.tif temp/datasets/boundaries.gpkg
    """
    logger.info("Converting raster boundaries to polygons.")
    if True:
        parser = argparse.ArgumentParser(description="Load building heights raster data.")
        parser.add_argument("raster_data_path", type=str, help="Path to the raster data.")
        parser.add_argument("bounds_output_path", type=str, help="Path to the data output.")
        args = parser.parse_args()
        extract_boundary_polys(args.raster_data_path, args.bounds_output_path)
    else:
        extract_boundary_polys("temp/HDENS-CLST-2021/HDENS_CLST_2021.tif", "temp/datasets/boundaries.gpkg")
