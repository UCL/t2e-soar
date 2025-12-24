""" """

from pathlib import Path

import geopandas as gpd
import momepy
import numpy as np
import rasterio
from cityseer.metrics import layers, networks
from rasterio.mask import mask
from tqdm import tqdm

from src import landuse_categories, tools

logger = tools.get_logger(__name__)

OVERTURE_SCHEMA = tools.generate_overture_schema()  # type: ignore
DISTANCES_LU = [200, 400, 800, 1200, 1600]
DISTANCES_CENT = [400, 800, 1200, 1600, 4800]
DISTANCES_MORPH = [100, 200]
DISTANCES_GREEN_REACH = [1600]
DISTANCES_GREEN_AGG = [200, 400, 800]


def process_centrality(nodes_gdf: gpd.GeoDataFrame, network_structure) -> gpd.GeoDataFrame:
    """ """
    logger.info("Computing centrality")
    nodes_gdf = networks.node_centrality_shortest(network_structure, nodes_gdf, distances=DISTANCES_CENT)
    return nodes_gdf


def process_places(
    nodes_gdf: gpd.GeoDataFrame, places_gdf: gpd.GeoDataFrame, infrast_gdf: gpd.GeoDataFrame, network_structure
) -> gpd.GeoDataFrame:
    """ """
    logger.info("Computing places")
    # apply standardized category merging
    places_gdf = landuse_categories.merge_landuse_categories(places_gdf)
    # landuses
    landuse_keys = places_gdf["merged_cats"].unique().tolist()
    # compute accessibilities
    nodes_gdf, places_gdf = layers.compute_accessibilities(
        places_gdf,  # type: ignore
        landuse_column_label="merged_cats",
        accessibility_keys=landuse_keys,
        nodes_gdf=nodes_gdf,
        network_structure=network_structure,
        distances=DISTANCES_LU,
    )
    nodes_gdf, places_gdf = layers.compute_mixed_uses(
        places_gdf,
        landuse_column_label="merged_cats",
        nodes_gdf=nodes_gdf,
        network_structure=network_structure,
        distances=DISTANCES_LU,
    )
    # infrastructure
    street_furn_keys = [
        "bench",
        "drinking_water",
        "fountain",
        "picnic_table",
        "plant",
        "planter",
        "post_box",
    ]
    parking_keys = [
        # "bicycle_parking",
        "motorcycle_parking",
        "parking",
    ]
    transport_keys = [
        "aerialway_station",
        "airport",
        "bus_station",
        "bus_stop",
        "ferry_terminal",
        "helipad",
        "international_airport",
        "railway_station",
        "regional_airport",
        "seaplane_airport",
        "subway_station",
    ]
    infrast_gdf["class"] = infrast_gdf["class"].replace(street_furn_keys, "street_furn")  # type: ignore
    infrast_gdf["class"] = infrast_gdf["class"].replace(parking_keys, "parking")  # type: ignore
    infrast_gdf["class"] = infrast_gdf["class"].replace(transport_keys, "transport")  # type: ignore
    landuse_keys = ["street_furn", "parking", "transport"]
    infrast_gdf = infrast_gdf[infrast_gdf["class"].isin(landuse_keys)]  # type: ignore
    # compute accessibilities
    nodes_gdf, infrast_gdf = layers.compute_accessibilities(
        infrast_gdf,  # type: ignore
        landuse_column_label="class",
        accessibility_keys=landuse_keys,
        nodes_gdf=nodes_gdf,
        network_structure=network_structure,
        distances=DISTANCES_LU,
    )
    return nodes_gdf


def process_blocks_buildings(
    nodes_gdf: gpd.GeoDataFrame,
    bldgs_gdf: gpd.GeoDataFrame,
    blocks_gdf: gpd.GeoDataFrame,
    hts_path: str | Path | None,
    network_structure,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """ """
    logger.info("Computing morphology")
    # placeholders
    for col_key in [
        "area",
        "perimeter",
        "compactness",
        "orientation",
        "volume",
        "floor_area_ratio",
        "form_factor",
        "corners",
        "shape_index",
        "shared_walls",
        "fractal_dimension",
    ]:
        bldgs_gdf[col_key] = np.nan
    if not bldgs_gdf.empty:
        # explode
        bldgs_gdf = bldgs_gdf.explode(index_parts=False)  # type: ignore
        bldgs_gdf.reset_index(drop=True, inplace=True)
        bldgs_gdf.index = bldgs_gdf.index.astype(str)
        bldgs_gdf["mean_height"] = np.nan
        # sample heights when a raster is available
        if hts_path is not None:
            hts_path = Path(hts_path)
            if hts_path.exists():
                with rasterio.open(hts_path) as rast_src:
                    logger.info("Sampling building heights")
                    heights = []
                    for _idx, bldg_row in tqdm(bldgs_gdf.iterrows(), total=len(bldgs_gdf)):
                        try:
                            # raster values within building polygon
                            out_image, _ = mask(
                                rast_src,
                                [bldg_row.geometry.buffer(10)],
                                all_touched=True,
                                crop=True,
                                nodata=rast_src.nodata,
                            )
                            # Filter out nodata values before computing mean
                            raster_data = out_image[0]
                            if rast_src.nodata is not None:
                                # Mask out nodata values
                                valid_data = raster_data[raster_data != rast_src.nodata]
                            else:
                                valid_data = raster_data
                            # Compute mean, excluding NaN values as well
                            mean_height = np.nanmean(valid_data) if len(valid_data) > 0 else np.nan
                            heights.append(mean_height)
                        except ValueError:
                            heights.append(np.nan)
                    bldgs_gdf["mean_height"] = heights
            else:
                logger.warning("Building heights raster not available at %s", hts_path)
        else:
            logger.warning("No building heights raster available; leaving height metrics empty")
        # bldg metrics
        area = bldgs_gdf.area
        ht = bldgs_gdf.loc[:, "mean_height"]
        bldgs_gdf["area"] = area
        bldgs_gdf["perimeter"] = bldgs_gdf.length
        bldgs_gdf["compactness"] = momepy.circular_compactness(bldgs_gdf)
        bldgs_gdf["orientation"] = momepy.orientation(bldgs_gdf)
        # height-based metrics
        bldgs_gdf["volume"] = momepy.volume(area, ht)
        bldgs_gdf["floor_area_ratio"] = momepy.floor_area(area, ht, 3)
        bldgs_gdf["form_factor"] = momepy.form_factor(bldgs_gdf, ht)
        # complexity metrics
        bldgs_gdf["corners"] = momepy.corners(bldgs_gdf)
        bldgs_gdf["shape_index"] = momepy.shape_index(bldgs_gdf)
        bldgs_gdf["shared_walls"] = momepy.shared_walls(bldgs_gdf, strict=False, tolerance=0.5)
        bldgs_gdf["fractal_dimension"] = momepy.fractal_dimension(bldgs_gdf)
    # calculate
    bldgs_gdf["centroid"] = bldgs_gdf.geometry.centroid
    bldgs_gdf.set_geometry("centroid", inplace=True)
    bldg_stats_cols = [
        "area",
        "mean_height",  # already computed prior
        "perimeter",
        "compactness",
        "orientation",
        "volume",
        "floor_area_ratio",
        "form_factor",
        "corners",
        "shape_index",
        "shared_walls",
        "fractal_dimension",
    ]
    nodes_gdf, bldgs_gdf = layers.compute_stats(
        data_gdf=bldgs_gdf,
        stats_column_labels=bldg_stats_cols,
        nodes_gdf=nodes_gdf,
        network_structure=network_structure,
        distances=DISTANCES_MORPH,
    )
    for bldg_stats_col in bldg_stats_cols:
        trim_columns = []
        for column_name in nodes_gdf.columns:
            if column_name.startswith(f"cc_{bldg_stats_col}") and not (
                column_name.startswith(f"cc_{bldg_stats_col}_median")
                or column_name.startswith(f"cc_{bldg_stats_col}_mad")
            ):
                trim_columns.append(column_name)
        nodes_gdf.drop(columns=trim_columns, inplace=True)
    finite_idx = np.isfinite(bldgs_gdf["perimeter"])
    bldgs_gdf.loc[finite_idx, "shared_wall_ratio"] = (
        bldgs_gdf.loc[finite_idx, "shared_walls"] / bldgs_gdf.loc[finite_idx, "perimeter"]
    )
    bldgs_gdf["type"] = "building"  # for downstream use
    nodes_gdf, bldgs_gdf = layers.compute_accessibilities(
        bldgs_gdf,  # type: ignore
        landuse_column_label="type",
        accessibility_keys=["building"],
        nodes_gdf=nodes_gdf,
        network_structure=network_structure,
        distances=DISTANCES_MORPH,
    )
    nodes_gdf = nodes_gdf.drop(columns=[f"cc_building_nearest_max_{max(DISTANCES_MORPH)}"])
    # placeholders
    for col_key in [
        "block_area",
        "block_perimeter",
        "block_compactness",
        "block_orientation",
        "block_covered_ratio",
    ]:
        blocks_gdf[col_key] = np.nan
    # block metrics
    if not blocks_gdf.empty:
        blocks_gdf.index = blocks_gdf.index.astype(str)
        blocks_gdf["block_area"] = blocks_gdf.area
        blocks_gdf["block_perimeter"] = blocks_gdf.length
        blocks_gdf["block_compactness"] = momepy.circular_compactness(blocks_gdf)
        blocks_gdf["block_orientation"] = momepy.orientation(blocks_gdf)
    # joint metrics require spatial join
    if not blocks_gdf.empty and not bldgs_gdf.empty:
        blocks_gdf["uID"] = blocks_gdf.index.values
        merged_gdf = gpd.sjoin(
            bldgs_gdf,
            blocks_gdf,
            how="left",
            predicate="intersects",
            lsuffix="bldg",
            rsuffix="block",
        )
        # Calculate covered ratio: sum of building areas per block / block area
        # Group buildings by block ID and sum their areas
        building_area_per_block = merged_gdf.groupby("uID")["area"].sum()
        # Divide by block area to get coverage ratio
        blocks_gdf["block_covered_ratio"] = building_area_per_block / blocks_gdf["block_area"]
        # Fill NaN values (blocks with no buildings) with 0
        blocks_gdf["block_covered_ratio"] = blocks_gdf["block_covered_ratio"].fillna(0)
    # block stats
    blocks_gdf["centroid"] = blocks_gdf.geometry.centroid
    blocks_gdf.set_geometry("centroid", inplace=True)
    block_stats_cols = [
        "block_area",
        "block_perimeter",
        "block_compactness",
        "block_orientation",
        "block_covered_ratio",
    ]
    nodes_gdf, blocks_gdf = layers.compute_stats(
        data_gdf=blocks_gdf,
        stats_column_labels=block_stats_cols,
        nodes_gdf=nodes_gdf,
        network_structure=network_structure,
        distances=DISTANCES_MORPH,
    )
    for block_stats_col in block_stats_cols:
        trim_columns = []
        for column_name in nodes_gdf.columns:
            if column_name.startswith(f"cc_{block_stats_col}") and not (
                column_name.startswith(f"cc_{block_stats_col}_median")
                or column_name.startswith(f"cc_{block_stats_col}_mad")
            ):
                trim_columns.append(column_name)
        nodes_gdf.drop(columns=trim_columns, inplace=True)
    blocks_gdf["type"] = "block"  # for downstream use
    nodes_gdf, blocks_gdf = layers.compute_accessibilities(
        blocks_gdf,  # type: ignore
        landuse_column_label="type",
        accessibility_keys=["block"],
        nodes_gdf=nodes_gdf,
        network_structure=network_structure,
        distances=DISTANCES_MORPH,
    )
    nodes_gdf = nodes_gdf.drop(columns=[f"cc_block_nearest_max_{max(DISTANCES_MORPH)}"])
    # reset geometry
    bldgs_gdf.set_geometry("geometry", inplace=True)
    bldgs_gdf.drop(columns=["centroid"], inplace=True)
    blocks_gdf.set_geometry("geometry", inplace=True)
    blocks_gdf.drop(columns=["centroid"], inplace=True)

    return nodes_gdf, bldgs_gdf, blocks_gdf


def process_green(
    nodes_gdf: gpd.GeoDataFrame, green_gdf: gpd.GeoDataFrame, trees_gdf: gpd.GeoDataFrame, network_structure
) -> gpd.GeoDataFrame:
    """ """
    # Intentionally using points for handling extra large features like rivers
    logger.info("Computing green")
    # check Polygons
    green_gdf = green_gdf.explode(index_parts=False)  # type: ignore
    green_gdf.reset_index(drop=True, inplace=True)
    # check Polygons
    trees_gdf = trees_gdf.explode(index_parts=False)  # type: ignore
    trees_gdf.reset_index(drop=True, inplace=True)

    # function for extracting points
    def generate_points(fid, categ, polygon, area, interval=20, simplify=20):
        if polygon.is_empty or polygon.exterior.length == 0:
            return []
        ring = polygon.exterior.simplify(simplify)
        num_points = int(ring.length // interval)
        return [
            (fid, categ, area, ring.interpolate(distance)) for distance in range(0, num_points * interval, interval)
        ]

    # extract points
    points = []
    # for green
    for fid, geom in zip(green_gdf.index, green_gdf.geometry, strict=True):  # type: ignore
        if geom.geom_type == "Polygon":
            points.extend(generate_points(fid, "green", geom, geom.area, interval=20, simplify=10))
    # for trees
    for fid, geom in zip(trees_gdf.index, trees_gdf.geometry, strict=True):  # type: ignore
        if geom.geom_type == "Polygon":
            points.extend(generate_points(fid, "trees", geom, geom.area, interval=20, simplify=5))
    # create GDF
    points_gdf = gpd.GeoDataFrame(  # type: ignore
        points,
        columns=["fid", "cat", "area", "geometry"],
        geometry="geometry",
        crs=trees_gdf.crs,  # type: ignore
    )
    points_gdf.index = points_gdf.index.astype(str)
    # relabel area to green_area and trees_area
    green_idx = points_gdf["cat"] == "green"
    trees_idx = points_gdf["cat"] == "trees"
    points_gdf.loc[green_idx, "green_area"] = points_gdf.loc[green_idx, "area"]
    points_gdf.loc[green_idx, "trees_area"] = 0.0
    points_gdf.loc[trees_idx, "trees_area"] = points_gdf.loc[trees_idx, "area"]
    points_gdf.loc[trees_idx, "green_area"] = 0.0
    points_gdf = points_gdf.drop(columns=["area"])
    # compute accessibilities
    nodes_gdf, points_gdf = layers.compute_accessibilities(
        points_gdf,  # type: ignore
        landuse_column_label="cat",
        accessibility_keys=["green", "trees"],
        nodes_gdf=nodes_gdf,
        network_structure=network_structure,
        distances=DISTANCES_GREEN_REACH,
        data_id_col="fid",  # deduplicate
    )
    # drop - aggregation columns since these are not meaningful for interpolated aggs - only using distances
    nodes_gdf = nodes_gdf.drop(
        columns=[
            "cc_green_1600_nw",
            "cc_green_1600_wt",
            "cc_trees_1600_nw",
            "cc_trees_1600_wt",
        ]
    )
    # set contained green nodes to zero
    contained_green_idx = gpd.sjoin(nodes_gdf, green_gdf, predicate="intersects", how="inner")
    nodes_gdf.loc[contained_green_idx.index, "cc_green_nearest_max_1600"] = 0
    # same for trees
    contained_trees_idx = gpd.sjoin(nodes_gdf, trees_gdf, predicate="intersects", how="inner")
    nodes_gdf.loc[contained_trees_idx.index, "cc_trees_nearest_max_1600"] = 0
    # sum areas within buffer distances
    points_gdf["green_area"] = points_gdf["green_area"] / (1000**2)  # m2 to km2
    points_gdf["trees_area"] = points_gdf["trees_area"] / (1000**2)  # m2 to km2
    nodes_gdf, points_gdf = layers.compute_stats(
        data_gdf=points_gdf,
        stats_column_labels=["green_area", "trees_area"],
        nodes_gdf=nodes_gdf,
        network_structure=network_structure,
        distances=DISTANCES_GREEN_AGG,
        data_id_col="fid",  # deduplicate
    )
    # drop unnecessary columns
    for area_col in ["green_area", "trees_area"]:
        trim_columns = []
        for column_name in nodes_gdf.columns:
            if column_name.startswith(f"cc_{area_col}") and not column_name.startswith(f"cc_{area_col}_sum"):
                trim_columns.append(column_name)
        nodes_gdf.drop(columns=trim_columns, inplace=True)
    return nodes_gdf
