""" """

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
from cityseer.tools import graphs, io
from scipy.interpolate import griddata
from tqdm import tqdm

from src import tools
from src.processing import processors

logger = tools.get_logger(__name__)

# Layers expected to be present in each per-boundary GeoPackage
REQUIRED_LAYERS = [
    "buildings",
    "blocks",
    "streets",
]
WORKING_CRS = 3035


def process_metrics(
    bounds_in_path: str,
    overture_data_dir: str,
    blocks_path: str,
    trees_path: str,
    hts_raster_data_dir: str,
    stats_path: str,
    processed_data_dir: str,
    overwrite: bool = False,
):
    """ """
    tools.validate_filepath(bounds_in_path)
    bounds_gdf = gpd.read_file(bounds_in_path, layer="bounds")
    bounds_gdf = bounds_gdf.to_crs(WORKING_CRS)
    # process each boundary
    for bounds_fid, bounds_row in bounds_gdf.iterrows():
        logger.info(f"\n\nProcessing metrics for bounds fid: {bounds_fid}")
        tools.validate_directory(overture_data_dir)
        overture_path = Path(overture_data_dir) / f"overture_{bounds_fid}.gpkg"
        if not overture_path.exists():
            logger.warning(f"Missing overture file for bounds fid {bounds_fid}, skipping: {overture_path}")
            continue
        # output path
        tools.validate_directory(processed_data_dir, create=True)
        output_path = Path(processed_data_dir) / f"metrics_{bounds_fid}.gpkg"
        # check if already exists
        if output_path.exists() and not overwrite:
            has_all = tools.gpkg_has_all_layers(str(output_path), REQUIRED_LAYERS)
            if has_all:
                logger.info(f"Skipping existing file with all layers: {output_path}")
                continue
            else:
                logger.info(f"File missing some layers, will overwrite: {output_path}")
        # CENTRALITY
        clean_edges_gdf = gpd.read_file(overture_path, layer="clean_edges")
        clean_edges_gdf = clean_edges_gdf.to_crs(WORKING_CRS)
        # DUAL CLEAN NETWORK
        nx_clean = io.nx_from_generic_geopandas(clean_edges_gdf)
        # decompose
        nx_decomp = graphs.nx_decompose(nx_clean, 80)  # corresponding to 1min walking distance used in distances
        # cast to dual
        nx_dual = graphs.nx_to_dual(nx_decomp)
        # back to GDF
        nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(nx_dual)
        # mMark nodes as live if they're within the boundary
        nodes_gdf["live"] = nodes_gdf.geometry.within(bounds_row.geometry)
        # process centrality
        nodes_gdf = processors.process_centrality(nodes_gdf, network_structure)
        # POI
        places_gdf = gpd.read_file(overture_path, layer="places")
        places_gdf = places_gdf.to_crs(WORKING_CRS)
        # infrast
        infrast_gdf = gpd.read_file(overture_path, layer="infrastructure")
        infrast_gdf = infrast_gdf.to_crs(WORKING_CRS)
        nodes_gdf = processors.process_places(nodes_gdf, places_gdf, infrast_gdf, network_structure)
        # buildings
        bldgs_gdf = gpd.read_file(overture_path, layer="buildings")
        bldgs_gdf = bldgs_gdf.to_crs(WORKING_CRS)
        # blocks
        tools.validate_filepath(blocks_path)
        blocks_gdf = gpd.read_file(blocks_path, bbox=bounds_row.geometry.bounds)
        blocks_gdf = blocks_gdf.to_crs(WORKING_CRS)
        # process
        tools.validate_directory(hts_raster_data_dir)
        hts_path = Path(hts_raster_data_dir) / f"bldg_hts_{bounds_fid}.tif"
        if not hts_path.exists():
            logger.warning(
                "Missing building heights raster for bounds fid %s, continuing without height sampling", bounds_fid
            )
            hts_path = None
        nodes_gdf, bldgs_gdf, blocks_gdf = processors.process_blocks_buildings(
            nodes_gdf, bldgs_gdf, blocks_gdf, hts_path, network_structure
        )
        if not bldgs_gdf.empty:
            bldgs_gdf["bounds_fid"] = bounds_fid
            if overwrite is True:
                tools.remove_layer_if_exists(output_path, "buildings")
            bldgs_gdf.to_file(output_path, driver="GPKG", layer="buildings")
        if not blocks_gdf.empty:
            blocks_gdf["bounds_fid"] = bounds_fid
            if overwrite is True:
                tools.remove_layer_if_exists(output_path, "blocks")
            blocks_gdf.to_file(output_path, driver="GPKG", layer="blocks")
        # green spaces
        green_gdf = blocks_gdf[
            blocks_gdf["class_2018"].isin(
                [
                    "Arable land (annual crops)",
                    "Complex and mixed cultivation patterns",
                    "Forests",
                    "Green urban areas",
                    "Herbaceous vegetation associations (natural grassland, moors...)",
                    "Open spaces with little or no vegetation (beaches, dunes, bare rocks, glaciers)",
                    "Orchards at the fringe of urban classes",
                    "Pastures",
                    "Permanent crops (vineyards, fruit trees, olive groves)",
                    # "Sports and leisure facilities",
                    "Water",
                    "Wetlands",
                ]
            )
        ].copy()
        # trees - simplify
        tools.validate_filepath(trees_path)
        trees_gdf = gpd.read_file(trees_path, bbox=bounds_row.geometry.bounds)
        trees_gdf = trees_gdf.to_crs(WORKING_CRS)
        trees_gdf.geometry = trees_gdf.geometry.simplify(2.0)
        nodes_gdf = processors.process_green(nodes_gdf, green_gdf, trees_gdf, network_structure)
        # stats
        logger.info("Computing stats")
        # fetch stats
        tools.validate_filepath(stats_path)
        stats_gdf = gpd.read_file(stats_path, bbox=bounds_row.geometry.bounds)
        stats_gdf = stats_gdf.to_crs(WORKING_CRS)
        # prepare for interpolation
        stats_gdf = stats_gdf.rename(columns={col: col.lower() for col in stats_gdf.columns})
        cols = [
            "density",
            "t",
            "m",
            "f",
            "y_lt15",
            "y_1564",
            "y_ge65",
            "emp",
            "nat",
            "eu_oth",
            "oth",
            "same",
            "chg_in",
            "chg_out",
        ]
        # ratios
        stats_gdf["density"] = stats_gdf["t"] / stats_gdf["land_surface"]
        for col in cols:
            if col == "density" or col == "t" or "%" in col:
                continue
            col_perc = f"{col}_%"
            stats_gdf[col_perc] = stats_gdf[col] / stats_gdf["t"]
            cols.append(col_perc)  # guard against re-adding
        # interpolate
        grid_coords = np.array([(point.x, point.y) for point in stats_gdf.geometry.centroid])  # type: ignore
        target_coords = np.column_stack((nodes_gdf.x, nodes_gdf.y))  # type: ignore
        for col in tqdm(cols):
            grid_values = stats_gdf[col].values  # type: ignore
            # Filter out invalid values (NaN, inf, and common sentinel values like -9999, -9902, etc.)
            # Create mask for valid data points (finite, non-negative for counts/percentages)
            valid_mask = np.isfinite(grid_values) & (grid_values >= 0)  # 0 threshold catches common sentinel values
            if not np.any(valid_mask):
                logger.warning(f"No valid values for column {col}, skipping interpolation")
                nodes_gdf[col] = np.nan
                continue
            # Only use valid grid points for interpolation
            valid_grid_coords = grid_coords[valid_mask]
            valid_grid_values = grid_values[valid_mask]
            # use linear because cubic goes negative
            # fill_value=np.nan ensures out-of-bounds points get NaN rather than extrapolated values
            nodes_gdf[col] = griddata(
                valid_grid_coords, valid_grid_values, target_coords, method="linear", fill_value=np.nan
            )  # type: ignore
        # keep only live nodes within the boundary
        if not nodes_gdf.empty:
            nodes_gdf["bounds_fid"] = bounds_fid
            nodes_gdf_live = nodes_gdf.loc[nodes_gdf["live"]].copy()
            if not nodes_gdf_live.empty:
                if overwrite is True:
                    tools.remove_layer_if_exists(output_path, "streets")
                nodes_gdf_live.to_file(output_path, driver="GPKG", layer="streets")


if __name__ == "__main__":
    """
    python -m src.processing.generate_metrics \
        temp/datasets/boundaries.gpkg \
            temp/cities_data/overture \
                temp/datasets/blocks.gpkg \
                    temp/datasets/tree_canopies.gpkg \
                        temp/cities_data/heights \
                            temp/Eurostat_Census-GRID_2021_V2/ESTAT_Census_2021_V2.gpkg \
                               temp/cities_data/processed
    """
    if True:
        parser = argparse.ArgumentParser(description="Load overture networks.")
        parser.add_argument("bounds_in_path", type=str, help="Input data directory with boundary GPKG.")
        parser.add_argument("overture_data_dir", type=str, help="Input data directory for overture GPKG files.")
        parser.add_argument("blocks_path", type=str, help="Input data directory with Urban Atlas blocks GPKG.")
        parser.add_argument("trees_path", type=str, help="Input data directory with Urban Atlas tree canopy GPKG.")
        parser.add_argument(
            "hts_raster_data_dir", type=str, help="Input data directory with building height raster files."
        )
        parser.add_argument("stats_path", type=str, help="Input data directory with population stats GPKG.")
        parser.add_argument("processed_data_dir", type=str, help="Output data directory for metrics GPKG files.")
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Overwrite existing files (default: False)",
            default=False,
        )
        args = parser.parse_args()
        process_metrics(
            bounds_in_path=args.bounds_in_path,
            overture_data_dir=args.overture_data_dir,
            blocks_path=args.blocks_path,
            trees_path=args.trees_path,
            hts_raster_data_dir=args.hts_raster_data_dir,
            stats_path=args.stats_path,
            processed_data_dir=args.processed_data_dir,
            overwrite=args.overwrite,
        )
    else:
        process_metrics(
            bounds_in_path="temp/datasets/boundaries.gpkg",
            overture_data_dir="temp/cities_data/overture",
            blocks_path="temp/datasets/blocks.gpkg",
            trees_path="temp/datasets/tree_canopies.gpkg",
            hts_raster_data_dir="temp/cities_data/heights",
            stats_path="temp/Eurostat_Census-GRID_2021_V2/ESTAT_Census_2021_V2.gpkg",
            processed_data_dir="temp/cities_data/processed",
            overwrite=False,
        )
