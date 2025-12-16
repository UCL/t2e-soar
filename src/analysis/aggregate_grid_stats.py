"""Aggregate POI statistics at the 1km census grid level with buffered analysis."""

from pathlib import Path

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree

from src import tools
from src.landuse_categories import COMMON_LANDUSE_CATEGORIES, merge_landuse_categories

logger = tools.get_logger(__name__)


def compute_neighborhood_populations(
    grid_gdf: gpd.GeoDataFrame,
    grid_spacing_m: float = 1000.0,
    full_census_gdf: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """Compute multi-scale population neighborhoods for each grid cell.

    Computes population aggregates at three scales:
    - local: single cell (1x1)
    - intermediate: 4x4 neighborhood (includes self + ~15 neighbors)
    - large: 9x9 neighborhood (includes self + ~80 neighbors)

    For grid cells organized in a regular grid, neighbors are identified by
    proximity. Assumes grid cells are approximately square with edge length
    ~grid_spacing_m.

    Parameters
    ----------
    grid_gdf
        GeoDataFrame with grid cells, must have 'population' column
    grid_spacing_m
        Approximate edge length of grid cells in meters
    full_census_gdf
        Optional full census grid for neighbor lookups. If provided, neighbors
        are found from this GeoDataFrame (useful when grid_gdf is a subset,
        e.g., grids within city boundaries that need neighbors outside).
        If None, neighbors are found within grid_gdf itself.

    Returns
    -------
    GeoDataFrame with added columns:
        - pop_local: population in cell
        - pop_intermediate: sum of 3x3 neighborhood
        - pop_large: sum of 5x5 neighborhood

    """
    grid_gdf = grid_gdf.copy()

    # Use full census grid for lookups if provided, otherwise use input grid
    lookup_gdf = full_census_gdf if full_census_gdf is not None else grid_gdf

    # Build spatial index on lookup grid for efficient neighbor finding
    lookup_centroids = lookup_gdf.geometry.centroid
    lookup_coords = np.array([[pt.x, pt.y] for pt in lookup_centroids])
    tree = cKDTree(lookup_coords)

    # Get target grid centroids for querying
    target_centroids = grid_gdf.geometry.centroid
    target_coords = np.array([[pt.x, pt.y] for pt in target_centroids])

    # Define neighborhood radii
    # 4x4 neighborhood: diagonal distance ~2.0 * grid_spacing
    radius_intermediate = 2.0 * grid_spacing_m
    # 9x9 neighborhood: diagonal distance ~4.5 * grid_spacing
    radius_large = 4.5 * grid_spacing_m

    logger.info("Computing multi-scale population neighborhoods...")

    # Get population data from lookup grid
    pop_data = lookup_gdf["population"].astype(float).values

    # Local population (just the cell itself)
    grid_gdf["pop_local"] = grid_gdf["population"].astype(float)

    # Intermediate neighborhood (3x3)
    pop_intermediate = []
    for centroid in target_coords:
        neighbors = tree.query_ball_point(centroid, radius_intermediate)
        pop_sum = pop_data[neighbors].sum()
        pop_intermediate.append(pop_sum)
    grid_gdf["pop_intermediate"] = pop_intermediate

    # Large neighborhood (5x5)
    pop_large = []
    for centroid in target_coords:
        neighbors = tree.query_ball_point(centroid, radius_large)
        pop_sum = pop_data[neighbors].sum()
        pop_large.append(pop_sum)
    grid_gdf["pop_large"] = pop_large

    logger.info(f"  Local pop: mean={grid_gdf['pop_local'].mean():.0f}, median={grid_gdf['pop_local'].median():.0f}")
    logger.info(
        f"  Intermediate pop: mean={grid_gdf['pop_intermediate'].mean():.0f}, "
        f"median={grid_gdf['pop_intermediate'].median():.0f}"
    )
    logger.info(f"  Large pop: mean={grid_gdf['pop_large'].mean():.0f}, median={grid_gdf['pop_large'].median():.0f}")

    return grid_gdf


def aggregate_grid_stats(
    bounds_gdf: gpd.GeoDataFrame,
    census_gdf: gpd.GeoDataFrame,
    overture_data_dir: str,
) -> gpd.GeoDataFrame:
    """Aggregate POI counts at census grid level.

    Parameters
    ----------
    bounds_gdf
        GeoDataFrame with city boundaries (must have 'bounds_fid' column)
    census_gdf
        GeoDataFrame with census grid cells
    overture_data_dir
        Path to directory containing individual city POI files (e.g., overture_0.gpkg)

    Returns
    -------
    gpd.GeoDataFrame
        Grid cells with aggregated POI counts and population statistics

    Notes
    -----
    - Neighborhoods are computed on the FULL census grid to ensure complete spatial context
    - Filtering by city boundaries happens AFTER neighborhood computation
    - POI data is loaded per-city from individual files (overture_0.gpkg, overture_1.gpkg, etc.)
    - No minimum population threshold is applied (all grid cells are included)
    """

    logger.info("=" * 80)
    logger.info("GRID-LEVEL POI STATISTICS AGGREGATION")
    logger.info("=" * 80)

    logger.info(f"Processing {len(bounds_gdf)} city boundaries")
    logger.info(f"Total grid cells: {len(census_gdf)}")
    logger.info(f"POI data directory: {overture_data_dir}")

    # Ensure same CRS
    if bounds_gdf.crs != census_gdf.crs:
        logger.info(f"Reprojecting census grid from {census_gdf.crs} to {bounds_gdf.crs}")
        census_gdf = census_gdf.to_crs(bounds_gdf.crs)

    # Rename population column if needed (census grid may use different column names)
    if "T" in census_gdf.columns and "population" not in census_gdf.columns:
        logger.info("Renaming census population column 'T' to 'population'")
        census_gdf = census_gdf.rename(columns={"T": "population"})
    elif "population" not in census_gdf.columns:
        logger.error(f"Could not find population column. Available columns: {list(census_gdf.columns)}")
        raise ValueError("Census grid must have 'T' or 'population' column")

    # Filter grids: keep only cells completely contained within city boundaries
    logger.info("Filtering grid cells to those completely within city boundaries...")
    census_gdf["GRD_ID"] = census_gdf.index.astype(str)

    # Spatial join to find which grids are within which cities
    grids_in_cities = gpd.sjoin(
        census_gdf[["GRD_ID", "population", "geometry"]],
        bounds_gdf[["bounds_fid", "geometry"]],
        how="inner",
        predicate="within",  # Grid completely within city boundary
    )

    logger.info(f"  Found {len(grids_in_cities)} grid cells completely within city boundaries")

    # Remove duplicate grids (shouldn't happen with 'within', but just in case)
    grids_in_cities = grids_in_cities.drop_duplicates(subset=["GRD_ID"])

    # Compute multi-scale population neighborhoods using shared module
    # Uses full census grid for accurate neighbor lookups across boundaries
    logger.info("Computing multi-scale population neighborhoods...")
    grids_in_cities = compute_neighborhood_populations(
        grid_gdf=grids_in_cities,
        full_census_gdf=census_gdf,
        grid_spacing_m=1000.0,
    )
    logger.info(f"  Computed neighborhoods for {len(grids_in_cities)} boundary cells")

    # Initialize POI count columns for all categories
    for cat in COMMON_LANDUSE_CATEGORIES:
        grids_in_cities[f"{cat}_count"] = 0

    # Iterate through boundaries and load POI data per-city
    logger.info("Counting POIs within grid cells (iterating by city)...")
    overture_path = Path(overture_data_dir)
    city_count = 0

    for bounds_fid in grids_in_cities["bounds_fid"].unique():
        city_count += 1
        city_grids = grids_in_cities[grids_in_cities["bounds_fid"] == bounds_fid].copy()

        # Load POI data for this city
        places_file = overture_path / f"overture_{bounds_fid}.gpkg"

        if not places_file.exists():
            logger.warning(
                f"  [{city_count}/{len(grids_in_cities['bounds_fid'].unique())}] "
                f"City {bounds_fid}: No places file found, skipping"
            )
            continue

        try:
            places_gdf = gpd.read_file(places_file, layer="places")

            if places_gdf.crs != grids_in_cities.crs:
                places_gdf = places_gdf.to_crs(grids_in_cities.crs)

            # Merge granular categories into common land-use categories
            if len(places_gdf) > 0:
                places_gdf = merge_landuse_categories(places_gdf)

                # For each grid in this city, count POIs within grid cell
                for idx, grid_row in city_grids.iterrows():
                    grid_geom = grid_row["geometry"]

                    # Find POIs intersecting this grid cell
                    pois_in_grid = places_gdf[places_gdf.intersects(grid_geom)]

                    # Count by category
                    for cat in COMMON_LANDUSE_CATEGORIES:
                        cat_pois = pois_in_grid[pois_in_grid["merged_cats"] == cat]
                        grids_in_cities.loc[idx, f"{cat}_count"] = len(cat_pois)

            logger.info(
                f"  [{city_count}/{len(grids_in_cities['bounds_fid'].unique())}] "
                f"City {bounds_fid}: Processed {len(city_grids)} grids, "
                f"{len(places_gdf)} total POIs"
            )

        except Exception as e:
            logger.error(
                f"  [{city_count}/{len(grids_in_cities['bounds_fid'].unique())}] "
                f"City {bounds_fid}: Error processing - {e}"
            )
            continue

    # Drop unnecessary columns and keep only what we need
    cols_to_drop = ["index_right"]
    grids_in_cities = grids_in_cities.drop(columns=cols_to_drop, errors="ignore")

    # Summary statistics
    logger.info("\nGrid Statistics Summary:")
    logger.info(f"  Total grids: {len(grids_in_cities)}")
    logger.info(f"  Cities with grids: {grids_in_cities['bounds_fid'].nunique()}")
    logger.info(f"  Mean population per grid: {grids_in_cities['population'].mean():.0f}")
    logger.info(f"  Median population per grid: {grids_in_cities['population'].median():.0f}")

    logger.info("\nGrid aggregation complete!")

    return grids_in_cities
