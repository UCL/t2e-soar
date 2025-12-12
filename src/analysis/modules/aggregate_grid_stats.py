"""Aggregate POI statistics at the 1km census grid level with buffered analysis."""

from pathlib import Path

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

from src import landuse_categories, tools

logger = tools.get_logger(__name__)


def aggregate_grid_stats(
    bounds_path: str,
    overture_data_dir: str,
    census_path: str,
    output_dir: str,
) -> None:
    """Aggregate POI counts at census grid level.

    Parameters
    ----------
    bounds_path
        Path to city boundaries GeoPackage
    overture_data_dir
        Directory containing Overture places data (places.gpkg per city)
    census_path
        Path to Eurostat census grid GeoPackage
    output_dir
        Directory for output files

    Notes
    -----
    - Neighborhoods are computed on the FULL census grid to ensure complete spatial context
    - Filtering by city boundaries happens AFTER neighborhood computation
    - No minimum population threshold is applied (all grid cells are included)

    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("GRID-LEVEL POI STATISTICS AGGREGATION")
    logger.info("=" * 80)

    # Load city boundaries
    logger.info(f"Loading city boundaries from {bounds_path}")
    bounds_gdf = gpd.read_file(bounds_path)
    logger.info(f"  Loaded {len(bounds_gdf)} city boundaries")

    # Load census grid
    logger.info(f"Loading census grid from {census_path}")
    census_gdf = gpd.read_file(census_path)
    logger.info(f"  Loaded {len(census_gdf)} census grid cells")
    logger.info(f"  Columns: {list(census_gdf.columns)}")

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

    # Compute multi-scale population neighborhoods ONLY for boundary cells
    # This is efficient: only ~5-10% of cells need neighbor computation
    logger.info("Computing multi-scale population neighborhoods for boundary cells...")

    # Build spatial index on full census grid for fast neighbor lookups
    centroids = census_gdf.geometry.centroid
    coords = np.array([[pt.coords[0][0], pt.coords[0][1]] for pt in centroids])
    tree = cKDTree(coords)

    # Define neighborhood radii (same as multiscale_neighborhood.py)
    radius_intermediate = 1500.0  # 1.5 * 1000m grid spacing
    radius_large = 4500.0  # 4.5 * 1000m grid spacing

    # Initialize arrays for population columns
    pop_local_values = []
    pop_int_values = []
    pop_large_values = []

    # For each boundary grid cell, find neighbors and aggregate population
    pop_data = census_gdf["population"].values
    grid_geoms = grids_in_cities.geometry.values
    grid_pops = grids_in_cities["population"].values

    for geom, pop_local in tqdm(zip(grid_geoms, grid_pops, strict=True), total=len(grid_geoms)):
        grid_centroid = geom.centroid
        coord = [[grid_centroid.x, grid_centroid.y]]

        # Local population already have from grids_in_cities
        pop_local_values.append(pop_local)

        # Find intermediate neighbors (3x3)
        neighbors_int = tree.query_ball_point(coord, r=radius_intermediate)[0]
        pop_int = float(pop_data[neighbors_int].sum())
        pop_int_values.append(pop_int)

        # Find large neighbors (9x9)
        neighbors_large = tree.query_ball_point(coord, r=radius_large)[0]
        pop_large = float(pop_data[neighbors_large].sum())
        pop_large_values.append(pop_large)

    # Assign computed values to dataframe
    grids_in_cities = grids_in_cities.copy()
    grids_in_cities["pop_local"] = pop_local_values
    grids_in_cities["pop_intermediate"] = pop_int_values
    grids_in_cities["pop_large"] = pop_large_values

    logger.info(f"  Computed neighborhoods for {len(grids_in_cities)} boundary cells")

    # Initialize POI count columns
    for cat in landuse_categories.COMMON_LANDUSE_CATEGORIES:
        grids_in_cities[f"{cat}_count"] = 0

    # Process each city and count POIs in grid cells
    logger.info("Counting POIs within grid cells for each city...")

    overture_dir = Path(overture_data_dir)
    city_count = 0

    for bounds_fid in grids_in_cities["bounds_fid"].unique():
        city_count += 1
        city_grids = grids_in_cities[grids_in_cities["bounds_fid"] == bounds_fid].copy()

        # Load POI data for this city
        places_file = overture_dir / f"overture_{bounds_fid}.gpkg"

        if not places_file.exists():
            logger.warning(
                f"  [{city_count}/{len(grids_in_cities['bounds_fid'].unique())}] City {bounds_fid}: No places file found, skipping"
            )
            continue

        try:
            places_gdf = gpd.read_file(places_file, layer="places")
            if places_gdf.crs != grids_in_cities.crs:
                places_gdf = places_gdf.to_crs(grids_in_cities.crs)

            # Merge granular categories into common land-use categories
            places_gdf = landuse_categories.merge_landuse_categories(places_gdf)

            # For each grid in this city, count POIs within grid cell
            for idx, grid_row in city_grids.iterrows():
                grid_geom = grid_row["geometry"]

                # Find POIs intersecting this grid cell
                pois_in_grid = places_gdf[places_gdf.intersects(grid_geom)]

                # Count by category
                for cat in landuse_categories.COMMON_LANDUSE_CATEGORIES:
                    cat_pois = pois_in_grid[pois_in_grid["merged_cats"] == cat]
                    grids_in_cities.loc[idx, f"{cat}_count"] = len(cat_pois)

            logger.info(
                f"  [{city_count}/{len(grids_in_cities['bounds_fid'].unique())}] "
                f"City {bounds_fid}: Processed {len(city_grids)} grids, "
                f"{len(places_gdf)} total POIs"
            )

        except Exception as e:
            logger.error(
                f"  [{city_count}/{len(grids_in_cities['bounds_fid'].unique())}] City {bounds_fid}: Error processing - {e}"
            )
            continue

    # Drop unnecessary columns and keep only what we need
    # (index_right from spatial join, any extra columns)
    cols_to_drop = ["index_right"]
    grids_in_cities = grids_in_cities.drop(columns=cols_to_drop, errors="ignore")

    # Save results
    output_file = output_path / "grid_stats.gpkg"
    grids_in_cities.to_file(output_file, driver="GPKG", layer="grid_stats")
    logger.info(f"Saved grid statistics to {output_file}")

    # Summary statistics
    logger.info("\nGrid Statistics Summary:")
    logger.info(f"  Total grids: {len(grids_in_cities)}")
    logger.info(f"  Cities with grids: {grids_in_cities['bounds_fid'].nunique()}")
    logger.info(f"  Mean population per grid: {grids_in_cities['population'].mean():.0f}")
    logger.info(f"  Median population per grid: {grids_in_cities['population'].median():.0f}")

    logger.info("\nPOI counts by category:")
    for cat in landuse_categories.COMMON_LANDUSE_CATEGORIES:
        col = f"{cat}_count"
        total_pois = grids_in_cities[col].sum()
        grids_with_pois = (grids_in_cities[col] > 0).sum()
        logger.info(f"  {cat}: {int(total_pois)} POIs across {grids_with_pois} grids")

    logger.info("\nGrid aggregation complete!")
