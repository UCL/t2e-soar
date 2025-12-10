"""Aggregate city-level statistics from boundaries, census, and Overture POI data."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

from src import landuse_categories, tools

logger = tools.get_logger(__name__)
WORKING_CRS = 3035


def aggregate_city_stats(
    bounds_path: str,
    overture_data_dir: str,
    census_path: str,
    output_path: str,
) -> None:
    """
    Aggregate city-level statistics including population, area, and POI counts.

    Args:
        bounds_path: Path to boundaries GPKG file
        overture_data_dir: Directory containing overture_{bounds_fid}.gpkg files
        census_path: Path to census GPKG file
        output_path: Output directory for city_stats.gpkg
    """
    tools.validate_filepath(bounds_path)
    tools.validate_directory(overture_data_dir)
    tools.validate_filepath(census_path)
    tools.validate_directory(output_path, create=True)

    # Load boundaries
    logger.info(f"Loading boundaries from {bounds_path}")
    bounds_gdf = gpd.read_file(bounds_path, layer="bounds")
    bounds_gdf = bounds_gdf.to_crs(WORKING_CRS)

    # Load census data
    logger.info(f"Loading census data from {census_path}")
    census_gdf = gpd.read_file(census_path)
    census_gdf = census_gdf.to_crs(WORKING_CRS)
    # Standardize column names to lowercase
    census_gdf = census_gdf.rename(columns={col: col.lower() for col in census_gdf.columns})

    city_stats = []

    # Process each boundary
    for bounds_fid, bounds_row in tqdm(bounds_gdf.iterrows(), total=len(bounds_gdf), desc="Processing cities"):
        # Initialize stats dict (bounds_fid used as index for merging)
        stats = {
            "bounds_fid": bounds_fid,
        }

        # Compute area in km²
        stats["area_km2"] = bounds_row.geometry.area / 1_000_000

        # Intersect with census to get population
        try:
            census_intersect = census_gdf[census_gdf.intersects(bounds_row.geometry)]
            # Sum total population (column 't')
            stats["population"] = census_intersect["t"].sum() if not census_intersect.empty else 0
        except Exception as e:
            logger.warning(f"Error computing population for bounds_fid {bounds_fid}: {e}")
            stats["population"] = None

        # Load Overture places data
        overture_path = Path(overture_data_dir) / f"overture_{bounds_fid}.gpkg"
        if not overture_path.exists():
            logger.warning(f"Missing overture file for bounds_fid {bounds_fid}: {overture_path}")
            # Set all POI counts to None
            for cat in landuse_categories.COMMON_LANDUSE_CATEGORIES:
                stats[f"{cat}_count"] = None
            city_stats.append(stats)
            continue

        try:
            # Load places layer
            places_gdf = gpd.read_file(overture_path, layer="places")
            places_gdf = places_gdf.to_crs(WORKING_CRS)

            # Apply category merging
            places_gdf = landuse_categories.merge_landuse_categories(places_gdf)

            # Count POIs by category
            category_counts = places_gdf["merged_cats"].value_counts()

            # Add counts for common categories
            for cat in landuse_categories.COMMON_LANDUSE_CATEGORIES:
                stats[f"{cat}_count"] = int(category_counts.get(cat, 0))

            # Also add all other categories that exist
            for cat in category_counts.index:
                if cat not in landuse_categories.COMMON_LANDUSE_CATEGORIES:
                    stats[f"{cat}_count"] = int(category_counts[cat])

        except Exception as e:
            logger.warning(f"Error processing places for bounds_fid {bounds_fid}: {e}")
            # Set all POI counts to None
            for cat in landuse_categories.COMMON_LANDUSE_CATEGORIES:
                stats[f"{cat}_count"] = None

        city_stats.append(stats)

    # Create DataFrame and merge with boundaries to create GeoDataFrame
    stats_df = pd.DataFrame(city_stats)

    # Merge stats with boundaries to create a GeoDataFrame
    # Set bounds_fid as index for the merge to avoid duplicate columns
    stats_df = stats_df.set_index("bounds_fid")
    stats_gdf = bounds_gdf.join(stats_df, how="left")

    # Save as GeoPackage
    output_file = Path(output_path) / "city_stats.gpkg"
    stats_gdf.to_file(output_file, driver="GPKG", layer="city_stats")
    logger.info(f"Saved city statistics with geometries to {output_file}")
    logger.info(f"Processed {len(stats_gdf)} cities")
    logger.info(f"Columns: {list(stats_gdf.columns)}")
