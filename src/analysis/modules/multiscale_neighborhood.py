"""Compute multi-scale population neighborhoods for grid cells."""

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree

from src import tools

logger = tools.get_logger(__name__)


def compute_neighborhood_populations(
    grid_gdf: gpd.GeoDataFrame,
    grid_spacing_m: float = 1000.0,
) -> gpd.GeoDataFrame:
    """Compute multi-scale population neighborhoods for each grid cell.

    Computes population aggregates at three scales:
    - local: single cell (1x1)
    - intermediate: 3x3 neighborhood (includes self + 8 neighbors)
    - large: 5x5 neighborhood (includes self + 24 neighbors)

    For grid cells organized in a regular grid, neighbors are identified by
    proximity. Assumes grid cells are approximately square with edge length
    ~grid_spacing_m.

    Parameters
    ----------
    grid_gdf
        GeoDataFrame with grid cells, must have 'population' column
    grid_spacing_m
        Approximate edge length of grid cells in meters

    Returns
    -------
    GeoDataFrame with added columns:
        - pop_local: population in cell
        - pop_intermediate: sum of 3x3 neighborhood
        - pop_large: sum of 5x5 neighborhood

    """
    grid_gdf = grid_gdf.copy()

    # Get grid cell centroids
    centroids = grid_gdf.geometry.centroid
    coords = np.array([[pt.x, pt.y] for pt in centroids])

    # Build spatial index for efficient neighbor finding
    tree = cKDTree(coords)

    # Define neighborhood radii
    # 3x3 neighborhood: diagonal distance ~1.5 * grid_spacing
    radius_intermediate = 1.5 * grid_spacing_m
    # 5x5 neighborhood: diagonal distance ~2.5 * grid_spacing
    radius_large = 2.5 * grid_spacing_m

    logger.info("Computing multi-scale population neighborhoods...")

    # Local population (just the cell itself)
    grid_gdf["pop_local"] = grid_gdf["population"].astype(float)

    # Intermediate neighborhood (3x3)
    pop_intermediate = []
    for centroid in coords:
        neighbors = tree.query_ball_point(centroid, radius_intermediate)
        pop_sum = grid_gdf.iloc[neighbors]["population"].astype(float).sum()
        pop_intermediate.append(pop_sum)
    grid_gdf["pop_intermediate"] = pop_intermediate

    # Large neighborhood (5x5)
    pop_large = []
    for centroid in coords:
        neighbors = tree.query_ball_point(centroid, radius_large)
        pop_sum = grid_gdf.iloc[neighbors]["population"].astype(float).sum()
        pop_large.append(pop_sum)
    grid_gdf["pop_large"] = pop_large

    logger.info(f"  Local pop: mean={grid_gdf['pop_local'].mean():.0f}, median={grid_gdf['pop_local'].median():.0f}")
    logger.info(
        f"  Intermediate pop: mean={grid_gdf['pop_intermediate'].mean():.0f}, "
        f"median={grid_gdf['pop_intermediate'].median():.0f}"
    )
    logger.info(f"  Large pop: mean={grid_gdf['pop_large'].mean():.0f}, median={grid_gdf['pop_large'].median():.0f}")

    return grid_gdf
