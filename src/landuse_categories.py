"""Shared land-use category merging logic for Overture POI data."""

import geopandas as gpd

from src import tools

logger = tools.get_logger(__name__)
OVERTURE_SCHEMA = tools.generate_overture_schema()  # type: ignore


def merge_landuse_categories(places_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Apply standardized land-use category merging to Overture places data.

    Creates a 'merged_cats' column with consolidated categories:
    - eat_and_drink: restaurant, bar, cafe
    - business_and_services: automotive, beauty_and_spa, pets, real_estate, travel,
      home_service, financial_service, professional_services, business_to_business,
      private_establishments_and_corporates, mass_media
    - public_services: renamed from public_service_and_government
    - religious: renamed from religious_organization
    - Other categories remain unchanged

    Args:
        places_gdf: GeoDataFrame with 'major_lu_schema_class' column

    Returns:
        Modified GeoDataFrame with 'merged_cats' column added
    """
    # filter to valid schema classes
    valid_schema_mask = places_gdf["major_lu_schema_class"].isin(list(OVERTURE_SCHEMA.keys()))
    places_gdf = places_gdf.loc[valid_schema_mask].copy()

    # create merged categories column
    places_gdf["merged_cats"] = places_gdf["major_lu_schema_class"]

    # merge eat_and_drink
    places_gdf.loc[places_gdf["major_lu_schema_class"].isin(["restaurant", "bar", "cafe"]), "merged_cats"] = (
        "eat_and_drink"
    )

    # merge business_and_services
    places_gdf.loc[
        places_gdf["major_lu_schema_class"].isin(
            [
                "automotive",
                "beauty_and_spa",
                "pets",
                "real_estate",
                "travel",
                "home_service",
                "financial_service",
                "private_establishments_and_corporates",
                "business_to_business",
                "professional_services",
                "mass_media",
            ]
        ),
        "merged_cats",
    ] = "business_and_services"

    # rename some categories
    places_gdf.loc[places_gdf["merged_cats"] == "public_service_and_government", "merged_cats"] = "public_services"
    places_gdf.loc[places_gdf["merged_cats"] == "religious_organization", "merged_cats"] = "religious"

    return places_gdf


# Define common land-use categories that should be present in most cities
COMMON_LANDUSE_CATEGORIES = [
    "business_and_services",
    "active_life",
    "arts_and_entertainment",
    "public_services",
    "retail",
    "health_and_medical",
    "eat_and_drink",
    "education",
    "attractions_and_activities",
    "religious",
    "accommodation",
]
