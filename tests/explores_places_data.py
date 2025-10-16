# %%
import geopandas as gpd

from src import tools

# %%
OVERTURE_SCHEMA = tools.generate_overture_schema()

# %%
OVERTURE_SCHEMA.keys()

# %%
places_gdf = gpd.read_file("temp/cities_data/overture/overture_0.gpkg", layer="places")
places_gdf.head()

# %%
places_gdf["major_lu_schema_class"].unique()

# %%
places_gdf = places_gdf[places_gdf["major_lu_schema_class"].isin(OVERTURE_SCHEMA.keys())]
places_gdf["major_lu_schema_class"].unique()

# %%
places_gdf["merged_cats"] = places_gdf["major_lu_schema_class"]
# merge eat_and_drink
places_gdf.loc[places_gdf["major_lu_schema_class"].isin(["restaurant", "bar", "cafe"]), "merged_cats"] = "eat_and_drink"
# merge services
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
places_gdf["merged_cats"].unique()

# %%
# rename some categories
places_gdf.loc[places_gdf["merged_cats"] == "public_service_and_government", "merged_cats"] = "public_services"
places_gdf.loc[places_gdf["merged_cats"] == "religious_organization", "merged_cats"] = "religious"

places_gdf["merged_cats"].unique()

# %%
