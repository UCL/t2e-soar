# %%
import geopandas as gpd

# %%
infrast_gdf = gpd.read_file("temp/cities_data/overture/overture_0.gpkg", layer="infrastructure")
infrast_gdf.head()

# %%
infrast_gdf["class"].unique()

# %%
# infrastructure
street_furn_keys = [
    "bench",
    "drinking_water",
    "fountain",
    "picnic_table",
    "plant",
    "planter",
    "post_box",
    "artwork",
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
