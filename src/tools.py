""" """

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any, cast

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import geometry, ops
from shapely.ops import transform

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def get_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(level=log_level)
    return logging.getLogger(name)


logger = get_logger(__name__)


def validate_filepath(path: str) -> str:
    """ """
    if not Path(path).exists():
        raise ValueError(f"Path does not exist: {path}")
    return path


def validate_directory(path: str, create: bool = False) -> str:
    """ """
    # handle if path is a file
    if Path(path).is_file() or Path(path).suffix != "":
        path = str(Path(path).parent)
    # handle if path is not a dir
    if not Path(path).is_dir():
        if create:
            Path(path).mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError(f"Directory does not exist: {path}")
    return path


def convert_ndarrays(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return convert_ndarrays(obj.tolist())
    if isinstance(obj, list | tuple):
        return [convert_ndarrays(item) for item in obj]
    if isinstance(obj, dict):
        return {key: convert_ndarrays(value) for key, value in obj.items()}
    if obj is None or obj == "":
        return None
    if isinstance(obj, str | int | float):
        return obj
    raise ValueError(f"Unhandled type when converting: {type(obj).__name__}")


def col_to_json(obj: Any) -> str | None:
    """Extracts JSON from a geoparquet / geopandas column"""
    if obj is None or (isinstance(obj, str) and obj == ""):
        return "null"
    obj = convert_ndarrays(obj)
    return json.dumps(obj)


Connector = tuple[str, geometry.Point]


def split_street_segment(
    line_string: geometry.LineString, connector_infos: list[Connector]
) -> list[tuple[geometry.LineString, Connector, Connector]]:
    """ """
    # overture segments can span multiple intersections
    # sort through and split until pairings are ready for insertion to the graph
    node_segment_pairs: list[tuple[geometry.LineString, Connector, Connector]] = []
    node_segment_lots: list[tuple[geometry.LineString, list[Connector]]] = [(line_string, connector_infos)]
    # start iterating
    while node_segment_lots:
        old_line_string, old_connectors = node_segment_lots.pop()
        # filter down connectors
        new_connectors: list[tuple[str, geometry.Point]] = []
        # if the point doesn't touch the line, discard
        for _fid, _point in old_connectors:
            if _point.distance(old_line_string) > 0:
                continue
            new_connectors.append((_fid, _point))
        # if only two connectors
        if len(new_connectors) == 2:
            node_segment_pairs.append((old_line_string, new_connectors[0], new_connectors[1]))
            continue
        # look for splits
        for _fid, _point in new_connectors:
            splits = ops.split(old_line_string, _point)
            # continue if an endpoint
            if len(splits.geoms) == 1:
                continue
            # otherwise unpack
            line_string_a, line_string_b = splits.geoms
            # otherwise split into two bundles and reset
            node_segment_lots.append((cast(geometry.LineString, line_string_a), new_connectors))
            node_segment_lots.append((cast(geometry.LineString, line_string_b), new_connectors))
            break
    return node_segment_pairs


def generate_graph(
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    drop_road_types: list[str] | None = None,
) -> nx.MultiGraph:
    """ """
    if drop_road_types is None:
        drop_road_types = []
    # create graph
    multigraph = nx.MultiGraph()
    # filter by boundary and build nx
    # dedupe nodes
    node_map = {}
    for node_row in nodes_gdf.itertuples():
        # catch duplicates in case of overture dupes by xy or database dupes
        x = node_row.geom.x  # type: ignore
        y = node_row.geom.y  # type: ignore
        xy_key = f"{x}-{y}"
        if xy_key not in node_map:
            node_map[xy_key] = node_row.Index
        # merged key
        merged_key = node_map[xy_key]
        # only insert if new
        if not multigraph.has_node(node_row.Index):
            multigraph.add_node(
                merged_key,
                x=x,
                y=y,
            )
    dropped_road_types = set()
    kept_road_types = set()
    for edge_idx, edges_data in edges_gdf.iterrows():
        road_class = edges_data["class"]
        if road_class in drop_road_types:
            dropped_road_types.add(road_class)
            continue
        kept_road_types.add(road_class)
        uniq_fids = set()
        connector_fids: list[str] = [connector["connector_id"] for connector in edges_data["connectors"]]
        connector_infos: list[tuple[str, geometry.Point]] = []
        missing_connectors = False
        for connector_fid in connector_fids:
            # skip malformed edges - this happens at boundary thresholds with missing nodes in relation to edges
            if connector_fid not in multigraph:
                missing_connectors = True
                break
            # deduplicate
            x, y = multigraph.nodes[connector_fid]["x"], multigraph.nodes[connector_fid]["y"]
            xy_key = f"{x}-{y}"
            merged_key = node_map[xy_key]
            if merged_key in uniq_fids:
                continue
            uniq_fids.add(merged_key)
            # track
            connector_point = geometry.Point(x, y)
            connector_infos.append((merged_key, connector_point))
        if missing_connectors is True:
            continue
        if len(connector_infos) < 2:
            # logger.warning("Only one connector pair for edge")
            continue
        # extract levels, names, routes, highways
        # do this once instead of for each new split segment
        levels = set([])
        if edges_data["level_rules"] is not None:
            for level_info in edges_data["level_rules"]:
                levels.add(level_info["value"])
        names = []  # takes list form for nx
        if edges_data["names"] is not None and "primary" in edges_data["names"]:
            names.append(edges_data["names"]["primary"])
        routes = set([])
        if edges_data["routes"] is not None:
            for routes_info in edges_data["routes"]:
                if "ref" in routes_info:
                    routes.add(routes_info["ref"])
        is_tunnel = False
        is_bridge = False
        if edges_data["road_flags"] is not None:
            for flags_info in edges_data["road_flags"]:
                if "is_tunnel" in flags_info["values"]:
                    is_tunnel = True
                if "is_bridge" in flags_info["values"]:
                    is_bridge = True
        highways = []  # takes list form for nx
        if road_class is not None and road_class not in ["unknown"]:
            highways.append(road_class)
        # split segments and build
        street_segs = split_street_segment(edges_data.geom, connector_infos)
        for seg_geom, node_info_a, node_info_b in street_segs:
            if not node_info_a[1].touches(seg_geom) or not node_info_b[1].touches(seg_geom):
                raise ValueError(
                    "Edge and endpoint connector are not touching. "
                    f"See connectors: {node_info_a[0]} and {node_info_b[0]}"
                )
            # don't add duplicates
            dupe = False
            if multigraph.has_edge(node_info_a[0], node_info_b[0]):
                edges = multigraph[node_info_a[0]][node_info_b[0]]
                for _edge_idx, edge_val in dict(edges).items():
                    if edge_val["geom"].buffer(1).contains(seg_geom):
                        dupe = True
                        break
            if dupe is False:
                multigraph.add_edge(
                    node_info_a[0],
                    node_info_b[0],
                    edge_idx=edge_idx,
                    geom=seg_geom,
                    levels=list(levels),
                    names=names,
                    routes=list(routes),
                    highways=highways,
                    is_bridge=is_bridge,
                    is_tunnel=is_tunnel,
                )

    return multigraph


def generate_overture_schema() -> dict[str, list[str]]:
    """ """
    logger.info("Preparing Overture schema")
    overture_csv_file_path = "./src/raw_landuse_schema.csv"
    schema = {
        # "eat_and_drink": [], - don't use because places overriden by more specific categories
        "restaurant": [],
        "bar": [],
        "cafe": [],
        "accommodation": [],
        "automotive": [],
        "arts_and_entertainment": [],
        "attractions_and_activities": [],
        "active_life": [],
        "beauty_and_spa": [],
        "education": [],
        "financial_service": [],
        "private_establishments_and_corporates": [],
        "retail": [],
        "health_and_medical": [],
        "pets": [],
        "business_to_business": [],
        "public_service_and_government": [],
        "religious_organization": [],
        "real_estate": [],
        "travel": [],
        "mass_media": [],
        "home_service": [],
        "professional_services": [],
        # "structure_and_geography": [],
    }
    for category, _list_val in schema.items():
        with open(overture_csv_file_path) as schema_csv:
            logger.info(f"Gathering category: {category}")
            for line in schema_csv:
                # remove header line
                if "Overture Taxonomy" in line:
                    continue
                splits = line.split(";")
                if "[" not in splits[1]:
                    logger.info(f"Skipping line {line}")
                    continue
                cats = splits[1].strip("\n[]")
                cats = cats.split(",")
                if category in cats:
                    schema[category].append(splits[0])
    return schema


def bounds_fid_type(value):
    if value == "all":
        return value
    try:
        return [int(value)]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"{value} is not a valid bounds_fid. It must be an integer or 'all'.") from e


def reproject_geometry(geom, from_crs, to_crs):
    """ """
    transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    reprojected_geom = transform(transformer.transform, geom)

    return reprojected_geom
