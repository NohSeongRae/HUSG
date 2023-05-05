import osmnx as ox
import json
import osmnx as ox
import geopandas as gpd
from shapely.geometry import LineString

def data_download(city_name, location):
    tags = {
        "amenity": True,
        "building": True,
        "craft": True,
        "emergency": True,
        "healthcare": True,
        "historic": True,
        "landuse": True,
        "leisure": True,
        "man_made": True,
        "military": True,
        "natural": True,
        "office": True,
        "place": True,
        "shop": True,
        "sport": True,
        "tourism": True,
        "water": True,
        "waterway": True
    }

    data = {"type": "FeatureCollection", "features": []}

    for tag in tags:
        gdf = ox.geometries_from_place(location, {tag: tags[tag]})
        geojson_data = json.loads(gdf.to_json())

        for feature in geojson_data["features"]:
            feature["properties"] = {k:v for k, v in feature["properties"].items() if v is not None}
            data["features"].append(feature)

    data_filepath = city_name + "_dataset/" + city_name + "_all_features.geojson"

    with open(data_filepath, 'w') as f:
        json.dump(data, f)

    """

    tags = '["highway"~"motorway|trunk|primary|secondary|tertiary|street_limited|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|living_street|residential"]'

    place_gdf = ox.geocode_to_gdf(location)

    expanded_boundary = place_gdf.buffer(0.1)

    graph = ox.graph_from_polygon(expanded_boundary.geometry.values[0], network_type="all", custom_filter=tags)

    gdf = ox.graph_to_gdfs(graph, nodes=False, edges=True)

    gdf["properties"] = gdf["highway"].apply(lambda x: {"highway": x})
    gdf = gpd.GeoDataFrame(gdf[["properties", "geometry"]], geometry="geometry")

    output_file = city_name + "_dataset/" + city_name + "_roads.geojson"
    gdf.to_file(output_file, driver="GeoJSON")

    """
    tags = '["highway"~"motorway|trunk|primary|secondary|tertiary|street_limited|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|living_street|residential"]'
    graph = ox.graph_from_place(location, network_type="all", custom_filter=tags)

    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)

    edges["geometry"] = edges["geometry"].apply(lambda x: LineString(x))

    gdf = gpd.GeoDataFrame(edges[["geometry"]], geometry="geometry")

    output_file = city_name + "_dataset/" + city_name + "_roads.geojson"
    gdf.to_file(output_file, driver="GeoJSON")


    print("data download complete")