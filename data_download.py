import osmnx as ox
import json

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

    print("data download complete")