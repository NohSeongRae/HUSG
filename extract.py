import json

def extract(city_name):
    data_filepath = f"{city_name}_dataset/{city_name}_all_features.geojson"
    with open(data_filepath, "r", encoding="UTF-8") as file:
        data_geojson = json.load(file)

    point_features = {"type": "FeatureCollection", "features": []}
    polygon_features = {"type": "FeatureCollection", "features": []}

    for feature in data_geojson["features"]:
        geom_type = feature["geometry"]["type"]

        if geom_type == "Point":
            if feature["properties"].get("amenity"):
                point_features["features"].append(feature)
        elif geom_type == "Polygon":
            properties = feature["properties"]
            if any(key in properties for key in ["building", "shop", "amenity", "office", "tourism", "government", "military", "landuse", "healthcare", "leisure", "natural", "historic", "water", "waterway"]):
                polygon_features["features"].append(feature)

    point_data_filepath = f"{city_name}_dataset/{city_name}_point_data.geojson"
    with open(point_data_filepath, "w", encoding="UTF-8") as outfile:
        json.dump(point_features, outfile)

    polygon_data_filepath = f"{city_name}_dataset/{city_name}_polygon_data.geojson"
    with open(polygon_data_filepath, "w", encoding="UTF-8") as outfile:
        json.dump(polygon_features, outfile)

    print("Extract point / polygon data complete")
