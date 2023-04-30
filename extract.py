import json

def extract(city_name):
    data_filepath = city_name + "_dataset/" + city_name + "_all_features.geojson"
    with open(data_filepath, "r", encoding='UTF-8') as file:
        data_geojson = json.load(file)

    point_features = {"type": "FeatureCollection", "features": []}
    polygon_features = {"type": "FeatureCollection", "features": []}

    for feature in data_geojson["features"]:
        geom_type = feature["geometry"]["type"]

        if geom_type == "Point":
            point_features["features"].append(feature)
        elif geom_type == "Polygon":
            polygon_features["features"].append(feature)

    point_data_filepath = city_name + "_dataset/" + city_name + "_point_data.geojson"
    with open(point_data_filepath, "w", encoding='UTF-8') as outfile:
        json.dump(point_features, outfile)

    polygon_data_filepath = city_name + "_dataset/" + city_name + "_polygon_data.geojson"
    with open(polygon_data_filepath, "w", encoding='UTF-8') as outfile:
        json.dump(polygon_features, outfile)

    print("extract point / polygon complete")