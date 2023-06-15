import json
import filepath

building_filename = filepath.combined_filepath

with open(building_filename, "r", encoding='UTF8') as infile:
    whole_geojson_data = json.load(infile)

for i in range(len(whole_geojson_data['features'])):
    properties = whole_geojson_data['features'][i]['properties']
    # building
    if properties.get("building") != None:
        properties["key"] = "residence"
        if properties["building"] in ['civic']:
            properties["key"] = "government_office"

    # shop
    if properties.get("shop") != None:
        properties["key"] = "shop"
        if properties["shop"] in ['convenience', 'supermarket']:
            properties["key"] = "supermarket"
        if properties["shop"] in ['herbalist', 'nutrition_supplements']:
            properties["key"] = "alternative"

    # amenity
    if properties.get("amenity") != None:
        if properties["amenity"] == 'marketplace':
            properties["key"] = "supermarket"
        if properties["amenity"] in ['restaurant', 'fast_food', 'cafe', 'bar', 'pub']:
            properties["key"] = "restaurant"
        if properties["amenity"] in ['kindergarten']:
            properties["key"] = "kindergarten"
        if properties["amenity"] in ['school']:
            properties["key"] = "school"
        if properties["amenity"] in ['college']:
            properties["key"] = "college"
        if properties["amenity"] in ['university']:
            properties["key"] = "university"
        if properties["amenity"] in ['police']:
            properties["key"] = "police_station"
        if properties["amenity"] in ['fire_station']:
            properties["key"] = "fire_station"
        if properties["amenity"] in ['bank']:
            properties["key"] = "bank"
        if properties["amenity"] in ['bureau_de_change']:
            properties["key"] = "bureau_de_change"
        if properties["amenity"] in ['court_house', 'townhall']:
            properties["key"] = "government_office"
        if properties["amenity"] in ['embassy']:
            properties['key'] = 'embassy'
        if properties["amenity"] in ['post_office']:
            properties['key'] = 'post_office'
        if properties["amenity"] in ['doctors']:
            properties['key'] = 'clinic'
        if properties["amenity"] in ['dentist']:
            properties['key'] = 'clinic'
        if properties["amenity"] in ['clinic']:
            properties['key'] = 'clinic'
        if properties["amenity"] in ['hospital']:
            properties['key'] = 'hospital'
        if properties["amenity"] in ['pharmacy']:
            properties['key'] = 'pharmacy'
        if properties["amenity"] in ['grave_yard']:
            properties['key'] = 'cemetery'
        if properties["amenity"] in ['place_of_worship']:
            properties['key'] = 'place_of_worship'
        if properties['amenity'] in ['community_centre']:
            properties['key'] = 'community_centre'
        if properties['amenity'] in ['library']:
            properties['key'] = 'library'

    # office
    if properties.get("office") != None:
        if properties["office"] in ['government']:
            properties["key"] = 'government_office'

    # tourism
    if properties.get("tourism") != None:
        properties["key"] = "tourism"
        if properties["tourism"] in ['hotel', 'chalet', 'guest_house', 'hostel', 'motel']:
            properties["key"] = "accommodation"

    # government
    if properties.get("government") != None:
        properties["key"] = "government_office"

    # militray
    if properties.get("military") != None:
        properties["key"] = "military"

    # landuse
    if properties.get("landuse") != None:
        if properties["landuse"] in ['military']:
            properties["key"] = "military"
        if properties["landuse"] in ['cemetery']:
            properties["key"] = "cemetery"
        if properties["landuse"] in ['farmland', 'farmyard', 'greenhouse_horticulture']:
            properties["key"] = "agriculture"
        if properties["landuse"] in ['landfill']:
            properties["key"] = "solid_waste"
        if properties["landuse"] in ['forest']:
            properties["key"] = "forest"
        if properties["landuse"] in ['reservoir']:
            properties["key"] = "reservoir"

    # health_care
    if properties.get("healthcare") != None:
        if properties["healthcare"] in ['alternative']:
            properties["key"] = "alternative"

    # leisure
    if properties.get("leisure") != None:
        if properties["leisure"] in ['park']:
            properties["key"] = "park"
        if properties["leisure"] in ['stadium']:
            properties["key"] = "stadium"
        if properties["leisure"] in ['swimming_pool']:
            properties["key"] = "swimming_pool"
        if properties["leisure"] in ['pitch']:
            properties["key"] = "pitch"
        if properties["leisure"] in ['sport_centre']:
            properties["key"] = "sport_centre"

    # natural
    if properties.get("natural") != None:
        if properties["natural"] in ['water']:
            properties["key"] = "water_body"
        if properties["natural"] in ['grassland']:
            properties["key"] = "grassland"
        if properties["natural"] in ["wetland"]:
            properties["key"] = "wetland"
        if properties["natural"] in ["water"]:
            properties["key"] = "reservoir"

    # historic
    if properties.get("historic") != None:
        properties["key"] = "historic"

    # water
    if properties.get("water") != None:
        if properties["water"] in ["reservoir"]:
            properties["key"] = "reservoir"

    # waterway
    if properties.get("waterway") != None:
        properties["key"] = "waterway"

filtered_features = []
for feature in whole_geojson_data["features"]:
    if feature["properties"].get("building:levels") != None:
        building_level = feature["properties"]["building:levels"]
        key_value = feature["properties"].get("key")
        if key_value:
            feature["properties"] = {"key": key_value}
            feature["properties"]["building:levels"] = building_level
            filtered_features.append(feature)

    if feature["properties"].get("height") != None:
        height = feature["properties"]["height"]
        key_value = feature["properties"].get("key")
        if key_value:
            feature["properties"] = {"key": key_value}
            feature["properties"]["building:levels"] = height
            filtered_features.append(feature)

whole_geojson_data["features"] = filtered_features

with open(filepath.buildinglevel_filepath, "w") as f:
    json.dump(whole_geojson_data, f)


import json

def get_buildinglevel():
    with open(filepath.buildinglevel_filepath, 'r') as f:
        data = json.load(f)

    with open(filepath.category_filepath, 'r') as f:
        category_keys = json.load(f)

    category_sums = [0] * len(category_keys)
    category_counts = [0] * len(category_keys)

    for feature in data['features']:
        properties = feature['properties']
        if properties.get("building:levels") is not None:
            levels = properties['building:levels']
            keys = properties['key']
            if levels.isdigit():
                for i, (category, _) in enumerate(category_keys):
                    if keys in category:
                        category_sums[i] += int(levels)
                        category_counts[i] += 1
                        break

    category_avgs = [category_sums[i] // category_counts[i] if category_counts[i] != 0 else 0 for i in range(len(category_sums))]
    return category_avgs

if __name__=='__main__':
    avg = get_buildinglevel()
    print(avg)
