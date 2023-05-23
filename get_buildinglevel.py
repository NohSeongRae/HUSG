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


def get_buildinglevel():
    with open(filepath.buildinglevel_filepath, 'r') as f:
        data = json.load(f)

    # category : commercial, education, emergency, financial, government, healthcare, landuse,
    #            natural, public, sport, water, residence

    category = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for feature in data['features']:
        properties = feature['properties']
        if properties.get("building:levels") != None:
            levels = properties['building:levels']
            keys = properties['key']
            if levels.isdigit():
                if keys in ['shop', 'supermarket', 'restaurant', 'tourism', 'accommodation']:
                    category[0] += int(levels)
                    num[0] += 1
                if keys in ['kindergarten', 'school', 'college', 'university']:
                    category[1] += int(levels)
                    num[1] += 1
                if keys in ['police_station', 'ambulance_station', 'fire_station']:
                    category[2]+= int(levels)
                    num[2] += 1
                if keys in ['bank', 'bureau_de_change']:
                    category[3] += int(levels)
                    num[3] += 1
                if keys in ['government_office', 'embassy', 'military', 'post_office']:
                    category[4]+= int(levels)
                    num[4] += 1
                if keys in ['doctor', 'dentist', 'clinic', 'hospital', 'pharmacy', 'alternative']:
                    category[5] += int(levels)
                    num[5] += 1
                if keys in ['park', 'cemetery', 'argriculture', 'solid_waste']:
                    category[6] += int(levels)
                    num[6] += 1
                if keys in ['forest', 'grassland']:
                    category[7] += int(levels)
                    num[7] += 1
                if keys in ['place_of_worship', 'community_centre', 'library', 'historic', 'toilet']:
                    category[8] += int(levels)
                    num[8] += 1
                if keys in ['stadium', 'swimming_pool', 'pitch', 'sport_centre']:
                    category[9] += int(levels)
                    num[9] += 1
                if keys in ['reservoir', 'waterway', 'coastline', 'water_body', 'wetland']:
                    category[10] += int(levels)
                    num[10] += 1
                if keys in ['residence']:
                    category[11] += int(levels)
                    num[11] += 1

    for j in range(len(num)):
        if num[j] != 0:
            category[j] = int(category[j]/num[j])
        else:
            category[j] = 0

    return category[0], category[1], category[2], category[3], category[4], category[5], category[6], category[7], category[8], category[9], category[10], category[11]



