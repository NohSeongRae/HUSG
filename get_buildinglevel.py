import json

def get_buildinglevel():
    with open('firenze_buildinglevel.geojson', 'r') as f:
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

    for i in range(len(num)):
        if num[i] != 0:
            category[i] = int(category[i]/num[i])
        else:
            category[i] = 0

    return category[0], category[1], category[2], category[3], category[4], category[5], category[6], category[7], category[8], category[9], category[10], category[11]

