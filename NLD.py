import json
import os

# city_name = "firenze"

def NLD(city_name):
    city_name = city_name.capitalize()

    dir_path = f'{city_name}_dataset/Image/'
    files = os.listdir(dir_path)
    num_file = len(files)

    result_list = []

    for i in range(1, num_file):
        building_filename = f'{city_name}_dataset/Buildings/{city_name}_buildings{i}.geojson'
        with open(building_filename, "r", encoding='UTF-8') as file:
            building_data = json.load(file)

        key_count = {}

        if not building_data["features"]:
            continue
        else:
            for feature in building_data['features']:
                key = feature['properties']['key']
                if key in key_count:
                    key_count[key] += 1
                else:
                    key_count[key] = 1

            num_to_word = {1: 'one', 2: 'two', 3: 'three'}

            for key, value in key_count.items():
                if value in num_to_word:
                    key_count[key] = num_to_word[value]
                else:
                    key_count[key] = 'many'

            result = " and ".join([f"{count} {key.replace('_', ' ')}" for key, count in key_count.items()])
            # print(result)
            if result != '':
                result = f"{result} in {city_name}"
            print(result)

            result_list.append(result)

    final_result = '\n'.join(result_list)

    final_result = '\n'.join([line for line in final_result.split('\n') if line.strip() != ''])

    nld_filename = f'{city_name}_dataset/NLD/' + city_name + "_NLD.txt"

    with open(nld_filename, 'w') as f:
        f.write(final_result)