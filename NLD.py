import json
import os
import re


def remove_duplicate_coordinates(features):
    seen_coordinates = set()
    new_features = []

    for feature in features:
        coordinates = feature['geometry']['coordinates']

        # Convert the list of coordinates to a string
        string_coordinates = json.dumps(coordinates)

        if string_coordinates not in seen_coordinates:
            seen_coordinates.add(string_coordinates)
            new_features.append(feature)

    return new_features


def NLD(city_name):
    city_name = city_name.capitalize()

    dir_path = "./2023_City_Team/" + f'{city_name}_dataset/Boundaries/'
    files = os.listdir(dir_path)
    filenum = len(files)

    result_list = []

    for i in range(1, filenum+1):
        building_filename = "./2023_City_Team/" + f'{city_name}_dataset/Buildings/{city_name}_buildings{i}.geojson'
        if os.path.exists(building_filename):
            with open(building_filename, "r", encoding='UTF-8') as file:
                building_data = json.load(file)

            # 각 key(category semantic)들이 몇번 들어가 있는지 숫자세기
            key_count = {}

            features = building_data['features']
            features = remove_duplicate_coordinates(features)

            if not features:
                continue
            else:
                for feature in features:
                    key = feature['properties']['key']
                    # 이미 추가했으면 1더하기
                    if key in key_count:
                        key_count[key] += 1
                    # dictionary에 없으면 해당 key값 추가
                    else:
                        key_count[key] = 1
                # 1은 one, 2는 two, ..
                num_to_word = {1: 'one', 2: 'two', 3: 'three'}

                for key, value in key_count.items():
                    # 1,2,3 중하나면 num_to_word의 value값 할당해주기
                    if value in num_to_word:
                        key_count[key] = num_to_word[value]
                    # 아니라면 many
                    else:
                        key_count[key] = 'many'
                # 문자열 생성 (one residence and ~~ )
                result = " and ".join([f"{count} {key.replace('_', ' ')}" for key, count in key_count.items()])

                # 뒤에 도시이름 추가
                if result != '':
                    result = f"{result} in {city_name}"

                result_list.append(result)

        # 엔터 추가
        final_result = '\n'.join(result_list)
        final_result = re.sub('\n+', '\n', final_result)

        nld_filename = "./2023_City_Team/" + f'{city_name}_dataset/NLD/' + city_name + "_NLD.txt"

        with open(nld_filename, 'w') as f:
            f.write(final_result)
