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

        # 각 key(category semantic)들이 몇번 들어가 있는지 숫자세기
        key_count = {}

        if not building_data["features"]:
            continue
        else:
            for feature in building_data['features']:
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
            print(result)

            result_list.append(result)

    # 엔터 추가
    final_result = '\n'.join(result_list)

    nld_filename = f'{city_name}_dataset/NLD/' + city_name + "_NLD.txt"

    with open(nld_filename, 'w') as f:
        f.write(final_result)