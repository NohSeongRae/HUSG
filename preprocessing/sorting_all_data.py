from tqdm import tqdm
import pickle
import os
import numpy as np

city_names = ["atlanta", "dallas", "houston", "lasvegas", "littlerock",
              "philadelphia", "phoenix", "portland", "richmond", "saintpaul",
              "sanfrancisco", "miami", "seattle", "boston", "providence",
              "neworleans", "denver", "pittsburgh", "washington"]

for city_name in city_names:
    print(city_name)
    # if all dataset
    # load_dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '2_transformer', 'new_dataset', 'train_dataset',
    #                              f'{city_name}')
    # save_dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '2_transformer', 'new_dataset', 'sorted_train_dataset',
    #                              f'{city_name}')
    # if fix dataset
    # load_dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '2_transformer', 'new_dataset', 'fix_node_feature_dataset',
    #                              f'{city_name}')
    # save_dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '2_transformer', 'new_dataset', 'sorted_fix_node_feature_dataset',
    #                              f'{city_name}')
    load_dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '7_CVAE', 'train_dataset_graph',
                                 f'{city_name}')
    save_dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '7_CVAE', 'sorted_train_dataset',
                                 f'{city_name}')
    load_dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '7_CVAE', 'train_dataset_mask',
                                 f'{city_name}')
    save_dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '7_CVAE', 'sorted_train_dataset',
                                 f'{city_name}')

    # 파일 이름을 정의합니다.
    file_names = ["boundary_filenames.pkl", "building_filenames.pkl", "building_polygons.pkl",
                  "building_semantics.pkl", "edge_indices.pkl",
                  "node_features.pkl", "unit_road_street_indices.pkl"]
    file_names = ["file_paths.pkl", "insidemask.pkl"]

    # 파일로부터 데이터를 불러옵니다.
    lists = []
    for file_name in tqdm(file_names):
        file_path = os.path.join(load_dir_path, file_name)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            lists.append(data)

    # 모든 리스트의 길이를 확인합니다.
    lengths = [len(lst) for lst in lists]

    # 길이가 동일한지 확인합니다.
    if len(set(lengths)) > 1:
        print("Lists have different lengths:", lengths)
    else:
        print("All lists have the same length.")

    # 4번째 리스트를 기준으로 정렬 순서를 얻습니다.
    sort_idx = 0
    extracted_numbers = []
    for file_path in lists[sort_idx]:
        split_path = file_path.split('/')
        last_element = split_path[-1]
        number = int(''.join(filter(str.isdigit, last_element)))
        extracted_numbers.append(number)
    sort_order = sorted(range(len(extracted_numbers)), key=lambda k: extracted_numbers[k])

    # 각 리스트를 동일한 순서로 정렬합니다.
    sorted_lists = [[lst[i] for i in sort_order] for lst in lists]

    # 결과를 다시 pickle 파일로 저장합니다.
    for i, file_name in enumerate(tqdm(file_names)):
        file_path = os.path.join(save_dir_path, file_name)
        # 폴더가 없다면 생성합니다.
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)
        with open(file_path, 'wb') as f:
            data = sorted_lists[i]
            pickle.dump(data, f)
