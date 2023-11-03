from tqdm import tqdm
import pickle
import os
import numpy as np

city_names = ["atlanta", "dallas", "houston", "lasvegas", "littlerock",
              "philadelphia", "phoenix", "portland", "richmond", "saintpaul",
              "sanfrancisco", "miami", "seattle", "boston", "providence",
              "neworleans", "denver", "pittsburgh", "washington"]

for city_name in city_names:
    # load_dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '2_transformer', 'new_dataset', 'train_dataset',
    #                              f'{city_name}')
    # save_dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '2_transformer', 'new_dataset', 'sorted_train_dataset',
    #                              f'{city_name}')
    load_dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '2_transformer', 'new_dataset', 'fix_node_feature_dataset',
                                 f'{city_name}')
    save_dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '2_transformer', 'new_dataset', 'sorted_fix_node_feature_dataset',
                                 f'{city_name}')

    # 파일 이름을 정의합니다.
    # file_names = ["adj_matrices.pkl", "boundary_filenames.pkl", "building_exist_sequences.pkl",
    #               "building_filenames.pkl", "building_polygons.pkl",
    #               "node_features.pkl", "street_index_sequences.pkl", "street_unit_position_datasets.pkl",
    #               "unit_coords_datasets.pkl", "unit_position_datasets.pkl"]
    file_names = ["building_filenames.pkl", "node_features.pkl"]

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
    sort_order = sorted(range(len(lists[sort_idx])), key=lambda k: lists[sort_idx][k])

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

    # 정렬된 4번째 리스트를 기준으로 정렬 여부 확인
    sorted_building_filenames = sorted_lists[sort_idx]
    is_sorted = all(sorted_building_filenames[i] <= sorted_building_filenames[i + 1] for i in range(len(sorted_building_filenames) - 1))

    # 각 리스트의 원소 순서가 동일한지 확인
    is_same_order = all(np.array_equal(lists[i][j], sorted_lists[i][j]) for i in range(len(file_names)) for j in range(len(lists[0])))

    print("Is the list sorted correctly?:", is_sorted)
    print("Are the elements in the same order in all lists?:", is_same_order)
