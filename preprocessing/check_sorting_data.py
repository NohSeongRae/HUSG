from tqdm import tqdm
import pickle
import os
import numpy as np

city_name = 'atlanta'
first_dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '7_CVAE', 'sorted_train_dataset',
                                 f'{city_name}')
second_dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '7_CVAE', 'sorted_train_dataset',
                                 f'{city_name}')

# 파일 이름을 정의합니다.
file_names = ["building_filenames.pkl", "file_paths.pkl"]

# 파일로부터 데이터를 불러옵니다.
for file_name in tqdm(file_names):
    file_path = os.path.join(first_dir_path, file_name)
    with open(file_path, 'rb') as f:
        data_1 = pickle.load(f)

    file_path = os.path.join(second_dir_path, file_name)
    with open(file_path, 'rb') as f:
        data_2 = pickle.load(f)

    print(file_name, data_1[:2])
    print(file_name, data_2[:2])
    print('-------------------')
