from tqdm import tqdm
import pickle
import os
import numpy as np

city_names = ["atlanta"]

for city_name in city_names:
    building_polygons_dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '2_transformer',
                                              'new_dataset', 'sorted_train_dataset',  f'{city_name}', 'building_polygons.pkl')
    building_filename_dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '2_transformer',
                                              'new_dataset', 'sorted_train_dataset', f'{city_name}', 'building_filenames.pkl')

    with open(building_polygons_dir_path, 'rb') as f:
        building_polygons = pickle.load(f)
    with open(building_filename_dir_path, 'rb') as f:
        building_filenames = pickle.load(f)
    for building in building_polygons:
        print(len(building))
    for name in building_filenames:
        print(name)
