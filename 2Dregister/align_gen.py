import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
import os
import pickle
import glob

city_name = "atlanta"
vertexnum = 64

# OBJ 파일로 메시를 저장하는 함수
def save_to_obj(filename, points, simplices):
    with open(filename, 'w') as f:
        for p in points:
            f.write("v {} {} {}\n".format(p[0], p[1], 0))  # z값은 0으로 설정
        for s in simplices:
            f.write("f {} {} {}\n".format(s[0]+1, s[1]+1, s[2]+1))

# pickle
def load_points_from_pickle(filename):
    df = pd.read_pickle(filename)
    return df.values

def load_points_from_csv(filename):
    df = pd.read_csv(filename)
    return df.values


def normalize_coordinates(coords):
    # Convert to numpy array for easier operations
    coords_array = np.array(coords)

    # Find the min and max for x and y coordinates
    x_min, y_min = np.min(coords_array, axis=0)
    x_max, y_max = np.max(coords_array, axis=0)

    # Normalize the coordinates
    coords_array[:, 0] = (coords_array[:, 0] - x_min) / (x_max - x_min)
    coords_array[:, 1] = (coords_array[:, 1] - y_min) / (y_max - y_min)

    # Convert back to list of lists
    normalized_coords = coords_array.tolist()

    return normalized_coords

square_root_path = os.path.join("Z:", "iiixr-drive", "Projects", "2023_City_Team", "5_ACAP", "ref_data")
normalized_pickle_path = os.path.join(square_root_path, 'normalized_square.pickle')

points_set1 = load_points_from_pickle(normalized_pickle_path)

# 첫 번째 점 세트에 대한 Delaunay 삼각분할 수행
tri = Delaunay(points_set1)

# 첫 번째 메시를 OBJ 파일로 저장
square_obj_path = os.path.join(square_root_path, f'square_{vertexnum}.obj')

save_to_obj(square_obj_path, points_set1, tri.simplices)

building_pickle_path = "64_point_polygon_exteriors.pkl"

with open(building_pickle_path, 'rb') as file:
    building_data = pickle.load(file)

for idx, buildings in enumerate(building_data):
    folder_path = os.path.join("Z:", "iiixr-drive", "Projects", "2023_City_Team", '5_ACAP', f"buildingobj_{vertexnum}", city_name, f'{city_name}_{idx+1}')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for i, building in enumerate(buildings):
        try:
            normalized_building = normalize_coordinates(building)
            polygon_path = os.path.join(folder_path, f'{city_name}_{idx+1}_{i+1}.obj')
            # save_path = f'/mnt/c/Users/82103/Desktop/get_dr_py/mesh/milan_obj{idx}.obj'
            save_to_obj(polygon_path, normalized_building, tri.simplices)
            print(f"Saved {polygon_path}")  # 저장한 파일 경로 출력
        except Exception as e:
            print(f"Error processing : {e}")  # 오류 발생 시 출력
