import pickle
import numpy as np
import geopandas as gpd
import re
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, MultiPoint, Point, MultiPolygon, MultiLineString
from scipy.spatial import Voronoi
from shapely.ops import cascaded_union
import os

unit_length = 0.04

unit_coords_path = './dataset/husg_unit_coords.pkl'

#boundary_root_path = 'Z:/iiixr-drive/Projects/2023_City_Team/dublin_dataset/Normalized/Boundaries/'
#inference_image_root_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'inference_image')


def count_files_in_directory(directory_path):
    return sum([1 for entry in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, entry))])

def extract_number_from_string(s):
    return int(re.search(r'(\d+)(?=\.\w+$)', s).group())

def extract_numbers_from_boundaryfile(s):
    return int(re.search(r'(\d+)', s).group())

def plot(transformer_output, gt_output, unit_coord_seq, mask, test_idx):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # ax1: 예측 결과 시각화
    for idx in range(len(unit_coord_seq)):
        x = [unit_coord_seq[idx][0][0], unit_coord_seq[idx][1][0]]
        y = [unit_coord_seq[idx][0][1], unit_coord_seq[idx][1][1]]
        if mask[idx] == 0:
            break
        elif transformer_output[idx][1] >= 0.5:
            ax1.plot(x, y, 'r-')
        else:
            ax1.plot(x, y, 'black')
    ax1.set_title('Prediction')
    ax1.grid(True)

    # ax2: Ground Truth 시각화
    for idx in range(len(unit_coord_seq)):
        x = [unit_coord_seq[idx][0][0], unit_coord_seq[idx][1][0]]
        y = [unit_coord_seq[idx][0][1], unit_coord_seq[idx][1][1]]
        if mask[idx] == 0:
            break
        elif gt_output[idx] >= 0.5:
            ax2.plot(x, y, 'r-')
        else:
            ax2.plot(x, y, 'black')
    ax2.set_title('Ground Truth')
    ax2.grid(True)

    plt.tight_layout()

    directory = "./images"  # 변경: 저장 경로를 /mnt/data/ 아래로 지정
    if not os.path.exists(directory):
        os.makedirs(directory)

    save_path = os.path.join(directory, "building_exist_" + str(test_idx) + ".png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
