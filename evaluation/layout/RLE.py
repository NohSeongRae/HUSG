from PIL import Image
import numpy as np
from pycocotools import mask
import os
import json

def binary_mask_to_rle(binary_mask):
    # COCO API를 사용하여 바이너리 마스크를 RLE로 인코딩
    rle = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    return rle

def png_to_binary_mask(png_path):
    # PNG 이미지를 불러와서 numpy 배열로 변환
    image = Image.open(png_path)
    binary_mask = np.array(image)
    # 그레이스케일 이미지의 경우, 0보다 큰 값을 모두 1로 변환하여 바이너리 마스크를 생성
    binary_mask = (binary_mask > 0).astype(np.uint8)
    return binary_mask


def compute_bbox(binary_mask):
    # 값이 1인 픽셀의 인덱스를 찾습니다.
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # bounding box의 [x_min, y_min, width, height]를 계산합니다.
    return [x_min, y_min, x_max - x_min, y_max - y_min]

counts_list = []
size_list = []
area_list = []
image_id_list = []
bbox_list = []
id_list = []

def extract_indices(file_name):
    # 파일 이름에서 image_index와 building_index 추출
    parts = file_name.split('_')
    image_index = int(parts[1])
    building_index = int(parts[2].replace('buildingmask', '').replace('.png', ''))
    print(image_index, building_index)
    return image_index, building_index


def get_annotation(city_name, image_id, id):
    folder_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '1_evaluation', 'layoutvae',
                               'buildingmask', city_name)

    file_names = os.listdir(folder_path)

    sorted_file_names = sorted(file_names, key=extract_indices)

    for file_name in sorted_file_names:
        if os.path.isfile(os.path.join(folder_path, file_name)):
            print(sorted_file_names)
            binary_mask_path = os.path.join(folder_path, file_name)
            binary_mask = png_to_binary_mask(binary_mask_path)
            segmentation_pixel_count = np.sum(binary_mask)
            bbox = compute_bbox(binary_mask)
            rle = binary_mask_to_rle(binary_mask)

            counts_list.append(rle['counts'])
            size_list.append(rle['size'])
            area_list.append(segmentation_pixel_count)
            bbox_list.append(bbox)
            image_id_list.append(image_id)
            id_list.append(id)
            id += 1

    return image_id, id

get_annotation("atlanta", 0, 0)

annotations_data = []

print(size_list)

