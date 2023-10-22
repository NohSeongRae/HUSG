from PIL import Image
import numpy as np
from pycocotools import mask
import os
import json
from tqdm import tqdm

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

    # 만약 바이너리 마스크에 값이 1인 픽셀이 없다면, 기본 bbox 값을 반환
    if not np.any(rows) or not np.any(cols):
        return [0, 0, 0, 0]

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
file_name_list = []

def extract_indices(file_name):
    # 파일 이름에서 image_index와 building_index 추출
    parts = file_name.split('_')
    image_index = int(parts[1])
    building_index = int(parts[2].replace('buildingmask', '').replace('.png', ''))
    return image_index, building_index


def extract_and_remove_duplicates(file_names):
    processed_names = []
    result = []

    for file_name in file_names:
        # 파일 이름을 "_"로 나눕니다.
        parts = file_name.split('_')

        # 마지막 부분에서 "buildingmask" 뒤의 숫자를 제거합니다.
        new_name = '_'.join(parts[:-1]) + '_buildingmask.png'

        # 중복되지 않은 경우에만 결과 리스트에 추가합니다.
        if new_name not in processed_names:
            processed_names.append(new_name)
            result.append(new_name)

    return result


def get_annotation(city_name, image_id, b_id):
    folder_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '1_evaluation', 'layoutvae',
                               'individualbuildingmask', city_name)

    file_names = os.listdir(folder_path)
    sorted_file_names = sorted(file_names, key=extract_indices)

    # 초기 image_index 설정
    prev_image_index = -1

    for file_name in tqdm(sorted_file_names):
        current_image_index, _ = extract_indices(file_name)

        if current_image_index != prev_image_index:
            image_id += 1
            prev_image_index = current_image_index

        if os.path.isfile(os.path.join(folder_path, file_name)):
            binary_mask_path = os.path.join(folder_path, file_name)
            binary_mask = png_to_binary_mask(binary_mask_path)
            bbox = compute_bbox(binary_mask)
            if bbox != [0, 0, 0, 0]:
                segmentation_pixel_count = np.sum(binary_mask)
                rle = binary_mask_to_rle(binary_mask)

                counts_list.append(rle['counts'])
                size_list.append(rle['size'])
                area_list.append(segmentation_pixel_count)
                bbox_list.append(bbox)
                image_id_list.append(image_id)
                id_list.append(b_id)
                file_name_list.append(file_name)
                b_id += 1

    return image_id, b_id

# city_names = ["atlanta", "dallas", "houston", "lasvegas", "littlerock",
#               "philadelphia", "phoenix", "portland", "richmond", "saintpaul",
#               "sanfrancisco", "miami", "seattle", "boston", "providence",
#               "neworleans", "denver", "pittsburgh", "tampa", "washington"]

city_names = ["atlanta", "dallas"]

for city_name in city_names:
    if city_name == "atlanta":
        image_id, b_id = get_annotation(city_name, 0, 10000000)
    else:
        image_id, b_id = get_annotation(city_name, image_id, b_id)

annotations_data = []

for i in range(len(counts_list)):
    annotation = {
        "segmentation": {"counts": counts_list[i], "size": size_list[i]},
        "area": area_list[i],
        "iscrowd": 0,
        "image_id": image_id_list[i],
        "bbox": bbox_list[i],
        "category_id": 121,
        "id": id_list[i]
    }
    annotations_data.append(annotation)

categories_data = [
    {
        "id": 121,
        "name": "building",
        "supercategory": "building"
    }
]

image_file_name = extract_and_remove_duplicates(file_name_list)

images_data = []
for i in range(len(image_file_name)):
    image_data = {
        "id": str(i),
        "width": 256,
        "height": 256,
        "file_name": image_file_name[i]
    }
    images_data.append(image_data)

coco_format = {
    "images": images_data,
    "annotations": annotations_data,
    "categories": categories_data
}

with open('coco_format_data.json', 'w') as json_file:
    json.dump(coco_format, json_file, indent=4)