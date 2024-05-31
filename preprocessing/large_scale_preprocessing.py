import json
import os
from tqdm import tqdm
# 경계 상자 좌표 정의
bbox = {
    "left": -2.27406,
    "right": -2.2603,
    "top": 53.45674,
    "bottom": 53.45054
}


def is_within_bbox(coordinates, bbox):
    """폴리곤의 좌표가 경계 상자 내에 있는지 확인합니다."""
    for coord in coordinates[0]:  # 폴리곤의 좌표는 리스트 안의 리스트로 구성되어 있습니다.
        lon, lat = coord
        if not (bbox['left'] <= lon <= bbox['right'] and bbox['bottom'] <= lat <= bbox['top']):
            return False
    return True


def check_geojson_in_bbox(file_path, bbox):
    """GeoJSON 파일의 모든 피처가 경계 상자 내에 있는지 확인합니다."""
    with open(file_path, 'r') as file:
        geojson_content = json.load(file)

    for feature in geojson_content['features']:
        if not is_within_bbox(feature['geometry']['coordinates'], bbox):
            return False
    return True


def check_folder_geojson(folder_path, bbox):
    """폴더 내 모든 GeoJSON 파일을 검사하여 경계 상자 내에 있는지 확인합니다."""
    valid_files = []
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.geojson'):
            file_path = os.path.join(folder_path, filename)
            if check_geojson_in_bbox(file_path, bbox):
                valid_files.append(filename)
    return valid_files


# 검사할 폴더 경로
folder_path = 'Z:/iiixr-drive/Projects/2023_City_Team/manchester_dataset/Boundaries'  # 실제 폴더 경로로 변경 필요

# 폴더 내 모든 파일 검사
valid_files = check_folder_geojson(folder_path, bbox)

# 결과 출력
print(valid_files)
