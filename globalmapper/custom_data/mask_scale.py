from shapely.geometry import Polygon
from shapely.affinity import rotate
from math import atan2
import pickle
import sys
sys.path.append('../')  # 상위 디렉토리를 모듈 검색 경로에 추가
from geo_utils import norm_block_to_horizonal, get_block_parameters
import shapely.affinity as sa


def compute_mask_scale(block_polygon):
    # 블록의 OBB 계산
    rect = block_polygon.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)

    # 두 가지 가능한 긴 변 계산
    side_1 = ((coords[0][0] - coords[1][0])**2 + (coords[0][1] - coords[1][1])**2)**0.5
    side_2 = ((coords[1][0] - coords[2][0])**2 + (coords[1][1] - coords[2][1])**2)**0.5

    # 가장 긴 변 선택
    longest_side = max(side_1, side_2)

    # 마스크의 너비 (또는 높이)
    mask_dimension = 64

    # 스케일링 계수 계산
    l = mask_dimension / longest_side

    print("longest_side", longest_side)

    return l


def scale_block_to_unit(block_polygon):
    # 블록의 OBB 계산
    rect = block_polygon.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)

    # 두 가지 가능한 긴 변 계산
    side_1 = ((coords[0][0] - coords[1][0]) ** 2 + (coords[0][1] - coords[1][1]) ** 2) ** 0.5
    side_2 = ((coords[1][0] - coords[2][0]) ** 2 + (coords[1][1] - coords[2][1]) ** 2) ** 0.5

    # 가장 긴 변 선택
    longest_side = max(side_1, side_2)

    # longest_side가 1보다 크면 스케일링
    if longest_side > 1:
        scaling_factor = 1 / longest_side
        block_polygon = sa.scale(block_polygon, xfact=scaling_factor, yfact=scaling_factor, origin=(0, 0))
        longest_side = 1

    return block_polygon, longest_side

def mask_to_polygons(mask):
    # 연결된 구성 요소 탐지
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    polygons = []
    for i in range(1, num_labels):  # 0은 배경이므로 제외
        component_mask = (labels == i).astype(np.uint8)

        # 외곽선 탐지
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if contour.shape[0] >= 3:  # 폴리곤을 형성하기 위해서는 최소 3개의 점이 필요합니다.
                contour = contour.squeeze(axis=1)  # (num_points, 1, 2) -> (num_points, 2)
                polygon = Polygon(contour)
                polygons.append(polygon)

    return polygons


# 예제 블록 정의#
# block = Polygon([(0, 0), (5, 1), (6, 6), (0, 5)])

raw_geo_path = "../globalmapper_dataset/raw_geo/0"

with open(raw_geo_path, 'rb') as file:
    content = pickle.load(file)
# print(content[0])

block = content[0]

azimuth, bbx = get_block_parameters(block)

block_polygon, longest_side = scale_block_to_unit(block)

print("new longestside", longest_side)

block = norm_block_to_horizonal([block], azimuth, bbx)

print(block)

# 스케일링 계수 출력
print(compute_mask_scale(block[0]))

