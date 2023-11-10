import numpy as np
from shapely.geometry import Polygon
import skgeom
from cal_iou import cal_iou
from canonical_transform import get_polyskeleton_longest_path, modified_skel_to_medaxis, warp_bldg_by_midaxis
from geo_utils import get_block_aspect_ratio, norm_block_to_horizonal, get_block_parameters
import networkx as nx
import rasterio
from rasterio.features import geometry_mask
import matplotlib.pyplot as plt
import os

globalmapper_custom_raw_data_root = os.path.join("Z:", "iiixr-drive", "Projects", "2023_City_Team", "1_evaluation", "globalmapper_custom_data", "raw_geo", "atlanta")
globalmapper_custom_raw_data = os.path.join(globalmapper_custom_raw_data_root, "9.pkl")

# globalmapper_custom_raw_data_root = os.path.join("../globalmapper_dataset/raw_geo")
# globalmapper_custom_raw_data = os.path.join(globalmapper_custom_raw_data_root, "0")

with open(globalmapper_custom_raw_data, 'rb') as file:
    raw_data = pickle.load(file)

def get_square_bounds(polygon, padding_percentage=0):
    # building data 전체를 geodataframe형태로 저장
    # gdf = gpd.read_file(geojson_path)

    # 그 전체 data를 감싸는 boundary 찾기
    bounds = polygon.bounds
    # data를 감싸는 사각형의 가로 세로
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    # 정사각형 만들기
    # square_size = max(width, height)
    square_size = max(width, height) * (1 + padding_percentage / 100)

    # 중심좌표 반환
    center_x = (bounds[0] + bounds[2]) / 2
    center_y = (bounds[1] + bounds[3]) / 2
    # width, height 중 더 값이 큰 것을 한 변의 길이로 하는 정사각형 생성
    square_coords = [
        (center_x - square_size / 2, center_y - square_size / 2),
        (center_x - square_size / 2, center_y + square_size / 2),
        (center_x + square_size / 2, center_y + square_size / 2),
        (center_x + square_size / 2, center_y - square_size / 2),
        (center_x - square_size / 2, center_y - square_size / 2)
    ]

    # left, upper, right, lower 값 추출
    left = square_coords[0][0]
    upper = square_coords[0][1]
    right = square_coords[2][0]
    lower = square_coords[2][1]

    return left, upper, right, lower

def plot_mask(mask):
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='gray')
    plt.title("Mask Visualization")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.colorbar()
    plt.show()


def insidemask(boundary_polygon, rectangle_polygon, image_size=64):
    boundary_line = boundary_polygon.boundary
    boundaries_list = [boundary_line]

    width, height = image_size, image_size
    # 이미지 경계 설정
    # left, bottom, right, top = get_square_bounds(rectangle_polygon)
    left, upper, right, lower = get_square_bounds(rectangle_polygon)

    rectangle_width = right - left
    rectangle_height = lower - upper

    transform = rasterio.transform.from_bounds(left, upper, right, lower, width, height)

    mask = geometry_mask([boundary_polygon], transform=transform, invert=True, out_shape=(height, width))

    scaled_mask = (mask * 1).astype(np.uint8)

    plot_mask(scaled_mask)

    return scaled_mask


raw_polygon = raw_data[0]
azimuth, bbx = get_block_parameters(raw_polygon)
block = norm_block_to_horizonal([raw_polygon], azimuth, bbx)
normalized_polygon = block[0]
min_rotated_rect = normalized_polygon.minimum_rotated_rectangle
insidemask(normalized_polygon, min_rotated_rect)