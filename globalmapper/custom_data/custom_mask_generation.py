import sys
sys.path.append('../')
from geo_utils import norm_block_to_horizonal, get_block_parameters
from canonical_transform import get_polyskeleton_longest_path, modified_skel_to_medaxis, get_size_with_vector
import pickle
import os
import matplotlib.pyplot as plt
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import Polygon
import numpy as np
import imageio
import skgeom

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

def plot_polygon(polygon):
    x, y = polygon.exterior.xy
    plt.figure(figsize=(6, 6))
    plt.fill(x, y, alpha=0.5, fc='blue', ec='black')
    plt.title("Polygon Visualization")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)

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

globalmapper_raw_data_root  = "../globalmapper_dataset/raw_geo"
globalmapper_raw_data = os.path.join(globalmapper_raw_data_root, "0")

with open(globalmapper_raw_data, 'rb') as file:
    data = pickle.load(file)

raw_polygon = data[0]

norm_bldg_poly = []
norm_bldg_poly.append(raw_polygon)

# print(raw_polygon


norm_blk_poly = data[0]

exterior_polyline = list(norm_blk_poly.exterior.coords)[:-1]
exterior_polyline.reverse()
poly_list = []
for ix in range(len(exterior_polyline)):
    poly_list.append(exterior_polyline[ix])
sk_norm_blk_poly = skgeom.Polygon(poly_list)

# print(sk_norm_blk_poly)

skel = skgeom.skeleton.create_interior_straight_skeleton(sk_norm_blk_poly)
G, longest_skel = get_polyskeleton_longest_path(skel, sk_norm_blk_poly)


plot_polygon(norm_blk_poly)

longside, shortside, long_vec, short_vec, _, _ = get_size_with_vector(norm_bldg_poly[0].minimum_rotated_rectangle)



medaxis = modified_skel_to_medaxis(longest_skel, norm_blk_poly)
# print("medaxis ", medaxis)
medaxis_length = medaxis.length

#####

azimuth, bbx = get_block_parameters(raw_polygon)

block = norm_block_to_horizonal([raw_polygon], azimuth, bbx)

normalized_polygon = block[0]

min_rotated_rect = normalized_polygon.minimum_rotated_rectangle

insidemask(normalized_polygon, min_rotated_rect)

# plt.show()