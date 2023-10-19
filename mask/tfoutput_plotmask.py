import os
import json
import geopandas as gpd
import matplotlib.pyplot as plt
import sys
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from skimage.morphology import dilation, square
from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.ops import unary_union
from tqdm import tqdm
import numpy as np
import imageio
from tqdm import tqdm
from rtree import index

current_script_path = os.path.dirname(os.path.abspath(__file__))
husg_directory_path = os.path.dirname(current_script_path)
sys.path.append(husg_directory_path)


def get_square_bounds(polygon, padding_percentage=10):
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


def divide_polygon_into_grids(polygon, num_grids):
    """
    polygon을 grid로 나누기
    """
    minx, miny, maxx, maxy = polygon.bounds
    width = maxx - minx
    height = maxy - miny
    dx = width / num_grids
    dy = height / num_grids
    grids = []
    for i in range(num_grids):
        for j in range(num_grids):
            lower_left = (minx + i*dx, miny + j*dy)
            upper_right = (minx + (i+1)*dx, miny + (j+1)*dy)
            grid = Polygon([lower_left, (lower_left[0], upper_right[1]), upper_right, (upper_right[0], lower_left[1]), lower_left])
            if polygon.intersects(grid):
                grids.append(grid)
    return grids

def rectangle_polygon_from_lines(line1, line2):
    # 꼭지점 계산
    point1 = line1[0]
    point2 = line1[1]
    point3 = line2[1]
    point4 = [point2[0] + (point3[0] - point1[0]), point2[1] + (point3[1] - point1[1])]

    return Polygon([point1, point2, point4, point3])


def rotated_lines_90(p1, p2, unit_length, scale=5):
    v = p2 - p1
    R = np.array([[0, 1], [-1, 0]])
    v_rotated = np.dot(R, v)
    magnitude_v = np.linalg.norm(v_rotated)
    v_normalized = v_rotated / magnitude_v

    line_from_p1 = [p1, p1 + v_normalized * unit_length * scale]
    line_from_p2 = [p2, p2 + v_normalized * unit_length * scale]

    return np.array(line_from_p1), np.array(line_from_p2)


def optimized_assign_grids_to_linestrings(unit_coords_dataset, grids):
    # Create an R-tree index for linestrings
    idx = index.Index()
    for i, segment in enumerate(unit_coords_dataset):
        idx.insert(i, LineString(segment).bounds)

    linestring_assignments = {i: [] for i in range(len(unit_coords_dataset))}
    for grid in grids:
        center = grid.centroid
        nearest_index = list(idx.nearest(center.bounds, 1))[0]
        linestring_assignments[nearest_index].append(grid)

    return linestring_assignments



def tfoutput_plotmask(city_name, image_size, unit_coords_datasets, building_index_sequences, linewidth, num_grids, unit_length):
    for idx in tqdm(range(len(unit_coords_datasets))):
        unit_coords_dataset = unit_coords_datasets[idx][np.any(unit_coords_datasets[idx] != 0, axis=(1, 2))]

        building_exists_index = [exist_idx for exist_idx, val in enumerate(building_index_sequences[idx]) if val == 1]
        no_building_indices = [exist_idx for exist_idx, val in enumerate(building_index_sequences[idx]) if val == 0]

        unit_coords = [list(line.coords) if isinstance(line, LineString) else line.tolist() for line in unit_coords_dataset]
        unit_coords = np.array(unit_coords)

        rotated_lines = []
        for unit_road_idx, unit_road in enumerate(unit_coords):
            p1 = np.array(unit_road[0])
            p2 = np.array(unit_road[1])

            lines_from_p1, lines_from_p2 = rotated_lines_90(p1, p2, unit_length)
            rotated_lines.append([unit_road_idx, lines_from_p1.tolist(), lines_from_p2.tolist()])

        rectangle_polygons = [rectangle_polygon_from_lines([unit_coords[unit_idx][0], unit_coords[unit_idx][1]], rotated_lines[unit_idx][1]) for unit_idx in no_building_indices]

        coordinates = [segment[0] for segment in unit_coords_dataset]
        coordinates.append(unit_coords_dataset[-1][1])
        boundary_polygon = Polygon(coordinates)

        # Use the optimized function
        grids = divide_polygon_into_grids(boundary_polygon, num_grids)

        # Use the optimized function
        linestring_assignments = optimized_assign_grids_to_linestrings(unit_coords_dataset, grids)

        linestring_polygon_map = {key: unary_union(value) for key, value in linestring_assignments.items() if value}

        building_polygons = []
        for exist_idx in building_exists_index:
            linestring_area_polygon = linestring_polygon_map.get(exist_idx)
            if isinstance(linestring_area_polygon, Polygon):
                building_polygons.append(linestring_area_polygon)
            elif isinstance(linestring_area_polygon, MultiPolygon):
                building_polygons.extend([poly for poly in linestring_area_polygon.geoms])

        width, height = image_size, image_size
        left, bottom, right, top = get_square_bounds(boundary_polygon)
        transform = rasterio.transform.from_bounds(left, bottom, right, top, width, height)

        building_mask = geometry_mask(building_polygons, transform=transform, invert=True, out_shape=(height, width))
        rectangle_mask = geometry_mask(rectangle_polygons, transform=transform, invert=False, out_shape=(height, width))
        combined_mask = np.where(rectangle_mask == 0, 0, building_mask)
        scaled_mask = (combined_mask * 1).astype(np.uint8)

        tfoutput_plotmask_folderpath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '3_mask',
                                             f'{city_name}', 'tfoutput_plotmask')
        if not os.path.exists(tfoutput_plotmask_folderpath):
            os.makedirs(tfoutput_plotmask_folderpath)

        tfoutput_plotmask_filepath = os.path.join(tfoutput_plotmask_folderpath, f'{city_name}_{idx+1}.png')
        imageio.imsave(tfoutput_plotmask_filepath, scaled_mask)