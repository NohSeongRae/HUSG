import pickle
import numpy as np
import geopandas as gpd
import re
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, MultiPoint, Point, MultiPolygon, MultiLineString
from scipy.spatial import Voronoi
from shapely.ops import cascaded_union
import os

city_name = "dublin"
unit_length = 0.04

unit_coords_path = './dataset/husg_unit_coords.pkl'
npz_path = './dataset/husg_transformer_dataset.npz'

boundary_root_path = 'Z:/iiixr-drive/Projects/2023_City_Team/dublin_dataset/Normalized/Boundaries/'
inference_image_root_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'inference_image')


def count_files_in_directory(directory_path):
    return sum([1 for entry in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, entry))])

total_file_num = count_files_in_directory(boundary_root_path)

print(total_file_num)


def extract_number_from_string(s):
    return int(re.search(r'(\d+)(?=\.\w+$)', s).group())

def extract_numbers_from_boundaryfile(s):
    return int(re.search(r'(\d+)', s).group())

def perpendicular_line_through_point(line, point):
    """선분에 수직하고 주어진 점을 지나는 선을 반환합니다."""
    dx = line.boundary.geoms[1].x - line.boundary.geoms[0].x
    dy = line.boundary.geoms[1].y - line.boundary.geoms[0].y

    # dx가 0이면 선분은 수직이므로 수평선을 반환
    if dx == 0:
        return LineString([(point.x, -9999), (point.x, 9999)])

    # 선분의 기울기
    m = dy / dx

    # 수직선의 기울기
    m_perp = -1 / m

    # y = mx + c => c = y - mx
    c = point.y - m_perp * point.x
    return LineString([(-9999, m_perp * -9999 + c), (9999, m_perp * 9999 + c)])


def get_intersection_coords(intersection):
    if intersection.is_empty:
        return []
    elif isinstance(intersection, Point):
        return [intersection.coords[0]]
    elif isinstance(intersection, (MultiPoint, LineString)):
        return list(intersection.coords)
    elif isinstance(intersection, MultiLineString):
        return [coord for part in intersection.geoms for coord in part.coords]
    elif isinstance(intersection, Polygon):
        return list(intersection.exterior.coords)
    elif isinstance(intersection, MultiPolygon):
        coords = []
        for part in intersection.geoms:
            coords.extend(list(part.exterior.coords))
        return coords
    else:
        return []


def create_polygon_from_line(line, polygon):
    perp_line1 = perpendicular_line_through_point(line, line.boundary.geoms[0])
    perp_line2 = perpendicular_line_through_point(line, line.boundary.geoms[1])

    intersections1 = get_intersection_coords(perp_line1.intersection(polygon))
    intersections2 = get_intersection_coords(perp_line2.intersection(polygon))

    new_polygon_coords = intersections1 + intersections2[::-1]

    if len(new_polygon_coords) < 4:
        return None  # 충분한 좌표가 없으므로 None 반환

    new_polygon = Polygon(new_polygon_coords)
    return new_polygon

def rotated_lines_90(p1, p2, unit_length, scale=5):
    v = p2 - p1
    R = np.array([[0, 1], [-1, 0]])
    v_rotated = np.dot(R, v)
    magnitude_v = np.linalg.norm(v_rotated)
    v_normalized = v_rotated / magnitude_v

    line_from_p1 = [p1, p1 + v_normalized * unit_length * scale]
    line_from_p2 = [p2, p2 + v_normalized * unit_length * scale]

    return np.array(line_from_p1), np.array(line_from_p2)



def rectangle_polygon_from_lines(line1, line2):
    # 꼭지점 계산
    point1 = line1[0]
    point2 = line1[1]
    point3 = line2[1]
    point4 = [point2[0] + (point3[0] - point1[0]), point2[1] + (point3[1] - point1[1])]

    return Polygon([point1, point2, point4, point3])


# 폴리곤을 그리드로 나누기 위한 함수
def divide_polygon_into_grids(polygon, num_grids):
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

for index in range(4, total_file_num):
    with open(unit_coords_path, 'rb') as f:
        unit_coords_data = pickle.load(f)


    unique_boundary_indices = list(set([seg[0] for seg in unit_coords_data]))
    sorted_boundary_indices = sorted(unique_boundary_indices, key=extract_number_from_string)
    original_boundary = sorted_boundary_indices[index]
    print(original_boundary)

    boundary_polygon_path = boundary_root_path + original_boundary

    gdf = gpd.read_file(boundary_polygon_path)
    boundary_polygon = gdf.iloc[0]['geometry']

    npz_data = np.load(npz_path)
    building_index_sequences = npz_data['building_index_sequences'][index]

    unit_coords = []

    for unit_coord in unit_coords_data:
        if unit_coord[0] == original_boundary:
            unit_coords.append(unit_coord[1])

    building_exists_index = []

    for idx in range(len(building_index_sequences)):
        if building_index_sequences[idx] == 1:
            building_exists_index.append(idx)

    unit_with_building = []

    for exist_idx in building_exists_index:
        unit_with_building.append(unit_coords[exist_idx])


    lines = [LineString(coords) for coords in unit_with_building]

    all_lines = [LineString(coords) for coords in unit_coords]

    coords = []
    for line in all_lines:
        coords.extend(list(line.coords[:-1]))
    polygon = Polygon(coords)


    rotated_lines = []

    print("len", len(unit_coords))


    for unit_road_idx, unit_road in enumerate(unit_coords):
        p1 = np.array(unit_road[0])
        p2 = np.array(unit_road[1])

        lines_from_p1, lines_from_p2 = rotated_lines_90(p1, p2, unit_length)

        rotated_lines.append([unit_road_idx, lines_from_p1.tolist(), lines_from_p2.tolist()])


    unit_line_squares = []

    for unit_idx in range(len(unit_coords)):
        unit_line = [unit_coords[unit_idx][0][0], unit_coords[unit_idx][0][1]], [unit_coords[unit_idx][1][0], unit_coords[unit_idx][1][1]]
        unit_line = list(unit_line)
        unit_line_square = []
        unit_line_square += [unit_line, rotated_lines[unit_idx][1]]
        unit_line_squares.append(unit_line_square)

    num_grids = 100
    grids = divide_polygon_into_grids(polygon, num_grids)

    linestring_polygon_map = {}

    linestring_assignments = {i: [] for i in range(len(all_lines))}
    for grid in grids:
        center = grid.centroid
        nearest_index = min(range(len(all_lines)), key=lambda i: center.distance(all_lines[i]))
        linestring_assignments[nearest_index].append(grid)


    for key, value in linestring_assignments.items():
        if value:
            polygon = cascaded_union(value)
            linestring_polygon_map[key] = polygon


    fig, ax = plt.subplots()

    all_indices = set(linestring_polygon_map.keys())
    no_building_indices = all_indices - set(building_exists_index)

    rectangle_polygons = []

    for idx, unit_line in enumerate(unit_line_squares):
        rectangle_polygon = rectangle_polygon_from_lines(unit_line[0], unit_line[1])
        if idx in no_building_indices:
            rectangle_polygons.append(rectangle_polygon)

    #
    # for idx in building_exists_index:
    #     try:
    #         linestring_area_polygon = linestring_polygon_map[idx]
    #
    #         if isinstance(linestring_area_polygon, Polygon):
    #             x, y = linestring_area_polygon.exterior.xy
    #             # ax.plot(x, y, 'black')
    #             ax.fill(x, y, color='blue', alpha=0.2)
    #
    #         elif isinstance(linestring_area_polygon, MultiPolygon):
    #             for poly in linestring_area_polygon.geoms:
    #                 x, y = poly.exterior.xy
    #                 # ax.plot(x, y, 'black')
    #                 ax.fill(x, y, color='blue', alpha=0.2)
    #
    #     except KeyError:
    #         pass

    x, y = boundary_polygon.exterior.xy
    ax.plot(x, y, 'black')


    unit_polygons = []

    for line in lines:
        new_polygon = create_polygon_from_line(line, boundary_polygon)
        unit_polygons.append(new_polygon)

        # plot unit
        x, y = line.xy
        ax.plot(x, y, 'r-')

    # plot unit polygons

    # for unit_polygon in unit_polygons:
    #     if unit_polygon is not None:
    #         x, y = unit_polygon.exterior.xy
    #         ax.fill(x, y, color='skyblue', alpha=0.3)
    #
    # for except_polygon in rectangle_polygons:
    #     x, y = except_polygon.exterior.xy
    #     ax.fill(x, y, color = 'white', alpha=0.8)


    ax.legend()
    ax.grid(True)

    fileindex = extract_numbers_from_boundaryfile(original_boundary)

    inference_image_path = os.path.join(inference_image_root_path, f'{city_name}_{fileindex}.png')

    # plt.savefig(inference_image_path)

    plt.show()

