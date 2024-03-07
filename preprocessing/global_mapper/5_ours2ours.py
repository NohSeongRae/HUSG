import pickle
import networkx as nx
import matplotlib.pyplot as plt
import shapely
from shapely.geometry import Polygon, MultiLineString, LineString
from shapely.ops import unary_union, nearest_points
import skgeom
from skgeom.draw import draw
import random
from tqdm import tqdm
import numpy as np
import math
from canonical_transform import get_polyskeleton_longest_path, modified_skel_to_medaxis, warp_bldg_by_midaxis, sparse_generate_graph_from_ftsarray
from simple_iou import cal_simple_iou
from a_load_datasets import *

random.seed(327)

def simplify_polygon_randomly(polygon, area_tolerance):
    original_area = polygon.area
    simplified_polygon = polygon
    points = list(polygon.exterior.coords[:-1])  # 마지막 점(첫 번째 점과 동일) 제외하고 리스트 생성
    tried_indices = set()  # 시도한 점의 인덱스 저장

    while len(points) > 3:  # 최소한 삼각형은 유지
        if len(tried_indices) >= len(points):  # 모든 점을 시도했으면 종료
            break

        random_index = random.randint(0, len(points) - 1)
        if random_index in tried_indices:  # 이미 시도한 인덱스라면 다시 시도
            continue
        tried_indices.add(random_index)  # 시도한 인덱스 추가

        new_points = points[:random_index] + points[random_index + 1:]
        new_polygon = Polygon(new_points + [new_points[0]])  # 다각형을 닫기 위해 첫 번째 점을 마지막에 추가

        # 면적 차이 확인
        if abs(original_area - new_polygon.area) <= area_tolerance:
            simplified_polygon = new_polygon
            original_area = new_polygon.area  # 업데이트된 다각형의 면적을 새 기준으로 설정
            points = new_points  # 업데이트된 점 목록
            tried_indices.clear()  # 시도한 인덱스 초기화
        # 면적 차이가 허용 오차를 초과하면 시도한 인덱스만 업데이트하고 반복 계속

    return simplified_polygon

def draw_skeleton(polygon, skeleton, show_time=False):
    draw(polygon)

    for h in skeleton.halfedges:
        if h.is_bisector:
            p1 = h.vertex.point
            p2 = h.opposite.vertex.point
            plt.plot([p1.x(), p2.x()], [p1.y(), p2.y()], 'r-', lw=2)

    if show_time:
        for v in skeleton.vertices:
            plt.gcf().gca().add_artist(plt.Circle(
                (v.point.x(), v.point.y()),
                v.time, color='blue', fill=False))

def get_bbox_corners(x, y, w, h):
    half_w = w / 2
    half_h = h / 2

    top_left = [x - half_w, y - half_h]
    top_right = [x + half_w, y - half_h]
    bottom_left = [x - half_w, y + half_h]
    bottom_right = [x + half_w, y + half_h]

    return [top_left, top_right, bottom_right, bottom_left]

def rotate_points_around_center(points, center, theta_deg):
    theta_rad = np.radians(theta_deg)

    rotation_matrix = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad), np.cos(theta_rad)]
    ])

    points = np.array(points)
    center = np.array(center)
    translated_points = points - center

    rotated_points = np.dot(translated_points, rotation_matrix.T)
    rotated_points = rotated_points + center

    return rotated_points

def create_rotated_rectangle(x, y, w, h, theta):
    # 사각형의 중심을 기준으로 초기 꼭짓점을 계산합니다.
    dx = w / 2
    dy = h / 2
    corners = [(-dx, -dy), (-dx, dy), (dx, dy), (dx, -dy)]

    # 중심점을 기준으로 꼭짓점을 회전시킨 후, 실제 위치로 이동시킵니다.
    theta = np.radians(theta)
    rotated_corners = [
        (math.cos(theta) * cx - math.sin(theta) * cy + x,
         math.sin(theta) * cx + math.cos(theta) * cy + y) for cx, cy in corners
    ]

    # 회전된 꼭짓점들로 Polygon 객체를 생성합니다.
    rotated_rectangle = Polygon(rotated_corners)
    return rotated_rectangle

def generate_datasets(idx, data_type):
    with open(f'datasets/eu_graph_condition_train_datasets/{data_type}/{str(idx)}.pkl', 'rb') as file:
        buildings = pickle.load(file)

    building_polygons = []
    original_building_polygons = []

    graph = nx.read_gpickle(f'datasets/eu_graph_condition_train_datasets/{data_type}/{str(idx)}.gpickle')

    n_node = graph.number_of_nodes()
    n_building = len(buildings)
    n_chunk = n_node - n_building

    # for node in graph.nodes():
    #     if node >= n_chunk:
    #         x, y, w, h, theta = graph.nodes[node]['node_features']
    #         polygon = create_rotated_rectangle(x, y, w, h, (theta * 2 - 1) * 45)
    #         x, y = polygon.exterior.coords.xy
    #         plt.plot(x, y, '-', color='red', label='Boundary')

    boundary_points = []
    for node in graph.graph['condition'].nodes():
        x_min, y_min, x_max, y_max, theta = graph.graph['condition'].nodes[node]['chunk_features']
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2

        boundary_points.append([cx * 2, cy * 2])

    original_boundary_polygon = Polygon(boundary_points)
    # x, y = original_boundary_polygon.exterior.coords.xy
    # plt.plot(x, y, '-', color='blue', label='Boundary')


    # for node1 in graph.nodes():
    #     for node2 in graph.nodes():
    #         if graph.has_edge(node1, node2):
    #             x1, y1, w, h, theta = graph.nodes[node1]['node_features']
    #             x2, y2, w, h, theta = graph.nodes[node2]['node_features']
    #
    #             plt.plot([x1, x2], [y1, y2], color='green')
    # plt.show()

    area_tolerance = 0.0005  # 허용되는 최대 면적 차이
    simplified_polygon = simplify_polygon_randomly(original_boundary_polygon, area_tolerance)
    scaled_mask, dx = insidemask(simplified_polygon, image_size=224)

    graph.graph['condition'] = scaled_mask

    if n_building >= 2:
        output_file_path = f'eu_ours_graph_datasets/{data_type}'
        with open(f'{output_file_path}/{idx}.gpickle', 'wb') as f:
            nx.write_gpickle(graph, f)

if __name__ == '__main__':
    end_index = 208622 + 1
    data_type = 'val'

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = []

        # tqdm의 total 파라미터를 설정합니다.
        progress = tqdm(total=end_index, desc='Processing files', position=0, leave=True)

        # submit 대신 map을 사용하여 future 객체를 얻고, 각 future가 완료될 때마다 진행 상황을 업데이트합니다.
        futures = [executor.submit(generate_datasets, file_index, data_type) for file_index in range(end_index)]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            progress.update(1)

        progress.close()