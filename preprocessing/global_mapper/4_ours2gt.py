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
    with open(f'datasets/new_city_datasets/graph_condition_train_datasets/{data_type}/{str(idx)}.pkl', 'rb') as file:
        buildings = pickle.load(file)

    building_polygons = []
    original_building_polygons = []
    # for building in buildings:
    #     poly = Polygon(building.T)
    #     x, y = poly.exterior.xy
    #     plt.plot(x, y)

    graph = nx.read_gpickle(f'datasets/new_city_datasets/graph_condition_train_datasets/{data_type}/{str(idx)}.gpickle')

    n_node = graph.number_of_nodes()
    n_building = len(buildings)
    n_chunk = n_node - n_building

    for node in graph.nodes():
        if node >= n_chunk:
            x, y, w, h, theta = graph.nodes[node]['node_features']
            polygon = create_rotated_rectangle(x, y, w, h, theta)
            building_polygons.append(polygon)
            original_building_polygons.append(polygon)

    adj_matrix = nx.adjacency_matrix(graph).todense()
    building_adj_matrix = adj_matrix[n_chunk:, n_chunk:]

    boundary_points = []
    for node in graph.graph['condition'].nodes():
        x_min, y_min, x_max, y_max, theta = graph.graph['condition'].nodes[node]['chunk_features']
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2

        boundary_points.append([cx * 2, cy * 2])

    original_boundary_polygon = Polygon(boundary_points)

    area_tolerance = 0.0005  # 허용되는 최대 면적 차이
    simplified_polygon = simplify_polygon_randomly(original_boundary_polygon, area_tolerance)

    exterior_polyline = list(simplified_polygon.exterior.coords)[:-1]
    exterior_polyline.reverse()
    poly_list = []
    for ix in range(len(exterior_polyline)):
        poly_list.append(exterior_polyline[ix])
    sk_boundary = skgeom.Polygon(poly_list)

    try:
        skel = skgeom.skeleton.create_interior_straight_skeleton(sk_boundary)
    except:
        return

    G, longest_skel = get_polyskeleton_longest_path(skel, sk_boundary)
    ### get the medial axis of block
    medaxis = modified_skel_to_medaxis(longest_skel, simplified_polygon)
    if medaxis == None:
        print('저장 하지 않음 1')
        return

    ### warp all building locations and sizes
    pos_xsorted, size_xsorted, xsort_idx, aspect_rto = warp_bldg_by_midaxis(building_polygons, simplified_polygon, medaxis)
    if pos_xsorted.all() == None:
        print('저장 하지 않음 2')
        return

    th = 0.25
    x_pos = [coord[0] for coord in pos_xsorted]
    y_pos = [coord[1] for coord in pos_xsorted]
    x_th = th / (max(x_pos) - min(x_pos))
    y_th = th / (max(y_pos) - min(y_pos))
    edge_index = []
    # generate edge index
    for i, data_i in enumerate(zip(pos_xsorted, size_xsorted)):
        edge_index.append([i, i])
        pos, size = data_i
        x, y, = pos
        w, h, = size
        # 네 꼭짓점 계산
        points = [
            (x - w / 2, y + h / 2),  # 왼쪽 상단
            (x + w / 2, y + h / 2),  # 오른쪽 상단
            (x + w / 2, y - h / 2),  # 오른쪽 하단
            (x - w / 2, y - h / 2)  # 왼쪽 하단
        ]

        # Polygon 객체 생성
        polygon_i = Polygon(points)
        # x, y = polygon_i.exterior.xy
        # plt.plot(x, y)

        for j, data_j in enumerate(zip(pos_xsorted, size_xsorted)):
            if i >= j:
                continue

            pos, size = data_j
            x, y, = pos
            w, h, = size
            # 네 꼭짓점 계산
            points = [
                (x - w / 2, y + h / 2),  # 왼쪽 상단
                (x + w / 2, y + h / 2),  # 오른쪽 상단
                (x + w / 2, y - h / 2),  # 오른쪽 하단
                (x - w / 2, y - h / 2)  # 왼쪽 하단
            ]

            # Polygon 객체 생성
            polygon_j = Polygon(points)

            def calculate_axis_distances(polygon_i, polygon_j):
                bounds_i = polygon_i.bounds
                bounds_j = polygon_j.bounds

                # X 축 거리 계산
                distance_x = max(0, max(bounds_i[0], bounds_j[0]) - min(bounds_i[2], bounds_j[2]))

                # Y 축 거리 계산
                distance_y = max(0, max(bounds_i[1], bounds_j[1]) - min(bounds_i[3], bounds_j[3]))

                return distance_x, distance_y

            x_dist, y_dist = calculate_axis_distances(polygon_i, polygon_j)

            if x_dist < x_th and y_dist < y_th:
                is_invalid = False
                for k, data_k in enumerate(zip(pos_xsorted, size_xsorted)):
                    if i != k and j != k:
                        pos, size = data_k
                        x, y, = pos
                        w, h, = size
                        # 네 꼭짓점 계산
                        points = [
                            (x - w / 2, y + h / 2),  # 왼쪽 상단
                            (x + w / 2, y + h / 2),  # 오른쪽 상단
                            (x + w / 2, y - h / 2),  # 오른쪽 하단
                            (x - w / 2, y - h / 2)  # 왼쪽 하단
                        ]

                        # Polygon 객체 생성
                        polygon_k = Polygon(points)
                        check_line = LineString(nearest_points(polygon_i, polygon_j))
                        if polygon_k.intersects(check_line):
                            is_invalid = True
                            break

                        # center_1 = (polygon_i.centroid.x, polygon_i.centroid.y)
                        # center_2 = (polygon_j.centroid.x, polygon_j.centroid.y)
                        # check_line = LineString([center_1, center_2])
                        # if polygon_k.intersects(check_line):
                        #     is_invalid = True
                        #     break

                if not is_invalid:
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    # center_1 = (polygon_i.centroid.x, polygon_i.centroid.y)
                    # center_2 = (polygon_j.centroid.x, polygon_j.centroid.y)
                    # plt.plot([center_1[0], center_2[0]],
                    #          [center_1[1], center_2[1]])
    edge_count = np.zeros(n_building)
    for edge in edge_index:
        edge_count[edge[0]] += 1
        edge_count[edge[1]] += 1

    for count_idx, count in enumerate(edge_count):
        if count == 2:
            cur_building_idx = count_idx
            min_idx = -1
            min_distance = 999

            pos, size = pos_xsorted[cur_building_idx], size_xsorted[cur_building_idx]
            x, y, = pos
            w, h, = size
            # 네 꼭짓점 계산
            points = [
                (x - w / 2, y + h / 2),  # 왼쪽 상단
                (x + w / 2, y + h / 2),  # 오른쪽 상단
                (x + w / 2, y - h / 2),  # 오른쪽 하단
                (x - w / 2, y - h / 2)  # 왼쪽 하단
            ]

            # Polygon 객체 생성
            polygon_i = Polygon(points)
            for j, data_j in enumerate(zip(pos_xsorted, size_xsorted)):
                pos, size = data_j
                x, y, = pos
                w, h, = size
                # 네 꼭짓점 계산
                points = [
                    (x - w / 2, y + h / 2),  # 왼쪽 상단
                    (x + w / 2, y + h / 2),  # 오른쪽 상단
                    (x + w / 2, y - h / 2),  # 오른쪽 하단
                    (x - w / 2, y - h / 2)  # 왼쪽 하단
                ]

                # Polygon 객체 생성
                polygon_j = Polygon(points)
                if cur_building_idx != j:
                    if min_distance > polygon_j.distance(polygon_i):
                        min_idx = j
                        min_distance = polygon_j.distance(polygon_i)

            edge_index.append([count_idx, min_idx])
            edge_index.append([min_idx, count_idx])

    building_adj_matrix = np.zeros((n_building, n_building), dtype=int)
    # edge_index를 사용하여 인접 행렬의 해당 위치를 1로 설정
    for start_node, end_node in edge_index:
        building_adj_matrix[start_node, end_node] = 1

    is_vis = False
    if is_vis:
        # x, y = medaxis.xy
        # # plt.plot(x, y, '-', color='blue', label='Line')
        # for i in range(len(x) - 1):
        #     colors = ['red', 'green', 'blue', 'cyan', 'magenta']
        #     plt.plot(x[i:i + 2], y[i:i + 2], '-', color=colors[i % len(colors)],
        #              label=f'Line {i + 1}' if i == 0 else "")
        #
        # x, y = simplified_polygon.exterior.coords.xy
        # plt.plot(x, y, '-', color='red', label='Boundary')
        #
        # for i, building in enumerate(building_polygons):
        #     x, y = building.exterior.coords.xy
        #     plt.plot(x, y, '-', color='green')
        #     for j in range(len(building_polygons)):
        #         if building_adj_matrix[i, j] == 1:
        #             plt.plot([building_polygons[i].centroid.x, building_polygons[j].centroid.x],
        #                      [building_polygons[i].centroid.y, building_polygons[j].centroid.y])

        x, y = pos_xsorted[:, 0], pos_xsorted[:, 1]
        for i in range(len(x)):
            for j in range(len(x)):
                if building_adj_matrix[i, j] == 1:
                    plt.plot([x[i], x[j]], [y[i], y[j]])

        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.show()
        plt.close()

    is_test_save = True
    if is_test_save:
        scaled_mask, dx = insidemask(simplified_polygon)

        if abs(dx) == 0:
            print('저장 하지 않음')
            return

        x_pos = [coord[0] for coord in pos_xsorted]
        y_pos = [coord[1] for coord in pos_xsorted]

        size_x = [coord[0] for coord in size_xsorted]
        size_y = [coord[1] for coord in size_xsorted]

        b_shape = []
        b_iou = []

        for building in building_polygons:
            shape, iou = cal_simple_iou(building)
            b_shape.append(shape)
            b_iou.append(iou)

        b_shape = [b_shape[i] for i in xsort_idx]
        b_iou = [b_iou[i] for i in xsort_idx]

        G = nx.DiGraph(building_adj_matrix)

        graph_nodes_list = graph_node()

        is_issue = False
        for building_index in range(len(buildings)):
            values_to_check = [x_pos[building_index], y_pos[building_index], size_x[building_index],
                               size_y[building_index],
                               b_shape[building_index], b_iou[building_index]]
            if any(math.isinf(val) or math.isnan(val) for val in values_to_check):
                is_issue = True
                break

        graph_values_to_check = [aspect_rto, medaxis, simplified_polygon, scaled_mask, dx]
        if any(math.isinf(val) or math.isnan(val) for val in graph_values_to_check if isinstance(val, (int, float))):
            is_issue = True

        if is_issue:
            print('저장 하지 않음 3')
            return

        for node in G.nodes():
            G.nodes[node]['old_label'] = graph_nodes_list[node]
            G.nodes[node]['posx'] = x_pos[node]
            G.nodes[node]['posy'] = y_pos[node]
            G.nodes[node]['exist'] = 1.0
            G.nodes[node]['merge'] = 0
            G.nodes[node]['size_x'] = size_x[node]
            G.nodes[node]['size_y'] = size_y[node]
            G.nodes[node]['shape'] = b_shape[node]
            G.nodes[node]['iou'] = b_iou[node]
            G.nodes[node]['height'] = 0.0
            G.nodes[node]['polygon'] = original_building_polygons[node]

        G.graph['aspect_ratio'] = aspect_rto
        G.graph['long_side'] = 0.0
        G.graph['midaxis'] = medaxis
        G.graph['polygon'] = simplified_polygon
        G.graph['binary_mask'] = scaled_mask
        G.graph['block_scale'] = 1 / abs(dx)
        G.graph['building_polygons'] = building_polygons

        output_file_path = f'gt_graph_datasets/{data_type}'
        with open(f'{output_file_path}/{idx}.gpickle', 'wb') as f:
            nx.write_gpickle(G, f)

if __name__ == '__main__':
    end_index = 208622 + 1
    data_type = 'test'

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