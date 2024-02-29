import pickle
import networkx as nx
import matplotlib.pyplot as plt
import shapely
from shapely.geometry import Polygon, MultiLineString
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

def generate_datasets(idx, data_type):
    with open(f'datasets/new_city_datasets/graph_condition_train_datasets/{data_type}/{str(idx)}.pkl', 'rb') as file:
        buildings = pickle.load(file)

    building_polygons = []
    for building in buildings:
        building_polygon = shapely.minimum_rotated_rectangle(Polygon(building.T))
        building_polygons.append(building_polygon)

    graph = nx.read_gpickle(f'datasets/new_city_datasets/graph_condition_train_datasets/{data_type}/{str(idx)}.gpickle')

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

    is_vis = True
    if is_vis:
        x, y = medaxis.xy
        # plt.plot(x, y, '-', color='blue', label='Line')
        for i in range(len(x) - 1):
            colors = ['red', 'green', 'blue', 'cyan', 'magenta']
            plt.plot(x[i:i + 2], y[i:i + 2], '-', color=colors[i % len(colors)],
                     label=f'Line {i + 1}' if i == 0 else "")

        x, y = simplified_polygon.exterior.coords.xy
        plt.plot(x, y, '-', color='red', label='Boundary')

        for building in building_polygons:
            x, y = building.exterior.coords.xy
            plt.plot(x, y, '-', color='green')

        x, y = pos_xsorted[:, 0], pos_xsorted[:, 1]
        size_x, size_y = size_xsorted[:, 0], size_xsorted[:, 1]
        for i in range(len(x)):
            plt.gca().add_patch(
                plt.Rectangle((x[i], y[i]), size_x[i], size_y[i], linewidth=1, edgecolor='r', facecolor='none'))

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

        node_indices = mapping(x_pos, y_pos)

        G = nx.Graph()
        G.add_edges_from(make_edge())

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
            if node_indices[node] > 0:
                building_index = node_indices[node] - 1

                G.nodes[node]['old_label'] = graph_nodes_list[node]
                G.nodes[node]['posx'] = x_pos[building_index]
                G.nodes[node]['posy'] = y_pos[building_index]
                G.nodes[node]['exist'] = 1.0
                G.nodes[node]['merge'] = 0
                G.nodes[node]['size_x'] = size_x[building_index]
                G.nodes[node]['size_y'] = size_y[building_index]
                G.nodes[node]['shape'] = b_shape[building_index]
                G.nodes[node]['iou'] = b_iou[building_index]
                G.nodes[node]['height'] = 0.0

            else:
                G.nodes[node]['old_label'] = graph_nodes_list[node]
                G.nodes[node]['posx'] = 0  # ((node % 40) / 39 * 2 - 1) / 42 * 40
                G.nodes[node]['posy'] = 0  # ((node // 40) / 2 * 2 - 1) / 5 * 3
                G.nodes[node]['exist'] = 0.0
                G.nodes[node]['merge'] = 0
                G.nodes[node]['size_x'] = 0.0
                G.nodes[node]['size_y'] = 0.0
                G.nodes[node]['shape'] = 0.0
                G.nodes[node]['iou'] = 0.0
                G.nodes[node]['height'] = 0.0

        G.graph['aspect_ratio'] = aspect_rto
        G.graph['long_side'] = 0.0
        G.graph['midaxis'] = medaxis
        G.graph['polygon'] = simplified_polygon
        G.graph['binary_mask'] = scaled_mask
        G.graph['block_scale'] = 1 / abs(dx)

        plt.show()

        output_file_path = 'datasets/test'
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