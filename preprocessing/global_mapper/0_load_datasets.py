import math
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.affinity import affine_transform, rotate
import numpy as np
import skgeom
from canonical_transform import get_polyskeleton_longest_path, modified_skel_to_medaxis, warp_bldg_by_midaxis, sparse_generate_graph_from_ftsarray
from simple_iou import cal_simple_iou
import networkx as nx
from skgeom.draw import draw
from tqdm import tqdm
from rasterio.features import geometry_mask
import rasterio
from shapely.geometry import Point
import shapely

# MultiPolygon을 플롯하기
# fig, ax = plt.subplots()

def get_obb_rotation_angle(polygon):
    """
    Calculate the oriented bounding box (OBB) for the given polygon
    and return the angle (in degrees) to align the longest side of the OBB with x-axis.
    """
    # Get the oriented bounding box (OBB) of the polygon
    obb = polygon.minimum_rotated_rectangle

    # Get the coordinates of the OBB and remove the duplicate point
    obb_coords = list(obb.exterior.coords)[:-1]

    # Initialize variables to find the longest side
    max_length = 0
    for i in range(len(obb_coords)):
        p1 = obb_coords[i]
        p2 = obb_coords[(i + 1) % len(obb_coords)]
        length = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if length > max_length:
            max_length = length
            longest_edge = (p1, p2)

    # Calculate the angle of the longest side relative to the x-axis
    dx = longest_edge[1][0] - longest_edge[0][0]
    dy = longest_edge[1][1] - longest_edge[0][1]
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)

    # Adjust the angle to ensure it's within 0 to 90 degrees
    angle_deg = np.mod(angle_deg, 360)
    if angle_deg > 180:
        angle_deg -= 360
    if angle_deg < 0:
        angle_deg += 180
    if angle_deg > 90:
        angle_deg = 180 - angle_deg

    return -angle_deg, dx

def plot_mask(mask):
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='gray')
    plt.title("Mask Visualization")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.colorbar()
    plt.show()

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

def insidemask(boundary_polygon, image_size=64):
    angle, dx = get_obb_rotation_angle(boundary_polygon)

    boundary_polygon = shapely.affinity.rotate(boundary_polygon, angle)
    boundary_line = boundary_polygon.boundary

    width, height = image_size, image_size

    left, upper, right, lower = get_square_bounds(boundary_polygon.minimum_rotated_rectangle)

    transform = rasterio.transform.from_bounds(left, upper, right, lower, width, height)

    mask = geometry_mask([boundary_polygon], transform=transform, invert=True, out_shape=(height, width))

    scaled_mask = (mask * 1).astype(np.uint8)

    return scaled_mask, dx

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

def make_edge():
    # 그리드의 크기
    rows, cols = 3, 40

    # 각 노드의 상하좌우 인접 노드와의 연결을 나타내는 간선 인덱스 생성
    edge_indices = []

    for row in range(rows):
        for col in range(cols):
            node_index = row * cols + col  # 현재 노드의 인덱스

            if row == 0 or row == rows - 1:
                # 상하좌우 인접 노드의 인덱스 계산
                neighbors = [
                    (row - 1, col),  # 상
                    (row + 1, col),  # 하
                    (row, col - 1),  # 좌
                    (row, col + 1)  # 우
                ]
            else:
                # 상하좌우 인접 노드의 인덱스 계산
                neighbors = [
                    # (row - 1, col),  # 상
                    # (row + 1, col),  # 하
                    (row, col - 1),  # 좌
                    (row, col + 1)  # 우
                ]

            # edge_indices.append([node_index, node_index])
            for n_row, n_col in neighbors:
                # 인접 노드가 그리드 범위 내에 있는지 확인
                if 0 <= n_row < rows and 0 <= n_col < cols:
                    neighbor_index = n_row * cols + n_col
                    # 간선 인덱스에 추가 (방향성이 없는 그래프 가정)
                    edge_indices.append([node_index, neighbor_index])
    return edge_indices

def mapping(x_pos, y_pos):
    # 그리드 사이즈 + 2
    rows, cols = 5, 42

    # 행과 열에 대한 분할 점 계산
    y_divisions = np.linspace(-1, 1, rows)[1: -1]
    x_divisions = np.linspace(-1, 1, cols)[1: -1]

    # 모든 교차점의 x, y 좌표를 계산
    x_coords, y_coords = np.meshgrid(x_divisions, y_divisions)

    # 2차원 좌표 배열로 변환
    coordinates = np.dstack([x_coords, y_coords])
    coordinates = np.reshape(coordinates, (-1, 2))
    x_all = coordinates[:, 0].flatten()
    y_all = coordinates[:, 1].flatten()

    # plt.scatter(x_all, y_all)  # scatter 플롯 생성

    node_indices = np.zeros(120, dtype=int)

    used_indices = []
    for idx, pos in enumerate(zip(x_pos, y_pos)):
        x_pos, y_pos = pos

        deltas = coordinates - np.array([x_pos, y_pos])
        dist_squared = np.sum(deltas ** 2, axis=1)
        for used_index in used_indices:
            dist_squared[used_index] = np.inf

        # 가장 짧은 거리의 인덱스 찾기
        closest_index = np.argmin(dist_squared)
        used_indices.append(closest_index)

        # plt.scatter(coordinates[closest_index][0], coordinates[closest_index][1], color='r')
        node_indices[closest_index] = idx + 1

    return node_indices

# def plot(boundary, buildings):
#     x, y = boundary.exterior.xy  # 외곽선의 x, y 좌표
#     ax.plot(x, y)
#
#     # 각 Polygon에 대해 반복
#     for building in buildings:
#         x, y = building.exterior.xy  # 외곽선의 x, y 좌표
#         ax.plot(x, y)
#
#     ax.set_title('MultiPolygon Plot')

def graph_node():
    graph_nodes_list = []
    for i in range(4):
        for j in range(30):
            graph_nodes_list.append((i, j))

    return graph_nodes_list

output_file_num = 0
for file_index in tqdm(range(120000)):
    file_path = str(file_index)
    input_file_path = f'raw_datasets/globalmapper_dataset/raw_geo/{file_path}'
    with open(input_file_path, 'rb') as file:
        data = pickle.load(file)
        boundary = data[0]
        buildings = data[1]

    boundary = boundary.simplify(0.5, preserve_topology=True)
    scaled_mask, dx = insidemask(boundary)

    if abs(dx) == 0:
        print('저장하지 않음')
        continue

    exterior_polyline = list(boundary.exterior.coords)[:-1]
    exterior_polyline.reverse()
    poly_list = []
    for ix in range(len(exterior_polyline)):
        poly_list.append(exterior_polyline[ix])
    sk_boundary = skgeom.Polygon(poly_list)

    skel = skgeom.skeleton.create_interior_straight_skeleton(sk_boundary)
    G, longest_skel = get_polyskeleton_longest_path(skel, sk_boundary)

    ### get the medial axis of block
    medaxis = modified_skel_to_medaxis(longest_skel, boundary)
    if medaxis == None:
        print('저장하지 않음')
        continue

    ### warp all building locations and sizes
    pos_xsorted, size_xsorted, xsort_idx, aspect_rto = warp_bldg_by_midaxis(buildings, boundary, medaxis)
    if pos_xsorted.all() == None:
        print('저장하지 않음')
        continue

    x_pos = [coord[0] for coord in pos_xsorted]
    y_pos = [coord[1] for coord in pos_xsorted]

    size_x = [coord[0] for coord in size_xsorted]
    size_y = [coord[1] for coord in size_xsorted]

    b_shape = []
    b_iou = []

    for building in buildings:
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
        values_to_check = [x_pos[building_index], y_pos[building_index], size_x[building_index], size_y[building_index],
                           b_shape[building_index], b_iou[building_index]]
        if any(math.isinf(val) or math.isnan(val) for val in values_to_check):
            is_issue = True
            break

    graph_values_to_check = [aspect_rto, medaxis, boundary, scaled_mask, dx]
    if any(math.isinf(val) or math.isnan(val) for val in graph_values_to_check if isinstance(val, (int, float))):
        is_issue = True

    if is_issue:
        print('저장하지 않음')
        continue

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
            G.nodes[node]['posx'] = 0 # ((node % 40) / 39 * 2 - 1) / 42 * 40
            G.nodes[node]['posy'] = 0 # ((node // 40) / 2 * 2 - 1) / 5 * 3
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
    G.graph['polygon'] = boundary
    G.graph['binary_mask'] = scaled_mask
    G.graph['block_scale'] = 1 / abs(dx)
    output_file_path = 'raw_datasets/globalmapper_dataset/processed'
    with open(f'{output_file_path}/{output_file_num}.gpickle', 'wb') as f:
        nx.write_gpickle(G, f)
        output_file_num += 1

# # 노드 위치 정보를 기반으로 위치 사전(pos) 생성
# pos = {node: (G.nodes[node]['posx'], G.nodes[node]['posy']) for node in G.nodes()}
#
# # 존재하는 노드와 존재하지 않는 노드를 구분하기 위한 색상 및 크기 설정
# node_colors = ['green' if G.nodes[node]['exist'] == 1.0 else 'red' for node in G.nodes()]
# node_sizes = [50 if G.nodes[node]['exist'] == 1.0 else 20 for node in G.nodes()]
#
# # 그래프 시각화
# plt.figure(figsize=(12, 8))  # 시각화할 창의 크기 설정
# nx.draw(G, pos, node_color=node_colors, node_size=node_sizes, with_labels=True, font_weight='bold')
#
# # 시각화한 그래프 표시
# plt.title('Graph Visualization')
# plt.show()