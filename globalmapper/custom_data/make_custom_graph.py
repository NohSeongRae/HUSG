import numpy as np
from shapely.geometry import Polygon
import skgeom
from cal_iou import cal_iou
from canonical_transform import get_polyskeleton_longest_path, modified_skel_to_medaxis, warp_bldg_by_midaxis
from geo_utils import get_block_aspect_ratio, norm_block_to_horizonal, get_block_parameters
import networkx as nx
import matplotlib.pyplot as plt
import math
import os
import pickle
from shapely.geometry import MultiLineString

graph_coords_list = []


def find_closest_coords_indices(posx, posy, coords):
    closest_indices = []

    for x, y in zip(posx, posy):
        point = np.array([x, y])

        distances = np.sqrt(np.sum((coords - point) ** 2, axis=1))

        closest_index = np.argmin(distances)
        closest_indices.append(closest_index)
    return closest_indices


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

def graph_coords():
    start_points = [np.array([-1, 1]), np.array([-1, 0.3333]), np.array([-1, -0.3333]), np.array([-1, -1])]
    end_points = [np.array([1, 1]), np.array([1, 0.3333]), np.array([1, -0.3333]), np.array([1, -1])]

    graph_nodes = []

    for i in range(len(start_points)):
        # Define the start and end points
        start_point = start_points[i]
        end_point = end_points[i]

        # Calculate the number of segments (since we include the endpoints, we subtract 1 from the total number of points)
        num_segments = 30 - 1

        # Calculate the difference between the start and end point
        delta = (end_point - start_point) / num_segments

        # Initialize an array to hold all the points
        points = np.array([start_point + i*delta for i in range(num_segments + 1)])

        points.tolist()

        for i in range(len(points)):
            graph_nodes.append(points[i])

    for i in range(len(graph_nodes)):
        graph_coords_list.append(graph_nodes[i].tolist())

    return graph_coords_list

graph_nodes_list = []

def graph_node():
    for i in range(4):
        for j in range(30):
            graph_nodes_list.append((i, j))

    return graph_nodes_list

graph_coords_list = graph_coords()
graph_nodes_list = graph_node()

def calculate_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def globalmapper_graph(globalmapper_custom_raw_data, city_name, index):
    with open(globalmapper_custom_raw_data, 'rb') as file:
        raw_data = pickle.load(file)

    #### a random shaply.Polygon
    norm_blk_poly = raw_data[0]

    polygons = raw_data[1]


    #### a random list of shaply.Polygon
    norm_bldg_poly = []

    for i in range(len(raw_data[1])):
        norm_bldg_poly.append(raw_data[1][i])

    b_shape = []
    b_iou = []

    for polygon in polygons:
        shape, iou = cal_iou(polygon)
        b_shape.append(shape)
        b_iou.append(iou)

    exterior_polyline = list(norm_blk_poly.exterior.coords)[:-1]
    exterior_polyline.reverse()
    poly_list = []
    for ix in range(len(exterior_polyline)):
        poly_list.append(exterior_polyline[ix])
    sk_norm_blk_poly = skgeom.Polygon(poly_list)

    #### get the skeleton of block
    skel = skgeom.skeleton.create_interior_straight_skeleton(sk_norm_blk_poly)
    G, longest_skel = get_polyskeleton_longest_path(skel, sk_norm_blk_poly)

    template_height = 4  # opt['template_height']
    template_width = 30  # opt['template_width']

    ### get the medial axis of block
    medaxis = modified_skel_to_medaxis(longest_skel, norm_blk_poly)

    medaxis_length = medaxis.length

    #############   wrap all building locations and sizes ###############################################################
    pos_xsorted, size_xsorted, xsort_idx, aspect_rto = warp_bldg_by_midaxis(norm_bldg_poly, norm_blk_poly, medaxis)

    x_pos = [coord[0] for coord in pos_xsorted]
    y_pos = [coord[1] for coord in pos_xsorted]

    size_x = [coord[0] for coord in size_xsorted]
    size_y = [coord[1] for coord in size_xsorted]

    b_shape = [b_shape[i] for i in xsort_idx]
    b_iou = [b_iou[i] for i in xsort_idx]


    # raw_polygon = raw_data[0]
    # azimuth, bbx = get_block_parameters(raw_polygon)
    # block = norm_block_to_horizonal([raw_polygon], azimuth, bbx)
    # normalized_polygon = block[0]
    # min_rotated_rect = normalized_polygon.minimum_rotated_rectangle
    # insidemask(normalized_polygon, min_rotated_rect)

    closest_indices_for_all_points = find_closest_coords_indices(x_pos, y_pos, graph_coords_list)

    globalmapper_stubby_graph_path = os.path.join("Z:", "iiixr-drive", "Projects", "2023_City_Team",
                                             "1_evaluation", "globalmapper", "stubby_edge.gpickle")

    stubby_graph = nx.read_gpickle(globalmapper_stubby_graph_path)


    num_of_buildings = len(x_pos)

    count = 0

    for node in stubby_graph.nodes():
        if node in closest_indices_for_all_points:
            stubby_graph.nodes[node]['old_label'] = graph_nodes_list[node]
            stubby_graph.nodes[node]['posx'] = x_pos[count]
            stubby_graph.nodes[node]['posy'] = y_pos[count]
            stubby_graph.nodes[node]['exist'] = 1.0
            stubby_graph.nodes[node]['merge'] = 0
            stubby_graph.nodes[node]['size_x'] = size_x[count]
            stubby_graph.nodes[node]['size_y'] = size_y[count]
            stubby_graph.nodes[node]['shape'] = b_shape[count]
            stubby_graph.nodes[node]['iou'] = b_iou[count]
            stubby_graph.nodes[node]['height'] = 0.0
            count += 1
        else:
            stubby_graph.nodes[node]['old_label'] = graph_nodes_list[node]
            stubby_graph.nodes[node]['posx'] = 0.0
            stubby_graph.nodes[node]['posy'] = 0.0
            stubby_graph.nodes[node]['exist'] = 0.0
            stubby_graph.nodes[node]['merge'] = 0
            stubby_graph.nodes[node]['size_x'] = 0.0
            stubby_graph.nodes[node]['size_y'] = 0.0
            stubby_graph.nodes[node]['shape'] = 0.0
            stubby_graph.nodes[node]['iou'] = 0.0
            stubby_graph.nodes[node]['height'] = 0.0

        stubby_graph.graph['aspect_ratio'] = aspect_rto
        stubby_graph.graph['long_side'] = 0.0

    globalmapper_custom_graph_root = os.path.join("Z:", "iiixr-drive", "Projects", "2023_City_Team", "1_evaluation",
                                                     "globalmapper_custom_data", "processed", f"{city_name}")

    if not os.path.exists(globalmapper_custom_graph_root):
        os.makedirs(globalmapper_custom_graph_root)

    globalmapper_custom_graph_path = os.path.join(globalmapper_custom_graph_root, f"{index}.gpickle")

    nx.write_gpickle(stubby_graph, globalmapper_custom_graph_path)

city_names = ["atlanta", "boston", "dallas",  "denver","houston",
             "lasvegas", "littlerock","miami","neworleans",
             "philadelphia", "phoenix", "portland", "providence","pittsburgh",
             "richmond", "saintpaul","sanfrancisco", "seattle","washington"]

import re
from tqdm import tqdm

for city_name in city_names:
    globalmapper_custom_raw_data_root = os.path.join("Z:", "iiixr-drive", "Projects", "2023_City_Team", "1_evaluation", "globalmapper_custom_data", "raw_geo", f"{city_name}")
    all_files_and_directories = os.listdir(globalmapper_custom_raw_data_root)
    all_files = [f for f in all_files_and_directories if os.path.isfile(os.path.join(globalmapper_custom_raw_data_root, f))]
    extracted_numbers = [int(re.search(r'\d+', name).group()) for name in all_files]

    for index in tqdm(extracted_numbers):
        globalmapper_custom_raw_data = os.path.join(globalmapper_custom_raw_data_root, f"{index}.pkl")
        globalmapper_graph(globalmapper_custom_raw_data, city_name, index)


