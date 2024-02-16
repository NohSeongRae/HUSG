import numpy as np
import skgeom
from simple_iou import cal_simple_iou
from canonical_transform import get_polyskeleton_longest_path, modified_skel_to_medaxis, warp_bldg_by_midaxis, sparse_generate_graph_from_ftsarray
import networkx as nx
import math
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
import os
import pickle

graph_coords_list = []

def fill_indices_with_ones(total_length, selected_indices):
    filled_list = [0] * total_length

    for i in selected_indices:
        filled_list[i] = 1

    return filled_list

def visual_block_graph(G, draw_edge = True, draw_nonexist = False):
    print("vs")
    pos = []
    size = []
    edge = []
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')

    if not draw_nonexist:
        for i in range(G.number_of_nodes()):
            if G.nodes[i]['exist'] == 0:
                G.remove_node(i)

    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')
    for i in range(G.number_of_nodes()):
        pos.append([G.nodes[i]['posx'], G.nodes[i]['posy']])
        size.append([G.nodes[i]['size_x'], G.nodes[i]['size_y']])

    for e in G.edges:
        edge.append(e)

    pos = np.array(pos, dtype = np.double)
    size = np.array(size, dtype = np.double)
    edge = np.array(edge, dtype = np.int16)


    if len(pos) > 0:
        plt.scatter(pos[:, 0], pos[:, 1], c = 'red', s=50)
        ax = plt.gca()
        for i in range(size.shape[0]):
            ax.add_patch(Rectangle((pos[i, 0] - size[i, 0] / 2.0, pos[i, 1] - size[i, 1] / 2.0), size[i, 0], size[i, 1], linewidth=2, edgecolor='r', facecolor='b', alpha=0.3))

        if draw_edge:
            for i in range(edge.shape[0]):
                l = mlines.Line2D([pos[edge[i, 0], 0], pos[edge[i, 1], 0]], [pos[edge[i, 0], 1], pos[edge[i, 1], 1]])
                ax.add_line(l)

    plt.show()

def find_closest_coords_indices(x_pos, y_pos, size_x, size_y, b_shape, b_iou, graph_coords_list):
    points = np.array([x_pos, y_pos]).T
    coords = np.array(graph_coords_list)

    # 가장 가까운 coords의 idx
    distances = np.sqrt(((points[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    closest_indices = np.argmin(distances, axis=1)

    # 원래의 x_pos, y_pos, size_x, size_y, b_shape, b_iou
    return closest_indices, x_pos, y_pos, size_x, size_y, b_shape, b_iou


def fill_with_zeros_and_values(total_length, closest_indices, values):
    filled_lists = [[0] * total_length for _ in values]

    for idx, closest_idx in enumerate(closest_indices):
        for i, value in enumerate(values):
            if closest_idx < total_length:
                filled_lists[i][closest_idx] = value[idx]

    return filled_lists


def get_square_bounds(polygon, padding_percentage=0):
    # building data 전체를 geodataframe형태로 저장
    # gdf = gpd.read_file(geojson_path)

    # boundary
    bounds = polygon.bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]

    # square_size = max(width, height)
    square_size = max(width, height) * (1 + padding_percentage / 100)

    # 중심좌표
    center_x = (bounds[0] + bounds[2]) / 2
    center_y = (bounds[1] + bounds[3]) / 2

    square_coords = [
        (center_x - square_size / 2, center_y - square_size / 2),
        (center_x - square_size / 2, center_y + square_size / 2),
        (center_x + square_size / 2, center_y + square_size / 2),
        (center_x + square_size / 2, center_y - square_size / 2),
        (center_x - square_size / 2, center_y - square_size / 2)
    ]

    left = square_coords[0][0]
    upper = square_coords[0][1]
    right = square_coords[2][0]
    lower = square_coords[2][1]

    return left, upper, right, lower

def graph_coords():
    start_points = [np.array([-1, 1]), np.array([-1, 0.3333]), np.array([-1, -0.3333]), np.array([-1, -1])]
    end_points = [np.array([1, 1]), np.array([1, 0.3333]), np.array([1, -0.3333]), np.array([1, -1])]

    graph_nodes = []

    for i in range(len(start_points)):

        start_point = start_points[i]
        end_point = end_points[i]

        num_segments = 30 - 1

        delta = (end_point - start_point) / num_segments

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

def globalmapper_graph(globalmapper_custom_raw_data):
    with open(globalmapper_custom_raw_data, 'rb') as file:
        raw_data = pickle.load(file)

    norm_blk_poly = raw_data[0]

    polygons = raw_data[1]

    norm_bldg_poly = []

    for i in range(len(raw_data[1])):
        norm_bldg_poly.append(raw_data[1][i])

    b_shape = []
    b_iou = []

    for polygon in polygons:
        shape, iou = cal_simple_iou(polygon)
        b_shape.append(shape)
        b_iou.append(iou)

    exterior_polyline = list(norm_blk_poly.exterior.coords)[:-1]
    exterior_polyline.reverse()
    poly_list = []
    for ix in range(len(exterior_polyline)):
        poly_list.append(exterior_polyline[ix])
    sk_norm_blk_poly = skgeom.Polygon(poly_list)

    ### get the skeleton of block (skeletonization)
    skel = skgeom.skeleton.create_interior_straight_skeleton(sk_norm_blk_poly)
    G, longest_skel = get_polyskeleton_longest_path(skel, sk_norm_blk_poly)

    ### get the medial axis of block
    medaxis = modified_skel_to_medaxis(longest_skel, norm_blk_poly)

    ### warp all building locations and sizes
    pos_xsorted, size_xsorted, xsort_idx, aspect_rto = warp_bldg_by_midaxis(norm_bldg_poly, norm_blk_poly, medaxis)

    x_pos = [coord[0] for coord in pos_xsorted]
    y_pos = [coord[1] for coord in pos_xsorted]

    size_x = [coord[0] for coord in size_xsorted]
    size_y = [coord[1] for coord in size_xsorted]

    b_shape = [b_shape[i] for i in xsort_idx]
    b_iou = [b_iou[i] for i in xsort_idx]

    """
    여기 위에 부분이 cst 하고 pos, size, shape, iou 등 계산하고 밑에가 stubby graph에 mapping 하는 코드 
    """
    closest_indices_for_all_points, closest_x, closest_y, closest_size_x, closest_size_y, closest_shapes, closest_ious = find_closest_coords_indices(
        x_pos, y_pos, size_x, size_y, b_shape, b_iou, graph_coords_list)

    pos_x_120, pos_y_120, size_x_120, size_y_120, shapes_120, ious_120 = \
        fill_with_zeros_and_values(120, closest_indices_for_all_points, [x_pos, y_pos, size_x, size_y, b_shape, b_iou])

    long_side = [0] * 120
    exist = fill_indices_with_ones(120, closest_indices_for_all_points)

    g_add = sparse_generate_graph_from_ftsarray(4, 30, pos_x_120, pos_y_120, size_y_120, size_x_120,
                                                exist, aspect_rto, long_side, shapes_120, ious_120)

    for node, data in g_add.nodes(data=True):
        print("g add")
        print(f"Node: {node}")
        print(f"Data: {data}")
        print(f"Graph: {g_add.graph}")
        print("------")

    visual_block_graph(g_add)

    globalmapper_stubby_graph_path = os.path.join("Z:", "iiixr-drive", "Projects", "2023_City_Team",
                                             "1_evaluation", "globalmapper", "stubby_edge.gpickle")

    stubby_graph = nx.read_gpickle(globalmapper_stubby_graph_path)

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

