import pickle
from geo_utils import inverse_warp_bldg_by_midaxis
import networkx as nx
from urban_dataset import graph2vector_processed
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter
from shapely.affinity import translate
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.affinity import scale
from shapely.ops import transform
import os
from tqdm import tqdm
import shapely
import math
def get_bbox_details(rotated_rectangle):
    # 사각형의 꼭짓점들을 얻음
    x, y = rotated_rectangle.exterior.coords.xy

    min_angle = 999
    min_idx = -1
    for i in range(len(x) - 1):
        dx = x[(i+1) % (len(x)-1)] - x[i]
        dy = y[(i+1) % (len(y)-1)] - y[i]
        angle = math.degrees(math.atan2(dy, dx))
        if min_angle > abs(angle):
            min_angle = abs(angle)
            min_idx = i

    idx = min_idx
    w = math.sqrt((x[(idx + 1) % (len(y)-1)] - x[idx]) ** 2 + (y[(idx + 1) % (len(y)-1)] - y[idx]) ** 2)
    h = math.sqrt((x[(idx + 2) % (len(y)-1)] - x[(idx + 1) % (len(y)-1)]) ** 2 + (y[(idx + 2) % (len(y)-1)] - y[(idx + 1) % (len(y)-1)]) ** 2)

    dx = x[idx + 1] - x[idx]
    dy = y[idx + 1] - y[idx]
    theta = math.degrees(math.atan2(dy, dx))
    theta = (theta + 45) / 90

    # 좌하단 꼭짓점을 x, y로 선택
    x, y = rotated_rectangle.centroid.x, rotated_rectangle.centroid.y
    return x, y, w, h, theta

def move_polygon_center_to_midpoint(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    center_x, center_y = (minx + maxx) / 2, (miny + maxy) / 2

    shift_x = 0.5 - center_x
    shift_y = 0.5 - center_y

    moved_polygon = translate(polygon, shift_x, shift_y)

    return moved_polygon, (shift_x, shift_y)

for idx in tqdm(range(0, 13039)):
    path = 'grid_graph_output'
    test_gt_graph = f"./{path}/gt/{str(idx)}.gpickle"
    test_pred_graph = f"./{path}/pred/{str(idx)}.gpickle"

    try:
        pred_G = nx.read_gpickle(test_pred_graph)
    except:
        continue

    midaxis = pred_G.graph['midaxis']
    aspect_rto = pred_G.graph['aspect_ratio']
    polygon = pred_G.graph['polygon']

    node_size, node_pos, node_attr, edge_list, node_idx, asp_rto, longside, b_shape, b_iou, building_polygons = graph2vector_processed(pred_G)

    org_bldg, org_pos, org_size = inverse_warp_bldg_by_midaxis(node_pos, node_size, midaxis, aspect_rto,
                                                               rotate_bldg_by_midaxis=True,
                                                               output_mode=False)
    from shapely.wkt import dumps

    wkt_polygons = [dumps(poly) for poly in org_bldg]
    counter = Counter(wkt_polygons)

    org_bldg = [poly for poly, wkt in zip(org_bldg, wkt_polygons) if counter[wkt] < 3]
    building_polygons = [poly for poly, wkt in zip(building_polygons, wkt_polygons) if counter[wkt] < 3]

    minx, miny, maxx, maxy = polygon.bounds
    width, height = maxx - minx, maxy - miny
    longer_side = max(width, height)

    def normalize(x, y):
        return ((x - minx) / longer_side, (y - miny) / longer_side)

    minx, miny, maxx, maxy = polygon.bounds
    width, height = maxx - minx, maxy - miny

    normalized_polygon = transform(normalize, polygon)

    moved_polygon, shift_amount = move_polygon_center_to_midpoint(normalized_polygon)

    normalized_org_bldg = [transform(normalize, poly) for poly in org_bldg]
    shifted_org_bldg = [translate(poly, xoff=shift_amount[0], yoff=shift_amount[1]) for poly in normalized_org_bldg]

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
    fig, ax2 = plt.subplots(1, 1, figsize=(6, 6))

    pred_output_list = []
    for norm_polygon in shifted_org_bldg:
        x, y = norm_polygon.exterior.xy
        ax1.plot(x, y, color='k', label='Rotated Box')

        # 중심 좌표 구하기
        center = norm_polygon.centroid.coords[0]
        if not (0 <= center[0] <= 1) or not (0 <= center[0] <= 1):
            print('111111')
            continue

        x, y, w, h, theta = get_bbox_details(norm_polygon)
        pred_output_list.append([x, y, w, h, theta])
    pred_output_list = sorted(pred_output_list, key=lambda item: (item[0], item[1]))

    gt_graph = nx.read_gpickle(test_gt_graph)
    building_polygons = gt_graph.graph['building_polygons']

    gt_output_list = []
    for building in building_polygons:
        x, y = building.exterior.xy
        ax2.plot(x, y, color='k', label='Rotated Box')

        x, y, w, h, theta = get_bbox_details(building)
        gt_output_list.append([x, y, w, h, theta])

    gt_output_list = sorted(gt_output_list, key=lambda item: (item[0], item[1]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    directory = path.replace('output', 'figure')
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.set_axis_off()
    save_path_1 = os.path.join(directory, "prediction_" + str(idx) + ".png")
    ax1.figure.savefig(save_path_1, dpi=300, bbox_inches='tight')
    plt.close(ax1.figure)  # ax1에 연결된 figure 닫기

    ax2.set_aspect('equal', adjustable='box')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.0])
    ax2.set_axis_off()
    save_path_2 = os.path.join(directory, "ground_truth_" + str(idx) + ".png")
    ax2.figure.savefig(save_path_2, dpi=300, bbox_inches='tight')
    plt.close(ax2.figure)  # ax2에 연결된 figure 닫기

    with open(save_path_1.replace('.png', '.pkl'), 'wb') as file:
        pickle.dump(pred_output_list, file)

    with open(save_path_2.replace('.png', '.pkl'), 'wb') as file:
        pickle.dump(gt_output_list, file)
