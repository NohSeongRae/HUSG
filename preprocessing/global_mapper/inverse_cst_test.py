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

def move_polygon_center_to_midpoint(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    center_x, center_y = (minx + maxx) / 2, (miny + maxy) / 2

    shift_x = 0.5 - center_x
    shift_y = 0.5 - center_y

    moved_polygon = translate(polygon, shift_x, shift_y)

    return moved_polygon, (shift_x, shift_y)

for idx in range(0, 100):
    test_gt_graph = f"./output/gt/{str(idx)}.gpickle"
    test_pred_graph = f"./output/pred/{str(idx)}.gpickle"

    pred_G = nx.read_gpickle(test_pred_graph)
    midaxis = pred_G.graph['midaxis']
    aspect_rto = pred_G.graph['aspect_ratio']
    polygon = pred_G.graph['polygon']

    node_size, node_pos, node_attr, edge_list, node_idx, asp_rto, longside, b_shape, b_iou = graph2vector_processed(pred_G)

    org_bldg, org_pos, org_size = inverse_warp_bldg_by_midaxis(node_pos, node_size, midaxis, aspect_rto,
                                                               rotate_bldg_by_midaxis=False,
                                                               output_mode=False)
    from shapely.wkt import dumps

    wkt_polygons = [dumps(poly) for poly in org_bldg]
    counter = Counter(wkt_polygons)

    org_bldg = [poly for poly, wkt in zip(org_bldg, wkt_polygons) if counter[wkt] < 3]

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

    for norm_polygon in shifted_org_bldg:
        x, y = norm_polygon.exterior.xy

        ax1.plot(x, y, color='k', label='Rotated Box')

    gt_graph = nx.read_gpickle(test_gt_graph)
    for building in gt_graph.graph['building_polygon']:
        x, y = building
        ax2.plot(x, y, color='k', label='Rotated Box')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    directory = 'figure'
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.set_axis_off()
    save_path_1 = os.path.join(directory, "prediction_" + str(idx) + ".png")
    ax1.figure.savefig(save_path_1, dpi=300, bbox_inches='tight')

    ax2.set_aspect('equal', adjustable='box')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.0])
    ax2.set_axis_off()
    save_path_2 = os.path.join(directory, "ground_truth_" + str(idx) + ".png")
    ax2.figure.savefig(save_path_2, dpi=300, bbox_inches='tight')
