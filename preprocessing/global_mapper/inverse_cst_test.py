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

test_gt_graph = "./output/1.gpickle"
test_gt_graph = "./datasets/globalmapper_datasets/1.gpickle"

def move_polygon_center_to_midpoint(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    center_x, center_y = (minx + maxx) / 2, (miny + maxy) / 2

    shift_x = 0.5 - center_x
    shift_y = 0.5 - center_y

    moved_polygon = translate(polygon, shift_x, shift_y)

    return moved_polygon, (shift_x, shift_y)

G = nx.read_gpickle(test_gt_graph)
midaxis = G.graph['midaxis']
aspect_rto = G.graph['aspect_ratio']
polygon = G.graph['polygon']

node_size, node_pos, node_attr, edge_list, node_idx, asp_rto, longside, b_shape, b_iou = graph2vector_processed(G)

org_bldg, org_pos, org_size = inverse_warp_bldg_by_midaxis(node_pos, node_size, midaxis, aspect_rto,
                                                           rotate_bldg_by_midaxis=True,
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

fig, ax = plt.subplots(figsize=(8, 8))

for norm_polygon in shifted_org_bldg:
    x, y = norm_polygon.exterior.xy
    ax.plot(x, y, color='black', linewidth=0.5)

ax.set_xticks([])
ax.set_yticks([])

# plt.grid(False)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.savefig('inverse_cst_test_normalized.png', dpi=32, bbox_inches='tight', pad_inches=0)