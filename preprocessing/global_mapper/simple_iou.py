from shapely.geometry import MultiPolygon
from scipy.spatial.distance import directed_hausdorff
from scipy.optimize import minimize
from shapely.geometry import Polygon
import numpy as np

# Compute Hausdorff distance between two sets of points
def hausdorff_distance(A, B):
    return max(directed_hausdorff(A, B)[0], directed_hausdorff(B, A)[0])

def normalize_polygon(polygon):
    # Polygon 객체의 좌표를 NumPy 배열로 변환
    coords = np.array(polygon.exterior.coords)

    # 정규화
    min_x, min_y = np.min(coords, axis=0)
    max_x, max_y = np.max(coords, axis=0)
    scale_x, scale_y = max_x - min_x, max_y - min_y
    normalized = np.array([[ (x - min_x) / scale_x, (y - min_y) / scale_y ] for x, y in coords])
    return normalized


templates = {
    0: [np.array(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]).exterior.coords[:-1])],
    1: [np.array(Polygon([(0, 0), (1, 0), (1, 0.5), (0.5, 0.5), (0.5, 1), (0, 1)]).exterior.coords[:-1]),
        np.array(Polygon([(0, 0), (1, 0), (1, 1), (0.5, 1), (0.5, 0.5), (0, 0.5)]).exterior.coords[:-1]),
        np.array(Polygon([(0.5, 0), (1, 0), (1, 1), (0, 1), (0, 0.5), (0.5, 0)]).exterior.coords[:-1]),
        np.array(Polygon([(0, 0), (0.5, 0), (0.5, 0.5), (1, 0.5), (1, 1), (0, 1)]).exterior.coords[:-1])],
    2: [np.array(Polygon([(0, 0), (1, 0), (1, 1), (0.75, 1), (0.75, 0.25), (0.25, 0.25), (0.25, 1), (0, 1)]).exterior.coords[:-1]),
        np.array(Polygon([(0, 0), (1, 0), (1, 0.25), (0.25, 0.25), (0.25, 0.75), (1, 0.75), (1, 1), (0, 1)]).exterior.coords[:-1]),
        np.array(Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0.75), (0.75, 0.75), (0.75, 0.25), (0, 0.25)]).exterior.coords[:-1]),
        np.array(Polygon([(0, 0), (0.25, 0), (0.25, 0.75), (0.75, 0.75), (0.75, 0), (1, 0), (1, 1), (0, 1)]).exterior.coords[:-1])],
    3: [np.array(Polygon([(0.25, 0), (0.75, 0), (0.75, 0.25), (1, 0.25), (1, 0.75), (0.75, 0.75), (0.75, 1), (0.25, 1), (0.25, 0.75), (0, 0.75), (0, 0.25), (0.25, 0.25)]).exterior.coords[:-1])]
}

def cal_simple_iou(polygon):
    normalized_polygon = normalize_polygon(polygon)
    min_distance = float('inf')
    closest_template_key = None

    for key, template_list in templates.items():
        for template in template_list:
            distance = hausdorff_distance(normalized_polygon, template)
            if distance < min_distance:
                min_distance = distance
                closest_template_key = key

    iou = 100 - min_distance
    iou = iou / 100

    return closest_template_key, iou


