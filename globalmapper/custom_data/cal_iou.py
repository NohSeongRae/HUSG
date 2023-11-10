from scipy.spatial.distance import directed_hausdorff
from scipy.optimize import minimize
from shapely.geometry import Polygon
import numpy as np
from shapely.ops import unary_union
from shapely.geometry import MultiPolygon

# Compute Hausdorff distance between two sets of points
def hausdorff_distance(A, B):
    return max(directed_hausdorff(A, B)[0], directed_hausdorff(B, A)[0])


# Optimize the position and rotation of the given polygon to match the template
def match_polygon_to_template(polygon_coords, template_coords):
    def objective_function(params):
        # Extract translation and rotation
        dx, dy, theta = params
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])

        # Apply transformation
        transformed_coords = np.dot(polygon_coords, R.T) + np.array([dx, dy])

        return hausdorff_distance(transformed_coords, template_coords)

    # Initial guess
    initial_params = [0, 0, 0]

    # Use Powell's method
    result = minimize(objective_function, initial_params, method='Powell')

    return result.fun

from scipy.spatial.distance import directed_hausdorff
from scipy.optimize import minimize
from shapely.geometry import Polygon
import numpy as np


# Compute Hausdorff distance between two sets of points
def hausdorff_distance(A, B):
    return max(directed_hausdorff(A, B)[0], directed_hausdorff(B, A)[0])


# Optimize the position and rotation of the given polygon to match the template
def match_polygon_to_template(polygon_coords, template_coords):
    def objective_function(params):
        # Extract translation and rotation
        dx, dy, theta = params
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])

        # Apply transformation
        transformed_coords = np.dot(polygon_coords, R.T) + np.array([dx, dy])

        return hausdorff_distance(transformed_coords, template_coords)

    # Initial guess
    initial_params = [0, 0, 0]

    # Use Powell's method
    result = minimize(objective_function, initial_params, method='Powell')

    return result.fun


def compute_min_hausdorff_distance(polygon_coords, template_coords):
    # Original
    distance_original = match_polygon_to_template(polygon_coords, template_coords)

    # Flip horizontally
    flipped_horizontal = np.copy(polygon_coords)
    flipped_horizontal[:, 0] = -flipped_horizontal[:, 0]
    distance_flipped_horizontal = match_polygon_to_template(flipped_horizontal, template_coords)

    # Flip vertically
    flipped_vertical = np.copy(polygon_coords)
    flipped_vertical[:, 1] = -flipped_vertical[:, 1]
    distance_flipped_vertical = match_polygon_to_template(flipped_vertical, template_coords)

    # Flip both horizontally and vertically
    flipped_both = np.copy(polygon_coords)
    flipped_both[:, 0] = -flipped_both[:, 0]
    flipped_both[:, 1] = -flipped_both[:, 1]
    distance_flipped_both = match_polygon_to_template(flipped_both, template_coords)

    return min(distance_original, distance_flipped_horizontal, distance_flipped_vertical, distance_flipped_both)

def cal_iou(polygon):
    # Define the templates
    rectangle = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    l_shape = Polygon([(0, 0), (1, 0), (1, 0.5), (0.5, 0.5), (0.5, 1), (0, 1)])
    u_shape = Polygon([(0, 0), (1, 0), (1, 1), (0.75, 1), (0.75, 0.25), (0.25, 0.25), (0.25, 1), (0, 1)])
    x_shape = Polygon([(0.25, 0), (0.75, 0), (0.75, 0.25), (1, 0.25), (1, 0.75), (0.75, 0.75), (0.75, 1), (0.25, 1), (0.25, 0.75), (0, 0.75), (0, 0.25), (0.25, 0.25)])


    # Given polygon
    # polygon = Polygon([(0, 0), (1, 0), (1, 1), (0.5, 1), (0.5, 0.5), (0, 0.5)])
    # polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    templates = {
        0: rectangle,
        1: l_shape,
        2: u_shape,
        3: x_shape
    }

    # Compute the minimum Hausdorff distance for each template and select the best one
    min_distances = {}

    for name, template in templates.items():

        # Check if the given polygon is a MultiPolygon
        if isinstance(polygon, MultiPolygon):
            min_distance_polygons = []
            for single_polygon in polygon:
                min_distance_polygons.append(compute_min_hausdorff_distance(np.array(single_polygon.exterior.coords[:-1]),
                                                                            np.array(template.exterior.coords[:-1])))
            min_distance = min(min_distance_polygons)
        else:
            min_distance = compute_min_hausdorff_distance(np.array(polygon.exterior.coords[:-1]),
                                                          np.array(template.exterior.coords[:-1]))

        # Check if the template is a MultiPolygon
        if isinstance(template, MultiPolygon):
            min_distance_templates = []
            for single_template in template:
                min_distance_templates.append(compute_min_hausdorff_distance(np.array(polygon.exterior.coords[:-1]),
                                                                             np.array(
                                                                                 single_template.exterior.coords[:-1])))
            min_distance = min(min_distance, min(min_distance_templates))

        min_distances[name] = min_distance

    # Get the template with the smallest distance
    best_template_name = min(min_distances, key=min_distances.get)
    best_distance = min_distances[best_template_name]

    shape = best_template_name
    iou = 100 - best_distance

    return shape, iou