import os
import numpy as np
from shapely.geometry import Polygon
import json
from shapely.geometry import shape


def resample_polygon_to_8_points(polygon):
    coords = list(polygon.exterior.coords)
    # Compute the total perimeter of the polygon
    distances = np.sqrt(np.sum(np.diff(coords, axis=0, append=[coords[0]]) ** 2, axis=1))
    perimeter = np.sum(distances)

    # Compute the desired distance between points
    desired_distance = perimeter / 8

    # Resample the polygon
    resampled_points = [coords[0]]
    distance_accumulated = 0
    point_idx = 0
    for i in range(1, 8):
        while distance_accumulated < desired_distance * i:
            point_idx += 1
            if point_idx >= len(coords):
                point_idx = 0
            distance_accumulated += distances[point_idx]
        resampled_points.append(coords[point_idx])

    return np.array(resampled_points)


import matplotlib.pyplot as plt


def plot_polygon(polygon, title="", color='b', show=True):
    """Plots the given polygon."""
    if isinstance(polygon, Polygon):
        x, y = polygon.exterior.xy
    elif isinstance(polygon, np.ndarray):
        x, y = polygon[:, 0], polygon[:, 1]
    else:
        raise ValueError("Unsupported type for polygon")
    plt.plot(x, y, color + '-o', label='Polygon')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    if show:
        plt.show()


def geojson_to_shapely_polygon(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)

    # Assuming the GeoJSON contains only one feature (a polygon)
    geometry = data['features'][0]['geometry']
    polygon = shape(geometry)

    return polygon


building_file_path = os.path.join("Z:", "iiixr-drive", "Projects", "2023_City_Team", "atlanta_dataset",
                                  "density20_building120_rotate_normalized", "Buildings",
                                  "atlanta_buildings133.geojson")
complex_polygon = geojson_to_shapely_polygon(building_file_path)

resampled_complex_polygon = resample_polygon_to_8_points(complex_polygon)

# Plot the original and resampled polygons
plot_polygon(complex_polygon, title="Original Polygon", color='b', show=False)
plot_polygon(resampled_complex_polygon, title="Resampled to 8 Points Polygon", color='r')
plt.legend(["Original Polygon", "Resampled Polygon"])
# plt.show()
