import numpy as np
import matplotlib.pyplot as plt
import random
from shapely.geometry import shape
from shapely.geometry import Polygon, Point, LineString
import pickle

seed = 327
random.seed(seed)
np.random.seed(seed)


def subdivide_polygon_exterior(polygon, distance):
    """
    다각형의 외부 좌표를 특정 거리마다 세분화하는 함수.

    Parameters:
        polygon (Polygon): Shapely Polygon 객체
        distance (float): 세분화할 거리

    Returns:
        list: 세분화된 좌표 목록 [(x1, y1), (x2, y2), ...]
    """
    # 다각형의 외곽 라인 추출
    exterior_coords = polygon.exterior.coords
    exterior_line = LineString(exterior_coords)

    # 외곽 라인의 전체 길이
    line_length = exterior_line.length

    # 세분화된 좌표를 저장할 리스트
    subdivided_coords = []

    # 0부터 전체 길이까지 distance 간격으로 좌표를 추출
    for i in np.arange(0, line_length, distance):
        point = exterior_line.interpolate(i)
        subdivided_coords.append((point.x, point.y))

    # 마지막 점 추가
    if subdivided_coords[-1] != (exterior_line.coords[-1][0], exterior_line.coords[-1][1]):
        subdivided_coords.append([exterior_line.coords[-1][0], exterior_line.coords[-1][1]])

    return subdivided_coords

# Function to generate a pastel color
def generate_pastel_color():
    base_color = np.array([random.random(), random.random(), random.random()])
    white = np.array([1.0, 1.0, 1.0])
    pastel_color = (base_color + white) / 2
    return pastel_color

def draw_line_with_thickness(x1, y1, x2, y2, thickness):
    # Calculate the angle of the line
    angle = np.arctan2(y2 - y1, x2 - x1)

    # Calculate the offset for the thickness
    dx = thickness / 2 * np.sin(angle)
    dy = thickness / 2 * -np.cos(angle)

    # Coordinates of the four corners of the rectangle
    corners_x = [x1 - dx, x1 + dx, x2 + dx, x2 - dx, x1 - dx]
    corners_y = [y1 - dy, y1 + dy, y2 + dy, y2 - dy, y1 - dy]

    plt.plot(corners_x, corners_y, 'black')

# To load the pickle file and visualize the data again
with open('block_building_info.pkl', 'rb') as f:
    loaded_block_data = pickle.load(f)

# Extract all coordinates from block polygons
all_x_coords = []
all_y_coords = []

for block_idx, block in enumerate(loaded_block_data):
    if block_idx != 47:
        continue

    block_polygon = Polygon(block['block_polygon']['coordinates'][0])
    x_coords, y_coords = block_polygon.exterior.coords.xy
    all_x_coords.extend(x_coords)
    all_y_coords.extend(y_coords)

# Calculate min and max values for x and y coordinates
minx, maxx = min(all_x_coords), max(all_x_coords)
miny, maxy = min(all_y_coords), max(all_y_coords)

block_colors = {}

# Iterate through the loaded block data and add polygons to the map
for block_idx, block_info in enumerate(loaded_block_data):
    if block_idx != 47:
        continue
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))

    block_polygon = shape(block_info["block_polygon"])

    for bbox in block_info["buildings_bbox"]:
        building_bbox = shape(bbox)

        if block_idx not in block_colors:
            block_colors[block_idx] = generate_pastel_color()
        facecolor = block_colors[block_idx]

        px, py = building_bbox.exterior.coords.xy
        ax.fill(px, py, edgecolor='black', facecolor=facecolor)

    # Draw block boundary
    # block_polygon = Polygon(block_info['block_polygon']['coordinates'][0])
    # bx, by = block_polygon.exterior.coords.xy
    sub_coord = np.array(subdivide_polygon_exterior(block_polygon, distance=2))
    for idx in range(len(sub_coord) - 1):
        draw_line_with_thickness(sub_coord[idx, 0], sub_coord[idx, 1],
                                 sub_coord[idx + 1, 0], sub_coord[idx + 1, 1], thickness=1)


    # Set dynamic limits for the plot
    ax.set_xlim([minx - 1, maxx + 1])
    ax.set_ylim([miny - 1, maxy + 1])
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()

    # Save the plot as an image file for each block
    plt.savefig(f'restored_predictions_map_block_{block_idx}.png', dpi=300, bbox_inches='tight')

