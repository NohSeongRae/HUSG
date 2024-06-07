import numpy as np
from shapely.geometry import Polygon
from skimage.draw import polygon as draw_polygon
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import networkx as nx

def normalize_polygon(polygon, scale_factor, center_x, center_y):
    xx, yy = polygon.exterior.xy
    normalized_coords = []
    for x, y in zip(xx, yy):
        norm_x = (x - center_x) * scale_factor + 0.5  # Shift by 0.5 for center
        norm_y = (y - center_y) * scale_factor + 0.5  # Shift by 0.5 for center
        normalized_coords.append((norm_x, norm_y))
    return Polygon(normalized_coords)

def create_mask(boundary_polygon, mask_size=(224, 224)):
    # Create a blank mask
    mask = np.zeros(mask_size, dtype=np.uint8)

    # Get the coordinates of the polygon's exterior
    x, y = boundary_polygon.exterior.xy

    # Scale coordinates from [0, 1] to the size of the mask
    x = np.array(x) * (mask_size[1] - 1)
    y = np.array(y) * (mask_size[0] - 1)

    # Draw the polygon on the mask
    rr, cc = draw_polygon(y, x, mask.shape)
    mask[rr, cc] = 1

    return mask

def calculate_centroid(polygon):
    centroid = polygon.centroid
    return centroid.x, centroid.y

def calculate_rotation_angle(polygon):
    poly = polygon
    min_rot_rect = poly.minimum_rotated_rectangle
    exterior_coords = list(min_rot_rect.exterior.coords)

    # Find the longest side
    max_length = 0
    angle = 0
    for i in range(len(exterior_coords) - 1):
        p1 = exterior_coords[i]
        p2 = exterior_coords[i + 1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = np.sqrt(dx**2 + dy**2)
        if length > max_length:
            max_length = length
            angle = np.arctan2(dy, dx)  # Angle in radians

    return -angle  # Negative angle to rotate to horizontal

def rotate_polygon(polygon, angle, center_x, center_y):
    rotated_coords = []
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    xx, yy = polygon.exterior.xy
    for x, y in zip(xx, yy):
        tx = x - center_x
        ty = y - center_y
        rot_x = tx * cos_angle - ty * sin_angle + center_x
        rot_y = tx * sin_angle + ty * cos_angle + center_y
        rotated_coords.append((rot_x, rot_y))
    return Polygon(rotated_coords)

bx = [299442.08043341816, 299429.0, 299452.0, 299445.0, 299596.2023476951, 299616.1036794798, 299611.0758156141, 299612.00289841415, 299580.0, 299572.0, 299489.0, 299497.1, 299448.3674750405, 299447.3830464642, 299442.08043341816]
by = [4630850.075960333, 4630894.4, 4630901.0, 4630926.0, 4630969.54915682, 4630900.149130106, 4630898.456773022, 4630895.052578355, 4630886.0, 4630915.0, 4630892.0, 4630861.5, 4630847.7545811, 4630851.482672085, 4630850.075960333]

coord = np.array([bx, by]).transpose()
polygon = Polygon(coord)
min_rot_rect = polygon.minimum_rotated_rectangle
min_x, min_y, max_x, max_y = min_rot_rect.bounds

center_x, center_y = calculate_centroid(polygon)

# Determine the scale factor to normalize the longest side to 1
bbox_width = max_x - min_x
bbox_height = max_y - min_y
max_length = max(bbox_width, bbox_height)
scale_factor = 1 / max_length
polygon = normalize_polygon(polygon, scale_factor, center_x, center_y)

rotation_angle = calculate_rotation_angle(polygon)
rotated_block = rotate_polygon(polygon, rotation_angle, 0.5, 0.5)

mask = create_mask(rotated_block)

# 검사할 방향: 상하좌우 및 대각선
directions = [
    (-1, 0),  # 위
    (1, 0),  # 아래
    (0, -1),  # 왼쪽
    (0, 1),  # 오른쪽
    (-1, -1),  # 왼쪽 위 대각선
    (-1, 1),  # 오른쪽 위 대각선
    (1, -1),  # 왼쪽 아래 대각선
    (1, 1)  # 오른쪽 아래 대각선
]

rows = len(mask)
cols = len(mask[0])

# 2중 포문으로 각 셀 검사
for i in range(rows):
    for j in range(cols):
        is_able = False

        # 각 방향으로 트리거 검사
        for direction in directions:
            ni = i + direction[0]
            nj = j + direction[1]

            # 유효한 인덱스인지 확인
            if 0 <= ni < rows and 0 <= nj < cols:
                if mask[ni][nj] == 0:
                    is_able = True

        if not is_able and 0 < i < rows - 1 and 0 < j < cols - 1:
            mask[i][j] = 2

for i in range(rows):
    for j in range(cols):
        if mask[i][j] == 2:
            mask[i][j] = 0

# Plot the heatmap for the 'condition' attribute with specified colors
custom_cmap = ListedColormap(['black', 'white'])
plt.figure(figsize=(10, 10))
sns.heatmap(mask, cmap=custom_cmap, cbar=False, vmin=0, vmax=1, xticklabels=False, yticklabels=False)
plt.axis('off')  # Remove axes

# Adjust layout to fill the plot area completely
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
