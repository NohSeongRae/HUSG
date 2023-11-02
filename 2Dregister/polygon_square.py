import math
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import csv
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
from shapely.wkt import loads

import os


def normalize_polygon_coordinates(coordinates):
    # Extract x and y coordinates
    x_coords = [coord[0] for coord in coordinates]
    y_coords = [coord[1] for coord in coordinates]

    # Find the min and max values for x and y
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # Normalize the coordinates
    normalized_coords = [
        ((x - min_x) / (max_x - min_x), (y - min_y) / (max_y - min_y))
        for x, y in coordinates
    ]

    return normalized_coords

root_path = os.path.join("Z:", "iiixr-drive", "Projects", "2023_City_Team", "5_ACAP")

square_pkl_path = os.path.join(root_path, "normalized_square_test.pkl")
polygon_pkl_path = os.path.join(root_path, "normalized_polygon_test.pkl")

pickle_file_path = os.path.join(root_path, "example.pkl")

with open(pickle_file_path, 'rb') as f:
    coordinates = pickle.load(f)

normalized_coordinates = normalize_polygon_coordinates(coordinates.exterior.coords)

# 폴리곤 객체 생성
polygon_obj = Polygon(normalized_coordinates)


# 폴리곤의 중심 좌표 (정확히는 폴리곤 영역의 중심)
poly_centroid = polygon_obj.centroid
poly_cx, poly_cy = poly_centroid.x, poly_centroid.y

# 정사각형의 중심 좌표
min_x, min_y, max_x, max_y = polygon_obj.bounds
cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2

# 폴리곤이 아래쪽에 정렬되도록 하는 이동
dx, dy = cx - poly_cx, cy - max_y
moved_coordinates = [[x+dx, y+dy] for x, y in list(normalized_coordinates)]

# 정사각형의 점들을 생성
num_points = 64
points_per_side = num_points // 4
points = []
side = max(max_x - min_x, max_y - min_y)

# 각 변에 점들 배치
for i in range(points_per_side):
    t = i / points_per_side
    points.append((min_x + t * side, min_y))  # 아래쪽 변
    points.append((min_x + side, min_y + t * side))  # 오른쪽 변
    points.append((min_x + (1 - t) * side, min_y + side))  # 위쪽 변
    points.append((min_x, min_y + (1 - t) * side))  # 왼쪽 변

# 데이터 정규화
scaler = MinMaxScaler()
all_points = points + moved_coordinates
scaler.fit(all_points)

normalized_points = scaler.transform(points)
normalized_polygon = scaler.transform(moved_coordinates)



# Save the normalized points to a pickle file
with open(square_pkl_path, 'wb') as f:
    pickle.dump(normalized_points, f)

# Save the normalized polygon coordinates to a pickle file
with open(polygon_pkl_path, 'wb') as f:
    pickle.dump(normalized_polygon, f)

# 주어진 폴리곤의 점들 그리기
plt.plot(*zip(*normalized_polygon), 'ro')  # 'ro'는 red dots를 의미합니다.

# 정사각형의 점들 그리기
plt.plot(*zip(*normalized_points), 'bo')  # 'bo'는 blue dots를 의미합니다.

# 정사각형과 폴리곤을 동시에 보여주기 위해 x, y 축의 범위를 동일하게 설정합니다.
plt.axis('equal')

plt.show()
