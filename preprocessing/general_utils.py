import numpy as np
import math
from shapely.geometry import LineString

def rotated_line_90(p1, p2, unit_length, scale=5):
    v = p2 - p1
    R = np.array([[0, 1], [-1, 0]])
    v_rotated = np.dot(R, v)
    magnitude_v = np.linalg.norm(v_rotated)
    v_normalized = v_rotated / magnitude_v

    v_rotated = [(p1 + p2) / 2, (p1 + p2) / 2 + v_normalized * unit_length * scale]
    return np.array(v_rotated)

def rotated_line_90_v2(p1, p2, unit_length, scale=5):
    v = p2 - p1
    R = np.array([[0, 1], [-1, 0]])
    v_rotated = np.dot(R, v)
    magnitude_v = np.linalg.norm(v_rotated)
    v_normalized = v_rotated / magnitude_v

    v_rotated = [(p1 + p2) / 2 + v_normalized * unit_length * -scale,
                 (p1 + p2) / 2 + v_normalized * unit_length * scale]
    return np.array(v_rotated)

def pad_list(input_list, target_length, pad_idx):
    while len(input_list) < target_length:
        input_list.append(pad_idx)
    return input_list

def get_segments_as_lists(polygon):
    # Polygon의 외곽선 좌표를 가져옴
    exterior_coords = list(polygon.exterior.coords)

    # 연속적인 좌표쌍을 사용하여 선분의 좌표 리스트 생성
    segments_coords = [[list(exterior_coords[i]), list(exterior_coords[i + 1])] for i in
                       range(len(exterior_coords) - 1)]

    return segments_coords

def extract_line_segments(polygon):
    coords = list(polygon.exterior.coords)
    line_segments = []

    # Iterate through all coordinate pairs and create a line segment
    for i in range(1, len(coords) - 1):  # Skipping the last segment to not connect the last and first points
        segment = [coords[i - 1], coords[i]]
        line_segments.append(segment)

    return line_segments

def random_sample_points_on_multiple_lines(lines, m):
    """
    lines: [(A1, B1), (A2, B2), ...]의 형태로 주어지는 n개의 직선
    m: 샘플링할 점의 개수
    """

    sampled_points = []

    for _ in range(m):
        # n개의 직선 중 하나를 랜덤하게 선택
        A, B = lines[np.random.randint(len(lines))]
        # 선택된 직선에서 점을 랜덤 샘플링
        point = random_sample_points_on_line(A, B, 1)[0]
        sampled_points.append(point)

    return np.array(sampled_points)

def random_sample_points_on_line(A, B, target_number):
    # 두 점 A와 B 사이의 벡터를 계산
    AB = np.array(B) - np.array(A)

    # 직선 위의 점들을 저장할 리스트 초기화
    sampled_points = []

    for _ in range(target_number):
        # 랜덤한 t 값을 [0, 1] 범위에서 선택
        t = np.random.uniform(0, 1)
        # 직선 위의 랜덤한 점을 계산
        point = A + t * AB
        sampled_points.append(point)

    return np.array(sampled_points)

def compute_distance(point1, point2):
    """Compute the Euclidean distance between two points."""
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)