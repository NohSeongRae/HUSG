import random
import math

def sample_points(p1, p2, n):
    """
    Uniformly sample `n` points between two endpoints `p1` and `p2`.

    Args:
    - p1 (list): Starting point [x, y].
    - p2 (list): Ending point [x, y].
    - n (int): Number of points to sample.

    Returns:
    - list: List of sampled points including `p1` and `p2`.
    """
    if n < 2:
        raise ValueError("n should be at least 2 to include both endpoints.")

    x1, y1 = p1
    x2, y2 = p2
    sampled_points = [p1, p2]

    for _ in range(n - 2):
        x = random.uniform(x1, x2)
        y = random.uniform(y1, y2)
        sampled_points.append([x, y])

    return sampled_points


def compute_distance(p1, p2):
    """
    Compute the Euclidean distance between two points `p1` and `p2`.

    Args:
    - p1 (list): First point [x, y].
    - p2 (list): Second point [x, y].

    Returns:
    - float: Euclidean distance between `p1` and `p2`.
    """
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def interpolate_points(p1, p2, step=2):
    """
    Interpolate points between two endpoints `p1` and `p2` based on a given step.

    Args:
    - p1 (list): Starting point [x, y].
    - p2 (list): Ending point [x, y].
    - step (int, optional): Distance between interpolated points. Default is 2.

    Returns:
    - list: List of interpolated points between `p1` and `p2`.
    """
    distance = compute_distance(p1, p2)
    num_points = int(distance / step)
    points = [p1]

    for i in range(1, num_points):
        t = i / num_points
        x = round((1 - t) * p1[0] + t * p2[0])
        y = round((1 - t) * p1[1] + t * p2[1])
        points.append([x, y])

    points.append(p2)

    sampled_points = []
    for i in range(len(points) - 1):
        sampled_points.append(sample_points(p1=points[i], p2=points[i + 1], n=8))

    return sampled_points