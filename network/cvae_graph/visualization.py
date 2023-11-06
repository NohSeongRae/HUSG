import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from torch_geometric.utils import to_dense_adj
import os

def get_bbox_corners(x, y, w, h):
    # Calculate half width and half height
    half_w = w / 2
    half_h = h / 2

    # Calculate corners
    top_left = [x - half_w, y - half_h]
    top_right = [x + half_w, y - half_h]
    bottom_left = [x - half_w, y + half_h]
    bottom_right = [x + half_w, y + half_h]

    return [top_left, top_right, bottom_right, bottom_left]


def rotate_points_around_center(points, center, theta_deg):
    # if theta_deg > 90:
    #     theta_deg = 180 - theta_deg
    # else:
    #     theta_deg = theta_deg

    # Convert theta from degrees to radians
    theta_rad = np.radians(theta_deg)

    # Create a rotation matrix
    rotation_matrix = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad), np.cos(theta_rad)]
    ])

    # Convert points and center to numpy arrays
    points = np.array(points)
    center = np.array(center)

    # Translate points so that center is at the origin
    translated_points = points - center

    # Rotate points
    rotated_points = np.dot(translated_points, rotation_matrix.T)

    # Translate points back
    rotated_points = rotated_points + center

    return rotated_points

def plot(pos, size, rot, building_exist_mask, gt, condition, idx, condition_type, is_chunk_graph, edge_index):
    node_x = []
    node_y = []

    # Create a figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    rotation_scale = 45

    if condition_type == 'image':
        ax1.imshow(condition, cmap='gray', extent=[0, 1, 0, 1], alpha=0.5)
        ax2.imshow(condition, cmap='gray', extent=[0, 1, 0, 1], alpha=0.5)
    else:
        if not is_chunk_graph:
            for street in condition:
                x = street[:, 0]
                y = street[:, 1]
                ax1.scatter(x, y, s=0.01)
                ax2.scatter(x, y, s=0.01)
        else:
            for i in range(len(condition)):
                x, y, w, h, theta = condition[i][0], condition[i][1], condition[i][2], condition[i][3], condition[i][4]
                points = get_bbox_corners(x, y, w, h)
                rotated_points = rotate_points_around_center(points, [x, y], theta)

                rotated_points = np.array(rotated_points)
                rotated_box = np.concatenate((rotated_points, [rotated_points[0]]), axis=0)
                ax2.plot(rotated_box[:, 0], rotated_box[:, 1], label='Rotated Box')
                ax2.plot(rotated_box[:, 0], rotated_box[:, 1], label='Rotated Box')

                node_x.append(x)
                node_y.append(y)

    for i in range(len(pos)):
        if building_exist_mask[i] == 0:
            continue

        x, y, w, h, theta = pos[i][0], pos[i][1], size[i][0], size[i][1], (rot[i][0] * 2 - 1) * rotation_scale
        points = get_bbox_corners(x, y, w, h)
        rotated_points = rotate_points_around_center(points, [x, y], theta)

        rotated_points = np.array(rotated_points)
        rotated_box = np.concatenate((rotated_points, [rotated_points[0]]), axis=0)
        ax1.plot(rotated_box[:, 0], rotated_box[:, 1], 'r-', label='Rotated Box')

    for i in range(len(pos)):
        if building_exist_mask[i] == 0:
            continue

        x, y, w, h, theta = gt[i][0], gt[i][1], gt[i][2], gt[i][3], (gt[i][4] * 2 - 1) * rotation_scale
        points = get_bbox_corners(x, y, w, h)
        rotated_points = rotate_points_around_center(points, [x, y], theta)

        rotated_points = np.array(rotated_points)
        rotated_box = np.concatenate((rotated_points, [rotated_points[0]]), axis=0)
        ax2.plot(rotated_box[:, 0], rotated_box[:, 1], 'r-', label='Rotated Box')

        node_x.append(x)
        node_y.append(y)

    if is_chunk_graph:
        num_nodes = len(pos)
        adj_matrix = np.zeros((num_nodes, num_nodes))
        adj_matrix[edge_index[0], edge_index[1]] = 1
        print(adj_matrix)

        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix)):
                if adj_matrix[i][j] == 1 and adj_matrix[j][i] == 1:
                    ax2.plot([node_x[i], node_x[j]], [node_y[i], node_y[j]], alpha=0.5)

    # Set the limits of the plot
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])

    # Set the aspect of the plot to be equal
    ax1.set_aspect('equal', adjustable='box')
    ax2.set_aspect('equal', adjustable='box')

    ax1.set_title('Prediction')
    ax1.grid(True)
    ax2.set_title('Ground Truth')
    ax2.grid(True)

    # Set the limits of the plot
    ax1.set_xlim([-0.1, 1.1])  # x축 범위 설정
    ax1.set_ylim([-0.1, 1.1])  # y축 범위 설정
    ax2.set_xlim([-0.1, 1.1])  # x축 범위 설정
    ax2.set_ylim([-0.1, 1.1])  # y축 범위 설정

    # Save the plot
    directory = "./images"  # 변경: 저장 경로를 /mnt/data/ 아래로 지정
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 이미지 파일로 저장
    save_path = os.path.join(directory, "cvae" + str(idx) + ".png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')