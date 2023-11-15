import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from torch_geometric.utils import to_dense_adj
import os
import pickle

def get_random_color(seed):
    """
    Returns a distinct color for each index in the range 0-9.
    """
    # Define a color palette (using matplotlib's tab10 palette)
    palette = plt.cm.tab10
    # Normalize index to be between 0 and 1
    norm_index = seed / 10
    # Get the corresponding color
    color = palette(norm_index)
    return color

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

def plot(pos, size, rot, building_exist_mask, gt_features, gt_semantics, condition, idx, condition_type, edge_index, polygon_path=None, save_dir_path=''):
    # Create a figure and axes
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))  # 한 개의 서브플롯만 생성
    fig, ax2 = plt.subplots(1, 1, figsize=(6, 6))  # 한 개의 서브플롯만 생성
    rotation_scale = 45

    # if condition_type == 'image_resnet34':
    #     ax1.imshow(condition[0], cmap='gray', extent=[0, 1, 0, 1], alpha=0.5)
    #     ax2.imshow(condition[0], cmap='gray', extent=[0, 1, 0, 1], alpha=0.5)

    pred_output_list = []
    for i in range(len(pos)):
        if building_exist_mask[i] == 0:
            continue

        x, y, w, h, theta = pos[i][0], pos[i][1], size[i][0], size[i][1], (rot[i][0] * 2 - 1) * rotation_scale
        pred_output_list.append([x, y, w, h, theta])

        points = get_bbox_corners(x, y, w, h)
        rotated_points = rotate_points_around_center(points, [x, y], theta)

        rotated_points = np.array(rotated_points)
        rotated_box = np.concatenate((rotated_points, [rotated_points[0]]), axis=0)

        ax1.plot(rotated_box[:, 0], rotated_box[:, 1], color='k', label='Rotated Box')

    gt_output_list = []
    for i in range(len(pos)):
        if building_exist_mask[i] == 0:
            continue
        x, y, w, h, theta = gt_features[i][0], gt_features[i][1], gt_features[i][2], gt_features[i][3], (gt_features[i][4] * 2 - 1) * rotation_scale,
        gt_output_list.append([x, y, w, h, theta])

    if polygon_path == None:
        filepath = f'../../../..//local_datasets/{condition_type}_condition_train_datasets/' + 'test/' + str(idx - 1) + '.pkl'
        with open(filepath, 'rb') as f:
            building_polygons = pickle.load(f)
    else:
        filepath = f'../../../..//local_datasets/{condition_type}_condition_train_datasets/' + 'test/' + polygon_path[0]
        with open(filepath, 'rb') as f:
            building_polygons = pickle.load(f)

    for building_polygon in building_polygons:
        x, y = building_polygon
        ax2.plot(x, y, color='k', label='Rotated Box')

    # Set the limits of the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    directory = f"./images_{condition_type}/{save_dir_path}/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # ax1 (예측) 이미지 저장
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.set_axis_off()  # 축 표기 제거
    save_path_1 = os.path.join(directory, "prediction_" + str(idx) + ".png")
    ax1.figure.savefig(save_path_1, dpi=300, bbox_inches='tight')

    # ax2 (실제) 이미지 저장
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.0])
    ax2.set_axis_off()  # 축 표기 제거
    save_path_2 = os.path.join(directory, "ground_truth_" + str(idx) + ".png")
    ax2.figure.savefig(save_path_2, dpi=300, bbox_inches='tight')

    with open(save_path_1.replace('png', '.pkl'), 'wb') as file:
        pickle.dump(pred_output_list, file)

    with open(save_path_2.replace('png', '.pkl'), 'wb') as file:
        pickle.dump(gt_output_list, file)

    print(pred_output_list)
    print(gt_output_list)

    print(save_path_1)

def test_plot(pos, size, rot, semantics, building_exist_mask, gt_features, gt_semantics, condition, idx, condition_type, is_chunk_graph, edge_index):
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
                x, y, w, h, theta = condition[i][0], condition[i][1], condition[i][2], condition[i][3], (condition[i][4] * 2 - 1) * rotation_scale
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

        x, y, w, h, theta, semantic = pos[i][0], pos[i][1], size[i][0], size[i][1], (rot[i][0] * 2 - 1) * rotation_scale, semantics[i]
        points = get_bbox_corners(x, y, w, h)
        rotated_points = rotate_points_around_center(points, [x, y], theta)

        rotated_points = np.array(rotated_points)
        rotated_box = np.concatenate((rotated_points, [rotated_points[0]]), axis=0)

        semantic = np.argmax(semantic)
        ax1.plot(rotated_box[:, 0], rotated_box[:, 1], color=get_random_color(semantic), label='Rotated Box')

    for i in range(len(pos)):
        if building_exist_mask[i] == 0:
            continue

        x, y, w, h, theta = gt_features[i][0], gt_features[i][1], gt_features[i][2], gt_features[i][3], (gt_features[i][4] * 2 - 1) * rotation_scale,
        semantic = gt_semantics[i]
        points = get_bbox_corners(x, y, w, h)
        rotated_points = rotate_points_around_center(points, [x, y], theta)

        rotated_points = np.array(rotated_points)
        rotated_box = np.concatenate((rotated_points, [rotated_points[0]]), axis=0)
        ax2.plot(rotated_box[:, 0], rotated_box[:, 1], color=get_random_color(semantic), label='Rotated Box')

        node_x.append(x)
        node_y.append(y)

    if is_chunk_graph and condition_type == 'graph':
        num_nodes = len(pos)
        adj_matrix = np.zeros((num_nodes, num_nodes))
        adj_matrix[edge_index[0], edge_index[1]] = 1

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