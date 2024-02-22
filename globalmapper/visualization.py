import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

def get_random_color(seed):
    palette = plt.cm.tab10
    norm_index = seed / 10
    color = palette(norm_index)
    return color

def get_bbox_corners(x, y, w, h):
    half_w = w / 2
    half_h = h / 2

    top_left = [x - half_w, y - half_h]
    top_right = [x + half_w, y - half_h]
    bottom_left = [x - half_w, y + half_h]
    bottom_right = [x + half_w, y + half_h]

    return [top_left, top_right, bottom_right, bottom_left]

def rotate_points_around_center(points, center, theta_deg):
    theta_rad = np.radians(theta_deg)

    rotation_matrix = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad), np.cos(theta_rad)]
    ])

    points = np.array(points)
    center = np.array(center)
    translated_points = points - center

    rotated_points = np.dot(translated_points, rotation_matrix.T)
    rotated_points = rotated_points + center

    return rotated_points

def plot(pos, size, rot, building_exist_mask, gt_building_exist_mask, gt_features, idx, condition_type, polygon_path=None, save_dir_path='', data_path=None):
    directory = f"./images_{condition_type}/{save_dir_path}/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    save_path_2 = os.path.join(directory, "file_name_" + str(idx) + ".png")
    with open(save_path_2.replace('.png', '.pkl'), 'wb') as file:
        file_name = [data_path[0]]
        print(file_name)

        if 'stockholm' in file_name[0]:
            print('It is stockholm')
            return

        pickle.dump(file_name, file)

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
    fig, ax2 = plt.subplots(1, 1, figsize=(6, 6))
    rotation_scale = 45

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
        if gt_building_exist_mask[i] == 0:
            continue
        x, y, w, h, theta = gt_features[i][0], gt_features[i][1], gt_features[i][2], gt_features[i][3], (gt_features[i][4] * 2 - 1) * rotation_scale,
        gt_output_list.append([x, y, w, h, theta])

    if polygon_path == None:
        filepath = f'/local_datasets/{condition_type}_condition_train_datasets/' + 'test/' + str(idx - 1) + '.pkl'
        with open(filepath, 'rb') as f:
            building_polygons = pickle.load(f)
    else:
        filepath = f'/local_datasets/{condition_type}_condition_train_datasets/' + 'test/' + polygon_path[0]
        with open(filepath, 'rb') as f:
            building_polygons = pickle.load(f)

    if data_path != None:
        filepath = f'/local_datasets/{condition_type}_condition_train_datasets/' + 'test/' + data_path[0]
        with open(filepath, 'rb') as f:
            gpickle_file = pickle.load(f)

    for building_polygon in building_polygons:
        x, y = building_polygon
        ax2.plot(x, y, color='k', label='Rotated Box')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.set_axis_off()
    save_path_1 = os.path.join(directory, "prediction_" + str(idx) + ".png")
    ax1.figure.savefig(save_path_1, dpi=300, bbox_inches='tight')

    ax2.set_aspect('equal', adjustable='box')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.0])
    ax2.set_axis_off()
    save_path_2 = os.path.join(directory, "ground_truth_" + str(idx) + ".png")
    ax2.figure.savefig(save_path_2, dpi=300, bbox_inches='tight')

    with open(save_path_1.replace('.png', '.pkl'), 'wb') as file:
        pickle.dump(pred_output_list, file)

    with open(save_path_2.replace('.png', '.pkl'), 'wb') as file:
        pickle.dump(gt_output_list, file)

    with open(save_path_2.replace('.png', '.gpickle'), 'wb') as file:
        pickle.dump(gpickle_file, file)

    save_path_2 = os.path.join(directory, "real_polygon_" + str(idx) + ".png")
    with open(save_path_2.replace('.png', '.pkl'), 'wb') as file:
        pickle.dump(building_polygons, file)

    print(save_path_1)