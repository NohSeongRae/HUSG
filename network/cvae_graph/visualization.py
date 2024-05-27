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
    # This function should return the four corners of the bounding box
    return [
        [x - w / 2, y - h / 2],
        [x + w / 2, y - h / 2],
        [x + w / 2, y + h / 2],
        [x - w / 2, y + h / 2]
    ]

def rotate_points_around_center(points, center, angle):
    # This function should rotate points around the given center by the given angle
    angle_rad = np.radians(angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    cx, cy = center
    rotated_points = []
    for x, y in points:
        tx, ty = x - cx, y - cy
        rx = tx * cos_angle - ty * sin_angle
        ry = tx * sin_angle + ty * cos_angle
        rotated_points.append([rx + cx, ry + cy])
    return rotated_points

def calculate_angle(p1, p2):
    """
    Calculate the angle of the line segment connecting points p1 and p2.
    """
    return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])  # Correct order

def find_closest_boundary_segment(building_center, boundary_coords):
    """
    Find the closest boundary segment to the given building center
    and return the angle of that segment and the points of the closest segment.
    """
    min_distance = float('inf')
    closest_angle = None
    closest_segment = None

    for i in range(len(boundary_coords) - 1):
        p1 = boundary_coords[i]
        p2 = boundary_coords[i + 1]
        segment_angle = calculate_angle(p1, p2)

        # Calculate the midpoint of the boundary segment
        segment_midpoint = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]

        # Calculate the distance between the building center and the segment midpoint
        dist = np.linalg.norm(np.array(building_center) - np.array(segment_midpoint))

        if dist < min_distance:
            min_distance = dist
            closest_angle = segment_angle
            closest_segment = (p1, p2)

    return closest_angle, closest_segment

def find_best_alignment_angle(building_points, boundary_coords):
    """
    Find the best alignment angle for the building to align with the boundary.
    """
    min_angle_diff = float('inf')
    best_alignment_angle = None

    for i in range(len(building_points)):
        p1 = building_points[i]
        p2 = building_points[(i + 1) % len(building_points)]
        building_segment_angle = calculate_angle(p1, p2)

        for j in range(len(boundary_coords) - 1):
            b1 = boundary_coords[j]
            b2 = boundary_coords[j + 1]
            boundary_segment_angle = calculate_angle(b1, b2)

            angle_diff = (boundary_segment_angle - building_segment_angle + np.pi) % (2 * np.pi) - np.pi

            if abs(angle_diff) < abs(min_angle_diff):
                min_angle_diff = angle_diff
                best_alignment_angle = building_segment_angle + angle_diff

    return best_alignment_angle

def plot(pos, size, rot, building_exist_mask, gt_features, idx, condition_type, polygon_path=None, save_dir_path='', data_path=None):
    directory = f"./synthetic_images_{condition_type}/{save_dir_path}/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    boundary_coords = []
    for i in range(len(pos)):
        if building_exist_mask[i] == 1:
            continue
        x, y = gt_features[i][0], gt_features[i][1]
        boundary_coords.append([x, y])

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
    fig, ax2 = plt.subplots(1, 1, figsize=(6, 6))

    pred_output_list = []
    for i in range(len(pos)):
        if building_exist_mask[i] == 0:
            continue

        x, y, w, h = pos[i][0], pos[i][1], size[i][0], size[i][1]

        points = get_bbox_corners(x, y, w, h)
        best_alignment_angle = find_best_alignment_angle(points, boundary_coords)
        angle_deg = np.degrees(best_alignment_angle)

        # Normalize the angle to be within -180 to 180 degrees
        angle_to_apply = (angle_deg + 180) % 360 - 180

        # Apply rotation only if the angle difference is within -45 to 45 degrees
        if -45 <= angle_to_apply <= 45:
            pred_output_list.append([x, y, w, h, angle_to_apply])
        else:
            pred_output_list.append([x, y, w, h, 0])

        rotated_points = rotate_points_around_center(points, [x, y], angle_to_apply)

        rotated_points = np.array(rotated_points)
        rotated_box = np.concatenate((rotated_points, [rotated_points[0]]), axis=0)

        ax1.plot(rotated_box[:, 0], rotated_box[:, 1], color='k', label='Rotated Box')

    gt_output_list = []
    for i in range(len(pos)):
        if building_exist_mask[i] == 0:
            continue
        x, y, w, h, theta = gt_features[i][0], gt_features[i][1], gt_features[i][2], gt_features[i][3], (gt_features[i][4] * 2 - 1) * 45,
        gt_output_list.append([x, y, w, h, theta])

        points = get_bbox_corners(x, y, w, h)
        rotated_points = rotate_points_around_center(points, [x, y], theta)

        rotated_points = np.array(rotated_points)
        rotated_box = np.concatenate((rotated_points, [rotated_points[0]]), axis=0)

        ax2.plot(rotated_box[:, 0], rotated_box[:, 1], color='k', label='Rotated Box')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.set_axis_off()
    save_path_1 = os.path.join(directory, "prediction_" + data_path.replace('.gpickle', '') + ".png")
    ax1.figure.savefig(save_path_1, dpi=300, bbox_inches='tight')
    plt.close(ax1.figure)  # ax1에 연결된 figure 닫기

    ax2.set_aspect('equal', adjustable='box')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.0])
    ax2.set_axis_off()
    save_path_2 = os.path.join(directory, "ground_truth_" + data_path.replace('.gpickle', '') + ".png")
    ax2.figure.savefig(save_path_2, dpi=300, bbox_inches='tight')
    plt.close(ax2.figure)  # ax2에 연결된 figure 닫기

    with open(save_path_1.replace('.png', '.pkl'), 'wb') as file:
        pickle.dump(pred_output_list, file)

    with open(save_path_2.replace('.png', '.pkl'), 'wb') as file:
        pickle.dump(gt_output_list, file)