import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
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
    # if theta_deg < 45:
    #     theta_deg = theta_deg
    # else:
    #     theta_deg = 90 - theta_deg

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

def plot(pos, size, rot, mask, gt, idx):
    # Create a figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    for i in range(len(pos)):
        if mask[i] == 0:
            continue

        x, y, w, h, theta = pos[i][0], pos[i][1], size[i][0], size[i][1], rot[i][0] * -180
        points = get_bbox_corners(x, y, w, h)
        rotated_points = rotate_points_around_center(points, [x, y], theta)

        rotated_points = np.array(rotated_points)
        rotated_box = np.concatenate((rotated_points, [rotated_points[0]]), axis=0)
        original_box = np.concatenate((points, [points[0]]), axis=0)
        ax1.plot(rotated_box[:, 0], rotated_box[:, 1], 'r-', label='Rotated Box')
        ax1.plot(original_box[:, 0], original_box[:, 1], 'b-', label='Rotated Box')

    # Set the limits of the plot
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])

    # Set the aspect of the plot to be equal
    ax1.set_aspect('equal', adjustable='box')

    for i in range(len(pos)):
        if mask[i] == 0:
            continue

        x, y, w, h, theta = gt[i][0], gt[i][1], gt[i][2], gt[i][3], gt[i][4] * 90
        points = get_bbox_corners(x, y, w, h)
        rotated_points = rotate_points_around_center(points, [x, y], theta)

        rotated_points = np.array(rotated_points)
        rotated_box = np.concatenate((rotated_points, [rotated_points[0]]), axis=0)
        original_box = np.concatenate((points, [points[0]]), axis=0)
        ax2.plot(rotated_box[:, 0], rotated_box[:, 1], 'r-', label='Rotated Box')
        ax2.plot(original_box[:, 0], original_box[:, 1], 'b-', label='Rotated Box')


    # Set the aspect of the plot to be equal
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