import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import os

def plot(pos, size, rot, mask, idx):
    # Create a figure and axes
    fig, ax = plt.subplots()

    for i in range(len(pos)):
        if mask[i] == 0:
            continue

        x, y, w, h, theta = pos[i][0], pos[i][1], size[i][0], size[i][1], rot[i] * 90
        # Calculate top-left corner coordinates from center (x, y)
        top_left_x = x - w / 2
        top_left_y = y - h / 2

        # Create a rectangle patch
        rect = patches.Rectangle((0, 0), w, h, linewidth=1, edgecolor='r', facecolor='none')

        # Create an Affine transformation
        t = transforms.Affine2D().rotate_deg_around(x, y, theta).translate(top_left_x, top_left_y) + ax.transData

        # Set the transformation to the rectangle
        rect.set_transform(t)

        # Add the patch to the Axes
        ax.add_patch(rect)

    # Set the limits of the plot
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])

    # Set the aspect of the plot to be equal
    ax.set_aspect('equal', adjustable='box')

    # Save the plot
    directory = "./images"  # 변경: 저장 경로를 /mnt/data/ 아래로 지정
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 이미지 파일로 저장
    save_path = os.path.join(directory, "cvae" + str(idx) + ".png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')