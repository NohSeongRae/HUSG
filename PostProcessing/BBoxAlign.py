import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def calculate_angle(p1, p2):
    """
    Calculate the angle of the line segment connecting points p1 and p2.
    """
    return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])


def find_closest_boundary_segment(building_facade_centers, boundary_coords):
    """
    Find the closest boundary segment to the given building facade centers
    and return the angle of that segment and the points of the closest segment.
    """
    min_distance = float('inf')
    closest_angle = None
    closest_segment = None
    closest_facade = None

    for facade_center in building_facade_centers:
        for i in range(len(boundary_coords) - 1):
            p1 = boundary_coords[i]
            p2 = boundary_coords[i + 1]
            segment_angle = calculate_angle(p1, p2)

            # Calculate the midpoint of the boundary segment
            segment_midpoint = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]

            # Calculate the distance between the facade center and the segment midpoint
            dist = np.linalg.norm(np.array(facade_center) - np.array(segment_midpoint))

            if dist < min_distance:
                min_distance = dist
                closest_angle = segment_angle
                closest_segment = (p1, p2)
                closest_facade = facade_center

    return closest_angle, closest_segment, closest_facade


def get_building_facade_centers(center, width, height, angle):
    """
    Calculate the centers of the four facades of the building given its center, width, height, and angle.
    """
    angle_rad = np.radians((angle * 2 - 1) * 45)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    w, h = width / 2, height / 2

    facades = [
        (0, -h),
        (0, h),
        (-w, 0),
        (w, 0)
    ]

    rotated_facades = []
    for x, y in facades:
        x_rot = center[0] + x * cos_angle - y * sin_angle
        y_rot = center[1] + x * sin_angle + y * cos_angle
        rotated_facades.append((x_rot, y_rot))

    return rotated_facades


# Load data
file_path = './sample_data/prediction_10267.pkl'
boundary_file_path = './sample_data/10267.pickle'

with open(file_path, 'rb') as f:
    polygon_data = pickle.load(f)
with open(boundary_file_path, 'rb') as f:
    boundary_data = pickle.load(f)

boundary_pos_feature = boundary_data['boundary_pos_feature']
boundary_pos_feature = np.vstack([boundary_pos_feature, boundary_pos_feature[0]])

# Process angles
boundary_coords = boundary_pos_feature
building_angles = []
facade_to_boundary_lines = []

# Copy original polygon data for comparison
original_polygon_data = [rect.copy() for rect in polygon_data]

for i, rect in enumerate(polygon_data):
    x_center, y_center, width, height, angle = rect
    angle = np.degrees(np.radians((angle * 2 - 1) * 45))
    building_center = [x_center, y_center]
    facade_centers = get_building_facade_centers(building_center, width, height, angle)
    closest_angle, closest_segment, closest_facade = find_closest_boundary_segment(facade_centers, boundary_coords)
    # rect[4] = closest_angle / np.pi  # Update the angle
    rect[4] = closest_angle
    building_angles.append(closest_angle)
    facade_to_boundary_lines.append((closest_facade, closest_segment))

# Visualization with building indices

# Plot original data
fig, ax = plt.subplots(1, 2, figsize=(14, 7))

# Original polygon data
ax[0].set_title("Original Data")
x, y = zip(*boundary_coords)
ax[0].plot(x, y, color='blue', label='Boundary Polygon')
for i, rect in enumerate(original_polygon_data):
    x_center, y_center, width, height, angle = rect
    print(angle)
    angle = np.degrees(np.radians((angle * 2 - 1) * 45))  # Convert original angle to radians and then to degrees
    rect_patch = patches.Rectangle(
        (x_center - width / 2, y_center - height / 2), width, height,
        angle=angle, edgecolor='black', facecolor='none'
    )
    ax[0].add_patch(rect_patch)
    ax[0].text(x_center, y_center, str(i), color='red', ha='center', va='center')

ax[0].set_xlim(0, 1.1)
ax[0].set_ylim(0, 1.1)
ax[0].set_aspect('equal')

# Transformed polygon data
ax[1].set_title("Transformed Data")
ax[1].plot(x, y, color='blue', label='Boundary Polygon')
for i, rect in enumerate(polygon_data):
    x_center, y_center, width, height, angle = rect
    # angle = np.radians((angle * 2 - 1) * 45)   # Convert radian angle to degrees
    rect_patch = patches.Rectangle(
        (x_center - width / 2, y_center - height / 2), width, height,
        angle=angle, edgecolor='black', facecolor='none'
    )
    ax[1].add_patch(rect_patch)
    ax[1].text(x_center, y_center, str(i), color='red', ha='center', va='center')

# Plot the facade to boundary segment lines
for facade_center, boundary_segment in facade_to_boundary_lines:
    p1, p2 = boundary_segment
    segment_midpoint = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
    closest_point = facade_center
    ax[1].plot([closest_point[0], segment_midpoint[0]], [closest_point[1], segment_midpoint[1]], 'r--')

ax[1].set_xlim(0, 1.1)
ax[1].set_ylim(0, 1.1)
ax[1].set_aspect('equal')

plt.show()

# Print building indices and their angles
for i, angle in enumerate(building_angles):
    print(f"Building {i}: Angle (radians) = {angle}, Angle (degrees) = {np.degrees(angle)}")