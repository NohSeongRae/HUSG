import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms

def calculate_angle(p1, p2):
    """
    Calculate the angle of the line segment connecting points p1 and p2.
    """
    return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

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

# Load data
file_path = './sample_data/prediction_16.pkl'
boundary_file_path = './sample_data/16.pickle'

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

# Visualization with building indices

# Plot original data
fig, ax = plt.subplots(1, 2, figsize=(14, 7))

# Original polygon data
ax[0].set_title("Original Data")
x, y = zip(*boundary_coords)
ax[0].plot(x, y, color='blue', label='Boundary Polygon')
for i, rect in enumerate(original_polygon_data):
    x_center, y_center, width, height, _ = rect  # Ignore the original angle
    rect_patch = patches.Rectangle(
        (x_center - width / 2, y_center - height / 2), width, height,
        angle=0, edgecolor='black', facecolor='none'
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
    x_center, y_center, width, height, _ = rect  # Ignore the original angle
    closest_angle, closest_segment = find_closest_boundary_segment([x_center, y_center], boundary_coords)
    angle = np.degrees(closest_angle)  # Convert radians to degrees
    rect[4] = angle  # Update the angle in the polygon data
    building_angles.append(closest_angle)
    facade_to_boundary_lines.append(([x_center, y_center], closest_segment))

    rect_patch = patches.Rectangle(
        (x_center - width / 2, y_center - height / 2), width, height,
        edgecolor='black', facecolor='none'
    )

    # Create a transformation for the rotation around the center of the rectangle
    t = transforms.Affine2D().rotate_deg_around(x_center, y_center, angle) + ax[1].transData
    rect_patch.set_transform(t)

    ax[1].add_patch(rect_patch)
    ax[1].text(x_center, y_center, str(i), color='red', ha='center', va='center')

# Plot the lines from building center to the closest boundary segment midpoint
for building_center, boundary_segment in facade_to_boundary_lines:
    p1, p2 = boundary_segment
    segment_midpoint = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
    closest_point = building_center
    ax[1].plot([closest_point[0], segment_midpoint[0]], [closest_point[1], segment_midpoint[1]], 'r--')

ax[1].set_xlim(0, 1.1)
ax[1].set_ylim(0, 1.1)
ax[1].set_aspect('equal')

plt.show()

# Print building indices and their angles
for i, angle in enumerate(building_angles):
    print(f"Building {i}: Angle (radians) = {angle}, Angle (degrees) = {np.degrees(angle)}")