import numpy as np
import matplotlib.pyplot as plt
import os
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import LineString
import json
from shapely.geometry import shape
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')
# Function to numerically approximate the gradient of the loss function
def approximate_gradient(original_polygon, approx_polygon, epsilon=1e-6):
    gradient = np.zeros_like(approx_polygon)
    for i in range(len(approx_polygon)):
        for j in range(2):  # We have 2 coordinates x and y
            original_value = approx_polygon[i, j]
            approx_polygon[i, j] += epsilon
            loss_plus_epsilon = calculate_loss(original_polygon, approx_polygon)
            approx_polygon[i, j] = original_value  # Reset to original value
            approx_polygon[i, j] -= epsilon
            loss_minus_epsilon = calculate_loss(original_polygon, approx_polygon)
            approx_polygon[i, j] = original_value  # Reset to original value
            gradient[i, j] = (loss_plus_epsilon - loss_minus_epsilon) / (2 * epsilon)
    return gradient

# Function to calculate the loss function as the sum of squared distances
def calculate_loss(original_polygon, approx_polygon):
    shapely_original_polygon = Polygon(original_polygon)
    dense_points = [shapely_original_polygon.exterior.interpolate(t, normalized=True)
                    for t in np.linspace(0, 1, 200)]
    edges = [LineString([approx_polygon[i], approx_polygon[(i + 1) % len(approx_polygon)]])
             for i in range(len(approx_polygon))]
    loss = sum(min(edge.distance(point)**2 for edge in edges)
               for point in dense_points)
    return loss

# Function to perform a single gradient descent step without enforcing rectilinear shape
def gradient_descent_step_non_rectilinear(original_polygon, approx_polygon, learning_rate=0.01):
    gradient = approximate_gradient(original_polygon, approx_polygon)
    new_approx_polygon = approx_polygon - learning_rate * gradient
    return new_approx_polygon
def geojson_to_shapely_polygon(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)

    # Assuming the GeoJSON contains only one feature (a polygon)
    geometry = data['features'][0]['geometry']
    polygon = shape(geometry)

    return polygon
# Function to convert an optimized polygon into a rectilinear shape
def convert_to_rectilinear(polygon):
    rectilinear_polygon = polygon.copy()
    for i in range(0,8):
        if ((i+1)%2)!=0:
            if i==0:
                print('first point:','x',rectilinear_polygon[0,0],'y',rectilinear_polygon[0,1])
            y_next=(rectilinear_polygon[i, 1] + polygon[i+1, 1])/2
            rectilinear_polygon[i, 1]= y_next
            rectilinear_polygon[i+1, 1] = y_next
        else:
            if i!= 7:
                x_next = (rectilinear_polygon[i, 0] + polygon[i + 1, 0]) / 2
                rectilinear_polygon[i, 0] = x_next
                rectilinear_polygon[i + 1, 0] = x_next
            else:
                # rectilinear_polygon[i, 0] = (rectilinear_polygon[i, 0] + polygon[i + 1, 0])/2
                rectilinear_polygon[i, 0] = rectilinear_polygon[0, 0]
    # rectilinear_polygon[1, 0] = (polygon[0, 0] + polygon[2, 0]) / 2  # Average x for left edge
    # rectilinear_polygon[5, 0] = (polygon[4, 0] + polygon[6, 0]) / 2  # Average x for right edge
    # rectilinear_polygon[3, 1] = (polygon[2, 1] + polygon[4, 1]) / 2  # Average y for top edge
    # rectilinear_polygon[7, 1] = (polygon[0, 1] + polygon[6, 1]) / 2  # Average y for bottom edge
    return rectilinear_polygon

building_file_path = os.path.join("Z:", "iiixr-drive", "Projects", "2023_City_Team", "atlanta_dataset",
                                  "density20_building120_rotate_normalized", "Buildings",
                                  "atlanta_buildings133.geojson")
convex_polygon = geojson_to_shapely_polygon(building_file_path)
convex_polygon_coords = np.array(convex_polygon.exterior.coords)

# Now you can calculate the centroid as the mean of the vertices
centroid = np.mean(convex_polygon_coords, axis=0)
# print(convex_polygon)
# centroid = Point(np.mean(convex_polygon, axis=0))
distance_from_centroid = np.max(np.linalg.norm(convex_polygon_coords - centroid, axis=1)) / np.sqrt(2)
rectilinear_polygon = np.array([
    [centroid[0] - distance_from_centroid, centroid[1] - distance_from_centroid],
    [centroid[0], centroid[1] - distance_from_centroid],
    [centroid[0] + distance_from_centroid, centroid[1] - distance_from_centroid],
    [centroid[0] + distance_from_centroid, centroid[1]],
    [centroid[0] + distance_from_centroid, centroid[1] + distance_from_centroid],
    [centroid[0], centroid[1] + distance_from_centroid],
    [centroid[0] - distance_from_centroid, centroid[1] + distance_from_centroid],
    [centroid[0] - distance_from_centroid, centroid[1]]
])
non_rectilinear_polygon = rectilinear_polygon.copy()
# Perform the gradient descent optimization
num_iterations = 10
learning_rate = 0.01

losses = []

for i in tqdm(range(num_iterations), desc='Optimizing'):
    non_rectilinear_polygon = gradient_descent_step_non_rectilinear(convex_polygon, non_rectilinear_polygon, learning_rate)
    loss = calculate_loss(convex_polygon, non_rectilinear_polygon)
    losses.append(loss)

# Convert the optimized polygon into a rectilinear shape
final_rectilinear_polygon = convert_to_rectilinear(non_rectilinear_polygon)
final_loss = calculate_loss(convex_polygon, final_rectilinear_polygon)

# Plot the optimization process
fig = plt.figure(figsize=(8, 8), dpi=100)

# Plot the loss over iterations
# plt.subplot(1, 2, 1)
# plt.plot(losses)
# plt.title('Loss During Optimization')
# plt.xlabel('Iteration')
# plt.ylabel('Loss')

# Plot the final rectilinear polygon
fig = plt.figure(figsize=(8, 8), dpi=100)  # Create a figure with a fixed size and resolution

# Plot the final rectilinear polygon
ax = fig.add_subplot(1, 1, 1)  # Since we're plotting only the polygon, we use 1 subplot instead of 2

# Extract the coordinates from the original Shapely polygon
x, y = convex_polygon.exterior.xy

# Plot the original polygon
ax.plot(np.append(x, x[0]), np.append(y, y[0]), 'b-o', label='Original Polygon')
# Fill the original polygon
ax.fill(x, y, 'skyblue', alpha=0.3)

# Plot the final rectilinear polygon (assuming final_rectilinear_polygon is a numpy array)
ax.plot(np.append(final_rectilinear_polygon[:, 0], final_rectilinear_polygon[0, 0]),
         np.append(final_rectilinear_polygon[:, 1], final_rectilinear_polygon[0, 1]), 'r-o', label='Final Rectilinear Polygon')
# Fill the final rectilinear polygon
ax.fill(final_rectilinear_polygon[:, 0], final_rectilinear_polygon[:, 1], 'salmon', alpha=0.3)

# Set the title and axis labels
ax.set_title('Original and Final Rectilinear Polygon')
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Set the x and y axis limits to fixed [0,1] plane
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)

# Enable the grid and legend
ax.grid(True)
ax.legend()

# Set the aspect of the plot to be equal
ax.set_aspect('equal')

# Display the plot with a fixed aspect ratio
plt.show()

# Output the initial and final loss

print("Final loss:", final_loss)
