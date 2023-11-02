import numpy as np
import pandas as pd
import os
import pickle

# Initialize the square
x1 = 0
y1 = 0
x2 = 1
y2 = 1

# Compute the number of points
num_points = 64

# Calculate an approximate number of points per edge
num_points_edge = int(np.round(num_points / 4))

# Create points for each edge
edge1 = np.column_stack((np.linspace(x1, x2, num_points_edge+1)[:-1], np.full(num_points_edge, y1)))
edge2 = np.column_stack((np.full(num_points_edge, x2), np.linspace(y1, y2, num_points_edge+1)[:-1]))
edge3 = np.column_stack((np.linspace(x2, x1, num_points_edge+1)[:-1], np.full(num_points_edge, y2)))
edge4 = np.column_stack((np.full(num_points_edge, x1), np.linspace(y2, y1, num_points_edge+1)[:-1]))

# Concatenate the points
points = np.concatenate((edge1, edge2, edge3, edge4))

# Make sure we have exactly 'num_points' points
assert len(points) == num_points, f"Expected {num_points} points, but got {len(points)}."

# Create a DataFrame
df = pd.DataFrame(points)
# Save as .csv file
square_root_path = os.path.join("Z:", "iiixr-drive", "Projects", "2023_City_Team", "5_ACAP")
# csv_path = os.path.join(square_root_path, 'normalized_points.csv')
pickle_path = os.path.join(square_root_path, 'normalized_square.pickle')

df.to_pickle(pickle_path)