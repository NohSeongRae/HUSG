import numpy as np
import pandas as pd

# Initialize the square
x1 = 0
y1 = 0
x2 = 1
y2 = 1

# Compute the number of points and the distance between them
num_points = 594
dist = np.sqrt((x2-x1)*(y2-y1) / num_points)

# Calculate number of points per edge
num_points_edge = int(np.round(num_points / 4))

# Create the points for each edge
edge1 = np.column_stack((np.linspace(x1, x2, num_points_edge), np.linspace(y1, y1, num_points_edge)))
edge2 = np.column_stack((np.linspace(x2, x2, num_points_edge), np.linspace(y1, y2, num_points_edge)))
edge3 = np.column_stack((np.linspace(x2, x1, num_points_edge), np.linspace(y2, y2, num_points_edge)))
edge4 = np.column_stack((np.linspace(x1, x1, num_points_edge), np.linspace(y2, y1, num_points_edge)))

# Concatenate the points
points = np.concatenate((edge1, edge2, edge3, edge4))

# Make sure we have exactly 'num_points' points
points = points[:num_points]

# Create a DataFrame
df = pd.DataFrame(points)
# Save as .csv file
df.to_csv('C:/Users/rlaqhdrb/Desktop/misong/normalized_points.csv', index=False, header=False)
