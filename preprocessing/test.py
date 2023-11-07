import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon

from plot_utils import plot_graph, get_random_color

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

dir_path = r"C:\Users\SeungWon Seo\Downloads\train_dataset\train_dataset\washington"

datasets = [
   'node_features',
   'edge_indices',
   'unit_road_street_indices',
   'building_filenames',
   'boundary_filenames',
   'building_polygons',
   'building_semantics'
]

with open(dir_path + '/' + datasets[0] + '.pkl', 'rb') as file:
   node_feature = pickle.load(file)

with open(dir_path + '/' + datasets[1] + '.pkl', 'rb') as file:
   edge_index = pickle.load(file)

with open(dir_path + '/' + datasets[2] + '.pkl', 'rb') as file:
   unit_road_street_indices = pickle.load(file)

with open(dir_path + '/' + datasets[5] + '.pkl', 'rb') as file:
   building_polygons = pickle.load(file)

with open(dir_path + '/' + datasets[6] + '.pkl', 'rb') as file:
   building_semantics = pickle.load(file)

for file_idx in range(len(node_feature)):
   for idx, d in enumerate(node_feature[file_idx]):
      x, y, w, h, theta = d[0], d[1], d[2], d[3], d[4] * 90 - 45
      points = get_bbox_corners(x, y, w, h)
      rotated_points = rotate_points_around_center(points, [x, y], theta)

      rotated_points = np.array(rotated_points)
      rotated_box = np.concatenate((rotated_points, [rotated_points[0]]), axis=0)
      if idx < len(unit_road_street_indices[file_idx]):
         plt.plot(rotated_box[:, 0], rotated_box[:, 1], color=get_random_color(unit_road_street_indices[file_idx][idx]), label='Rotated Box')
      else:
         plt.plot(rotated_box[:, 0], rotated_box[:, 1], 'r-', label='Rotated Box')
   plot_graph(node_feature[file_idx], edge_index[file_idx])

   for idx, building_polygon in enumerate(building_polygons[file_idx]):
      x, y = building_polygon
      plt.fill(x, y, alpha=0.8)
      plt.text(np.mean(x), np.mean(y), building_semantics[file_idx][idx])

   plt.show()
