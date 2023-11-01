import os
import numpy as np
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely.geometry import Polygon, LineString, Point
import copy
matplotlib.use('TkAgg')
building_file_path = os.path.join("Z:", "iiixr-drive", "Projects", "2023_City_Team", "atlanta_dataset",
                                  "density20_building120_rotate_normalized", "Buildings",
                                  "atlanta_buildings133.geojson")
gdf_utm = gpd.read_file(building_file_path)
# gdf_utm = gdf_utm.to_crs(building_gdf.estimate_utm_crs())
# xmin, ymin, xmax, ymax = gdf_utm.total_bounds
# width = xmax - xmin
# height = ymax - ymin
# max_range = max(width, height)
# scale_factor = 1 / max_range
# gdf_utm['geometry'] = gdf_utm.scale(xfact=scale_factor, yfact=scale_factor, origin=(xmin,ymin))
# gdf_utm['geometry'] = gdf_utm.translate(-xmin * scale_factor, -ymin * scale_factor)
fig2, ax2 = plt.subplots(figsize=(8, 8))

gdf_utm.plot(ax=ax2, color='red', label='Buildings')
ax2.set_title("Aligned Building Block with Buildings")
ax2.set_aspect('equal')
ax2.legend()
plt.show()