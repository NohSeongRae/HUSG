import os
import json
import geopandas as gpd
import matplotlib.pyplot as plt
import sys
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
import numpy as np
import imageio

current_script_path = os.path.dirname(os.path.abspath(__file__))
husg_directory_path = os.path.dirname(current_script_path)
sys.path.append(husg_directory_path)

city_name = "philadelphia"

dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Boundaries')
files = os.listdir(dir_path)
filenum = len(files)

def insidemask(city_name):
    for i in range(1, filenum + 1):
        boundary_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                         'Boundaries', f'{city_name}_boundaries{i}.geojson')

        if os.path.exists(boundary_filename):
            gdf = gpd.read_file(boundary_filename)

            # print(gdf)

            width, height = 224, 224
            transform = rasterio.transform.from_bounds(gdf.total_bounds[0], gdf.total_bounds[1], gdf.total_bounds[2],
                                                       gdf.total_bounds[3], width, height)

            mask = geometry_mask(gdf.geometry, transform=transform, invert=False, out_shape=(height, width))

            scaled_mask = (mask * 255).astype(np.uint8)

            # plt.imshow(mask, cmap='gray')
            # plt.colorbar()
            # plt.show()

            insidemask_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                             'insidemask', f'{city_name}_insidemask{i}.png')

            imageio.imsave(insidemask_filename, scaled_mask)

insidemask(city_name="philadelphia")