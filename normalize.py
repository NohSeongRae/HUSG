import os
import numpy as np
import geopandas as gpd

import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point

city_name = "dublin"

dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Boundaries')
files = os.listdir(dir_path)
filenum = len(files)

building_polygons = []

def plot_boundary_building(building_polygons, boundary_polygon):
    for poly in building_polygons:
        # print(poly)
        x, y = poly.exterior.xy

        plt.plot(x, y)

    boundary_x, boundary_y = boundary_polygon.exterior.xy
    plt.plot(boundary_x, boundary_y)

    plt.show()

fig, ax = plt.subplots()

for i in range(30, 33):
    building_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Combined_Buildings', f'{city_name}_buildings{i}.geojson')
    boundary_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Boundaries', f'{city_name}_boundaries{i}.geojson')

    if os.path.exists(building_filename):
        boundary_gdf = gpd.read_file(boundary_filename)
        building_gdf = gpd.read_file(building_filename)

        # Get building polygons for the current file and add them to the building_polygon list
        building_polygon = [row['geometry'] for idx, row in building_gdf.iterrows()]

        # Accumulate building polygons from all files into the building_polygons list

        """
        building_polygons : list of building polygons in boundary  ex. [POLYGON ((x1 y1, x2 y2, ...)), POLYGON ((x1 y1, x2 y2, ...))]
        boundary_polygon : boundary polygon  ex. POLYGON ((x1 y1, x2 y2, ...))  
        
        x1, y1은 현재 위도 경도 
        """

        building_polygons.extend(building_polygon)
        boundary_polygon = boundary_gdf.iloc[0]['geometry']

        # 시각화
        plot_boundary_building(building_polygons, boundary_polygon)

        building_polygons.clear()

