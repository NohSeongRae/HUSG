import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely.geometry import Polygon, LineString, Point

# from etc.cityname import *

# city_name = city_name

building_polygons = []


def plot_boundary_building(building_polygons, boundary_polygon):
    for poly in building_polygons:
        # print(poly)
        x, y = poly.exterior.xy

        plt.plot(x, y)

    boundary_x, boundary_y = boundary_polygon.exterior.xy
    plt.plot(boundary_x, boundary_y)

    plt.show()


def normalize_coordinates(geometry, min_x, min_y, max_x, max_y):
    def normalize_point(point):
        return ((point[0] - min_x) / (max_x - min_x),
                (point[1] - min_y) / (max_y - min_y))

    if isinstance(geometry, Polygon):
        exterior = [normalize_point(point) for point in list(geometry.exterior.coords)]
        interiors = []
        for interior in geometry.interiors:
            interiors.append([normalize_point(point) for point in list(interior.coords)])
        return Polygon(exterior, interiors)

    # Add additional handling for other geometry types (Point, LineString, etc.) if needed

    return geometry


# Load data
# building_gdf = gpd.read_file('path_to_building_file.geojson')
# boundary_gdf = gpd.read_file('path_to_boundary_file.geojson')
#
# # Find bounding box
# min_x = min(building_gdf.geometry.bounds.minx.min(), boundary_gdf.geometry.bounds.minx.min())
# min_y = min(building_gdf.geometry.bounds.miny.min(), boundary_gdf.geometry.bounds.miny.min())
# max_x = max(building_gdf.geometry.bounds.maxx.max(), boundary_gdf.geometry.bounds.maxx.max())
# max_y = max(building_gdf.geometry.bounds.maxy.max(), boundary_gdf.geometry.bounds.maxy.max())
#
# # Normalize coordinates
# building_gdf['geometry'] = building_gdf['geometry'].apply(normalize_coordinates, args=(min_x, min_y, max_x, max_y))
# boundary_gdf['geometry'] = boundary_gdf['geometry'].apply(normalize_coordinates, args=(min_x, min_y, max_x, max_y))
#
# # Save the normalized data
# new_dir_path = "path_to_new_directory"
# os.makedirs(new_dir_path, exist_ok=True)
#
# building_gdf.to_file(os.path.join(new_dir_path, "normalized_buildings.geojson"), driver='GeoJSON')
# boundary_gdf.to_file(os.path.join(new_dir_path, "normalized_boundaries.geojson"), driver='GeoJSON')


city_names = ["atlanta", "dallas", "houston", "lasvegas", "littlerock",
"philadelphia", "phoenix", "portland", "richmond", "saintpaul",
"sanfrancisco", "miami", "seattle", "boston", "providence",
"neworleans", "denver", "pittsburgh", "tampa", "washington"]

def sort_key(filename):
    # 파일 이름에서 숫자만 추출
    num = int(''.join(filter(str.isdigit, filename)))
    return num


for city_name in city_names:
    print("city : ", city_name)
    building_dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                     'density20_building120_filtered_data', 'Buildings')
    boundary_dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                     'density20_building120_filtered_data', 'Boundaries')

    # Iterate over all .geojson files in the directory
    for building_filepath in tqdm(sorted([f for f in os.listdir(building_dir_path) if f.endswith('.geojson')], key=sort_key)):
        num = sort_key(building_filepath)
        boundary_filepath = building_filepath.replace('buildings', 'boundaries')

        # Construct the full paths
        building_filename = os.path.join(building_dir_path, building_filepath)
        boundary_filename = os.path.join(boundary_dir_path, boundary_filepath)

        if os.path.exists(building_filename):

            boundary_gdf = gpd.read_file(boundary_filename)
            building_gdf = gpd.read_file(building_filename)
            # for i in range(1, 10000):
            #     building_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
            #                                      'filtered_data','Buildings', f'{city_name}_buildings{i}.geojson')
            #     boundary_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
            #                                      'filtered_data','Boundaries', f'{city_name}_boundaries{i}.geojson')
            #
            #     if os.path.exists(building_filename):
            #         boundary_gdf = gpd.read_file(boundary_filename)
            #         building_gdf = gpd.read_file(building_filename)

            # Find bounding box
            min_x = min(building_gdf.geometry.bounds.minx.min(), boundary_gdf.geometry.bounds.minx.min())
            min_y = min(building_gdf.geometry.bounds.miny.min(), boundary_gdf.geometry.bounds.miny.min())
            max_x = max(building_gdf.geometry.bounds.maxx.max(), boundary_gdf.geometry.bounds.maxx.max())
            max_y = max(building_gdf.geometry.bounds.maxy.max(), boundary_gdf.geometry.bounds.maxy.max())

            # Normalize coordinates
            building_gdf['geometry'] = building_gdf['geometry'].apply(normalize_coordinates,
                                                                      args=(min_x, min_y, max_x, max_y))
            boundary_gdf['geometry'] = boundary_gdf['geometry'].apply(normalize_coordinates,
                                                                      args=(min_x, min_y, max_x, max_y))

            # Save the normalized data
            # new_dir_path = "path_to_new_directory"
            building_new_dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team',
                                                 f'{city_name}_dataset',
                                                 'density20_building120_Normalized', 'Buildings')
            boundary_new_dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team',
                                                 f'{city_name}_dataset',
                                                 'density20_building120_Normalized', 'Boundaries')
            os.makedirs(building_new_dir_path, exist_ok=True)
            os.makedirs(boundary_new_dir_path, exist_ok=True)

            building_gdf.to_file(os.path.join(building_new_dir_path, f'{city_name}_buildings{num}.geojson'),
                                 driver='GeoJSON')
            boundary_gdf.to_file(os.path.join(boundary_new_dir_path, f'{city_name}_boundaries{num}.geojson'),
                                 driver='GeoJSON')

            # # Get building polygons for the current file and add them to the building_polygon list
            # building_polygon = [row['geometry'] for idx, row in building_gdf.iterrows()]
            #
            # # Accumulate building polygons from all files into the building_polygons list
            #
            # """
            # building_polygons : list of building polygons in boundary  ex. [POLYGON ((x1 y1, x2 y2, ...)), POLYGON ((x1 y1, x2 y2, ...))]
            # boundary_polygon : boundary polygon  ex. POLYGON ((x1 y1, x2 y2, ...))
            #
            # x1, y1은 현재 위도 경도
            # """
            #
            # building_polygons.extend(building_polygon)
            # boundary_polygon = boundary_gdf.iloc[0]['geometry']
            #
            # # 시각화
            # plot_boundary_building(building_polygons, boundary_polygon)
            #
            # building_polygons.clear()
