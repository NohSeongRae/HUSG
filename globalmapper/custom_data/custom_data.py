import networkx as nx
import os
import pickle
import shapely
import geopandas as gpd
from shapely.geometry import Polygon
import os
from pyproj import Proj, Transformer
from shapely.geometry import Point, Polygon

# TODO
# 미국 도시에 대해
# normalize 후 돌리기

city_names =["atlanta","boston", "dallas",  "denver","houston",
             "lasvegas", "littlerock","miami","neworleans",
             "philadelphia", "phoenix", "portland", "providence","pittsburgh",
             "richmond", "saintpaul","sanfrancisco", "seattle","washington"]

def coord_transform(gdf):
    utm_polygons = []
    for geometry in gdf.geometry:
        if isinstance(geometry, Polygon):
            utm_exterior_coords = []

            for exterior_coords in geometry.exterior.coords:
                x, y = exterior_coords
                utm_x, utm_y = transformer.transform(x, y)
                utm_exterior_coords.append((utm_x, utm_y))

            utm_polygon = Polygon(utm_exterior_coords)
            utm_polygons.append(utm_polygon)

    return utm_polygons

count = -1

for city in city_names:
    drive_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city}_dataset')
    custom_data_root = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '1_evaluation',
                                    'globalmapper_custom_data', 'raw_geo', city)

    if not os.path.exists(custom_data_root):
        os.makedirs(custom_data_root)

    dir_path = os.path.join(drive_path, 'Boundaries')
    files = os.listdir(dir_path)
    filenum = len(files)

    in_proj = Proj(proj='latlong', datum='WGS84')
    out_proj = Proj(proj='utm', zone=33, datum='WGS84')

    transformer = Transformer.from_proj(in_proj, out_proj)


    utm_polygons = []

    for i in range(1, filenum+1):
        boundary_filename = os.path.join(drive_path, 'density20_building120_rotate_normalized_v2', 'Boundaries', f'{city}_boundaries{i}.geojson')
        building_filename = os.path.join(drive_path, 'density20_building120_rotate_normalized_v2', 'Buildings', f'{city}_buildings{i}.geojson')

        if os.path.exists(building_filename):
            count += 1
            custom_data = []

            gdf_building = gpd.read_file(building_filename)
            transformed_buildings = coord_transform(gdf_building)

            gdf_boundary = gpd.read_file(boundary_filename)
            transformed_boundaries = coord_transform(gdf_boundary)[0]

            custom_data.append(transformed_boundaries)
            custom_data.append(transformed_buildings)

            custom_data_path = os.path.join(custom_data_root, f'{i}.pkl')

            with open(custom_data_path, 'wb') as f:
                pickle.dump(custom_data, f)


# custom_data_root = os.path.join("../globalmapper_dataset/raw_geo")
#
# file_path = os.path.join(custom_data_root, '0')
#
# with open(file_path, 'rb') as file:
#     my_object = pickle.load(file)
#
# print("custom data: ", my_object)
