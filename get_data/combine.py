import os
import geopandas as gpd
import sys

current_script_path = os.path.dirname(os.path.abspath(__file__))
husg_directory_path = os.path.dirname(current_script_path)
sys.path.append(husg_directory_path)

from etc.cityname import city_name
from add_key import add_key

add_key(city_name)

directory = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                         'Combined_Buildings')

if not os.path.exists(directory):
    os.makedirs(directory)

dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Boundaries')
files = os.listdir(dir_path)
filenum = len(files)

index = 0

for i in range(1, filenum + 1):
    building_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                     'Buildings', f'{city_name}_buildings{i}.geojson')

    if os.path.exists(building_filename):
        index += 1
        # Use geopandas to read the GeoJSON file
        gdf = gpd.read_file(building_filename)

        if gdf.empty:
            continue

        # Dissolve the polygons into single one by the 'key' attribute
        if gdf['key'].apply(lambda x: isinstance(x, list)).any():
            gdf['key'] = gdf['key'].apply(lambda x: x[0] if isinstance(x, list) else x)

        gdf_dissolved = gdf.dissolve(by='key')

        # gdf_dissolved = gdf.dissolve(by='key')

        # Output the dissolved GeoDataFrame into a new GeoJSON file
        building_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                         'Combined_Buildings', f'{city_name}_buildings{index}.geojson')

        gdf_dissolved.to_file(building_filename, driver='GeoJSON')

import shutil

dir = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Buildings')
if os.path.exists(dir):
    shutil.rmtree(dir)

temp_dir = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Combined_Buildings')

os.rename(temp_dir, dir)