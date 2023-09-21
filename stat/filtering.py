import os
import sys

current_script_path = os.path.dirname(os.path.abspath(__file__))
husg_directory_path = os.path.dirname(current_script_path)
sys.path.append(husg_directory_path)

from lengtharea import lengtharea
from lengthnum import lengthnum
from ratio import ratio
from blocksemantic import block_category
from density import density
from boundarysize import boundarysize
from buildingnum import buildingnum
from tqdm import tqdm

import shutil
from etc import filepath as filepath
import json

# if not os.path.exists(filepath.stat):
#    os.makedirs(filepath.stat)

# city_names_USA = ["atlanta", "dallas", "dublin", "houston", "lasvegas", "littlerock", "minneapolis", "phoenix", "portland", "richmond", "sanfrancisco", "washington"]

# city_names_all = ["barcelona", "budapest", "firenze", "manchester", "milan", "nottingham", "paris", "singapore", "toronto", "vienna", "zurich"]
# city_names_all = ["losangeles", "miami", "seattle", "boston", "providence", "tampa"]

city_names_all = ["pittsburgh"]


for city in tqdm(range(len(city_names_all))):
    city_name = city_names_all[city]

    filtereddata_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                'filtered_data')
    filteredbuilding_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                    'filtered_data', 'Buildings')
    filteredboundary_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                    'filtered_data', 'Boundaries')

    if not os.path.exists(filtereddata_path):
        os.makedirs(filtereddata_path)

    if not os.path.exists(filteredbuilding_path):
        os.makedirs(filteredbuilding_path)

    if not os.path.exists(filteredboundary_path):
        os.makedirs(filteredboundary_path)


    boundarysize_filtered = boundarysize([city_name], upperlimit=1.8 * 1e-6, lowerlimit=0.2 * 1e-6)
    buildingnum_filtered = buildingnum([city_name], boundarysize_filtered, upperlimit=20, lowerlimit=2)
    density_filtered = density([city_name], buildingnum_filtered, lowerlimit=40)
    print(len(density_filtered))


    whole_filtered = density_filtered


    dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Boundaries')
    files = os.listdir(dir_path)
    filenum = len(files)

    for i in range(1, filenum+1):
        if i in whole_filtered:
            boundary_filtered_data_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team',
                                             f'{city_name}_dataset', 'filtered_data', 'Boundaries')

            building_filtered_data_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team',
                                             f'{city_name}_dataset', 'filtered_data', 'Buildings')

            boundary_data_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team',
                                             f'{city_name}_dataset',
                                             'Boundaries')

            building_data_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team',
                                             f'{city_name}_dataset',
                                             'Combined_Buildings')

            boundary_source_file_path = os.path.join(boundary_data_filepath, f'{city_name}_boundaries{i}.geojson')
            building_source_file_path = os.path.join(building_data_filepath, f'{city_name}_buildings{i}.geojson')

            boundary_dest_file_path = os.path.join(boundary_filtered_data_filepath, f'{city_name}_boundaries{i}.geojson')
            building_dest_file_path = os.path.join(building_filtered_data_filepath, f'{city_name}_buildings{i}.geojson')

            shutil.copy(boundary_source_file_path, boundary_dest_file_path)
            shutil.copy(building_source_file_path, building_dest_file_path)