import os
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import tqdm
import pickle
from boundarymask import boundarymask
from insidemask import insidemask
from streetmask import streetmask
from tfoutput_seqmask import tfoutput_seqmask
from tfoutput_plotmask import tfoutput_plotmask
from inbuildingcpmask import inbuildingcpmask
from inedgemask import inedgemask
from allmask import allmask
from groundtruthmask import groundtruthmask

# city_names = ["atlanta", "dallas", "houston", "lasvegas", "littlerock",
# "philadelphia", "phoenix", "portland", "richmond", "saintpaul",
# "sanfrancisco", "miami", "seattle", "boston", "providence",
# "neworleans", "denver", "pittsburgh", "tampa", "washington"]

# city_names = ["philadelphia", "phoenix", "portland", "richmond", "saintpaul"]

city_names = ["phoenix", "miami", "littlerock"]

image_size = 120
linewidth = 5
num_grids = 60
unit_length = 0.04
cp_node_size = 5

pickle_folder_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '2_transformer', 'train_dataset')

for city_name in city_names:
    print("city name", city_name)
    building_exist_sequences_path = os.path.join(pickle_folder_path, f'{city_name}', 'building_exist_sequences.pkl')
    street_index_sequences_path = os.path.join(pickle_folder_path, f'{city_name}', 'street_index_sequences.pkl')
    unit_coords_datasets_path = os.path.join(pickle_folder_path, f'{city_name}', 'unit_coords_datasets.pkl')
    node_features_path = os.path.join(pickle_folder_path, f'{city_name}', 'node_features.pkl')

    with open(building_exist_sequences_path, 'rb') as f:
        building_exist_sequences = pickle.load(f)

    with open(street_index_sequences_path, 'rb') as f:
        street_index_sequences = pickle.load(f)

    with open(unit_coords_datasets_path, 'rb') as f:
        unit_coords_datasets = pickle.load(f)

    with open(node_features_path, 'rb') as f:
        building_center_position_datasets = pickle.load(f)

    # print(node_features) # building 여부 , x y w h theta

    # boundarymask(city_name, image_size, unit_coords_datasets)
    # insidemask(city_name, image_size, unit_coords_datasets)
    # streetmask(city_name, image_size, unit_coords_datasets, street_index_sequences)
    # tfoutput_seqmask(city_name, image_size, unit_coords_datasets, building_exist_sequences)

    inbuildingcpmask(city_name, image_size, unit_coords_datasets, building_center_position_datasets, cp_node_size)
    inedgemask(city_name, image_size, unit_coords_datasets, building_center_position_datasets, node_size=3, line_width=2)

    # allmask(city_name, image_size, unit_coords_datasets, street_index_sequences, building_exist_sequences)

    # tfoutput_plotmask(city_name, image_size, unit_coords_datasets, building_exist_sequences, linewidth, num_grids, unit_length)

    groundtruthmask(city_name, image_size, unit_coords_datasets, building_center_position_datasets, node_size=3, line_width=2)
