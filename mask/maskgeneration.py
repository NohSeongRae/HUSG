import os
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import tqdm
from boundarymask import boundarymask
from insidemask import insidemask
from streetmask import streetmask
from tfoutput_seqmask import tfoutput_seqmask
from tfoutput_plotmask import tfoutput_plotmask
from inbuildingcpmask import inbuildingcpmask
from inedgemask import inedgemask
from allmask import allmask

# city_names = ["atlanta", "dallas", "houston", "lasvegas", "littlerock",
# "philadelphia", "phoenix", "portland", "richmond", "saintpaul",
# "sanfrancisco", "miami", "seattle", "boston", "providence",
# "neworleans", "denver", "pittsburgh", "tampa", "washington"]

city_names = ["atlanta"]

image_size = 120
linewidth = 5
num_grids = 60
unit_length = 0.04
cp_node_size = 5

for city_name in city_names:
    city_npz_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '2_transformer', 'train_dataset', f'{city_name}', 'husg_transformer_dataset.npz')
    building_center_npz_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '2_transformer', 'train_dataset', f'{city_name}', 'husg_building_center_position.npz')

    city_npz_data = np.load(city_npz_path)
    building_npz_data = np.load(building_center_npz_path)

    building_index_sequences = city_npz_data['building_index_sequences']
    street_index_sequences = city_npz_data['street_index_sequences']
    # unit_position_datasets = city_npz_data['unit_position_datasets']
    # street_unit_position_datasets = city_npz_data['street_unit_position_datasets']
    unit_coords_datasets = city_npz_data['unit_coords_datasets']

    building_center_position_datasets = building_npz_data['building_center_position_datasets']

    # boundarymask(city_name, image_size, unit_coords_datasets)
    # insidemask(city_name, image_size, unit_coords_datasets)
    # streetmask(city_name, image_size, unit_coords_datasets, street_index_sequences)
    # tfoutput_seqmask(city_name, image_size, unit_coords_datasets, building_index_sequences)
    # inbuildingcpmask(city_name, image_size, building_center_position_datasets, cp_node_size)
    # inedgemask(city_name, image_size, unit_coords_datasets, building_center_position_datasets, cp_node_size)
    # allmask(city_name, image_size, unit_coords_datasets, street_index_sequences, building_index_sequences)
    tfoutput_plotmask(city_name, image_size, unit_coords_datasets, building_index_sequences, linewidth, num_grids, unit_length)
