import os
import sys

current_script_path = os.path.dirname(os.path.abspath(__file__))
husg_directory_path = os.path.dirname(current_script_path)
sys.path.append(husg_directory_path)

from .cityname import city_name

data_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', f'{city_name}_all_features.geojson')
polygon_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', f'{city_name}_polygon_data.geojson')
point_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', f'{city_name}_point_data.geojson')
combined_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', f'{city_name}_polygon_data_combined.geojson')
roads_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', f'{city_name}_roads.geojson')
buildinglevel_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', f'{city_name}_buildinglevel.geojson')
graph_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', f'{city_name}_graph.csv')
removed_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', f'{city_name}_removed_filenum.csv')
category_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', 'category.json')

lengtharea_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', f'{city_name}_lengtharea.png')
lengthnum_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', f'{city_name}_lengthnum.png')
ratio_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', f'{city_name}_ratio.png')
density_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', f'{city_name}_density.png')

category_semantic = ["commercial", "education", "emergency", "financial", "government", "healthcare", "public", "sport", "building"]

commercial_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'blocksemantic', 'commerical.png')
education_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'blocksemantic', 'education.png')
emergency_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'blocksemantic', 'emergency.png')
financial_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'blocksemantic', 'financial.png')
government_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'blocksemantic', 'government.png')
healthcare_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'blocksemantic', 'healthcare.png')
public_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'blocksemantic', 'public.png')
sport_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'blocksemantic', 'sport.png')
building_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'blocksemantic', 'building.png')

dataset = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset')
boundaries = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Boundaries')
buildings = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Buildings')
nld = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'NLD')
image = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Image')

output_directories = [
    dataset,
    boundaries,
    buildings,
    nld,
    image
]






