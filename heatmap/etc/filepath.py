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

lengtharea_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Statistics', f'{city_name}_lengtharea.png')
lengthnum_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Statistics', f'{city_name}_lengthnum')
ratio_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Statistics', f'{city_name}_ratio.png')
density_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Statistics', f'{city_name}_density.png')
boundarysize_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Statistics', f'{city_name}_boundarysize.png')
buildingnum_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Statistics', f'{city_name}_buildingnum.png')

category_semantic = ["commercial", "education", "emergency", "financial", "government", "healthcare", "public", "sport", "building"]

commercial_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Statistics', 'blocksemantics', 'commerical.png')
education_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Statistics', 'blocksemantics', 'education.png')
emergency_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Statistics', 'blocksemantics', 'emergency.png')
financial_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Statistics', 'blocksemantics', 'financial.png')
government_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Statistics', 'blocksemantics', 'government.png')
healthcare_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Statistics', 'blocksemantics', 'healthcare.png')
public_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Statistics', 'blocksemantics', 'public.png')
sport_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Statistics', 'blocksemantics', 'sport.png')
building_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Statistics', 'blocksemantics', 'building')

dataset = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset')
boundaries = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Boundaries')
buildings = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Buildings')
nld = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'NLD')
image = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Image')

boundaryimage = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'BoundaryImage')
buildingimage = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'BuildingImage')

stat = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Statistics')
blocksemantic = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Statistics', 'blocksemantics')



output_directories = [
    dataset,
    boundaries,
    buildings,
    nld,
    image
]

city_names = ["atlanta", "barcelona", "budapest", "dallas", "dublin", "firenze", "houston", "lasvegas",
                  "littlerock", "manchester", "milan", "minneapolis",
                  "nottingham", "paris", "philadelphia", "phoenix", "portland", "richmond", "saintpaul", "sanfrancisco",
                  "singapore", "toronto", "vienna",
                  "washington", "zurich"]
