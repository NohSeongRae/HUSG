from cityname import city_name

data_filepath = f"./2023_City_Team/{city_name}_dataset/{city_name}_all_features.geojson"
polygon_filepath = f'./2023_City_Team/{city_name}_dataset/{city_name}_polygon_data.geojson'
point_filepath = f'./2023_City_Team/{city_name}_dataset/{city_name}_point_data.geojson'
combined_filepath = f'./2023_City_Team/{city_name}_dataset/{city_name}_polygon_data_combined.geojson'
roads_filepath = f'./2023_City_Team/{city_name}_dataset/{city_name}_roads.geojson'
buildinglevel_filepath = f'./2023_City_Team/{city_name}_dataset/{city_name}_buildinglevel.geojson'
graph_filepath = f'./2023_City_Team/{city_name}_dataset/{city_name}_graph.csv'

output_directories = [
    f'./2023_City_Team/{city_name}_dataset/',
    f'./2023_City_Team/{city_name}_dataset/Boundaries',
    f'./2023_City_Team/{city_name}_dataset/Buildings',
    f'./2023_City_Team/{city_name}_dataset/NLD',
    f'./2023_City_Team/{city_name}_dataset/Image'
]

