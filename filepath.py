from cityname import city_name

data_filepath = f"2023_cityteam/{city_name}_dataset/{city_name}_all_features.geojson"
polygon_filepath = f'2023_cityteam/{city_name}_dataset/{city_name}_polygon_data.geojson'
point_filepath = f'2023_cityteam/{city_name}_dataset/{city_name}_point_data.geojson'
combined_filepath = f'2023_cityteam/{city_name}_dataset/{city_name}_polygon_data_combined.geojson'
roads_filepath = f'2023_cityteam/{city_name}_dataset/{city_name}_roads.geojson'

output_directories = [
    f'2023_cityteam/{city_name}_dataset/',
    f'2023_cityteam/{city_name}_dataset/Boundaries',
    f'2023_cityteam/{city_name}_dataset/Buildings',
    f'2023_cityteam/{city_name}_dataset/NLD',
    f'2023_cityteam/{city_name}_dataset/Image'
]

