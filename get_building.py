import os
import json
import geopandas as gpd
import filepath
from concurrent.futures import ProcessPoolExecutor
from cityname import city_name

dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Boundaries')
files = os.listdir(dir_path)
filenum = len(files)

with open(filepath.combined_filepath, "r") as f:
    geojson_polygons = json.load(f)

polygons_gdf = gpd.GeoDataFrame.from_features(geojson_polygons['features'])

def process_boundary(i):
    boundary_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                    'Boundaries', f'{city_name}_boundaries{i}.geojson')

    with open(boundary_filename, "r") as f:
        geojson_boundary = json.load(f)

    if geojson_boundary['type'] == 'Feature':
        geojson_boundary = [geojson_boundary]

    boundary_gdf = gpd.GeoDataFrame.from_features(geojson_boundary)

    intersection_gdf = gpd.overlay(polygons_gdf, boundary_gdf, how='intersection')

    geojson_polygons_clean = json.loads(intersection_gdf.to_json())

    for feature in geojson_polygons_clean['features']:
        feature['properties'] = {k: v for k, v in feature['properties'].items() if v is not None}

    building_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Buildings',
                 f'{city_name}_buildings{i}.geojson')

    if geojson_polygons_clean['features']:
        with open(building_filepath, 'w') as f:
            json.dump(geojson_polygons_clean, f)

    return intersection_gdf.index

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=5) as executor:
        for inside_boundary_indices in executor.map(process_boundary, range(1, filenum+1)):
            polygons_gdf = polygons_gdf[~polygons_gdf.index.isin(inside_boundary_indices)]
