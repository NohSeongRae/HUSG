import os
import json
import geopandas as gpd
import filepath
from concurrent.futures import ProcessPoolExecutor

city_name = "portland"

dir_path = "./2023_City_Team/" + city_name + '_dataset/Boundaries/'
files = os.listdir(dir_path)
filenum = len(files)

with open(filepath.combined_filepath, "r") as f:
    geojson_polygons = json.load(f)

polygons_gdf = gpd.GeoDataFrame.from_features(geojson_polygons['features'])

def process_boundary(i):
    boundary_filename = "./2023_City_Team/" + city_name + '_dataset/Boundaries/' + city_name + f'_boundaries{i}.geojson'
    with open(boundary_filename, "r") as f:
        geojson_boundary = json.load(f)

    if geojson_boundary['type'] == 'Feature':
        geojson_boundary = [geojson_boundary]

    boundary_gdf = gpd.GeoDataFrame.from_features(geojson_boundary)

    inside_boundary = gpd.sjoin(polygons_gdf, boundary_gdf, how='inner', predicate='within')

    inside_polygons_gdf = polygons_gdf[polygons_gdf.index.isin(inside_boundary.index)]

    geojson_polygons_clean = json.loads(inside_polygons_gdf.to_json())

    for feature in geojson_polygons_clean['features']:
        feature['properties'] = {k: v for k, v in feature['properties'].items() if v is not None}

    building_filepath = "./2023_City_Team/" + city_name + '_dataset/Buildings/' + city_name + f'_buildings{i}.geojson'

    if geojson_polygons_clean['features']:
        with open(building_filepath, 'w') as f:
            json.dump(geojson_polygons_clean, f)

    return inside_boundary.index

if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        for inside_boundary_indices in executor.map(process_boundary, range(1, filenum + 1)):
            polygons_gdf = polygons_gdf[~polygons_gdf.index.isin(inside_boundary_indices)]
