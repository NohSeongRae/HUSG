import os
import geopandas as gpd
import concurrent.futures
import json


def process_boundary(city_name, i, data_filepath):
    # boundary file 불러오기
    boundary_filename = city_name + '_dataset/Boundaries/' + city_name + f'_boundaries{i}.geojson'
    boundary_gdf = gpd.read_file(boundary_filename)
    data_gdf = gpd.read_file(data_filepath)

    intersections = gpd.sjoin(data_gdf, boundary_gdf, how="left", predicate='within')

    if not intersections.empty:  # building이 하나라도 존재하는 경우에만 저장
        building_filename = city_name + '_dataset/' + 'Buildings/' + city_name + f'_buildings{i}.geojson'

        # Geopandas dataframe을 GeoJSON 형식으로 변환
        geojson = json.loads(intersections.to_json())

        # null 값을 가진 features 제거
        geojson['features'] = [feature for feature in geojson['features'] if feature['properties'] is not None]

        # GeoJSON 데이터를 파일로 저장
        with open(building_filename, 'w') as f:
            json.dump(geojson, f)


def get_building(city_name):
    dir_path = city_name + '_dataset/Boundaries/'
    files = os.listdir(dir_path)
    filenum = len(files)

    data_filepath = city_name + '_dataset/' + city_name + "_polygon_data_combined.geojson"

    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        futures = []
        for i, _ in enumerate(range(1, filenum), start=1):
            futures.append(executor.submit(process_boundary, city_name, i, data_filepath))


if __name__ == '__main__':
    get_building('littlerock')
