import os
import geopandas as gpd
import concurrent.futures

def process_boundary(city_name, i, data_filepath):
    # boundary file 불러오기
    boundary_filename = city_name + '_dataset/Boundaries/' + city_name + f'_boundaries{i}.geojson'
    boundary_gdf = gpd.read_file(boundary_filename)
    data_gdf = gpd.read_file(data_filepath)

    """
    # boundary polygon과 building polygon의 공간 관계를 파악해 boundary polygon 내에 속하는 building polygon 찾기
    # geopandas의 sjoin 함수는 r-tree를 기반으로 탐색을 수행
    # how="left", predicate="within"은 왼쪽 param(chunk_points)가 오른쪽 param(polygons)에 속하는지(within)를 판단하는 것을 의미
    :param chunk_points: POI chunk
    :param polygons: building
    :return: GeoDataFrame 형태로 point, polygon index 정보 반환 - 각각의 point가 어떤 index의 polygon에 속하는지
     (columns: index(point), geometry(point), index_right(polygon index))
    """

    intersections = gpd.sjoin(data_gdf, boundary_gdf, how="left", predicate='within')

    if not intersections.empty:  # building이 하나라도 존재하는 경우에만 저장
        building_filename = city_name + '_dataset/' + 'Buildings/' + city_name + f'_buildings{i}.geojson'
        intersections.to_file(building_filename, driver='GeoJSON')

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