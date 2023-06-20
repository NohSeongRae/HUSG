import geopandas as gpd
import pandas as pd
import json
from concurrent import futures
import os
import sys

current_script_path = os.path.dirname(os.path.abspath(__file__))
husg_directory_path = os.path.dirname(current_script_path)
sys.path.append(husg_directory_path)

from etc import filepath as filepath

def process_chunk(chunk_points, polygons):
    """
    # point와 polygon의 공간 관계를 파악해 polygon 내에 속하는 point 찾기
    # geopandas의 sjoin 함수는 r-tree를 기반으로 탐색을 수행
    # how="left", predicate="within"은 왼쪽 param(chunk_points)가 오른쪽 param(polygons)에 속하는지(within)를 판단하는 것을 의미
    :param chunk_points: POI chunk
    :param polygons: building
    :return: GeoDataFrame 형태로 point, polygon index정보 반환 - 각각의 point가 어떤 index의 polygon에 속하는지
     (columns: index(point), geometry(point), index_right(polygon index))
    """
    return gpd.sjoin(chunk_points, polygons, how="left", predicate="within")

if __name__ == '__main__':
    polygons = gpd.read_file(filepath.polygon_filepath)
    points = gpd.read_file(filepath.point_filepath)

    num_chunks = 5
    # 병렬 처리를 위한 point data 분할
    chunk_size = len(points) // num_chunks
    chunks = [points[i:i+chunk_size] for i in range(0, len(points), chunk_size)]
    # process_chunk 함수를 병렬적으로 처리
    with futures.ProcessPoolExecutor(max_workers=5) as executor:
        results = [executor.submit(process_chunk, chunk, polygons) for chunk in chunks]
    # 병렬처리 결과물을 합침
    joined = pd.concat([result.result() for result in results])

    # point의 index와 polygon index를 mapping 하기위한 dictionary
    index_mapping = {}

    for i, row in joined.iterrows():
        # 만약 point가 어떠한 polygon에 속한다면 (polygon의 index가 None이 아니라면)
        if not pd.isna(row['index_right']):
            # point index - polygon index mapping
            index_mapping[i] = int(row['index_right'])

    """
    index_mapping을 기반으로 point data와 polygon data 병합 
    :return: polygon, point 가 합쳐진 geojson 파일 
    """
    with open(filepath.polygon_filepath, 'r', encoding='UTF-8') as file:
        polygon_json = json.load(file)

    with open(filepath.point_filepath, 'r', encoding='UTF-8') as file:
        point_json = json.load(file)

    # source_index = point index / target_index = polygon_index
    for source_index, target_index in index_mapping.items():
        # propertiese_to_add : point_index properties (POI)
        properties_to_add = point_json['features'][source_index]['properties']
        for key, value in properties_to_add.items():
            # 만약 해당 POI 속성이 합치고자하는 polygon data에 존재하지 않는다면 정보 추가하기
            if key not in polygon_json['features'][target_index]['properties']:
                polygon_json['features'][target_index]['properties'][key] = value

    with open(filepath.combined_filepath, "w") as outfile:
        json.dump(polygon_json, outfile)

    print("POI 합치기 완료")