import json
from shapely.geometry import shape
import concurrent.futures
import os


def process_boundary(city_name, i, filenum, data_geojson):
    # boundary file을 하나씩 읽기
    boundary_filename = city_name + '_dataset/Boundaries/' + city_name + f'_boundaries{i}.geojson'

    with open(boundary_filename, "r", encoding='UTF-8') as file:
        boundary_geojson = json.load(file)

    # boundary_polygon에 boundary의 coordinate을 추출해서 shapely의 polygon data로 만들기
    # + ) features : [{"geometry": [coords]}] features는 list안에 dictionary가 존재하는 형태임
    boundary_polygon = shape(boundary_geojson["features"][0]["geometry"])
    intersections = {"type": "FeatureCollection", "features": []}

    # R-tree index 생성
    # 지금은 r-tree에 하나씩 넣고 있는 방식인데 이 부분은 한번에 넣을 수 있도록 POI 수정하면서 함께 수정할 예정
    idx = index.Index()
    for pos, feature in enumerate(data_geojson["features"]):
        # geom에 담기는 정보는 building polygon
        geom = shape(feature["geometry"])
        idx.insert(pos, geom.bounds)  # geom.bound??? 가 뭐임? coords인가? 주석 추가해주셈

    # R-tree를 사용하여 겹치는(포함관계인) 객체 찾기
    intersecting_indices = list(idx.intersection(boundary_polygon.bounds))

    # 진짜 속하는지 찾기 - R-tree의 intersection함수에서 겹친다고(포함관계) 판단된 경우에 대해서만 수행
    # 이 과정이 정확히 왜 필요한지 궁금함. intersecting_indices = list(idx.intersection(boundary_polygon.bounds))가 실제로는 포함관계가 아닌데 포함관계라고 인식하는 경우가 있는건가?
    # 만약 그렇다면, 어떤 경우에 그런건지 여기에 주석으로 달아두셈
    for intersecting_index in intersecting_indices:
        feature = data_geojson["features"][intersecting_index]
        # r-tree에 의해 선택된 building_polygon이 boundary_polygon에 속하는지를 검사 (within)
        if geom.within(boundary_polygon):
            # 속하면 intersection 이라는 이름의 geojson 형태에 추가하기
            intersections["features"].append(feature)

    # building geojson 으로 저장
    building_filename = city_name + '_dataset/Buildings/' + city_name + f'_buildings{i}.geojson'
    with open(building_filename, 'w', encoding='UTF-8') as outfile:
        json.dump(intersections, outfile)


def get_building(city_name):
    # Boundary가 몇개 추출되었는지를 파악 (filenum == boundary 수)
    dir_path = f'{city_name}_dataset/Boundaries/'
    files = os.listdir(dir_path)
    filenum = len(files)
    # building data 불러오기
    data_filepath = city_name + '_dataset/' + city_name + "_polygon_data_combined.geojson"

    # data_geojson은 building이 building data를 포함함
    with open(data_filepath, "r", encoding='UTF-8') as file:
        data_geojson = json.load(file)

    # process_boundary 함수의 실행을 병렬로 처리
    # ThreadPoolExecutor()의 경우 사용가능한 thread에서 task들(process_boundary)을 병렬로 처리하고
    # 다 처리했으면 그다음거 가져와서 처리하고 이런식이라고 함
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i, _ in enumerate(range(1, filenum), start=1):
            # process_boundary(city_name, i, filenum, data_geojson)을 넣어주면
            # process_boundary에서는 i번째 boundary에 대한 building data를 추출하고 저장
            futures.append(executor.submit(process_boundary, city_name, i, filenum, data_geojson))
