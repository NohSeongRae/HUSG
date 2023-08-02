from shapely.geometry import Point, Polygon, LineString
from shapely.geometry import shape
import networkx as nx
import osmnx as ox
import json
import geopandas as gpd
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys

current_script_path = os.path.dirname(os.path.abspath(__file__))
husg_directory_path = os.path.dirname(current_script_path)
sys.path.append(husg_directory_path)

from etc import filepath as filepath

def get_boundary(city_name, location):

    """
    # 전체 roads geojson -> Largest subgraph 추출
    :param city_name: processing city name
    :param location: city full name for distinguishing duplicated city name
    :return: None
    """

    with open(filepath.roads_filepath, "r") as file:
        data = json.load(file)
    # data json파일에서 property와 geometry를 추출해서 gdf에 저장
    gdf = gpd.GeoDataFrame.from_features(data["features"])

    # LineString에 해당하는 것들만 edges 에 넣기
    edges = gdf[gdf["geometry"].apply(lambda x: isinstance(x, LineString))]

    # 위 edges 변수에 담긴 edge들을 가지는 Graph G 정의
    # 즉 G는 LineString에 해당하는 것들만 담긴 전체 road graph임
    G = nx.Graph()
    G.add_edges_from([(row["u"], row["v"]) for _, row in edges.iterrows()])
    # graph들을 list에 넣음
    connected_components = list(nx.connected_components(G))
    # 가장 edge 수가 많은 graph를 largest_subgraph_nodes 변수에 할당
    largest_subgraph_nodes = max(connected_components, key=len)

    """
    for component in connected_components:
        if component != largest_subgraph_nodes:
            for node in component:
                G.remove_node(node)
    """

    # largest subgraph를 geopandasframe 형태로 생성
    largest_subgraph_edges_gdf = edges[
        edges.apply(lambda row: row['u'] in largest_subgraph_nodes and row['v'] in largest_subgraph_nodes, axis=1)]
    # geometry만 추출해서 geojson 파일로 만들기
    geometry_only = largest_subgraph_edges_gdf["geometry"].tolist()

    # geom.__geo_interface__는 shapely 라이브러리의 함수 : type(linestring), linestring을 dictionary 형태로 반환해줌
    # 이를 이용해 coordinates 정보만 따와서 geojson 파일로 저장
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": geom.__geo_interface__} for geom in geometry_only
        ],
    }

    # 원래의 roads geojson 파일 덮어쓰기
    with open(filepath.roads_filepath, "w") as file:
        json.dump(geojson_data, file)

    # osmnx를 통해 특정 도시의 boundary geodataframe 얻기
    city_boundary = ox.geocode_to_gdf(location)
    city_boundary_geojson = json.loads(city_boundary.to_json())

    # boundary를 polygon 형태로 만들기
    polygon = shape(city_boundary_geojson['features'][0]['geometry'])

    with open(filepath.roads_filepath) as file:
        data = json.load(file)

    # data_download.py에서 저장한 road geojson 파일의 linestring 선들 얻기
    linestrings = []
    features = data['features']
    for feature in features:
        geometry = feature['geometry']
        if geometry['type'] == 'LineString':
            coordinates = geometry['coordinates']
            linestring = LineString(coordinates)
            linestrings.append(linestring)

    # result_polygon이 쓰이지 않는거같아서 주석처리함
    # result_polygon = polygon

    # difference : 차집합을 찾는 함수 (여기선 polygon - linestring)
    # buffer : linestring에 두께를 줌 (두께의 크기, cap_style=3는 linestring의 모양(끝부분이 각지도록하기))
    def buffered_difference(polygon, linestring):
        """
        #linestring으로 polygon 자르기
        :param polygon: city boundary
        :param linestring: road segment
        :return: 주어진 road segment로 잘린 polygon
        """
        buffered_linestring = linestring.buffer(0.000019, cap_style=3)  # 0.000019는 경험적으로 산출된건지?
        return polygon.difference(buffered_linestring)

    def process_linestrings(polygon, linestrings, num_threads=4):
        """
        4개의 thread를 이용하여 buffered_difference 함수를 전체 linestring에 대해 병렬적으로 처리함
        :param polygon: city boundary
        :param linestrings: road segment
        :param num_threads: 4
        :return: 모든 road segment에 대해 잘린 city boundary
        """

        # 나눠진 chunk 들에 대해 각각 buffered_difference 연산(polygon-linestring)을 수행
        def worker(polygon, linestrings_chunk):
            """
            각 thread에 들어가는 함수. 주어진 road segment chunk를 순회하며 buffered_difference함수를 실행한다.
            :param polygon:city boundary
            :param linestrings_chunk: road segment chunck. thread에 들어가는 각 chunck는 중복되지 않는다.
            :return:주어진 road segment chunck에 대해 잘린 city boundary
            """
            for linestring in linestrings_chunk:
                polygon = buffered_difference(polygon, linestring)
            return polygon

        # 병렬 처리, num_thread=4
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            chunk_size = len(linestrings) // num_threads
            # chunks에 0~chunk_size / chunk_size~2*chunk_size / 2*chunk_size~3*chunk_size ... 넣기
            chunks = [linestrings[i:i + chunk_size] for i in range(0, len(linestrings), chunk_size)]
            # 각각의 chunk들을 input으로 넣어줘서 병렬로 task가 수행되도록 함
            tasks = [executor.submit(worker, polygon, chunk) for chunk in chunks]
            # 완료되면 result_polygons에 담기
            result_polygons = []
            for future in as_completed(tasks):
                result_polygons.append(future.result())

        return result_polygons

    result_polygons = process_linestrings(polygon, linestrings)

    # len(result_polygons) == num_threads
    # 병렬로 처리되어 잘린 polygon들의 합집합을 구함
    final_polygon = result_polygons[0]
    for i in range(1, len(result_polygons)):
        final_polygon = final_polygon.intersection(result_polygons[i])

    # 여러개의 linestring에 대해 잘린 polygon은 자동으로 data type이 multipolygon으로 할당됨
    # 이 multipolygon의 조각들을 빼 poly_list에 저장
    # 즉 poly_list에는 모든 boundary가 shapely 라이브러리의 'polygon' 데이터 타입으로 저장됨
    poly_list = []
    # multipolygon인 경우 그 multipolygon에 속하는 polygon들을 떼어내어 poly_list에 넣어주기
    if final_polygon.geom_type == "MultiPolygon":
        for poly in final_polygon.geoms:
            poly_list.append(poly)
    else:
        # 단일 polygon이면 그냥 넣기
        poly_list.append(final_polygon)

    print(len(poly_list))

    # poly_list에 담긴 polygon들을 하나씩 geojson 형태로 저장
    def save_polygon(i):
        """
        주어진 i번째 polygon을 geojson형태로 저장
        :param i: ploy_list의 i번째 폴리곤
        :return: None
        """
        poly = poly_list[i]
        gdf = gpd.GeoDataFrame(geometry=[poly], columns=["POLYGON"])

        polygon_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Boundaries',
                     f'{city_name}_boundaries{i+1}.geojson')

        gdf.to_file(polygon_filename, driver="GeoJSON")

    # 파일 저장을 병렬로 처리해주는 코드
    with ThreadPoolExecutor() as executor:
        executor.map(save_polygon, range(len(poly_list)))

    executor.shutdown()

    print("boundary 추출 완료")
