from shapely.geometry import Point, Polygon, LineString
from shapely.geometry import shape
import networkx as nx
import osmnx as ox
import json
import geopandas as gpd
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_boundary(city_name, location):
    # Largest subgraph 추출
    roads_filename = city_name + "_dataset/" + city_name + "_roads.geojson"

    with open(roads_filename, "r") as file:
        data = json.load(file)

    gdf = gpd.GeoDataFrame.from_features(data["features"])

    edges = gdf[gdf["geometry"].apply(lambda x: isinstance(x, LineString))]

    G = nx.Graph()
    G.add_edges_from([(row["u"], row["v"]) for _, row in edges.iterrows()])

    connected_components = list(nx.connected_components(G))

    largest_subgraph_nodes = max(connected_components, key=len)

    for component in connected_components:
        if component != largest_subgraph_nodes:
            for node in component:
                G.remove_node(node)

    largest_subgraph_edges_gdf = edges[edges.apply(lambda row: row['u'] in largest_subgraph_nodes and row['v'] in largest_subgraph_nodes, axis=1)]

    geometry_only = largest_subgraph_edges_gdf["geometry"].tolist()

    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": geom.__geo_interface__} for geom in geometry_only
        ],
    }

    with open(roads_filename, "w") as file:
        json.dump(geojson_data, file)

    city_boundary = ox.geocode_to_gdf(location)

    city_boundary_geojson = json.loads(city_boundary.to_json())

    polygon = shape(city_boundary_geojson['features'][0]['geometry'])

    with open(roads_filename) as file:
        data = json.load(file)

    linestrings = []
    features = data['features']
    for feature in features:
        geometry = feature['geometry']
        if geometry['type'] == 'LineString':
            coordinates = geometry['coordinates']
            linestring = LineString(coordinates)
            linestrings.append(linestring)

    result_polygon = polygon

    def buffered_difference(polygon, linestring):
        buffered_linestring = linestring.buffer(0.000019, cap_style=3)
        return polygon.difference(buffered_linestring)

    def process_linestrings(polygon, linestrings, num_threads=4):
        def worker(polygon, linestrings_chunk):
            for linestring in linestrings_chunk:
                polygon = buffered_difference(polygon, linestring)
            return polygon

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            chunk_size = len(linestrings) // num_threads
            chunks = [linestrings[i:i + chunk_size] for i in range(0, len(linestrings), chunk_size)]

            tasks = [executor.submit(worker, polygon, chunk) for chunk in chunks]

            result_polygons = []
            for future in as_completed(tasks):
                result_polygons.append(future.result())

        return result_polygons

    result_polygons = process_linestrings(polygon, linestrings)

    final_polygon = result_polygons[0]
    for i in range(1, len(result_polygons)):
        final_polygon = final_polygon.intersection(result_polygons[i])

    poly_list = []
    if final_polygon.geom_type == "MultiPolygon":
        for poly in final_polygon.geoms:
            poly_list.append(poly)
    else:
        poly_list.append(final_polygon)

    def save_polygon(i):
        poly = poly_list[i]
        gdf = gpd.GeoDataFrame(geometry=[poly], columns=["POLYGON"])
        polygon_filename = city_name + "_dataset/Boundaries/" + city_name + f"_boundaries{i+1}.geojson"
        gdf.to_file(polygon_filename, driver="GeoJSON")

    with ThreadPoolExecutor() as executor:
        executor.map(save_polygon, range(len(poly_list)))

    executor.shutdown()

    print("boundary 추출 완료")
