from shapely.geometry import Polygon, LineString
from shapely.geometry import shape
import osmnx as ox
import json
import geopandas as gpd
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_boundary(city_name, location):
    city_boundary = ox.geocode_to_gdf(location)

    city_boundary_geojson = json.loads(city_boundary.to_json())

    polygon = shape(city_boundary_geojson['features'][0]['geometry'])

    roads_filename = city_name + "_dataset/" + city_name + "_roads.geojson"
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


    def divide_polygon(polygon, linestrings):
        result_polygons = []

        for linestring in linestrings:
            buffered_linestring = linestring.buffer(0.000019, cap_style=3)
            result_polygon = polygon.difference(buffered_linestring)
            result_polygons.append(result_polygon)

        return result_polygons


    result_polygons = divide_polygon(polygon, linestrings)

    polys = []

    for result_polygon in result_polygons:
        polys.append(result_polygon)

    result_poly = polys[0]

    print("intersection 시작")

    def intersection_operation(i):
        return polys[i].intersection(result_poly)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(intersection_operation, i) for i in range(1, len(polys))]

        for future in as_completed(futures):
            result = future.result()
            result_poly = result_poly.intersection(result)

    print("intersection 끝")

    poly_list = []
    if result_poly.geom_type == "MultiPolygon":
        for poly in result_poly.geoms:
            poly_list.append(poly)
    else:
        poly_list.append(result_poly)

    print(len(poly_list))


    def save_polygon(i):
        poly = poly_list[i]
        gdf = gpd.GeoDataFrame(geometry=[poly], columns=["POLYGON"])
        polygon_filename = city_name + "_dataset/Boundaries/" + city_name + f"_boundaries{i}.geojson"
        gdf.to_file(polygon_filename, driver="GeoJSON")

    with ThreadPoolExecutor() as executor:
        executor.map(save_polygon, range(len(poly_list)))

    return len(poly_list)
