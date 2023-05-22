import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
import networkx as nx
import os
from get_buildinglevel import get_buildinglevel

commercial, education, emergency, financial, government, healthcare, landuse, natural, public, sport, water, residence = get_buildinglevel()

def polar_angle(origin, point):
    delta_x = point[0] - origin[0]
    delta_y = point[1] - origin[1]
    angle = np.arctan2(delta_y, delta_x)
    return angle if angle >= 0 else 2 * np.pi + angle


def sort_points_ccw(points):
    centroid = np.mean(points, axis=0)
    angles = np.array([polar_angle(centroid, point) for point in points])
    sorted_indices = np.argsort(angles)
    return points[sorted_indices]


new_coords = []


def extract_polygon_coordinates(geom):
    # normalize
    minx, miny, maxx, maxy = geom.bounds
    for point in list(geom.exterior.coords):
        lat, lon = point
        lat_norm = 2 * ((lat - minx) / (maxx - minx)) - 1
        lon_norm = 2 * ((lon - miny) / (maxy - miny)) - 1
        new_coords.append((lat_norm, lon_norm))
    geom = Polygon(new_coords)
    return list(geom.exterior.coords)


def graph_dataloader(city_name):
    dir_path = "./2023_City_Team/" + city_name + '_dataset/Buildings/'
    files = os.listdir(dir_path)
    num_file = len(files)

    graph_list = []
    graph_features_list = []

    for i in range(1, num_file + 1):
        geojson_filepath = "./2023_City_Team/" + city_name + '_dataset/Buildings/' + city_name + f'_buildings{i}.geojson'
        gdf = gpd.read_file(geojson_filepath)

        polygons = []
        semantics = []

        for _, row in gdf.iterrows():
            geom = row['geometry']
            coords = extract_polygon_coordinates(geom)
            polygons.append(coords)
            semantic = row['key']
            semantics.append(semantic)

        polygon_objects = [Polygon(p) for p in polygons]

        mbrs = [p.minimum_rotated_rectangle for p in polygon_objects]

        widths_heights = []
        for mbr in mbrs:
            coords = np.array(mbr.exterior.coords)
            width = np.linalg.norm(coords[0] - coords[1])
            height = np.linalg.norm(coords[1] - coords[2])
            widths_heights.append((width, height))

        centroids = [p.centroid for p in polygon_objects]

        point_list = []

        for idx, centroid in enumerate(centroids):
            point_list.append((centroid.x, centroid.y))

        point_list = list(set(point_list))
        points = np.array(point_list)

        sorted_points = sort_points_ccw(points)

        G = nx.Graph()

        for i, point in enumerate(sorted_points):
            G.add_node(i, pos=point)

        for i in range(len(sorted_points)):
            G.add_edge(i, (i + 1) % len(sorted_points))

        if G.number_of_nodes() > 0:
            graph_list.append(G)
            for node in G.nodes:
                centroid = centroids[node]
                width_height = widths_heights[node]
                semantic = semantics[node]
                coord = polygons[node]
                if semantic in ['shop', 'supermarket', 'restaurant', 'tourism', 'accommodation']:
                    building_level = commercial
                if semantic in ['kindergarten', 'school', 'college', 'university']:
                    building_level = education
                if semantic in ['police_station', 'ambulance_station', 'fire_station']:
                    building_level = emergency
                if semantic in ['bank', 'bureau_de_change']:
                    building_level = financial
                if semantic in ['government_office', 'embassy', 'military', 'post_office']:
                    building_level = government
                if semantic in ['doctor', 'dentist', 'clinic', 'hospital', 'pharmacy', 'alternative']:
                    building_level = healthcare
                if semantic in ['park', 'cemetery', 'argriculture', 'solid_waste']:
                    building_level = landuse
                if semantic in ['forest', 'grassland']:
                    building_level = natural
                if semantic in ['place_of_worship', 'community_centre', 'library', 'historic', 'toilet']:
                    building_level = public
                if semantic in ['stadium', 'swimming_pool', 'pitch', 'sport_centre']:
                    building_level = sport
                if semantic in ['reservoir', 'waterway', 'coastline', 'water_body', 'wetland']:
                    building_level = water
                if semantic in ['residence']:
                    building_level = residence
                graph_features_list.append([(centroid.x, centroid.y), coord, width_height, building_level, semantic])

    print(graph_features_list)

    return graph_list, graph_features_list

graph_dataloader('littlerock')