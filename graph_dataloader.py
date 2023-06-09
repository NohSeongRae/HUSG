import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
import networkx as nx
import os
import filepath
from get_buildinglevel import get_buildinglevel
import pandas as pd
import matplotlib.pyplot as plt

commercial, education, emergency, financial, government, healthcare, landuse, natural, public, sport, water, residence = get_buildinglevel()

def polar_angle(origin, point):
    delta_x = point[0] - origin[0]
    delta_y = point[1] - origin[1]
    angle = np.arctan2(delta_y, delta_x)
    return angle if angle >= 0 else 2 * np.pi + angle


def sort_points_ccw(points):
    if len(points) == 0:
        return []

    centroid = np.mean(points, axis=0)
    angles = np.array([polar_angle(centroid, point) for point in points])
    sorted_indices = np.argsort(angles)
    return points[sorted_indices]

new_coords = []

def extract_polygon_coordinates(geom):
    minx, miny, maxx, maxy = geom.bounds
    midx = (minx + maxx) / 2
    midy = (miny + maxy) / 2

    # use the larger range as the basis for normalization
    width = maxx - minx
    height = maxy - miny
    max_range = max(width, height)

    new_coords = []
    for point in list(geom.exterior.coords):
        lat, lon = point
        lat_norm = 2 * ((lat - midx) / max_range)
        lon_norm = 2 * ((lon - midy) / max_range)
        new_coords.append((lat_norm, lon_norm))

    return list(Polygon(new_coords).exterior.coords)


def graph_dataloader(city_name):
    dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Boundaries')
    files = os.listdir(dir_path)
    num_file = len(files)

    graph_list = []
    graph_features_list = []

    centroidx_list = []
    centroidy_list = []
    width_list = []
    height_list = []
    semantic_list = []
    coord_list = []
    buildinglevel_list = []
    group_list = []

    group = 1

    for i in range(1, num_file + 1):
        geojson_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                         'Buildings', f'{city_name}_buildings{i}.geojson')

        if os.path.exists(geojson_filepath):
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

            widths = []
            heights = []

            for mbr in mbrs:
                coords = np.array(mbr.exterior.coords)
                width = np.linalg.norm(coords[0] - coords[1])
                height = np.linalg.norm(coords[1] - coords[2])
                widths.append(width)
                heights.append(height)

            centroids = [p.centroid for p in polygon_objects]

            point_list = []

            attributes = {}
            for idx, (centroid, width, height, semantic, coords) in enumerate(
                    zip(centroids, widths, heights, semantics, polygons)):
                attributes[idx] = {
                    'centroid': centroid,
                    'width': width,
                    'height': height,
                    'semantic': semantic,
                    'coords': coords,
                }

            for idx, centroid in enumerate(centroids):
                point_list.append((centroid.x, centroid.y))

            point_list = list(set(point_list))
            points = np.array(point_list)

            sorted_points = sort_points_ccw(points)

            if i==2:
                plt.scatter(sorted_points[:, 0], sorted_points[:, 1], color='blue')

                # Plot index of each point
                for i, point in enumerate(sorted_points):
                    plt.text(point[0], point[1], str(i))

                # Connect points in order
                plt.plot(np.append(sorted_points[:, 0], sorted_points[0, 0]),
                         np.append(sorted_points[:, 1], sorted_points[0, 1]), 'r-')

                plt.show()

            G = nx.Graph()

            for i, point in enumerate(sorted_points):
                G.add_node(i, pos=point, **attributes[i])

            for i in range(len(sorted_points)):
                G.add_edge(i, (i + 1) % len(sorted_points))

            if G.number_of_nodes() > 0:
                graph_list.append(G)
                for node in G.nodes:
                    data = G.nodes[node]
                    if i == 2:
                        print(centroid)
                    centroid = data['centroid']
                    width = data['width']
                    height = data['height']
                    semantic = data['semantic']
                    coord = data['coords']
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
                    if semantic in ['park', 'cemetery', 'agriculture', 'solid_waste']:
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

                    centroidx_list.append(centroid.x)
                    centroidy_list.append(centroid.y)
                    coord_list.append(coord)
                    width_list.append(width)
                    height_list.append(height)
                    buildinglevel_list.append(building_level)
                    semantic_list.append(semantic)
                    group_list.append(group)
                    # graph_features_list.append([centroid.x, centroid.y, coord, width, height, building_level, semantic])

                group += 1

    df = pd.DataFrame({
        'group': group_list,
        'centroid.x': centroidx_list,
        'centroid.y': centroidy_list,
        'coord': coord_list,
        'width': width_list,
        'height': height_list,
        'building_level': buildinglevel_list,
        'semantic': semantic_list
    })

    df = df.drop_duplicates(subset=['centroid.x', 'centroid.y'])

    df.to_csv(filepath.graph_filepath, index=False)

    return graph_list, graph_features_list

# from cityname import city_name
# graph_dataloader(city_name)

if __name__ == '__main__':
    graph_dataloader('firenze')