import pandas as pd
import io
import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import os
import geopandas as gpd
from tqdm import tqdm
from shapely import affinity
from shapely.affinity import translate

matplotlib.use('TkAgg')

"""
attribute 
{xc, yc, w, h, n}
"""


def calculate_rotation_angle(polygon):
    min_rect = polygon.minimum_rotated_rectangle
    coords = list(min_rect.exterior.coords)
    edge1 = np.array(coords[1]) - np.array(coords[0])
    edge2 = np.array(coords[2]) - np.array(coords[1])

    if np.linalg.norm(edge1) > np.linalg.norm(edge2):
        angle = np.degrees(np.arctan2(edge1[1], edge1[0]))
    else:
        angle = np.degrees(np.arctan2(edge2[1], edge2[0]))

    return -angle


def rotate_gdf(gdf, angle):
    return gdf.apply(lambda row: affinity.rotate(row['geometry'], angle, use_radians=False), axis=1)


def polar_angle(origin, point):
    """Compute the polar angle of a point relative to an origin."""
    delta_x = point[0] - origin[0]
    delta_y = point[1] - origin[1]
    angle = np.arctan2(delta_y, delta_x)
    return angle if angle >= 0 else 2 * np.pi + angle


def sort_key(filename):
    # 파일 이름에서 숫자만 추출
    num = int(''.join(filter(str.isdigit, filename)))
    return num


def save_graph(graph, city_name, graph_index):
    """Save the graph to a .npz file."""

    # Capture adjacency list content directly into a binary stream
    bio = io.BytesIO()
    nx.write_adjlist(graph, bio)
    content = bio.getvalue().decode('utf-8')
    bio.close()

    # Define the output path
    output_directory = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                    f'{city_name}_graphs_blockplanner')
    os.makedirs(output_directory, exist_ok=True)
    output_path = os.path.join(output_directory, f'graph_{graph_index}.npz')

    # Save the content in a .npz file
    np.savez_compressed(output_path, adjacency_list=content)


def normalize(geometry, minx, miny, max_range):
    # Calculate the current center of the bbox
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2

    # Calculate the range for x and y dimensions
    x_range = maxx - minx
    y_range = maxy - miny

    max_range = max(x_range, y_range)

    # Translate geometry to have the current center at 0, 0
    geometry = affinity.translate(geometry, xoff=-center_x, yoff=-center_y)

    # Scale the geometry to fit within a unit range, maintaining aspect ratio
    scale_factor = 1 / max_range
    geometry = affinity.scale(geometry, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))

    # 축소된 도형의 새로운 bbox 계산
    new_minx, new_miny, new_maxx, new_maxy = geometry.bounds

    new_x_range = new_maxx - new_minx
    new_y_range = new_maxy - new_miny
    new_max_range = max(new_x_range, new_y_range)

    # Translate the geometry to have the bbox center at 0.5, 0.5
    geometry = affinity.translate(geometry, xoff=0.5, yoff=0.5)

    return geometry


def plot_graph(graph):
    """Visualize the graph."""
    pos = nx.get_node_attributes(graph, 'pos')
    nx.draw(graph, pos, node_color='blue', node_size=10)
    labels = nx.draw_networkx_labels(graph, pos, font_color='red')
    for node, label in labels.items():
        label.set_text(f"{node}")
    plt.show()


counter = 0
if __name__ == '__main__':
    # Change this to the city you want to load data for
    city_names = ["littlerock"]
    for city_name in city_names:
        print("city : ", city_name)
        building_dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                         'density20_building120_rotate_normalized', 'Buildings')
        boundary_dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                         'density20_building120_rotate_normalized', 'Boundaries')
        for building_filepath in tqdm(
                sorted([f for f in os.listdir(building_dir_path) if f.endswith('.geojson')], key=sort_key)):
            num = sort_key(building_filepath)
            boundary_filepath = building_filepath.replace('buildings', 'boundaries')
            building_filename = os.path.join(building_dir_path, building_filepath)
            boundary_filename = os.path.join(boundary_dir_path, boundary_filepath)
            if os.path.exists(building_filename):
                boundary_gdf = gpd.read_file(boundary_filename)
                building_gdf = gpd.read_file(building_filename)

                boundary_gdf = boundary_gdf.to_crs(boundary_gdf.estimate_utm_crs())
                building_gdf = building_gdf.to_crs(building_gdf.estimate_utm_crs())

                # scale
                scale_bounds = boundary_gdf.total_bounds  # [minx, miny, maxx, maxy]
                scale_minx, scale_miny, scale_maxx, scale_maxy = scale_bounds

                scale_x_range = scale_maxx - scale_minx
                scale_y_range = scale_maxy - scale_miny
                scale_factor = max(scale_x_range, scale_y_range)

                scale_factor = 1 / scale_factor

                boundary_polygon = boundary_gdf.iloc[0]['geometry']
                rotation_angle = calculate_rotation_angle(boundary_polygon)

                boundary_centroid = boundary_polygon.centroid


                def rotate_geometry(geometry, angle, origin):
                    return affinity.rotate(geometry, angle, origin=origin, use_radians=False)


                building_gdf['geometry'] = building_gdf['geometry'].apply(rotate_geometry, angle=rotation_angle,
                                                                          origin=(
                                                                              boundary_centroid.x, boundary_centroid.y))

                boundary_gdf['geometry'] = boundary_gdf['geometry'].apply(rotate_geometry, angle=rotation_angle,
                                                                          origin=(
                                                                              boundary_centroid.x, boundary_centroid.y))

                rotated_boundary_polygon = boundary_gdf.iloc[0]['geometry']

                center_x = rotated_boundary_polygon.centroid.x
                center_y = rotated_boundary_polygon.centroid.y

                move_vector_x = 0.0
                move_vector_y = 0.5 - center_y

                moved_vector = [move_vector_x, move_vector_y]

                moved_polygon = translate(rotated_boundary_polygon, xoff=move_vector_x, yoff=move_vector_y)

                ###
                boundary_gdf['geometry'] = gpd.GeoSeries(moved_polygon)
                # Translate the geometries in building_gdf
                building_gdf['geometry'] = building_gdf['geometry'].apply(
                    lambda geom: translate(geom, xoff=move_vector_x, yoff=move_vector_y)
                )

                bounds = boundary_gdf.total_bounds  # [minx, miny, maxx, maxy]
                minx, miny, maxx, maxy = bounds

                range_x = maxx - minx
                range_y = maxy - miny
                max_range = max(range_x, range_y)

                # 0 ~ 1 정규화
                boundary_gdf['geometry'] = boundary_gdf['geometry'].apply(normalize, minx=minx, miny=miny,
                                                                          max_range=max_range)

                min_x_values = boundary_gdf['geometry'].bounds.minx
                max_x_values = boundary_gdf['geometry'].bounds.maxx

                for min_value in min_x_values:
                    if min_value < 0.0:
                        translate_x = 0.0 - min_value
                        translate_y = 0.0

                        moved_vector = [translate_x, translate_y]

                        boundary_gdf['geometry'] = boundary_gdf['geometry'].apply(
                            lambda geom: translate(geom, xoff=translate_x, yoff=translate_y)
                        )
                        building_gdf['geometry'] = building_gdf['geometry'].apply(
                            lambda geom: translate(geom, xoff=translate_x, yoff=translate_y)
                        )

                max_x_values = boundary_gdf['geometry'].bounds.maxx

                for max_value in max_x_values:
                    if max_value > 1.0:
                        translate_x = 1.0 - max_value
                        translate_y = 0.0

                        moved_vector = [translate_x, translate_y]

                        ###
                        boundary_gdf['geometry'] = boundary_gdf['geometry'].apply(
                            lambda geom: translate(geom, xoff=translate_x, yoff=translate_y)
                        )
                        # Translate the geometries in building_gdf
                        building_gdf['geometry'] = building_gdf['geometry'].apply(
                            lambda geom: translate(geom, xoff=translate_x, yoff=translate_y)
                        )

                building_gdf['geometry'] = building_gdf['geometry'].apply(normalize, minx=minx, miny=miny,
                                                                          max_range=max_range)
                building_new_dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team',
                                                     f'{city_name}_dataset',
                                                     'density20_building120_rotate_normalized_v2', 'Buildings')
                boundary_new_dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team',
                                                     f'{city_name}_dataset',
                                                     'density20_building120_rotate_normalized_v2', 'Boundaries')
                os.makedirs(building_new_dir_path, exist_ok=True)
                os.makedirs(boundary_new_dir_path, exist_ok=True)

                building_gdf.to_file(
                    os.path.join(building_new_dir_path, f'{city_name}_buildings{num}.geojson'),
                    driver='GeoJSON')
                boundary_gdf.to_file(os.path.join(boundary_new_dir_path, f'{city_name}_boundaries{num}.geojson'),
                                     driver='GeoJSON')
