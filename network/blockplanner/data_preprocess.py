import pandas as pd
import io
import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import os
import geopandas as gpd
from tqdm import tqdm

matplotlib.use('TkAgg')


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


def create_graph_from_df(df):
    """Create a graph from a DataFrame."""
    # Get the centroid of each polygon in the 'geometry' column
    point_list = [(polygon.centroid.x, polygon.centroid.y) for polygon in df['geometry']]
    point_list = np.unique(point_list, axis=0)

    # Sort coordinates by polar angle
    sorted_indices = np.argsort([polar_angle(np.mean(point_list, axis=0), point) for point in point_list])
    sorted_points = point_list[sorted_indices]

    G = nx.Graph()

    # Add nodes and edges to the graph
    G.add_nodes_from([(i, {"pos": point}) for i, point in enumerate(sorted_points)])
    for i in range(len(sorted_points)):
        G.add_edge(i, (i + 1) % len(sorted_points))

    # Print node features
    print("Node features:")
    for node, attributes in G.nodes(data=True):
        print(f"Node {node}: {attributes}")

    # Print adjacency matrix
    adj_matrix = nx.adjacency_matrix(G)
    print("\nAdjacency matrix:")
    print(adj_matrix.toarray())


    return G


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
    # target_city = 'atlanta'
    # graph_list = graph_dataloader(target_city)
    #
    # for index, graph in enumerate(graph_list):
    #     save_graph(graph, target_city, index)
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

                # fig2, ax2 = plt.subplots(figsize=(8, 8))
                # boundary_gdf.boundary.plot(ax=ax2, color='blue', label='Rotated Block Boundary')
                # building_gdf.plot(ax=ax2, color='red', label='Rotated Buildings')
                # ax2.set_title("Aligned Building Block with Buildings")
                # ax2.legend()
                # plt.show()

                graph = create_graph_from_df(building_gdf)
                # plot_graph(graph)
                save_graph(graph, city_name, num)

            # counter += 1
            # if counter > 5:
            #     break
