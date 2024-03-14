import concurrent.futures
from tqdm import tqdm
import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def generate_datasets(idx, data_type):
    with open(f'datasets/eu_graph_condition_train_datasets/{data_type}/{str(idx)}.pkl', 'rb') as file:
        buildings = pickle.load(file)

    graph = nx.read_gpickle(f'datasets/eu_graph_condition_train_datasets/{data_type}/{str(idx)}.gpickle')

    n_node = graph.number_of_nodes()
    n_building = len(buildings)
    n_chunk = n_node - n_building

    if n_building >= 2:
        adj_matrix = nx.adjacency_matrix(graph).todense()
        boundary_adj_matrix = adj_matrix[:n_chunk, :n_chunk]
        building_adj_matrix = adj_matrix[n_chunk:, n_chunk:]
        bb_adj_matrix = adj_matrix[n_chunk:, :n_chunk]
        boundary_pos_feature = []

        for node in graph.nodes():
            if node < n_chunk:
                boundary_pos_feature.append(graph.nodes[node]['node_features'][:2])
        boundary_pos_feature = np.array(boundary_pos_feature)

        data = {'boundary_adj_matrix': boundary_adj_matrix,
                'building_adj_matrix': building_adj_matrix,
                'bb_adj_matrix': bb_adj_matrix,
                'boundary_pos_feature': boundary_pos_feature,
                'n_boundary': n_chunk,
                'n_building': n_building}

        output_file_path = f'eu_graph_generation_datasets/{data_type}/'
        with open(f'{output_file_path}/{idx}.pickle', 'wb') as f:
            pickle.dump(data, f)

if __name__ == '__main__':
    end_index = 208622 + 1
    data_type = 'val'

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = []

        # tqdm의 total 파라미터를 설정합니다.
        progress = tqdm(total=end_index, desc='Processing files', position=0, leave=True)

        # submit 대신 map을 사용하여 future 객체를 얻고, 각 future가 완료될 때마다 진행 상황을 업데이트합니다.
        futures = [executor.submit(generate_datasets, file_index, data_type) for file_index in range(end_index)]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            progress.update(1)

        progress.close()