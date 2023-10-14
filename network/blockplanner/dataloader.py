import torch
from torch_geometric.data import Data, Dataset
import numpy as np
import networkx as nx
import os
import random

class BlockplannerDataset(Dataset):
    def __init__(self, n_building, n_semantic):
        super().__init__()

        self.dataset_path = './dataset/'
        self.file_names = [f for f in os.listdir(self.dataset_path) if os.path.isfile(os.path.join(self.dataset_path, f))]
        self.n_building = n_building
        self.n_semantic = n_semantic

        self.test = True
    def get(self, index):
        if self.test:
            return self.generate_random_graph(num_nodes=self.n_building, n_semantic=self.n_semantic)

        tmp_graph = nx.read_gpickle(os.path.join(self.dataset_path, '{}.gpickle'.format(index)))
        num_nodes = tmp_graph.number_of_nodes()

        node_index = np.stack((np.arange(num_nodes) / num_nodes, np.arange(num_nodes) / num_nodes), axis=1)
        edge_index = tmp_graph.edge_index
        adj_matrix = nx.adjacency_matrix(tmp_graph)

        x = list(nx.get_node_attributes(tmp_graph, 'x').items())
        y = list(nx.get_node_attributes(tmp_graph, 'y').items())
        w = list(nx.get_node_attributes(tmp_graph, 'w').items())
        h = list(nx.get_node_attributes(tmp_graph, 'h').items())
        n = list(nx.get_node_attributes(tmp_graph, 'n').items())
        geometry = np.stack((x, y, w, h, n), 1)

        semantic = list(nx.get_node_attributes(tmp_graph, 'semantic').items())
        semantic = np.zeros((len(semantic), self.n_semantic))

        aspect_ratio = np.array(list(nx.get_node_attributes(tmp_graph, 'aspect_ratio').items()))

        node_exists_prob = np.array(list(nx.get_node_attributes(tmp_graph, 'node_exists_prob').items()))

        return Data(geometry=torch.tensor(geometry, dtype=torch.float32),
                    semantic=torch.tensor(semantic, dtype=torch.long),
                    adj_matrix=torch.tensor(adj_matrix, dtype=torch.long),
                    node_index=torch.tensor(node_index, dtype=torch.long),
                    aspect_ratio=torch.tensor(aspect_ratio, dtype=torch.float32),
                    node_exists_prob=torch.tensor(node_exists_prob, dtype=torch.float32),
                    edge_index=torch.tensor(edge_index, dtype=torch.long))

    def len(self):
        if self.test:
            return 100
        return len(self.file_names)

    def generate_random_graph(self, num_nodes=10, num_edges=15, n_semantic=11):
        # Create a random graph using networkx
        G = nx.gnm_random_graph(num_nodes, num_edges)

        # Randomly generate node attributes
        for i in range(num_nodes):
            G.nodes[i]['x'] = random.uniform(0, 1)
            G.nodes[i]['y'] = random.uniform(0, 1)
            G.nodes[i]['w'] = random.uniform(0, 1)
            G.nodes[i]['h'] = random.uniform(0, 1)
            G.nodes[i]['n'] = random.uniform(0, 1)
            G.nodes[i]['semantic'] = random.randint(0, n_semantic - 1)
            G.nodes[i]['aspect_ratio'] = random.uniform(1, 2)
            G.nodes[i]['node_exists_prob'] = random.uniform(0, 1)

        # Convert node attributes to numpy arrays
        geometry = np.array([[G.nodes[i]['x'], G.nodes[i]['y'], G.nodes[i]['w'], G.nodes[i]['h'], G.nodes[i]['n']] for i in G.nodes])
        semantic = np.array([G.nodes[i]['semantic'] for i in G.nodes])
        aspect_ratio = np.array([G.nodes[i]['aspect_ratio'] for i in G.nodes])
        node_exists_prob = np.array([G.nodes[i]['node_exists_prob'] for i in G.nodes])

        # Convert to torch tensors and create a Data object
        data = Data(
            geometry=torch.tensor(geometry, dtype=torch.float32),
            semantic=torch.tensor(semantic, dtype=torch.long),
            aspect_ratio=torch.tensor(aspect_ratio, dtype=torch.float32).view(-1, 1),
            node_exists_prob=torch.tensor(node_exists_prob, dtype=torch.float32).view(-1, 1)
        )

        # Convert networkx graph to edge_index format used by torch_geometric
        edge_index = torch.tensor(list(G.edges)).t().contiguous()
        data.edge_index = edge_index.view(2, -1)

        # Get adjacency matrix and convert to torch tensor
        adj_matrix = nx.adjacency_matrix(G).toarray()
        data.adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)

        return data