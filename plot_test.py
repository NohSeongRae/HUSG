import os
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx

graph_data_path = "./preprocessing/dataset/husg_diffusion_dataset_graphs.gpickle"

graph_list = nx.read_gpickle(graph_data_path)

