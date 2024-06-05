import os
from tqdm import tqdm
import numpy as np
import pickle
from shapely.geometry import Polygon
import math
import matplotlib.pyplot as plt
import networkx as nx

base = './globalmapper_figure'
path = ''
path = os.path.join(base, path)

list_output_all = os.listdir(path)

list_output = []

for name in list_output_all:
    if 'prediction' in name and 'png' in name:
        list_output.append(name)

for output in tqdm(list_output):
    file_path = f'{path}/{output}'
    file_idx = file_path.split('/')[-1].replace('prediction_', '').replace('.pkl', '')
    if not os.path.isfile(f'./badge_figure_pred_sample/{file_idx}.png'):
        os.remove(path + f'{output}')