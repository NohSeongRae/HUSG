import os
import torch
from tqdm import tqdm
import numpy as np
from scipy.stats import wasserstein_distance
import pickle


base = 'grid_graph_figure'
base = 'ours_output/synthetic_graph'
path = ''
path = os.path.join(base, path)

list_output_all = os.listdir(path)

list_output = []
list_output_gt = []

for name in list_output_all:
    if 'ground_truth' in name and 'pkl' in name:
        list_output_gt.append(name)
    elif 'prediction' in name and 'pkl' in name:
        list_output.append(name)

xy = []
xy_gt = []
wh = []
wh_gt = []
theta = []
theta_gt = []

for o, o_gt in zip(list_output, list_output_gt):
    with open(f'{path}/{o}', 'rb') as f:
        tmp = pickle.load(f)

    with open(f'{path}/{o_gt}', 'rb') as f:
        tmp_gt = pickle.load(f)

    for t in tmp:
        xy.append([t[0], t[1]])
        wh.append([t[2], t[3]])
        theta.append(t[4])

    for t_gt in tmp_gt:
        xy_gt.append([t_gt[0], t_gt[1]])
        wh_gt.append([t_gt[2], t_gt[3]])
        theta_gt.append(t[4])

xy = np.array(xy)
xy_gt = np.array(xy_gt)
wh = np.array(wh)
wh_gt = np.array(wh_gt)
theta = np.array(theta)
theta_gt = np.array(theta_gt)
# theta = (np.array(theta) * 2 - 1) * 45
# theta_gt = (np.array(theta_gt) * 2 - 1) * 45

wd_xy = np.mean([
    wasserstein_distance(xy[:, i], xy_gt[:, i])
    for i in range(2)
])

wd_wh = np.mean([
    wasserstein_distance(wh[:, i], wh_gt[:, i])
    for i in range(2)
])

wd_theta = wasserstein_distance(theta, theta_gt)

print(wd_xy, wd_wh, wd_theta)