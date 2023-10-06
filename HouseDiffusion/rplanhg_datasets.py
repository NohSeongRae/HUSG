import math
import random
import torch

from PIL import Image, ImageDraw
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
from glob import glob
import json
import os
import cv2 as cv
from tqdm import tqdm
from shapely import geometry as gm
from shapely.ops import unary_union
from collections import defaultdict
import copy


def load_rplanhg_data(batch_size, analog_bit, target_set=8, set_name='train'):
    """
    For a dataset, create a generator over (shape, kwargs) pairs
    """
    print(f"loading {set_name} of target set {target_set}")
    determinisitic = False if set_name=='train' else True
    dataset=RPlanghgDataset(set_name, analog_bit, target_set)



def make_non_manhattan(poly, polygon, house_poly):
    pass


get_bin = lambda x, z: [int(y) for y in format(x, 'b').zfill(z)]
get_one_hot = lambda x, z: np.eye(z)[x]

class RPlanghgDataset(Dataset):
    def __init__(self, set_name, analog_bit, target_set, non_manhattan=False):
        super().__init__()
        base_dir='.../datasets/rplan'
        self.non_manhattan=non_manhattan
        self.set_name=set_name
        self.analog_bit=analog_bit
        self.target_set=target_set
        self.subgraphs=[]
        self.org_graphs=[]
        self.org_houses=[]
        max_num_points=100
        if self.set_name=='eval':
            cnumber_dist=np.load(f'processed_rplan/rplan_train_{target_set}_cndist.npz', allow_pickle=True)['cnumber_dist'].item()
        if os.path.exists(f'processed_rplan/rplan_{set_name}_{target_set}.npz'):
            data=np.load(f'processed_rplan/rplan_{set_name}_{target_set}.npz', allow_pickle=True)
            self.graphs=data['graphs']
            self.houses=data['houses']
            self.door_masks=data['self_masks']
            self.gen_masks=data['gen_masks']
            self.num_coords=2
            self.max_num_points=max_num_points
            cnumber_dist=np.load(f'processed_rplan/rplan_train_{target_set}_cndist.npz', allow_pickle=True)['cnumber_dist'].item()
            if self.set_name=='eval':
                data=np.load(f'processed_rplan/rplan_{set_name}_{target_set}_syn.npz', allow_pickle=True)
                self.syn_graphs=data['graphs']
                self.syn_houses=data['houses']
                self.syn_door_masks=data['door_masks']
                self.syn_self_masks=data['self_masks']
                self.syn_gen_masks=data['gen_masks']
        else:
            with open(f'{base_dir}/list.txt') as f:
                lines=f.readlines()
            cnt=0
            for line in tqdm(lines):
                cnt=cnt+1
                file_name=f'{base_dir}/{line[:-1]}'
                rms_type, fd_eds, rms_bbs, eds_to_rms=reader(file_name)
                fp_size=len([x for x in rms_type if x != 15 and x !=17])
    def __len__(self):
        pass
    def __getitem__(self, idx):
        pass

    def make_sequence(self, edges):
        pass

    def build_graph(self, rms_type, fp_eds, eds_to_rms, out_size=64):
        pass

def is_adjacent(box_a, box_b, threshold=0.03):
    pass

def reader(filename):
    with open(filename) as f:
        info = json.load(f)
        rms_bbs=np.asarray(info['boxes'])
        fp_eds=info['edges']
        rms_type=info['room_type']
        eds_to_rms=info['ed_rm']
        s_r=0
        for rmk in range(len(rms_type)):
            if(rms_type[rmk]!=17):
                s_r=s_r+1
        rms_bbs=np.array(rms_bbs)/256.0
        fp_eds=np.array(fp_eds)/256.0
        fp_eds=fp_eds[:, :4]
        t1=np.min(rms_bbs[:,:2], 0)
        br=np.max(rms_bbs[:,2:], 0)
        shift=(t1+br)/2.0-0.5
        rms_bbs[:, :2]-=shift
        rms_bbs[:, 2:]-=shift


if __name__=='__main__':
    dataset=RPlanghgDataset('eval', False, 8)
