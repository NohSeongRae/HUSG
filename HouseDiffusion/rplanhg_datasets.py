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
    if determinisitic:
        loader=DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False
        )
    else:
        loader=DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False
        )
    while True:
        yield from loader



# def make_non_manhattan(poly, polygon, house_poly):
#     pass


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
            self.door_masks=data['door_masks']
            self.self_masks=data['self_masks']
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
        #preprocessing code
        # else:
        #     with open(f'{base_dir}/list.txt') as f:
        #         lines=f.readlines()
        #     cnt=0
        #     for line in tqdm(lines):
        #         cnt=cnt+1
        #         file_name=f'{base_dir}/{line[:-1]}'
        #         rms_type, fp_eds, rms_bbs, eds_to_rms=reader(file_name)
        #         fp_size=len([x for x in rms_type if x != 15 and x !=17])
        #         if self.set_name=='train' and fp_size == target_set:
        #             continue
        #         if self.set_name=='eval' and fp_size != target_set:
        #             continue
        #         a = [rms_type, rms_bbs, fp_eds, eds_to_rms]
        #         self.subgraphs.append(a)
        #     for graph in tqdm(self.subgraphs):
        #         rms_type = graph[0]
        #         rms_bbs=graph[1]
        #         fp_eds=graph[2]
        #         eds_to_rms=graph[3]
        #         rms_bbs=np.array(rms_bbs)
        #         fp_eds=np.array(fp_eds)
        #
        #         #extract boundary box and centralize
        #         tl=np.min(rms_bbs[:, :2], 0)
        #         br=np.max(rms_bbs[:, 2:], 0)
        #         shift=(tl+br)/2.0-0.5
        #         rms_bbs[:, :2]-=shift
        #         rms_bbs[:, 2:]-=shift
        #         fp_eds[:, :2]-=shift
        #         fp_eds[:, 2:]-=shift
        #         tl-=shift
        #         br-=shift
        #
        #         # build input graph
        #         graph_nodes, graph_edges, rooms_mks=self.build_graph(rms_type, fp_eds, eds_to_rms)

    def __len__(self):
        return len(self.houses)
    def __getitem__(self, idx):
        # idx= int(idx//20)
        arr =self.houses[idx][:, :self.num_coords]
        graph=np.concatenate((self.graphs[idx], np.zeros([200-len(self.graphs[idx]), 3])), 0)

        cond ={
            'door_mask':self.door_masks[idx],
            'self_mask': self.self_masks[idx],
            'gen_mask': self.gen_masks[idx],
            'room_types': self.houses[idx][:, self.num_coords:self.num_coords+25],
            'corner_indices': self.houses[idx][:, self.num_coords+25:self.num_coords+57],
            'room_indices': self.houses[idx][:, self.num_coords+57: self.num_coords+89],
            'src_key_padding_mask': 1-self.houses[idx][:, self.num_coords+89],
            'connections':self.houses[idx][:, self.num_coords+90:self.num_coords+92],
            'graph':graph,
        }
        if self.set_name=='eval':
            syn_graph=np.concatenate((self.syn_graphs[idx], np.zeros([200-len(self.syn_graphs[idx]),3])),0)
            assert (graph==syn_graph).all(), idx
            cond.update({
                'syn_door_mask':self.syn_door_masks[idx],
                'syn_self_mask': self.syn_self_masks[idx],
                'syn_gen_mask': self.syn_gen_masks[idx],
                'syn_room_types':self.syn_houses[idx][:, self.num_coords:self.num_coords+25],
                'syn_corner_indices': self.syn_houses[idx][:, self.num_coords+25:self.num_coords+57],
                'syn_room_indices': self.syn_houses[idx][:, self.num_coords+57:self.num_coords+89],
                'syn_src_key_padding_mask': 1-self.syn_houses[idx][:, self.num_coords+89],
                'syn_connections': self.syn_houses[idx][:, self.num_coords+90:self.num_coords+92],
                'syn_graph': syn_graph,
            })
        if self.set_name == 'train':
            # Random rotation
            rotation=random.randint(0,3)
            if rotation == 1:
                arr[:, [0,1]]=arr[:, [1,0]]
                arr[:, 0]=-arr[:,0]
            elif rotation==2:
                arr[:, [0,1]]=-arr[:, [1,0]]
            elif rotation ==3:
                arr[:, [0,1]] = arr[:, [1,0]]
                arr[:, 1]=-arr[:,1]

            if self.non_manhattan:
                theta=random.random()*np.pi/2
                rot_mat=np.array([[np.cos(theta), -np.sin(theta), 0],
                                  [np.sin(theta), np.cos(theta),0]])
                arr=np.matmul(arr, rot_mat)[:,2]

        if not self.analog_bit:
            arr =np.transpose(arr, [1,0])
            return arr.astype(float), cond
        else:
            ONE_HOT_RES = 256
            arr_onehot = np.zeros((ONE_HOT_RES * 2, arr.shape[1])) - 1
            xs = ((arr[:, 0] + 1) * (ONE_HOT_RES / 2)).astype(int)
            ys = ((arr[:, 1] + 1) * (ONE_HOT_RES / 2)).astype(int)
            xs = np.array([get_bin(x, 8) for x in xs])
            ys = np.array([get_bin(x, 8) for x in ys])
            arr_onehot = np.concatenate([xs, ys], 1)
            arr_onehot = np.transpose(arr_onehot, [1, 0])
            arr_onehot[arr_onehot == 0] = -1
            return arr_onehot.astype(float), cond

    # def make_sequence(self, edges):
    #     pass

    # def build_graph(self, rms_type, fp_eds, eds_to_rms, out_size=64):
    #     """
    #     Build a graph and room masks based on input data
    #     :param rms_type: List, indicating the types of rooms/nodes.
    #     :param fp_eds:
    #     :param eds_to_rms: List, encoding relation between entities and rooms
    #     :param out_size: Int, desired output size for mask images (default 64)
    #     :return:
    #     - nodes: Array, representing node in the graph
    #     - triples: Array, storing adjacency information between nodes.
    #     - rms_masks: Array, storing binary mask images of rooms
    #     """
    #     #create edges
    #     triples=[]
    #     nodes=rms_type
    #     #encode connections
    #     for k in range(len(nodes)):
    #         for l in range(len(nodes)):
    #             if l>k:
    #                 is_adjacent=any([True for e_map in eds_to_rms if ( l in e_map) and (k in e_map)])
    #                 if is_adjacent:
    #                     triples.append([k,1,l])
    #                 else:
    #                     triples.append([k,-1,l])
    #             else:
    #                 break
    #     #get rooms masks
    #     eds_to_rms_tmp=[]
    #     for l in range(len(eds_to_rms)):
    #         eds_to_rms_tmp.append([eds_to_rms[l][0]])
    #     rms_masks=[]
    #     im_size=256
    #     fp_mk=np.zeros((out_size, out_size))
    #     for k in range(len(nodes)):
    #         #add roms and doors
    #         eds=[]
    #         for l , e_map in enumerate(eds_to_rms_tmp):
    #             if (k in e_map):
    #                 eds.append(l)
    #         # draw rooms
    #         rm_im = Image.new('L', (im_size, im_size))
    #         dr=ImageDraw.Draw(rm_im)
    #         for eds_poly in [eds]:
    #             poly=self.make_sequence(np.array([fp_eds[l][:4] for l in eds_poly]))[0]
    #             poly = [(im_size*x, im_size*y) for x, y, in poly]
    #             if len(poly)>=2:
    #                 dr.polygon(poly, fill='white')
    #             else:
    #                 print("Empty room")
    #                 exit(0)
    #         rm_im=rm_im.resize((out_size, out_size))
    #         rm_arr=np.array(rm_im)
    #         inds=np.where(rm_arr>0)
    #         rm_arr[inds]=1.0
    #         rms_masks.append(rm_arr)
    #         if rms_type[k]!=15 and rms_type[k] != 17:
    #             fp_mk[inds]=k+1
    #     #trick to remove overlap
    #     for k in range(len(nodes)):
    #         if rms_type[k]!=15 and rms_type[k]!=17:
    #             rm_arr = np.zeros((out_size, out_size))
    #             inds=np.where(fp_mk==k+1)
    #             rm_arr[inds]=1.0
    #             rms_masks[k]=rm_arr
    #     #convert to array
    #     nodes=np.array(nodes)
    #     triples=np.array(triples)
    #     rms_masks=np.array(rms_masks)
    #     return nodes, triples, rms_masks


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
