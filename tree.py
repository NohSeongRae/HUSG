import sys
import os
import json
import torch
import torch_geometric
import numpy as np
from torch.utils import data
from sklearn.decomposition import PCA
from collections import namedtuple
# from utils import one_hot #for semantic encoding
# import trimesh #for 3D triangular mesh importing

class Tree(object):
    part_name2id=dict()
    part_id2name=dict()
    part_name2cids=dict()

    part_non_leaf_sem_names=[]
    num_sem=None
    root_sem=None

    @staticmethod
    def load_category_info(cat):
        with open(os.path.join('../stats/building')) as fin:
            pass #some building json related semantic relationship exporting

    class polygon(object):
        def __init__(self, vertices=None, ):
            self.vertices=vertices #let vertices as list or set of local (x,y) coordinates
            self.dir_vectors=get_dir_vectors(self.vertices)
        def get_dir_vectors(self):
            for i in self.vertices:
                coords_i=i
                next_coords=next(i)

            return d_vecs
    class Node(object):
        def __init__(self, node_id=0,  box=None,poly=None, center=None, size=None, ratio=None, level=None, semantic=None, children=None):
            self.node_id=node_id
            self.box=box #[c_x, c_y, width, height, x_dir, y_dir]
            self.ploy=poly
            self.center=center
            self.size=size
            self.ratio=ratio
            self.level=level
            self.semantic=semantic
            self.children=[] if children is None else children

        def get_semantic_id(self):
            return Tree.part_name2id[self.semantic]

        def get_box_edgeside_polyline(self):
            box=self.box
            center=box[:2]
            size=box[2]*box[3]
            x_dir=box[4]
            y_dir=box[5]





















