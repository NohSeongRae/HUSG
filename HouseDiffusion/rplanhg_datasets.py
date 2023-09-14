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
    pass


def make_non_manhattan(poly, polygon, house_poly):
    pass


get_bin = lambda x, z: [int(y) for y in format(x, 'b').zfill(z)]
get_one_hot = lambda x, z: np.eye(z)[x]

class RPlanghgDataset(Dataset):
    def __init__(self, set_name, analog_bit, target_set, non_manhattan=False):
        pass
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
    pass

if __name__=='__main__':
    dataset=RPlanghgDataset('eval', False, 8)
