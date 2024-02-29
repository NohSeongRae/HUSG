import os
import pickle
import networkx as nx
import torch
from torch_geometric.data import Dataset, download_url
import numpy as np
from scipy import stats
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
# from Spatial_Encoder import TheoryGridCellSpatialRelationEncoder
from PIL import Image
import torchvision.transforms as transforms
import math


# bldg_type_num = 6

def __add_gaussain_noise(img, scale):
    ow, oh = img.size
    mean = 12
    sigma = 4
    gauss = np.random.normal(mean, sigma, (ow, oh))
    gauss = gauss.reshape(ow, oh)
    img = np.array(img, dtype=np.uint8)

    mask_0 = (img == 0)
    mask_255 = (img == 255)

    img[mask_0] = img[mask_0] + gauss[mask_0]
    img[mask_255] = img[mask_255] - gauss[mask_255]
    return img

from functools import partial

def add_gaussian_noise_with_range(img, noise_range):
    return __add_gaussain_noise(img, noise_range)



def get_transform(noise_range=0.0, noise_type=None, isaug=False, rescale_size=64):
    transform_list = []

    if noise_type == 'gaussian':
        noise_func = partial(add_gaussian_noise_with_range, noise_range=noise_range)
        transform_list.append(transforms.Lambda(noise_func))


    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5), (0.25))]
    return transforms.Compose(transform_list)


GRAPH_EXTENSIONS = [
    '.gpickle',
]

PROCESSED_EXTENSIONS = [
    '.gpickle',
]


# ### global defintion
# quantile_level = 9
# node_attr_onehot_classnum = [2, 10, 10, 11] # node_type, posx, posy, bldg_area
# edge_attr_onehot_classnum = [10, 4] #edge_dist, edge_type


# frequency_num = 32
# f_act = 'relu'
# max_radius = 10000
# min_radius = 10
# freq_init = 'geometric'


# spa_enc = TheoryGridCellSpatialRelationEncoder(
#     64,
#     coord_dim = 2,
#     frequency_num = frequency_num,
#     max_radius = max_radius,
#     min_radius = min_radius,
#     freq_init = freq_init)


# idx_enc = TheoryGridCellSpatialRelationEncoder(
#     64,
#     coord_dim = 2,
#     frequency_num = frequency_num,
#     max_radius = max_radius,
#     min_radius = min_radius,
#     freq_init = freq_init)


def is_graph_file(filename):
    return any(filename.endswith(extension) for extension in GRAPH_EXTENSIONS)


def is_processed_file(filename):
    return any(filename.endswith(extension) for extension in PROCESSED_EXTENSIONS)


def get_node_attribute(g, keys, dtype, default=None):
    attri = list(nx.get_node_attributes(g, keys).items())
    for i in range(len(list(attri))):
        _, attri[i] = attri[i]
    if attri:
        if isinstance(attri[0], torch.Tensor) and attri[0].is_cuda:
            attri = [x.cpu().numpy() for x in attri]
        else:
            attri = np.array(attri, dtype=dtype)
    else:
        attri = np.full(g.number_of_nodes(), default, dtype=dtype)

    attri = np.array(attri)
    return attri


def get_edge_attribute(g, keys, dtype, default=None):
    attri = list(nx.get_edge_attributes(g, keys).items())
    attri = np.array(attri)
    attri = attri[:, 1]
    attri = np.array(attri, dtype=dtype)
    return attri


# def get_node_attribute(g, attribute_name, dtype):
#     attri = [g.nodes[node][attribute_name] for node in g.nodes()]
#
#     # Tensor가 CUDA Tensor인 경우 CPU로 이동
#     if attri and isinstance(attri[0], torch.Tensor) and attri[0].is_cuda:
#         attri = [x.cpu() for x in attri]
#
#     attri = np.array(attri, dtype=dtype)
#     return attri


def graph2vector_processed(g):
    num_nodes = g.number_of_nodes()

    asp_rto = g.graph['aspect_ratio']
    longside = g.graph['long_side']

    posx = get_node_attribute(g, 'posx', np.double)
    posy = get_node_attribute(g, 'posy', np.double)

    size_x = get_node_attribute(g, 'size_x', np.double)
    size_y = get_node_attribute(g, 'size_y', np.double)

    exist = get_node_attribute(g, 'exist', np.int_)
    merge = get_node_attribute(g, 'merge', np.int_)

    b_shape = get_node_attribute(g, 'shape', np.int_)
    b_iou = get_node_attribute(g, 'iou', np.double)

    node_attr = np.stack((exist.squeeze(), merge), 1)

    edge_list = np.array(list(g.edges()), dtype=np.int_)
    edge_list = np.transpose(edge_list)

    node_pos = np.stack((posx, posy), 1)
    node_size = np.stack((size_x, size_y), 1)

    node_idx = np.stack((np.arange(num_nodes) / num_nodes, np.arange(num_nodes) / num_nodes), axis=1)

    if isinstance(node_size, torch.Tensor) and node_size.is_cuda:
        node_size = node_size.cpu().numpy()
    else:
        node_size = np.array(node_size)

    if isinstance(node_pos, torch.Tensor) and node_pos.is_cuda:
        node_pos = node_pos.cpu().numpy()
    else:
        node_pos = np.array(node_pos)

    if isinstance(node_attr, torch.Tensor) and node_attr.is_cuda:
        node_attr = node_attr.cpu().numpy()
    else:
        node_attr = np.array(node_attr)

    if isinstance(edge_list, torch.Tensor) and edge_list.is_cuda:
        edge_list = edge_list.cpu().numpy()
    else:
        edge_list = np.array(edge_list)

    if isinstance(node_idx, torch.Tensor) and node_idx.is_cuda:
        node_idx = node_idx.cpu().numpy()
    else:
        node_idx = np.array(node_idx)

    if isinstance(asp_rto, torch.Tensor) and asp_rto.is_cuda:
        asp_rto = asp_rto.cpu().numpy()
    else:
        asp_rto = np.array(asp_rto)

    if isinstance(longside, torch.Tensor) and longside.is_cuda:
        longside = longside.cpu().numpy()
    else:
        longside = np.array(longside)

    if isinstance(b_shape, torch.Tensor) and b_shape.is_cuda:
        b_shape = b_shape.cpu().numpy()
    else:
        b_shape = np.array(b_shape)

    if isinstance(b_iou, torch.Tensor) and b_iou.is_cuda:
        b_iou = b_iou.cpu().numpy()
    else:
        b_iou = np.array(b_iou)

    return node_size, node_pos, node_attr, edge_list, node_idx, asp_rto, longside, b_shape, b_iou


def test_graph_transform(data):
    midaxis = data[0].midaxis

    boundary_polygon = data[0].boundary_polygon

    num_nodes = data[0].x.shape[0]
    node_feature = data[0].x
    edge_idx = data[0].edge_index

    org_node_size = data[0].node_size
    org_node_size = torch.tensor(org_node_size, dtype=torch.float32)

    node_size = np.expand_dims(org_node_size, axis=0)
    node_size = node_size.squeeze(0)

    org_node_pos = data[0].node_pos
    org_node_pos = torch.tensor(org_node_pos, dtype=torch.float32)

    node_pos = np.expand_dims(org_node_pos, axis=0)
    node_pos = node_pos.squeeze(0)

    b_shape_gt = torch.tensor(data[0].b_shape, dtype=torch.int64)
    b_shape = torch.tensor(F.one_hot(b_shape_gt.clone().detach(), num_classes=6), dtype=torch.float32)
    b_iou = torch.tensor(data[0].b_iou, dtype=torch.float32).unsqueeze(1)

    node_feature = torch.tensor(node_feature, dtype=torch.float32)

    edge_idx = torch.tensor(edge_idx, dtype=torch.long)

    node_idx = torch.tensor(data[0].node_idx, dtype=torch.float32)
    node_idx = np.expand_dims(data[0].node_idx, axis=0)
    node_idx = node_idx.squeeze(0)

    long_side = torch.tensor(data[0].long_side, dtype=torch.float32).repeat(num_nodes).unsqueeze(1)
    asp_rto = torch.tensor(data[0].asp_rto, dtype=torch.float32).repeat(num_nodes).unsqueeze(1)

    long_side_gt = torch.tensor(data[0].long_side, dtype=torch.float32)
    asp_rto_gt = torch.tensor(data[0].asp_rto, dtype=torch.float32)

    blockshape_latent = torch.tensor(data[0].blockshape_latent / 10.0, dtype=torch.float32).repeat(num_nodes, 1)
    block_scale = torch.tensor(data[0].block_scale / 100.0, dtype=torch.float32).repeat(num_nodes).unsqueeze(1)

    blockshape_latent_gt = torch.tensor(data[0].blockshape_latent / 10.0, dtype=torch.float32)
    block_scale_gt = torch.tensor(data[0].block_scale / 100.0, dtype=torch.float32)

    trans_data = Data(x=node_feature, edge_index=edge_idx, node_pos=node_pos, org_node_pos=org_node_pos,
                      node_size=node_size, org_node_size=org_node_size,
                      node_idx=node_idx, asp_rto=asp_rto, long_side=long_side, asp_rto_gt=asp_rto_gt,
                      long_side_gt=long_side_gt,
                      b_shape=b_shape, b_iou=b_iou, b_shape_gt=b_shape_gt,
                      blockshape_latent=blockshape_latent, blockshape_latent_gt=blockshape_latent_gt,
                      block_scale=block_scale, block_scale_gt=block_scale_gt,
                      block_condition=data[0].block_condition, org_binary_mask=data[0].org_binary_mask, midaxis=midaxis, boundary_polygon=boundary_polygon)

    return trans_data


def graph_transform(data):

    midaxis = data.midaxis
    boundary_polygon = data.boundary_polygon

    num_nodes = data.x.shape[0]
    node_feature = data.x
    edge_idx = data.edge_index

    org_node_size = data.node_size
    org_node_pos = data.node_pos
    b_shape_org = data.b_shape
    b_iou = data.b_iou

    randm = torch.rand(1)
    if randm < 0.5:
        org_node_size = np.flip(org_node_size, 0).copy()
        org_node_pos = np.flip(org_node_pos, 0).copy()
        b_shape_org = np.flip(b_shape_org, 0).copy()
        b_iou = np.flip(b_iou, 0).copy()
        node_feature = np.flip(node_feature, 0).copy()

    org_node_size = torch.tensor(org_node_size, dtype=torch.float32)

    # node_size = spa_enc(np.expand_dims(org_node_size, axis=0))
    node_size = np.expand_dims(org_node_size, axis=0)
    node_size = node_size.squeeze(0)

    org_node_pos = torch.tensor(org_node_pos, dtype=torch.float32)

    node_pos = np.expand_dims(org_node_pos, axis=0)
    node_pos = node_pos.squeeze(0)

    b_shape_gt = torch.tensor(b_shape_org, dtype=torch.int64)
    b_shape = torch.tensor(F.one_hot(b_shape_gt.clone().detach(), num_classes=6), dtype=torch.float32)

    b_iou = torch.tensor(b_iou, dtype=torch.float32).unsqueeze(1)

    node_feature = torch.tensor(node_feature, dtype=torch.float32)

    edge_idx = torch.tensor(edge_idx, dtype=torch.long)

    node_idx = torch.tensor(data.node_idx, dtype=torch.float32)
    # node_idx = idx_enc(np.expand_dims(data.node_idx, axis=0))
    node_idx = np.expand_dims(data.node_idx, axis=0)
    node_idx = node_idx.squeeze(0)

    long_side = torch.tensor(data.long_side, dtype=torch.float32).repeat(num_nodes).unsqueeze(1)
    asp_rto = torch.tensor(data.asp_rto, dtype=torch.float32).repeat(num_nodes).unsqueeze(1)

    long_side_gt = torch.tensor(data.long_side, dtype=torch.float32)
    asp_rto_gt = torch.tensor(data.asp_rto, dtype=torch.float32)

    blockshape_latent = torch.tensor(data.blockshape_latent / 10.0, dtype=torch.float32).repeat(num_nodes, 1)
    block_scale = torch.tensor(data.block_scale / 100.0, dtype=torch.float32).repeat(num_nodes).unsqueeze(1)

    blockshape_latent_gt = torch.tensor(data.blockshape_latent / 10.0, dtype=torch.float32)
    block_scale_gt = torch.tensor(data.block_scale / 100.0, dtype=torch.float32)

    trans_data = Data(x=node_feature, edge_index=edge_idx, node_pos=node_pos, org_node_pos=org_node_pos,
                      node_size=node_size, org_node_size=org_node_size,
                      node_idx=node_idx, asp_rto=asp_rto, long_side=long_side, asp_rto_gt=asp_rto_gt,
                      long_side_gt=long_side_gt,
                      b_shape=b_shape, b_iou=b_iou, b_shape_gt=b_shape_gt,
                      blockshape_latent=blockshape_latent, blockshape_latent_gt=blockshape_latent_gt,
                      block_scale=block_scale, block_scale_gt=block_scale_gt,
                      block_condition=data.block_condition, org_binary_mask=data.org_binary_mask, midaxis=midaxis, boundary_polygon=boundary_polygon)

    return trans_data


class UrbanGraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, cnn_transform=None, is_multigpu=False):
        super().__init__(root, transform, pre_transform)
        self.root_data_path = root
        self.cnn_transforms = cnn_transform
        self.base_transform = transforms.Compose([transforms.ToTensor()])

    @property
    def raw_file_names(self):
        raw_graph_dir = []
        for root, _, fnames in sorted(os.walk(self.raw_dir)):
            for fname in fnames:
                if is_graph_file(fname):
                    path = os.path.join(root, fname)
                    raw_graph_dir.append(path)
        return raw_graph_dir

    @property
    def processed_file_names(self):
        processed_graph_dir = []
        for root, _, fnames in sorted(os.walk(self.processed_dir)):
            for fname in fnames:
                if is_processed_file(fname):
                    path = os.path.join(root, fname)
                    processed_graph_dir.append(path)
        return processed_graph_dir

    def process(self):
        print('processed')
        # self.dataset_attribute_discretize() # self.root inherited from BaseDataset

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        # called by get_item() in BaseDatset, after get(), base dataset will implement transform()
        tmp_graph = nx.read_gpickle(os.path.join(self.processed_dir, '{}.gpickle'.format(idx)))

        block_mask = Image.fromarray(tmp_graph.graph['binary_mask'])
        block_scale = tmp_graph.graph['block_scale']

        midaxis = tmp_graph.graph['midaxis']
        boundary_polygon = tmp_graph.graph['polygon']

        trans_image = self.cnn_transforms(block_mask)
        scale_channel = math.log(block_scale, 20) / 2.0
        # if scale_channel < 0.0:
        #     scale_channel = 0.0

        scale_channel = torch.tensor([scale_channel]).expand(trans_image.shape)
        block_condition = torch.cat((trans_image, scale_channel), 0)

        blockshape_latent = np.zeros(40)
        node_size, node_pos, node_feature, edge_index, node_idx, asp_rto, long_side, b_shape, b_iou = graph2vector_processed(
            tmp_graph)

        data = Data(x=node_feature, node_pos=node_pos, edge_index=edge_index, node_size=node_size, node_idx=node_idx,
                    asp_rto=asp_rto,
                    long_side=long_side, b_shape=b_shape, b_iou=b_iou,
                    blockshape_latent=blockshape_latent, block_scale=block_scale, block_condition=block_condition,
                    org_binary_mask=self.base_transform(block_mask), midaxis=midaxis, boundary_polygon=boundary_polygon)

        return data, midaxis, boundary_polygon

# root = os.getcwd()
# a = UrbanGraphDataset(os.path.join(root,'dataset','synthetic'),transform = graph_transform)
# train_loader = DataLoader(a, batch_size=6, shuffle=True)
# count = 0
# for data in train_loader:
#     # print(data)
#     # print(count)
#     print(data.x, data.edge_index, data.node_pos)
#     count += 1