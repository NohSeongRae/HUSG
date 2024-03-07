import os
import argparse
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import networkx as nx
import numpy as np
import random
from tqdm import tqdm

from model import GraphTransformer
from dataloader import GraphDataset

import wandb

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Initialize a transformer with user-defined hyperparameters.")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--max_epoch", type=int, default=1000, help="Maximum number of epochs for training.")
    parser.add_argument("--d_model", type=int, default=512, help="Dimension of the model.")
    parser.add_argument("--n_layer", type=int, default=6, help="Number of transformer layers.")
    parser.add_argument("--n_head", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate used in the transformer model.")
    parser.add_argument("--seed", type=int, default=327, help="Random seed for reproducibility across runs.")
    parser.add_argument("--use_tensorboard", type=bool, default=True, help="Use tensorboard.")
    parser.add_argument("--use_checkpoint", type=bool, default=False, help="Use checkpoint model.")
    parser.add_argument("--checkpoint_epoch", type=int, default=0, help="Use checkpoint index.")
    parser.add_argument("--val_epoch", type=int, default=1, help="Use checkpoint index.")
    parser.add_argument("--save_epoch", type=int, default=10, help="Use checkpoint index.")
    parser.add_argument("--local-rank", type=int)
    parser.add_argument("--save_dir_path", type=str, default="transformer_graph", help="save dir path")
    parser.add_argument("--lr", type=float, default=3e-5, help="save dir path")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="save dir path")

    opt = parser.parse_args()

    # Set the random seed for reproducibility
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    device = torch.device('cpu')

    # Subsequent initializations will use the already loaded full dataset
    test_dataset = GraphDataset(data_type='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    transformer = GraphTransformer(opt.d_model, opt.d_model * 4, opt.n_layer, opt.n_head, opt.dropout).to(device=device)

    checkpoint = torch.load("./models/" + opt.save_dir_path + "/epoch_"+ 'best' + ".pth")
    transformer.load_state_dict(checkpoint['model_state_dict'])

    transformer.eval()
    with torch.no_grad():
        vis = False
        save = True

        count = 1
        for data in tqdm(test_dataloader):
            data, file = data
            file = file[0].replace('.pickle', '')
            if 'grid' in file:
                file_idx = file.replace('grid_', '')
                graph_type = 'grid_graph'
            elif 'ring' in file:
                file_idx = file.replace('ring_', '')
                graph_type = 'ring_graph'
            elif 'line' in file:
                file_idx = file.replace('line_', '')
                graph_type = 'line_graph'
            elif 'random' in file:
                file_idx = file.replace('random_', '')
                graph_type = 'random_graph'

            if 'small' in file:
                file_idx = file_idx.replace('small_', '')
                size_type = 'small'
            elif 'middle' in file:
                file_idx = file_idx.replace('middle_', '')
                size_type = 'middle'
            elif 'large' in file:
                file_idx = file_idx.replace('large_', '')
                size_type = 'large'


            building_adj_matrix_padded = data['building_adj_matrix_padded'].to(device=device)
            boundary_adj_matrix_padded = data['boundary_adj_matrix_padded'].to(device=device)
            bb_adj_matrix_padded = data['bb_adj_matrix_padded'].to(device=device)
            boundary_pos_padded = data['boundary_pos_padded'].to(device=device)
            building_pad_mask = data['building_pad_mask'].to(device=device)
            boundary_pad_mask = data['boundary_pad_mask'].to(device=device)
            bb_pad_mask = data['bb_pad_mask'].to(device=device)
            n_boundary = data['n_boundary']
            n_building = data['n_building']

            if boundary_adj_matrix_padded.shape[1] > 200:
                print(boundary_adj_matrix_padded.shape)
                continue

            output = transformer(building_adj_matrix_padded, boundary_adj_matrix_padded,
                                 building_pad_mask, boundary_pad_mask, boundary_pos_padded)
            output *= bb_pad_mask
            output_1 = (output >= 0.5).float()

            k = 3  # 상위 k개 값을 선택

            # 각 120x200 행렬에 대해 top k 값을 찾기
            topk_values, topk_indices = torch.topk(output, k, dim=2)

            # 모든 값을 0으로 설정한 다음, topk 인덱스에 해당하는 위치만 1로 설정
            output_2 = torch.zeros_like(output)
            output_2.scatter_(2, topk_indices, 1)

            output = torch.zeros_like(output_1)
            if torch.sum(output_1) <= n_building * k:
                output = output_2
            else:
                output = output_1

            pred_adj_matrix = np.zeros((n_boundary + n_building, n_boundary + n_building))
            gt_adj_matrix = np.zeros((n_boundary + n_building, n_boundary + n_building))

            pred_adj_matrix[:n_boundary, :n_boundary] = boundary_adj_matrix_padded.squeeze(0).detach().cpu().numpy()[:n_boundary, :n_boundary]
            pred_adj_matrix[n_boundary:, n_boundary:] = building_adj_matrix_padded.squeeze(0).detach().cpu().numpy()[:n_building, :n_building]
            pred_adj_matrix[n_boundary:, :n_boundary] = output.squeeze(0).detach().cpu().numpy()[:n_building, :n_boundary]
            pred_adj_matrix[:n_boundary, n_boundary:] = output.squeeze(0).detach().cpu().numpy().T[:n_boundary, :n_building]

            gt_adj_matrix[:n_boundary, :n_boundary] = boundary_adj_matrix_padded.squeeze(0).detach().cpu().numpy()[:n_boundary, :n_boundary]
            gt_adj_matrix[n_boundary:, n_boundary:] = building_adj_matrix_padded.squeeze(0).detach().cpu().numpy()[:n_building, :n_building]
            gt_adj_matrix[n_boundary:, :n_boundary] = bb_adj_matrix_padded.squeeze(0).detach().cpu().numpy()[:n_building, :n_boundary]

            if vis:
                G_pred = nx.DiGraph(pred_adj_matrix)
                G_gt = nx.DiGraph(gt_adj_matrix)

                union_graph = nx.compose(G_pred, G_gt)  # create a union of both graphs to ensure all nodes are considered
                pos = nx.spring_layout(union_graph)

                # Assign colors based on the threshold
                node_colors_pred = ['lightblue' if i < n_boundary else 'lightgreen' for i in G_pred.nodes()]
                node_colors_gt = ['lightblue' if i < n_boundary else 'lightgreen' for i in G_gt.nodes()]

                plt.figure(figsize=(12, 5))

                plt.subplot(1, 2, 1)
                nx.draw(G_pred, pos, with_labels=True, node_color=node_colors_pred, edge_color='gray')
                plt.title("Prediction Graph")

                plt.subplot(1, 2, 2)
                nx.draw(G_gt, pos, with_labels=True, node_color=node_colors_gt, edge_color='gray')
                plt.title("Ground Truth Graph")

                plt.tight_layout()

                # 저장할 경로 확인 및 폴더 생성
                directory = "./images"  # 변경: 저장 경로를 /mnt/data/ 아래로 지정
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # 이미지 파일로 저장
                save_path = os.path.join(directory, "graph_comparison_" + file + ".png")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')

            if save:
                path = f'../preprocessing/global_mapper/datasets/new_city_datasets/gt_train_datasets/test/{file_idx}.gpickle'
                graph = nx.read_gpickle(path)

                # 그래프의 모든 엣지를 제거
                graph.remove_edges_from(list(graph.edges()))

                # 사용자의 인접 행렬을 바탕으로 엣지 추가
                for i, row in enumerate(pred_adj_matrix):
                    for j, val in enumerate(row):
                        if val == 1:  # i와 j 사이에 연결이 있는 경우
                            graph.add_edge(i, j)

                for node in range(n_boundary + n_building):
                    if node < n_boundary:
                        graph.add_node(node, building_masks=[0], node_features=[0, 0, 0, 0, 0])
                    else:
                        graph.add_node(node, building_masks=[1], node_features=[0, 0, 0, 0, 0])
                # 결과 그래프 검증 또는 사용
                # 예: 그래프를 다시 gpickle 파일로 저장
                path = path.replace('gt_train_datasets', f'{graph_type}_{size_type}_train_datasets')

                # 폴더가 존재하는지 확인하고, 없으면 생성
                directory = os.path.dirname(path)
                if not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)

                nx.write_gpickle(graph, path)

            count += 1
            if count % 1001 == 0:
                break