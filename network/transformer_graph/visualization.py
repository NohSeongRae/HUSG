import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

def get_all_one_rows(matrix):
    return [i for i, row in enumerate(matrix) if np.sum(row) >= len(row)]

def plot(pred, gt, idx, cur_n_street):
    one_indices = get_all_one_rows(gt)
    for idx in range(len(pred)):
        print(idx)
        print(pred[idx])
        print(gt[idx])
    i1 = one_indices[0] + 1
    i2 = one_indices[1]

    # 그래프 객체 생성
    pred = pred[i1:i2, i1:i2]
    gt = gt[i1:i2, i1:i2]

    G_pred = nx.DiGraph(pred)
    G_gt = nx.DiGraph(gt)

    union_graph = nx.compose(G_pred, G_gt)  # create a union of both graphs to ensure all nodes are considered
    pos = nx.spring_layout(union_graph)

    # Assign colors based on the threshold
    node_colors_pred = ['lightblue' if i < cur_n_street else 'lightgreen' for i in G_pred.nodes()]
    node_colors_gt = ['lightblue' if i < cur_n_street else 'lightgreen' for i in G_gt.nodes()]

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
    save_path = os.path.join(directory, "graph_comparison_" + str(idx) + ".png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')