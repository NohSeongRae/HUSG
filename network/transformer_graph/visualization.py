import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

def get_all_one_rows(matrix):
    # 모든 값이 1인 행을 찾아 해당 행의 인덱스 리스트를 반환
    return [i for i, row in enumerate(matrix) if np.sum(row) >= len(row)]

def plot(pred, gt, idx):
    one_indices = get_all_one_rows(gt)
    i1 = one_indices[0] + 1
    i2 = one_indices[1]

    # 그래프 객체 생성
    G_pred = nx.from_numpy_matrix(pred[i1:i2][i1:i2])
    G_gt = nx.from_numpy_matrix(gt[i1:i2][i1:i2])

    # 그래프 시각화
    plt.figure(figsize=(12, 5))

    # Prediction Graph
    plt.subplot(1, 2, 1)
    pos = nx.spring_layout(G_pred)
    nx.draw(G_pred, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.title("Prediction Graph")

    # Ground Truth Graph
    plt.subplot(1, 2, 2)
    pos = nx.spring_layout(G_gt)
    nx.draw(G_gt, pos, with_labels=True, node_color='lightgreen', edge_color='gray')
    plt.title("Ground Truth Graph")

    plt.tight_layout()

    # 저장할 경로 확인 및 폴더 생성
    directory = "/mnt/data/images"  # 변경: 저장 경로를 /mnt/data/ 아래로 지정
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 이미지 파일로 저장
    save_path = os.path.join(directory, "graph_comparison_" + str(idx) + ".png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')