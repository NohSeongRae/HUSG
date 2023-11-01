import numpy as np
import matplotlib.pyplot as plt
import os

def plot(pred, gt, file_idx, mask):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # ax1: 예측 결과 시각화
    for idx in range(len(pred)):
        if mask[idx] == 0:
            continue

        x = [pred[idx][0], pred[idx][2]]
        y = [pred[idx][1], pred[idx][3]]
        ax1.plot(x, y, 'black')

    ax1.set_title('Prediction')
    ax1.grid(True)
    ax1.set_xlim([-0.1, 1.1])  # x축 범위 설정
    ax1.set_ylim([-0.1, 1.1])  # y축 범위 설정

    # ax2: Ground Truth 시각화
    for idx in range(len(pred)):
        if mask[idx] == 0:
            continue

        x = [gt[idx][0], gt[idx][2]]
        y = [gt[idx][1], gt[idx][3]]
        ax2.plot(x, y, 'black')
    ax2.set_title('Ground Truth')
    ax2.grid(True)
    ax2.set_xlim([-0.1, 1.1])  # x축 범위 설정
    ax2.set_ylim([-0.1, 1.1])  # y축 범위 설정

    plt.tight_layout()

    # 저장할 경로 확인 및 폴더 생성
    directory = "./images"  # 변경: 저장 경로를 /mnt/data/ 아래로 지정
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 이미지 파일로 저장
    save_path = os.path.join(directory, "boundary_" + str(file_idx) + ".png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')