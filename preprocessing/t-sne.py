import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 데이터 생성 예시 (실제로는 고차원 데이터 사용)
# 여기서는 100개의 샘플과 50개의 특성으로 이루어진 데이터셋을 생성
data = np.load('C:/Users/Dobby/Downloads/latent_vectors.npy')

# t-SNE를 사용하여 2차원으로 변환 (perplexity와 learning rate 조정)
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
data_2d = tsne.fit_transform(data)

# 결과 시각화
plt.figure(figsize=(8, 6))
plt.scatter(data_2d[:, 0], data_2d[:, 1], c='blue', marker='o', edgecolor='k', s=50)
plt.title("2D visualization of the latent space using t-SNE (Perplexity 30, Learning rate 200)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()