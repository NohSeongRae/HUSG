import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import networkx as nx

# Load the graph from the file
graph_path = './47.gpickle'
G = nx.read_gpickle(graph_path)

# Define the custom colormap
graph_condition = G.graph.get('condition', None)

# 검사할 방향: 상하좌우 및 대각선
directions = [
    (-1, 0),  # 위
    (1, 0),  # 아래
    (0, -1),  # 왼쪽
    (0, 1),  # 오른쪽
    (-1, -1),  # 왼쪽 위 대각선
    (-1, 1),  # 오른쪽 위 대각선
    (1, -1),  # 왼쪽 아래 대각선
    (1, 1)  # 오른쪽 아래 대각선
]

rows = len(graph_condition)
cols = len(graph_condition[0])
# 2중 포문으로 각 셀 검사
for i in range(rows):
    for j in range(cols):
        is_able = False

        # 각 방향으로 트리거 검사
        for direction in directions:
            ni = i + direction[0]
            nj = j + direction[1]

            # 유효한 인덱스인지 확인
            if 0 <= ni < rows and 0 <= nj < cols:
                if graph_condition[ni][nj] == 0:
                    is_able = True

        if not is_able and 0 < i < rows - 1 and 0 < j < cols - 1:
            graph_condition[i][j] = 2

for i in range(rows):
    for j in range(cols):
        graph_condition[i][j] = 0

custom_cmap = ListedColormap(['black', 'white'])

# Plot the heatmap for the 'condition' attribute with specified colors
plt.figure(figsize=(10, 10))
sns.heatmap(graph_condition, cmap=custom_cmap, cbar=False, vmin=0, vmax=1, xticklabels=False, yticklabels=False)
plt.axis('off')  # Remove axes

# Adjust layout to fill the plot area completely
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
