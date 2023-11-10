import networkx as nx

height = 4
width = 30
g = nx.grid_2d_graph(height, width)

posx = [2, 3, 1, 0]
posy = [25, 2, 29, 17]

def normalize(values):
    min_val = min(values)
    max_val = max(values)
    return [(v - min_val) / (max_val - min_val) for v in values]

# 정규화
norm_posx = normalize(posx)
norm_posy = normalize(posy)

# 4x30 그리드 좌표로 변환
grid_posx = [round(x * 3) for x in norm_posx]  # 0-3 범위로
grid_posy = [round(y * 29) for y in norm_posy]  # 0-29 범위로

# 4x30 그리드 좌표 출력
for x, y in zip(grid_posx, grid_posy):
    print((x, y))
