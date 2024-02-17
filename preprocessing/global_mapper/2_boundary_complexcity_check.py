from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt


def plot(boundary, buildings):
    x, y = boundary.exterior.xy  # 외곽선의 x, y 좌표
    ax.plot(x, y)

    # 각 Polygon에 대해 반복
    for building in buildings:
        x, y = building.exterior.xy  # 외곽선의 x, y 좌표
        ax.plot(x, y)

    ax.set_title('MultiPolygon Plot')

for file_index in tqdm(range(10)):
    fig, ax = plt.subplots()
    file_path = str(file_index)
    input_file_path = f'raw_datasets/globalmapper_dataset/raw_geo/{file_path}'
    with open(input_file_path, 'rb') as file:
        data = pickle.load(file)
        boundary = data[0]
        buildings = data[1]

    plot(boundary,buildings)
    boundary = boundary.simplify(10, preserve_topology=True)
    plot(boundary,buildings)
    plt.show()