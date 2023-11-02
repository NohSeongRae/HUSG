import os
import pickle
import matplotlib.pyplot as plt


city_name = "atlanta"

folder_path = os.path.join('/home', 'rhosunr99', 'HUSG', 'preprocessing', '6_64_polygon', 'train_dataset',
                           f'{city_name}')

base_dir = os.path.join("Z:", "iiixr-drive", "Projects", "2023_City_Team", "6_64_polygon", "atlanta")
# atlanta_polygon = os.path.join(base_dir, "64_point_polygon_exteriors.pkl")

atlanta_polygon = os.path.join(folder_path, "64_point_polygon_exteriors.pkl")

with open(atlanta_polygon, 'rb') as file:
    data = pickle.load(file)

data_ = data[5]

print(len(data))

for i in range(len(data)):
    print(i)
    data_ = data[i]
    for d_ in data_:
        # coords = data_[0]

        # x와 y 좌표 추출
        x = d_[:, 0]
        y = d_[:, 1]

        # 그래프 그리기
        plt.scatter(x, y)  # 좌표에 점을 찍습니다.
        plt.plot(x, y)     # 좌표를 연결하는 선을 그립니다.

    plt.show()
