import os
import re
import pickle
from PIL import Image

import os
import shutil
import pickle

# ["allmask", "boundarymask", "insidemask", "streetmask", "tfoutput_plotmask", "tfoutput_seqmask"]

def remove_extension(filename):
    """파일 확장자를 제거하는 함수"""
    return os.path.splitext(filename)[0]


city_name = "atlanta"
root_path = os.path.join("Z:", "iiixr-drive", "Projects", "2023_City_Team", "3_mask", f"{city_name}")
boundarymask_path = os.path.join(root_path, "boundarymask")
allmask_path = os.path.join(root_path, "allmask")
insidemask_path = os.path.join(root_path, "insidemask")
streetmask_path = os.path.join(root_path, "streetmask")
tfoutput_plotmask_path = os.path.join(root_path, "tfoutput_plotmask")
tfoutput_seqmask_path = os.path.join(root_path, "tfoutput_seqmask")

inbuildingcpmask_path = os.path.join(root_path, "inbuildingcpmask")
inedgemask_path = os.path.join(root_path, "inedgemask")
groundtruthmask_path = os.path.join(root_path, "groundtruthmask")

# 폴더 경로 설정
file_folders = [
    boundarymask_path,
    allmask_path,
    insidemask_path,
    streetmask_path,
    tfoutput_plotmask_path,
    tfoutput_seqmask_path
]

dir_folders = [
    inbuildingcpmask_path,
    inedgemask_path,
    groundtruthmask_path
]

all_folders = file_folders + dir_folders

# 모든 폴더에서 파일 및 폴더 이름의 집합을 만듭니다.
folder_sets = [set(os.listdir(folder)) for folder in all_folders]

all_folders = file_folders + dir_folders

# 모든 폴더에서 파일 및 폴더 이름의 집합을 만듭니다.
folder_sets = [set(map(remove_extension, os.listdir(folder))) for folder in file_folders]
folder_sets += [set(os.listdir(folder)) for folder in dir_folders]

# 모든 폴더의 파일 및 폴더 이름의 합집합을 만듭니다.
all_names = set.union(*folder_sets)

# 각 이름에 대해서
for name in all_names:
    # 해당 이름이 모든 폴더에 존재하는지 확인합니다.
    if sum(name in folder_set for folder_set in folder_sets) != len(all_folders):
        print(name)
        # 이름이 모든 폴더에 존재하지 않는 경우, 해당 이름으로 된 파일 또는 폴더를 삭제합니다.
        for folder in all_folders:
            path = os.path.join(folder, name)
            if os.path.exists(path):
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)


def get_file_counts_in_order(base_path):
    all_folders = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    sorted_folders = sorted(all_folders, key=lambda x: int(x.split('_')[-1]))

    file_counts = []
    for folder in sorted_folders:
        folder_path = os.path.join(base_path, folder)
        count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
        file_counts.append(count)

    return file_counts

# file_counts_pickle_path = os.path.join("Z:", "iiixr-drive", "Projects", "2023_City_Team", "3_mask", 'train_dataset', f"{city_name}")

# file_counts_list = get_file_counts_in_order(root_path)
# with open(file_counts_pickle_path, 'wb') as f:
#     pickle.dump(file_counts_list, f)

# # 폴더 수 세기
# import os
#
# folder_path = "YOUR_FOLDER_PATH_HERE"  # 여기에 폴더 경로를 입력하세요.
# file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
#
# print(f"Number of files in {folder_path}: {file_count}")
