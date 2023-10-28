import os
import re
import pickle
from PIL import Image


import os
import shutil

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


# def extract_number(filename):
#     match = re.search(r'(\d+).png$', filename)
#     return int(match.group(1)) if match else 0
#
# def save_mask_pkl(city_name, masktype):
#     folder_path = os.path.join("Z:", "iiixr-drive", "Projects", "2023_City_Team", "3_mask", f"{city_name}",
#                                         f"{masktype}")
#     all_files = os.listdir(folder_path)
#     sorted_files = sorted(all_files, key=extract_number)
#
#     images = [Image.open(os.path.join(folder_path, file)) for file in sorted_files]
#
#     save_folder_path = os.path.join("Z:", "iiixr-drive", "Projects", "2023_City_Team", "3_mask", 'train_dataset', f"{city_name}")
#
#     if not os.path.exists(save_folder_path):
#         os.makedirs(save_folder_path)
#
#     save_path = os.path.join(save_folder_path, f"{masktype}.pkl")
#
#     with open(save_path, "wb") as f:
#         pickle.dump(images, f)
#
# # single_mask_type_list = ["allmask", "boundarymask", "insidemask", "streetmask", "tfoutput_plotmask", "tfoutput_seqmask"]
# single_mask_type_list = ["allmask", "streetmask", "tfoutput_plotmask", "tfoutput_seqmask"]
#
# city_names = ["atlanta"]
#
# for city_name in city_names:
#     for mask_type in single_mask_type_list:
#         save_mask_pkl(city_name, mask_type)
