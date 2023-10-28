import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

"""
available mask type 
1. boundarymask
2. insidemask
3. streetmask
4. allmask
5. tfoutput_seqmask
6. tfoutput_plotmask
7. inbuildingcpmask
8. inedgemask
9. groundtruthmask 
"""

masktype = "inbuildingcpmask"
# inmasktype = "inedgemask"

city_name = "atlanta"
dataset_idx = 1
indata_idx = 3
# 마스크 저장

if masktype == "inbuildingcpmask" or masktype == "inedgemask" or masktype == "groundtruthmask":
    maskfolderpath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '3_mask', f'{city_name}',
                                  masktype, f'{city_name}_{dataset_idx}')
    mask_filename = os.path.join(maskfolderpath, f'{city_name}_{dataset_idx}_{indata_idx}.png')
else:
    maskfolderpath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '3_mask', f'{city_name}', masktype)
    mask_filename = os.path.join(maskfolderpath, f'{city_name}_{dataset_idx}.png')

image = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)

# 이미지를 출력하고 각 픽셀 위치에 해당 픽셀의 값을 표시
plt.figure(figsize=(12, 12))
plt.imshow(image, cmap='gray')
for i in range(120):
    for j in range(120):
        plt.text(j, i, str(image[i, j]), ha='center', va='center', color='red', fontsize=6)

plt.axis('off')
plt.show()