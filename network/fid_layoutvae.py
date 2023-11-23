import os
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
import PIL
from torchmetrics.image.fid import FrechetInceptionDistance

# base = 'Z:/iiixr-drive/Projects/2023_City_Team/0_others/vis/cvae_graph/Abilation'
# path = 'Abilation(T5 + GIN)'
# path = os.path.join(base, path)
path = 'Z:/iiixr-drive/Projects/2023_City_Team/0_others/vis/cvae_graph/Abilation/building graph + street graph'
list_output_all = os.listdir(path)

path2 = 'C:/Users/SeungWon Seo/Downloads/eu_blockplanner_image'
list_output_all2 = os.listdir(path2)

list_output = []
list_output_gt = []

for name in list_output_all:
    if 'ground_truth' in name and not 'pkl' in name:
        list_output_gt.append(name)
for name in list_output_all2:
    list_output.append(name)

assert len(list_output_gt) == 1000
assert len(list_output) == 1000

image_tensor = torch.zeros(1000, 3, 299, 299).type(torch.uint8)
image_gt_tensor = torch.zeros(1000, 3, 299, 299).type(torch.uint8)

for idx, (o, o_gt) in tqdm(enumerate(zip(list_output, list_output_gt)), total=len(list_output)):
    image_sample = PIL.Image.open(os.path.join(path2, o)).convert("RGB").resize((299, 299))
    image_sample_gt = PIL.Image.open(os.path.join(path, o_gt)).convert("RGB").resize((299, 299))

    # image_sample_tensor = (totensor(image_sample)*255).type(torch.uint8)
    # image_sample_gt_tensor = (totensor(image_sample)*255).type(torch.uint8)
    image_sample_tensor = torch.LongTensor(np.array(image_sample)).permute(2, 0, 1)
    image_sample_gt_tensor = torch.LongTensor(np.array(image_sample_gt)).permute(2, 0, 1)

    image_tensor[idx] = image_sample_tensor
    image_gt_tensor[idx] = image_sample_gt_tensor

fid = FrechetInceptionDistance(feature=2048)
fid.update(image_gt_tensor, real=True)
fid.update(image_tensor, real=False)
score = fid.compute()

print(score)
