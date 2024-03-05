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
path = 'grid_graph_figure'
path = 'output/synthetic_T4_dnpcs_ariel-k1/cvae_graph_20240305_145257'

list_output_all = os.listdir(path)

list_output = []
list_output_gt = []

for name in list_output_all:
    if 'ground_truth' in name and 'png' in name:
        list_output_gt.append(name)
    elif 'prediction' in name and 'png' in name:
        list_output.append(name)

list_output_gt = list_output_gt[:1000]
list_output = list_output[:1000]

assert len(list_output_gt) == 1000
assert len(list_output) == 1000

image_tensor = torch.zeros(1000, 3, 299, 299).type(torch.uint8)
image_gt_tensor = torch.zeros(1000, 3, 299, 299).type(torch.uint8)

for idx, (o, o_gt) in tqdm(enumerate(zip(list_output, list_output_gt)), total=len(list_output)):
    try:
        image_sample = PIL.Image.open(os.path.join(path, o)).convert("RGB").resize((299, 299))
        image_sample_gt = PIL.Image.open(os.path.join(path, o_gt)).convert("RGB").resize((299, 299))
    except:
        continue

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