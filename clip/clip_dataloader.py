from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import warnings

warnings.filterwarnings("ignore")

plt.ion()  # interactive mode

landmarks_frame = pd.read_csv('../data/faces/face_landmarks.csv')
n = 65
img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:]
landmarks = np.asarray(landmarks)
landmarks = landmarks.astype('float').reshape(-1, 2)

print(f"Image name: {img_name}")
print(f"Landmarks shape: {landmarks.shape}")
print(f"First 4 Landmarks: {landmarks[:4]}")


def show_landmarks(image, landmarks):
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)


plt.figure()
show_landmarks(io.imread(os.path.join('../data/faces/', img_name)), landmarks)
plt.show()


class FaceLandmarksDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.asarray(landmarks)
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

face_dataset=FaceLandmarksDataset(csv_file='../data/faces/face_landmarks.csv', root_dir='../data/faces/')
fig=plt.figure()
for i in range(len(face_dataset)):
    sample=face_dataset[i]
    print(i, sample['image'].shape, sample['landmarks'].shape)
    ax=plt.subplot(1,4,i+1)
    plt.tight_layout()
    ax.set_title(f"Sample #{i}")
    ax.axis('off')
    show_landmarks(**sample)
    if i==3:
        plt.show()
        break