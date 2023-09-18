import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from data_refine import load_mask
import paths
import random
import argparse


class BuildingDataset(Dataset):
    def __init__(self, boundary_masks, inside_masks, centroid_masks):
        self.boundary_masks = boundary_masks
        self.inside_masks = inside_masks
        self.centroid_masks = centroid_masks

    def __len__(self):
        return len(self.boundary_masks)

    def __getitem__(self, idx):
        boundary = self.boundary_masks[idx].unsqueeze(0)
        inside = self.inside_masks[idx].unsqueeze(0)
        centroid = self.centroid_masks[idx].unsqueeze(0)

        x = torch.cat([boundary, inside], dim=0)
        y = centroid

        return x, y

def k_fold_split(dataset, n_splits):
    fold_length=len(dataset)//n_splits
    indices=list(range(len(dataset)))
    return [Subset(dataset, indices[i*fold_length: (i+1)*fold_length]) for i in range(n_splits)]
def get_datasets_and_loaders(args,  n_splits=5):
    boundarymask_hdf5 = paths.hdf5_boundarymask
    insidemask_hdf5 = paths.hdf5_insidemask
    centroidmask_hdf5 = paths.hdf5_centroidmask

    # boundary_masks = load_mask(boundarymask)
    # inside_masks = load_mask(insidemask)
    # centroid_masks = load_mask(centroidmask)
    boundary_masks = [torch.tensor(mask) for mask in load_mask(boundarymask_hdf5)]
    inside_masks = [torch.tensor(mask) for mask in load_mask(insidemask_hdf5)]
    centroid_masks = [torch.tensor(mask) for mask in load_mask(centroidmask_hdf5)]

    all_dataset = BuildingDataset(boundary_masks, inside_masks, centroid_masks)

    indices = list(range(len(all_dataset)))
    random.shuffle(indices) #random shuffle entire dataset for better generalization

    folds = k_fold_split(all_dataset, n_splits) # K-fold cross validation
    loaders=[]

    for i in range(n_splits):
        train_subsets = [folds[j] for j in range(n_splits) if j != i]
        train_dataset = torch.utils.data.ConcatDataset(train_subsets)
        val_dataset = folds[i]

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False
        )

        loaders.append((train_loader, val_loader))

    return loaders

