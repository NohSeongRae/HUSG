import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from data_refine import load_mask
import paths
import random
import argparse


class BuildingDataset(Dataset):
    def __init__(self, boundary_masks, inside_masks, centroid_masks=None, mode='train'):
        self.boundary_masks = boundary_masks
        self.inside_masks = inside_masks
        self.centroid_masks = centroid_masks
        self.mode = mode

        assert mode in ['train', 'inference'], "Mode should be either 'train' or 'inference'"

        if mode == 'train':
            assert self.centroid_masks is not None, "For training mode, centroid_masks should be provided."

    def __len__(self):
        return len(self.boundary_masks)

    def __getitem__(self, idx):
        boundary = self.boundary_masks[idx].unsqueeze(0)
        inside = self.inside_masks[idx].unsqueeze(0)

        x = torch.cat([boundary, inside], dim=0)

        if self.mode == 'train':
            centroid = self.centroid_masks[idx].unsqueeze(0)
            y = centroid
            return x, y
        else:
            return x


def k_fold_split(dataset, n_splits):
    fold_length = len(dataset) // n_splits
    indices = list(range(len(dataset)))
    return [Subset(dataset, indices[i * fold_length: (i + 1) * fold_length]) for i in range(n_splits)]


def get_datasets_and_loaders(args, n_splits=5):
    boundarymask_hdf5 = paths.boundarymask_USA
    insidemask_hdf5 = paths.insidemask_USA
    centroidmask_hdf5 = paths.centroidmask_USA
    if args.train_sample:
        boundarymask_hdf5 = paths.boundarymask_sample
        insidemask_hdf5 = paths.insidemask_sample
        centroidmask_hdf5 = paths.centroidmask_sample

    if args.use_Kfold:
        # boundary_masks = load_mask(boundarymask)
        # inside_masks = load_mask(insidemask)
        # centroid_masks = load_mask(centroidmask)
        boundary_masks = [torch.tensor(mask) for mask in load_mask(boundarymask_hdf5)]
        inside_masks = [torch.tensor(mask) for mask in load_mask(insidemask_hdf5)]
        centroid_masks = [torch.tensor(mask) for mask in load_mask(centroidmask_hdf5)]

        all_dataset = BuildingDataset(boundary_masks, inside_masks, centroid_masks)

        if args.use_total_shuffle:
            indices = list(range(len(all_dataset)))
            random.shuffle(indices)  # random shuffle entire dataset for better generalization

        folds = k_fold_split(all_dataset, n_splits)  # K-fold cross validation
        loaders = []

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
    else:
        boundary_masks = [torch.tensor(mask) for mask in load_mask(boundarymask_hdf5)]
        inside_masks = [torch.tensor(mask) for mask in load_mask(insidemask_hdf5)]
        centroid_masks = [torch.tensor(mask) for mask in load_mask(centroidmask_hdf5)]

        dataset_size=len(boundary_masks)
        train_size = int(dataset_size * 0.9)
        val_size = (dataset_size - train_size) // 2
        test_size = val_size
        train_dataset = BuildingDataset(
            boundary_masks=boundary_masks[:train_size],
            inside_masks=inside_masks[:train_size],
            centroid_masks=centroid_masks[:train_size]
        )
        val_dataset = BuildingDataset(
            boundary_masks=boundary_masks[train_size:train_size + val_size],
            inside_masks=inside_masks[train_size:train_size + val_size],
            centroid_masks=centroid_masks[train_size:train_size + val_size]
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False
        )
        return train_loader, val_loader

from torchvision.utils import save_image

def save_masks(masks, prefix):
    for idx, mask in enumerate(masks):
        save_image(mask.cpu(), f"./output/gt/{prefix}_mask_{idx}.png")

def get_inference_loader(args):
    boundarymask_test_path = paths.boundarymask_test
    insidemask_test_path = paths.insidemask_test

    boundary_masks_test = [torch.tensor(mask) for mask in load_mask(boundarymask_test_path)]
    inside_masks_test = [torch.tensor(mask) for mask in load_mask(insidemask_test_path)]

    save_masks(boundary_masks_test, "boundary")
    save_masks(inside_masks_test, "inside")

    test_dataset = BuildingDataset(boundary_masks_test, inside_masks_test, mode="inference")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )

    return test_loader