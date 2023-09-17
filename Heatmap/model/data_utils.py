import torch
from torch.utils.data import DataLoader, Dataset
from data_refine import load_mask
import paths
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


def get_datasets_and_loaders(args, mode):
    # boundarymask = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', 'mask', 'boundarymask')
    # insidemask = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', 'mask', 'insidemask')
    # centroidmask = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', 'mask', 'centroidmask')
    boundarymask = paths.boundarymask
    insidemask = paths.insidemask
    centroidmask = paths.centroidmask
    # sample
    # boundarybuildingmask = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', 'mask', 'boundarybuildingmask_sample')
    # boundarymask = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', 'mask', 'boundarymask_sample')
    # buildingmask = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', 'mask', 'buildingmask_sample')
    # insidemask = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', 'mask', 'insidemask_sample')
    # inversemask = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', 'mask', 'inversemask_sample')
    # centroidmask = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', 'mask', 'centroidmask_sample')
    if mode == 'all':
        # boundarybuilding_masks = load_mask(boundarybuildingmask)
        boundary_masks = load_mask(boundarymask)
        # building_masks = load_mask(buildingmask)
        inside_masks = load_mask(insidemask)
        # inverse_masks = load_mask(inversemask)
        centroid_masks = load_mask(centroidmask)

        dataset_size = len(boundary_masks)
        train_size = int(dataset_size * 0.80)
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
    else:
        boundary_masks = load_mask(boundarymask)
        inside_masks = load_mask(insidemask)
        centroid_masks = load_mask(centroidmask)

        dataset_size = len(boundary_masks)
        train_size = int(dataset_size * 0.80)
        val_size = (dataset_size - train_size) // 2
        test_size = val_size

        test_dataset = BuildingDataset(
            boundary_masks=boundary_masks[train_size + val_size:],
            inside_masks=inside_masks[train_size + val_size:],
            centroid_masks=centroid_masks[train_size + val_size:]
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=args.num_workers,
            shuffle=False
        )
        return test_loader
