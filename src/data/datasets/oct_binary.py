import json
from torch.utils.data import Dataset, ConcatDataset, Subset
from torchvision import datasets, transforms
import os

# Import the original datasets to use them without modification
from .oct_c8 import OCTC8Dataset
from .oct_2017 import OCT2017Dataset


class OCTBinaryDataset(Dataset):
    def __init__(self, oct2017_root, octc8_root, split):
        """
        Args:
            oct2017_root (string): Root directory path of the OCT 2017 dataset.
            octc8_root (string): Root directory path of the OCT-C8 dataset.
            split (string): Specifies the dataset split ('train', 'val', 'test').
        """
        self.target_classes = ["CNV", "NORMAL"]

        # Initialize original datasets
        oct2017_full_dataset = OCT2017Dataset(oct2017_root, split)
        octc8_full_dataset = OCTC8Dataset(octc8_root, split)

        # Filter OCT2017Dataset for CNV and NORMAL using Subset
        oct2017_indices = [
            i
            for i, (path, label_idx) in enumerate(oct2017_full_dataset.dataset.samples)
            if oct2017_full_dataset.classes[label_idx] in self.target_classes
        ]
        oct2017_filtered = Subset(oct2017_full_dataset, oct2017_indices)

        # Filter OCTC8Dataset for CNV and NORMAL using Subset
        octc8_indices = [
            i
            for i, (path, label_idx) in enumerate(octc8_full_dataset.dataset.samples)
            if octc8_full_dataset.classes[label_idx] in self.target_classes
        ]
        octc8_filtered = Subset(octc8_full_dataset, octc8_indices)

        # Concatenate the filtered datasets
        self.dataset = ConcatDataset([oct2017_filtered, octc8_filtered])

        # Define the classes and class_to_idx for this combined dataset
        self.classes = self.target_classes
        self.class_to_idx = {cls: i for i, cls in enumerate(self.target_classes)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
