from torch.utils.data import Dataset
from torchvision import datasets
import os


class OCTDiagnosisDataset(Dataset):
    def __init__(self, root, split, transform=None):
        """
        Args:
            root (string): Root directory path of the dataset.
            split (string): Specifies the dataset split ('train', 'val', 'test').
        """
        self.split = split
        self.dataset = datasets.ImageFolder(os.path.join(root, split))
        self.classes = self.dataset.classes
        self.transform = transform
        self.class_to_idx = self.dataset.class_to_idx
        self.instruction = "What is the diagnosis for this OCT scan, Normal or wet-AMD?"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 1. Get the raw image and label from the internal ImageFolder dataset
        image, label_idx = self.dataset[idx]
        label = self.classes[label_idx]

        # 2. Apply augmentations
        if self.transform:
            image = self.transform(image)

        # 4. Format the data into the final dictionary
        sample = {
            "image": image,
            "label": label,
            "instruction": self.instruction,
            "response": label,
        }

        return sample
