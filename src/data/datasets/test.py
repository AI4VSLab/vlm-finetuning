from torch.utils.data import Dataset
from torchvision import datasets


class OCTTestDataset(Dataset):
    def __init__(self, root):
        """
        Args:
            root (string): Root directory path of the dataset.
            split (string): Specifies the dataset split ('train', 'val', 'test').
        """
        self.dataset = datasets.ImageFolder(root)
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        self.instruction = "[DIAGNOSIS] What is the diagnosis for this OCT scan, Normal or wet-AMD?"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the raw image and label from the internal ImageFolder dataset
        image, label_idx = self.dataset[idx]
        label = self.classes[label_idx]

        # Format the data into the final dictionary
        sample = {
            "image": image,
            "label": label,
            "instruction": self.instruction,
            "response": label,
        }

        return sample
