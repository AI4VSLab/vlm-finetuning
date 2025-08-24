import json
from torch.utils.data import Dataset
from torchvision import datasets


class OCTAMDDataset(Dataset):
    def __init__(self, root):
        """
        Args:
            root (string): Root directory path of the dataset.
        """
        self.dataset = datasets.ImageFolder(root)
        self.classes = ["NORMAL", "CNV"]
        self.class_to_idx = {"NORMAL": 0, "CNV": 1}

        # Dictionary to map class names to descriptions
        self.class_descriptions = {
            "CNV": "Wet Age-related Macular Degeneration (CNV)",
            "NORMAL": "Healthy eyes with no abnormalities (NORMAL)",
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 1. Get the raw image and label from the internal ImageFolder dataset
        image, label_idx = self.dataset[idx]
        label_name = self.classes[label_idx]

        # 2. Create the structured text output
        diagnosis_description = self.class_descriptions.get(
            label_name, label_name
        )  # Get description, default to class name if not found
        target = {
            "diagnosis": diagnosis_description,
            "biomarkers": [],  # No biomarkers in this dataset
        }
        text_output = json.dumps(target)

        # 3. Format the data into the final dictionary
        sample = {
            "image": image,
            "label": label_name,
            "text": text_output,
        }

        return sample
