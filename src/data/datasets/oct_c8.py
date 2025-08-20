import json
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os


class OCTC8Dataset(Dataset):
    def __init__(self, root, split):
        """
        Args:
            root (string): Root directory path of the dataset.
            split (string): Specifies the dataset split ('train', 'val', 'test').
        """
        self.split = split
        self.dataset = datasets.ImageFolder(os.path.join(root, split))
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

        # Dictionary to map class names to descriptions
        self.class_descriptions = {
            "AMD": "Wet Age-related Macular Degeneration (AMD)",
            "CSR": "Central Serous Retinopathy (CSR)",
            "DME": "Diabetic Macular Edema (DME)",
            "DR": "Diabetic Retinopathy (DR)",
            "DRUSEN": "Dry Age-related Macular Degeneration (DRUSEN)",
            "MH": "Macular Hole (MH)",
            "NORMAL": "Healthy eyes with no abnormalities (NORMAL)",
        }

        # Define augmentations for the training split
        if self.split == "train":
            self.augmentations = transforms.Compose(
                [
                    # 1. Flip the image horizontally.
                    transforms.RandomHorizontalFlip(p=0.5),
                    # 2. Apply random geometric transformations.
                    transforms.RandomAffine(
                        degrees=5,
                        translate=(0.05, 0.05),
                        scale=(0.95, 1.05),
                    ),
                    # 3. Adjust brightness and contrast.
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                ]
            )
        else:
            # For validation or test, just return the image without augmentation
            self.augmentations = transforms.Compose([])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 1. Get the raw image and label from the internal ImageFolder dataset
        image, label_idx = self.dataset[idx]
        label_name = self.classes[label_idx]
        label_name = "AMD" if label_name == "CNV" else label_name

        # 2. Apply augmentations
        image = self.augmentations(image)

        # 3. Create the structured text output
        # Since this dataset only has labels, we'll create a JSON structure
        # with an empty list for biomarkers to maintain consistency.
        # Format the diagnosis as "<description> (<class>)"
        diagnosis_description = self.class_descriptions.get(
            label_name, label_name
        )  # Get description, default to class name if not found
        target = {
            "diagnosis": diagnosis_description
        }
        text_output = json.dumps(target)

        # 4. Format the data into the final dictionary
        sample = {
            "image": image,
            "label": label_name,
            "text": text_output,
        }

        return sample
