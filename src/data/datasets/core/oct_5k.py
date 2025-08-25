from PIL import Image
import torchvision.transforms as transforms
import json  # Import the json library
import pandas as pd
from torch.utils.data import Dataset
import os


class OCT5KDataset(Dataset):
    def __init__(self, images_path, csv_path, use_grouped_biomarkers=True):
        self.images_path = images_path
        self.df = pd.read_csv(csv_path)
        self.use_grouped_biomarkers = use_grouped_biomarkers
        self.augmentations = transforms.Compose(
            [
                # 1. Flip the image horizontally. This is the most effective augmentation.
                transforms.RandomHorizontalFlip(p=0.5),
                # 2. Apply a set of random geometric transformations.
                transforms.RandomAffine(
                    degrees=5,  # Rotate by +/- 10 degrees
                    translate=(
                        0.05,
                        0.05,
                    ),  # Shift horizontally and vertically by up to 5%
                    scale=(0.95, 1.05),  # Zoom in/out by up to 5%
                ),
                # 3. Adjust brightness and contrast to simulate different scanner settings.
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_filename = self.df.iloc[idx]["image"]
        if self.use_grouped_biomarkers:
            biomarkers = self.df.iloc[idx]["grouped_biomarkers"]
        else:
            biomarkers = self.df.iloc[idx]["biomarkers"]

        # Convert the string representation of list to actual list
        biomarkers = eval(biomarkers)

        img_path = os.path.join(self.images_path, img_filename)

        img = Image.open(img_path).convert("RGB")

        # Apply augmentations
        img = self.augmentations(img)

        # Create the text field as a JSON string
        target = {
            "diagnosis": "Wet Age-related Macular Degeneration (CNV)",
            "biomarkers": biomarkers,
        }
        text_output = json.dumps(target)

        return {"image": img, "label": "CNV", "text": text_output}
