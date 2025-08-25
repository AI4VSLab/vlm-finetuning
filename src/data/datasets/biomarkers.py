import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class OCTBiomarkersDataset(Dataset):
    def __init__(self, images_dir, csv_path='/content/OCT-Biomarkers/sheets/combined.csv', transform = None):
        """
        Args:
            images_dir (string): Directory with all the images.
            csv_path (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images_dir = images_dir
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.instruction = "What biomarkers are visible in this OCT scan?"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_filename = self.df.iloc[idx]['image_filename']
        img_path = os.path.join(self.images_dir, img_filename)
        image = Image.open(img_path).convert('RGB')

        biomarkers_str = self.df.iloc[idx]['biomarkers']
        # Convert the string representation of the list back to a list
        biomarkers = eval(biomarkers_str)

        label = self.df.iloc[idx]['label'] # Get the label

        if self.transform:
            image = self.transform(image)

        sample = {
            "image": image,
            "label": label,
            "instruction": self.instruction,
            "response": ', '.join(biomarkers)
        }


        return sample
