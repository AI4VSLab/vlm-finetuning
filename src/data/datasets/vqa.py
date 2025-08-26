import os
import pandas as pd
from PIL import Image
import torch
import random
from torch.utils.data import Dataset
import ast


class OCTVQADataset(Dataset):
    def __init__(self, images_dir, csv_path, transform=None):
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
        all_biomarkers = {
                    "Drusen", "Photoreceptor Degeneration", "Pigment Epithelial Detachment",
                    "Geographic Atrophy", "Choroidal Fold", "Epiretinal Membrane",
                    "Hyperfluorescent Spots", "Intraretinal Fluid", "Posterior Vitreous Detachment",
                    "Fluid", "Subretinal Fluid"
                }
        self.instruction_template = "[VQA] Is {} present in this OCT scan?"

        # Create a list of all possible biomarker questions and their corresponding image index
        self.samples = []
        for index, row in self.df.iterrows():
            image_filename = row["image_filename"]
            present_biomarkers = ast.literal_eval(row["biomarkers"])
            label = row["label"]

            # Add all "Yes" samples
            present_biomarkers = [b for b in present_biomarkers if b in all_biomarkers]
            for biomarker in present_biomarkers:
                    instruction = self.instruction_template.format(biomarker)
                    self.samples.append({
                        "image_filename": image_filename, "instruction": instruction,
                        "response": "Yes", "label": label,
                    })

            # Add an equal number of "No" samples
            absent_biomarkers = [b for b in all_biomarkers if b not in present_biomarkers]
            num_no_samples = len(present_biomarkers) # Match the number of "Yes" samples

            # If there are no present biomarkers, add a few "No" samples to still learn from the image
            if num_no_samples == 0:
                num_no_samples = 2

            no_samples = random.sample(absent_biomarkers, min(num_no_samples, len(absent_biomarkers)))

            for biomarker in no_samples:
                instruction = self.instruction_template.format(biomarker)
                self.samples.append({
                    "image_filename": image_filename, "instruction": instruction,
                    "response": "No", "label": label,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_info = self.samples[idx]
        img_filename = sample_info["image_filename"]
        img_path = os.path.join(self.images_dir, img_filename)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        sample = {
            "image": image,
            "label": sample_info["label"],
            "instruction": sample_info["instruction"],
            "response": sample_info["response"],
        }

        return sample
