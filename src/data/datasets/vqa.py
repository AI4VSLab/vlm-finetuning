import os
import pandas as pd
from PIL import Image
import torch
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
        all_biomarkers = [
            "Pigment Epithelial Detachment",
            "Drusen",
            "Intraretinal Fluid",
            "Subretinal Fluid",
            "Choroidal Neovascularization",
            "Posterior Vitreous Detachment",
            "Geographic Atrophy",
            "Epiretinal Membrane",
            "Photoreceptor Layer Disruption",
            "Hemorrhage",
            "Retinal Pigment Epithelium Migration",
            "Retinal Pigment Epithelium Atrophy",
            "Disciform Scar",
            "Outer Retinal Tubulation",
            "Subretinal Hyperreﬂective Material",
            "Retinal Pigment Epithelium Tear",
            "Choroidal Fold",
            "Fluid",
        ]
        self.instruction_template = "Is {} present in this OCT scan?"

        # Create a list of all possible biomarker questions and their corresponding image index
        self.samples = []
        for index, row in self.df.iterrows():
            image_filename = row["image_filename"]
            biomarkers = ast.literal_eval(row["biomarkers"])
            label = row["label"]
            for biomarker in all_biomarkers:
                instruction = self.instruction_template.format(biomarker)
                response = "Yes" if biomarker in biomarkers else "No"
                self.samples.append(
                    {
                        "image_filename": image_filename,
                        "instruction": instruction,
                        "response": response,
                        "label": label,
                    }
                )

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
