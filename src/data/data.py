# amd-vlm-finetuning/src/data/unified_data.py
import logging
import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import ConcatDataset, Subset
from .datasets import (
    OCTDiagnosisDataset,
    OCTBiomarkersDataset,
    OCTVQADataset,
    OCTTestDataset,
)

logger = logging.getLogger(__name__)


class OCTUnifiedData:
    def __init__(self, config):
        logger.info("Loading Unified Datasets...")

        self.stage = config.stage

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

        # Initialize individual datasets
        self.diagnosis_dataset = OCTDiagnosisDataset(
            root=config.diagnosis_dir, split="train", transform=self.augmentations
        )

        images_dir = config.biomarkers_images_dir
        train_csv_path = os.path.join(config.biomarkers_csv_path, "train.csv")
        self.biomarkers_dataset = OCTBiomarkersDataset(
            images_dir=images_dir, csv_path=train_csv_path, transform=self.augmentations
        )
        self.vqa_dataset = OCTVQADataset(images_dir=images_dir, csv_path=train_csv_path, transform=self.augmentations)

        if self.stage == "foundation":
            self.train_data = ConcatDataset([self.diagnosis_dataset, self.vqa_dataset])
        else:
            # Combine datasets for training
            self.train_data = ConcatDataset(
                [self.diagnosis_dataset, self.biomarkers_dataset, self.vqa_dataset]
            )
        logger.info(
            f"Unified training dataset loaded with {len(self.train_data)} samples."
        )
        logger.info(f"  - Diagnosis samples (train): {len(self.diagnosis_dataset)}")
        if self.stage == "unified":
            logger.info(f"  - Biomarker samples (train): {len(self.biomarkers_dataset)}")
        logger.info(f"  - VQA samples (train): {len(self.vqa_dataset)}")

        # Validation Dataset
        val_diagnosis_dataset = OCTDiagnosisDataset(
            root=config.diagnosis_dir, split="val"
        )

        images_dir = config.biomarkers_images_dir
        val_csv_path = os.path.join(config.biomarkers_csv_path, "val.csv")
        val_biomarkers_dataset = OCTBiomarkersDataset(
            images_dir=images_dir, csv_path=val_csv_path
        )
        val_vqa_dataset = OCTVQADataset(images_dir=images_dir, csv_path=val_csv_path)

        target_size = len(val_biomarkers_dataset)
        logger.info(
            f"Target size for each validation task set to {target_size} samples."
        )

        # Subsample the larger datasets
        diag_indices = np.random.choice(
            len(val_diagnosis_dataset), target_size, replace=False
        )
        val_diagnosis_dataset = Subset(val_diagnosis_dataset, diag_indices)

        vqa_indices = np.random.choice(len(val_vqa_dataset), target_size, replace=False)
        val_vqa_dataset = Subset(val_vqa_dataset, vqa_indices)

        if self.stage == "foundation":
            self.val_data = ConcatDataset(
                [val_diagnosis_dataset, val_vqa_dataset]
            )
        else:
            self.val_data = ConcatDataset(
                [val_diagnosis_dataset, val_biomarkers_dataset, val_vqa_dataset]
            )
        logger.info(
            f"Unified validation dataset loaded with {len(self.val_data)} samples."
        )
        logger.info(f"  - Diagnosis samples (val): {len(val_diagnosis_dataset)}")
        logger.info(f"  - Biomarker samples (val): {len(val_biomarkers_dataset)}")
        logger.info(f"  - VQA samples (val): {len(val_vqa_dataset)}")

    def get_sampler_weights(self):
        weights = []

        if self.stage == "foundation":
            # For Stage 1, we can just balance between Diagnosis and VQA
            logger.info("Using foundation stage sampling ratios.")
            diag_weight = 0.5 / len(self.diagnosis_dataset)
            vqa_weight = 0.5 / len(self.vqa_dataset)
            weights.extend([diag_weight] * len(self.diagnosis_dataset))
            weights.extend([vqa_weight] * len(self.vqa_dataset))
        else: # Default "unified" stage
            # For Stage 2, use your tuned ratios for all three datasets
            logger.info("Using unified stage sampling ratios.")
            ratios = {
                'diagnosis': 0.35,
                'biomarkers': 0.40,
                'vqa': 0.25
            }
            diag_weight = ratios['diagnosis'] / len(self.diagnosis_dataset)
            biom_weight = ratios['biomarkers'] / len(self.biomarkers_dataset)
            vqa_weight = ratios['vqa'] / len(self.vqa_dataset)
            weights.extend([diag_weight] * len(self.diagnosis_dataset))
            weights.extend([biom_weight] * len(self.biomarkers_dataset))
            weights.extend([vqa_weight] * len(self.vqa_dataset))

        return torch.DoubleTensor(weights)
