# amd-vlm-finetuning/src/data/unified_data.py
import logging
import os
import torch
import numpy as np
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

        # Initialize individual datasets
        self.diagnosis_dataset = OCTDiagnosisDataset(
            root=config.diagnosis_dir, split="train"
        )

        images_dir = config.biomarkers_images_dir
        train_csv_path = os.path.join(config.biomarkers_csv_path, "train.csv")
        self.biomarkers_dataset = OCTBiomarkersDataset(
            images_dir=images_dir, csv_path=train_csv_path
        )
        self.vqa_dataset = OCTVQADataset(images_dir=images_dir, csv_path=train_csv_path)

        # Combine datasets for training
        self.train_data = ConcatDataset(
            [self.diagnosis_dataset, self.biomarkers_dataset, self.vqa_dataset]
        )
        logger.info(
            f"Unified training dataset loaded with {len(self.train_data)} samples."
        )
        logger.info(f"  - Diagnosis samples (train): {len(self.diagnosis_dataset)}")
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

        self.val_data = ConcatDataset(
            [val_diagnosis_dataset, val_biomarkers_dataset, val_vqa_dataset]
        )
        logger.info(
            f"Unified validation dataset loaded with {len(self.val_data)} samples."
        )
        logger.info(f"  - Diagnosis samples (val): {len(val_diagnosis_dataset)}")
        logger.info(f"  - Biomarker samples (val): {len(val_biomarkers_dataset)}")
        logger.info(f"  - VQA samples (val): {len(val_vqa_dataset)}")

        # For evaluation, we can use a held-out test set. Here we use the diagnosis test set as an example.
        # You might want to create a more comprehensive test set later.
        self.test_data = OCTTestDataset(root=config.test_dir)
        logger.info(f"Test dataset loaded with {len(self.test_data)} samples.")
        logger.info("Unified data loading complete.")

    def get_sampler_weights(self):
        """
        Calculates and returns weights for each sample in the training dataset
        for balanced sampling.

        Args:
            ratios (dict, optional): A dictionary specifying the desired sampling
                                        ratio for each dataset.
                                        Keys should be 'diagnosis', 'biomarkers', 'vqa'.
                                        If None, defaults to 1/N weighting for a 33/33/33 split.
        """
        ratios = {
            'diagnosis': 0.25,
            'biomarkers': 0.50,
            'vqa': 0.25
        }

        weights = []

        if ratios:
            # Ensure the ratios sum to 1 for clarity
            if sum(ratios.values()) != 1.0:
                logger.warning(f"Custom ratios {ratios} do not sum to 1. Normalizing internally.")

            logger.info(f"Using custom sampling ratios: {ratios}")
            # The weight for each sample is target_ratio / num_samples
            diag_weight = ratios['diagnosis'] / len(self.diagnosis_dataset)
            biom_weight = ratios['biomarkers'] / len(self.biomarkers_dataset)
            vqa_weight = ratios['vqa'] / len(self.vqa_dataset)
        else:
            logger.info("Using default 1/N balanced sampling for a 33/33/33 split.")
            # Default behavior: 1 / num_samples
            diag_weight = 1.0 / len(self.diagnosis_dataset)
            biom_weight = 1.0 / len(self.biomarkers_dataset)
            vqa_weight = 1.0 / len(self.vqa_dataset)

        # Assign weight to each sample
        weights.extend([diag_weight] * len(self.diagnosis_dataset))
        weights.extend([biom_weight] * len(self.biomarkers_dataset))
        weights.extend([vqa_weight] * len(self.vqa_dataset))

        return torch.DoubleTensor(weights)
