# amd-vlm-finetuning/src/data/unified_data.py
import logging
import torch
from torch.utils.data import ConcatDataset
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
        self.biomarkers_dataset = OCTBiomarkersDataset(
            images_dir=config.biomarkers_images_dir, csv_path=config.biomarkers_train_csv_path
        )
        self.vqa_dataset = OCTVQADataset(
            images_dir=config.biomarkers_images_dir, csv_path=config.biomarkers_train_csv_path
        )

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
        val_biomarkers_dataset = OCTBiomarkersDataset(
            images_dir=config.biomarkers_images_dir, csv_path=config.biomarkers_val_csv_path
        )
        val_vqa_dataset = OCTVQADataset(
            images_dir=config.biomarkers_images_dir, csv_path=config.biomarkers_val_csv_path
        )

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
        """
        weights = []
        # Calculate weight for each dataset (1 / num_samples)
        diag_weight = 1.0 / len(self.diagnosis_dataset)
        biom_weight = 1.0 / len(self.biomarkers_dataset)
        vqa_weight = 1.0 / len(self.vqa_dataset)

        # Assign weight to each sample
        weights.extend([diag_weight] * len(self.diagnosis_dataset))
        weights.extend([biom_weight] * len(self.biomarkers_dataset))
        weights.extend([vqa_weight] * len(self.vqa_dataset))

        return torch.DoubleTensor(weights)
