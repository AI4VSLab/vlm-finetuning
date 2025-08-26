# amd-vlm-finetuning/src/data/unified_data.py
import logging
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
        diagnosis_dataset = OCTDiagnosisDataset(
            root=config.diagnosis_dir, split="train"
        )
        biomarkers_dataset = OCTBiomarkersDataset(
            images_dir=config.biomarkers_images_dir, csv_path=config.biomarkers_csv_path
        )
        vqa_dataset = OCTVQADataset(
            images_dir=config.biomarkers_images_dir, csv_path=config.biomarkers_csv_path
        )

        # Combine datasets for training
        self.train_data = ConcatDataset(
            [diagnosis_dataset, biomarkers_dataset, vqa_dataset]
        )
        logger.info(
            f"Unified training dataset loaded with {len(self.train_data)} samples."
        )
        logger.info(f"  - Diagnosis samples: {len(diagnosis_dataset)}")
        logger.info(f"  - Biomarker samples: {len(biomarkers_dataset)}")
        logger.info(f"  - VQA samples: {len(vqa_dataset)}")

        # For evaluation, we can use a held-out test set. Here we use the diagnosis test set as an example.
        # You might want to create a more comprehensive test set later.
        self.test_data = OCTTestDataset(root=config.test_dir)
        logger.info(f"Test dataset loaded with {len(self.test_data)} samples.")
        logger.info("Unified data loading complete.")
