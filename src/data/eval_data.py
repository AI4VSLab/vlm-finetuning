# amd-vlm-finetuning/src/data/unified_evaluation_data.py
import logging
import os
from .datasets import (
    OCTDiagnosisDataset,
    OCTBiomarkersDataset,
    OCTVQADataset,
    OCTTestDataset,
)  # Assuming OCTTestDataset is for the generic test data

logger = logging.getLogger(__name__)


class OCTEvaluationData:
    def __init__(self, config):
        logger.info("Loading Evaluation Data for testing...")

        # Initialize individual datasets
        self.diagnosis_dataset = OCTDiagnosisDataset(
            root=config.diagnosis_dir, split="test"
        )
        logger.info(
            f"Diagnosis Test Dataset loaded with {len(self.diagnosis_dataset)} samples."
        )

        images_dir = config.biomarkers_images_dir
        test_csv_path = os.path.join(config.biomarkers_csv_path, "test.csv")

        self.biomarkers_dataset = OCTBiomarkersDataset(
            images_dir=images_dir, csv_path=test_csv_path
        )
        logger.info(
            f"Biomarker Test Dataset loaded with {len(self.biomarkers_dataset)} samples."
        )

        self.vqa_dataset = OCTVQADataset(images_dir=images_dir, csv_path=test_csv_path)
        logger.info(f"VQA Test Dataset loaded with {len(self.vqa_dataset)} samples.")

        # Generic Test Dataset (e.g., OCT_Test)
        self.test_dataset = OCTTestDataset(
            root=config.test_dir
        )  # Assuming a new generic test path in config
        logger.info(
            f"Generic Test Dataset loaded with {len(self.test_dataset)} samples."
        )

        logger.info("Evaluation Data Loaded.")
