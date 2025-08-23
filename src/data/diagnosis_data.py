import logging
from .datasets import OCTBinaryDataset

logger = logging.getLogger(__name__)


class OCTDiagnosisData:
    def __init__(self, config):
        logger.info("Loading Transcript Data... ")
        self.train_data = OCTBinaryDataset(config.oct2017_path, config.octc8_path, split="train")
        logger.info(f"Train dataset size: {len(self.train_data)}")
        self.test_data = OCTBinaryDataset(config.oct2017_path, config.octc8_path, split="test")
        logger.info(f"Test dataset size: {len(self.test_data)}")
        logger.info(f"Found classes: {self.test_data.classes}")
        logger.info(f"Class to index mapping: {self.test_data.class_to_idx}")
        logger.info("OCT Binary Data Loaded.")
