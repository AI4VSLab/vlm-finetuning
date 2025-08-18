import logging
from .datasets import OCTC8Dataset

logger = logging.getLogger(__name__)


class OCTDiagnosisData:
    def __init__(self, config):
        logger.info("Loading Transcript Data... ")
        self.train_data = OCTC8Dataset(config.path, split="train")
        logger.info(f"Train dataset size: {len(self.train_data)}")
        self.val_data = OCTC8Dataset(config.path, split="val")
        logger.info(f"Val dataset size: {len(self.val_data)}")
        self.test_data = OCTC8Dataset(config.path, split="test")
        logger.info(f"Test dataset size: {len(self.test_data)}")
        logger.info(f"Found classes: {self.test_data.classes}")
        logger.info(f"Class to index mapping: {self.test_data.class_to_idx}")
        logger.info("OCT-C8 Data Loaded.")
