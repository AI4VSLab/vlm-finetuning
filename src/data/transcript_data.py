import logging
from .datasets import OCTTranscriptDataset
from torchvision.datasets import ImageFolder

logger = logging.getLogger(__name__)


class OCTTranscriptData:
    def __init__(self, config):
        logger.info("Loading Transcript Data... ")
        self.train_data = OCTTranscriptDataset(config.train_path)
        logger.info(
            f"Transcript dataset loaded with {len(self.train_data.samples)} images"
        )

        self.test_data = ImageFolder(root=config.test_path)
        logger.info(f"Test Dataset loaded with {len(self.test_data)} images.")
        logger.info(f"Found classes: {self.test_data.classes}")
        logger.info(f"Class to index mapping: {self.test_data.class_to_idx}")
        logger.info("Data Loaded.")
