import logging
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

class OCTTestData:
    def __init__(self, config):
        self.config = config
        self.dataset = datasets.ImageFolder(root=self.config.test_path)

        logger.info(f"\nDataset loaded with {len(self.dataset)} images.")
        logger.info(f"Found classes: {self.dataset.classes}")
        logger.info(f"Class to index mapping: {self.dataset.class_to_idx}")
