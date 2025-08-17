import logging
from torchvision import datasets

logger = logging.getLogger(__name__)


class OCTTestDataset(datasets.ImageFolder):
    def __init__(self, path):
        datasets.ImageFolder.__init__(self, root=path)

        logger.info(f"Test Dataset loaded with {len(self.dataset)} images.")
        logger.info(f"Found classes: {self.dataset.classes}")
        logger.info(f"Class to index mapping: {self.dataset.class_to_idx}")
